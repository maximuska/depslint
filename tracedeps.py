#!/usr/bin/env python

# Inspired by 'fabricate'.

import os
import re
import subprocess
import sys
import tempfile

_keep_temps = True
_verbose = False
_DEPS_LOG = 'deps.lst'

_FILEOPS=r'open|openat|(sym)?link|rename|chdir|creat'
_PROCOPS=r'clone|execve|v?fork'
_UNUSED=r'l?chown|.etxattr|fchmodat|rmdir|unlinkat|utimensat|mkdir|getcwd|chmod|statfs(64)?|l?stat(64)?|access|readlink|unlink|exit_group|waitpid|wait4|arch_prctl|utime'

#TODO: handle escapes using lookbehind
_ARG=r'\{[^}]+\}|"[^"]+"|\[[^]]+\]|\S+'
_OPS='%s|%s|%s' % (_FILEOPS, _PROCOPS, _UNUSED)

class StraceRunner(object):
    # Regular expressions for parsing syscall in strace log
    # TBD: can it be faster?
    _file_re = re.compile(r'(?P<pid>\d+)\s+' +
                          r'(?P<op>%s)\(\s*' % _OPS +
                          r'(?P<arg1>%s)?(, (?P<arg2>%s))?(, (%s))*' % (_ARG,_ARG,_ARG) +
                          r'\s*\)\s+= (?P<ret>-?\d+|\?)')

    # Regular expressions for joining interrupted lines in strace log
    _unfinished_re = re.compile(r'(?P<body>(?P<pid>\d+).*)\s+<unfinished \.\.\.>$')
    _resumed_re   = re.compile(r'(?P<pid>\d+)\s+<\.\.\..*\s+resumed> (?P<body>.*)')

    def __init__(self, build_dir=None):
        self._test_strace_version()
        self.build_dir = os.path.abspath(build_dir or os.getcwd())
        self.deps = set()
        self.outputs = set()
        self.working_dirs = {} # pid -> cwd

    def norm_path(self, cwd, path):
        path = os.path.join(cwd, path)
        path = os.path.normpath(path)

        # Make paths relative to build_dir when possible
        if os.path.isabs(path) and path.startswith(self.build_dir):
            path = path[len(self.build_dir):]
            path = path.lstrip(os.path.sep)

        return path

    def add_dep(self, path):
        if self.is_relevant(path):
            self.deps.add(path)

    def add_output(self, path):
        if self.is_relevant(path):
            self.outputs.add(path)

    def is_relevant(self, name):
        # Here all paths in build tree are relative, therefore:
        in_build_tree = not os.path.isabs(name)
        if not in_build_tree:
            return False
        if (name.endswith('.pyc') or
            name.endswith('.rsp') or
            name.endswith('.tmp')):
            return False
        return True

    def _test_strace_version(self):
        try:
            subprocess.check_call(['strace', '-o/dev/null','-etrace=file,process', 'true'])
        except subprocess.CalledProcessError:
            print >>sys.stderr, "strace is missing or incompatible"
            sys.exit(-1)

    def _do_strace(self, outfile, outname, cmd):
        """ Run strace on: /bin/sh -c 'cmd'.
            Return (status code, list of dependencies, list of outputs). """
        # Collect traces
        # Note (*) we are tracing now all system calls classified as 'file' or 'process'
        #  and warn if we see something unrecognizable to make sure we don't miss something important.
        # TODO: this is wasteful, find out the complete listof relevant calls and fix.
        retcode = subprocess.call(['strace',
                                   '-o%s' % outname,
                                   '-f',  # Follow child processes
                                   '-a1', # Only one space before return values
                                   '-s1', # Print non-filename strings really short
                                   '-etrace=file,process', # Trace syscals related to file and process operations (*)
                                   '-esignal=none',        # Discard signals
                                   '/bin/sh', '-c', cmd])

        deps, outputs = self._parse_trace(outfile)

        return retcode, deps, outputs

    def _parse_trace(self, strace_output):
        interrupted_syscalls = {} # pid -> interrupted syscall log beginning
        for line in strace_output:
            #TODO: make a function
            # Join unfinished syscall traces to a single line
            match = self._unfinished_re.match(line)
            if match:
                pid, body = match.group('pid'), match.group('body')
                interrupted_syscalls[pid] = body
                continue
            match = self._resumed_re.match(line)
            if match:
                pid, body = match.group('pid'), match.group('body')
                line = interrupted_syscalls[pid] + body
                del interrupted_syscalls[pid]

            # Parse syscall line
            fop = self._file_re.match(line)
            if not fop:
                print "Warning: unmatched line:", line,
                continue

            pid, op, ret = fop.group('pid'), fop.group('op'), fop.group('ret')
            arg1 = fop.group('arg1')
            arg2 = fop.group('arg2')
            arg1 = arg1.strip('"') if arg1 else arg1
            arg2 = arg2.strip('"') if arg2 else arg2
            if _verbose:
                print "pid=%s" % pid,
                print "op='%s'" % op,
                print "arg1=%s" % arg1,
                print "arg2=%s" % arg2,
                print "ret=%s" % ret

            # Ignore failed syscalls
            if ret  == '-1':
                continue

            # Process successful system calls
            cwd = self.working_dirs.get(pid, '.')
            if op in ('clone', 'fork', 'vfork'):
                new_pid = ret
                self.working_dirs[new_pid] = cwd
            elif op == 'chdir':
                new_cwd = os.path.join(cwd, arg1)
                self.working_dirs[pid] = new_cwd
            elif op == 'open':
                path = self.norm_path(cwd, arg1)
                mode = arg2
                if 'O_DIRECTORY' in mode:
                    # Filter out 'opendir'-s.
                    # TBD: does this test worth the cycles?
                    continue
                if 'O_RDONLY' in mode:
                    self.add_dep(path)
                else:
                    self.add_output(path)
            elif op == 'execve':
                path = self.norm_path(cwd, arg1)
                self.add_dep(path)
            elif op == 'symlink':
                path = self.norm_path(cwd, arg2)
                self.add_output(path)
            elif op in ('rename', 'link'):
                from_path = self.norm_path(cwd, arg1)
                to_path = self.norm_path(cwd, arg2)
                self.add_dep(from_path)
                self.add_output(to_path)

        # Complex scripts may create intermediate outputs and then
        # reconsume these as inputs, therefore we don't consider modifed
        # files as dependendencies.
        deps = self.deps - self.outputs

        # Discard dependency on deptracer files (relevant if located in the build tree)
        deps.discard(os.path.normpath( __file__ ))

        # Filter out outputs which were deleted (consider these 'temporary' files)
        existing_outputs = [f for f in self.outputs if os.path.lexists(f)]

        return list(deps), existing_outputs

    def trace(self, cmd):
        """ Run command and return its dependencies and outputs, using strace
            to determine dependencies (by looking at what files are opened or
            modified). """
        handle, trcpath = tempfile.mkstemp(prefix="tracedeps-strace-", suffix=".txt")
        try:
            with os.fdopen(handle, 'r') as outfile:
                status, deps, outputs = self._do_strace(outfile=outfile, outname=trcpath, cmd=cmd)
        finally:
            if not _keep_temps:
                os.remove(trcpath)
        return status, deps, outputs, trcpath

def tracecmd(target, cmd):
    r = StraceRunner()

    # Run traced
    status, deps, outputs, strace_ofile = r.trace(cmd=cmd)
    if status:
        print >>sys.stderr, "**ERROR**: command execution has failed: '%s'" % cmd
        print >>sys.stderr, "**ERROR**: cwd:", os.getcwd()
        return status

    # Log results
    deps = sorted(deps)
    outputs = sorted(outputs)
    with file(_DEPS_LOG, "a") as f:
        f.write("{'target': %r, 'outputs': %r, 'inputs': %r, 'log': %r}\n" % (target, outputs, deps, strace_ofile))

    # Sanity:
    if os.path.normpath(target) not in outputs:
        print >>sys.stderr, "**WARNING**: target '%s' misdetected by strace for: '%s' (see %s)" % (
            target, cmd, strace_ofile)

    return 0

def restart_traced(target):
    if os.getenv('_DEP_TRACE_ON') is not None:
        return

    os.putenv('_DEP_TRACE_ON', '1')
    me = __file__
    if me.endswith(".pyc"):
        me = me[:-1]
    args = [me, target, "%s %s" % (sys.executable, " ".join(sys.argv))]
    print "**Restaring**:", args
    os.execv(me, args)

def dbg_parse_trace(trace_path):
    r = StraceRunner()
    deps, outputs = r._parse_trace(file(trace_path, "r"))
    print "deps: %r" % deps
    print "outs: %r" % outputs

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '-d':
        # Debug mode: process an existing strace output file
        #  (in the existing build tree) and print the findings.
        dbg_parse_trace(sys.argv[2])
        sys.exit(0)

    if len(sys.argv) < 3:
        # Note: requires a single string to make sure that symbols
        #   as '>' and '|' are quoted and interpreted properly.
        print  >>sys.stderr, "Usage: %s <target path> <string passed to '/bin/sh -c'>" % sys.argv[0]
        sys.exit(-1)
    target, cmd = sys.argv[1:]
    sys.exit(tracecmd(target, cmd))
