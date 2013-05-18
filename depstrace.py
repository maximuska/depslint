#!/usr/bin/env python
#
# Copyright 2013 Maxim Kalaev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import optparse
import os
import re
import subprocess
import sys
#import tempfile

_NINJA_PROG_NAME = 'ninja'
_DEFAULT_OUTFILE = 'deps.lst'
_STRACE_LOG = 'strace_log.txt'
_STRACE_FIFO = '/tmp/_strace_log_fifo' # TODO: use tempfile

_FILEOPS=r'open|(sym)?link|rename|chdir|creat' # TODO: handle |openat?
_PROCOPS=r'clone|execve|v?fork'
_UNUSED=r'l?chown(32)?|[gs]etxattr|fchmodat|rmdir|mkdir|unlinkat|utimensat|getcwd|chmod|statfs(64)?|l?stat(64)?|access|readlink|unlink|exit_group|waitpid|wait4|arch_prctl|utime'

_ARG = (r'\{[^}]+\}|' + # {st_mode=S_IFREG|0755, st_size=97736, ...}
        r'"[^"]+"|'   + # "tst.o"
        r'\[[^]]+\]|' + # [{WIFEXITED(s) && WEXITSTATUS(s) == 0}]
        r'\S+')         # O_WRONLY|O_CREAT|O_TRUNC|O_LARGEFILE, et. al.
_OPS = '%s|%s|%s' % (_FILEOPS, _PROCOPS, _UNUSED)

global _verbose
_verbose = 0

def V1(*strings):
    if _verbose >= 1:
        print " ".join(strings)

def V2(*strings):
    if _verbose >= 2:
        print " ".join(strings)

def V3(*strings):
    if _verbose >= 3:
        print " ".join(strings)

def warn(msg):
    if _verbose < 0:
        return
    print "\033[1;33mWARNING: %s\033[0m" % msg

def info(msg):
    if _verbose < 0:
        return
    print "\033[1;32mINFO: %s\033[0m" % msg

class TracedRule(object):
    def __init__(self, lineno):
        self.deps = set()
        self.outputs = set()

        # Debug info
        self.pids = set()
        self.lineno = lineno

    def add_dep(self, path):
        self.deps.add(path)

    def add_output(self, path):
        self.outputs.add(path)

    def add_pid(self, pid):
        self.pids.add(pid)

    def get_deps_filtered(self):
        # Complex scripts may create intermediate outputs and then
        # reconsume these as inputs, therefore we don't consider modifed
        # files as dependendencies.
        return self.deps - self.outputs

    def get_outputs_filtered(self):
        # Filter out outputs which were deleted (consider these 'temporary' files)
        # existing_outputs = [f for f in self.outputs if os.path.lexists(f)]
        # return existing_outputs
        return self.outputs

class DepsTracer(object):
    # Regular expressions for parsing syscall in strace log
    # TODO: this is VERY slow. We can easily improve this if anyone cares...
    _file_re = re.compile(r'(?P<pid>\d+)\s+' +
                          r'(?P<op>%s)\(' % _OPS +
                          r'(?P<arg1>%s)?(, (?P<arg2>%s))?(, (%s))*' % (_ARG,_ARG,_ARG) +
                          r'\) = (?P<ret>-?\d+|\?)')

    # Regular expressions for joining interrupted lines in strace log
    _unfinished_re = re.compile(r'(?P<body>(?P<pid>\d+).*)\s+<unfinished \.\.\.>$')
    _resumed_re   = re.compile(r'(?P<pid>\d+)\s+<\.\.\. \S+ resumed> (?P<body>.*)')

    def __init__(self, build_dir=None):
        self._test_strace_version()
        self.build_dir = os.path.abspath(build_dir or os.getcwd())
        self.logfile = None
        self.unmatched_lines = []
        self.traced_rules = list()
        self.cur_lineno = 0
        self.pid2rule = dict()     # pid -> TracedRule (many to one is allowed)
        self.working_dirs = dict() # pid -> cwd

    def createRule(self, pid):
        r = TracedRule(self.cur_lineno)
        r.add_pid(pid)
        self.traced_rules.append(r)
        self.pid2rule[pid] = r
        return r

    def norm_path(self, cwd, path):
        path = os.path.join(cwd, path)
        path = os.path.normpath(path)

        # Make paths relative to build_dir when possible
        if os.path.isabs(path) and path.startswith(self.build_dir):
            path = path[len(self.build_dir):]
            path = path.lstrip(os.path.sep)

        return path

    def add_dep(self, pid, path):
        if not self._is_in_buildtree(path):
            return
        rule = self.pid2rule.get(pid)
        if rule:
            rule.add_dep(path)

    def add_output(self, pid, path):
        if not self._is_in_buildtree(path):
            return
        rule = self.pid2rule.get(pid)
        if rule:
            rule.add_output(path)

    def _is_in_buildtree(self, norm_path):
        # All paths which are in build tree were converted to relative by here
        in_build_tree = not os.path.isabs(norm_path)
        return in_build_tree

    def _test_strace_version(self):
        try:
            subprocess.check_call(['strace', '-o/dev/null','-etrace=file,process', 'true'])
        except subprocess.CalledProcessError:
            print >>sys.stderr, "strace is missing or incompatible"
            sys.exit(-1)

    def trace(self, cmd):
        """
        Run build script cmd under strace as: 'strace <cmd>' and factor out a list of 'rules'
        with dependencies and outputs (judging by files opened or modified).

        Return (status code, list of rule objects).
        """
        # Note (*) we are tracing now all system calls classified as 'file' or 'process'
        #  and warn if we see something unrecognizable to make sure we don't miss something important.
        # TODO: this approach is cpu-expensive, consider alternatives.
        self.logfile = file(_STRACE_LOG, "w")
        fifopath = _STRACE_FIFO
        #os.unlink(fifopath) - TBD + catch exception
        os.mkfifo(fifopath)
        try:
            command = ['strace',
                       '-o%s' % fifopath,
                       '-f',  # Follow child processes
                       '-a1', # Only one space before return values
                       '-s0', # Print non-filename strings really short to keep parser simpler
                       '-etrace=file,process', # Trace syscals related to file and process operations (*)
                       '-esignal=none'] + cmd
            V1("Running: %r" % command)

            strace_popen = subprocess.Popen(command)
            rules = self.parse_trace(file(fifopath))
        finally:
            os.unlink(fifopath)

        # Strace return code.
        retcode = strace_popen.wait()
        return retcode, rules

    def parse_trace(self, strace_out):
        # Init strace log parser
        log_iterator = self._strace_log_iter(strace_out)

        # Look for 'ninja' process invocation
        ninja_pid = None
        for pid, op, ret, arg1, _ in log_iterator:
            if op == 'execve' and ret == '0':
                path = os.path.normpath(arg1)
                if path.endswith(_NINJA_PROG_NAME):
                    ninja_pid = pid
                    V1("detected ninja process invocation: '%s'" % self.cur_line.strip())
                    break
        if ninja_pid is None:
            print >>sys.stderr, "Ninja ('%s') process invocation could not be detected" % _NINJA_PROG_NAME
            sys.exit(-1)

        # Track processes spawn under 'ninja' and record their inputs/outputs,
        # grouped by 'rule'. 'Rule' is considered to be process tree
        # parented directly under 'ninja'.
        for pid, op, ret, arg1, arg2 in log_iterator:
            # Ignore failed syscalls
            if ret  == '-1':
                continue

            # Process successful system calls
            cwd = self.working_dirs.get(pid, '.')
            if op in ('clone', 'fork', 'vfork') and ret  != '?':
                new_pid = ret
                self.working_dirs[new_pid] = cwd
                # Consider all processes forked by ninja directly a 'build rule' process tree
                if pid == ninja_pid:
                    V2("Creating a build rule record for pid %s, line %d in strace log" % (new_pid, self.cur_lineno))
                    self.createRule(new_pid)
                else:
                    rul = self.pid2rule.get(pid)
                    rul.add_pid(new_pid)
                    self.pid2rule[new_pid] = rul
            elif op == 'chdir':
                new_cwd = os.path.join(cwd, arg1)
                self.working_dirs[pid] = new_cwd
            elif op == 'open':
                path = self.norm_path(cwd, arg1)
                mode = arg2
                if 'O_DIRECTORY' in mode:
                    # Filter out 'opendir'-s.TBD: does this test worth the cycles?
                    continue
                if 'O_RDONLY' in mode:
                    self.add_dep(pid, path)
                else:
                    self.add_output(pid, path)
            elif op == 'execve':
                path = self.norm_path(cwd, arg1)
                self.add_dep(pid, path)
            elif op == 'symlink':
                path = self.norm_path(cwd, arg2)
                self.add_output(pid, path)
            elif op in ('rename', 'link'):
                from_path = self.norm_path(cwd, arg1)
                to_path = self.norm_path(cwd, arg2)
                self.add_dep(pid, from_path)
                self.add_output(pid, to_path)

        return self.traced_rules

    def _strace_log_iter(self, strace_log):
        interrupted_syscalls = {} # pid -> interrupted syscall log beginning
        for self.cur_lineno, line in enumerate(strace_log, start=1):
            self.cur_line = line
            if self.logfile:
                self.logfile.write(line)

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
                self.unmatched_lines.append(line.strip())
                continue

            pid, op, ret = fop.group('pid'), fop.group('op'), fop.group('ret')
            arg1, arg2 = fop.group('arg1'), fop.group('arg2')
            arg1 = arg1.strip('"') if arg1 else arg1
            arg2 = arg2.strip('"') if arg2 else arg2
            V2("pid=%s, op='%s', arg1=%s, arg2=%s, ret=%s" % (pid, op, arg1, arg2, ret))
            yield (pid, op, ret, arg1, arg2)

def process_results(options, rules, unmatched_lines):
    # Display unmatched lines.. (e.g., fixme's)
    for l in unmatched_lines:
        warn("Unmatched: '%s'" % l)

    # Log results
    info("Detected %d build rules in total, writing log: %s" % (len(rules), options.outfile))
    with file(options.outfile, "w") as f:
        for rule in rules:
            deps = sorted(rule.get_deps_filtered())
            outputs = sorted(rule.get_outputs_filtered())
            f.write("{'OUT': %r, 'IN': %r, 'LINE': %d, 'PID': %r}\n" % (
                outputs, deps, rule.lineno, "|".join(rule.pids)))
    info("Done")

def tracecmd(options, args):
    tracer = DepsTracer()

    # Build & trace
    status, rules = tracer.trace(cmd=args)
    if status:
        print >>sys.stderr, "**ERROR**: command execution has failed: %r" % args
        print >>sys.stderr, "**ERROR**: cwd:", os.getcwd()
        return status

    process_results(options, rules, tracer.unmatched_lines)
    return 0

def parse_tracefile(args):
    tracer = DepsTracer()

    # Process pre-recorded tracefile
    rules = tracer.parse_trace(file(args.from_tracefile, "r"))
    process_results(options, rules, tracer.unmatched_lines)
    return 0

if __name__ == '__main__':
    parser = optparse.OptionParser(prog='depstrace',
                                   version='%prog: git',
                                   usage="usage: %prog [options] -- [command [arg ...]]")
    parser.add_option('-o', '--outfile', default=_DEFAULT_OUTFILE,
                      help="store output to the specified file [default: %default]")
    parser.add_option('-r', '--from_tracefile',
                      help="parse pre-recorded strace output"
                      " instead of tracing the command")
    parser.add_option('-v', '--verbose', action='count', default=0)
    (options, args) = parser.parse_args()

    # Global verbosity settings
    _verbose = options.verbose

    if options.from_tracefile:
        # Process an existing strace output file instead of
        #  actually running the command under strace
        info("""Processing tracefile: %r""" % options.from_tracefile)
        parse_tracefile(options)
        sys.exit(0)

    # Run process, trace it and process the traces
    if not args:
        print >>sys.stderr, "ERROR: invalid command line"
        print >>sys.stderr, "Either '-r<file>' or a 'command' should be specified."
        sys.exit(-1)
    info("""Tracing: %r""" % args)
    ret = tracecmd(options, args)
    sys.exit(ret)
