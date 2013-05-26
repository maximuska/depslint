#!/usr/bin/python
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

import unittest
import depstrace

# Silence all output
depstrace._verbose = -1

class DepsTraceTests(unittest.TestCase):
    def testDepStraceLogIterator(self):
        tracefile = """5082  execve("/bin/ninja", [...], [/* 45 vars */]) = 0
5082  chdir("/home/build/") = 0
5082  rename("version_info/contained_commits~", "version_info/contained_commits") = 0
5082  open("tst.c", O_RDONLY) = 5
5082  open("tst.o", O_WRONLY|O_CREAT|O_TRUNC|O_LARGEFILE) = 6
5082  symlink(".build/install", "install") = 0
5086  execve("gcc", [...], [/* 45 vars */] <unfinished ...>
5085  open("tst.h", O_RDONLY|O_NOCTTY|O_LARGEFILE <unfinished ...>
5082  clone(child_stack=0, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0xb77c8768) = 5555
5082  clone( <unfinished ...>
4294  vfork( <unfinished ...>
5086  <... execve resumed> ) = 0
5085  <... open resumed> ) = 7
4294  <... vfork resumed> ) = 4295
5082  <... clone resumed> child_stack=0, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x7f1483d089f0) = ? ERESTARTNOINTR (To be restarted)
4120  wait4(-1,  <unfinished ...>
4120  <... wait4 resumed> [{WIFEXITED(s) && WEXITSTATUS(s) == 0}], WNOHANG, NULL) = 4180
4181  stat("/bin/ls", {st_mode=S_IFREG|0755, st_size=97736, ...}) = 0
5085  exit_group(0) = ?
"""
        tracer = depstrace.DepsTracer()
        tokens = list(tracer._strace_log_iter(tracefile.splitlines(True)))
        self.assertItemsEqual(tracer.unmatched_lines, [])

        # execve
        self.assertEqual(tokens.pop(0), ('5082', 'execve', '0', '/bin/ninja', '[...]'))
        # chdir
        self.assertEqual(tokens.pop(0), ('5082', 'chdir', '0', '/home/build/', None))
        # rename
        self.assertEqual(tokens.pop(0), ('5082', 'rename', '0', 'version_info/contained_commits~',
                                         'version_info/contained_commits'))
        # open one flag
        self.assertEqual(tokens.pop(0), ('5082', 'open', '5', 'tst.c', 'O_RDONLY'))
        # open flags OR-ed
        self.assertEqual(tokens.pop(0), ('5082', 'open', '6', 'tst.o', 'O_WRONLY|O_CREAT|O_TRUNC|O_LARGEFILE'))
        # symlink
        self.assertEqual(tokens.pop(0), ('5082', 'symlink', '0', '.build/install', 'install'))
        # clone
        self.assertEqual(tokens.pop(0), ('5082', 'clone', '5555', 'child_stack=0',
                                         'flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD'))
        # execve unfinished and then resumed
        self.assertEqual(tokens.pop(0), ('5086', 'execve', '0', 'gcc', '[...]'))
        # open unfinished and then resumed
        self.assertEqual(tokens.pop(0), ('5085', 'open', '7', 'tst.h', 'O_RDONLY|O_NOCTTY|O_LARGEFILE'))
        # vfork unfinished and then resumed
        self.assertEqual(tokens.pop(0), ('4294', 'vfork', '4295', None, None))
        # clone non-return (ERESTARTNOINTR)
        self.assertEqual(tokens.pop(0), ('5082', 'clone', '?', 'child_stack=0',
                                         'flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD'))

        # Complex parameter expression
        self.assertEqual(tokens.pop(0), ('4120', 'wait4', '4180', '-1', '[{WIFEXITED(s) && WEXITSTATUS(s) == 0}]'))

        # Another complex parameter expression
        self.assertEqual(tokens.pop(0), ('4181', 'stat', '0', '/bin/ls', '{st_mode=S_IFREG|0755, st_size=97736, ...}'))

        # non-return
        self.assertEqual(tokens.pop(0), ('5085', 'exit_group', '?', '0', None))

        # Ensure all the tockens were fetched
        self.assertEqual(0, len(tokens))

    def testDepStraceTrackTwoRules(self):
        tracefile = """5082  execve("/home/maximk/programs/bin/ninja", [...], [/* 45 vars */]) = 0
5082  open("tst.d", O_RDONLY) = 5
5082  open("tst2.d", O_RDONLY) = 5
5082  clone(child_stack=0, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0xb77c8768) = 5085
5082  clone(child_stack=0, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0xb77c8768) = 5086
5085  execve("/usr/bin/gcc", [...], [/* 45 vars */] <unfinished ...>
5086  execve("gcc", [...], [/* 45 vars */] <unfinished ...>
5085  <... execve resumed> ) = 0
5086  <... execve resumed> ) = 0
5085  open("tst.c", O_RDONLY|O_NOCTTY|O_LARGEFILE <unfinished ...>
5086  open("tst.c", O_RDONLY|O_NOCTTY|O_LARGEFILE <unfinished ...>
5085  <... open resumed> ) = 4
5086  <... open resumed> ) = 4
5085  open("tst.h", O_RDONLY|O_NOCTTY|O_LARGEFILE <unfinished ...>
5086  open("tst.h", O_RDONLY|O_NOCTTY|O_LARGEFILE <unfinished ...>
5085  <... open resumed> ) = 5
5086  <... open resumed> ) = 5
5085  open("tst2.d", O_WRONLY|O_CREAT|O_TRUNC|O_LARGEFILE, 0666) = 4
5086  open("tst.d", O_WRONLY|O_CREAT|O_TRUNC|O_LARGEFILE, 0666) = 4
5086  open("tst", O_RDWR|O_CREAT|O_TRUNC|O_LARGEFILE, 0666 <unfinished ...>
5086  <... open resumed> ) = 5
5085  open("tst2", O_RDWR|O_CREAT|O_TRUNC|O_LARGEFILE, 0666) = 4
5086  open("tst", O_RDWR|O_LARGEFILE) = 13
5085  open("tst2", O_RDWR|O_LARGEFILE) = 13
"""
        tracer = depstrace.DepsTracer(None)
        rules = tracer.parse_trace(tracefile.splitlines(True))
        self.assertEqual(2, len(rules))

        r = rules.pop(0)
        self.assertItemsEqual(r.deps, ['tst.c', 'tst.h'])
        self.assertItemsEqual(r.outputs, ['tst2', 'tst2.d'])
        self.assertItemsEqual(r.pids, ['5085'])
        self.assertEqual(r.lineno, 4)

        r = rules.pop(0)
        self.assertItemsEqual(r.deps, ['tst.c', 'tst.h', 'gcc'])
        self.assertItemsEqual(r.outputs, ['tst', 'tst.d'])
        self.assertItemsEqual(r.pids, ['5086'])
        self.assertEqual(r.lineno, 5)

        self.assertItemsEqual(tracer.unmatched_lines, [])

    def testDepStraceUnmatched(self):
        tracefile = """5082  execve("/home/maximk/programs/bin/ninja", [...], [/* 45 vars */]) = 0
5082  clone(child_stack=0, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0xb77c8768) = 5085
5085  execve("tst.sh", [...], [/* 45 vars */]) = 0
5085  open("tst", O_RDWR) = 4
5085  unmatchedop("tst.c") = 5
"""
        tracer = depstrace.DepsTracer()
        rules = tracer.parse_trace(tracefile.splitlines(True))
        self.assertEqual(1, len(rules))

        r = rules.pop(0)
        self.assertItemsEqual(r.deps, ['tst.sh'])
        self.assertItemsEqual(r.outputs, ['tst'])
        self.assertItemsEqual(r.pids, ['5085'])
        self.assertEqual(r.lineno, 2)

        self.assertItemsEqual(tracer.unmatched_lines, ['5085  unmatchedop("tst.c") = 5'])

    def testDepStraceDoubleResumedBug(self):
        tracefile = """5082  execve("/home/maximk/programs/bin/ninja", [...], [/* 45 vars */]) = 0
5082  clone(child_stack=0, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0xb77c8768) = 21316
21316 execve("/usr/lib/gcc/x86_64-linux-gnu/4.4.3/cc1plus", [...], [/* 63 vars */] <unfinished ...>
21316 <... execve resumed> ) = 0
21316 open("../../wtf/Assertions.h", O_RDONLY|O_NOCTTY <unfinished ...>
21316 <... open resumed> ) = 4
21316 <... open resumed> ) = 0
"""
        tracer = depstrace.DepsTracer(build_dir="/xxx/yyy")
        rules = tracer.parse_trace(tracefile.splitlines(True))
        self.assertEqual(1, len(rules))

        r = rules.pop(0)
        self.assertItemsEqual(r.deps, ['../../wtf/Assertions.h'])
        self.assertItemsEqual(r.outputs, [])
        self.assertItemsEqual(r.pids, ['21316'])
        self.assertEqual(r.lineno, 2)

        self.assertItemsEqual(tracer.unmatched_lines, ['21316 <... open resumed> ) = 0'])

    def testDepStraceDoubleUnfinishedBug(self):
        tracefile = """5082  execve("/home/maximk/programs/bin/ninja", [...], [/* 45 vars */]) = 0
5082  clone(child_stack=0, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0xb77c8768) = 21316
21316 execve("/usr/lib/gcc/x86_64-linux-gnu/4.4.3/cc1plus", [...], [/* 63 vars */] <unfinished ...>
21316 <... execve resumed> ) = 0
21316 open("../../wtf/Assertions.h", O_RDONLY|O_NOCTTY <unfinished ...>
21316 open("Unexpected.h", O_RDONLY|O_NOCTTY <unfinished ...>
21316 <... open resumed> ) = 4
"""
        tracer = depstrace.DepsTracer(build_dir="/xxx/yyy")
        rules = tracer.parse_trace(tracefile.splitlines(True))
        self.assertEqual(1, len(rules))

        r = rules.pop(0)
        self.assertIn(r.deps.pop(), ['Unexpected.h', '../../wtf/Assertions.h'])
        self.assertItemsEqual(r.outputs, [])
        self.assertItemsEqual(r.pids, ['21316'])
        self.assertEqual(r.lineno, 2)

        self.assertItemsEqual(tracer.unmatched_lines, ['21316 open("Unexpected.h", O_RDONLY|O_NOCTTY <unfinished ...>'])

    def testDepStraceExcessiveUnfinishedBug(self):
        #TODO
        tracefile = """5082  execve("/home/maximk/programs/bin/ninja", [...], [/* 45 vars */]) = 0
5082  clone(child_stack=0, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0xb77c8768) = 21316
21316 execve("/usr/lib/gcc/x86_64-linux-gnu/4.4.3/cc1plus", [...], [/* 63 vars */] <unfinished ...>
21316 <... execve resumed> ) = 0
21316 open("../../wtf/Assertions.h", O_RDONLY|O_NOCTTY <unfinished ...>
"""
        tracer = depstrace.DepsTracer(build_dir="/xxx/yyy")
        rules = tracer.parse_trace(tracefile.splitlines(True))
        self.assertEqual(1, len(rules))

        r = rules.pop(0)
        self.assertItemsEqual(r.deps, [])
        self.assertItemsEqual(r.outputs, [])
        self.assertItemsEqual(r.pids, ['21316'])
        self.assertEqual(r.lineno, 2)

        self.assertItemsEqual(tracer.unmatched_lines, ['21316 open("../../wtf/Assertions.h", O_RDONLY|O_NOCTTY <unfinished ...>'])

    def testDepStraceTrackChdir(self):
        #TODO
        tracefile = """"""
