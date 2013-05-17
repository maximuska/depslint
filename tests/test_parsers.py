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
import depslint

# Silence all output
depslint._verbose = -1

class NinjaManifestParserIOHooked(depslint.NinjaManifestParser):
    def __init__(self, manifest, files_dict={}):
        self.files = files_dict
        super(NinjaManifestParserIOHooked, self).__init__(manifest)

    def _read_depfile(self, path):
        return self.files.get(path)

#TODO: rename edges to 'build rules'
class NinjaManifestParserTests(unittest.TestCase):
    def testBuildRulesParse(self):
        manifest = """
rule RULE
  command = cc $in -o $out

build out1 out2: RULE in1 in2 | ein1 ein2 || oin1 oin2

build out3: RULE in2 | in1
build out4: RULE || oin3
build out5: RULE | in4
build out6: RULE in5
build out7: RULE | || in6
build out8: RULE | ||
build out9: RULE ||
build outA: RULE |
"""
        parser = NinjaManifestParserIOHooked(manifest.splitlines())
        edges = list(parser.iterate_target_rules())

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out1", "out2"])
        self.assertItemsEqual(e.deps, ["in1", "in2", "ein1", "ein2"])
        self.assertItemsEqual(e.order_only_deps, ["oin1", "oin2"])
        self.assertEqual(e.rule_name, "RULE")
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "in1 in2")

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out3"])
        self.assertItemsEqual(e.deps, ["in1", "in2"])
        self.assertItemsEqual(e.order_only_deps, [])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "in2")
        self.assertEqual(parser._eval_edge_attribute(e, "out"), "out3")
        self.assertEqual(e.rule_name, "RULE")

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out4"])
        self.assertItemsEqual(e.deps, [])
        self.assertItemsEqual(e.order_only_deps, ["oin3"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "")
        self.assertEqual(parser._eval_edge_attribute(e, "out"), "out4")
        self.assertEqual(e.rule_name, "RULE")

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out5"])
        self.assertItemsEqual(e.deps, ["in4"])
        self.assertItemsEqual(e.order_only_deps, [])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "")
        self.assertEqual(parser._eval_edge_attribute(e, "out"), "out5")
        self.assertEqual(e.rule_name, "RULE")

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out6"])
        self.assertItemsEqual(e.deps, ["in5"])
        self.assertItemsEqual(e.order_only_deps, [])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "in5")
        self.assertEqual(parser._eval_edge_attribute(e, "out"), "out6")
        self.assertEqual(e.rule_name, "RULE")

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out7"])
        self.assertItemsEqual(e.deps, [])
        self.assertItemsEqual(e.order_only_deps, ["in6"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "")
        self.assertEqual(parser._eval_edge_attribute(e, "out"), "out7")
        self.assertEqual(e.rule_name, "RULE")

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out8"])
        self.assertItemsEqual(e.deps, [])
        self.assertItemsEqual(e.order_only_deps, [])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "")
        self.assertEqual(parser._eval_edge_attribute(e, "out"), "out8")
        self.assertEqual(e.rule_name, "RULE")

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out9"])
        self.assertItemsEqual(e.deps, [])
        self.assertItemsEqual(e.order_only_deps, [])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "")
        self.assertEqual(parser._eval_edge_attribute(e, "out"), "out9")
        self.assertEqual(e.rule_name, "RULE")

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["outA"])
        self.assertItemsEqual(e.deps, [])
        self.assertItemsEqual(e.order_only_deps, [])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "")
        self.assertEqual(parser._eval_edge_attribute(e, "out"), "outA")
        self.assertEqual(e.rule_name, "RULE")

        self.assertItemsEqual(edges, list())

    def testManifestPathsNormalization(self):
        manifest = """
# Paths normalization
build a/../outB: phony ./a/b | a//c || a/../a/d
"""
        parser = NinjaManifestParserIOHooked(manifest.splitlines())
        edges = list(parser.iterate_target_rules())

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["outB"])
        self.assertItemsEqual(e.deps, ["a/b", "a/c"])
        self.assertItemsEqual(e.order_only_deps, ["a/d"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "a/b")
        self.assertEqual(parser._eval_edge_attribute(e, "out"), "outB")
        self.assertEqual(e.rule_name, "phony")

        self.assertItemsEqual(edges, list())

    def testAtributes(self):
        manifest = """
v1 = out
v2 = in
v3 = g

rule RULE
  command = cc $in -o $out
  v3 = r

# Using variables in build contructs, with and w/o {}
# Build scope overrides definitions from global and rule scopes,
# including build block filenames.
build ${v1}: RULE $v2
  v1 = out1
  v2 = in1
  v4 = $v1/$v2 $v3

# Embedding varibles in a string
build ${v1}2: RULE prefix$v2.ext

# Rule scope and global scope
build ${v1}3: phony
"""
        parser = NinjaManifestParserIOHooked(manifest.splitlines())
        edges = list(parser.iterate_target_rules())

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out1"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "in1")
        self.assertEqual(parser._eval_edge_attribute(e, "v1"), "out1")
        self.assertEqual(parser._eval_edge_attribute(e, "v2"), "in1")
        self.assertEqual(parser._eval_edge_attribute(e, "v3"), "r")
        self.assertEqual(parser._eval_edge_attribute(e, "v4"), "out1/in1 r")

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out2"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "prefixin.ext")
        self.assertEqual(parser._eval_edge_attribute(e, "v1"), "out")
        self.assertEqual(parser._eval_edge_attribute(e, "v2"), "in")
        self.assertEqual(parser._eval_edge_attribute(e, "v3"), "r")

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out3"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "")
        self.assertEqual(parser._eval_edge_attribute(e, "v1"), "out")
        self.assertEqual(parser._eval_edge_attribute(e, "v2"), "in")
        self.assertEqual(parser._eval_edge_attribute(e, "v3"), "g")

        self.assertItemsEqual(edges, list())

    @unittest.skip("Fixme later")
    def testEscapes(self):
        manifest = """
v2 = in
v3 = g

# Escaping variables
build $${v1}4: phony $$v2
  v3 = $$v3
"""
        parser = NinjaManifestParserIOHooked(manifest.splitlines())
        edges = list(parser.iterate_target_rules())

        # Fix needed!
        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["${v1}4"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "$v2")
        self.assertEqual(parser._eval_edge_attribute(e, "v1"), "out")
        self.assertEqual(parser._eval_edge_attribute(e, "v2"), "in")
        self.assertEqual(parser._eval_edge_attribute(e, "v3"), "$v3")

        self.assertItemsEqual(edges, list())

    @unittest.skip("Fixme later")
    def testManifestRecursiveAttrs(self):
        manifest = """
# Recursive attribute redefinition
v2 = in

build out: phony $v2
  v2 = $v2.$v2
"""
        parser = NinjaManifestParserIOHooked(manifest.splitlines())
        edges = list(parser.iterate_target_rules())

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "in.in")
        self.assertEqual(parser._eval_edge_attribute(e, "v2"), "in.in")

        self.assertItemsEqual(edges, list())

    def testDepfilesLoading(self):
        manifest = """
rule RULE
  depfile = $out.d

build out1: RULE in1 | in2 || oin3

# Empty depfile
build out2: RULE || oin3

# Missing depfile
build out3: RULE
"""
        files = ({"out1.d":"out1: din1 din2",
                  "out2.d":"out2: "})
        parser = NinjaManifestParserIOHooked(manifest.splitlines(), files)
        edges = list(parser.iterate_target_rules())

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out1"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "in1")
        self.assertEqual(parser._eval_edge_attribute(e, "depfile"), "out1.d")
        self.assertItemsEqual(e.deps, ["in1", "in2"])
        self.assertItemsEqual(e.depfile_deps, ["din1", "din2"])
        self.assertItemsEqual(e.order_only_deps, ["oin3"])

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out2"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "")
        self.assertEqual(parser._eval_edge_attribute(e, "depfile"), "out2.d")
        self.assertItemsEqual(e.deps, [])
        self.assertItemsEqual(e.depfile_deps, [])
        self.assertItemsEqual(e.order_only_deps, ["oin3"])

        e = edges.pop(0)
        self.assertItemsEqual(e.targets,["out3"])
        self.assertEqual(parser._eval_edge_attribute(e, "in"), "")
        self.assertEqual(parser._eval_edge_attribute(e, "depfile"), "out3.d")
        self.assertItemsEqual(e.deps, [])
        self.assertItemsEqual(e.depfile_deps, [])
        self.assertItemsEqual(e.order_only_deps, [])

        self.assertItemsEqual(edges, list())

class DepfilesParsingTests(unittest.TestCase):
    def testDepfilesTypicalParse(self):
        depfile = """out1:"""
        parser = depslint.DepfileParser()
        targets, deps = parser.parse_depfile(depfile)
        self.assertItemsEqual(targets, ["out1"])
        self.assertItemsEqual(deps, [])

        depfile = """out: in1 in2"""
        targets, deps = parser.parse_depfile(depfile)
        self.assertItemsEqual(targets, ["out"])
        self.assertItemsEqual(deps, ["in1", "in2"])

        depfile = r"""out: \
in1 \
in2"""
        targets, deps = parser.parse_depfile(depfile)
        self.assertItemsEqual(targets, ["out"])
        self.assertItemsEqual(deps, ["in1", "in2"])

    def testDepfilesWhitespacesAndSpecials(self):
        depfile = r"""
out\ 1: in\ 1.h in\ 2.h \
in\3.h in\\4.h c:\ms(x86)\h @conf+-=.h"""
        parser = depslint.DepfileParser()
        targets, deps = parser.parse_depfile(depfile)
        self.assertItemsEqual(targets, ["out 1"])
        self.assertItemsEqual(deps, ["in 1.h", "in 2.h", r"in\3.h", r"in\4.h",
                                     r"c:\ms(x86)\h",
                                     "@conf+-=.h"])

    def testDepfilesMultitargets(self):
        depfile = r"""out\ 1 out\ 2: in\ 1.h in\ 2.h"""
        parser = depslint.DepfileParser()
        targets, deps = parser.parse_depfile(depfile)
        self.assertItemsEqual(targets, ["out 1", "out 2"])
        self.assertItemsEqual(deps, ["in 1.h", "in 2.h"])

    def testDepfilesPathsNormalization(self):
        depfile = r"""out//out1 ./out2: ./../in1.h /a/../in2.h"""
        parser = depslint.DepfileParser()
        targets, deps = parser.parse_depfile(depfile)
        self.assertItemsEqual(targets, ["out/out1", "out2"])
        self.assertItemsEqual(deps, ["../in1.h", "/in2.h"])


class TraceParserTests(unittest.TestCase):
    def testIterateTargetRules(self):
        input = """{'OUT': ['out1', 'out2'], 'IN': ['in1', 'in2']}
{'OUT': ['out3'], 'IN': []}
""".splitlines()
        parser = depslint.TraceParser(input)
        parser.iterate_target_rules()
        edges = list(parser.iterate_target_rules())

        e = edges.pop(0)
        self.assertItemsEqual(e.targets, ['out1', 'out2'])
        self.assertItemsEqual(e.deps, ['in1', 'in2'])
        self.assertItemsEqual(e.depfile_deps, [])
        self.assertItemsEqual(e.order_only_deps, [])

        e = edges.pop(0)
        self.assertItemsEqual(e.targets, ['out3'])
        self.assertItemsEqual(e.deps, [])
        self.assertItemsEqual(e.depfile_deps, [])
        self.assertItemsEqual(e.order_only_deps, [])

        self.assertItemsEqual(edges, list())
