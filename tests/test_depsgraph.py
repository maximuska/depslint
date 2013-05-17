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

from depslint import BuildRule as BR

# Silence all output
depslint._verbose = -1

def e2t(edges):
    """Edges to targets"""
    return depslint.sets_union(e.provides for e in edges)

class DepsGraphTestBasics(unittest.TestCase):
    build_rules = [
        BR(['outA'], ['A1', 'A2'], order_only_deps=['unused1']),
        BR(['outB'], ['B1', 'AB'], order_only_deps=['unused2']),
        BR(['A2'], ['AB']),
    ]
    def setUp(self):
        self.g = depslint.Graph(self.build_rules, [], is_clean_build_graph = False)

    def testBuild(self):
        g = self.g
        self.assertItemsEqual(g.target2edge.keys(), ["outA", "outB", "A2"])
        self.assertItemsEqual(g.source2edges.keys(), ["A1", "A2", "B1", "AB"])
        self.assertItemsEqual(g.duplicate_target_rules, [])

    def testTopTargets(self):
        g = self.g
        self.assertItemsEqual(g.top_targets, ["outA", "outB"])

    def testRanks(self):
        g = self.g
        self.assertItemsEqual(g.targets_by_ranks.keys(), [0, 1, 2])
        self.assertItemsEqual(g.targets_by_ranks[0], ["A1", "B1", "AB"])
        self.assertItemsEqual(g.targets_by_ranks[1], ["A2", "outB"])
        self.assertItemsEqual(g.targets_by_ranks[2], ["outA"])

    def testIsStatic(self):
        g = self.g
        self.assertTrue(g.is_static_target('A1'))
        self.assertTrue(g.is_static_target('B1'))
        self.assertTrue(g.is_static_target('AB'))
        self.assertFalse(g.is_static_target('A2'))
        self.assertFalse(g.is_static_target('outA'))
        self.assertFalse(g.is_static_target('outB'))

    def testGetPathToTop(self):
        g = self.g
        self.assertEqual(g.get_any_path_to_top('A1'), ['outA'])
        self.assertEqual(g.get_any_path_to_top('A2'), ['outA'])
        self.assertEqual(g.get_any_path_to_top('B1'), ['outB'])
        self.assertIn(g.get_any_path_to_top('AB'), (['outB'], ['A2','outA']))

    def testDepsClosure(self):
        g = self.g
        self.assertItemsEqual(g.get_deps_closure('AB'), [])
        self.assertItemsEqual(g.get_deps_closure('A1'), [])
        self.assertItemsEqual(g.get_deps_closure('B1'), [])
        self.assertItemsEqual(g.get_deps_closure('A2'), ["AB"])
        self.assertItemsEqual(g.get_deps_closure('outA'), ["A1", "A2", "AB"])
        self.assertItemsEqual(g.get_deps_closure('outB'), ["B1", "AB"])

    def testProductsClosures(self):
        g = self.g
        self.assertItemsEqual(e2t(g.get_product_rules_closure('A1')), ['outA'])
        self.assertItemsEqual(e2t(g.get_product_rules_closure('A2')), ['outA'])
        self.assertItemsEqual(e2t(g.get_product_rules_closure('B1')), ['outB'])
        self.assertItemsEqual(e2t(g.get_product_rules_closure('AB')), ['outB', 'A2', 'outA'])

class DepsGraphTestWanted(unittest.TestCase):
    build_rules = [
        BR(['outA'], ['A1', 'A2']),
        BR(['outB'], ['B1', 'AB']),
        BR(['A2'], ['AB']),
    ]
    def setUp(self):
        self.g = depslint.Graph(self.build_rules, targets_wanted=['outA'], is_clean_build_graph = False)

    def testBuild(self):
        g = self.g
        self.assertItemsEqual(g.target2edge.keys(), ["outA", "outB", "A2"])
        self.assertItemsEqual(g.source2edges.keys(), ["A1", "A2", "B1", "AB"])
        self.assertItemsEqual(g.duplicate_target_rules, [])

    def testTopTargets(self):
        g = self.g
        self.assertItemsEqual(g.top_targets, ["outA"])

    def testRanks(self):
        g = self.g
        self.assertItemsEqual(g.targets_by_ranks.keys(), [0, 1, 2])
        self.assertItemsEqual(g.targets_by_ranks[0], ["A1", "AB"])
        self.assertItemsEqual(g.targets_by_ranks[1], ["A2"])
        self.assertItemsEqual(g.targets_by_ranks[2], ["outA"])

    def testIsStatic(self):
        g = self.g
        self.assertTrue(g.is_static_target('A1'))
        self.assertTrue(g.is_static_target('AB'))
        self.assertFalse(g.is_static_target('A2'))
        self.assertFalse(g.is_static_target('outA'))
        # Not wanted
        self.assertFalse(g.is_static_target('outB')) # Not static & not wanted
        self.assertFalse(g.is_static_target('B1')) # Static, not wanted

    def testGetPathToTop(self):
        g = self.g
        self.assertEqual(g.get_any_path_to_top('A1'), ['outA'])
        self.assertEqual(g.get_any_path_to_top('A2'), ['outA'])
        self.assertEqual(g.get_any_path_to_top('AB'), ['A2', 'outA'])
        # Not wanted
        with self.assertRaises(Exception):
            g.get_any_path_to_top('B1')

    def testDepsClosure(self):
        g = self.g
        self.assertItemsEqual(g.get_deps_closure('AB'), [])
        self.assertItemsEqual(g.get_deps_closure('A1'), [])
        self.assertItemsEqual(g.get_deps_closure('A2'), ["AB"])
        self.assertItemsEqual(g.get_deps_closure('outA'), ["A1", "A2", "AB"])
        # Not wanted
        with self.assertRaises(Exception):
            g.get_deps_closure('outB')
        with self.assertRaises(Exception):
            g.get_deps_closure('B1')

    def testProductsClosures(self):
        g = self.g
        self.assertItemsEqual(e2t(g.get_product_rules_closure('A1')), ['outA'])
        self.assertItemsEqual(e2t(g.get_product_rules_closure('A2')), ['outA'])
        self.assertItemsEqual(e2t(g.get_product_rules_closure('AB')), ['A2', 'outA'])
        # Not wanted
        with self.assertRaises(Exception):
            g.get_product_rules_closure('B1')

class DepsGraphWithPhony(unittest.TestCase):
    build_rules = [
        BR(['outA'], ['A1', 'A2']),
        BR(['A1'], [], rule_name="phony"),
        BR(['A2'], ['AB']),
        BR(['AB'], [], rule_name="phony"),
        BR(['aliasA2'], ['A2'], rule_name="phony"),
    ]
    def setUp(self):
        self.g = depslint.Graph(self.build_rules, targets_wanted=[], is_clean_build_graph = False)

    def testBuild(self):
        g = self.g
        self.assertItemsEqual(g.target2edge.keys(), ["outA", "aliasA2", "A2", "A1", "AB"])
        self.assertItemsEqual(g.source2edges.keys(), ["A1", "A2", "AB"])
        self.assertItemsEqual(g.duplicate_target_rules, [])

    def testTopTargets(self):
        g = self.g
        self.assertItemsEqual(g.top_targets, ["outA", "aliasA2"])

    def testRanks(self):
        g = self.g
        self.assertItemsEqual(g.targets_by_ranks.keys(), [0, 1, 2])
        self.assertItemsEqual(g.targets_by_ranks[0], ["A1", "AB"])
        self.assertItemsEqual(g.targets_by_ranks[1], ["A2", "aliasA2"])
        self.assertItemsEqual(g.targets_by_ranks[2], ["outA"])

    def testIsStatic(self):
        g = self.g
        self.assertTrue(g.is_static_target('A1'))
        self.assertTrue(g.is_static_target('AB'))
        self.assertFalse(g.is_static_target('A2'))
        self.assertFalse(g.is_static_target('outA'))
        self.assertFalse(g.is_static_target('aliasA2'))

    def testResolvePhony(self):
        g = self.g
        self.assertItemsEqual(g.resolve_phony(['A1']), ['A1'])
        self.assertItemsEqual(g.resolve_phony(['aliasA2', 'A1']), ['A2', 'A1'])
        self.assertItemsEqual(g.resolve_phony(['aliasA2', 'A1']), ['A2', 'A1'])

    def testGetPathToTop(self):
        g = self.g
        self.assertIn(g.get_any_path_to_top('A1'), (['outA'],))
        self.assertIn(g.get_any_path_to_top('A2'), (['outA'],['aliasA2']))
        self.assertIn(g.get_any_path_to_top('AB'), (['A2', 'outA'], ['A2', 'aliasA2']))
        self.assertEqual(g.get_any_path_to_top('outA'), [])
        self.assertEqual(g.get_any_path_to_top('aliasA2'), [])

    def testDepsClosure(self):
        g = self.g
        self.assertItemsEqual(g.get_deps_closure('AB'), [])
        self.assertItemsEqual(g.get_deps_closure('A1'), [])
        self.assertItemsEqual(g.get_deps_closure('A2'), ['AB'])
        self.assertItemsEqual(g.get_deps_closure('outA'), ['A1', 'A2', 'AB'])
        self.assertItemsEqual(g.get_deps_closure('aliasA2'), ['A2', 'AB'])

    def testProductsClosures(self):
        g = self.g
        self.assertItemsEqual(e2t(g.get_product_rules_closure('A1')), ['outA'])
        self.assertItemsEqual(e2t(g.get_product_rules_closure('A2')), ['outA', 'aliasA2'])
        self.assertItemsEqual(e2t(g.get_product_rules_closure('AB')), ['A2', 'outA', 'aliasA2'])

class DepsGraphMultipleOutputs(unittest.TestCase):
    # TODO: highly relevant to trace graphs
    pass
