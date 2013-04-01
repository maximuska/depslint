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

import argparse
import os
import re
import sys
import cPickle as pickle
from collections import defaultdict

_DEFAULT_TRACEFILE = 'deps.lst'
_DEFAULT_MANIFEST = 'build.ninja'

_IGNORED_PREFIXES = []
_IGNORED_SUFFICES = ['.d', '.lock', '.pyc', '.rsp']

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

def debug(msg):
    if _verbose < 0:
        return
    print "\033[1;34mINFO: %s\033[0m" % msg

#TBD: is needed?
def is_ignored(target):
    return any(target.startswith(prefix) for prefix in _IGNORED_PREFIXES) or \
           any(target.endswith(suffix) for suffix in _IGNORED_SUFFICES)

# Implicit dependencies list. Stores pairs of regexeps matching a
# target and its implicit dependencies, useful to discard false/irrelevant
# alarms regrading missing dependencies detected by the tool.
_IMPLICIT_DEPS_MATCHERS = [
    (re.compile(r""), # 'Immutable dependencies list' TODO: narrow this down?
     re.compile(r"utils/build/xmake/lib/xmake/.*\.py|"
                "common/pypackages/xiv/.*\.py|"
                "common/pypackages/elementtree.py|"
                "common/python/nextra/.*\.py|"
                "system/elf_x86_64.x|"
                "common/conf/gen/conf_validate_xml.py|"
                "utils/build/common.py|"
                "utils/build/scripts/(pyrexc|tar).deterministic.*|"
                "utils/administrator/(admin_util.py|admin_acl.py)|"
                "utils/misc/(code_exceptions.py|call_tree.py)|"
                "utils/Xylophage/Xylowrap|" # FIX XMAKE
                "common/journal/gen/journal_filenames.py|" # FIX XMAKE
                "common/spec/tools/elementtree.py|" # FIX XMAKE
                "common/pypackages/xiv/osutils/vexec_helper.so" # FIX XMAKE - tool dep?
                )),
    (re.compile(r".*\.ko$"), re.compile(r".*\.h$")), # FIX XMAKE. Missing deps on headers in kernel modules.
    (re.compile(r"utils/mcelog/mcelog"), re.compile(r"utils/mcelog/sandy-bridge\.[hc]")), # FIX XMAKE
    (re.compile(r"utils/cmdline/iperf/iperf"), re.compile(r"utils/cmdline/iperf/.*")), # FIX XMAKE
    (re.compile(r"utils/cmdline/netcat/netcat"), re.compile(r"utils/cmdline/netcat/.*")), # FIX XMAKE
    (re.compile(r"common/python/nextra/mcl/nextra_api.c"), re.compile(r"common/python/nextra/mcl/py_rpc_utils.pxi")), # FIX XMAKE
    (re.compile(r"challenge_response/pam/pam_cr.so"), re.compile(r"challenge_response/pam/modules.map")), # FIX XMAKE
    (re.compile(r"outputs/kernel_modules.tar.gz"), re.compile(r"")),                       # WTF happens here? Fixme?
    (re.compile(r"utils/smp_utils/"), re.compile(r".*\.lock|utils/smp_utils/.*Makefile")), # WTF happens here? Fixme?
    (re.compile(r"version_info/(scm_info|contained_commits)"), re.compile(r"\.\./\.git/.*")),
    (re.compile(r"outputs/kerntypes"), re.compile(r"")), # Fix ME by implementing aliasing soflinks or fix XMAKE rules.. &What is e.g., 'kernel_modules-549ae3653fede49fbb6f75162309a91c'?
    (re.compile(r"outputs/vmlinuz"), re.compile(r"kernel/cached-out/(include/.*|Makefile|scripts/.*|vmlinux)")), # FIX XMAKE?
    (re.compile(r"outputs/vmlinuz"), re.compile(r"kernel/git-hash|kernel/cached-out/arch/x86_64/boot/bzImage"))  # FIX XMAKE?
    ]

def match_implicit_dependency(dep, targets):
    """Verify if any of paths in 'targets' depends implicitly on 'dep'
    to inhibit a 'missing dependency' error."""
    #print "Matching %r on '%r'" % (dep, targets)
    for target_re, dep_re in _IMPLICIT_DEPS_MATCHERS:
        if not re.match(dep_re, dep):
            continue
        #print "Dep re matched"
        for t in targets:
            if re.match(target_re, t):
                return t
    return None

def trc_filter_ignored(targets):
    return [t for t in targets if not is_ignored(t)]

def norm_paths(paths):
    return [os.path.normpath(p) for p in paths]

class DepfileParser(object):
    _depfile_parse_re = re.compile(r'\s*(?P<targets>.*?)\s*'
                                   r'(?<!\\):'
                                   r'\s*(?P<deps>.*?)\s*$', re.DOTALL)
    _depfile_split_re = re.compile(r'(?<!\\)[\s]+')
    _depfile_unescape_re = re.compile(r'\\([ \\?*|])') # TODO: handle $ and !

    def parse_depfile(self, buf):
        buf = buf.replace('\\\n','') # unescape '\n's
        match = re.match(self._depfile_parse_re, buf)
        if not match or not match.group('targets'):
            raise Exception("Error parsing depfile: '%s'" % buf)
        targets = re.split(self._depfile_split_re, match.group('targets'))
        if match.group('deps') is '':
            deps = []
        else:
            deps = re.split(self._depfile_split_re, match.group('deps'))
        return (norm_paths(self._unescape(t) for t in targets),
                norm_paths(self._unescape(i) for i in deps))

    def _unescape(self, string):
        return re.sub(self._depfile_unescape_re, r'\1', string)

#TODO: Convert to a tuple
class Edge(object):
    def __init__(self, targets, deps, depfile_deps=[], order_only_deps=[], rule=""):
        self.targets = targets
        self.deps = deps
        self.depfile_deps = list(depfile_deps)
        self.order_only_deps = list(order_only_deps)
        self.rule = rule

class TraceParser(object):
    def __init__(self, input):
        self.input = input
        self.lineno = 0

    def _iterate_edges(self, input):
        for self.lineno, line in enumerate(input, start=1):
            tok = eval(line)
            #TODO: temporary Hack - move this code to depstrace?
            targets = trc_filter_ignored(tok['OUT'])
            deps = trc_filter_ignored(tok['IN'])
            if not targets:
                warn("Trace record at line %d has no targets after filtering: %r" % (self.lineno, tok))
            yield Edge(targets=targets, deps=deps)

    def iterate_edges(self):
        return self._iterate_edges(self.input)

class NinjaManifestParser(object):
    def __init__(self, input):
        self.input = input
        self.lineno = 0

        self.global_attributes = dict()
        self.edges_attributes = dict()
        self.edges = list()
        self.targets = dict()

        # Initializing rules list with a 'phony' rule
        self.rules = dict(phony=dict(attributes=[]))

        self._parse()
        # FIXME: should be able to do this 'per tested target tree' / or on demand
        self._load_depfiles()

    def _parse(self):
        for blk in self._iterate_manifest_blocks(self.input):
            if blk[0].startswith('build '):
                self._handle_build_blk(blk)
            elif blk[0].startswith('rule '):
                self._handle_rule_blk(blk)
            elif blk[0].startswith('default '):
                self._handle_default_blk(blk)
            else:
                self._handle_globals(blk)

    def _read_depfile(self, path):
        if not os.path.isfile(path):
            return None
        return file(path).read()

    def _load_depfiles(self):
        parser = DepfileParser()
        for edge in self.edges:
            depfile = self._eval_edge_attribute(edge, 'depfile')
            if not depfile:
                continue
            buf = self._read_depfile(depfile)
            if buf is None:
                warn("Depfile '%s' couldn't be read, ignoring" % (depfile, ))
                continue
            dep_targets, dep_inputs = parser.parse_depfile(buf)
            if set(dep_targets) ^ set(edge.targets):
                # Note: ninja doesn't accept depfile specifying more
                # than one target, we currently do..
                raise Exception("Depfile targets list %r doesn't match edge targets %r" % (dep_targets, edge.targets))

            # Verify that 'reload' extension is used properly.
            # Handle 'reload' by using 'depfile_deps' as the first-class
            # manifest 'deps' (as these are taking effect even on the 1st build)
            # TODO: ditto for manifest dependencies
            if self._eval_edge_attribute(edge, 'reload'):
                debug("'reload' attribute set for edge generating: %r" % (edge.targets,))
                if depfile not in set(edge.targets):
                    warn("Edge with 'reload' attribute doesn't mention depfile in targets: %r" % (edge.targets,))
                edge.deps += dep_inputs
                continue

            edge.depfile_deps = dep_inputs

    def _handle_globals(self, blk):
        global_attr = self._parse_attributes(blk)
        self.global_attributes.update(global_attr)
        if _verbose > 1 and global_attr:
            print "** Set global attribute: %r" % global_attr

    def _handle_default_blk(self, blk):
        # TODO: store default target for future reference?
        pass

    _build_re = re.compile(r'build\s+(?P<out>.+)\s*'+
                           r'(?<!\$):\s*(?P<rule>\S+)'+
                           r'\s*(?P<all_deps>.*)\s*$')
    def _handle_build_blk(self, blk):
        V2("** Parsing build block: '%s'" % (blk[0], ))
        match = re.match(self._build_re, blk[0])
        if not match:
            raise Exception("Error parsing manifest at line:%d: '%s'" % (self.lineno-len(blk), blk[0]))
        targets, rule, all_deps = match.groups()
        ins, implicit, order = self._split_deps(all_deps)

        # TODO: fix to evaluate along the parsing to avoid possible cycles
        edge_attrs = self._parse_attributes(blk[1:])

        # prep attributes scope
        scope = self.global_attributes.copy()
        scope.update(self._get_rule_attrs(rule))
        scope.update(edge_attrs)

        # evaluate targets and dependencies
        targets = norm_paths(self._split_unescape_and_eval(targets, scope))
        ins = norm_paths(self._split_unescape_and_eval(ins, scope))
        implicit = norm_paths(self._split_unescape_and_eval(implicit, scope))
        order = norm_paths(self._split_unescape_and_eval(order, scope))

        # Add automatic variables
        edge_attrs.update({'out':" ".join(targets), 'in':" ".join(ins)})

        edge = Edge(targets = targets,
                    deps = ins + implicit,
                    depfile_deps = [],
                    order_only_deps = order,
                    rule = rule)
        V1("** Edge** ", repr(edge))
        self.edges.append(edge)
        self.edges_attributes[edge] = edge_attrs
        for t in targets:
            self.targets[t] = edge

    _rule_re = re.compile(r'rule\s+(?P<rule>.+?)\s*$')
    def _handle_rule_blk(self, blk):
        match = re.match(self._rule_re, blk[0])
        if not match:
            raise Exception("Error parsing manifest at line:%d: '%s'" % (self.lineno-len(blk), blk[0]))
        rule = match.group('rule')
        attributes = self._parse_attributes(blk[1:])
        self.rules[rule] = dict(attributes=attributes)

    _attr_re = re.compile(r'\s*(?P<k>\w+)\s*=\s*(?P<v>.*?)\s*$') # key = val
    def _parse_attributes(self, blk):
        #TODO: eval/expand attributes as we parse!

        attributes = dict()
        for line in blk:
            match = re.match(self._attr_re, line)
            if not match:
                raise Exception("Error parsing manifest, expecting key=val, got: '%s'" % line)
            attributes[match.group('k')] = match.group('v')
        return attributes

    def _iterate_manifest_blocks(self, fh):
        # After stripping comments and joining escaped EOLs,
        #  'block' always starts with a 'header' at position '0',
        #  then optionally followed by a couple of 'indented key = val' lines.
        blk = []
        for line in self._iterate_manifest_lines(fh):
            if blk and not line[0].isspace():
                yield blk
                blk = []
            blk.append(line)
        if blk:
            yield blk

    def _iterate_manifest_lines(self, fh):
        acc = []
        for line in fh:
            self.lineno += 1
            # Skip empty lines (?) and comments
            if not line or line.isspace() or line.startswith('#'):
                continue
            # Join escaped EOLs
            if line.endswith('$\n'):
                acc.append(line[:-2])
                continue
            yield str.rstrip("".join(acc) + line)
            acc = []
        if acc:
            raise Exception("Error parsing manifest, unexpected end of file after: %s" % acc[-1])

    _split_all_deps_re = re.compile(r'(?P<in>.*?)'                      # Explicit deps
                                    r'((?<!\$)\|(?P<deps>[^|].*?)?)?' # Unescaped | + implicit deps
                                    r'((?<!\$)\|\|(?P<ord>.*))?'     # Unescaped || + order deps
                                    r'$')
    def _split_deps(self, s):
        if not s or s.isspace():
            return ("", "", "")
        match = re.match(self._split_all_deps_re, s)
        if not match:
            raise Exception("Error parsing deps: '%s'" % (s,))
        ins, implicit, order = match.group('in'), match.group('deps'), match.group('ord')
        return (ins or "", implicit or "", order or "")

    def _unescape(self, string):
        # Unescape '$ ', '$:', '$$' sequences
        return re.sub(r'\$([ :$])', r'\1', string)

    _deps_sep_re = re.compile(r'(?<!\$)\s+') # Unescaped spaces
    def _split_and_unescape(self, s):
        if not s:
            return []
        return [self._unescape(s) for s in re.split(self._deps_sep_re, s) if s != '']

    def _get_rule_attrs(self, rule):
        return self.rules[rule]['attributes']

    def _get_edge_attrs(self, edge):
        return self.edges_attributes[edge]

    def _split_unescape_and_eval(self, s, scope):
        lst = [self._eval_attribute(scope, x) for x in self._split_and_unescape(s)]
        V3(">> split_unescape_and_eval('%s') -> '%s'" % (s, lst))
        return lst

    _attr_sub_re = re.compile('(?<!\$)\$(\{)?(?P<attr>\w+)(?(1)})') # $attr or ${attr}
    def _eval_attribute(self, scope, attribute):
        # TBD: use empty strings for undefined attributes or raise?
        V3(">>> evaluating attribute: '%s'" % attribute)
        def evaluator(match):
            V3(">>> Evaluating replacement for attr:", match.group('attr'))
            attribute_val = scope.get(match.group('attr'), "")
            return self._eval_attribute(scope, attribute_val)
        evaluated_attr = re.sub(self._attr_sub_re, evaluator, attribute)
        V3(">>> evaluated attribute: '%s' -> '%s'" % (attribute, evaluated_attr))
        return evaluated_attr

    # TODO: fix variables substituation according to ninja's rules. E.g., need to eval as we parse.
    def _eval_edge_attribute(self, edge, attribute):
        scope = self.global_attributes.copy()
        scope.update(self._get_rule_attrs(edge.rule))
        scope.update(self._get_edge_attrs(edge))
        V3(">> eval_edge_attribute('%s': '%s')" % (scope, attribute))
        attribute_val = scope.get(attribute, "")
        return self._unescape(self._eval_attribute(scope, attribute_val))

    def get_edge(self, path):
        return self.targets[path]

    def iterate_edges(self):
        return iter(self.edges)

class Node(object):
    def __init__(self, provides, requires, built_after, is_phony):
        self.provides = frozenset(provides)
        self.requires = frozenset(requires)
        self.built_after = frozenset(built_after)
        self.is_phony = is_phony
        self.rank = 0

class Graph(object):
    def __init__(self):
        self.nodes = list()
        self.targets = dict()
        self.input_targets = set()
        self.duplicate_target_rules = set()

        self.top_targets = list()

        self.target_ranks = dict()
        self.targets_by_ranks = defaultdict(list)
        self.top_rank_targets = list()

        self.target_deps_closure = defaultdict(set)
        self.target_build_order_closure = defaultdict(set)

    def add_edge(self, edge):
        # TODO: consolidate Node and edge.
        # Make sure 'filter_target_paths' is called on all paths for consistency.
        node = Node(provides=edge.targets,
                    requires=edge.deps+edge.depfile_deps,
                    built_after=edge.deps+edge.order_only_deps,
                    is_phony=(edge.rule == "phony"))
        self.nodes.append(node)

        # Populate targets dictionary, take notes of duplicate targets
        for o in node.provides:
            if self.targets.get(o):
                self.duplicate_target_rules.add(o)
            self.targets[o] = node

        self.input_targets.update(node.requires)

    def _init_eval_graph_properties(self):
        debug("Finding top targets...")
        self._eval_top_targets()
        #debug("Top targets: %r" % self.top_targets)

        debug("Calculating ranks...")
        for target in self.top_targets:
            self._calc_ranks(target)
        self.top_rank = sorted(self.targets_by_ranks.keys())[-1]
        self.top_rank_targets = self.targets_by_ranks[self.top_rank]

    def build_graph(self, parser):
        for e in parser.iterate_edges():
            self.add_edge(e)

        self._init_eval_graph_properties()

    def get_target_rank(self, target):
        if not self.targets.get(target):
            raise Exception("ERROR: unknown target: '%s'" % target)
        return self.targets[target].rank

    def is_phony_target(self, target):
        node = self.targets.get(target)
        if not node:
            return False
        return node.is_phony

    def _eval_top_targets(self):
        top_targets_set = set(self.targets.keys()) - self.input_targets
        # top_targets_set = set(self.targets.keys())
        # for target, node in self.targets.iteritems():
        #     top_targets_set.difference_update(node.requires)
        if not top_targets_set and self.targets:
            raise Exception("ERROR: could not isolate top targets, check inputs for dependency loops")
        self.top_targets = sorted(top_targets_set)

    def target_iterate_deps_closure(self, root, targets_only=False, depends={}):
        frontend = [root]
        visited = set(frontend)
        depends[root] = None
        while frontend:
            path = frontend.pop(0)
            node = self.targets.get(path, None)
            if node:
                for p in node.requires:
                    if p in visited:
                        continue
                    frontend.append(p)
                    visited.add(p)
                    depends[p] = path
                    if not targets_only or p in self.targets:
                        yield p

    def _calc_ranks(self, target):
        visited = list()
        return self._do_calc_ranks(target, visited)

    def _do_calc_ranks(self, target, visited):
        # Cycles detection
        if target in visited:
            raise Exception("Dependencies loop detected: %r" % (visited + [target],))

        node = self.targets.get(target, None)
        if not node:
            # Static input rank
            return 0

        if node.rank:
            return node.rank

        rank = node.rank
        if node.requires:
            max_children_rank = max(self._do_calc_ranks(p, visited + [target]) for p in node.requires)
            # Phony targets don't climb ranks
            rank = node.rank = max_children_rank + (0 if node.is_phony else 1)

            closure = set(node.requires)
            for p in node.requires:
                closure.update(self.target_deps_closure[p])
            self.target_deps_closure[target] = closure

        self.targets_by_ranks[rank].append(target)
        return rank

    def get_deps_closure(self, target_path):
        return self.target_deps_closure[target_path]

    def iterate_target_paths_by_ranks(self):
        for rank in sorted(self.targets_by_ranks.keys()):
            for tpath in self.targets_by_ranks[rank]:
                yield tpath

    @classmethod
    def backtrace_to_root(cls, depends, p):
        l = list()
        while p:
            l.append(p)
            p = depends[p]
        return l

    def print_duplicate_target_edges(self):
        if self.duplicate_target_rules:
            warn("Detected multiple rules modifying targets:")
            for t in sorted(self.duplicate_target_rules):
                print "  %s" % t

    def print_target_deps(self, root, targets_only=False):
        for t in self.target_iterate_deps_closure(root, targets_only):
            print t

    def print_targets_by_ranks(self):
        for rank in reversed(sorted(self.targets_by_ranks.keys())):
            lst = ", ".join(self.targets_by_ranks[rank])
            if len(lst) > 120:
                lst = lst[:110] + "....."
            print "Rank %2d: %4d target(s) [%s]" % (rank, len(self.targets_by_ranks[rank]), lst)

    def print_top_rank_targets(self):
        for target in self.top_rank_targets:
            print "Target: '%s'" % target
            print " + rank: %d" % self.get_target_rank(target)
            print " + deps closure size: %d" % len(self.get_deps_closure(target))

def build_graph(path, parser_cls):
    cached_graph_path = path + ".pkl"
    if os.path.exists(cached_graph_path) and \
       os.path.getmtime(cached_graph_path) > os.path.getmtime(path) and \
       os.path.getmtime(cached_graph_path) > os.path.getmtime(__file__):
        warn("Loading a cached verion of graph for '%s'" % path)
        with file(cached_graph_path) as fh:
            g = pickle.load(fh)
        return g

    info("Building graph from '%s'" % path)
    g = Graph()
    with file(path, "r") as fh:
        g.build_graph(parser_cls(fh))
    with file(cached_graph_path + "~", "w") as fh:
        pickle.dump(g, fh)
    os.rename(cached_graph_path + "~", cached_graph_path)
    return g

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='depslint')
    parser.add_argument('--manifest', default=_DEFAULT_MANIFEST)
    parser.add_argument('-r', '--tracefile', default=_DEFAULT_TRACEFILE)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--version', action='version', version='%(prog)s: v0.1')
    parser.add_argument('targets', nargs='*')
    args = parser.parse_args()

    # TODO: make some order here
    _verbose = args.verbose

    ### Load Ninja graph
    manifest = args.manifest
    ng = build_graph(manifest, NinjaManifestParser)

    ### Load Strace graph
    g = build_graph(args.tracefile, TraceParser)

    ### Pass #1: checking full dependencies
    info("=== Pass #1: checking for missing dependencies ===")
    info("=== (these may lead to incomlete incremental builds) ===")
    for tp in ng.iterate_target_paths_by_ranks():
        if ng.is_phony_target(tp):
            continue

        target_in_trace = g.targets.get(tp)
        if not target_in_trace:
            warn("WARNING: manifest target '%s' doesn't present in strace graph" % tp)
            continue

        missing, ignored = [], []
        for depp in target_in_trace.requires - ng.get_deps_closure(tp):
            matching_dep = match_implicit_dependency(depp, [tp] + list(ng.get_deps_closure(tp)))
            if matching_dep:
                ignored.append(depp)
                continue
            missing.append(depp)
        if missing:
            print "ERROR: target '%s' dependencies missing: %r" % (tp, missing)
        # if ignored:
        #     warn("WARNING: ignored missing dependencies of '%s' on %r" % \
        #          (tp, ignored))

    # Currently requiring and picking up the first target, etc
    target = args.targets[0]

    warn("=== Statistics ===")
    gdepends, ngdepends = {}, {}
    gtargets  = set(g.target_iterate_deps_closure(target, depends=gdepends))
    ngtargets = set(ng.target_iterate_deps_closure(target, depends=ngdepends))

    info("=== Targets in manifest, not in the traces ===")
    info("=== (these may indicate somebody's mistakes somewhere) ===")
    info("=== (and are adding an unnecessary overhead on the build system) ===")
    for x in sorted(ngtargets - gtargets):
        if ng.is_phony_target(x):
            continue
        print "'%s': %s" % (x, " > ".join(Graph.backtrace_to_root(ngdepends, x)[1:]))

    ### Statistics
    # TODO
    # info("Totally: %d targets were ignored (prefixed with one of %r or ending with: %r)" % (
    #     len(ignored_set),_IGNORED_PREFIXES, _IGNORED_SUFFICES))

    info("Top targets from '%s':" % manifest)
    print "%r" % (sorted(trc_filter_ignored(ng.top_targets)),)

    info("Targets from '%s' by rank:" % manifest)
    ng.print_targets_by_ranks()
    info("Top RANK non-phony target(s) from '%s':" % manifest)
    ng.print_top_rank_targets()

    info("Targets from TRACE by rank:")
    g.print_targets_by_ranks()
    info("Top RANK target(s) from TRACE:")
    g.print_top_rank_targets()

    # Debug
    sys.exit(0)

    # Warn about any duplicate target build detected when tracing.
    # FIXME: disabled until will factor the test out to ignore 'reload' targets
    #g.print_duplicate_target_edges()
    ###
    ###
    ###

    sys.exit(0)
