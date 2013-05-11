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
import itertools
import os
import re
import sys
import cPickle as pickle
from collections import defaultdict

_DEPSLINT_CFG = '.depslint'
_DEFAULT_TRACEFILE = 'deps.lst'
_DEFAULT_MANIFEST = 'build.ninja'

# Matching targets are silently dropped when loading trace file, as if these
# were never accessed.
_IGNORED_SUFFICES = ['.d', '.pyc', '.rsp']

# Implicit dependencies list. Stores pairs of regexeps matching a
# target and its implicit dependencies, and used to discard false/irrelevant
# alarms regrading missing dependencies detected by the tool.
#
# For example, you may have a tool (e.g., calc_crc.sh) in the build tree
# invoked for each target built. It will be considered a dependency by deplint,
# but you couldn't care less. Adding ("", r"calc_crc\.sh") to the list
# will suppress the errors.

_IMPLICIT_DEPS_MATCHERS = ['']

global _verbose
_verbose = 0

_module_path = os.path.join(os.getcwd(), __file__)

def V1(*strings):
    if _verbose >= 1:
        print " ".join(strings)

def V2(*strings):
    if _verbose >= 2:
        print " ".join(strings)

def V3(*strings):
    if _verbose >= 3:
        print " ".join(strings)

def fatal(msg, ret=-1):
    sys.stdout.flush()
    print >>sys.stderr, "\033[1;41mFATAL: %s\033[0m" % msg
    sys.exit(ret)

def error(msg):
    if _verbose < 0:
        return
    print "\033[1;31mERROR: %s\033[0m" % msg

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

def is_ignored(target):
    return any(target.endswith(suffix) for suffix in _IGNORED_SUFFICES)

def match_implicit_dependency(dep, targets):
    """Verify if any of paths in 'targets' depends implicitly on 'dep'
    to inhibit a 'missing dependency' error."""
    V3("looking for an implicit dependency for any of %r on %r" % (targets, dep))
    for target_re, dep_re in _IMPLICIT_DEPS_MATCHERS:
        if not re.match(dep_re, dep):
            continue
        V3("Found a rule matching %r, checking for any target match.." % dep)
        for t in targets:
            if re.match(target_re, t):
                return t
    return None

def trc_filter_ignored(targets):
    return [t for t in targets if not is_ignored(t)]

def norm_paths(paths):
    return [os.path.normpath(p) for p in paths]

def sets_union(iterable_of_sets):
    u = set()
    for s in iterable_of_sets:
        u.update(s)
    return u

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

class BuildRule(object):
    def __init__(self, targets, deps, depfile_deps=[], order_only_deps=[], rule_name=""):
        self.targets = targets
        self.deps = deps
        self.depfile_deps = list(depfile_deps)
        self.order_only_deps = list(order_only_deps)
        self.rule_name = rule_name

    def __str__(self):
        return "%s %s: %s | %s || %s" % (
            self.targets, self.rule_name,
            self.deps, self.depfile_deps,
            self.order_only_deps)

class TraceParser(object):
    def __init__(self, input):
        self.input = input
        self.lineno = 0

    def _iterate_target_rules(self, input):
        for self.lineno, line in enumerate(input, start=1):
            tok = eval(line)
            #TODO: move the filtering to depstrace?
            targets = trc_filter_ignored(tok['OUT'])
            deps = trc_filter_ignored(tok['IN'])
            if not targets:
                warn("Trace record at line %d has no targets after filtering: %r" % (self.lineno, tok))
            yield BuildRule(targets=targets, deps=deps)

    def iterate_target_rules(self):
        return self._iterate_target_rules(self.input)

class NinjaManifestParser(object):
    def __init__(self, input):
        self.input = input
        self.lineno = 0

        self.global_attributes = dict()
        self.edges_attributes = dict()
        self.edges = list()

        # Initializing rules list with a 'phony' rule
        self.rules = dict(phony=dict(attributes=[]))

        self._parse()
        # FIXME: should be able to do this 'per tested target tree' /
        # or on demand
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
            # TODO: treat manifest dependencies the same
            if self._eval_edge_attribute(edge, 'reload'):
                V3("'reload' attribute set for edge generating: %r" % (edge.targets,))
                if depfile not in set(edge.targets):
                    warn("Rule with 'reload' attribute doesn't mention depfile in targets: %r" % (edge.targets,))
                edge.deps += dep_inputs
                continue

            edge.depfile_deps = dep_inputs

    def _handle_globals(self, blk):
        global_attr = self._parse_attributes(blk)
        self.global_attributes.update(global_attr)
        if global_attr:
            V2("** Set global attribute: %r" % global_attr)

    def _handle_default_blk(self, blk):
        # TODO: store default target for future reference?
        pass

    _build_re = re.compile(r'build\s+(?P<out>.+)\s*'+
                           r'(?<!\$):\s*(?P<rule>\S+)'+
                           r'\s*(?P<all_deps>.*)\s*$')
    def _handle_build_blk(self, blk):
        V3("** Parsing build block: '%s'" % (blk[0], ))
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

        edge = BuildRule(targets=targets,
                    deps=ins + implicit,
                    depfile_deps=[],
                    order_only_deps=order,
                    rule_name=rule)
        V2("** BuildRule** ", str(edge))
        self.edges.append(edge)
        self.edges_attributes[edge] = edge_attrs

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

    _split_all_deps_re = re.compile(r'(?P<in>.*?)'                    # Explicit deps
                                    r'((?<!\$)\|(?P<deps>[^|].*?)?)?' # Unescaped | + implicit deps
                                    r'((?<!\$)\|\|(?P<ord>.*))?'      # Unescaped || + order deps
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
        scope.update(self._get_rule_attrs(edge.rule_name))
        scope.update(self._get_edge_attrs(edge))
        V3(">> eval_edge_attribute('%s': '%s')" % (scope, attribute))
        attribute_val = scope.get(attribute, "")
        return self._unescape(self._eval_attribute(scope, attribute_val))

    def iterate_target_rules(self):
        return iter(self.edges)

class Edge(object):
    def __init__(self, provides, requires, is_phony):
        self.provides = frozenset(provides)
        self.requires = frozenset(requires)
        self.is_phony = is_phony
        self.rank = 0

class Graph(object):
    def __init__(self, from_brules, top_targets, is_clean_build_graph):
        self.target2edge  = dict()
        self.source2edges = defaultdict(set)

        self.duplicate_target_rules = set()

        self.top_targets = list()
        self.targets_by_ranks = defaultdict(set)

        self.target_deps_closure = defaultdict(set)
        self.target_products_closure = defaultdict(set)

        for brule in from_brules:
            # Clean build graph - as everything is rebuilt, only build
            # order matters. Depfiles do not exist.  Incremental build
            # - depfiles exist, and order rules can be neglected
            # assuming that clean order build is correct (and a
            # missing depfile triggers target rebuild).  Implicit and
            # explicit dependencies from manifest always play.
            deps = brule.deps + (brule.order_only_deps if is_clean_build_graph else brule.depfile_deps)
            edge = Edge(provides=brule.targets,
                        requires=deps,
                        is_phony=(brule.rule_name == "phony"))
            self._add_edge(edge)

        self._eval_graph_properties(top_targets)

    def _add_edge(self, edge):
        # Populate targets dictionary, take note of duplicate target
        # rules.
        for t in edge.provides:
            if self.target2edge.get(t):
                self.duplicate_target_rules.add(t)
            self.target2edge[t] = edge
        for s in edge.requires:
            self.source2edges[s].add(edge)

    def _eval_graph_properties(self, top_targets):
        if not top_targets:
            V1("Finding all terminal targets...")
            top_targets = self._find_top_targets()
        self.top_targets = list(top_targets)

        V1("Terminal targets (up to first 5): %r" % top_targets[:5])
        V1("Calculating deps closures and nodes ranks...")
        debug("Started _calc_deps_closure_in_tree")
        for target in top_targets:
            self._calc_deps_closure_in_tree(target)
        debug("Finished _calc_deps_closure_in_tree")

        debug("Started _calc_products_closure_in_tree")
        self._calc_products_closure_in_tree()
        debug("Finished _calc_products_closure_in_tree")

    def get_edge(self, target):
        """Returns an edge corresponding to target build rule, or
        'None' if there is no rule to build the target (e.g., a static target)."""
        return self.target2edge.get(target, None)

    def get_target_rank(self, target):
        if not self.target2edge.get(target):
            raise Exception("ERROR: unknown target: '%s'" % target)
        return self.target2edge[target].rank

    def is_phony_target(self, target):
        edge = self.target2edge.get(target)
        if not edge:
            return False
        return edge.is_phony

    def is_static_target(self, target):
        """Returns False if the specific target is output of a
        non-all-phony edges tree. Returns True otherwise."""
        e = self.target2edge.get(target, None)
        return not e or e.rank == 0

    def get_any_path_to_top(self, target):
        out_edges = self.source2edges.get(target)
        if not out_edges:
            # Terminal or unknown target..
            return []
        for edge in out_edges:
            for out in edge.provides:
                return [out] + self.get_any_path_to_top(out)
        # Unreachable
        assert False

    def _find_top_targets(self):
        # Top targets are not required by any other target in the build graph
        top_targets_set = set(self.target2edge.keys()) - set(self.source2edges.keys())
        if not top_targets_set and self.target2edge:
            raise Exception("ERROR: could not isolate top targets, check inputs for dependency loops")
        return sorted(top_targets_set)

    def _calc_deps_closure_in_tree(self, target):
        visited = list()
        return self._do_calc_deps_closure(target, visited)

    def _do_calc_deps_closure(self, target, visited):
        # Cycles detection
        if target in visited:
            raise Exception("Dependencies loop detected: %r" % (visited + [target],))

        edge = self.target2edge.get(target, None)
        # Static source?
        if not edge:
            self.targets_by_ranks[0].add(target)
            return 0

        # Already processed?
        if edge.rank:
            return edge.rank

        if edge.requires:
            max_children_rank = max(self._do_calc_deps_closure(p, visited + [target]) for p in edge.requires)
            # Special: phony targets don't climb ranks
            rank = edge.rank = max_children_rank + (0 if edge.is_phony else 1)

            closure = set(edge.requires)
            for p in edge.requires:
                closure.update(self.target_deps_closure[p])
            self.target_deps_closure[target] = closure
        else:
            if edge.is_phony:
                rank = edge.rank = 0
            else:
                # Non-static target w/no dependencies
                rank = edge.rank = 1

        self.targets_by_ranks[rank].add(target)
        return rank

    def _calc_products_closure_in_tree(self):
        reachable_sources = self.targets_by_ranks[0]
        visited = list()
        # DFS-traverse the DAG bottom-up
        for source in reachable_sources:
            self._do_calc_products_closure(source, visited)

    def _do_calc_products_closure(self, source, visited):
        # TODO: rewrite to use top-down BFS (to limit scope to the specified targets)
        # See then if it is possible to do 'deps' closure calculation 'on the way back'?

        # Cycles detection (just in case)
        if source in visited:
            raise Exception("Dependencies loop detected: %r" % (visited + [source],))

        # Already calculated?
        my_products = self.target_products_closure[source]
        if my_products:
            return my_products

        for out_edge in self.source2edges.get(source, []):
            for p in out_edge.provides:
                products = self._do_calc_products_closure(p, visited + [source])
                my_products.update(products)
            my_products.update(out_edge.provides)
        return my_products

    def get_deps_closure(self, target_path):
        return self.target_deps_closure[target_path]

    def iterate_targets_by_rank(self, include_static_targets):
        for rank in sorted(self.targets_by_ranks.keys()):
            if rank == 0 and not include_static_targets:
                continue
            for tpath in self.targets_by_ranks[rank]:
                #TODO: sort by significance
                yield tpath

def create_graph(path, parser_cls, targets=[], clean_build_graph=False):
    cached_graph_path = path + "%s.pkl" % ("-order" if clean_build_graph else "")
    if os.path.exists(cached_graph_path) and \
       os.path.getmtime(cached_graph_path) > os.path.getmtime(path) and \
       os.path.getmtime(cached_graph_path) > os.path.getmtime(_module_path):
        warn("Loading a cached verion of graph for '%s'" % path)
        with file(cached_graph_path) as fh:
            g = pickle.load(fh)
        return g

    info("Building graph from '%s'" % path)
    with file(path, "r") as fh:
        parser = parser_cls(fh)
        g = Graph(parser.iterate_target_rules(), targets, clean_build_graph)
    with file(cached_graph_path + "~", "w") as fh:
        pickle.dump(g, fh)
    os.rename(cached_graph_path + "~", cached_graph_path)
    return g

def load_config(path):
    if not os.path.isfile(path):
        V2("Note: no custom configuration file at: %r" % path)
        return None

    try:
        conf = {}
        execfile(path, conf)
    except Exception, e:
        # TODO: give more helpfull errors
        fatal("Error loading configuration file: %r" % e)

    info("Loaded configuration file: %r" % config_path)
    if conf.get('IGNORED_SUFFICES'):
        global _IGNORED_SUFFICES
        _IGNORED_SUFFICES = list(conf.get('IGNORED_SUFFICES'))
        V1("Set ignored suffices to: %r" % _IGNORED_SUFFICES)
    if conf.get('IMPLICIT_DEPS_MATCHERS'):
        impl_deps = conf.get('IMPLICIT_DEPS_MATCHERS')
        global _IMPLICIT_DEPS_MATCHERS
        _IMPLICIT_DEPS_MATCHERS = list((re.compile(t), re.compile(s)) for t, s in impl_deps)
        V1("Set implicit matchers to: %r" % impl_deps)

    return conf

def compare_dependencies(trace_graph, manifest_graph, clean_build):
    missing = defaultdict(list)
    ignored_missing = defaultdict(list)
    for tp in manifest_graph.iterate_targets_by_rank(include_static_targets=False):
        if manifest_graph.is_phony_target(tp):
            V2("Skipping phony: %s" % tp)
            continue

        edge_in_trace = trace_graph.get_edge(tp)
        if not edge_in_trace:
            warn("manifest target '%s' doesn't present in strace graph" % tp)
            continue

        for dep in edge_in_trace.requires - manifest_graph.get_deps_closure(tp):
            if clean_build and manifest_graph.is_static_target(dep):
                # Only dependencies on the non-static targets are critical
                # for a clean build.
                continue

            if match_implicit_dependency(dep, [tp] + list(manifest_graph.get_deps_closure(tp))):
                # The dependency 'tp | deps' IS missing in manifest graph,
                #  but 'implicit' dependencies rules fix this. Inhibit the warning.
                ignored_missing[tp].append(dep)
                continue
            missing[tp].append(dep)
    return missing, ignored_missing

def print_missing_dependencies(manifest_graph, missing, ignored_missing, clean_build):
    dtype="ORDER " if clean_build else ""
    for t, t_deps in missing.iteritems():
        error("target '%s' is missing %sdependencies on: %r" % (t, dtype, t_deps))
        #TODO: print path from t to top in verbose mode.
    if _verbose > 0:
        for t, t_ignored_deps in ignored_missing.iteritems():
            warn("ignored missing %sdependencies of '%s' on %r" % (dtype, t, t_ignored_deps))
            #TODO: print path from t to top in verbose mode.
    else:
        if ignored_missing:
            warn("%sDependency errors inhibited for %d targets due to implicit dependency rules" % (dtype, len(ignored_missing)))
            all_missing = set(itertools.chain(*ignored_missing.values()))
            warn("Distinct unspecifed dependencies num: %d" % (len(all_missing),))

def print_targets_by_ranks(graph):
    for rank in sorted(graph.targets_by_ranks.keys(), reverse=True):
        lst = ", ".join(graph.targets_by_ranks[rank])
        if len(lst) > 120:
            lst = lst[:110] + "....."
        print "Rank %2d: %5d target(s) [%s]" % (rank, len(graph.targets_by_ranks[rank]), lst)

def print_targets_by_depending_products(graph):
    def by_products(x,y):
        return cmp(len(graph.target_products_closure[x]), len(graph.target_products_closure[y]))
    relevant_sources = (x for x in graph.target_products_closure.iterkeys() if graph.is_static_target(x))
    relevant_targets = len(graph.target_deps_closure)
    for t in sorted(relevant_sources, cmp=by_products):
        score = len(graph.target_products_closure[t])
        print "%5d (%2.0f%%): %r" % (score, score*100.0/relevant_targets, t)
    print "Total relevant sources: %d, targets: %d, edges %d" % (
        len(list(relevant_sources)), relevant_targets, len(graph.target2edge))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='depslint')
    parser.add_argument('-C', dest='dir', help='change to DIR before doing anything else')
    parser.add_argument('-f', dest='manifest', default=_DEFAULT_MANIFEST, help='specify input ninja manifest')
    parser.add_argument('-r', dest='tracefile', default=_DEFAULT_TRACEFILE, help='specify input trace file')
    parser.add_argument('--conf', help='load custom configuration from CONF')
    parser.add_argument('-v', dest='verbose', action='count', default=0, help='increase verbosity level')
    parser.add_argument('--version', action='version', version='%(prog)s: v0.1')
    parser.add_argument('targets', nargs='*', help='specify targets to verify, as passed to ninja when traced')
    args = parser.parse_args()

    # Set global verbosity level
    _verbose = args.verbose

    # Process "-C"
    if args.dir:
        V1("Changing working dir to: %r" % args.dir)
        os.chdir(args.dir)

    # TODO: validate files existence (manifest, traces & config)
    # TODO: validate top targets (verify exists, convert phony, etc)

    # Attempt loading custom configuration file
    # With custom 'IGNORED_SUFFICES' and 'IMPLICIT_DEPS_MATCHERS' lists
    if args.conf:
        if load_config(args.conf) is None:
            fatal("Couldn't load configuration file: %r" % args.conf)
    else:
        config_path = os.path.join(os.path.dirname(args.manifest), _DEPSLINT_CFG)
        load_config(config_path)

    ### Build graphs
    ninja_clean_build_graph = create_graph(args.manifest, NinjaManifestParser, args.targets, clean_build_graph=True)
    ninja_incremental_graph = create_graph(args.manifest, NinjaManifestParser, args.targets, clean_build_graph=False)
    trace_graph = create_graph(args.tracefile, TraceParser, args.targets)

    ### Verification passes
    info("=== Pass #1: checking clean build order constraints ===")
    info("=== (may lead to clean build failure or, rarely, to incorrect builds) ===")
    missing, ignored = compare_dependencies(trace_graph, ninja_clean_build_graph, clean_build=True)
    if missing or ignored:
        info("Errors: %d, Ignored: %d" % (len(missing), len(ignored)))
        print_missing_dependencies(ninja_clean_build_graph, missing, ignored, clean_build=True)
    else:
        info("No issues!")

    info("=== Pass #2: checking for missing dependencies ===")
    info("=== (may lead to incomlete incremental builds if any) ===")
    missing, ignored = compare_dependencies(trace_graph, ninja_incremental_graph, clean_build=False)
    if missing or ignored:
        info("Errors: %d, Ignored: %d" % (len(missing), len(ignored)))
        print_missing_dependencies(ninja_incremental_graph, missing, ignored, clean_build=False)
    else:
        info("No issues!")

    ### Statistics passes
    warn("=== Statistics ===")
    top_targets = ninja_clean_build_graph.top_targets

    info("=== Targets in manifest, not in the traces ===")
    info("=== (these may indicate somebody's mistakes somewhere) ===")
    info("=== (and are adding an unnecessary overhead on the build system) ===")
    gtargets  = sets_union(trace_graph.get_deps_closure(t) for t in top_targets)
    ngtargets = sets_union(ninja_incremental_graph.get_deps_closure(t) for t in top_targets)
    for x in sorted(ngtargets - gtargets):
        if ninja_incremental_graph.is_phony_target(x):
            continue
        print "%-30r: > '%s'" % (x, "' > '".join(ninja_incremental_graph.get_any_path_to_top(x)))

    info("=== Targets from '%s' by order-dependencies rank ===" % args.manifest)
    print_targets_by_ranks(ninja_clean_build_graph)

    info("=== Targets from '%s' by rebuild-dependencies rank ===" % args.manifest)
    print_targets_by_ranks(ninja_incremental_graph)

    info("=== Targets from TRACE by rank ===")
    print_targets_by_ranks(trace_graph)

    #TODO: warn about any duplicate target build detected when tracing.
    #TODO: factor out 'reload' targets
    # warn("Detected multiple rules modifying targets:")
    # print trace_graph.duplicate_target_rules()

    info("=== Targets sorted by number of products ===")
    info("=== (e.g., how many will be rebuilt if 'x' is touched) ===")
    print_targets_by_depending_products(ninja_incremental_graph)

    info("That's all!")
    sys.exit(0)

    # TODO: try-except..
