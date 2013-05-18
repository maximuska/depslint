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
import time
from collections import defaultdict

_DEPSLINT_CFG = '.depslint'
_DEFAULT_TRACEFILE = 'deps.lst'
_DEFAULT_MANIFEST = 'build.ninja'
_SUPPORTED_NINJA_VER = "1.2"

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

_module_path = os.path.join(os.getcwd(), __file__)

global _verbose
_verbose = 0

global _logfile
_logfile = None
def _set_logger(filename):
    if not filename:
        filename = '/dev/null'
    global _logfile
    _logfile = open(filename, "w")

def log_msg(level, msg, trunc_lines=True, ansi=None, fd=sys.stdout):
    if _verbose < 0:
        # Testing mode: be silent
        return

    if _verbose >= level:
        _logfile.write("[%s] %s\n" % (time.asctime(), msg))

        if trunc_lines and len(msg) > 140:
            msg = msg[:137] + "..."
        if ansi:
            _ANSI_END = '\033[0m'
            msg = ansi + msg + _ANSI_END
        print >>fd, msg
        fd.flush()
        return

    # Always log levels 0 and 1
    if level in (0, 1):
        _logfile.write("[%s] %s\n" % (time.asctime(), msg))

def H0():
    log_msg(0, "")

def V0(msg, trunc_lines=True):
    log_msg(0, msg, trunc_lines)

def V1(msg, trunc_lines=True):
    log_msg(1, msg, trunc_lines)

def V2(msg, trunc_lines=True):
    log_msg(2, msg, trunc_lines)

def V3(msg, trunc_lines=True):
    log_msg(3, msg, trunc_lines)

def fatal(msg, ret=-1):
    msg = "FATAL: %s" % msg
    log_msg(0, msg, trunc_lines=False, ansi='\033[1;41m', fd=sys.stderr)
    sys.exit(ret)

def error(msg):
    msg = "ERROR: %s" % msg
    log_msg(0, msg, trunc_lines=False, ansi='\033[1;31m')

def warn(msg):
    msg = "WARNING: %s" % msg
    log_msg(0, msg, trunc_lines=False, ansi='\033[1;33m')

def info(msg):
    msg = "INFO: %s" % msg
    log_msg(0, msg, ansi='\033[1;32m')

def debug(msg):
    msg = "DEBUG: %s" % msg
    log_msg(0, msg, ansi='\033[1;34m')

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
        buf = buf.replace('\\\n','') # unescape and remove '\n's
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
        self.default_targets = []
        self.ninja_required_version = 0.0

        # Initializing rules list with a 'phony' rule
        self.rules = dict(phony=dict(attributes=[]))

        self._parse()
        # FIXME: should do this 'per tested target tree' / or on
        # demand. Becomes less urgent with ninja 'deps' support.
        self._load_depfiles()

    def iterate_target_rules(self):
        return iter(self.edges)

    def get_default_targets(self):
        return self.default_targets

    def _parse(self):
        for blk in self._iterate_manifest_blocks(self.input):
            if blk[0].startswith('build '):
                self._handle_build_blk(blk)
            elif blk[0].startswith('rule '):
                self._handle_rule_blk(blk)
            elif blk[0].startswith('default '):
                self._handle_default_blk(blk)
            elif blk[0].startswith('pool '):
                self._handle_pool_blk(blk)
            elif blk[0].startswith('include '):
                self._handle_include(blk)
            elif blk[0].startswith('subninja '):
                self._handle_subninja(blk)
            else:
                self._handle_globals(blk)

    def _read_depfile(self, path):
        if not os.path.isfile(path):
            return None
        return file(path).read()

    def _load_depfiles(self):
        parser = DepfileParser()
        for edge in self.edges:
            # TODO: support 'deps' dependencies format
            if self._eval_edge_attribute(edge, 'deps'):
                warn("'deps' dependencies format specified for targets %r is not supported" % (list(edge.provides), ))

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
        global_attr = dict(self._parse_attributes(blk))
        self.global_attributes.update(global_attr)
        if global_attr:
            V2("** Set global attribute: %r" % global_attr)
        self._check_required_version(global_attr)

    def _handle_default_blk(self, blk):
        targets_str = blk[0][len('default '):]
        self.default_targets = self._split_unescape_and_eval(targets_str, self.global_attributes)

    def _handle_pool_blk(self, blk):
        # Just skipping over
        pass

    def _handle_include(self, blk):
        fatal("'include' keyword support not implemented, wanna help?")

    def _handle_subninja(blk):
        fatal("'subninja' keyword support not implemented, wanna help?")

    def _check_required_version(self, attrs):
        if self.ninja_required_version:
            return
        self.ninja_required_version = float(attrs.get('ninja_required_version', 0))
        V2("** ninja_required_version: %r" % self.ninja_required_version)
        if self.ninja_required_version > _SUPPORTED_NINJA_VER:
            warn("Ninja version required in manifest is newer than supported (%r vs %r)",
                 self.ninja_required_version, _SUPPORTED_NINJA_VER)
            warn("Trying to continue but the results may be meaningless...")

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
        edge_attrs = dict(self._parse_attributes(blk[1:]))

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
        attributes = dict(self._parse_attributes(blk[1:]))
        self.rules[rule] = dict(attributes=attributes)

    _attr_re = re.compile(r'\s*(?P<k>\w+)\s*=\s*(?P<v>.*?)\s*$') # key = val
    def _parse_attributes(self, blk):
        #TODO: fix to eval/expand attributes as we parse!

        for line in blk:
            match = re.match(self._attr_re, line)
            if not match:
                raise Exception("Error parsing manifest, expecting key=val, got: '%s'" % line)
            yield (match.group('k'), match.group('v'))

    def _iterate_manifest_blocks(self, fh):
        blk = []
        for line in self._iterate_manifest_lines(fh):
            # After stripping comments and joining escaped EOLs,
            #  'block' always starts with a 'header' at position '0',
            #  then optionally followed by a couple of 'indented key = val' lines.
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

class Edge(object):
    def __init__(self, provides, requires, is_phony):
        self.provides = frozenset(provides)
        self.requires = frozenset(requires)
        self.is_phony = is_phony
        self.rank = None

class Graph(object):
    def __init__(self, from_brules, targets_wanted, is_clean_build_graph):
        self.target2edge  = dict()
        self.source2edges = defaultdict(set)

        self.duplicate_target_rules = set()

        self.top_targets = list(targets_wanted)
        self.targets_by_ranks = defaultdict(set)

        self.target_deps_closure = dict()
        self.target_products_closure = dict()

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

        self._eval_graph_properties()

    def _add_edge(self, edge):
        # Populate targets dictionary, take note of duplicate target
        # rules.
        for t in edge.provides:
            if self.target2edge.get(t):
                self.duplicate_target_rules.add(t)
            self.target2edge[t] = edge
        for s in edge.requires:
            self.source2edges[s].add(edge)

    def _eval_graph_properties(self):
        if not self.top_targets:
            V1("Finding all terminal targets...")
            self.top_targets = list(self._find_top_targets())
        #TODO: filter out top_targets w/o an incoming edge?

        V1("Terminal targets (up to first 5): %r" % self.top_targets[:5])
        V1("Calculating deps closures and nodes ranks...")
        for target in self.top_targets:
            self._calc_deps_closure_in_tree(target)

        V1("Calculating targets product closures...")
        self._calc_products_closure_in_tree()
        V1("Done")

    def get_edge(self, target):
        """Returns an edge corresponding to target build rule, or
        'None' if there is no rule to build the target (e.g., a static target)."""
        return self.target2edge.get(target, None)

    def is_phony_target(self, target):
        edge = self.target2edge.get(target)
        if not edge:
            return False
        return edge.is_phony

    def is_static_target(self, target):
        """Returns 'True' if the target is 'wanted' and unless the
        target is a product of the non-all-phony edges path. 'False'
        otherwise."""
        return target in self.targets_by_ranks[0]

    def get_any_path_to_top(self, target):
        out_edges = self.source2edges.get(target, [])
        out_edges = [e for e in out_edges if self._is_wanted(e)]
        if not out_edges and target not in self.top_targets:
            # If the queried target was 'wanted', there should be a
            # path to a 'wanted' top target.
            raise Exception("Unknown or unwanted target: %r" % target)

        for edge in out_edges:
            for out in edge.provides:
                return [out] + self.get_any_path_to_top(out)

        # No outgoing edges, reached the top of the 'wanted' sub-graph
        return []

    def get_deps_closure(self, target):
        try:
            return self.target_deps_closure[target]
        except KeyError:
            raise Exception("Unknown or unwanted target: %r" % target)

    def get_product_rules_closure(self, target):
        try:
            return self.target_products_closure[target]
        except KeyError:
            raise Exception("Unknown or unwanted target: %r" % target)

    def resolve_phony(self, targets):
        """Substitute phone targets by non-phony dependecies, unless
        target dependencies list is empty."""
        resolved = []
        for t in targets:
            edge = self.target2edge.get(t)
            if not edge or not edge.is_phony or not edge.requires:
                resolved.append(t)
                continue
            resolved.extend(self.resolve_phony(edge.requires))
        return resolved

    def iterate_targets_by_rank(self, include_static_targets):
        for rank in sorted(self.targets_by_ranks.keys()):
            if rank == 0 and not include_static_targets:
                continue
            for tpath in self.targets_by_ranks[rank]:
                #TODO: sort by significance
                yield tpath

    def sorted_by_products_num(self, targets, reverse=False):
        def by_products(x,y):
            return cmp(len(self.get_product_rules_closure(x)), len(self.get_product_rules_closure(y)))
        return sorted(targets, cmp=by_products, reverse=reverse)

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
        if not edge:
            # Static source
            self.targets_by_ranks[0].add(target)
            self.target_deps_closure[target] = set()
            return 0

        # Already processed?
        if edge.rank is not None:
            return edge.rank

        max_children_rank = 0
        closure = set(edge.requires)
        for p in edge.requires:
            max_children_rank = max(self._do_calc_deps_closure(p, visited + [target]), max_children_rank)
            closure.update(self.target_deps_closure[p])

        # Note: phony targets don't climb ranks,
        # non-static target w/no dependencies are ranked '1'
        rank = edge.rank = max_children_rank + (0 if edge.is_phony else 1)
        for t in edge.provides:
            self.target_deps_closure[t] = closure
            self.targets_by_ranks[rank].add(t)
        return rank

    def _is_wanted(self, edge):
        return edge.rank is not None

    def _calc_products_closure_in_tree(self):
        reachable_sources = self.targets_by_ranks[0]
        visited = list()
        # DFS-traverse the DAG bottom-up
        for source in reachable_sources:
            self._do_calc_products_closure(source, visited)

    def _do_calc_products_closure(self, source, visited):
        # TODO: rewrite to use top-down BFS (to limit scope to the specified targets),
        # see then if it is possible to do 'deps' closure calculation 'on the way back'?

        # Cycles detection (just in case)
        if source in visited:
            raise Exception("Dependencies loop detected: %r" % (visited + [source],))

        # Already calculated?
        my_products = self.target_products_closure.get(source, set())
        if my_products:
            return my_products

        for out_edge in self.source2edges.get(source, []):
            if not self._is_wanted(out_edge):
                continue
            for p in out_edge.provides:
                products = self._do_calc_products_closure(p, visited + [source])
                my_products.update(products)
            my_products.add(out_edge)
        self.target_products_closure[source] = my_products
        return my_products

def create_graph(path, parser, targets=[], clean_build_graph=False):
    info("Building %s graph for '%s'.." % (
        "order-only" if clean_build_graph else "dependency", path))
    g = Graph(parser.iterate_target_rules(), targets, clean_build_graph)
    return g

def load_config(path):
    if not os.path.isfile(path):
        V2("Note: no custom configuration file at: %r" % path)
        return None

    try:
        conf = {}
        execfile(path, conf)
    except Exception, e:
        # TODO: give more helpful errors
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
            if clean_build and trace_graph.is_static_target(dep):
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

def print_excessive_manifest_dependencies(manifest_graph, trace_graph):
    manifest_targets = set(manifest_graph.target_deps_closure.iterkeys())
    traced_targets = set(trace_graph.target_deps_closure.iterkeys())
    excessive_by_targets = defaultdict(list)
    for x in sorted(manifest_targets - traced_targets):
        if ninja_incremental_graph.is_phony_target(x):
            continue
        path_to_top = manifest_graph.get_any_path_to_top(x)
        immediate_parent = path_to_top[0]
        excessive_by_targets[immediate_parent].append(x)

    if not excessive_by_targets:
        info("No issues!")
        return []

    warn("Targets with excessive dependenies: %d" % len(excessive_by_targets))
    for t in manifest_graph.sorted_by_products_num(excessive_by_targets.keys(), reverse=True):
        deps = excessive_by_targets[t]
        V1("%s (%d excessive deps): %r {> '%s'}" % (t, len(deps), deps, "' > '".join(path_to_top)))
    return excessive_by_targets

def print_missing_dependencies(manifest_graph, missing, ignored_missing, clean_build):
    dtype="ORDER " if clean_build else ""
    for t, t_deps in missing.iteritems():
        error("target '%s' is missing %sdependencies on: %r" % (t, dtype, t_deps))
        #TODO: print path from t to top in verbose mode?

    if ignored_missing:
        warn("%sDependency errors inhibited for %d targets due to implicit dependency rules" % (dtype, len(ignored_missing)))
    for t, t_ignored_deps in ignored_missing.iteritems():
        V1("Ignoring missing %sdependencies of '%s' on %r" % (dtype, t, t_ignored_deps))
        #TODO: print path from t to top in verbose mode?

def print_targets_by_ranks(graph):
    for rank in sorted(graph.targets_by_ranks.keys(), reverse=True):
        lst = ", ".join(graph.targets_by_ranks[rank])
        V0("Rank %2d: %5d targets) [%s]" % (rank, len(graph.targets_by_ranks[rank]), lst))

def print_targets_by_depending_products(graph):
    static_wanted_sources = graph.targets_by_ranks[0]
    nonstatic_wanted_rules = list(e for e in graph.target2edge.itervalues() if e.rank)
    bins = [list() for x in xrange(0,10)]
    for t in graph.sorted_by_products_num(static_wanted_sources, reverse=True):
        score = len(graph.get_product_rules_closure(t))
        prct = score*100.0/len(nonstatic_wanted_rules)
        bin = int(prct / 10)
        bins[bin].append((t, score, prct))
        V2("%5d (%2.0f%%): %r" % (score, prct, t))
    for i, bin in enumerate(bins):
        if not bin:
            continue
        V0("[%2d-%2d%%]: %5d targets [%s]" % (
            10*i, 10*i+10, len(bin),
            ", ".join("%s(%d%%)" % (b[0], b[2]) for b in bin)))
    return bins

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='depslint')
    parser.add_argument('-C', dest='dir', help='change to DIR before doing anything else')
    parser.add_argument('-f', dest='manifest', default=_DEFAULT_MANIFEST, help='specify input ninja manifest')
    parser.add_argument('-r', dest='tracefile', default=_DEFAULT_TRACEFILE, help='specify input trace file')
    parser.add_argument('--conf', help='load custom configuration from CONF')
    parser.add_argument('--stats', choices=['all'], help='Evaluate and print build tree statitics')
    parser.add_argument('-v', dest='verbose', action='count', default=0, help='increase verbosity level')
    parser.add_argument('--version', action='version', version='%(prog)s: git')
    parser.add_argument('targets', nargs='*', help='specify targets to verify, as passed to ninja when traced')
    args = parser.parse_args()

    # Set global verbosity level
    _verbose = args.verbose
    _set_logger(filename='depslint.log')

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

    ### Parsing inputs
    info("Parsing Ninja manifest..")
    ninja_parser = NinjaManifestParser(file(args.manifest, "r"))
    info("Parsing Trace log..")
    trace_parser = TraceParser(file(args.tracefile, "r"))

    # If 'default' was specified in nija manifest, use it if no targets
    #  were selected in command line.
    wanted = args.targets or ninja_parser.get_default_targets()

    ### Build graphs
    ninja_clean_build_graph = create_graph(args.manifest, ninja_parser, wanted, clean_build_graph=True)
    ninja_incremental_graph = create_graph(args.manifest, ninja_parser, wanted, clean_build_graph=False)
    # Note: for now, always build a complete (e.g., all-targets-wanted) trace-graph
    trace_graph = create_graph(args.tracefile, trace_parser, targets=[])

    ### Verification passes
    H0()
    info("=== Pass #1: checking clean build order constraints ===")
    info("=== (may lead to clean build failure or, rarely, to incorrect builds) ===")
    missing, ignored = compare_dependencies(trace_graph, ninja_clean_build_graph, clean_build=True)
    if missing or ignored:
        info("Errors: %d, Ignored: %d" % (len(missing), len(ignored)))
        print_missing_dependencies(ninja_clean_build_graph, missing, ignored, clean_build=True)
    else:
        info("No issues!")

    H0()
    info("=== Pass #2: checking for missing dependencies ===")
    info("=== (may lead to incomlete incremental builds if any) ===")
    missing, ignored = compare_dependencies(trace_graph, ninja_incremental_graph, clean_build=False)
    if missing or ignored:
        info("Errors: %d, Ignored: %d" % (len(missing), len(ignored)))
        print_missing_dependencies(ninja_incremental_graph, missing, ignored, clean_build=False)
    else:
        info("No issues!")

    ### Statistics passes
    if args.stats:
        H0()
        info("=== Statistics ===")

        info("=== Listing targets in manifest, not in the traces ===")
        info("=== (these are adding an unnecessary overhead on the build system) ===")
        info("=== (or indicating incomplete trace file - have you traced a clean build?) ===")
        print_excessive_manifest_dependencies(ninja_incremental_graph, trace_graph)

        H0()
        info("=== Targets rank histograms ===")
        info("=== ('rank' is target distance from the bottom of the graph) ===")
        info("=== (e.g., a minimal number of sequential tasks to rebuild a target) ===")
        V0("=== Targets from '%s' by order-dependencies rank ===" % args.manifest)
        print_targets_by_ranks(ninja_clean_build_graph)

        V0("=== Targets from '%s' by rebuild-dependencies rank ===" % args.manifest)
        print_targets_by_ranks(ninja_incremental_graph)

        V0("=== Targets from TRACE by rank ===")
        print_targets_by_ranks(trace_graph)

        #TODO: warn about any duplicate target build detected when tracing.
        #TODO: factor out 'reload' targets
        # warn("Detected multiple rules modifying targets:")
        # print trace_graph.duplicate_target_rules()

        H0()
        info("=== Targets by number of products ===")
        info("=== (e.g., how many targets are rebuilt if 'x' is touched) ===")
        print_targets_by_depending_products(ninja_incremental_graph)

    info("=== That's all! ===")
    sys.exit(0)

    # TODO: try-except..
