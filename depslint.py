#!/usr/bin/env python
import os
import re
import sys
from collections import namedtuple

_DEPS_LOG       = 'deps.lst'
_BUILD_MANIFEST = 'build.ninja'

_IGNORED_PREFIXES = ['kernel/cached-out', '../.git', '.mkmp', '/usr']
_IGNORED_SUFFICES = ['.dd', '.d']
_IGNORED_TARGETS = ['version_info/scm_info', 'version_info/contained_commits']

ignored_set = set()
def is_ignored(target):
    return any(target.startswith(prefix) for prefix in _IGNORED_PREFIXES) or \
           any(target.endswith(suffix) for suffix in _IGNORED_SUFFICES)

def filter_ignored(targets):
    return [t for t in targets if not is_ignored(t)]

class MakeParser:
    _depfile_parse_re = re.compile(r'\s*(?P<targets>.*?)\s*'
                                   r'(?<!\\):'
                                   r'\s*(?P<deps>.*?)\s*$', re.DOTALL)
    _depfile_split_re = re.compile(r'(?<!\\)[\s]+')
    _depfile_unescape_re = re.compile(r'\\([ \\?*$|])')

    def parse_depfile(self, path):
        with file(path) as fh:
            buf = fh.read()
            buf = buf.replace('\\\n','') # unescape '\n's
            match = re.match(self._depfile_parse_re, buf)
            if not match or not match.group('targets'):
                raise Exception("Error parsing depfile: '%s'" % buf)
            targets = re.split(self._depfile_split_re, match.group('targets'))
            deps = re.split(self._depfile_split_re, match.group('deps'))
            return ([os.path.normpath(self._unescape(t)) for t in targets],
                    [os.path.normpath(self._unescape(i)) for i in deps])

    def _unescape(self, string):
        return re.sub(self._depfile_unescape_re, r'\1', string)


class NinjaManifestParser:
    def __init__(self, path):
        self.path = path
        self.lineno = 0

        self.global_attributes = dict()
        self.edges = list()
        self.targets = dict()

        # Initializing rules list with a 'phony' rule
        self.rules = dict(phony=dict(attributes=[]))

        self._parse()
        # FIXME: should be able to do this 'per tested target tree'
        self._load_depfiles()

    def _parse(self):
        info("Parsing manifest...")
        with file(self.path) as fh:
            for blk in self._iterate_manifest_blocks(fh):
                #print "blk >> %s .. (%d lines)" % (blk[0], len(blk))
                if blk[0].startswith('build '):
                    self._handle_build_blk(blk)
                elif blk[0].startswith('rule '):
                    self._handle_rule_blk(blk)
                elif blk[0].startswith('default '):
                    self._handle_default_blk(blk)
                else:
                    self._handle_globals(blk)
        info("done")

    def _load_depfiles(self):
        info("Loading depfiles...")
        parser = MakeParser()
        for edge in self.edges:
            depfile = self.eval_edge_attribute(edge, 'depfile')
            if not depfile:
                continue
            if not os.path.isfile(depfile):
                warn("Depfile '%s' couldn't be found, ignoring" % (depfile, ))
                continue
            targets, inputs = parser.parse_depfile(depfile)
            if set(targets) ^ set(edge['targets']):
                # Note: ninja does't accept depfile specifying more
                # than one target, we can't care less.
                raise Exception("Depfile targets list %r doesn't match edge targets %r" % (targets, edge['targets']))
            edge['indepfile'] = inputs
        info("done")

    def _handle_globals(self, blk):
        self.global_attributes.update(self._parse_attributes(blk))

    def _handle_default_blk(self, blk):
        pass

    _build_re = re.compile(r'build\s+(?P<out>.+)\s*'+
                           r'(?<!\$):\s*(?P<rule>\S+)'+
                           r'\s*(?P<all_deps>.*)\s*$')
    def _handle_build_blk(self, blk):
        match = re.match(self._build_re, blk[0])
        if not match:
            raise Exception("Error parsing manifest at line:%d: '%s'" % (self.lineno-len(blk), blk[0]))
        outs, rule, all_deps = match.groups()
        ins, implicit, order = self._split_deps(all_deps)
        attributes = self._parse_attributes(blk[1:])
        targets = self._split_and_unescape(outs)
        attributes.update({'out':outs, 'in':ins})
        # TODO: use a named tupple instead
        edge = {'targets':targets,
                'in':self._split_and_unescape(ins),
                'implicit':self._split_and_unescape(implicit),
                'indepfile':[],
                'order':self._split_and_unescape(order),
                'rule':rule,
                'attributes':attributes}
        #print "Edge>> ", edge
        self.edges.append(edge)
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
        attributes = dict()
        for line in blk:
            match = re.match(self._attr_re, line)
            if not match:
                raise Exception("Error parsing manifest, expecting key=val, got: '%s'" % line)
            attributes[match.group('k')] = match.group('v')
        return attributes

    def _iterate_manifest_blocks(self, fh):
        # Following stripping comments and handling escaped EOLs,
        #  'block' always starts with a 'header' at position '0',
        #  then optionally followed by a couple indented key = val lines.
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
        for line in file(self.path):
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

    def _unescape(self, string):
        # Unescape '$ ', '$:', '$$' sequences
        return re.sub(r'\$([ :$])', r'\1', string)

    _deps_sep_re = re.compile(r'(?<!\$)\s+') # Unescaped spaces
    def _split_and_unescape(self, s):
        if not s:
            return []
        return [self._unescape(s) for s in re.split(self._deps_sep_re, s) if s != '']

    _split_all_deps_re = re.compile(r'(?P<in>.*?)'                  # Explicit deps
                                    r'(\s*(?<!\$)\|(?P<deps>.*?))?' # Unescaped | + implicit deps
                                    r'(\s*(?<!\$)\|\|(?P<ord>.*?))?'# Unescaped || + order deps
                                    r'\s*$')
    def _split_deps(self, s):
        if not s or s.isspace():
            return ("", "", "")
        match = re.match(self._split_all_deps_re, s)
        if not match:
            raise Exception("Error parsing deps: '%s'" % (s,))
        ins, implicit, order = match.group('in'), match.group('deps'), match.group('ord')
        return (ins or "", implicit or "", order or "")

    _attr_sub_re = re.compile('(?<!\$)\$(\{)?(?P<attr>\w+)(?(1)})') # $attr or ${attr}
    def _eval_edge_attribute(self, edge, attribute):
        raw_attribute = self.get_edge_raw_attribute(edge, attribute)
        #print ">> raw attribute: '%s'" % raw_attribute
        def evaluator(match):
            #print ">> Evaluating replacement for attr:", match.group('attr')
            return self._eval_edge_attribute(edge, match.group('attr'))
        return re.sub(self._attr_sub_re, evaluator, raw_attribute)

    def get_edge_raw_attribute(self, edge, attribute):
        rule = self.rules[edge['rule']]
        for scope in (edge['attributes'], rule['attributes'], self.global_attributes):
            if attribute in scope:
                return scope[attribute]
        # TBD: using empty strings for undefined attributes
        return ""

    def eval_edge_attribute(self, edge, attribute):
        return self._unescape(self._eval_edge_attribute(edge, attribute))

    def get_edge(self, path):
        return self.targets[path]

    def iterate_edges(self):
        return iter(self.edges)

class Graph:
    Node = namedtuple('Node', ['provides', 'requires'])

    def __init__(self):
        self.nodes = list()
        self.targets = dict()
        self.declared_targets = list()
        self.duplicate_target_rules = set()

    def add_node_from_tok(self, tok):
            assert 'inputs' in tok and 'outputs' in tok
            node = Graph.Node(provides=tok['outputs'], requires=tok['inputs'])
            self.nodes.append(node)
            for o in tok['outputs']:
                if self.targets.get(o, None):
                    self.duplicate_target_rules.add(o)
                self.targets[o] = node

    def check_for_duplicate_targets(self):
        if self.duplicate_target_rules:
            warn("Detected multiple rules modifying targets:")
            for t in sorted(self.duplicate_target_rules):
                print "  %s" % t

    def build_from_trace(self, path):
        info("Loading build traces...")
        for line in file(path):
            tok = eval(line)
            self.add_node_from_tok(tok)
            self.declared_targets.append(tok['target'])

    def build_from_ninja_manifest(self, path):
        parser = NinjaManifestParser(path)
        for e in parser.iterate_edges():
            tok = dict(outputs=e['targets'], inputs=e['in'] + e['implicit'] + e['indepfile'] + e['order'])
            self.add_node_from_tok(tok)

    def get_top_targets(self):
        top_targets_set = set(self.targets.keys())
        for target, node in self.targets.iteritems():
            top_targets_set.difference_update(node.requires)
        return top_targets_set

    def target_iterate_deps_closure(self, root, targets_only=False, depends={}):
        frontend = [root]
        visited = set(frontend)
        depends[root] = None
        while frontend:
            path = frontend.pop(0)
            node = self.targets.get(path, None)
            if node and path not in _IGNORED_TARGETS:
                for p in node.requires:
                    if p in visited:
                        continue
                    frontend.append(p)
                    visited.add(p)
                    depends[p] = path
                    if not targets_only or p in self.targets:
                        yield p

    @classmethod
    def backtrace_to_root(cls, depends, p):
        l = list()
        while p:
            l.append(p)
            p = depends[p]
        return l

    def print_target_deps(self, root, targets_only=False):
        for t in self.target_iterate_deps_closure(root, targets_only):
            print t

def warn(msg):
    print "\033[1;33mWARNING: %s\033[0m" % msg

def info(msg):
    print "\033[1;32mINFO: %s\033[0m" % msg

if __name__ == '__main__':
    if len(sys.argv) < 2:
        # Note: requires a single string to make sure that symbols
        #   as '>' and '|' are quoted and interpreted properly.
        print  >>sys.stderr, "Usage: %s [<target list>]" % sys.argv[0]
        sys.exit(-1)
    target = sys.argv[1]
    manifest = 'build.ninja'

    #p = MakeParser()
    #print p.parse_depfile('dleia/b.d')
    #sys.exit(-1)

    # Testing ninja manifest parser
    #parser = NinjaManifestParser(manifest)
    # print "** Rules", parser.rules
    # print "** Globals", parser.globals
    # 'common/system/no_trace/.xmake.build/sys_tasklabels.o-05c9cb07af9de7885c3fb651a271e4b1.o'
    # 'version_info/scm_info'
    # attr: 'target_rule_pickle'
    #t = parser.get_edge('default')
    #t = parser.get_edge('version_info/scm_info')
    #t = parser.get_edge('common/system/no_trace/.xmake.build/sys_tasklabels.o-05c9cb07af9de7885c3fb651a271e4b1.o')
    #print 'Target:', t
    #print ' depfile:', parser.eval_edge_attribute(t, 'depfile')
    #print ' command:', parser.eval_edge_attribute(t, 'command')
    #sys.exit(-1)

    # Testing concepts
    g = Graph()
    g.build_from_trace(_DEPS_LOG)

    # Warn about any duplicate target build detected when tracing.
    # FIXME: disabled until will factor the test out to ignore 'reload' targets
    g.check_for_duplicate_targets()

    ng = Graph()
    ng.build_from_ninja_manifest(manifest)

    info("Top targets")
    info("ng")
    print "\n".join(sorted(filter_ignored(ng.get_top_targets())))
    info("g")
    print "\n".join(sorted(filter_ignored(g.get_top_targets())))

    warn("Now mining..")
    gdepends, ngdepends = {}, {}
    gtargets  = set(g.target_iterate_deps_closure(target, depends=gdepends))
    ngtargets = set(ng.target_iterate_deps_closure(target, depends=ngdepends))
    warn("Targets in traces, not in manifest")
    for x in sorted(gtargets - ngtargets):
        if not is_ignored(x):
            print "'%s': %s" % (x, " -> ".join(Graph.backtrace_to_root(gdepends, x)))
        else:
            ignored_set.add(x)

    warn("Target in manifest, not in the traces")
    for x in sorted(ngtargets - gtargets):
        if not is_ignored(x):
            print "'%s': %s" % (x, " -> ".join(Graph.backtrace_to_root(ngdepends, x)))
        else:
            ignored_set.add(x)

    info("Totally: %d targets were ignored (prefixed with one of %r or ending with: %r)" % (
        len(ignored_set),_IGNORED_PREFIXES, _IGNORED_SUFFICES))

    # Test
    #print "Manifest deps:"
    #ng.print_target_deps('build.ninja')
    #sys.exit(0)
