"""Safe bounded AND/OR search for the ideal count-vector oracle.

Depth 1 is exact (until an explicit budget is hit). For deeper nodes, UNSAT is
returned only by a sound lower bound or after exhaustive query enumeration.
Heuristic/beam exhaustion returns UNKNOWN and is never cached as UNSAT.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import combinations, product
import math
import time
import numpy as np
from model import (N, CAPS, FULL, MASKS, POPC, ALL_QUERIES, codes,
                   canonical_rows, state_key, fingerprint, partition)


class Status(str, Enum):
    SAT = 'SAT'
    UNSAT = 'UNSAT'
    UNKNOWN = 'UNKNOWN'


@dataclass
class Result:
    status: Status
    tree: dict | None = None
    reason: str = ''


class BudgetExceeded(Exception):
    pass


def check_deadline(deadline):
    if time.perf_counter() > deadline:
        raise BudgetExceeded


def bitmask_of_bool(values) -> int:
    return int.from_bytes(np.packbits(values, bitorder='little').tobytes(), 'little')


@lru_cache(maxsize=50000)
def active_query_masks(members: tuple[int,...]) -> tuple[int,...]:
    union = 0
    for m in members: union |= int(m)
    active = [i for i in range(N) if union >> i & 1]
    # Complement-equivalent questions: fixing one ACTIVE bit to zero is safe
    # because membership cardinality is constant within this team.
    if not active: return (0,)
    active = active[:-1]
    return tuple(sum((1 << x) for j,x in enumerate(active) if k >> j & 1)
                 for k in range(1 << len(active)))


@lru_cache(maxsize=100000)
def scalar_one(members: tuple[int,...]) -> bool:
    members = tuple(sorted(set(members)))
    if len(members) <= 1: return True
    cap = int(members[0]).bit_count()
    if len(members) > cap+1: return False
    for q in active_query_masks(members):
        if len({(int(m)&q).bit_count() for m in members}) == len(members):
            return True
    return False


@lru_cache(maxsize=100000)
def fiber_valid(members: tuple[int,...]) -> int:
    """Exact set of FIRST masks allowing a scalar fiber to finish in two tests.

    A child block is one-query separable iff it contains no forbidden subset.
    Minimal forbidden subsets have size <= cap+2 (at most 4 here). Checking
    them uses 1024-bit masks, not a Python loop over 1024 first queries.
    """
    f = tuple(sorted(set(map(int, members))))
    if len(f) <= 1: return ALL_QUERIES
    cap = f[0].bit_count()
    assert all(m.bit_count() == cap for m in f)
    if len(f) > (cap+1)**2: return 0
    responses = POPC[np.array(f, dtype=np.uint16)[:,None] & MASKS[None,:]]
    valid = ALL_QUERIES
    for k in range(2, min(len(f),cap+2)+1):
        for js in combinations(range(len(f)), k):
            sub = tuple(f[j] for j in js)
            if scalar_one(sub): continue
            same = np.all(responses[list(js)] == responses[js[0]], axis=0)
            valid &= ALL_QUERIES ^ bitmask_of_bool(same)
            if valid == 0: return 0
    return valid


def fiber_groups(states, team):
    """Rows sharing the exact assignments of every OTHER team."""
    other = [t for t in range(6) if t != team]
    if not len(states): return []
    _, inv, counts = np.unique(states[:,other], axis=0,
                               return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind='stable')
    edges = np.cumsum(np.r_[0, counts])
    return [order[edges[g]:edges[g+1]] for g in np.flatnonzero(counts > 1)]


def depth2_domains(states) -> list[int]:
    out = []
    for t in range(6):
        allowed = ALL_QUERIES
        for ix in fiber_groups(states,t):
            allowed &= fiber_valid(tuple(sorted(map(int,states[ix,t]))))
            if allowed == 0: break
        out.append(allowed)
    return out


class FailureCache:
    """Monotone, proved UNSAT subsets; UNKNOWN must never enter this cache.

    A depth-d core contained in one query cell excludes that query at depth
    d+1. Uniform-mask domains turn this into query-generation nogoods.
    Exact row bytes/sets, rather than fingerprints, decide cache equality.
    """
    def __init__(self):
        self.entries = []
        self.keys = set()
        self.hits = 0

    def add(self, states, depth, result):
        if result.status != Status.UNSAT:
            return False
        s = canonical_rows(states)
        key = (state_key(s), int(depth))
        if key in self.keys: return False
        self.keys.add(key)
        uniform = []
        for t in range(6):
            members = np.unique(s[:,t])
            h = POPC[members[:,None] & MASKS[None,:]]
            uniform.append(np.all(h == h[:1], axis=0))
        self.entries.append({'states':s, 'rows':frozenset(map(tuple,s.tolist())),
                             'depth':int(depth), 'reason':result.reason,
                             'uniform':uniform})
        return True

    def contained(self, states, depth):
        possible = [e for e in self.entries if e['depth'] == depth and len(e['states']) <= len(states)]
        if not possible: return None
        rows = frozenset(map(tuple,states.tolist()))
        for e in possible:
            if e['rows'] <= rows:
                self.hits += 1
                return e
        return None

    def rejects(self, query, child_depth, states=None):
        q = [min(int(x), FULL ^ int(x)) for x in query]
        rows = None if states is None else frozenset(map(tuple,states.tolist()))
        for e in self.entries:
            if (e['depth'] == child_depth and (rows is None or e['rows'] <= rows)
                and all(e['uniform'][t][q[t]] for t in range(6))):
                self.hits += 1
                return e
        return None


class ParentScreen:
    """Amortize fixed-other-team fibers across many candidate parent queries.

    Result: all children pass a necessary depth-2 condition, NOT a solution.
    Warm reuse is per (team, mask), not only a repeat of the identical query.
    """
    def __init__(self, states, failure_cache=None):
        self.states = canonical_rows(states)
        self.groups = [fiber_groups(self.states,t) for t in range(6)]
        self.memo = {}
        self.hits = 0
        self.misses = 0
        self.failure_cache = failure_cache or FailureCache()
        self.learned = []

    def constraints(self, t, q):
        q = min(int(q), FULL ^ int(q))  # same partition after complement
        key = (t,q)
        if key in self.memo:
            self.hits += 1
            return self.memo[key]
        self.misses += 1
        s = self.states
        cons = []
        for ix in self.groups[t]:
            h = POPC[s[ix,t] & q]
            for r in np.unique(h):
                sub = ix[h == r]
                if len(sub) <= 1: continue
                allowed = fiber_valid(tuple(sorted(map(int,s[sub,t]))))
                if allowed != ALL_QUERIES:
                    cons.append((int(sub[0]),allowed,sub))
        self.memo[key] = cons
        return cons

    def evaluate(self, query, early=False):
        self.learned = []
        c = codes(self.states,query)
        counts = np.bincount(c,minlength=729)
        failures = []
        for t in range(6):
            allowed_by_child = {}
            contributors = {}
            for row,bits,ix in self.constraints(t,query[t]):
                r = int(c[row])
                if allowed_by_child.get(r,1) == 0: continue
                value = allowed_by_child.get(r,ALL_QUERIES) & bits
                allowed_by_child[r] = value
                contributors.setdefault(r,[]).append((bits,ix))
                if value == 0:
                    failures.append((r,t,int(counts[r])))
                    # Remove redundant fibers, keeping an explicit small
                    # subset whose exact valid-mask intersection is empty.
                    core = contributors[r].copy()
                    for j in range(len(core)-1,-1,-1):
                        intersection = ALL_QUERIES
                        for k,(b,_) in enumerate(core):
                            if j != k: intersection &= b
                        if intersection == 0: core.pop(j)
                    # Shrinking within fibers strengthens the generation
                    # nogood: more masks leave a smaller proved core intact.
                    for j in range(len(core)):
                        other = ALL_QUERIES
                        for k,(b,_) in enumerate(core):
                            if k != j: other &= b
                        _,ix = core[j]
                        for row in list(ix):
                            trial = ix[ix != row]
                            bits = fiber_valid(tuple(sorted(map(int,self.states[trial,t]))))
                            if other & bits == 0:
                                ix = trial;core[j] = (bits,ix)
                    rows = np.unique(np.concatenate([x for _,x in core]))
                    subset = self.states[rows]
                    assert depth2_domains(subset)[t] == 0
                    ans = Result(Status.UNSAT,reason=f'fixed-other-team depth-2 obstruction Team-index {t}')
                    if self.failure_cache.add(subset,2,ans):
                        self.learned.append(self.failure_cache.entries[-1])
                    if early: return failures,counts
        return sorted(failures),counts


def normalize_labels(row):
    first = {}; out = np.empty(len(row),dtype=np.uint8)
    for i,v in enumerate(row):
        v = int(v)
        if v not in first: first[v] = len(first)
        out[i] = first[v]
    return out


def query_options(states, t, allowed=ALL_QUERIES):
    """One representative for each partition; disallowed domains respected."""
    unique, inv = np.unique(states[:,t],return_inverse=True)
    masks = [q for q in range(1024) if allowed >> q & 1]
    if not masks: return []
    mat = POPC[np.array(masks,dtype=np.uint16)[:,None] & unique[None,:]]
    result = []; seen = set()
    for j,q in enumerate(masks):
        norm = normalize_labels(mat[j])
        key = norm.tobytes()
        if key in seen: continue
        seen.add(key)
        result.append((q,norm[inv]))
    return result


def independent_witness_ok(states, query):
    # Pure Python: deliberately not calling model.codes.
    sigs = [tuple((int(m)&int(q)).bit_count() for m,q in zip(row,query))
            for row in states]
    return len(set(sigs)) == len(sigs)


class Solver:
    def __init__(self, beam_width=24, query_budget=96, node_limit=100000,
                 exhaustive_product_limit=5000, sample_queries=0, seed_queries=None):
        self.beam_width = beam_width
        self.query_budget = query_budget
        self.node_limit = node_limit
        self.exhaustive_product_limit = exhaustive_product_limit
        self.sample_queries = sample_queries
        self.seed_queries = seed_queries or []
        self.cache = {}
        self.failure_cache = FailureCache()
        self.cores = []  # negative one-query witnesses (exact state subsets)
        self.stats = {'nodes':0, 'cache_hits':0, 'last_query_calls':0,
                      'fiber_prunes':0, 'query_trials':0, 'core_prunes':0}

    def last_query(self, states, deadline=float('inf')) -> Result:
        s = canonical_rows(states); key = (state_key(s),1)
        if key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[key]
        self.stats['last_query_calls'] += 1
        try:
            check_deadline(deadline)
            if len(s) <= 1:
                result = Result(Status.SAT, {'leaf':True})
            elif len(s) > math.prod(c+1 for c in CAPS):
                result = Result(Status.UNSAT, reason='one-query alphabet bound')
            else:
                n = len(s); a,b = np.triu_indices(n,1)
                full = (1 << len(a)) - 1
                families = []
                for t in range(6):
                    check_deadline(deadline)
                    opts = query_options(s,t)
                    covmap = {}
                    for q,labels in opts:
                        cov = bitmask_of_bool(labels[a] != labels[b])
                        covmap.setdefault(cov,q)
                    # Dominance safe for equal-cost ideal queries. The actual
                    # scorer/padding must be handled in a separate layer.
                    keep = []
                    for cov in sorted(covmap,key=int.bit_count,reverse=True):
                        if any(cov|x == x for x,_ in keep): continue
                        keep.append((cov,covmap[cov]))
                    families.append(keep)
                order = sorted(range(6),key=lambda t:len(families[t]))
                unions = [0]*7
                for i in range(5,-1,-1):
                    u = 0
                    for cov,_ in families[order[i]]: u |= cov
                    unions[i] = unions[i+1] | u
                failed = set(); chosen = [0]*6; nodes = 0
                def dfs(i,covered):
                    nonlocal nodes
                    nodes += 1
                    if nodes > self.node_limit: raise BudgetExceeded
                    if nodes % 256 == 0: check_deadline(deadline)
                    if covered == full: return chosen.copy()
                    if i == 6 or covered|unions[i] != full: return None
                    k = (i,covered)
                    if k in failed: return None
                    t = order[i]
                    ranked = sorted(families[t],
                        key=lambda item:(item[0] & ~covered).bit_count(),reverse=True)
                    seen = set()
                    for cov,q in ranked:
                        new = covered|cov
                        if new in seen: continue
                        seen.add(new); chosen[t] = q
                        # A team may add zero coverage: it MUST NOT be skipped.
                        found = dfs(i+1,new)
                        if found is not None: return found
                    failed.add(k); return None
                witness = dfs(0,0)
                self.stats['nodes'] += nodes
                if witness is None:
                    result = Result(Status.UNSAT, reason='complete pair-cover search')
                    if len(s) <= 24:
                        self.cores.append(frozenset(map(tuple,s.tolist())))
                else:
                    assert independent_witness_ok(s,witness)
                    result = Result(Status.SAT, {'query':witness,'terminal':True})
            self.cache[key] = result
            self.failure_cache.add(s,1,result)
            return result
        except BudgetExceeded:
            return Result(Status.UNKNOWN, reason='time/node budget')

    def candidates(self, states, domains, deadline, child_depth=1):
        parent_rows = frozenset(map(tuple,states.tolist()))
        applicable = [e for e in self.failure_cache.entries
                      if e['depth'] == child_depth and e['rows'] <= parent_rows]
        def rejected(q):
            return any(all(e['uniform'][t][min(int(q[t]),FULL^int(q[t]))]
                           for t in range(6)) for e in applicable)
        if self.sample_queries:
            # A bounded alternative for large nodes: sample only exact fiber
            # domains, then beam-order by complete-query minimax/sumsq. The
            # local moves are a limited-discrepancy extension of the best seed.
            # Inactive bits and response-label complements otherwise create
            # many copies of the same q5 partition in the top beam.
            masks = [[q for q,_ in query_options(states,t,domains[t])] for t in range(6)]
            def score(q):
                cnt = np.bincount(codes(states,q),minlength=729).astype(np.int64)
                return int(cnt.max()),int(cnt@cnt)
            if math.prod(map(len,masks)) <= self.exhaustive_product_limit:
                return sorted((q for q in product(*masks) if not rejected(q)),key=score), True
            rng = np.random.default_rng(int(fingerprint(states)[:16],16))
            trials = set()
            for q in self.seed_queries:
                q = tuple(min(int(x),FULL^int(x)) for x in q)
                if all(domains[t] >> q[t] & 1 for t in range(6)): trials.add(q)
            for _ in range(self.sample_queries):
                trials.add(tuple(int(rng.choice(m)) for m in masks))
            check_deadline(deadline)
            ranked = sorted((q for q in trials if not rejected(q)),key=score)
            if ranked:
                base = ranked[0];alternatives = []
                for t in range(6):
                    check_deadline(deadline)
                    moves = []
                    for m in masks[t]:
                        q = list(base);q[t] = m;q = tuple(q)
                        if not rejected(q): moves.append((score(q),m,q))
                    moves.sort()
                    alternatives.append([m for _,m,_ in moves if m != base[t]][:3])
                    trials.update(q for _,_,q in moves[:4])
                for t,u in combinations(range(6),2):
                    for m,n in product(alternatives[t],alternatives[u]):
                        q = list(base);q[t] = m;q[u] = n;trials.add(tuple(q))
            ranked = sorted((q for q in trials if not rejected(q)),key=score)
            return ranked[:self.query_budget], False
        opts = [query_options(states,t,domains[t]) for t in range(6)]
        count = math.prod(len(o) for o in opts)
        if count <= self.exhaustive_product_limit:
            rows = [tuple(o[0] for o in selected) for selected in product(*opts)
                    if not rejected(tuple(o[0] for o in selected))]
            def score(q):
                cnt = np.bincount(codes(states,q),minlength=729).astype(np.int64)
                return (int(cnt.max()),int(cnt@cnt))
            return sorted(rows,key=score), True
        # Bounded beam: never interpret exhausting this list as UNSAT.
        current = [((),np.zeros(len(states),dtype=np.int32))]
        for t in range(6):
            check_deadline(deadline)
            ranked = []; seen = set()
            for qs,base in current:
                for q,labels in opts[t]:
                    if t == 5 and rejected(qs+(q,)): continue
                    c = base*3 + labels.astype(np.int32)
                    # Canonicalize actual joint partition for deduplication.
                    _,first,inv = np.unique(c,return_index=True,return_inverse=True)
                    ren = np.empty(len(first),dtype=np.int32)
                    ren[np.argsort(first)] = np.arange(len(first))
                    norm = ren[inv]
                    sig = norm.tobytes()
                    if sig in seen: continue
                    seen.add(sig)
                    cnt = np.bincount(norm).astype(np.int64)
                    ranked.append(((int(cnt.max()),int(cnt@cnt)),qs+(q,),c))
            ranked.sort(key=lambda x:x[0])
            current = [(q,c) for _,q,c in ranked[:self.beam_width]]
        return [q for q,_ in current[:self.query_budget]], False

    def solve(self, states, depth, seconds=30.0) -> Result:
        if depth < 0: raise ValueError('depth must be nonnegative')
        try:
            return self._solve(canonical_rows(states),depth,time.perf_counter()+seconds)
        except BudgetExceeded:
            return Result(Status.UNKNOWN, reason='time budget')

    def _solve(self, s, d, deadline):
        check_deadline(deadline)
        key = (state_key(s),d)
        if key in self.cache:
            self.stats['cache_hits'] += 1; return self.cache[key]
        core = self.failure_cache.contained(s,d)
        if core is not None:
            self.stats['core_prunes'] += 1
            return Result(Status.UNSAT,reason='contains proved failure subproblem')
        if len(s) <= 1: return Result(Status.SAT, {'leaf':True})
        if d == 0: return Result(Status.UNSAT, reason='multiple states at depth zero')
        if d == 1: return self.last_query(s,deadline)
        if len(s) > math.prod(c+1 for c in CAPS)**d:
            return Result(Status.UNSAT, reason='alphabet bound')
        domains = depth2_domains(s) if d == 2 else [ALL_QUERIES]*6
        if any(v == 0 for v in domains):
            self.stats['fiber_prunes'] += 1
            r = Result(Status.UNSAT, reason='fixed-other-team depth-2 obstruction')
            self.cache[key] = r
            self.failure_cache.add(s,d,r)
            return r
        queries, exhaustive = self.candidates(s,domains,deadline,d-1)
        unknown_seen = False
        for q in queries:
            check_deadline(deadline); self.stats['query_trials'] += 1
            if self.failure_cache.rejects(q,d-1,s) is not None:
                self.stats['core_prunes'] += 1
                continue
            children = partition(s,q)
            if len(children) <= 1: continue
            children = sorted(children.items(),key=lambda kv:-len(kv[1]))
            subtree = {}; failed = False; this_unknown = False
            for code,child in children:
                ans = self._solve(child,d-1,deadline)
                if ans.status == Status.UNSAT: failed = True; break
                if ans.status == Status.UNKNOWN: this_unknown = True; break
                subtree[str(code)] = ans.tree
            if not failed and not this_unknown:
                result = Result(Status.SAT,{'query':list(q),'children':subtree})
                self.cache[key] = result; return result
            unknown_seen |= this_unknown
        if exhaustive and not unknown_seen:
            result = Result(Status.UNSAT, reason='all query partitions exhausted')
            self.cache[key] = result
            self.failure_cache.add(s,d,result)
            return result
        return Result(Status.UNKNOWN, reason='bounded query generator exhausted')
