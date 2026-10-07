"""Unoptimized, exact reference. Reconstructs the earlier mask-enumeration method.

This is a benchmark baseline, not a claim to reproduce past wall-clock logs.
Its arithmetic and loops are separate from the optimized implementation.
"""
from functools import lru_cache
from itertools import combinations
from collections import defaultdict

@lru_cache(maxsize=None)
def one(f):
    if len(f) <= 1: return True
    if len(f) > int(f[0]).bit_count()+1: return False
    for q in range(1024):
        if len({(int(m)&q).bit_count() for m in f}) == len(f): return True
    return False

@lru_cache(maxsize=None)
def valid(f):
    value = 0
    for q in range(1024):
        groups = defaultdict(list)
        for m in f: groups[(int(m)&q).bit_count()].append(int(m))
        if all(one(tuple(sorted(g))) for g in groups.values()): value |= 1 << q
    return value


def domains(states):
    result = []
    for t in range(6):
        groups = defaultdict(set)
        for row in states:
            key = tuple(int(row[u]) for u in range(6) if u != t)
            groups[key].add(int(row[t]))
        allowed = (1 << 1024)-1
        for f in groups.values():
            allowed &= valid(tuple(sorted(f)))
            if not allowed: break
        result.append(allowed)
    return result


def screen(states, query):
    children = defaultdict(list)
    for row in states:
        code = sum((int(m)&int(q)).bit_count()*3**t
                   for t,(m,q) in enumerate(zip(row,query)))
        children[code].append(row)
    bad = []
    for code,rows in children.items():
        for t,d in enumerate(domains(rows)):
            if d == 0: bad.append((code,t,len(rows)))
    return sorted(bad)


def last_query(states):
    n = len(states)
    if n <= 1: return [0]*6
    pairs = list(combinations(range(n),2)); full = (1 << len(pairs))-1
    families = []
    for t in range(6):
        lookup = {}
        for q in range(1024):
            vals = [(int(row[t])&q).bit_count() for row in states]
            coverage = sum(1 << k for k,(i,j) in enumerate(pairs) if vals[i] != vals[j])
            lookup.setdefault(coverage,q)
        families.append(list(lookup.items()))
    order = sorted(range(6),key=lambda t:len(families[t]))
    suffix = [0]*7
    for i in range(5,-1,-1):
        u = 0
        for c,_ in families[order[i]]: u |= c
        suffix[i] = suffix[i+1]|u
    memo = set(); choice=[0]*6
    def dfs(i, covered):
        if covered == full: return choice.copy()
        if i == 6 or covered|suffix[i] != full: return None
        key=(i,covered)
        if key in memo: return None
        t=order[i]
        for cov,q in sorted(families[t],key=lambda cq:(cq[0]&~covered).bit_count(),reverse=True):
            choice[t]=q
            r=dfs(i+1,covered|cov)
            if r is not None: return r
        memo.add(key); return None
    return dfs(0,0)
