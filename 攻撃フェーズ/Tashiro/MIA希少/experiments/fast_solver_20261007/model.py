"""Exact, integer-only model for the six-team identification experiment.

No network, contest submission, scorer, or non-MIA writes. This is the ideal
count-vector oracle, not the mean-TPR oracle used by the contest.
"""
from itertools import combinations, permutations, product
import hashlib
import numpy as np

IDS = (47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675)
TEAMS = (6, 10, 11, 16, 17, 25)
CAPS = (1, 2, 1, 2, 1, 2)
Q3 = (590, 436, 1547, 440, 1095, 433)
Q4_30 = (432, 1598, 1108, 721, 557, 1096)
N = 11
FULL = (1 << N) - 1
POPC = np.array([x.bit_count() for x in range(1 << N)], dtype=np.uint8)
MASKS = np.arange(1 << (N-1), dtype=np.uint16)
WEIGHTS = np.array([3**t for t in range(6)], dtype=np.int32)
ALL_QUERIES = (1 << len(MASKS)) - 1


def decode(code: int) -> tuple[int, ...]:
    return tuple((int(code) // (3**t)) % 3 for t in range(6))


def validate(states: np.ndarray) -> np.ndarray:
    x = np.asarray(states)
    if x.ndim != 2 or x.shape[1] != 6 or not np.issubdtype(x.dtype, np.integer):
        raise ValueError('states must be an N by 6 integer matrix')
    if np.any(x < 0) or np.any(x > FULL):
        raise ValueError('out-of-range membership mask')
    x = x.astype(np.uint16)
    if not np.all(POPC[x] == np.array(CAPS)):
        raise ValueError('membership cardinality mismatch')
    for t in range(6):
        for u in range(t):
            if np.any(x[:,t] & x[:,u]):
                raise ValueError('overlapping membership')
    if len(np.unique(x, axis=0)) != len(x):
        raise ValueError('duplicate states')
    return x


def canonical_rows(states: np.ndarray) -> np.ndarray:
    x = np.asarray(states, dtype=np.uint16)
    if len(x) == 0: return x.reshape(0, 6)
    return x[np.lexsort(tuple(x[:,t] for t in range(5,-1,-1)))]


def state_key(states: np.ndarray) -> bytes:
    # Exact bytes, not only a hash: hash collisions must not prove UNSAT.
    return canonical_rows(states).astype('<u2').tobytes()


def fingerprint(states: np.ndarray) -> str:
    return hashlib.sha256(state_key(states)).hexdigest()


def codes(states: np.ndarray, query) -> np.ndarray:
    q = np.asarray(query, dtype=np.int64)
    if q.shape != (6,) or np.any(q < 0) or np.any(q > FULL):
        raise ValueError('query must contain six 11-bit masks')
    # Cast BEFORE multiplying: uint8 overflow would silently merge outcomes.
    h = POPC[np.asarray(states, dtype=np.uint16) & q.astype(np.uint16)]
    return h.astype(np.int32) @ WEIGHTS


def generate(code: int | None = None) -> np.ndarray:
    target = None if code is None else decode(code)
    opts = []
    for t, cap in enumerate(CAPS):
        masks = [sum(1 << i for i in c) for c in combinations(range(N), cap)]
        if target is not None:
            masks = [m for m in masks if (m & Q3[t]).bit_count() == target[t]]
        opts.append(masks)
    result = []
    def rec(t, used, row):
        if t == 6:
            result.append(tuple(row)); return
        for m in opts[t]:
            if not used & m:
                rec(t+1, used|m, row+[m])
    rec(0, 0, [])
    return np.asarray(result, dtype=np.uint16).reshape(-1,6)


def partition(states, query):
    c = codes(states, query)
    return {int(r):states[c == r] for r in np.unique(c)}


def verified_q3_automorphisms():
    """Find actual query-incidence automorphisms, not arbitrary team swaps.

    Each result contains old->new candidate and team permutations. Only exact
    incidence-preserving permutations are accepted; generated transformations
    preserve capacities and the legal-query family of the ideal model.
    """
    profiles = [tuple((Q3[t] >> i) & 1 for t in range(6)) for i in range(N)]
    dest = {}
    for i,p in enumerate(profiles): dest.setdefault(p, []).append(i)
    result = []
    for ps in permutations((0,2,4)):
        for pp in permutations((1,3,5)):
            tm = [0]*6
            for src,dst in zip((0,2,4),ps): tm[src] = dst
            for src,dst in zip((1,3,5),pp): tm[src] = dst
            src_profiles = {}
            for i,p in enumerate(profiles):
                newp = [0]*6
                for t in range(6): newp[tm[t]] = p[t]
                src_profiles.setdefault(tuple(newp), []).append(i)
            if set(src_profiles) != set(dest): continue
            if any(len(src_profiles[p]) != len(dest[p]) for p in dest): continue
            keys = sorted(dest)
            for choices in product(*(permutations(dest[p]) for p in keys)):
                pm = [0]*N
                for p,targets in zip(keys,choices):
                    for old,new in zip(src_profiles[p],targets): pm[old] = new
                assert sorted(pm) == list(range(N))
                assert all(((Q3[t] >> i)&1) == ((Q3[tm[t]] >> pm[i])&1)
                           for t in range(6) for i in range(N))
                result.append((tuple(pm), tuple(tm)))
    return result


def transform_code(code, tm):
    h = decode(code); out = [0]*6
    for t in range(6): out[tm[t]] = h[t]
    return sum(out[t]*3**t for t in range(6))
