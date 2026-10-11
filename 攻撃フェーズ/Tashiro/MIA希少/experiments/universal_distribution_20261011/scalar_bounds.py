"""Independent exact scalar lower-bound replay, no search implementation imports."""
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
import hashlib
import json
import time
FULL = 2047

def response(row: tuple[int, ...], query: tuple[int, ...]) -> int:
    # Deliberately not calling model.codes or partition.
    return sum((m & q).bit_count() * 3 ** t
               for t, (m, q) in enumerate(zip(row, query)))

def normalize(query: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    inv = tuple(q.bit_count() > 5 for q in query)
    return tuple(FULL ^ q if flip else q for q, flip in zip(query, inv)), inv

def rate(n: int, h: int, m: int = 5) -> Fraction:
    return Fraction((24 - n) * h, 100 * n * (m - h))

def scalar_states(n: int) -> list[tuple[int, ...]]:
    return sorted(((1 << j) | (1 << k), 0, 0, 0, 0, 0)
                  for j, k in combinations(range(n), 2))

def scalar_replay(tree: dict, rows: list[tuple[int, ...]]) -> dict:
    leaves = []
    hist = Counter()
    query_count = 0
    trace = hashlib.sha256()
    padding_values = set()
    mean_values = {rate(2, h): h for h in range(3)}
    assert len(mean_values) == 3

    def walk(node, states, depth):
        nonlocal query_count
        assert states and depth <= 4
        if node.get("leaf") is True:
            assert len(states) == 1
            leaves.extend(states); hist[depth] += 1
            return
        q = tuple(node["query"])
        assert len(q) == 6 and q[1:] == (0,) * 5 and depth < 4
        assert all(0 <= x <= FULL for x in q)
        query_count += 1
        physical, inv = normalize(q)
        padding = 5 - physical[0].bit_count()
        assert 0 <= padding <= 13
        padding_values.add(padding)
        groups = {}
        for row in states:
            r = response(row, q)
            groups.setdefault(r, []).append(row)
            h = (row[0] & physical[0]).bit_count()
            recovered = mean_values[rate(2, h)]
            if inv[0]: recovered = 2 - recovered
            assert r == recovered
            trace.update(json.dumps([depth, row, q, r, physical, h, str(rate(2, h))], separators=(",", ":")).encode() + b"\n")
        if node.get("terminal") is True:
            assert all(len(v) == 1 for v in groups.values())
            for r in sorted(groups): walk({"leaf": True}, groups[r], depth + 1)
        else:
            assert set(node["children"]) == {str(r) for r in groups}
            for r in sorted(groups): walk(node["children"][str(r)], groups[r], depth + 1)

    walk(tree, rows, 0)
    assert sorted(leaves) == rows
    return {"status": "PASS", "leaf_count": len(leaves), "all_leaves_singleton": True,
            "maximum_queries": max(hist), "path_length_histogram": dict(sorted(hist.items())),
            "query_node_count": query_count, "trace_sha256": trace.hexdigest(),
            "normalized_subset_size_at_most": 5, "M": 5,
            "padding_values": sorted(padding_values), "formula_mean_decode_replay": True}

def mathematical_pair_lower_bound(n: int) -> dict:
    assert n in (10, 11)
    checks = []
    for s in range(n // 2 + 1):
        branches = [(n - s) * (n - s - 1) // 2, s * (n - s), s * (s - 1) // 2]
        if max(branches) > 3 ** 3:
            checks.append({"selected_candidates": s, "first_branch_counts": branches,
                           "obstruction": "first branch exceeds 27 remaining leaves"})
        else:
            cuts = []
            for x in range(s + 1):
                for y in range(n - s + 1):
                    sizes = [(s - x) * (n - s - y),
                             x * (n - s - y) + (s - x) * y, x * y]
                    cuts.append({"x": x, "y": y, "second_branch_counts": sizes,
                                 "maximum_branch": max(sizes)})
            best = min(c["maximum_branch"] for c in cuts)
            assert best > 9
            checks.append({"selected_candidates": s, "first_branch_counts": branches,
                           "obstruction": "cross-pair branch leaves more than 9 states after every possible second query",
                           "all_second_query_cell_size_choices": cuts, "best_second_maximum_branch": best})
    return {"status": "UNSAT", "candidate_count": n, "pair_count": n * (n - 1) // 2,
            "depth": 4, "complement_symmetry_first_subset_sizes": [0, n // 2],
            "all_first_subset_sizes_checked": checks,
            "scope": "all adaptive subset-count queries, including target-specific choices"}

def fixed_seven_certificate() -> dict:
    words = [0, 1, 3, 7, 9, 10, 12]
    signatures = [tuple(((words[i] >> t) & 1) + ((words[j] >> t) & 1) for t in range(4))
                  for i, j in combinations(range(7), 2)]
    assert len(set(signatures)) == 21
    return {"status": "SAT_FIXED_FOUR", "adaptive_solver_run": False,
            "reason": "fixed four queries already solve all pairs; user requested skipping adaptive search in fixed-SAT cases",
            "candidate_count": 7, "state_count": 21, "codewords_decimal": words,
            "all_distinct_pair_signatures_unique": True,
            "query_masks": [sum(((w >> t) & 1) << i for i, w in enumerate(words)) for t in range(4)]}

def independently_check_pair_nine() -> dict:
    """Complete pure-Python scalar AND/OR search; no old Solver functions."""
    initial = tuple(row[0] for row in scalar_states(9))
    nodes = Counter()

    @lru_cache(maxsize=None)
    def feasible(states: tuple[int, ...], depth: int) -> bool:
        nodes[depth] += 1
        if len(states) <= 1:
            return True
        if depth == 0 or len(states) > 3 ** depth:
            return False
        active = 0
        for member_pair in states:
            active |= member_pair
        # Inactive candidate bits do not affect answers.  Fixing one active
        # candidate to zero leaves one of every complement-equivalent pair;
        # all member pairs have the same size two, so complements return 2-h.
        bits = [i for i in range(11) if active >> i & 1][:-1]
        query_masks = [sum(1 << bits[j] for j in range(len(bits)) if word >> j & 1)
                       for word in range(1 << len(bits))]
        seen = set()
        choices = []
        for q in query_masks:
            groups = [[], [], []]
            for member_pair in states:
                groups[(member_pair & q).bit_count()].append(member_pair)
            children = tuple(sorted(tuple(v) for v in groups if v))
            if len(children) <= 1 or children in seen:
                continue
            seen.add(children)
            if max(map(len, children)) > 3 ** (depth - 1):
                continue
            if depth == 1:
                return True
            choices.append(children)
        choices.sort(key=lambda p: (max(map(len, p)), sum(len(v) ** 2 for v in p)))
        for children in choices:
            if all(feasible(child, depth - 1) for child in sorted(children, key=len, reverse=True)):
                return True
        return False

    start = time.perf_counter()
    answer = feasible(initial, 4)
    assert not answer
    info = feasible.cache_info()
    return {"status": "UNSAT", "candidate_count": 9, "state_count": 36, "depth": 4,
            "method": "complete independent pure-Python AND/OR search of all active subset masks modulo complement and identical response partitions",
            "old_solver_imported_or_called": False, "query_generator_complete": True,
            "nodes_by_remaining_depth": dict(sorted(nodes.items())),
            "cache_hits": info.hits, "cache_misses": info.misses,
            "elapsed_seconds": time.perf_counter() - start}
