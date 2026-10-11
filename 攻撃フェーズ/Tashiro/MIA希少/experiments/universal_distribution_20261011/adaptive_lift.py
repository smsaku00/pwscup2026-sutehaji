"""Read-only independent functions for lifting a3b3 certificates to a4b3.

No search is performed. Candidate slots 0..10, all concrete assignments,
physical padding, exact means, restored paths and extra singleton codes are checked."""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import hashlib
from itertools import combinations, product
import json
from math import factorial, lcm
from pathlib import Path
import sys
from time import monotonic

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

OLD = ROOT / "analysis/full_depth4_20261007"
OLD_REPLAY = ROOT / "analysis/universal_distribution_20261011/adaptive4/a3b3/replay.json"

CAPS_OLD = (1, 2, 1, 2, 1, 2)
CAPS = (1, 1, 1, 1, 2, 2, 2)
M = (5, 6, 7, 8, 5, 6, 8)
Q3 = (590, 436, 1547, 440, 1095, 433)
FULL = (1 << 11)-1
OLD_TO_NEW = (0, 4, 1, 5, 2, 6)
WORDS = (0, 1, 2, 4, 6, 7, 8, 9, 11, 13, 14)
EXTRA_QUERY = tuple(sum(1 << i for i, word in enumerate(WORDS) if word >> q & 1)
                    for q in range(4))
POPC = np.array([x.bit_count() for x in range(2048)], dtype=np.uint8)
OLD_WEIGHTS = np.array([3**t for t in range(6)], dtype=np.uint16)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, sources):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def generate_projection(first_code):
    options = []
    for t, cap in enumerate(CAPS_OLD):
        h = first_code // 3**t % 3
        options.append([sum(1 << i for i in group) for group in combinations(range(11), cap)
                        if sum(bool(Q3[t] >> i & 1) for i in group) == h])
    rows = []
    def visit(t, used, row):
        if t == 6:
            rows.append(tuple(row))
            return
        for mask in options[t]:
            if not mask & used:
                visit(t+1, used | mask, row+[mask])
    visit(0, 0, [])
    return np.asarray(rows, dtype=np.uint16).reshape(-1, 6)


def state_hash(rows):
    order = np.lexsort(tuple(rows[:, t] for t in range(5, -1, -1)))
    return hashlib.sha256(rows[order].astype("<u2", copy=False).tobytes()).hexdigest()


def replay_representative(rows, tree):
    """Every row traverses the actual certificate; record its four queries."""
    count = len(rows)
    queries = np.zeros((count, 4, 6), dtype=np.uint16)
    responses = np.zeros((count, 4), dtype=np.uint16)
    depths = np.zeros(count, dtype=np.uint8)
    queries[:, 0, :] = Q3
    responses[:, 0] = POPC[rows & np.asarray(Q3, dtype=np.uint16)].astype(np.uint16) @ OLD_WEIGHTS
    node_count = 0
    def walk(indices, node, depth):
        nonlocal node_count
        require(len(indices) > 0 and depth <= 4, "Empty or too-deep reachable node")
        if node.get("leaf"):
            require(len(indices) == 1, "Historical leaf contains multiple assignments")
            depths[indices] = depth
            return
        require(depth < 4, "Historical strategy exceeds four queries")
        raw_query = node["query"]
        require(len(raw_query) == 6 and all(type(q) is int and 0 <= q <= FULL for q in raw_query),
                "Historical query is not a legal six-target mask")
        query = np.asarray(raw_query, dtype=np.uint16)
        queries[indices, depth, :] = query
        response = POPC[rows[indices] & query].astype(np.uint16) @ OLD_WEIGHTS
        responses[indices, depth] = response
        node_count += 1
        outcomes = np.unique(response)
        if node.get("terminal"):
            require(len(outcomes) == len(indices), "Historical terminal query collides")
            depths[indices] = depth+1
            return
        require(set(node["children"]) == {str(int(code)) for code in outcomes},
                "Historical reachable children differ")
        for outcome in outcomes:
            walk(indices[response == outcome], node["children"][str(int(outcome))], depth+1)
    walk(np.arange(count), tree, 1)
    require(np.all((depths >= 1) & (depths <= 4)), "Historical path missing")
    return queries, responses, depths, node_count


def exact_encoder():
    answers = list(product(*(range(n+1) for n in CAPS)))
    rates = [[Fraction((24-n)*h, 100*n*(m-h)*7) for h in range(n+1)]
             for n, m in zip(CAPS, M)]
    means = [sum((rates[t][h] for t, h in enumerate(hs)), Fraction()) for hs in answers]
    require(len(means) == len(set(means)) == 432, "a4b3 exact mean alphabet collides")
    denominator = lcm(*(rate.denominator for table in rates for rate in table))
    unit_rates = [np.array([int(rate*denominator) for rate in table], dtype=np.uint64)
                  for table in rates]
    units = np.array([int(mean*denominator) for mean in means], dtype=np.uint64)
    order = np.argsort(units)
    gap = min(right-left for left, right in zip(sorted(means), sorted(means)[1:]))
    return unit_rates, units[order], np.asarray(answers, dtype=np.uint8)[order], denominator, gap


def path_keys(responses):
    out = np.zeros(len(responses), dtype=np.uint64)
    for q in range(4):
        out = out * np.uint64(729) + responses[:, q]
    return out


def concrete_branch(rows, raw_queries, expected_codes, depths, code, unit_rates, mean_units, mean_answers):
    projected_count = len(rows)
    used = np.bitwise_or.reduce(rows, axis=1)
    remaining = FULL ^ used
    require(np.all(POPC[used] == 9) and np.all(POPC[remaining] == 2), "Projection does not leave two candidates")
    first_extra = remaining & (-remaining.astype(np.int32)).astype(np.uint16)
    second_extra = remaining ^ first_extra
    extras = np.stack((first_extra, second_extra), axis=1).reshape(-1)
    full_rows = np.zeros((2*projected_count, 7), dtype=np.uint16)
    full_rows[:, list(OLD_TO_NEW)] = np.repeat(rows, 2, axis=0)
    full_rows[:, 3] = extras
    require(np.all(POPC[full_rows] == np.asarray(CAPS)), "Concrete member counts differ")
    require(np.all(POPC[np.bitwise_or.reduce(full_rows, axis=1)] == 10), "Concrete membership overlap")
    normalized = np.where(POPC[raw_queries] > 5, FULL ^ raw_queries, raw_queries).astype(np.uint16)
    flips = POPC[raw_queries] > 5
    physical = np.zeros((projected_count*2, 4, 7), dtype=np.uint16)
    physical[:, :, list(OLD_TO_NEW)] = np.repeat(normalized, 2, axis=0)
    physical[:, :, 3] = EXTRA_QUERY
    sizes = POPC[physical]
    padding = np.asarray(M, dtype=np.int16) - sizes.astype(np.int16)
    require(np.all((sizes <= 5) & (padding >= 0) & (padding <= 13)), "Known-negative padding cannot realize M")
    require(np.all(sizes + padding == np.asarray(M)), "Physical total high count differs from M")
    # Explicit 24-row masks: candidates are bits 0..10, known nonmembers
    # bits 11..23. Their masks cannot intersect any concrete membership.
    negative_masks = np.array([(1 << p)-1 for p in range(14)], dtype=np.uint32)
    questions24 = physical.astype(np.uint32) | (negative_masks[padding] << np.uint32(11))
    require(np.all(questions24 < 1 << 24), "Physical query exceeds twenty-four rows")
    require(np.all((questions24 >> 11) == negative_masks[padding]), "Known-negative masks differ")
    actual_h = POPC[full_rows[:, None, :] & physical]
    scores = np.zeros((len(full_rows), 4), dtype=np.uint64)
    for t in range(7):
        scores += unit_rates[t][actual_h[:, :, t]]
    positions = np.searchsorted(mean_units, scores)
    require(np.all(positions < len(mean_units)) and np.array_equal(mean_units[positions], scores),
            "Concrete mean is outside the exact response alphabet")
    recovered_h = mean_answers[positions]
    require(np.array_equal(actual_h, recovered_h), "Concrete mean count-vector recovery differs")
    old_recovered = recovered_h[:, :, list(OLD_TO_NEW)].copy()
    repeated_flips = np.repeat(flips, 2, axis=0)
    old_recovered = np.where(repeated_flips, np.asarray(CAPS_OLD)-old_recovered, old_recovered)
    recovered_codes = old_recovered.astype(np.uint16) @ OLD_WEIGHTS
    require(np.array_equal(recovered_codes, np.repeat(expected_codes, 2, axis=0)),
            "Restored mean answers do not follow the transported old tree")
    # Build the observable-path decoder independently of a supplied SAT flag.
    projected_keys = path_keys(expected_codes)
    order = np.argsort(projected_keys)
    require(len(np.unique(projected_keys)) == projected_count, "Historical complete response paths collide")
    observed_old_keys = path_keys(recovered_codes)
    found = np.searchsorted(projected_keys[order], observed_old_keys)
    require(np.array_equal(projected_keys[order][found], observed_old_keys), "Recovered leaf path missing")
    decoded_projection = rows[order[found]]
    require(np.array_equal(decoded_projection, full_rows[:, list(OLD_TO_NEW)]), "Old leaf identity recovery differs")
    extra_words = recovered_h[:, :, 3].astype(np.uint16) @ np.array([1, 2, 4, 8], dtype=np.uint16)
    inverse = np.zeros(16, dtype=np.uint16)
    for slot, word in enumerate(WORDS):
        inverse[word] = 1 << slot
    require(np.array_equal(inverse[extra_words], extras), "Additional singleton identity recovery differs")
    combined_keys = (observed_old_keys << np.uint64(4)) | extra_words.astype(np.uint64)
    distinct = len(np.unique(combined_keys))
    require(distinct == len(full_rows), "Concrete seven-target complete observation leaves collide")
    digest = hashlib.sha256()
    digest.update(full_rows.astype("<u2", copy=False).tobytes())
    digest.update(scores.astype("<u8", copy=False).tobytes())
    digest.update(combined_keys.astype("<u8", copy=False).tobytes())
    return {"q3_code": int(code), "projected_states": projected_count,
            "concrete_full_states": len(full_rows), "distinct_complete_observation_leaves": distinct,
            "collisions": 0, "maximum_total_queries": 4,
            "maximum_projection_queries": int(depths.max()),
            "mean_responses_replayed": len(full_rows)*4,
            "padding_minimum": int(padding.min()), "padding_maximum": int(padding.max()),
            "physical_24row_query_masks_checked": len(full_rows)*4*7,
            "trace_sha256": digest.hexdigest()}


