"""Read-only independent replay functions for three completed adaptive types.

No solver or certificate generator is imported. audit.py supplies public
relative paths and requires complete SAT witnesses for all requested types."""
from __future__ import annotations

import argparse
from collections import Counter
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
PROOF = ROOT / "analysis/universal_distribution_20261011/fixed5/results.json"

TYPES = {"a1b5": (1, 5, (1, 2, 2, 2, 2, 2)),
         "a2b4": (2, 4, (1, 1, 2, 2, 2, 2)),
         "a0b5": (0, 5, (2, 2, 2, 2, 2, 0))}
Q1 = (31,)*6

FULL = 2047
POPC = np.array([mask.bit_count() for mask in range(2048)], dtype=np.uint8)
WEIGHTS = np.array([3**t for t in range(6)], dtype=np.uint16)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, sources):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def manifest_path(run):
    for filename in ("manifest.json.gz", "manifest.json"):
        path = run / filename
        if path.exists():
            return path
    return None


def readiness(name, run, sources):
    path = manifest_path(run)
    a, b, caps = TYPES[name]
    if path is None:
        return {"type_name": name, "type": [a, b], "status": "SKIPPED_MISSING_MANIFEST", "run": str(run)}, None
    manifest = read(path, sources)
    require(manifest["type"] == [a, b] and manifest["capacities"] == list(caps),
            f"{name}: manifest type differs")
    first = manifest["first_query"]
    require(len(first) == 6 and all(type(mask) is int and (mask == 31 if n else mask in (0, 31))
                                  for mask, n in zip(first, caps)),
            f"{name}: first query must select the first five candidates in each positive column; dummy column may be 0 or 31")
    missing = []
    non_sat = []
    for orbit in manifest["representatives"]:
        representative = orbit["representative"]
        certificate = run / str(representative) / "certificate.json.gz"
        if not certificate.exists():
            missing.append(representative)
        else:
            entry = read(certificate, sources)
            if entry.get("status") != "SAT":
                non_sat.append({"representative": representative, "status": entry.get("status")})
    ready = not missing and not non_sat
    return {"type_name": name, "type": [a, b], "status": "READY_COMPLETE_SAT_CERTIFICATES" if ready else "SKIPPED_NOT_ALL_REPRESENTATIVES_SAT",
            "run": str(run), "representatives": len(manifest["representatives"]),
            "missing_certificates": missing, "non_SAT_certificates": non_sat}, manifest


def first_branch_counts(caps):
    """Exact labelled allocation counts in the five-selected/six-other split."""
    counts = {}
    total = sum(caps)
    for hs in product(*(range(n+1) for n in caps)):
        high = sum(hs)
        low = total-high
        if high > 5 or low > 6:
            continue
        numerator = factorial(5)*factorial(6)
        denominator = factorial(5-high)*factorial(6-low)
        for n, h in zip(caps, hs):
            denominator *= factorial(h)*factorial(n-h)
        require(numerator % denominator == 0, "First-branch cardinality is not integral")
        counts[sum(h*3**t for t, h in enumerate(hs))] = numerator//denominator
    return counts


def generate(caps, code):
    options = []
    for t, cap in enumerate(caps):
        h = code//3**t % 3
        options.append([sum(1 << slot for slot in members) for members in combinations(range(11), cap)
                        if sum(slot < 5 for slot in members) == h])
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


def tree_paths(rows, tree, first_query):
    count = len(rows)
    queries = np.zeros((count, 4, 6), dtype=np.uint16)
    codes = np.zeros((count, 4), dtype=np.uint16)
    depths = np.zeros(count, dtype=np.uint8)
    queries[:, 0, :] = first_query
    codes[:, 0] = POPC[rows & np.asarray(first_query, dtype=np.uint16)].astype(np.uint16) @ WEIGHTS
    node_count = 0
    def walk(indices, node, depth):
        nonlocal node_count
        require(len(indices) > 0 and depth <= 4, "Empty or too-deep reachable tree node")
        if node.get("leaf"):
            require(len(indices) == 1, "Certificate leaf is not singleton")
            depths[indices] = depth
            return
        require(depth < 4, "Certificate exceeds three queries after the first")
        raw = node["query"]
        require(len(raw) == 6 and all(type(mask) is int and 0 <= mask <= FULL for mask in raw),
                "Illegal six-column certificate query")
        query = np.asarray(raw, dtype=np.uint16)
        queries[indices, depth, :] = query
        response = POPC[rows[indices] & query].astype(np.uint16) @ WEIGHTS
        codes[indices, depth] = response
        outcomes = np.unique(response)
        node_count += 1
        if node.get("terminal"):
            require(len(outcomes) == len(indices), "Certificate terminal query collides")
            depths[indices] = depth+1
            return
        require(set(node["children"]) == {str(int(code)) for code in outcomes},
                "Certificate reachable child set differs")
        for code in outcomes:
            walk(indices[response == code], node["children"][str(int(code))], depth+1)
    walk(np.arange(count), tree, 1)
    require(np.all((depths >= 1) & (depths <= 4)), "Missing certificate path")
    return queries, codes, depths, node_count


def encoder(caps, ms, proof):
    positive = sum(n > 0 for n in caps)
    rates = [[Fraction((24-n)*h, 100*n*(m-h)*positive) for h in range(n+1)] if n else [Fraction()]
             for n, m in zip(caps, ms)]
    answers = list(product(*(range(n+1) for n in caps)))
    means = [sum((rates[t][h] for t, h in enumerate(hs)), Fraction()) for hs in answers]
    require(len(means) == len(set(means)) == proof["independent_response_count"], "Exact mean alphabet collides or differs")
    gap = min(right-left for left, right in zip(sorted(means), sorted(means)[1:]))
    require(str(gap) == proof["minimum_gap_exact"], "Exact mean gap differs from prior M certificate")
    denominator = lcm(*(value.denominator for table in rates for value in table))
    unit_rates = [np.array([int(value*denominator) for value in table], dtype=np.uint64) for table in rates]
    units = np.array([int(value*denominator) for value in means], dtype=np.uint64)
    order = np.argsort(units)
    return unit_rates, units[order], np.asarray(answers, dtype=np.uint8)[order], denominator, gap


def path_keys(codes):
    result = np.zeros(len(codes), dtype=np.uint64)
    for q in range(4):
        result = result*np.uint64(729) + codes[:, q]
    return result


def mean_replay(rows, queries, expected_codes, caps, ms, units, mean_units, mean_answers):
    flips = POPC[queries] > 5
    physical = np.where(flips, FULL ^ queries, queries).astype(np.uint16)
    for t, n in enumerate(caps):
        if n == 0:
            physical[:, :, t] = 0
    sizes = POPC[physical]
    padding = np.asarray(ms, dtype=np.int16)-sizes.astype(np.int16)
    require(np.all((sizes <= 5) & (padding >= 0) & (padding <= 13)), "24-row padding cannot realize M")
    require(np.all(sizes + padding == np.asarray(ms)), "High-side total differs from M")
    negative_masks = np.array([(1 << k)-1 for k in range(14)], dtype=np.uint32)
    physical24 = physical.astype(np.uint32) | (negative_masks[padding] << np.uint32(11))
    require(np.all(physical24 < 1 << 24), "High-side mask exceeds twenty-four rows")
    require(np.all((physical24 >> 11) == negative_masks[padding]), "Known-negative padding mask differs")
    actual_h = POPC[rows[:, None, :] & physical]
    scores = np.zeros((len(rows), 4), dtype=np.uint64)
    for t in range(6):
        scores += units[t][actual_h[:, :, t]]
    positions = np.searchsorted(mean_units, scores)
    require(np.all(positions < len(mean_units)) and np.array_equal(mean_units[positions], scores),
            "Concrete average is outside the exact decoder alphabet")
    recovered = mean_answers[positions]
    require(np.array_equal(recovered, actual_h), "Average fails to recover the full count vector")
    restored = np.where(flips, np.asarray(caps)-recovered, recovered)
    tree_codes = restored.astype(np.uint16) @ WEIGHTS
    require(np.array_equal(tree_codes, expected_codes), "Average does not restore the certificate response path")
    reference_keys = path_keys(expected_codes)
    observed_keys = path_keys(tree_codes)
    order = np.argsort(reference_keys)
    require(len(np.unique(reference_keys)) == len(rows), "Certificate observation paths collide")
    found = np.searchsorted(reference_keys[order], observed_keys)
    require(np.array_equal(reference_keys[order][found], observed_keys), "Observed path has no leaf")
    require(np.array_equal(rows[order[found]], rows), "Decoded leaf membership differs from concrete truth")
    distinct = len(np.unique(observed_keys))
    require(distinct == len(rows), "Complete average observations collide")
    digest = hashlib.sha256()
    for array, dtype in ((rows, "<u2"), (physical24, "<u4"), (scores, "<u8"), (observed_keys, "<u8")):
        digest.update(array.astype(dtype, copy=False).tobytes())
    return {"states": len(rows), "distinct_complete_observation_leaves": distinct, "collisions": 0,
            "mean_responses_replayed": len(rows)*4, "physical_24row_query_masks_checked": len(rows)*4*6,
            "padding_minimum": int(padding.min()), "padding_maximum": int(padding.max()),
            "trace_sha256": digest.hexdigest()}


def verify_type(name, run, manifest, proof, sources):
    started = monotonic()
    a, b, caps = TYPES[name]
    positive_ms = proof["M_1person_targets_in_ID_order"]+proof["M_2person_targets_in_ID_order"]
    ms = tuple(positive_ms+[0]*(6-a-b))
    require(len(ms) == 6 and all(5 <= ms[t] <= 13 for t, n in enumerate(caps) if n), "M construction differs")
    units, mean_units, mean_answers, denominator, gap = encoder(caps, ms, proof)
    counts = first_branch_counts(caps)
    first_query = manifest["first_query"]
    expected = factorial(11)//(2**b*factorial(11-a-2*b))
    stored_totals = [manifest[key] for key in ("full_state_count", "total_states", "states") if key in manifest]
    stored_branches = [manifest[key] for key in ("reachable_first_branches", "first_branch_count") if key in manifest]
    require(stored_totals and sum(counts.values()) == expected and all(value == expected for value in stored_totals),
            "Complete state count differs")
    require(stored_branches and all(value == len(counts) for value in stored_branches), "Initial branch count differs")
    require(len(manifest["representatives"]) == manifest["representative_count"], "Representative count differs")
    reports = []
    covered = set()
    query_nodes = 0
    for orbit in manifest["representatives"]:
        representative = orbit["representative"]
        require(representative in counts and orbit["states"] == counts[representative], "Representative branch cardinality differs")
        certificate = read(run / str(representative) / "certificate.json.gz", sources)
        require(certificate["status"] == "SAT" and certificate["type"] == [a, b]
                and certificate["representative"] == representative and certificate["depth"] == 3
                and certificate["first_query"] == first_query, "Certificate header differs")
        source_rows = generate(caps, representative)
        require(len(source_rows) == counts[representative] and state_hash(source_rows) == certificate["state_hash"],
                "Independently generated representative states differ")
        source_queries, source_codes, depths, nodes = tree_paths(source_rows, certificate["tree"], first_query)
        query_nodes += nodes
        for branch in orbit["branches"]:
            code = branch["code"]
            require(code in counts and code not in covered and branch["states"] == counts[code], "Duplicate, missing or miscounted initial branch")
            covered.add(code)
            permutation = branch["team_old_to_new"]
            require(sorted(permutation) == list(range(6)) and all(caps[t] == caps[permutation[t]] for t in range(6)),
                    "Team transport is not a capacity-preserving permutation")
            require(all(first_query[t] == first_query[permutation[t]] for t in range(6)), "First query is not invariant under transport")
            rows = np.zeros_like(source_rows)
            queries = np.zeros_like(source_queries)
            codes = np.zeros_like(source_codes)
            for old, new in enumerate(permutation):
                rows[:, new] = source_rows[:, old]
                queries[:, :, new] = source_queries[:, :, old]
                codes += ((source_codes//np.uint16(3**old)) % np.uint16(3))*np.uint16(3**new)
            require(len(rows) == counts[code] and np.all(codes[:, 0] == code), "Transported states do not cover the specified branch")
            require(np.all(POPC[rows] == np.asarray(caps)) and np.all(POPC[np.bitwise_or.reduce(rows, axis=1)] == a+2*b),
                    "Transported concrete state is not legal")
            result = mean_replay(rows, queries, codes, caps, ms, units, mean_units, mean_answers)
            result.update({"first_code": code, "representative": representative,
                           "maximum_queries": int(depths.max()), "state_sha256": state_hash(rows)})
            reports.append(result)
    require(covered == set(counts) and sum(row["states"] for row in reports) == expected, "Full initial-branch coverage differs")
    require(all(row["maximum_queries"] <= 4 and row["collisions"] == 0 for row in reports), "Maximum depth or collision failure")
    return {"type_name": name, "type": [a, b], "a": a, "b": b, "status": "SAT",
            "verification_status": "PASS_ALL_CONCRETE_ASSIGNMENTS_EXACT_MEAN_REPLAY",
            "run": str(run.relative_to(ROOT)), "total_states": expected,
            "all_leaves_singleton": True, "collisions": 0,
            "distinct_complete_observation_leaves": sum(row["distinct_complete_observation_leaves"] for row in reports),
            "maximum_queries": max(row["maximum_queries"] for row in reports),
            "maximum_total_queries": max(row["maximum_queries"] for row in reports),
            "reachable_first_branches": len(counts), "representative_certificates": len(manifest["representatives"]),
            "representative_query_nodes_replayed": query_nodes,
            "mean_responses_replayed": sum(row["mean_responses_replayed"] for row in reports),
            "physical_24row_query_masks_checked": sum(row["physical_24row_query_masks_checked"] for row in reports),
            "encoder": {"capacities": list(caps), "M_in_six_columns": list(ms), "positive_targets_averaged": a+b,
                        "response_count": len(mean_units), "unit_denominator": denominator, "minimum_gap_exact": str(gap),
                        "padding_minimum": min(row["padding_minimum"] for row in reports),
                        "padding_maximum": max(row["padding_maximum"] for row in reports),
                        "zero_column_policy": "Physical question empty and M=0; excluded from the mean; restored tree response digit is zero"},
            "coverage_proof": "Every representative fiber is independently generated. A capacity-preserving target permutation preserving the first question bijectively maps it onto its destination fiber. All moved states are concretely mean-replayed. Exact first-fiber allocation counts, complete coverage of every reachable code and unique leaf paths prove coverage of the full labelled disjoint-state universe",
            "inputs_sha256": {relative: digest for relative, digest in sources.items()
                              if (ROOT/relative).is_relative_to(run) or (ROOT/relative) == PROOF},
            "branches": reports, "elapsed_seconds": monotonic()-started}


