"""Read-only independent target-specific fixed-four certificate functions.

Disjoint pair assignments use OR/AND response signatures and residual masks,
independently of the search generator. UNKNOWN never establishes UNSAT."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import combinations, product
import json
from math import factorial
from pathlib import Path
import struct
import sys
from time import monotonic


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]



EXPECTED_TYPES = {(m - 2*b, b) for b in range(2, 6) for m in (10, 11)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def validate_words(words, label):
    require(len(words) == 11, f"{label}: candidate count is not eleven")
    require(all(type(word) is int and 0 <= word < 16 for word in words),
            f"{label}: codeword outside four-bit universe")
    require(len(set(words)) == 11, f"{label}: repeated single codewords")
    questions = [[i for i, word in enumerate(words) if word >> bit & 1]
                 for bit in range(4)]
    require(all(len(question) == 5 for question in questions),
            f"{label}: each query must select five candidates for the saved padding policy")
    return questions


def check_mean_alphabet(proof_type, a, b):
    """Check every one-round count vector as exact rational means."""
    counts = [1]*a + [2]*b
    ms = proof_type["M"]
    require(proof_type["counts"] == counts, "Proof count ordering differs")
    require(len(ms) == a+b and all(type(m) is int and 5 <= m <= 18 for m in ms),
            "M does not have a legal five-candidate plus known-negative construction")
    padding = [m-5 for m in ms]
    require(padding == proof_type["padding"] and all(0 <= p <= 13 for p in padding),
            "Saved padding differs or exceeds thirteen known nonmembers")
    alphabets = [[Fraction((24-n)*h, 100*n*(m-h)) for h in range(n+1)]
                 for n, m in zip(counts, ms)]
    means = sorted(sum(response, Fraction()) / (a+b)
                   for response in product(*alphabets))
    require(len(means) == len(set(means)) == 2**a * 3**b,
            "One-round exact mean alphabet has a collision")
    gap = min(right-left for left, right in zip(means, means[1:]))
    require(str(gap) == proof_type["minimum_gap_exact"], "Independent mean gap differs")
    require(proof_type["response_count"] == len(means), "Saved alphabet cardinality differs")
    return {
        "status": "PASS_EXACT_FRACTION_ALPHABET",
        "M": ms, "padding": padding, "response_count": len(means),
        "minimum_gap_exact": str(gap),
        "candidate_high_count_each_round": 5,
        "known_negative_count": 13,
        "formula": "per-target (24-n)*h/[100*n*(M-h)], averaged over positive targets",
        "construction": "For each target and each round select the five code-bit candidates and the first M-5 known nonmembers; every remaining row is low. The same per-target M is used in all four rounds",
        "scope": "Exact unrounded mean response; lower-precision leaderboard display alone is not assumed sufficient",
    }


def check_certificate(certificate, proof_type, source):
    started = monotonic()
    a, b = certificate["a"], certificate["b"]
    m = a + 2*b
    require((a, b) in EXPECTED_TYPES and certificate["total_members"] == m,
            "Certificate type outside eight requested types")
    words = certificate["pair_codewords"]
    require(len(words) == b, "Pair-target codebook count differs")
    pair_questions = [validate_words(row, f"pair target {t}") for t, row in enumerate(words)]
    questions_by_round = [[pair_questions[t][q] for t in range(b)] for q in range(4)]
    if "pair_questions_by_round" in certificate:
        require(certificate["pair_questions_by_round"] == questions_by_round,
                "Saved pair questions disagree with codeword bits")
    single_questions = validate_words(certificate["single_codewords"], "single targets")
    # A binary pair's four integer sums are in one-to-one correspondence with
    # its union (sum >= 1) and intersection (sum == 2). Use a base-256 layout,
    # unlike the main search's base-81 ternary pair-response representation.
    pair_signatures = []
    for row in words:
        table = {}
        for i, j in combinations(range(11), 2):
            union, intersection = row[i] | row[j], row[i] & row[j]
            signature = union | (intersection << 4)
            require(all(((union >> q & 1) + (intersection >> q & 1)) ==
                        ((row[i] >> q & 1) + (row[j] >> q & 1)) for q in range(4)),
                    "Union/intersection does not encode the pair response")
            table[(i, j)] = signature
        pair_signatures.append(table)
    seen = set()
    state_count = key_count = collisions = 0
    key_digest = hashlib.sha256()

    def record(signature, residual):
        nonlocal key_count, collisions
        # residual is the complement of the single-member candidate set.
        key = (signature << 11) | residual
        key_count += 1
        collisions += key in seen
        seen.add(key)
        key_digest.update(struct.pack("<Q", key))

    def enumerate_pairs(target, remaining, signature, used):
        nonlocal state_count
        if target == b:
            state_count += 1
            require(used.bit_count() == 2*b, "Independent pair generation reused a candidate")
            if m == 11:
                record(signature, used)
            else:
                for unassigned in remaining:
                    record(signature, used | (1 << unassigned))
            return
        for i, j in combinations(remaining, 2):
            enumerate_pairs(target+1, tuple(x for x in remaining if x != i and x != j),
                            signature | (pair_signatures[target][(i, j)] << (8*target)),
                            used | (1 << i) | (1 << j))

    enumerate_pairs(0, tuple(range(11)), 0, 0)
    expected_states = factorial(11) // (2**b * factorial(11-2*b))
    expected_keys = expected_states * (1 if m == 11 else 11-2*b)
    require(state_count == expected_states == certificate["pair_state_count"],
            "Pair-state enumeration cardinality differs")
    require(key_count == expected_keys == certificate["residual_keys_checked"],
            "Residual-key enumeration cardinality differs")
    require(collisions == 0 and len(seen) == expected_keys, "Claimed SAT certificate collides")
    if "collisions" in certificate:
        require(certificate["collisions"] == 0, "Saved SAT collision count is nonzero")
    if "fewest_collisions" in certificate:
        require(certificate["fewest_collisions"] == 0, "Saved optimized SAT collision count is nonzero")
    full_assignments = factorial(11) // (2**b * factorial(11-m))
    require(full_assignments == proof_type["full_assignment_count"],
            "Full assignment count differs from the known-count proof")
    mean_encoder = check_mean_alphabet(proof_type, a, b)
    return {
        "a": a, "b": b, "total_members": m,
        "status": "PASS_INDEPENDENT_FULL_PAIR_RESIDUAL_CERTIFICATE",
        "certificate_source": str(source.relative_to(ROOT)),
        "pair_state_count": state_count, "residual_keys_checked": key_count,
        "distinct_residual_keys": len(seen), "collisions": collisions,
        "independent_key_stream_sha256": key_digest.hexdigest(),
        "key_encoding": "four-bit OR + four-bit AND per labelled pair target, eight-bit blocks; then eleven-bit residual candidate mask",
        "single_questions_by_round": single_questions,
        "pair_questions_by_round": questions_by_round,
        "full_disjoint_assignment_count_covered_by_composition": full_assignments,
        "mean_encoder": mean_encoder,
        "composition_proof": "The exact mean alphabet recovers all per-target counts in each round. Unique single-target words recover each singleton identity. Their complement R is therefore fixed. For m=11, R is the pair-used mask; for m=10, R consists of the pair-used mask and one unassigned candidate. The exhaustive residual-key check proves that each possible R and all four pair-target responses identify at most one ordered collection of disjoint pairs. Thus every full assignment is identified, including all singleton target permutations",
        "elapsed_seconds": monotonic()-started,
    }


