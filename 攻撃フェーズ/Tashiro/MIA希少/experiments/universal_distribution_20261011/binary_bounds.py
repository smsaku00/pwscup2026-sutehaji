"""Read-only exact fixed-four binary bounds and common-code classification.

All finite word sets and constructive/collision witnesses are independently
recomputed. These are binary subset/count bounds after member counts are known."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from itertools import combinations, product
import json
from math import comb
from pathlib import Path
import sys
from time import monotonic


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

DIMENSION = 4
VERTICES = 16
N = 11
SINGLE_M = (5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digits(code):
    return tuple((code >> bit) & 1 for bit in range(DIMENSION))


def ternary_pair_sum(left, right):
    return sum(((left >> bit & 1) + (right >> bit & 1)) * 3**bit
               for bit in range(DIMENSION))


def exhaustive_subset_bound():
    """Directly test all 2^16 sets by the independent union/intersection key."""
    valid_by_size = [0] * 17
    best = ()
    masks_tested = 0
    for subset_mask in range(1 << VERTICES):
        subset = tuple(code for code in range(VERTICES) if subset_mask >> code & 1)
        seen = set()
        valid = True
        for left, right in combinations(subset, 2):
            # Binary coordinate sum is determined by the union (sum>=1)
            # and intersection (sum==2), avoiding the DFS's base-three key.
            key = (left | right) | ((left & right) << DIMENSION)
            if key in seen:
                valid = False
                break
            seen.add(key)
        masks_tested += 1
        if valid:
            valid_by_size[len(subset)] += 1
            if len(subset) > len(best):
                best = subset
    require(masks_tested == 65_536 and len(best) == 7, "Unexpected fixed-four bound")
    require(sum(valid_by_size) == 8193, "Valid-set total differs")
    # Complementary independent DFS checks every collision-free prefix.
    pair_keys = [[ternary_pair_sum(left, right) for right in range(VERTICES)]
                 for left in range(VERTICES)]
    dfs_histogram = [0] * 17
    def dfs(subset, start, used_pair_sums):
        dfs_histogram[len(subset)] += 1
        for new_code in range(start, VERTICES):
            additions = sum(1 << pair_keys[new_code][old_code] for old_code in subset)
            if not additions & used_pair_sums:
                dfs(subset + (new_code,), new_code + 1, used_pair_sums | additions)
    dfs((), 0, 0)
    require(dfs_histogram == valid_by_size, "Direct enumeration and DFS disagree")
    return {
        "status": "EXACT_MAXIMUM_PROVED", "dimension": 4,
        "binary_codeword_universe": 16, "all_subsets_directly_tested": masks_tested,
        "valid_subsets_by_cardinality": valid_by_size,
        "invalid_subsets_by_cardinality": [comb(16, k) - valid_by_size[k] for k in range(17)],
        "maximum_codewords_with_unique_distinct_pair_sums": len(best),
        "maximum_witness_decimal": list(best),
        "maximum_witness_bits": [list(digits(code)) for code in best],
        "independent_dfs_valid_prefixes": sum(dfs_histogram),
        "direct_and_dfs_histograms_equal": True,
        "eight_codeword_subsets_tested": comb(16, 8),
        "valid_eight_codeword_subsets": valid_by_size[8],
        "eleven_candidate_single_and_pair_fixed4_possible": False,
        "scope": "Nonadaptive binary subset questions; distinct candidate codewords; unordered pairs of two different candidates; member count is known, so single/pair cross-cardinality collisions are irrelevant",
    }


def single_pair_adaptive_lower_bound():
    branches = []
    for selected in range(12):
        sizes = (comb(11 - selected, 2) if 11 - selected >= 2 else 0,
                 selected * (11 - selected),
                 comb(selected, 2) if selected >= 2 else 0)
        require(sum(sizes) == 55, "First-query branches do not partition the pairs")
        branches.append({"selected_candidates": selected, "h0_pairs": sizes[0],
                         "h1_pairs": sizes[1], "h2_pairs": sizes[2],
                         "largest_branch": max(sizes)})
    best_worst = min(item["largest_branch"] for item in branches)
    require(best_worst == 28 and best_worst > 3**3, "Adaptive lower bound failed")
    return {
        "status": "ADAPTIVE_FOUR_COUNT_QUERIES_IMPOSSIBLE",
        "count_type": {"a": 0, "b": 1},
        "candidate_count": 11, "possible_pairs": 55,
        "all_first_query_sizes": branches,
        "minimum_possible_largest_first_branch": best_worst,
        "remaining_three_query_leaf_upper_bound": 27,
        "minimum_binary_count_queries": 5,
        "proof": "Every first subset leaves at least 28 pairs in one branch; each subsequent query returns at most three counts, so three remaining queries distinguish at most 27 pairs",
        "other_targets": "All other targets have zero members, so supply no extra membership-dependent responses",
        "scope": "Adaptive binary subset/count questions, including team-specific padding; not an impossibility theorem for arbitrary multi-level confidence submissions or other returned channels",
    }


def balanced_single_code():
    witness = next(subset for subset in combinations(range(16), 11)
                   if all(sum(code >> bit & 1 for code in subset) == 5 for bit in range(4)))
    bits = tuple(digits(code) for code in witness)
    questions = tuple(tuple(slot for slot, code in enumerate(bits) if code[q]) for q in range(4))
    require(len(set(bits)) == 11 and all(len(question) == 5 for question in questions),
            "Single-member balanced construction failed")
    types = []
    for a in range(1, 12):
        ms = SINGLE_M[:a]
        rates = [Fraction(23, 100 * (m - 1)) for m in ms]
        means = sorted(sum((rate * h for rate, h in zip(rates, response)), Fraction()) / a
                       for response in product((0, 1), repeat=a))
        require(len(means) == len(set(means)) == 2**a, "Single-member mean encoder collides")
        gap = min(y - x for x, y in zip(means, means[1:]))
        types.append({"a": a, "b": 0, "status": "FIXED_FOUR_PROVED_BY_COMPOSITION",
                      "M": list(ms), "padding": [m - 5 for m in ms],
                      "response_count": len(means), "minimum_mean_gap_exact": str(gap)})
    return {
        "status": "BALANCED_SINGLE_MEMBER_FIXED4_CONSTRUCTED",
        "codewords_decimal": list(witness), "codewords_bits": [list(code) for code in bits],
        "four_questions_candidate_slots": [list(question) for question in questions],
        "candidates_selected_per_query": [len(question) for question in questions],
        "local_single_signature_count": 11, "local_single_signature_collisions": 0,
        "positive_single_only_types": types,
        "mean_score_domain": "24 rare rows, 11 candidate rows and 13 known negative rows; TPR at FPR .01; unrounded mean; known per-target counts",
        "target_policy": "Assign M in ascending positive-target ID order; keep M and the four candidate subsets fixed",
        "scope": "Every all-single type (a,0), 1<=a<=11, and every compatible disjoint individual assignment; this is a new fixed-four code, not deletion of one bit from the actually used fixed-five code",
    }


def type_bounds():
    types = []
    for b in range(6):
        for a in range(12 - 2*b):
            m = a + 2*b
            if not m:
                fixed_status = adaptive_status = "TRIVIAL_NO_MEMBERS"
            elif not b:
                fixed_status = adaptive_status = "FOUR_QUERIES_CONSTRUCTED_ALL_SINGLE"
            elif b == 1 and a >= 8:
                fixed_status = adaptive_status = "FOUR_QUERIES_CONSTRUCTED_COMMON_CODE"
            elif m <= 9:
                fixed_status = "NONADAPTIVE_FOUR_IMPOSSIBLE_BY_PAIR_COLLISION_EMBEDDING"
                adaptive_status = ("ADAPTIVE_FOUR_IMPOSSIBLE_BY_FIRST_BRANCH_BOUND"
                                   if (a, b) == (0, 1) else "NOT_DECIDED_BY_THIS_BOUND")
            else:
                fixed_status = adaptive_status = "NOT_DECIDED_BY_THIS_BOUND"
            types.append({"a": a, "b": b, "total_members": m,
                          "nonadaptive_four_status": fixed_status,
                          "adaptive_four_status": adaptive_status,
                          "remaining_candidates_in_pair_target_fiber": 13 - m if b else None})
    require(len(types) == 42, "Type coverage differs")
    return {
        "scope": "Targets may use different four fixed binary subsets; full per-target count responses are granted, a stronger observation than their average",
        "local_fiber_proof": "Fix the memberships of all targets except one two-member target. They consume m-2 candidates, leaving 11-(m-2)=13-m choices. Any nonadaptive four-query policy gives these choices binary four-bit words. A duplicate word yields a pair collision; otherwise the Sidon maximum seven implies a collision whenever 13-m>=8. Keep all other memberships fixed to obtain two disjoint legal full assignments with identical responses to every query",
        "stronger_embedding_proof": "Choose one two-member target before fixing the other memberships. If its eleven words repeat, add any common third candidate to the two repeated words to obtain two pairs with equal sums and a three-candidate union. If all words differ, the exact Sidon maximum seven gives two colliding pairs with a four-candidate union. For m<=9, assign the other m-2 candidates identically outside that union: at least seven candidates are available. All other targets then have the same memberships and thus the same responses even when they use different questions. Every query's complete per-target responses and the average are equal",
        "impossible_nonadaptive_condition": "b>0 and m=a+2*b<=9",
        "local_fiber_only_condition": "b>0 and m<=5 (six types); subsumed by the stronger twenty-type embedding bound",
        "why_not_an_adaptive_fiber_proof": "In an adaptive policy the pair-target questions may change after earlier answers; one fixed four-bit code does not represent all paths",
        "types": types,
        "summary": {
            "types": 42,
            "all_single_positive_fixed4_constructed": sum(bool(item["a"]) and item["b"] == 0 for item in types),
            "nonadaptive_types_impossible_by_embedding": sum(item["nonadaptive_four_status"].startswith("NONADAPTIVE_FOUR_IMPOSSIBLE") for item in types),
            "local_fiber_only_types": sum(bool(item["b"]) and item["total_members"] <= 5 for item in types),
            "positive_types_constructed_fixed4": sum(item["nonadaptive_four_status"].startswith("FOUR_QUERIES_CONSTRUCTED") for item in types),
            "fixed4_types_not_decided_by_this_bound": sum(item["nonadaptive_four_status"] == "NOT_DECIDED_BY_THIS_BOUND" for item in types),
            "adaptive_types_impossible_by_first_branch": sum(item["adaptive_four_status"].startswith("ADAPTIVE_FOUR_IMPOSSIBLE") for item in types),
            "zero_types": 1,
        },
    }


def first_pair_collision(codewords):
    seen = {}
    for pair in combinations(range(len(codewords)), 2):
        signature = tuple(sum(digits(codewords[slot])[q] for slot in pair) for q in range(4))
        if signature in seen:
            other = seen[signature]
            require(not set(pair) & set(other), "Distinct words cannot collide on pairs sharing a candidate")
            return other, pair, signature
        seen[signature] = pair
    return None


def common_fixed4_classification(codewords):
    """Complete classification for a common codebook shared by all targets."""
    tested = 0
    collision_hash = hashlib.sha256()
    for subset in combinations(range(16), 11):
        collision = first_pair_collision(subset)
        require(collision is not None, "An eleven-word Sidon set exists")
        collision_hash.update(json.dumps([subset, collision], separators=(",", ":")).encode())
        tested += 1
    require(tested == 4368, "Eleven-word subset enumeration differs")
    collision = first_pair_collision(codewords)
    require(collision is not None, "Example balanced code has no required pair collision")
    pair_x, pair_y, shared_signature = collision
    excluded = set(pair_x) | set(pair_y)
    types = []
    for b in range(6):
        for a in range(12 - 2*b):
            m = a + 2*b
            counts = (1,) * a + (2,) * b
            if not b:
                status = "SAT_COMMON_FIXED4" if a else "TRIVIAL_NO_MEMBERS"
                proof_kind = "distinct single codewords" if a else "no members"
                example = None
            elif b == 1 and a >= 8:
                status = "SAT_COMMON_FIXED4"
                proof_kind = "decode all singles first; remaining candidate pool has at most three; pair sums of distinct codewords among three candidates are unique"
                example = None
                # Check this constructive argument for every possible residual
                # two/three-candidate set, not merely one selected assignment.
                for residual in combinations(range(11), 11 - a):
                    require(first_pair_collision(tuple(codewords[i] for i in residual)) is None,
                            "Residual pair decoder collides")
            else:
                status = "UNSAT_COMMON_FIXED4"
                left = [None] * len(counts)
                right = [None] * len(counts)
                left[a], right[a] = pair_x, pair_y
                if m <= 9:
                    proof_kind = "one-pair-target substitution; the alternative colliding pair is unassigned"
                else:
                    require(b >= 2 and m in (10, 11), "Counterexample cases do not cover the type")
                    proof_kind = "two-pair-target exchange"
                    left[a + 1], right[a + 1] = pair_y, pair_x
                unused = iter(slot for slot in range(11) if slot not in excluded)
                for target, n in enumerate(counts):
                    if left[target] is None:
                        members = tuple(next(unused) for _ in range(n))
                        left[target] = right[target] = members
                require(left != right, "Counterexample states are equal")
                for state in (left, right):
                    flattened = [slot for group in state for slot in group]
                    require(tuple(map(len, state)) == counts and len(set(flattened)) == m,
                            "Counterexample violates count/disjointness constraints")
                response_left = [list(tuple(sum(digits(codewords[slot])[q] for slot in group)
                                             for q in range(4))) for group in left]
                response_right = [list(tuple(sum(digits(codewords[slot])[q] for slot in group)
                                              for q in range(4))) for group in right]
                require(response_left == response_right, "Counterexample questions distinguish the states")
                example = {"state_1_candidate_slots": [list(group) for group in left],
                           "state_2_candidate_slots": [list(group) for group in right],
                           "identical_four_responses_per_target": response_left,
                           "counts": list(counts), "synthetic_truth_only": True}
            types.append({"a": a, "b": b, "total_members": m, "status": status,
                          "proof_kind": proof_kind, "counterexample_template_instance": example})
    sat = sum(item["status"] in ("SAT_COMMON_FIXED4", "TRIVIAL_NO_MEMBERS") for item in types)
    unsat = sum(item["status"] == "UNSAT_COMMON_FIXED4" for item in types)
    require(sat == 14 and unsat == 28 and len(types) == 42, "Common-code classification differs")
    return {
        "scope": "All targets use the same four fixed binary candidate subsets; target-specific padding may differ. Full per-target counts are granted. This is distinct from four fixed team-specific subsets and from adaptive search",
        "unique_codewords_necessary": "For m<=10, put one of two equal-code candidates in a target and leave the other unassigned, then exchange them while fixing all other memberships. For m=11, put them in distinct targets and exchange them. Such target positions always exist because every target has at most two members. Thus any universally identifying common codebook must have eleven distinct words",
        "all_eleven_word_subsets_checked": tested,
        "all_eleven_word_subsets_have_disjoint_pair_collision": True,
        "eleven_word_collision_witness_stream_sha256": collision_hash.hexdigest(),
        "why_collision_pairs_disjoint": "Cancelling a shared codeword from x+y=x+z would give y=z, contrary to distinct codewords",
        "counterexample_template_1": "For b>=1 and m<=9, put colliding pair X or Y in one two-member target. Assign the other m-2 members identically from the seven candidates outside X union Y. The unused alternative pair changes no other target response",
        "counterexample_template_2": "For b>=2 and m in {10,11}, put disjoint colliding pairs X,Y in two two-member targets and swap them. Assign the other m-4 members identically from the remaining seven candidates",
        "positive_template": "b=0 uses the distinct four-bit singleton code. For (a,b)=(8,1),(9,1), recover the eight/nine singleton targets first; then recover the pair among the remaining three/two candidates",
        "positive_average_encoder": "Use the existing injective type-specific M certificate; each new four-query subset selects exactly five candidates, so the same score formula and 13-negative padding remain applicable",
        "example_common_codewords_decimal": list(codewords),
        "example_disjoint_colliding_pairs_candidate_slots": [list(pair_x), list(pair_y)],
        "example_equal_pair_four_response": list(shared_signature),
        "types": types,
        "summary": {"all_types": 42, "SAT_including_zero": sat, "nonzero_SAT": sat - 1,
                    "UNSAT": unsat, "unresolved": 0},
        "no_adaptive_inference": "UNSAT here does not establish adaptive UNSAT; adaptive (0,1) is proved separately by the first-branch bound",
    }


