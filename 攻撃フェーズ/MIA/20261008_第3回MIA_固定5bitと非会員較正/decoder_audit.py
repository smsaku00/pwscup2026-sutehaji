"""Reproducible independent audit of 結果読取.py with official-score means.

Only an explicitly selected new JSON report is written. The decoder itself
and the submitted candidate remain unchanged. Synthetic truths are for
exhaustive local coverage and testing, not inferred real memberships.
"""
from __future__ import annotations

import argparse
import ast
import csv
from decimal import Decimal
from fractions import Fraction
import hashlib
import importlib.util
from itertools import combinations, permutations
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DECODER = HERE / "結果読取.py"
CANDIDATE = HERE / "候補/F_mia.csv"
TARGETS = tuple(range(18)) + (22, 25, 26, 27)
TEAMS = (6, 10, 11, 16, 17, 25)
N = (1, 2, 1, 2, 1, 2)
M = (6, 7, 7, 11, 10, 12)
IDS = (47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675)
NEG = (126, 136, 271, 273, 317, 440, 480, 625, 637, 643, 668, 814, 841)
QUESTIONS = (
    (115, 169, 264, 383, 462), (47, 115, 304, 462, 675),
    (47, 90, 264, 462, 495), (47, 76, 264, 304, 383),
    (47, 76, 90, 115, 169),
)
SEED = 20261008


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_decoder():
    spec = importlib.util.spec_from_file_location("fixed_five_bit_decoder_under_review", DECODER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.TEAMS == TEAMS and module.N == N and module.M == M, "decoder teams/counts/padding differ from the human agreement")
    require(module.IDS == IDS and module.QUESTIONS == QUESTIONS, "decoder ID/bit-order constants differ from the human agreement")
    require(module.TOLERANCE == Fraction(1, 10 ** 12), "decoder exact-match tolerance differs")
    return module


def official_sources():
    kits = list((ROOT / "本戦").glob("配布*/pwscup2026-kit-main_revised0912"))
    require(len(kits) == 1, "official kit path is not unique")
    src = kits[0] / "starter/src"
    score_path = src / "pwscup2026/scoring/attack_score.py"
    mean_path = src / "pwscup2026/kit/codabench/scoring_io.py"
    require(sha(score_path) == "1e62a591bf5f564eef30bd64edb4e8146563ab497a249ac15230ef61178186c6", "official score source changed")
    require(sha(mean_path) == "32df3a36762ca876449f6c635f4ed8d5fc3ecb5f322697ff167d6c17949d5a9b", "official aggregate source changed")
    sys.path.insert(0, str(src))
    from pwscup2026.scoring.attack_score import score_mia
    tree = ast.parse(mean_path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("_finite", "_nanmean")]
    require(len(nodes) == 2, "official mean functions are missing")
    env = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(mean_path), "exec"), env)
    return score_mia, env["_nanmean"], {str(p.relative_to(ROOT)): sha(p) for p in (score_path, mean_path)}


def source_arrays():
    bpaths = list((ROOT / "本戦").glob("配布*/PWSCup2026_final_team28_*/B_self.csv"))
    require(len(bpaths) == 1, "official own cohort path is not unique")
    bpath = bpaths[0]
    qpath = HERE.parent / "20260928_配布データ基本分析/入力/aia_challenge_28.csv"
    with bpath.open(encoding="utf-8-sig", newline="") as h:
        b = list(csv.DictReader(h))
    with qpath.open(encoding="utf-8-sig", newline="") as h:
        q = list(csv.DictReader(h))
    qi = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")
    def key(row):
        return tuple(row[c] if c == "prefecture" else Decimal(row[c]) for c in qi)
    qkeys = {key(row) for row in q}
    require(len(b) == 1049 and len(qkeys) == len(q) == 326, "original data dimensions differ")
    rare = [i for i, row in enumerate(b) if row["is_rare"] == "1"]
    private = {i for i, row in enumerate(b) if key(row) in qkeys}
    require(tuple(sorted(set(rare) & private)) == NEG and tuple(sorted(set(rare) - private)) == IDS, "original seven-QI rare partition differs")
    with CANDIDATE.open(encoding="utf-8-sig", newline="") as h:
        rows = list(csv.DictReader(h))
    require(len(rows) == 1049 and [row["record_id"] for row in rows] == [str(i) for i in range(1049)], "candidate record IDs/order differ")
    confidence = {team: np.array([float(row[str(team)]) for row in rows]) for team in TARGETS}
    mask = np.zeros(1049, dtype=bool)
    mask[rare] = True
    return confidence, mask, {str(p.relative_to(ROOT)): sha(p) for p in (bpath, qpath, CANDIDATE)}


def legal_state(shuffled):
    groups = []
    start = 0
    for count in N:
        groups.append(tuple(sorted(shuffled[start:start + count])))
        start += count
    return tuple(groups)


def local_signature_check():
    single = {i: tuple(int(i in q) for q in QUESTIONS) for i in IDS}
    pairs = {pair: tuple(single[pair[0]][k] + single[pair[1]][k] for k in range(5)) for pair in combinations(IDS, 2)}
    require(len(set(single.values())) == 11 and len(set(pairs.values())) == 55, "independent fixed-bit signatures collide")
    return {"single_cases": 11, "pair_cases": 55, "signature_collisions": 0}


def mean_values(state, confidence, mask, score_mia, official_mean):
    cfg = {"scoring": {"low_fpr": 0.01}}
    ordinary = np.flatnonzero(~mask)
    values = []
    count_vectors = []
    for k, selected in enumerate(QUESTIONS):
        scores = []
        counts = []
        for j, team in enumerate(TEAMS):
            c = confidence[team].copy()
            c[mask] = 0.1
            c[list(selected) + list(NEG[:M[j] - 5])] = 0.4
            if k == 0:
                require(np.array_equal(c[mask], confidence[team][mask]), "actual first-question rare CSV differs from the independently constructed agreed question")
                c = confidence[team]
            truth = np.zeros(1049, dtype=int)
            truth[list(state[j])] = 1
            truth[ordinary[:20]] = 1
            got = score_mia(c, truth, cfg, rare_mask=mask).breakdown["tpr_rare"]
            scores.append(got)
            counts.append(len(set(state[j]) & set(selected)))
        values.append(repr(official_mean(scores + [float("nan")] * 16)))
        count_vectors.append(tuple(counts))
    return values, count_vectors


def check_final(result, state):
    require(result["all_identified"] is True, "five official means did not identify all six teams")
    expected = {str(t): list(group) for t, group in zip(TEAMS, state)}
    require(result["identified_members"] == expected, "decoded team membership differs from synthetic legal truth")
    require(result["not_in_any_target"] == sorted(set(IDS) - {i for g in state for i in g}), "decoded outside-two IDs differ")
    require(sum(len(g) for g in result["identified_members"].values()) == 9, "decoded member total differs")


def reject(module, values, description):
    try:
        module.interpret(values)
    except ValueError as exc:
        return {"case": description, "rejected": True, "reason": str(exc)}
    raise ValueError(f"decoder failed to reject malformed or incompatible input: {description}")


def check_cli(values, expected=None, invalid=False, labeled=False):
    completed = subprocess.run([sys.executable, "-B", str(DECODER),
                                "--round-means" if labeled else "--means", *values],
                               capture_output=True, text=True, timeout=15)
    if invalid:
        require(completed.returncode != 0, "decoder CLI accepted invalid input")
        return {"returncode": completed.returncode, "rejected": True}
    require(completed.returncode == 0, f"decoder CLI failed: {completed.stderr}")
    result = json.loads(completed.stdout)
    require(result["writes_files"] is False and result["submitted"] is False, "decoder side-effect markers differ")
    if expected is not None:
        check_final(result, expected)
    return {"returncode": 0, "rounds_read": len(values), "all_identified": result["all_identified"],
            "input_mode": "round-means" if labeled else "means"}


def audit():
    decoder_digest = sha(DECODER)
    module = load_decoder()
    score_mia, official_mean, official_hashes = official_sources()
    confidence, mask, input_hashes = source_arrays()
    signatures = local_signature_check()
    rng = random.Random(SEED)
    samples = []
    while len(samples) < 100:
        shuffled = list(IDS)
        rng.shuffle(shuffled)
        state = legal_state(shuffled)
        if state not in samples:
            samples.append(state)
    cache = {}
    partial_cases = 0
    for state in samples:
        values, counts = mean_values(state, confidence, mask, score_mia, official_mean)
        cache[state] = values
        for prefix in range(1, 6):
            result = module.interpret(values[:prefix])
            require(result["read_rounds"] == list(range(3, 3 + prefix)), "decoder round numbering differs")
            require([tuple(o["selected_member_counts"][str(t)] for t in TEAMS) for o in result["observations"]] == counts[:prefix], "decoded official count vectors differ")
            for team, truth in zip(TEAMS, state):
                possibilities = result["remaining"][str(team)]["candidate_assignments"]
                require(list(truth) in possibilities, "partial decoding eliminated the true member subset")
            if prefix < 5:
                require(result["next_round"] == 3 + prefix, "fixed next round differs")
                require(result["next_fixed_question"] == list(QUESTIONS[prefix]), "next question became adaptive or has wrong bit order")
                require(result["next_padding_count"] == dict(zip(map(str, TEAMS), [m - 5 for m in M])), "next padding differs")
                partial_cases += 1
            if prefix == 5:
                check_final(result, state)
    targeted = 0
    coverage = {}
    for j, (team, count) in enumerate(zip(TEAMS, N)):
        covered = 0
        for members in combinations(IDS, count):
            remaining = [i for i in IDS if i not in members]
            state = [None] * 6
            state[j] = members
            start = 0
            for other in range(6):
                if other != j:
                    state[other] = tuple(remaining[start:start + N[other]])
                    start += N[other]
            state = tuple(state)
            if state not in cache:
                cache[state] = mean_values(state, confidence, mask, score_mia, official_mean)[0]
            check_final(module.interpret(cache[state]), state)
            covered += 1
            targeted += 1
        coverage[str(team)] = covered
    require(targeted == 198, "targeted legal-state coverage differs")
    bad_cases = [reject(module, [value], label) for value, label in (
        ("NaN", "NaN"), ("inf", "infinity"), ("-inf", "negative infinity"),
        ("1.0001", "above one"), ("-0.01", "negative score"),
        ("not-a-number", "malformed score"), ("1/0", "zero denominator"),
    )]
    bad_cases.append(reject(module, [], "empty observation history"))
    bad_cases.append(reject(module, cache[samples[0]] + [cache[samples[0]][0]], "six observations instead of five"))
    bad_cases.append(reject(module, ["0"], "all-zero first counts cannot assign nine distinct members to the six unselected candidates"))
    bad_cases.append(reject(module, [cache[samples[0]][0]] * 5, "repeated inconsistent five-round history"))
    rounded = None
    for state in samples:
        for value in cache[state]:
            four = f"{float(value):.4f}"
            if abs(Fraction(value) - Fraction(four)) > Fraction(1, 10 ** 12):
                try:
                    module.decode_mean(four)
                except ValueError:
                    rounded = (value, four)
                    break
        if rounded:
            break
    require(rounded is not None, "no meaningful four-digit rounding rejection case found")
    bad_cases.append(reject(module, [rounded[1]], "four-decimal rounded display incompatible with the exact alphabet"))
    forced = list(samples[0])
    forced[2] = forced[0]
    forced = tuple(forced)
    forced_means = mean_values(forced, confidence, mask, score_mia, official_mean)[0]
    bad_cases.append(reject(module, forced_means, "same identified individual forced into Team6 and Team11"))
    reorder_rejected = 0
    reorder_accepted = []
    for state in samples:
        values = cache[state][::-1]
        try:
            result = module.interpret(values)
        except ValueError:
            reorder_rejected += 1
        else:
            require(result["all_identified"], "five accepted reordered observations are incomplete")
            reorder_accepted.append({"original": {str(t): list(g) for t, g in zip(TEAMS, state)},
                                     "decoded_reversed": result["identified_members"],
                                     "means_reversed": values})
    require(reorder_rejected > 0, "no meaningful round-order rejection case found")
    # A concrete undetectable round-order error: the fixed codebook itself has
    # a nonidentity permutation symmetry. This is an input-origin limitation,
    # not a failure under the decoder's chronological-input contract.
    codes = {rid: tuple(int(rid in q) for q in QUESTIONS) for rid in IDS}
    inverse = {code: rid for rid, code in codes.items()}
    automorphisms = []
    for perm in permutations(range(5)):
        if perm == tuple(range(5)):
            continue
        moved = {rid: tuple(code[k] for k in perm) for rid, code in codes.items()}
        if set(moved.values()) == set(inverse):
            automorphisms.append((perm, {rid: inverse[code] for rid, code in moved.items()}))
    require(len(automorphisms) == 1, "independent codebook symmetry differs")
    perm, remap = automorphisms[0]
    symmetry_cases = []
    changed_memberships = 0
    for state in samples:
        values = [cache[state][k] for k in perm]
        remapped = tuple(tuple(sorted(remap[rid] for rid in group)) for group in state)
        result = module.interpret(values)
        check_final(result, remapped)
        changed_memberships += remapped != state
        if len(symmetry_cases) < 2 and remapped != state:
            symmetry_cases.append({"original": {str(t): list(g) for t, g in zip(TEAMS, state)},
                                   "decoded_misordered": result["identified_members"],
                                   "misordered_means": values})
    cli = [check_cli(cache[state], expected=state) for state in samples[:5]]
    cli.append(check_cli(cache[samples[0]][:2]))
    cli.append(check_cli(["NaN"], invalid=True))
    cli.append(check_cli(forced_means, invalid=True))
    symmetry_state = samples[0]
    remapped = tuple(tuple(sorted(remap[rid] for rid in group)) for group in symmetry_state)
    cli.append(check_cli([cache[symmetry_state][k] for k in perm], expected=remapped))
    labeled_random_cases = 0
    for state in samples:
        entries = [f"{k + 3}={value}" for k, value in enumerate(cache[state])]
        rng.shuffle(entries)
        ordered = module.labeled_values(entries)
        require(ordered == cache[state], "round labels did not restore chronological values")
        check_final(module.interpret(ordered), state)
        labeled_random_cases += 1
    first = cache[samples[0]]
    labeled_partial_cases = 0
    for prefix in range(1, 5):
        entries = [f"{k + 3}={value}" for k, value in enumerate(first[:prefix])][::-1]
        result = module.interpret(module.labeled_values(entries))
        require(result["next_fixed_question"] == list(QUESTIONS[prefix]), "reordered partial labels changed the fixed next question")
        labeled_partial_cases += 1
    label_rejections = []
    def reject_labels(entries, description):
        try:
            module.interpret(module.labeled_values(entries))
        except ValueError as exc:
            label_rejections.append({"case": description, "rejected": True, "reason": str(exc)})
        else:
            raise ValueError(f"labeled decoder accepted invalid input: {description}")
    for entries, description in (
        ([f"3={first[0]}", f"3={first[0]}"], "duplicate round"),
        ([f"3={first[0]}", f"03={first[0]}"], "numeric duplicate with leading zero"),
        ([f"4={first[1]}"], "missing round three"),
        ([f"3={first[0]}", f"5={first[2]}"], "missing intermediate round"),
        ([f"2={first[0]}"], "round below three"),
        ([f"8={first[0]}"], "round above seven"),
        ([f"three={first[0]}"], "non-numeric label"),
        ([f"３={first[0]}"], "non-ASCII label"),
        ([first[0]], "missing equals sign"),
        (["3="], "empty mean value"),
        (["3=NaN"], "labeled NaN"),
        (["3=inf"], "labeled infinity"),
        (["3=1.0001"], "labeled mean above one"),
        ([], "empty labeled history"),
    ):
        reject_labels(entries, description)
    labeled_cli = []
    canonical_entries = [f"{k + 3}={value}" for k, value in enumerate(first)]
    labeled_cli.append(check_cli(canonical_entries[::-1], expected=samples[0], labeled=True))
    # The same bit permutation that misdecodes bare values is harmless when
    # the values retain correct round labels and only argument order changes.
    labeled_cli.append(check_cli([canonical_entries[k] for k in perm], expected=samples[0], labeled=True))
    labeled_cli.append(check_cli(canonical_entries[:2][::-1], labeled=True))
    labeled_cli.append(check_cli([canonical_entries[0], canonical_entries[0]], invalid=True, labeled=True))
    labeled_cli.append(check_cli([canonical_entries[0], canonical_entries[2]], invalid=True, labeled=True))
    labeled_cli.append(check_cli(["3=NaN"], invalid=True, labeled=True))
    both = subprocess.run([sys.executable, "-B", str(DECODER), "--means", first[0],
                           "--round-means", canonical_entries[0]], capture_output=True, text=True, timeout=15)
    require(both.returncode != 0, "mutually exclusive input options were accepted together")
    labeled_cli.append({"case": "means and round-means combined", "rejected": True, "returncode": both.returncode})
    wrong_labels = [f"{k + 3}={first[perm[k]]}" for k in range(5)]
    wrong_result = module.interpret(module.labeled_values(wrong_labels))
    check_final(wrong_result, remapped)
    require(sha(DECODER) == decoder_digest, "decoder source changed during the audit")
    return {
        "audit_status": "PASS", "decoder_sha256": decoder_digest, "auditor_sha256": sha(Path(__file__).resolve()),
        "seed": SEED, "random_legal_configuration_cases": 100,
        "partial_prefix_cases": partial_cases, "targeted_legal_configuration_cases": targeted,
        "targeted_single_pair_coverage_by_team": coverage,
        "all_five_mean_results_identify_nine_and_outside_two": True,
        "partial_prefix_true_candidate_retention": True, "next_fixed_question_and_padding_match": True,
        "mean_generation": "actual official score_mia on 1049-row confidence arrays and AST-extracted official _nanmean; actual candidate used for round 3; agreed fixed rare scores substituted in memory for rounds 4-7",
        "synthetic_truth_notice": "all test assignments are synthetic; no real individual membership is asserted",
        "local_signatures": signatures, "rejections": bad_cases,
        "cli_checks": cli, "round_order_experiment": {
            "reversed_histories_tested": 100, "rejected": reorder_rejected,
            "accepted_as_legal_alternative": len(reorder_accepted),
            "accepted_examples": reorder_accepted[:2],
            "limitation": "A mean-only API cannot always detect shuffled round order if it happens to encode another legal assignment; chronological round identity must be supplied correctly by the operator.",
        },
        "recommended_labeled_input_checks": {
            "random_correctly_labeled_argument_permutations": labeled_random_cases,
            "reversed_partial_label_cases": labeled_partial_cases,
            "all_restore_correct_membership_or_next_question": True,
            "rejections": label_rejections, "cli_cases": labeled_cli,
            "both_modes_are_mutually_exclusive": True,
            "wrong_labels_cannot_prove_value_origin": {
                "mislabelled_example_accepted": True,
                "input": wrong_labels,
                "decoded": wrong_result["identified_members"],
                "limitation": "Round labels prevent argument-order mistakes, but a value deliberately or accidentally given the wrong label still needs submission/result provenance checking.",
            },
        },
        "concrete_undetectable_round_order_error": {
            "nonidentity_bit_permutation_automorphisms": len(automorphisms),
            "permutation_zero_based": list(perm),
            "input_round_order": [3 + k for k in perm],
            "candidate_id_remapping": {str(rid): mapped for rid, mapped in remap.items()},
            "legal_cases_accepted_after_this_misordering": len(samples),
            "cases_with_changed_member_assignments": changed_memberships,
            "examples": symmetry_cases,
            "required_operation": "Match each raw detailed-results mean to its actual submission CSV and round before chronological entry. Numeric validity and disjointness checks cannot replace that provenance check.",
        },
        "four_decimal_rejection_example": {"raw_official_mean": rounded[0], "four_decimal_display": rounded[1]},
        "precision_origin_limitation": "A decimal string alone cannot prove it is raw; a rounded display that coincides exactly with an alphabet value cannot be identified as rounded. Use the raw detailed-results key, not the display.",
        "source_sha256": {**input_hashes, **official_hashes},
        "review": "No membership misidentification, false true-candidate exclusion, wrong fixed next query, or failure to reject incompatible/no-overlap histories was found under the documented chronological raw-mean input contract.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(HERE) and out.suffix == ".json" and not out.exists(), "report must be a new explicit JSON file in this analysis folder")
    report = audit()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("audit_status", "random_legal_configuration_cases", "partial_prefix_cases", "targeted_legal_configuration_cases", "round_order_experiment")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
