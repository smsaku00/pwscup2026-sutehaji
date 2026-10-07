"""Independent auditor of the agreed, non-adaptive five-bit rare MIA plan.

This checker never imports the builder, its settings, or its output tables.
It reads original inputs and the explicitly selected candidate. Only an
explicit --output report is written. Historical inputs are never modified.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
from decimal import Decimal
from fractions import Fraction
import hashlib
from itertools import combinations, product
import json
from math import factorial
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ATTACK = HERE.parent
INPUT = ATTACK / "20260928_配布データ基本分析/入力"
TARGETS = tuple(range(18)) + (22, 25, 26, 27)
ACTIVE = (6, 10, 11, 16, 17, 25)
COUNTS = (1, 2, 1, 2, 1, 2)
TOTAL_HIGH = (6, 7, 7, 11, 10, 12)
QI = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")
AGREED_IDS = (47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675)
AGREED_NEGATIVE = (126, 136, 271, 273, 317, 440, 480, 625, 637, 643, 668, 814, 841)
# Left-to-right is submission round 3, 4, 5, 6, 7. These are the human-agreed
# questions, not values imported from a builder-generated plan.
QUESTIONS = (
    (115, 169, 264, 383, 462),
    (47, 115, 304, 462, 675),
    (47, 90, 264, 462, 495),
    (47, 76, 264, 304, 383),
    (47, 76, 90, 115, 169),
)
COUNT_SOURCE = ATTACK / (
    "20261006_MIA第2回結果と人数導出/公開用/MIA/"
    "20261006_MIA第2回_seo02と希少同点/採点結果/MIA結果.json"
)
COUNT_SHA256 = "411e00d38ffbe7cb38bb73c654b129ca6377a198ad9031e409cccdc19dd725a4"
SCORE_SHA256 = "1e62a591bf5f564eef30bd64edb4e8146563ab497a249ac15230ef61178186c6"
AGGREGATION_SHA256 = "32df3a36762ca876449f6c635f4ed8d5fc3ecb5f322697ff167d6c17949d5a9b"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_path(paths, message: str) -> Path:
    found = list(paths)
    require(len(found) == 1, message)
    return found[0].resolve()


def read_dicts(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV header missing: {path}")
        require(len(set(reader.fieldnames)) == len(reader.fieldnames), "duplicate CSV columns")
        rows = list(reader)
        require(all(None not in row and None not in row.values() for row in rows), "malformed CSV row")
        return tuple(reader.fieldnames), rows


def qi_key(row):
    # Decimal compares the original decimal values exactly; no rounded merge,
    # nearest-neighbour matching, or tolerance is used for private AIA rows.
    return tuple(row[c] if c == "prefecture" else Decimal(row[c]) for c in QI)


def original_inputs():
    bpath = unique_path((ROOT / "本戦").glob("配布*/PWSCup2026_final_team28_*/B_self.csv"), "B_self path is not unique")
    qpath = INPUT / "aia_challenge_28.csv"
    declared = INPUT / "mia_columns.json"
    _, b = read_dicts(bpath)
    _, q = read_dicts(qpath)
    require(len(b) == 1049, "B_self row count differs from 1049")
    require([int(row["record_id"]) for row in b] == list(range(1049)), "B_self record_id order differs")
    bkeys = [qi_key(row) for row in b]
    qkeys = [qi_key(row) for row in q]
    require(len(set(bkeys)) == len(bkeys), "B_self seven-QI duplicate")
    require(len(set(qkeys)) == len(qkeys) and len(q) == 326, "self AIA challenge duplicate or wrong row count")
    private = {i for i, key in enumerate(bkeys) if key in set(qkeys)}
    require(all(row["is_rare"] in ("0", "1") for row in b), "invalid B_self is_rare")
    rare = tuple(i for i, row in enumerate(b) if row["is_rare"] == "1")
    negative = tuple(sorted(set(rare) & private))
    unknown = tuple(sorted(set(rare) - private))
    require((len(private), len(rare), len(negative), len(unknown)) == (163, 24, 13, 11), "original-input rare/private partition differs")
    require(unknown == AGREED_IDS and negative == AGREED_NEGATIVE, "derived candidate or known-negative IDs differ from the agreed IDs")
    columns = json.loads(declared.read_text(encoding="utf-8"))["target_columns"]
    require(tuple(int(t) for t in columns if int(t) != 28) == TARGETS, "official target columns differ")
    require(digest(COUNT_SOURCE) == COUNT_SHA256, "prior official-score-derived count source changed")
    members = json.loads(COUNT_SOURCE.read_text(encoding="utf-8"))["derived_rare_members"]
    expected = {str(t): COUNTS[ACTIVE.index(t)] if t in ACTIVE else 0 for t in TARGETS}
    require(members == expected, "prior count result differs from the agreed rare counts")
    return b, rare, negative, unknown, {
        "private_records": len(private), "rare_records": len(rare),
        "known_negative_rare_records": len(negative), "unknown_rare_records": len(unknown),
        "matching": "exact decimal equality of seven QIs; one-to-one keys",
        "rare_ids": list(rare), "known_negative_ids": list(negative), "candidate_ids": list(unknown),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (bpath, qpath, declared, COUNT_SOURCE)},
        "count_status": "conditional result derived from prior official scores, not a released individual truth map",
    }


def read_candidate(path: Path):
    header, rows = read_dicts(path)
    require(header == ("record_id",) + tuple(map(str, TARGETS)), "candidate target columns/order differ")
    require(len(rows) == 1049, "candidate must contain exactly 1049 rows")
    for i, row in enumerate(rows):
        require(row["record_id"] == str(i), f"candidate record_id differs at row {i}")
        for team in TARGETS:
            value = Decimal(row[str(team)])
            require(value.is_finite() and Decimal(0) <= value <= Decimal(1), f"candidate confidence invalid at record {i}, Team{team}")
    return header, rows


def check_rare_cells(header, rows, rare, negative, unknown):
    require(header == ("record_id",) + tuple(map(str, TARGETS)), "target columns/order differ")
    require(len(rows) == 1049, "wrong candidate row count")
    require([row["record_id"] for row in rows] == [str(i) for i in range(1049)], "record IDs or order changed")
    for team in TARGETS:
        if team in ACTIVE:
            high_total = TOTAL_HIGH[ACTIVE.index(team)]
            high_ids = set(QUESTIONS[0]) | set(negative[:high_total - len(QUESTIONS[0])])
            require(len(high_ids) == high_total, "high-side size or padding overlap differs")
            for rid in rare:
                expected = 0.4 if rid in high_ids else 0.1
                # Official score_mia reads confidence as binary64. The 17-digit
                # round-trip CSV spelling of 0.1/0.4 is the same tied score,
                # although its decimal string differs from Decimal('0.1').
                # Compare parsed binary64 exactly, with no tolerance.
                require(float(rows[rid][str(team)]) == expected, f"rare confidence differs: record {rid}, Team{team}")
        else:
            for rid in rare:
                require(Decimal(rows[rid][str(team)]) == Decimal(0), f"zero-member rare cell is not zero: record {rid}, Team{team}")
    require(set(unknown) == set(AGREED_IDS), "unknown IDs changed")
    return {
        "all_rare_cells_checked": len(rare) * len(TARGETS),
        "active_rare_cells_checked": len(rare) * len(ACTIVE),
        "zero_member_team_cells_checked": len(rare) * (len(TARGETS) - len(ACTIVE)),
        "first_question_ids": list(QUESTIONS[0]), "team_order": list(ACTIVE),
        "counts": list(COUNTS), "total_high": list(TOTAL_HIGH),
        "padding_counts": [m - 5 for m in TOTAL_HIGH],
    }


def signatures(ids=AGREED_IDS, questions=QUESTIONS):
    require(len(questions) == 5 and all(len(q) == 5 and len(set(q)) == 5 for q in questions), "invalid five-bit question sizes")
    require(all(set(q) <= set(ids) for q in questions), "question contains a non-candidate ID")
    single = {rid: tuple(int(rid in q) for q in questions) for rid in ids}
    require(len(set(single.values())) == 11, "single-member five-bit collision")
    pairs = {pair: tuple(sum(int(rid in q) for rid in pair) for q in questions) for pair in combinations(ids, 2)}
    require(len(pairs) == 55 and len(set(pairs.values())) == 55, "two-member ternary-count signature collision")
    return single, pairs, {
        "questions": [list(q) for q in questions],
        "bit_order": "submission rounds 3,4,5,6,7 from left to right",
        "single_signatures": {str(i): list(v) for i, v in single.items()},
        "single_cases": 11, "pair_cases": 55, "single_collisions": 0, "pair_collisions": 0,
        "legal_joint_configurations": factorial(11) // (2 ** 3 * factorial(2)),
        "proof_type": "composition of independent team signatures; no joint-state replay claimed",
        "proof": "Each of the three one-member teams is uniquely identified among 11 single signatures, and each of the three two-member teams among 55 pair signatures. The six team answers are recovered separately from each of five means. Therefore any two distinct legal joint configurations differ in at least one team's unique five-count signature. This covers all 2,494,800 disjoint legal configurations and also the larger product set permitting team overlap.",
    }


def exact_roc(n: int, high: int, hit: int) -> Fraction:
    # Derive the low-FPR value from the two tied score groups as ROC points,
    # rather than trusting a builder's closed formula or saved lookup table.
    points = ((Fraction(0), Fraction(0)),
              (Fraction(high - hit, 24 - n), Fraction(hit, n)),
              (Fraction(1), Fraction(1)))
    target = Fraction(1, 100)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= target <= x1 and x1 > x0:
            return y0 + (target - x0) * (y1 - y0) / (x1 - x0)
    raise ValueError("target FPR falls outside the independently constructed ROC")


def official_functions():
    kit = unique_path((ROOT / "本戦").glob("配布*/pwscup2026-kit-main_revised0912"), "official kit path is not unique")
    src = kit / "starter/src"
    score_path = src / "pwscup2026/scoring/attack_score.py"
    agg_path = src / "pwscup2026/kit/codabench/scoring_io.py"
    require(digest(score_path) == SCORE_SHA256, "official score_mia source changed")
    require(digest(agg_path) == AGGREGATION_SHA256, "official aggregation source changed")
    sys.path.insert(0, str(src))
    from pwscup2026.scoring.attack_score import score_mia
    tree = ast.parse(agg_path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_finite", "_nanmean")]
    require({n.name for n in nodes} == {"_finite", "_nanmean"}, "official aggregation functions missing")
    env = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(agg_path), "exec"), env)
    return score_mia, env["_nanmean"], {str(p.relative_to(ROOT)): digest(p) for p in (score_path, agg_path)}


def check_score_alphabet(score_mia, official_mean):
    cfg = {"scoring": {"low_fpr": 0.01}}
    lookup = {}
    error = 0.0
    per_team = 0
    for team_index, n in enumerate(COUNTS):
        total = TOTAL_HIGH[team_index]
        for selected in range(6):
            require(1 <= total - selected <= 13, "illegal generic known-negative padding")
            for hit in range(min(n, selected) + 1):
                confidence = np.full(24, 0.1)
                confidence[:selected] = 0.4
                confidence[11:11 + total - selected] = 0.4
                truth = np.zeros(24, dtype=int)
                truth[:hit] = 1
                truth[selected:selected + n - hit] = 1
                got = score_mia(confidence, truth, cfg, rare_mask=np.ones(24, dtype=bool)).breakdown["tpr_rare"]
                expected = float(exact_roc(n, total, hit))
                delta = abs(got - expected)
                require(delta < 1e-14, "official per-team rare score differs from independent ROC")
                error = max(error, delta)
                lookup[team_index, hit] = got
                per_team += 1
    means = {}
    for hits in product(*(range(n + 1) for n in COUNTS)):
        exact = sum((exact_roc(n, m, h) for n, m, h in zip(COUNTS, TOTAL_HIGH, hits)), Fraction()) / 6
        means[hits] = exact
        got = official_mean([lookup[i, h] for i, h in enumerate(hits)] + [float("nan")] * 16)
        delta = abs(got - float(exact))
        require(delta < 1e-14, "official rare mean differs from independent fraction mean")
        error = max(error, delta)
    require(per_team == 78 and len(means) == 216 and len(set(means.values())) == 216, "score alphabet case count or collision differs")
    ordered = sorted(means.values())
    gap = min(b - a for a, b in zip(ordered, ordered[1:]))
    require(gap == Fraction(1, 54000), "score alphabet minimum gap differs")
    # Record what precision is sufficient, without interpreting the four-digit
    # leaderboard display as an exact score.
    rounded5 = [round(float(x), 5) for x in ordered]
    require(len(set(rounded5)) == 216, "five-decimal score alphabet collision")
    require(np.isnan(score_mia(np.zeros(24), np.zeros(24, dtype=int), cfg, rare_mask=np.ones(24, dtype=bool)).breakdown["tpr_rare"]), "zero-positive rare team must yield NaN")
    require(official_mean([float("nan")] * 22) == 0.0, "all-NaN official mean differs")
    return {
        "official_per_team_cases": per_team, "official_mean_cases": len(means),
        "score_collisions": 0, "minimum_gap_exact": str(gap),
        "maximum_official_score_error": error, "five_decimal_alphabet_unique": True,
        "operational_score_source": "raw detailed-results mia.tpr_rare; four-decimal leaderboard display is insufficient",
        "zero_positive_team_nan_exclusion": True,
    }


def check_actual_candidate_scores(rows, b, rare, unknown, score_mia):
    rare_mask = np.array([row["is_rare"] == "1" for row in b], dtype=bool)
    ordinary = [i for i in range(1049) if not rare_mask[i]]
    cfg = {"scoring": {"low_fpr": 0.01}}
    cases = 0
    error = 0.0
    zero_cases = 0
    for team in TARGETS:
        confidence = np.array([float(row[str(team)]) for row in rows])
        if team not in ACTIVE:
            truth = np.zeros(1049, dtype=int)
            truth[ordinary[:20]] = 1
            require(np.isnan(score_mia(confidence, truth, cfg, rare_mask=rare_mask).breakdown["tpr_rare"]), "actual zero-member team was not excluded from rare mean")
            zero_cases += 1
            continue
        i = ACTIVE.index(team)
        for members in combinations(unknown, COUNTS[i]):
            truth = np.zeros(1049, dtype=int)
            truth[list(members)] = 1
            truth[ordinary[:20]] = 1
            hit = len(set(members) & set(QUESTIONS[0]))
            expected = float(exact_roc(COUNTS[i], TOTAL_HIGH[i], hit))
            got = score_mia(confidence, truth, cfg, rare_mask=rare_mask).breakdown["tpr_rare"]
            require(abs(got - expected) < 1e-14, "actual CSV rare score differs for a legal single/pair candidate")
            error = max(error, abs(got - expected))
            changed = confidence.copy()
            changed[~rare_mask] = 1.0 - changed[~rare_mask]
            other = score_mia(changed, truth, cfg, rare_mask=rare_mask).breakdown["tpr_rare"]
            require(got == other, "nonrare confidence changes affected the rare score")
            cases += 1
    require(cases == 198 and zero_cases == 16, "actual CSV official score case count differs")
    return {"actual_csv_single_and_pair_cases": cases,
            "actual_zero_positive_team_cases": zero_cases,
            "nonrare_isolation_cases": cases,
            "maximum_actual_csv_score_error": error,
            "synthetic_truth_notice": "all single/pair possibilities used for scorer checking; no unreleased individual truth was used"}


def mutation_rejections(header, rows, rare, negative, unknown):
    results = []
    def reject(name, mutate):
        changed = copy.deepcopy(rows)
        changed_header = list(header)
        mutate(changed_header, changed)
        try:
            check_rare_cells(tuple(changed_header), changed, rare, negative, unknown)
        except (ValueError, KeyError, ArithmeticError):
            results.append({"mutation": name, "rejected": True})
        else:
            raise ValueError(f"auditor failed to reject deliberate mutation: {name}")
    reject("misidentified_record", lambda h, r: r[AGREED_IDS[0]].update(record_id="1049"))
    def wrong_bit(h, r):
        # Replace q3 by the last fixed question while preserving high-side size.
        for rid in unknown:
            r[rid]["6"] = "0.4" if rid in QUESTIONS[4] else "0.1"
    reject("reversed_first_and_last_bits", wrong_bit)
    reject("broken_high_score_tie", lambda h, r: r[QUESTIONS[0][0]].update({"6": "0.4000000000001"}))
    def moved_padding(h, r):
        r[negative[0]]["6"] = "0.1"
        r[negative[-1]]["6"] = "0.4"
    reject("shifted_known_negative_padding", moved_padding)
    reject("zero_member_team_rare_zero_omission", lambda h, r: r[rare[0]].update({"0": "0.1"}))
    reject("active_unknown_candidate_false_zero", lambda h, r: r[unknown[0]].update({"6": "0"}))
    reject("nonfinite_rare_confidence", lambda h, r: r[rare[0]].update({"6": "NaN"}))
    reject("swapped_team_column_order", lambda h, r: h.__setitem__(slice(1, 3), h[1:3][::-1]))
    reject("removed_candidate_row", lambda h, r: r.pop(rare[0]))
    duplicate_question = (QUESTIONS[0],) * 5
    try:
        signatures(questions=duplicate_question)
    except ValueError:
        results.append({"mutation": "colliding_five_bit_signatures", "rejected": True})
    else:
        raise ValueError("auditor failed to reject colliding five-bit signatures")
    return results


def audit(candidate: Path):
    b, rare, negative, unknown, original = original_inputs()
    header, rows = read_candidate(candidate)
    cells = check_rare_cells(header, rows, rare, negative, unknown)
    _, _, sig = signatures()
    score_mia, official_mean, sources = official_functions()
    alphabet = check_score_alphabet(score_mia, official_mean)
    actual = check_actual_candidate_scores(rows, b, rare, unknown, score_mia)
    mutations = mutation_rejections(header, rows, rare, negative, unknown)
    return {
        "audit_status": "PASS", "candidate": str(candidate), "candidate_sha256": digest(candidate),
        "auditor_sha256": digest(Path(__file__).resolve()),
        "auditor_independence": "no import of builder, generated settings, or generated encoding; exact original-input matching; direct official score_mia and AST-extracted official aggregation functions",
        "original_inputs": original, "rare_csv_cells": cells, "fixed_five_bit_signatures": sig,
        "official_score_alphabet": alphabet, "actual_csv_official_scoring": actual,
        "official_source_sha256": sources, "deliberate_mutations": mutations,
        "scope": "rare MIA and agreed fixed five-bit operation; nonrare calibrated ranking is audited separately",
        "conditional_guarantee": "Assumes the previously derived team counts and public sharing conditions, unmodified score_mia/aggregation, correct future five fixed question CSVs, and sufficient score precision. This audit does not claim to have observed the actual individual memberships.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    output = args.output.resolve()
    require(output.is_relative_to(HERE) and output.suffix == ".json", "report must be an explicitly named JSON file within this new analysis folder")
    require(output != candidate and not output.exists(), "report output must be a new file and must not overwrite the candidate")
    try:
        result = audit(candidate)
    except Exception as exc:
        result = {"audit_status": "FAIL", "candidate": str(candidate),
                  "error_type": type(exc).__name__, "error": str(exc),
                  "auditor_sha256": digest(Path(__file__).resolve())}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        return 1
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_status": "PASS", "candidate_sha256": result["candidate_sha256"],
                      "report": str(output), "rare_cells": 528, "official_actual_cases": 198,
                      "mutation_rejections": len(result["deliberate_mutations"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
