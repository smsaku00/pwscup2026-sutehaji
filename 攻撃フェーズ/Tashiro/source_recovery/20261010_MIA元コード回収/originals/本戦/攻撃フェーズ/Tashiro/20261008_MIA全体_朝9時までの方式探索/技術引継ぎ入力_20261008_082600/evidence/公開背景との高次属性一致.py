"""公開C42と公開A3の高次属性一致と、独立周辺抽出の固定記述を集計。

root .venv/bin/python -B 本script --write-plan を先に実行し、その後引数なし。
個人ID、行位置、値tupleをexportしない。B/Q/所属正解/採点を読まない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CLINICAL = ("BMI", "SBP", "TG", "HDL", "ALT", "FPG")
NUMERIC = ("age",) + CLINICAL + ("time",)
QI7 = ("age", "sex", "prefecture", "SBP", "TG", "HDL", "ALT")
ALL13 = NUMERIC + ("sex", "prefecture", "smoking", "onset", "death")
SETS = [(kind, "+".join(cols), cols) for size, kind in
        [(2, "clinical_pair"), (3, "clinical_triple"), (4, "clinical_fourset")]
        for cols in combinations(CLINICAL, size)] + [
        ("clinical6", "clinical6", CLINICAL), ("QI7", "QI7", QI7), ("all13", "all13", ALL13)]
GROUPS = ("all", "sex0", "sex1")
OLD_CSV = HERE / "公開背景との属性単位一致結果/属性組と行全体の公開値一致.csv"
METHOD = {
    "version": 1,
    "inputs": "3public A + 42attack C only; prelim own5 and main own28 C excluded",
    "groups": list(GROUPS), "group_reference": "C sex group uses the same sex A group; all uses all A",
    "clinical_columns": list(CLINICAL), "QI7": list(QI7), "all13": list(ALL13),
    "set_counts": {"clinical_pair": 15, "clinical_triple": 20, "clinical_fourset": 15,
                   "clinical6": 1, "QI7": 1, "all13": 1},
    "aggregate_rows": 42 * 3 * 3 * 53,
    "read_precision": "pandas float_precision=round_trip; exact numeric equality, no tolerance merging",
    "observed": "Number of C rows whose attribute tuple is in the unique matching A-group tuple support",
    "expected_matched_rows": "n_C * sum over unique A-group tuples a of product_j(freq_Cj(a_j)/n_C)",
    "counterfactual": "Within this C group, each attribute draws independently with replacement from its empirical marginal; A tuples deduplicated",
    "expected_is_not_estimated_actual_processing": True,
    "same_rule_all_targets": True,
    "independent_checks": ["Finite toy exact Fraction formula vs exhaustive independent row-index choices for each attribute",
                           "All observed aggregate counts: tuple Counter/set vs pandas MultiIndex.isin",
                           "All-group 15pairs/full6/QI7/all13 vs frozen old CSV, all 3A and 42C"],
    "exports": "Aggregate counts/fractions/expected counts only; no individual ID, row position, value tuple, or NPZ",
    "limits": ["Observed cooccurrence differs from a fixed marginal-independence counterfactual only descriptively.",
               "This counterfactual does not identify the true processing mechanism or its causal effect.",
               "No p-value, significance, algorithm proof, MIA performance or score/method selection.",
               "Only current-phase A has same-phase nonmember guarantee; historical A/kit guarantee not extended."],
    "b_used": False, "q_used": False, "membership_labels_used": False,
    "official_scores_used": False, "method_selection": False,
    "final_prose_or_github_created": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(data):
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def stamp():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def source_paths():
    pre = ROOT / "予備戦/配布データ"
    main_a = sorted((ROOT / "本戦/配布データ").glob("PWSCup2026_final_team28_*/A_bg.csv"))
    assert len(main_a) == 1
    refs = {"prelim_A": pre / "PWSCup2026_prelim_common/A_bg.csv", "main_A": main_a[0],
            "kit_A": ROOT / "スターターキット/初版/participant_data/A_bg.csv"}
    targets = []
    for phase, folder, own, expected in [
        ("prelim", pre / "PWSCup2026_prelim_attack", 5, 20),
        ("main", HERE.parent / "20260928_配布データ基本分析/入力", 28, 22)]:
        paths = sorted(p for p in folder.glob("C_*.csv") if int(p.stem.split("_")[1]) != own)
        assert len(paths) == expected
        targets.extend((phase, int(p.stem.split("_")[1]), p) for p in paths)
    return refs, targets


def frozen_hashes():
    files = [HERE / "公開背景との属性単位一致.py"] + list(OLD_CSV.parent.iterdir())
    return {str(p.resolve()): sha(p) for p in sorted(files) if p.is_file()}


def group_frame(frame, group):
    return frame if group == "all" else frame.loc[frame.sex.eq(int(group[-1]))]


def expected_count(unique_ref, marginals, n):
    weights = np.ones(len(unique_ref), dtype=float)
    for col in unique_ref.columns:
        weights *= unique_ref[col].map(marginals[col]).fillna(0).to_numpy(float) / n
    return n * math.fsum(weights.tolist())


def toy_checks():
    cases = [
        ([(0, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (1, 0)]),
        ([(0, "x", 1), (1, "x", 0), (1, "y", 1)], [(0, "y", 0), (1, "x", 1)]),
        ([(0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 0)],
         [(0, 0, 0, 0, 0, 0), (1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1)]),
        ([(0, 0), (1, 1)], [(2, 2), (2, 2)]),
    ]
    records = []
    for i, (c_rows, a_rows) in enumerate(cases):
        n, dim = len(c_rows), len(c_rows[0])
        ref = set(a_rows)
        margins = [Counter(row[j] for row in c_rows) for j in range(dim)]
        formula = sum((math.prod(Fraction(margins[j][row[j]], n) for j in range(dim))
                       for row in ref), Fraction(0)) * n
        matches = sum(tuple(c_rows[idx[j]][j] for j in range(dim)) in ref
                      for idx in product(range(n), repeat=dim))
        enumerated = Fraction(matches, n ** dim) * n
        columns = list(range(dim))
        frame = pd.DataFrame(list(ref), columns=columns)
        numeric = expected_count(frame, {j: margins[j] for j in columns}, n)
        assert formula == enumerated
        assert abs(numeric - float(enumerated)) <= 1e-14
        records.append({"case": i, "C_rows": n, "dimensions": dim, "A_rows": len(a_rows),
                        "A_unique_tuples": len(ref), "enumerated_draws": n ** dim,
                        "expected_numerator": enumerated.numerator,
                        "expected_denominator": enumerated.denominator,
                        "numeric_absolute_error": abs(numeric - float(enumerated))})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "公開背景との高次属性一致結果")
    args = parser.parse_args()
    assert Path(sys.executable).parent == ROOT / ".venv/bin"
    refs, targets = source_paths()
    code, out = Path(__file__).resolve(), args.output_dir.resolve()
    if args.write_plan:
        assert not out.exists()
        out.mkdir()
        sources = {str(p.resolve()): sha(p) for p in list(refs.values()) + [p for _, _, p in targets]}
        (out / "方法.json").write_text(dump({"created_jst": stamp(), "method": METHOD,
            "code_sha256_before_execution": sha(code), "source_sha256_before_execution": sources,
            "frozen_artifact_sha256": frozen_hashes()}), encoding="utf-8")
        print(dump({"stage": "pre_execution_plan_fixed", "code_sha256": sha(code),
                    "source_count": len(sources), "expected_aggregate_rows": METHOD["aggregate_rows"]}), end="")
        return
    started, execution_started = time.perf_counter(), stamp()
    assert out.is_dir() and {p.name for p in out.iterdir()} == {"方法.json"}
    plan = json.loads((out / "方法.json").read_text())
    assert plan["method"] == METHOD and plan["code_sha256_before_execution"] == sha(code)
    assert plan["frozen_artifact_sha256"] == frozen_hashes()
    inputs = {}
    def read(path):
        assert path.name == "A_bg.csv" or path.name.startswith("C_")
        assert not any(s in str(path) for s in ("B_self", "aia_challenge", "pool_ids", "正解データ", "アーカイブ"))
        key = str(path.resolve()); inputs[key] = sha(path)
        assert inputs[key] == plan["source_sha256_before_execution"][key]
        d = pd.read_csv(path, float_precision="round_trip", usecols=list(ALL13))
        assert not d.isna().any().any() and np.isfinite(d[list(NUMERIC)].to_numpy(float)).all()
        assert set(d.sex.unique()) == {0, 1}
        return d
    backgrounds = {name: read(path) for name, path in refs.items()}
    ref_cache = {}
    for source, frame in backgrounds.items():
        assert len(frame) == 5143
        for group in GROUPS:
            a = group_frame(frame, group)
            for _, label, cols in SETS:
                unique = a[list(cols)].drop_duplicates()
                tuples = set(unique.itertuples(index=False, name=None))
                ref_cache[source, group, label] = (unique, tuples, pd.MultiIndex.from_frame(unique), len(a))
    toy = toy_checks()
    old = pd.read_csv(OLD_CSV, float_precision="round_trip")
    old = old.loc[old.reference_source.isin(refs)]
    old_lookup = {(r.phase, r.team, r.reference_source, r.attribute_set): r
                  for r in old.itertuples(index=False)}
    rows, checked, contrasted = [], 0, 0
    for index, (phase, team, path) in enumerate(targets):
        c_full = read(path); assert len(c_full) == 1049
        for group in GROUPS:
            c = group_frame(c_full, group); n = len(c); assert n > 0
            margins = {col: Counter(c[col].tolist()) for col in ALL13}
            for kind, label, cols in SETS:
                cc = Counter(c[list(cols)].itertuples(index=False, name=None))
                c_index = pd.MultiIndex.from_frame(c[list(cols)])
                for source in refs:
                    unique, ac, a_index, a_rows = ref_cache[source, group, label]
                    shared = set(cc) & ac
                    observed = sum(cc[key] for key in shared)
                    assert observed == int(c_index.isin(a_index).sum())
                    expected = expected_count(unique, margins, n)
                    assert math.isfinite(expected) and -1e-12 <= expected <= n + 1e-12
                    assert not observed or expected > 0
                    if group == "all" and kind in ("clinical_pair", "clinical6", "QI7", "all13"):
                        previous = old_lookup[phase, team, source, label]
                        assert observed == previous.exact_matching_c_rows
                        assert len(shared) == previous.exact_matching_unique_c_tuples
                        assert len(cc) == previous.c_unique_tuples
                        assert len(ac) == previous.reference_unique_tuples
                        contrasted += 1
                    rows.append({"phase": phase, "team": team, "reference_source": source, "group": group,
                        "attribute_set_kind": kind, "attribute_set": label, "attribute_count": len(cols),
                        "c_rows": n, "reference_rows": a_rows, "c_unique_tuples": len(cc),
                        "reference_unique_tuples": len(ac), "reference_duplicate_excess": a_rows - len(ac),
                        "exact_matching_c_rows": observed, "exact_matching_c_fraction": observed / n,
                        "exact_matching_unique_c_tuples": len(shared),
                        "independent_marginals_expected_matching_c_rows": expected,
                        "independent_marginals_expected_matching_fraction": expected / n,
                        "observed_minus_expected_rows": observed - expected,
                        "observed_to_expected_ratio": observed / expected if expected > 0 else None,
                        "counterfactual_descriptive_only": True, "membership_accuracy_unknown": True})
                    checked += 1
        if (index + 1) % 6 == 0:
            print(dump({"stage": "aggregate_progress", "targets_complete": index + 1,
                        "aggregate_rows": len(rows), "elapsed_seconds": time.perf_counter() - started}), end="", flush=True)
    assert len(rows) == checked == METHOD["aggregate_rows"]
    assert contrasted == 42 * 3 * 18
    assert set(inputs) == set(plan["source_sha256_before_execution"])
    for p, value in inputs.items(): assert sha(p) == value
    assert plan["frozen_artifact_sha256"] == frozen_hashes()
    pd.DataFrame(rows).sort_values(["phase", "team", "reference_source", "group", "attribute_set_kind", "attribute_set"]).to_csv(
        out / "高次属性の同時一致と独立周辺期待.csv", index=False, float_format="%.17g")
    (out / "独立検証.json").write_text(dump({"toy_exhaustive_checks": toy,
        "observed_tupleCounter_vs_MultiIndex_cases": checked, "maximum_observed_count_difference": 0,
        "frozen_previous_CSV_cases": contrasted, "maximum_previous_CSV_count_difference": 0,
        "input_files": len(inputs), "no_GT": True}), encoding="utf-8")
    outputs = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "実行記録.json").write_text(dump({"started_jst": execution_started, "completed_jst": stamp(),
        "elapsed_seconds": time.perf_counter() - started, "code_sha256": sha(code),
        "source_sha256": inputs, "output_sha256": outputs,
        "frozen_artifact_unchanged": True, "python": sys.executable, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "aggregate_rows": len(rows), "B_used": False, "Q_used": False, "GT_used": False,
        "official_scores_used": False, "final_prose_or_github_created": False}), encoding="utf-8")
    print(dump({"stage": "complete", "aggregate_rows": len(rows), "contrasts": contrasted,
                "elapsed_seconds": time.perf_counter() - started}), end="")


if __name__ == "__main__":
    main()
