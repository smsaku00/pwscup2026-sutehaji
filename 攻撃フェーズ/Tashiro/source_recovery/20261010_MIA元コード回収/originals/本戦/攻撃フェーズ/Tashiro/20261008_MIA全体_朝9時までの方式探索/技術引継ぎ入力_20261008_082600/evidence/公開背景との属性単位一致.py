"""3公開Aと42公開Cの属性単位・属性組の一致、time右端、age支持を診断する。

ルート .venv/bin/python -B で --write-plan を先に実行し、引数なしで計算する。
技術CSV/JSONのみを新規保存。B/Q/所属正解/採点結果は読み込まない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NUMERIC = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time")
CLINICAL = ("BMI", "SBP", "TG", "HDL", "ALT", "FPG")
QI7 = ("age", "sex", "prefecture", "SBP", "TG", "HDL", "ALT")
ALL13 = NUMERIC + ("sex", "prefecture", "smoking", "onset", "death")
ATTRIBUTE_SETS = [("clinical_pair", "+".join(cols), cols) for cols in combinations(CLINICAL, 2)] + [
    ("clinical6", "clinical6", CLINICAL), ("QI7", "QI7", QI7), ("all13", "all13", ALL13)]
GRIDS = {"integer": 1.0, "decimal1": .1, "decimal2": .01}
TOLERANCES = {"primary": 1e-8, "strict": 1e-10, "loose": 1e-6}
ENDPOINTS = {"near_9_998631": 9.998631074606434, "near_10": 10.0}
ENDPOINT_TOLERANCE = 1e-8
METHOD = {
    "version": 1, "inputs": "公開A3（予備戦・本戦・初版kit）と公開C42。自チームC除外。",
    "numeric_attributes": list(NUMERIC), "clinical_attributes": list(CLINICAL),
    "QI7": list(QI7), "all13": list(ALL13), "clinical_pair_count": 15,
    "read_precision": "pandas float_precision=round_trip",
    "exact_matching": "数値として完全一致。許容差で統合しない。非欠損・有限を入力時に確認。",
    "univariate": "値を共有するC行数、C値種類数、反復C値の共有、参照側の支持・反復を数える。",
    "tuples": "15clinical pairs/clinical6/QI7/all13の支持集合一致。行の多重度を保存して件数と種類数を分ける。",
    "source_union": "3Aをconcatした参照を追加。QI7/all13支持の相互重なりを確認し、unionは和集合で数える。",
    "time_endpoints": ENDPOINTS, "time_endpoint_absolute_tolerance": ENDPOINT_TOLERANCE,
    "time_nonendpoint_grid_steps": GRIDS, "time_nonendpoint_grid_tolerances": TOLERANCES,
    "time_grid_distance": "右端2種を除外した有限timeについてabs(x-rint(x/step)*step)を測る。",
    "time_endpoint_outcomes": "near_9_998631/near_10/nonendpointごとにonset/death組の集約件数。打切りの加工法を断定しない。",
    "age_support": "各公開A/Cのunique age列、min/max、隣接間隔・間隔の集約件数。個人行は保存しない。",
    "independent_check": "属性単位pandas.isinとPython Counter/set、tuple MultiIndex.isinとtupleCounterを全件比較。timeはスカラーroundでも照合。",
    "predeclared_rechecks": {
        "main3_all13": {"prelim_A": 92, "main_A": 40, "kit_A": 93, "union3_A": 225},
        "main3_QI7": {"prelim_A": 153, "main_A": 68, "kit_A": 160, "union3_A": 381},
        "age_prelim2_equals_main9": [40, 44, 47, 51, 54, 58, 61, 65, 68, 72, 74],
    },
    "guarantee_scope": "現行本戦Aの非会員保証のみ。historical予備戦A/kitA/3A unionへformal保証を拡張しない。予備戦Cの保証も本診断から新設しない。",
    "limits": [
        "属性値の共有、組の一致、行の一致は観測であり、コピー/シャッフル等の加工アルゴリズムを確定しない。",
        "FPGの125/100、time右端など公開背景自体に集中がある。反復だけで加工由来と断定しない。",
        "ageの疎支持は狭い年齢rangeかどうかと分けて記述し、加工法や同番号の継承は断定しない。",
        "所属ラベル、攻撃精度、private B、Qは使用しない。得点による追調整なし。",
    ],
    "b_used": False, "q_used": False, "membership_labels_used": False, "attack_scores_used": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def stamp():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def source_paths():
    pre = ROOT / "予備戦/配布データ"
    main_a = sorted((ROOT / "本戦/配布データ").glob("PWSCup2026_final_team28_*/A_bg.csv"))
    assert len(main_a) == 1
    backgrounds = {"prelim_A": pre / "PWSCup2026_prelim_common/A_bg.csv", "main_A": main_a[0],
                   "kit_A": ROOT / "スターターキット/初版/participant_data/A_bg.csv"}
    targets = []
    for phase, folder, own, expected in [("prelim", pre / "PWSCup2026_prelim_attack", 5, 20),
            ("main", HERE.parent / "20260928_配布データ基本分析/入力", 28, 22)]:
        paths = sorted(p for p in folder.glob("C_*.csv") if int(p.stem.split("_")[1]) != own)
        assert len(paths) == expected
        targets.extend((phase, int(p.stem.split("_")[1]), p) for p in paths)
    return backgrounds, targets


def frozen_hashes():
    files = [HERE / p for p in ("公開Cの特徴を整理.py", "公開分布の尾を確認.py", "公開値の刻みとカテゴリを確認.py")]
    for folder in ("公開加工特徴の比較結果", "公開分布の尾の診断結果", "公開値の刻みとカテゴリ結果"):
        files.extend(p for p in (HERE / folder).iterdir() if p.is_file())
    return {str(p.resolve()): sha(p) for p in sorted(files)}


def counters(frame, columns):
    return Counter(frame[list(columns)].itertuples(index=False, name=None))


def age_record(identity, frame):
    values = sorted(float(x) for x in frame.age.unique())
    gaps = [right - left for left, right in zip(values[:-1], values[1:])]
    return dict(identity, rows=len(frame), distinct_count=len(values), values=values,
                value_min=values[0], value_max=values[-1], value_range=values[-1] - values[0],
                adjacent_gaps=gaps, gap_count_records=[dict(gap=g, count=n) for g, n in sorted(Counter(gaps).items())])


def time_records(identity, frame):
    x = frame.time.to_numpy(float)
    label = np.full(len(x), "nonendpoint", dtype=object)
    for name, endpoint in ENDPOINTS.items():
        mask = np.abs(x - endpoint) <= ENDPOINT_TOLERANCE
        assert (label[mask] == "nonendpoint").all()
        label[mask] = name
    other = x[label == "nonendpoint"]
    record = dict(identity, rows=len(x), nonendpoint_rows=len(other),
                  nonendpoint_fraction=len(other) / len(x), time_min=float(x.min()), time_max=float(x.max()),
                  time_distinct_count=len(set(x.tolist())))
    for name in ENDPOINTS:
        record[name + "_rows"] = int((label == name).sum())
        record[name + "_fraction"] = float((label == name).mean())
    record["endpoint_union_rows"] = int((label != "nonendpoint").sum())
    record["endpoint_union_fraction"] = float((label != "nonendpoint").mean())
    checked = 0
    for name, step in GRIDS.items():
        distance = np.abs(other - np.rint(other / step) * step)
        for tolerance, threshold in TOLERANCES.items():
            count = int((distance <= threshold).sum())
            independent = sum(abs(float(value) / step - round(float(value) / step)) * step <= threshold for value in other)
            assert count == independent, (identity, name, tolerance)
            record[name + "_nonendpoint_near_rows_" + tolerance] = count
            record[name + "_nonendpoint_near_fraction_" + tolerance] = count / len(other) if len(other) else None
            checked += 1
    grouped = frame.assign(endpoint_group=label).groupby(["endpoint_group", "onset", "death"]).size()
    outcome_rows = [dict(identity, endpoint_group=g, onset=int(onset), death=int(death), count=int(count),
                          fraction_of_all_rows=count / len(frame)) for (g, onset, death), count in grouped.items()]
    assert sum(row["count"] for row in outcome_rows) == len(frame)
    return record, outcome_rows, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "公開背景との属性単位一致結果")
    args = parser.parse_args()
    out, code = args.output_dir.resolve(), Path(__file__).resolve()
    backgrounds, paths = source_paths()
    if args.write_plan:
        assert not out.exists()
        source_hashes = {str(p.resolve()): sha(p) for p in list(backgrounds.values()) + [p for _, _, p in paths]}
        out.mkdir()
        (out / "方法.json").write_text(dump(dict(created_jst=stamp(), method=METHOD,
            code_sha256_before_execution=sha(code), source_sha256_before_execution=source_hashes,
            frozen_artifact_sha256=frozen_hashes())), encoding="utf-8")
        print(dump(dict(stage="plan_and_sources_fixed_before_execution", path=str(out / "方法.json"))), end="")
        return
    assert Path(sys.executable).parent == ROOT / ".venv/bin"
    assert out.is_dir() and {p.name for p in out.iterdir()} == {"方法.json"}
    plan = json.loads((out / "方法.json").read_text())
    assert plan["method"] == METHOD and plan["code_sha256_before_execution"] == sha(code)
    assert plan["frozen_artifact_sha256"] == frozen_hashes()
    inputs = {}
    def read(path):
        assert path.name == "A_bg.csv" or path.name.startswith("C_")
        assert not any(word in str(path) for word in ("B_self", "pool_ids", "aia_challenge", "アーカイブ", "正解データ"))
        inputs[str(path.resolve())] = sha(path)
        assert inputs[str(path.resolve())] == plan["source_sha256_before_execution"][str(path.resolve())]
        frame = pd.read_csv(path, float_precision="round_trip")
        assert not frame[list(ALL13)].isna().any().any()
        assert np.isfinite(frame[list(NUMERIC)].to_numpy(float)).all()
        assert set(frame.onset.unique()) <= {0, 1} and set(frame.death.unique()) <= {0, 1}
        return frame
    refs = {name: read(path) for name, path in backgrounds.items()}
    disjoint_rows = []
    for left, right in combinations(refs, 2):
        for name, columns in [("QI7", QI7), ("all13", ALL13)]:
            lc, rc = counters(refs[left], columns), counters(refs[right], columns)
            observed = pd.MultiIndex.from_frame(refs[left][list(columns)]).isin(pd.MultiIndex.from_frame(refs[right][list(columns)]))
            shared = set(lc) & set(rc)
            count = sum(lc[key] for key in shared)
            assert int(observed.sum()) == count == 0
            disjoint_rows.append(dict(source1=left, source2=right, attribute_set=name,
                source1_rows=len(refs[left]), source2_rows=len(refs[right]), shared_unique_tuples=len(shared), source1_matching_rows=count))
    refs["union3_A"] = pd.concat(list(refs.values()), ignore_index=True)
    numeric_cache = {(name, attribute): Counter(frame[attribute].tolist()) for name, frame in refs.items() for attribute in NUMERIC}
    tuple_cache = {(name, label): counters(frame, cols) for name, frame in refs.items() for _, label, cols in ATTRIBUTE_SETS}
    index_cache = {(name, label): pd.MultiIndex.from_frame(frame[list(cols)]) for name, frame in refs.items() for _, label, cols in ATTRIBUTE_SETS}
    numeric_rows, tuple_rows, time_rows, outcome_rows, ages = [], [], [], [], []
    checked_numeric = checked_tuples = checked_time = 0
    for source, frame in refs.items():
        if source == "union3_A":
            continue
        identity = dict(phase="background", source=source, team=-1)
        ages.append(age_record(identity, frame))
        record, outcomes, count = time_records(identity, frame)
        time_rows.append(record); outcome_rows.extend(outcomes); checked_time += count
    for phase, team, path in paths:
        c = read(path)
        identity = dict(phase=phase, source="C", team=team)
        ages.append(age_record(identity, c))
        record, outcomes, count = time_records(identity, c)
        time_rows.append(record); outcome_rows.extend(outcomes); checked_time += count
        c_numeric = {attribute: Counter(c[attribute].tolist()) for attribute in NUMERIC}
        c_tuples = {label: counters(c, cols) for _, label, cols in ATTRIBUTE_SETS}
        c_index = {label: pd.MultiIndex.from_frame(c[list(cols)]) for _, label, cols in ATTRIBUTE_SETS}
        for source, ref in refs.items():
            common_identity = dict(phase=phase, team=team, reference_source=source, c_rows=len(c), reference_rows=len(ref))
            for attribute in NUMERIC:
                cc, ac = c_numeric[attribute], numeric_cache[source, attribute]
                shared, repeated = set(cc) & set(ac), {value for value, n in cc.items() if n > 1}
                shared_repeated = shared & repeated
                matched = c[attribute].isin(ref[attribute])
                observed = int(matched.sum()); independent = sum(cc[value] for value in shared)
                assert observed == independent
                assert int(c.loc[matched, attribute].nunique()) == len(shared)
                numeric_rows.append(dict(common_identity, attribute=attribute,
                    c_distinct_count=len(cc), reference_distinct_count=len(ac),
                    c_repeated_value_count=len(repeated), c_rows_on_repeated_values=sum(cc[v] for v in repeated),
                    reference_repeated_value_count=sum(n > 1 for n in ac.values()), reference_mode_count=max(ac.values()),
                    exact_shared_c_rows=observed, exact_shared_c_fraction=observed / len(c), exact_shared_c_unique_values=len(shared),
                    shared_repeated_c_values=len(shared_repeated), c_rows_on_shared_repeated_values=sum(cc[v] for v in shared_repeated),
                    reference_rows_on_c_values=sum(ac[v] for v in shared)))
                checked_numeric += 1
            for kind, label, columns in ATTRIBUTE_SETS:
                cc, ac = c_tuples[label], tuple_cache[source, label]
                shared = set(cc) & set(ac)
                observed = c_index[label].isin(index_cache[source, label])
                independent = sum(cc[key] for key in shared)
                assert int(observed.sum()) == independent
                observed_unique = len(c.loc[observed, list(columns)].drop_duplicates())
                assert observed_unique == len(shared)
                tuple_rows.append(dict(common_identity, attribute_set_kind=kind, attribute_set=label,
                    c_unique_tuples=len(cc), reference_unique_tuples=len(ac), c_duplicate_excess=len(c) - len(cc),
                    exact_matching_c_rows=independent, exact_matching_c_fraction=independent / len(c),
                    exact_matching_unique_c_tuples=len(shared), exact_matching_duplicate_excess=independent - len(shared),
                    reference_rows_on_c_tuples=sum(ac[key] for key in shared)))
                checked_tuples += 1
    numeric = pd.DataFrame(numeric_rows); tuples = pd.DataFrame(tuple_rows)
    assert len(numeric) == 42 * 4 * 8 and len(tuples) == 42 * 4 * 18
    for kind, expected_by_source in METHOD["predeclared_rechecks"].items():
        if kind == "age_prelim2_equals_main9":
            selected = [row for row in ages if (row["phase"], row["team"]) in [("prelim", 2), ("main", 9)]]
            assert len(selected) == 2 and all(row["values"] == expected_by_source for row in selected)
            continue
        metric = "all13" if kind == "main3_all13" else "QI7"
        main3 = tuples[(tuples.phase == "main") & (tuples.team == 3) & (tuples.attribute_set == metric)].set_index("reference_source")
        assert main3.exact_matching_c_rows.to_dict() == expected_by_source
    for metric in ("QI7", "all13"):
        subset = tuples[tuples.attribute_set == metric]
        for _, block in subset.groupby(["phase", "team"]):
            values = block.set_index("reference_source").exact_matching_c_rows
            assert values["union3_A"] == values.drop("union3_A").sum()
    numeric.sort_values(["phase", "team", "reference_source", "attribute"]).to_csv(out / "属性単位の公開値共有.csv", index=False, float_format="%.17g")
    tuples.sort_values(["phase", "team", "reference_source", "attribute_set_kind", "attribute_set"]).to_csv(out / "属性組と行全体の公開値一致.csv", index=False, float_format="%.17g")
    pd.DataFrame(disjoint_rows).to_csv(out / "3背景の行支持重なり.csv", index=False)
    pd.DataFrame(time_rows).sort_values(["phase", "source", "team"]).to_csv(out / "time右端と非右端の刻み.csv", index=False, float_format="%.17g")
    pd.DataFrame(outcome_rows).sort_values(["phase", "source", "team", "endpoint_group", "onset", "death"]).to_csv(out / "time区分別のonset_death件数.csv", index=False, float_format="%.17g")
    (out / "ageの値支持.json").write_text(dump(dict(records=ages)), encoding="utf-8")
    (out / "独立検証.json").write_text(dump(dict(numeric_isin_counter_checks=checked_numeric,
        tuple_isin_counter_checks=checked_tuples, time_scalar_grid_checks=checked_time,
        background_QI7_all13_disjoint_checks=len(disjoint_rows), main3_and_age_rechecks_passed=True,
        union_matching_sum_checks_passed=True)), encoding="utf-8")
    assert plan["source_sha256_before_execution"] == inputs
    for path, expected in inputs.items():
        assert sha(path) == expected
    assert plan["frozen_artifact_sha256"] == frozen_hashes()
    assert plan["code_sha256_before_execution"] == sha(code)
    output_hashes = {p.name: sha(p) for p in sorted(out.iterdir())}
    record = dict(completed_jst=stamp(), python=sys.executable, python_version=platform.python_version(),
        code_sha256=sha(code), source_sha256=inputs, output_sha256=output_hashes,
        output_hash_scope="実行記録.json自身は除外。", plan_fixed_before_execution=True,
        frozen_artifacts_unchanged=True, sources=3, targets=42, numeric_share_rows=len(numeric), tuple_share_rows=len(tuples),
        b_used=False, q_used=False, membership_labels_used=False, attack_scores_used=False,
        guarantee_scope=METHOD["guarantee_scope"], processing_algorithm_confirmed=False)
    (out / "実行記録.json").write_text(dump(record), encoding="utf-8")
    print(dump(dict(stage="public_attribute_sharing_complete", sources=3, targets=42,
                    numeric_share_rows=len(numeric), tuple_share_rows=len(tuples))), end="")


if __name__ == "__main__":
    main()
