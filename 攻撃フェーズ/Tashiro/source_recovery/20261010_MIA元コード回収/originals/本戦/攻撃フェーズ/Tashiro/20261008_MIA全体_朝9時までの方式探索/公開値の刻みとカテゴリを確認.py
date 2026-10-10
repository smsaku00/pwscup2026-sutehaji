"""公開A/Cだけで値の刻み、集中、カテゴリ/time支持を共通規則で診断する。

先に .venv/bin/python -B このファイル --write-plan を実行し、次に引数なしで計算。
既存の公開C/尾のコード・結果は凍結し、別の結果フォルダに技術CSV/JSONを保存する。
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
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NUMERIC = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG", "time")
CATEGORIES = ("sex", "prefecture", "smoking", "onset", "death")
GROUPS = ("all", "sex0", "sex1")
GRIDS = {"integer": 1.0, "decimal1": .1, "decimal2": .01}
TOLERANCES = {"primary": 1e-8, "strict": 1e-10, "loose": 1e-6}
FRACTION_CHANGE_TOLERANCE = .05
METHOD = {
    "version": 1,
    "population": "予備戦20対象、本戦22対象。自チームCは除外。参照Aを含め同一規則。",
    "numeric_attributes": list(NUMERIC), "categories": list(CATEGORIES), "groups": list(GROUPS),
    "numeric_input": "CSVはfloat_precision=round_tripで読み、数値のまま比較。表示桁数・文字列長は使わない。",
    "grid_steps_raw_units": GRIDS, "absolute_distance_tolerances_raw_units": TOLERANCES,
    "grid_distance": "abs(x-rint(x/step)*step)。許容差以内の有限値の比率を測る。",
    "primary_tolerance": "1e-8。1e-10/1e-6は事前固定の感度確認。属性・対象により変更しない。",
    "distinct_ratio": "数値として完全に等しい有限値の種類数/有限行数。許容差による値の統合は行わない。",
    "mode_fraction": "数値として完全に等しい最頻値の件数/有限行数。最頻値が複数なら最小値を表示。",
    "numeric_ranges": "有限値のmin/max/range/q25/q50/q75/IQR。欠損・非数値・無限値は別に数える。",
    "category_support": "非欠損の値の完全一致による支持集合、種類数、最頻値集中、A支持にない件数。",
    "time_support": "timeは数値診断と支持診断の両方に含める。丸めてから支持を数えない。",
    "category_values_output": "公開データのカテゴリ値ごとの集約件数のみ。timeの全値一覧は書き出さない。",
    "phase_comparison": "同番号を機械的に比較。fractionの差<=0.05、distinct_countの差=0を記述的近さとして表示。",
    "descriptor_tolerance": {"fraction_absolute_change": FRACTION_CHANGE_TOLERANCE, "distinct_count_absolute_change": 0},
    "existing_public_reference": "公開加工特徴の比較結果/同番号のフェーズ間変化.csvを同番号の比較表に併記。診断入力ではない。",
    "independent_check": "全数値群をPythonのCounter、組み込みroundとスカラー演算で再計算。カテゴリ支持もCounter/setで再計算。",
    "limits": [
        "数値距離への一致は加工アルゴリズム、元の測定精度、会員状態を証明しない。",
        "ageや二値カテゴリの整数性、timeの上限集中など元データ由来の特徴がある。",
        "コホート構成と加工効果の因果的な分離ではない。",
        "同番号が同じチーム・加工アルゴリズムであることは仮定しない。",
        "within_descriptor_toleranceは事前固定した記述値の近さであり、統計的同等性や攻撃精度の保証ではない。",
    ],
    "b_used": False, "q_used": False, "membership_labels_used": False, "attack_scores_used": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def stamp():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def sources():
    pre = ROOT / "予備戦/配布データ"
    bg = sorted((ROOT / "本戦/配布データ").glob("PWSCup2026_final_team28_*/A_bg.csv"))
    assert len(bg) == 1
    return [("prelim", pre / "PWSCup2026_prelim_common/A_bg.csv", pre / "PWSCup2026_prelim_attack", 5, 20),
            ("main", bg[0], HERE.parent / "20260928_配布データ基本分析/入力", 28, 22)]


def frozen_hashes():
    paths = [HERE / "公開Cの特徴を整理.py", HERE / "公開分布の尾を確認.py"]
    for name in ("公開加工特徴の比較結果", "公開分布の尾の診断結果"):
        paths.extend(p for p in (HERE / name).iterdir() if p.is_file())
    return {str(p.resolve()): sha(p) for p in sorted(paths)}


def group(frame, name):
    return frame if name == "all" else frame[frame.sex == int(name[-1])]


def as_number(series):
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return numeric[np.isfinite(numeric)], {
        "rows": len(series), "missing_rows": int(series.isna().sum()),
        "nonnumeric_rows": int((series.notna() & pd.to_numeric(series, errors="coerce").isna()).sum()),
        "infinite_rows": int(np.isinf(numeric).sum()), "finite_rows": int(np.isfinite(numeric).sum())}


def number_stats(series):
    x, record = as_number(series)
    assert len(x) > 0
    values, counts = np.unique(x, return_counts=True)
    q = np.quantile(x, [.25, .5, .75], method="linear")
    record.update(distinct_count=len(values), distinct_fraction=len(values) / len(x),
                  mode_count=int(counts.max()), mode_fraction=float(counts.max() / len(x)),
                  mode_value=float(values[np.flatnonzero(counts == counts.max())[0]]),
                  value_min=float(x.min()), value_max=float(x.max()), value_range=float(x.max() - x.min()),
                  q25=float(q[0]), median=float(q[1]), q75=float(q[2]), iqr=float(q[2] - q[0]))
    for label, step in GRIDS.items():
        distance = np.abs(x - np.rint(x / step) * step)
        record.update({label + "_distance_median": float(np.median(distance)),
                       label + "_distance_q95": float(np.quantile(distance, .95, method="linear")),
                       label + "_distance_max": float(distance.max())})
        for tolerance, threshold in TOLERANCES.items():
            record[label + "_near_count_" + tolerance] = int((distance <= threshold).sum())
            record[label + "_near_fraction_" + tolerance] = float((distance <= threshold).mean())
    return record


def support_stats(series, reference):
    s = series.dropna()
    counts = s.value_counts(dropna=False)
    support = set(s.tolist())
    a_support = set(reference.dropna().tolist())
    values = sorted(support, key=lambda value: (str(type(value)), str(value)))
    top = [v for v in values if counts.loc[v] == counts.max()]
    novel_rows = int((~s.isin(a_support)).sum())
    numeric = pd.to_numeric(s, errors="coerce").to_numpy(float)
    finite = numeric[np.isfinite(numeric)]
    return dict(rows=len(series), missing_rows=int(series.isna().sum()), nonmissing_rows=len(s),
                distinct_count=len(support), distinct_fraction=len(support) / len(s),
                mode_count=int(counts.max()), mode_fraction=float(counts.max() / len(s)),
                mode_value_json=json.dumps(top[0].item() if hasattr(top[0], "item") else top[0], ensure_ascii=False),
                a_distinct_count=len(a_support), support_intersection_count=len(support & a_support),
                support_outside_a_count=len(support - a_support), rows_outside_a_support=novel_rows,
                fraction_outside_a_support=novel_rows / len(s),
                numeric_value_min=float(finite.min()) if len(finite) else None,
                numeric_value_max=float(finite.max()) if len(finite) else None,
                numeric_value_range=float(finite.max() - finite.min()) if len(finite) else None)


def independent_check(frame, name, numeric, supports, reference):
    """Numpyによる集計を再使用せず、Counterと各値のスカラー演算で照合する。"""
    maxima = []
    checked = 0
    for attribute in NUMERIC:
        series = frame[attribute]
        data = [float(v) for v in series if pd.notna(v) and math.isfinite(float(v))]
        counts = Counter(data)
        observed = numeric[attribute]
        assert observed["finite_rows"] == len(data)
        assert observed["distinct_count"] == len(counts)
        assert observed["mode_count"] == max(counts.values())
        assert observed["value_min"] == min(data) and observed["value_max"] == max(data)
        for label, step in GRIDS.items():
            scale = 1 / step
            distances = [abs(v * scale - round(v * scale)) / scale for v in data]
            for tolerance, threshold in TOLERANCES.items():
                expected = sum(d <= threshold for d in distances)
                assert observed[label + "_near_count_" + tolerance] == expected, (name, attribute, label, tolerance)
                maxima.append(abs(observed[label + "_near_fraction_" + tolerance] - expected / len(data)))
                checked += 1
        assert observed["distinct_fraction"] == len(counts) / len(data)
        assert observed["mode_fraction"] == max(counts.values()) / len(data)
    for attribute in CATEGORIES + ("time",):
        values = [v for v in frame[attribute] if pd.notna(v)]
        counts = Counter(values)
        a_support = set(v for v in reference[attribute] if pd.notna(v))
        observed = supports[attribute]
        assert observed["distinct_count"] == len(counts)
        assert observed["mode_count"] == max(counts.values())
        assert observed["missing_rows"] == len(frame) - len(values)
        assert observed["support_intersection_count"] == len(set(counts) & a_support)
        assert observed["support_outside_a_count"] == len(set(counts) - a_support)
        assert observed["rows_outside_a_support"] == sum(v not in a_support for v in values)
    return checked, max(maxima)


def phase_changes(numeric, supports, previous):
    result = []
    metrics = {"numeric": [label + "_near_fraction_primary" for label in GRIDS] +
               ["distinct_fraction", "mode_fraction", "distinct_count", "value_min", "value_max", "value_range"],
               "support": ["distinct_count", "mode_fraction", "missing_rows", "support_outside_a_count", "fraction_outside_a_support"]}
    for kind, table in [("numeric", numeric), ("support", supports)]:
        c = table[table.source == "C"]
        left = c[c.phase == "prelim"].set_index(["team", "group", "attribute"])
        right = c[c.phase == "main"].set_index(["team", "group", "attribute"])
        for key in sorted(set(left.index) & set(right.index)):
            team, g, attribute = key
            for metric in metrics[kind]:
                pre, main = float(left.loc[key, metric]), float(right.loc[key, metric])
                tolerance = FRACTION_CHANGE_TOLERANCE if "fraction" in metric else (0 if metric == "distinct_count" else None)
                result.append(dict(team=team, group=g, attribute=attribute, feature=metric, feature_source=kind,
                                   prelim=pre, main=main, change=main - pre, descriptor_tolerance=tolerance,
                                   within_descriptor_tolerance=abs(main - pre) <= tolerance if tolerance is not None else None,
                                   same_processing_algorithm_unconfirmed=True))
    for row in previous.to_dict("records"):
        result.append(dict(team=int(row["team"]), group="existing", attribute="existing", feature=row["feature"],
                           feature_source="existing_public_profile", prelim=float(row["prelim"]), main=float(row["main"]),
                           change=float(row["change"]), descriptor_tolerance=None, within_descriptor_tolerance=None,
                           same_processing_algorithm_unconfirmed=True))
    return pd.DataFrame(result).sort_values(["team", "feature_source", "group", "attribute", "feature"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "公開値の刻みとカテゴリ結果")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    code = Path(__file__).resolve()
    reference = HERE / "公開加工特徴の比較結果/同番号のフェーズ間変化.csv"
    if args.write_plan:
        assert not out.exists()
        out.mkdir()
        (out / "方法.json").write_text(dump(dict(created_jst=stamp(), method=METHOD,
            code_sha256_before_execution=sha(code), frozen_artifact_sha256=frozen_hashes(),
            public_reference_sha256={str(reference): sha(reference)})), encoding="utf-8")
        print(dump(dict(stage="plan_fixed_before_execution", path=str(out / "方法.json"))), end="")
        return
    assert Path(sys.executable).parent == ROOT / ".venv/bin"
    assert out.is_dir() and {p.name for p in out.iterdir()} == {"方法.json"}
    plan = json.loads((out / "方法.json").read_text())
    assert plan["method"] == METHOD and plan["code_sha256_before_execution"] == sha(code)
    assert plan["frozen_artifact_sha256"] == frozen_hashes()
    assert plan["public_reference_sha256"][str(reference)] == sha(reference)
    numeric_rows, support_rows, level_rows, inputs, check_rows = [], [], [], {}, []
    for phase, a_path, target_dir, self_id, expected_count in sources():
        paths = sorted(p for p in target_dir.glob("C_*.csv") if int(p.stem.split("_")[1]) != self_id)
        assert len(paths) == expected_count
        a = pd.read_csv(a_path, float_precision="round_trip")
        for path in [a_path] + paths:
            assert path.name == "A_bg.csv" or path.name.startswith("C_")
            assert not any(w in str(path) for w in ("B_self", "pool_ids", "aia_challenge", "アーカイブ", "正解データ"))
            inputs[str(path.resolve())] = sha(path)
            frame = a if path == a_path else pd.read_csv(path, float_precision="round_trip")
            source = "A" if path == a_path else "C"
            team = -1 if source == "A" else int(path.stem.split("_")[1])
            for g in GROUPS:
                data, ref = group(frame, g), group(a, g)
                assert len(data) and len(ref)
                numbers = {attribute: number_stats(data[attribute]) for attribute in NUMERIC}
                supports = {attribute: support_stats(data[attribute], ref[attribute]) for attribute in CATEGORIES + ("time",)}
                identity = dict(phase=phase, source=source, team=team, group=g)
                numeric_rows.extend(dict(identity, attribute=attribute, **values) for attribute, values in numbers.items())
                support_rows.extend(dict(identity, attribute=attribute, **values) for attribute, values in supports.items())
                for attribute in CATEGORIES:
                    counts = data[attribute].dropna().value_counts()
                    for value, count in counts.items():
                        value = value.item() if hasattr(value, "item") else value
                        level_rows.append(dict(identity, attribute=attribute, value_json=json.dumps(value, ensure_ascii=False),
                                               count=int(count), fraction=float(count / counts.sum())))
                checked, error = independent_check(data, (phase, source, team, g), numbers, supports, ref)
                check_rows.append(dict(identity, numeric_grid_count_checks=checked, max_fraction_error=error,
                                       counter_and_category_support_verified=True))
    numeric = pd.DataFrame(numeric_rows).sort_values(["phase", "source", "team", "group", "attribute"])
    supports = pd.DataFrame(support_rows).sort_values(["phase", "source", "team", "group", "attribute"])
    assert len(numeric) == 44 * 3 * 8 and len(supports) == 44 * 3 * 6
    previous = pd.read_csv(reference, float_precision="round_trip")
    changes = phase_changes(numeric, supports, previous)
    numeric.to_csv(out / "数値の刻みと集中.csv", index=False, float_format="%.17g")
    supports.to_csv(out / "カテゴリとtimeの支持.csv", index=False, float_format="%.17g")
    pd.DataFrame(level_rows).sort_values(["phase", "source", "team", "group", "attribute", "value_json"]).to_csv(
        out / "カテゴリ値の件数.csv", index=False, float_format="%.17g")
    changes.to_csv(out / "同番号の特徴変化.csv", index=False, float_format="%.17g")
    (out / "独立検証.json").write_text(dump(dict(check_method=METHOD["independent_check"],
        checks=check_rows, numeric_groups_verified=len(numeric), category_time_groups_verified=len(supports),
        total_grid_count_checks=sum(x["numeric_grid_count_checks"] for x in check_rows),
        maximum_fraction_error=max(x["max_fraction_error"] for x in check_rows))), encoding="utf-8")
    for path, expected in inputs.items():
        assert sha(path) == expected
    assert plan["frozen_artifact_sha256"] == frozen_hashes()
    assert plan["code_sha256_before_execution"] == sha(code)
    output_hashes = {p.name: sha(p) for p in sorted(out.iterdir())}
    record = dict(completed_jst=stamp(), python=sys.executable, python_version=platform.python_version(),
        code_sha256=sha(code), source_sha256=inputs, public_reference_sha256=plan["public_reference_sha256"],
        output_sha256=output_hashes, output_hash_scope="実行記録.json自身を除く。",
        method_fixed_before_execution=True, frozen_artifacts_unchanged=True, targets=42, backgrounds=2,
        numeric_descriptor_rows=len(numeric), category_time_rows=len(supports), category_level_rows=len(level_rows),
        phase_comparison_rows=len(changes), b_used=False, q_used=False, membership_labels_used=False, attack_scores_used=False)
    (out / "実行記録.json").write_text(dump(record), encoding="utf-8")
    print(dump(dict(stage="public_grid_and_category_complete", targets=42, backgrounds=2,
                    numeric_rows=len(numeric), support_rows=len(supports), phase_comparison_rows=len(changes))), end="")


if __name__ == "__main__":
    main()
