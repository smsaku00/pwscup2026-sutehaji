"""公開済みの集計だけを用い、2026-09-25の保存結果を検算する。

個票・非公開入力・課題生成・攻撃実行にはアクセスしない。
通常は標準出力のみ。--output の指定時だけ検証JSONを保存する。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from scipy import __version__ as scipy_version
from scipy import stats

HERE = Path(__file__).resolve().parent
EPS = 1e-12
TEXT_COLUMNS = {
    "case", "name", "panel", "attack", "attack_name", "metric", "worst_attacks"
}
METRICS = ["raw", "adjusted", "minus_constant", "minus_background", "minus_phi"]


def read_csv(name):
    with (HERE / "集計" / (name + ".csv")).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert rows, name
        assert all(set(row) == set(reader.fieldnames) for row in rows), name
        assert all(None not in row.values() for row in rows), name
        return [
            {key: value if key in TEXT_COLUMNS else float(value) for key, value in row.items()}
            for row in rows
        ]


def close(actual, expected, label):
    assert np.isfinite(actual) and np.isfinite(expected), label
    assert abs(actual - expected) < EPS, (label, actual, expected)


def compare_table(name, generated, keys):
    saved = read_csv(name)
    assert len(saved) == len(generated), name
    index = {tuple(row[key] for key in keys): row for row in saved}
    assert len(index) == len(saved), name
    for row in generated:
        key = tuple(row[column] for column in keys)
        assert key in index, (name, key)
        old = index[key]
        assert set(old) == set(row), (name, key)
        for column, value in row.items():
            label = (name, key, column)
            if column in TEXT_COLUMNS:
                assert value == old[column], label
            else:
                close(value, old[column], label)
    return len(saved)


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": len(values), "mean": float(values.mean()),
        "median": float(np.median(values)), "minimum": float(values.min()),
        "maximum": float(values.max()), "p90": float(np.quantile(values, .9)),
        "p95": float(np.quantile(values, .95)),
        "zero_count": int((np.abs(values) < EPS).sum()),
    }


def verify():
    conditions = json.loads((HERE / "実行条件.json").read_text(encoding="utf-8"))
    assert conditions["panel_count"] == 300
    assert conditions["targets_per_panel"] == conditions["controls_per_panel"] == 163
    assert not conditions["old_results_in_primary"]
    count = conditions["panel_count"]
    cases = {case["id"]: case["name"] for case in conditions["cases"]}
    attacks = {attack["id"]: attack["name"] for attack in conditions["attacks"]}
    assert cases == {"C27": "Zhang04-HCv2-3", "C26": "Zhang11-HCv2-1"}
    assert len(attacks) == 12
    scores = read_csv("全7200採点")
    baselines = read_csv("各パネルの基準値")
    assert len(scores) == 7200 and len(baselines) == count
    panels = sorted(row["panel"] for row in baselines)
    assert panels == [f"F163_{i:03d}" for i in range(1, count + 1)]
    score_index = {(row["case"], row["panel"], row["attack"]): row for row in scores}
    assert len(score_index) == len(scores)
    assert set(score_index) == {
        (case, panel, attack) for case in cases for panel in panels for attack in attacks
    }
    for row in scores:
        for prefix in ("target", "control"):
            hits = row[prefix + "_hits"]
            assert 0 <= hits <= 163 and hits == int(hits)
            close(row[prefix + "_rate"], hits / 163, (row["panel"], prefix))
        calculated = np.clip(
            (row["target_rate"] - row["control_rate"])
            / max(1 - row["control_rate"], 1e-9), 0, 1
        )
        close(row["AIA"], calculated, (row["case"], row["panel"], row["attack"]))
    baseline_index = {row["panel"]: row for row in baselines}
    for row in baselines:
        assert 0 <= row["constant10"] <= 1 and 0 <= row["background1NN"] <= 1
        close(row["baseline_max"], max(row["constant10"], row["background1NN"]), row["panel"])
    panel_results = []
    for panel in panels:
        baseline = baseline_index[panel]
        for case, name in cases.items():
            attack_scores = [score_index[case, panel, attack] for attack in attacks]
            raw = max(row["AIA"] for row in attack_scores)
            panel_results.append({
                "case": case, "name": name, "panel": panel, "raw": raw,
                "constant10": baseline["constant10"],
                "background1NN": baseline["background1NN"],
                "baseline_max": baseline["baseline_max"],
                "adjusted": max(0, raw - baseline["baseline_max"]),
                "minus_constant": max(0, raw - baseline["constant10"]),
                "minus_background": max(0, raw - baseline["background1NN"]),
                "minus_phi": max(0, raw - conditions["phi_reference"]),
                "worst_attacks": " / ".join(
                    row["attack"] for row in attack_scores if abs(row["AIA"] - raw) < EPS
                ),
            })
    checked = {"全7200採点": len(scores), "各パネルの基準値": len(baselines)}
    checked["パネル別比較"] = compare_table("パネル別比較", panel_results, ["case", "panel"])
    panel_index = {(row["case"], row["panel"]): row for row in panel_results}
    summaries = []
    tails = []
    for case, name in cases.items():
        for metric in METRICS:
            summaries.append({"case": case, "name": name, "metric": metric, **summarize(
                [panel_index[case, panel][metric] for panel in panels]
            )})
        for threshold in conditions["exploratory_tail_thresholds"]:
            high = sum(panel_index[case, panel]["adjusted"] > threshold for panel in panels)
            ci = stats.binomtest(high, count).proportion_ci(.95, method="exact")
            tails.append({
                "case": case, "name": name, "threshold": threshold,
                "count": high, "total": count, "rate": high / count,
                "ci95_low": float(ci.low), "ci95_high": float(ci.high),
            })
    checked["候補別要約"] = compare_table("候補別要約", summaries, ["case", "metric"])
    checked["高スコアの発生率"] = compare_table("高スコアの発生率", tails, ["case", "threshold"])
    paired = [{
        "panel": panel,
        "Zhang04_adjusted": panel_index["C27", panel]["adjusted"],
        "Zhang11_adjusted": panel_index["C26", panel]["adjusted"],
        "difference": panel_index["C26", panel]["adjusted"] - panel_index["C27", panel]["adjusted"],
        "raw_difference": panel_index["C26", panel]["raw"] - panel_index["C27", panel]["raw"],
    } for panel in panels]
    checked["2案の対応差"] = compare_table("2案の対応差", paired, ["panel"])
    method_results = [{
        "case": case, "name": name, "attack": attack, "attack_name": attack_name,
        **summarize([score_index[case, panel, attack]["AIA"] for panel in panels]),
    } for case, name in cases.items() for attack, attack_name in attacks.items()]
    checked["手法別要約"] = compare_table("手法別要約", method_results, ["case", "attack"])
    tests = {}
    for metric in METRICS:
        differences = np.array([
            panel_index["C26", panel][metric] - panel_index["C27", panel][metric]
            for panel in panels
        ])
        mean = float(differences.mean())
        standard_error = float(stats.sem(differences))
        half_width = float(stats.t.ppf(.975, count - 1) * standard_error)
        rng = np.random.default_rng(conditions["bootstrap_seed"])
        bootstrap = differences[rng.integers(
            0, count, size=(conditions["bootstrap_replicates"], count)
        )].mean(axis=1)
        tests[metric] = {
            "mean_difference_11_minus_04": mean,
            "sd": float(differences.std(ddof=1)), "standard_error": standard_error,
            "ci95_low": mean - half_width, "ci95_high": mean + half_width,
            "p_two_sided": float(stats.ttest_1samp(differences, 0).pvalue),
            "bootstrap95_low": float(np.quantile(bootstrap, .025)),
            "bootstrap95_high": float(np.quantile(bootstrap, .975)),
            "Zhang11_better": int((differences < -EPS).sum()),
            "ties": int((np.abs(differences) <= EPS).sum()),
            "Zhang04_better": int((differences > EPS).sum()),
            "primary": metric == "adjusted",
        }
        reported = conditions["reported_paired_tests"][metric]
        assert set(tests[metric]) == set(reported)
        for key, value in tests[metric].items():
            close(value, reported[key], (metric, key))
    files = sorted((HERE / "集計").glob("*.csv")) + [HERE / "実行条件.json", HERE / "分析.py"]
    return {
        "passed": True, "validation_date": "2026-10-13",
        "scope": "Published aggregate consistency and conditional statistics only",
        "private_inputs_used": False, "individual_predictions_recomputed": False,
        "baseline_predictions_recomputed": False, "original_experiment_rerun": False,
        "panels": count, "cases": len(cases), "attacks": len(attacks),
        "aggregate_AIA_scores_recomputed_from_hit_counts": len(scores),
        "baseline_rows_checked": len(baselines), "table_rows_checked": checked,
        "paired_tests": tests,
        "official_anonymity_score": False,
        "tolerance_absolute": EPS,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy_version},
        "published_files_sha256": {
            str(file.relative_to(HERE)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in files
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="検証JSONの保存先。省略時は標準出力だけ。")
    arguments = parser.parse_args()
    result = verify()
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if arguments.output:
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
