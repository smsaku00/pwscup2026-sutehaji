"""標準ライブラリだけで集計・実行コード・掲載ファイルを検証する。

入力を持つローカル環境では --check-inputs を付ける。CSVを再計算・上書きせず、
対象別値から平均と改善数を照合する。提出回答の作成やネットワークアクセスはない。
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PAIRS = [
    ("結果/予備戦_対象別比較.csv", "結果/予備戦_集約比較.csv", ("method",), "mia_all", "global7", "auc"),
    ("追加結果_最終版/予備戦_追加対象別比較.csv", "追加結果_最終版/予備戦_追加集約比較.csv", ("method",), "mia_all", "global7", "auc"),
    ("補助結果/対照一致_対象別比較.csv", "補助結果/対照一致_集約比較.csv", ("subset", "method"), "proxy_tpr_1pct", "global7_k10", "proxy_auc"),
    ("参照結果_拡張版/参照_対象別比較.csv", "参照結果_拡張版/参照_集約比較.csv", ("method",), "mia_all", "global7", "auc"),
    ("背景再標本化結果/背景再標本化_対象別比較.csv", "背景再標本化結果/背景再標本化_集約比較.csv", ("method",), "mia_all", "global7", "auc"),
    ("学習結果_順位校正版/学習_対象別比較.csv", "学習結果_順位校正版/学習_集約比較.csv", ("method",), "mia_all", "global7", "auc"),
    ("属性部分集合結果/部分集合_対象別比較.csv", "属性部分集合結果/部分集合_集約比較.csv", ("method",), "mia_all", "global7", "auc"),
    ("低FPR結果/低FPR_対象別比較.csv", "低FPR結果/低FPR_集約比較.csv", ("method", "fpr"), "mia_all", "baseline", None),
    ("固定候補の感度結果/感度検証_対象別比較.csv", "固定候補の感度結果/感度検証_集約比較.csv", ("method",), "mia_all", "global7", "auc"),
    ("非会員較正結果/非会員較正_対象別比較.csv", "非会員較正結果/非会員較正_集約比較.csv", ("method",), "mia_all", "global7", "auc"),
    ("非会員較正の感度結果/較正感度_対象別比較.csv", "非会員較正の感度結果/較正感度_集約比較.csv", ("method",), "mia_all", "global7", "auc"),
    ("非会員較正の感度結果/較正低FPR_対象別比較.csv", "非会員較正の感度結果/較正低FPR_集約比較.csv", ("method", "fpr"), "mia_all", "full__baseline", None),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def key(row, columns):
    return tuple(float(row[c]) if c == "fpr" else row[c] for c in columns)


def close(actual, expected, label):
    assert math.isfinite(float(actual)) and math.isfinite(float(expected)), label
    assert abs(float(actual) - float(expected)) < 1e-12, (label, actual, expected)


def pair_check(d, detail_path, aggregate_path, columns, metric, baseline_method, auc):
    detail, aggregate = read_csv(d / detail_path), read_csv(d / aggregate_path)
    groups = defaultdict(list)
    for row in detail:
        value = float(row[metric])
        assert math.isfinite(value) and 0 <= value <= 1, (detail_path, row)
        groups[key(row, columns)].append(row)
    assert len(aggregate) == len(groups), aggregate_path
    for summary in aggregate:
        group = groups[key(summary, columns)]
        teams = {row["team"] for row in group}
        assert len(group) == len(teams) == 20, (aggregate_path, summary)
        close(summary[metric], statistics.mean(float(row[metric]) for row in group), aggregate_path)
        if summary.get("targets"):
            assert int(summary["targets"]) == 20
        if summary.get("other18_mean"):
            others = [float(row[metric]) for row in group if int(row["team"]) not in (1, 2)]
            assert len(others) == 18
            close(summary["other18_mean"], statistics.mean(others), aggregate_path)
        if auc is not None:
            close(summary[auc], statistics.mean(float(row[auc]) for row in group), aggregate_path)
        baseline = dict(summary)
        baseline["method"] = baseline_method
        base = {row["team"]: float(row[metric]) for row in groups[key(baseline, columns)]}
        delta = [float(row[metric]) - base[row["team"]] for row in group]
        if summary.get("wins"):
            assert int(summary["wins"]) == sum(value > 1e-12 for value in delta)
            assert int(summary["losses"]) == sum(value < -1e-12 for value in delta)
        if summary.get("paired_mean_delta"):
            close(summary["paired_mean_delta"], statistics.mean(delta), aggregate_path)
        if summary.get("same_reference_delta"):
            reference = summary["method"].split("__")[0]
            other = dict(summary)
            other["method"] = reference + "__baseline"
            base = {row["team"]: float(row[metric]) for row in groups[key(other, columns)]}
            delta = [float(row[metric]) - base[row["team"]] for row in group]
            close(summary["same_reference_delta"], statistics.mean(delta), aggregate_path)
            assert int(summary["same_reference_wins"]) == sum(value > 1e-12 for value in delta)
            assert int(summary["same_reference_losses"]) == sum(value < -1e-12 for value in delta)
    return {"detail": detail_path, "aggregate": aggregate_path, "target_rows": len(detail), "groups": len(groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--input-root", type=Path, help="別の場所に公式入力を置いた場合のPWS作業ルート")
    parser.add_argument("--output-file", type=Path, help="省略時は標準出力のみ。既存ファイルは上書きしない")
    args = parser.parse_args()
    if args.output_file is not None and args.output_file.exists():
        raise FileExistsError("既存の検証記録を上書きしません。")
    d = args.analysis_dir
    record = json.loads((d / "共有集計_確認版/実行記録.json").read_text())
    source_checks = []
    for run in record["runs"]:
        assert sha(d / run["code"]) == run["code_sha256"], run["code"]
        if (d / run["raw_result"]).exists():
            assert sha(d / run["raw_result"]) == run["raw_result_sha256"], run["raw_result"]
        source_checks.append(run["code"])
    assert sha(d / "分析.py") == record["common_code_sha256"]
    assert sha(d / "記録と図の作成.py") == record["summary_code_sha256"]
    for path in d.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    assert record["main_counts"] == {"B": 1049, "A": 5143, "targets": 22, "private": 163, "rare": 24, "possible_nonrare": 875}
    assert record["baseline_saved_tpr_max_error"] == 0
    official_errors = [run["official_tpr_max_error"] for run in record["runs"] if "official_tpr_max_error" in run]
    assert len(official_errors) == 2 and max(official_errors) == 0
    pairs = [pair_check(d, *pair) for pair in PAIRS]
    comparison = read_csv(d / "共有集計_確認版/主要候補の比較.csv")
    for row in comparison:
        found = [v for v in read_csv(d / row["source"]) if v["method"] == row["method"]]
        assert len(found) == 1
        for field in ("mia_all", "other18_mean", "auc", "wins", "losses"):
            close(row[field], found[0][field], row["label"])
    labels = {"基準Gen-LRA7": "基準 Gen-LRA7", "弱い補正0.1": "弱い補正 0.1", "弱い補正0.25": "弱い補正 0.25",
              "非会員較正500＋補正0.1": "非会員較正 500 + 補正0.1", "非会員較正100＋補正0.1": "非会員較正 100 + 補正0.1",
              "参照拡充＋補正0.1": "参照拡充 + 補正0.1", "性別Gen-LRA7": "性別に分ける Gen-LRA7",
              "9属性Gen-LRA": "9属性 Gen-LRA", "近傍比k=1": "近傍比 k=1", "近傍比k=10": "近傍比 k=10",
              "学習による2特徴補正": "学習: 2特徴", "学習による5特徴補正": "学習: 5特徴"}
    explanation = (d / "説明.md").read_text().replace("**", "")
    for visible, label in labels.items():
        line = [line for line in explanation.splitlines() if line.startswith(f"| {visible} |")]
        assert len(line) == 1, visible
        cells = [cell.strip() for cell in line[0].split("|")][1:-1]
        row = next(row for row in comparison if row["label"] == label)
        assert cells[1] == f"{float(row['mia_all']):.5f}", visible
        assert cells[2] == f"{float(row['other18_mean']):.5f}", visible
        if visible != "基準Gen-LRA7":
            assert cells[3] == f"{row['wins']}／{row['losses']}", visible
    checked_inputs = 0
    if args.check_inputs:
        original_root = Path(os.path.commonpath(list(record["input_sha256"])))
        for saved_path, expected in record["input_sha256"].items():
            path = Path(saved_path)
            if args.input_root is not None:
                path = args.input_root / path.relative_to(original_root)
            assert path.exists(), path
            assert sha(path) == expected, path
            checked_inputs += 1
    result = {"checked_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(), "status": "passed",
              "run_code_hashes_checked": len(source_checks), "input_files_checked": checked_inputs,
              "pairs": pairs, "target_rows_checked": sum(pair["target_rows"] for pair in pairs),
              "aggregate_groups_checked": sum(pair["groups"] for pair in pairs),
              "report_rows_checked": len(labels), "official_tpr_max_error": max(official_errors),
              "writes_submission_files": False, "verifier_sha256": sha(Path(__file__))}
    manifest_path = d / "掲載ファイル一覧.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["files"]:
            path = d / entry["path"]
            assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], path
        result["manifest_files_checked"] = len(manifest["files"])
    if args.output_file is not None:
        args.output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
