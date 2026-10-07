"""保存済みの対象別集計から比較表・図・実行記録をまとめる。攻撃入力は読まない。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = [
    ("基礎比較", "分析.py", "結果/再集計.json"),
    ("近傍・密度・補正", "追加検討.py", "追加結果_最終版/追加再集計.json"),
    ("対照一致の補助評価", "対照を揃えた補助検証.py", "補助結果/補助検証.json"),
    ("参照拡充", "参照データの検討.py", "参照結果_拡張版/参照再集計.json"),
    ("背景再標本化", "ブートストラップの検討.py", "背景再標本化結果/背景再標本化.json"),
    ("学習と二方向分割", "学習による順位補正の検討.py", "学習結果_順位校正版/学習再集計.json"),
    ("属性部分集合", "属性部分集合の検討.py", "属性部分集合結果/部分集合再集計.json"),
    ("低FPRと本戦順位", "低FPRと対象偏りの検討.py", "低FPR結果/低FPR再集計.json"),
    ("固定候補の感度", "固定候補の感度検証.py", "固定候補の感度結果/感度検証.json"),
    ("非会員による較正", "非会員を使う順位較正の検討.py", "非会員較正結果/非会員較正.json"),
    ("非会員較正の感度", "非会員較正の感度検証.py", "非会員較正の感度結果/較正感度検証.json"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("既存の集計を上書きしません。")
    d, output = args.analysis_dir, args.output_dir
    output.mkdir(parents=True)
    records, inputs = [], {}
    for name, code, source in RUNS:
        result = json.loads((d / source).read_text())
        record = dict(result["record"])
        assert record["code_sha256"] == sha(d / code)
        for path, expected in record.pop("input_sha256").items():
            assert path not in inputs or inputs[path] == expected
            inputs[path] = expected
        if "common_code_sha256" in record:
            assert record["common_code_sha256"] == sha(d / "分析.py")
        records.append({"name": name, "code": code, "raw_result": source,
                        "raw_result_sha256": sha(d / source), **record})
    record = {"scope": "方法の探索的比較。公式の本戦得点を新たに取得したものではない。",
              "goal_mia_all": .0647, "goal_source": "2026-10-07のユーザー共有値",
              "analysis_deadline_jst": "2026-10-07T09:00:00+09:00",
              "input_sha256": inputs, "runs": records,
              "common_code_sha256": sha(d / "分析.py"), "summary_code_sha256": sha(Path(__file__)),
              "main_counts": json.loads((d / "結果/再集計.json").read_text())["main_counts"],
              "baseline_saved_tpr_max_error": json.loads((d / "結果/再集計.json").read_text())["prelim_baseline_saved_tpr_max_error"],
              "writes_submission_files": False, "accesses_codabench": False,
              "raw_results_note": "大きい元JSONはローカルに保存。GitHubには対象別CSVとこの実行記録を掲載する。"}
    (output / "実行記録.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    selected = [
        ("基準 Gen-LRA7", "追加結果_最終版/予備戦_追加集約比較.csv", "global7"),
        ("弱い補正 0.1", "追加結果_最終版/予備戦_追加集約比較.csv", "leave_target_out_alpha_0.1"),
        ("弱い補正 0.25", "追加結果_最終版/予備戦_追加集約比較.csv", "global7_leave_target_out_cross025"),
        ("非会員較正 500 + 補正0.1", "非会員較正結果/非会員較正_集約比較.csv", "Q_prefecture_sex_shrink500_cross01"),
        ("非会員較正 100 + 補正0.1", "非会員較正結果/非会員較正_集約比較.csv", "Q_prefecture_sex_shrink100_cross01"),
        ("参照拡充 + 補正0.1", "参照結果_拡張版/参照_集約比較.csv", "full_aug_plain_cross0.1"),
        ("性別に分ける Gen-LRA7", "追加結果_最終版/予備戦_追加集約比較.csv", "sex7"),
        ("9属性 Gen-LRA", "追加結果_最終版/予備戦_追加集約比較.csv", "global9"),
        ("近傍比 k=1", "追加結果_最終版/予備戦_追加集約比較.csv", "global_nn_k1"),
        ("近傍比 k=10", "追加結果_最終版/予備戦_追加集約比較.csv", "global_nn_k10"),
        ("学習: 2特徴", "学習結果_順位校正版/学習_集約比較.csv", "full__learn_full_nuisance2"),
        ("学習: 5特徴", "学習結果_順位校正版/学習_集約比較.csv", "full__learn_full_signals5"),
    ]
    comparison = []
    for label, source, method in selected:
        table = pd.read_csv(d / source, float_precision="round_trip")
        found = table[table.method == method]
        assert len(found) == 1
        row = found.iloc[0].to_dict()
        comparison.append({"label": label, "source": source, **row})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(output / "主要候補の比較.csv", index=False, float_format="%.17g")
    fixed = pd.read_csv(d / "固定候補の感度結果/感度検証_集約比較.csv", float_precision="round_trip")
    calibrated = pd.read_csv(d / "非会員較正の感度結果/較正感度_集約比較.csv", float_precision="round_trip")
    sensitivity = []
    for reference in calibrated.reference.dropna().unique():
        for label, table, method in [
            ("弱い補正0.1", fixed, "baseline_cross0.1"),
            ("弱い補正0.25", fixed, "baseline_cross0.25"),
            ("参照拡充+補正0.1", fixed, "augmented_cross0.1"),
            ("非会員較正500+補正0.1", calibrated, "Q_pref_sex500_cross01"),
            ("非会員較正100+補正0.1", calibrated, "Q_pref_sex100_cross01"),
        ]:
            row = table[(table.reference == reference) & (table.local_method == method)].iloc[0].to_dict()
            sensitivity.append({"label": label, **row})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(output / "固定候補の背景感度.csv", index=False, float_format="%.17g")

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/pws20261007-mia-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt, font_manager
    font_path = Path("/Library/Fonts/Arial Unicode.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "svg.hashsalt": "pws20261007-mia"})
    plotted = comparison[~comparison.label.isin(["弱い補正 0.25", "近傍比 k=10", "学習: 2特徴", "学習: 5特徴"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.8, 5.8), constrained_layout=True)
    yy = np.arange(len(plotted))
    ax.barh(yy - .16, plotted.mia_all, height=.3, color="#3178ac", label="全20対象")
    ax.barh(yy + .16, plotted.other18_mean, height=.3, color="#66bba6", label="基準が高い対象1・2を除く18対象")
    ax.set_yticks(yy, plotted.label)
    ax.invert_yaxis()
    for y, value in zip(yy - .16, plotted.mia_all):
        ax.text(value + .001, y, f"{value:.5f}", va="center", fontsize=9)
    for y, value in zip(yy + .16, plotted.other18_mean):
        ax.text(value + .001, y, f"{value:.5f}", va="center", fontsize=9)
    ax.set_xlim(0, .12)
    ax.set_xlabel("対象別 TPR@FPR1% の平均（MIA全体）")
    ax.set_title("予備戦の公開正解による探索比較 — 本戦の得点ではありません", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", alpha=.2)
    fig.savefig(output / "主要候補の比較.png", dpi=170)
    fig.savefig(output / "主要候補の比較.svg", metadata={"Date": None})
    plt.close(fig)

    reference_order = ["full", "fraction0.9_seed20261007", "fraction0.9_seed20261008", "fraction0.8_seed20261007",
                       "fraction0.8_seed20261008", "fraction0.5_seed20261007", "fraction0.5_seed20261008"]
    fig, ax = plt.subplots(figsize=(10.8, 5.4), constrained_layout=True)
    for label, color, marker in [("弱い補正0.1", "#3178ac", "o"), ("弱い補正0.25", "#c08442", "x"),
                                 ("非会員較正500+補正0.1", "#2a957c", "s"), ("参照拡充+補正0.1", "#8a69b1", "^")]:
        frame = sensitivity[sensitivity.label == label].set_index("reference").loc[reference_order]
        ax.plot(range(7), frame.same_reference_delta, marker=marker, color=color, label=label, linewidth=1.7)
    ax.axhline(0, color="#666666", linewidth=1)
    ax.set_xticks(range(7), ["全5143人", "90%\n抽出1", "90%\n抽出2", "80%\n抽出1", "80%\n抽出2", "50%\n抽出1", "50%\n抽出2"])
    ax.set_ylabel("同じ背景標本の基準に対するMIA全体の差")
    ax.set_title("背景の人数・抽出標本を変えた感度分析（独立検証ではありません）", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(output / "背景を変えた感度.png", dpi=170)
    fig.savefig(output / "背景を変えた感度.svg", metadata={"Date": None})
    plt.close(fig)
    print(json.dumps({"output_dir": str(output), "runs": len(records), "input_files": len(inputs),
                      "comparison_methods": len(comparison), "summary_code_sha256": record["summary_code_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
