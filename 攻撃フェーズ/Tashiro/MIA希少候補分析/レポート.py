"""計算結果と、別に記録した暫定判断から共有用の表・説明・図を作る。"""
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def candidate_summary(detail, candidates):
    selections = json.loads((HERE/"候補選定.json").read_text(encoding="utf-8"))
    rows = []
    for entry in selections:
        g = detail[(detail.team == entry["team"]) & (detail.role == "candidate")
                   & (detail.method == "mixed9_L2")]
        nearest = g.sort_values(["d1", "record_id"]).iloc[0]
        specific = g.sort_values(["specificity", "record_id"], ascending=[False, True]).iloc[0]
        rows.append({**entry, "candidate_ids": ";".join(map(str, entry["candidate_ids"])),
                     "membership": "未確定", "selection_type": "計算後の暫定判断",
                     "primary_closest_id": int(nearest.record_id), "primary_closest_distance": nearest.d1,
                     "primary_most_specific_id": int(specific.record_id), "primary_max_specificity": specific.specificity})
    rows.append({"team": 28, "candidate_ids": ";".join(map(str, candidates)),
                 "assessment": "自チーム検証", "reason": "11行すべて既知会員。希少24行全体も検証する。",
                 "competing_teams": "対象外", "membership": "既知会員", "selection_type": "自チームB_selfによる既知情報"})
    return pd.DataFrame(rows)


def table(df):
    def fmt(x):
        if pd.isna(x): return "—"
        if isinstance(x, (float, np.floating)): return f"{x:.6f}"
        return str(x).replace("|", "/").replace("\n", " ")
    return "| " + " | ".join(df.columns) + " |\n| " + " | ".join(["---"]*len(df.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(fmt(x) for x in row) + " |" for row in df.itertuples(index=False, name=None))


def draw(pairs, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"pws2026_mia_publish_mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir())/"pws2026_mia_publish_cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output/"図"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), constrained_layout=True)
    specs = [("primary_d1", "Distance to nearest C row", "viridis_r", None),
             ("primary_known13_both_ge", "Known nonmembers at least as strong (of 13)", "Blues", 13)]
    for ax, (col, title, cmap, vmax) in zip(axes, specs):
        p = pairs.pivot(index="team", columns="record_id", values=col).sort_index()
        values = p.to_numpy(float)
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(p.columns)), p.columns, rotation=45)
        ax.set_yticks(range(len(p.index)), p.index)
        ax.set_xlabel("Candidate record_id")
        ax.set_ylabel("Target team")
        ax.set_title(title, fontsize=11, pad=12)
        if vmax is not None:
            for i in range(len(p.index)):
                for j in range(len(p.columns)):
                    ax.text(j, i, str(int(values[i, j])), ha="center", va="center", fontsize=8,
                            color="white" if values[i, j] >= 7 else "#222222")
        fig.colorbar(im, ax=ax, shrink=.75)
    fig.suptitle("Rare-record diagnostics: mixed9 L2 (not membership probabilities)", fontsize=13)
    fig.savefig(figures/"候補比較.png", dpi=150)
    plt.close(fig)


def write_reports(frames, run, output):
    summary = frames["チーム別候補.csv"]
    display = summary[["team", "candidate_ids", "assessment", "reason"]].rename(columns={
        "team": "チーム", "candidate_ids": "調査候補ID", "assessment": "扱い", "reason": "理由・留保"})
    display["調査候補ID"] = display["調査候補ID"].str.replace(";", "、", regex=False)
    own = frames["自チーム28検証.csv"]
    own_rows = []
    for method, g in own.groupby("method", sort=False):
        c = g[g.other_team_candidate]
        own_rows.append({"設定": method, "希少24行の相互最近傍数": int((g.C_to_B_rank_min == 1).sum()),
                         "候補11行の相互最近傍数": int((c.C_to_B_rank_min == 1).sum())})
    detail = frames["全設定診断.csv"]
    examples = detail[(detail.method == "mixed9_L2") & (((detail.team == 17) & (detail.record_id == 264)) |
                                                      ((detail.team == 9) & (detail.record_id == 462)))].copy()
    examples["sum_components"] = examples.d1**2*9
    examples = examples[["team", "record_id", "nearest_C_csv_line", "sum_components", "d1", "nearest_competitor_distance", "specificity",
                         "known13_distance_le_count", "known13_specificity_ge_count"]]
    examples = examples.rename(columns={"team": "チーム", "record_id": "ID", "nearest_C_csv_line": "CのCSV行番号",
        "sum_components": "9項目の寄与合計", "d1": "d1", "nearest_competitor_distance": "d_other", "specificity": "固有性",
        "known13_distance_le_count": "同等以上に近い非会員数", "known13_specificity_ge_count": "同等以上に固有な非会員数"})
    notes = (HERE/"説明テンプレート.md").read_text(encoding="utf-8")
    for token, replacement in {"{{TEAM_TABLE}}": table(display), "{{SELF_TABLE}}": table(pd.DataFrame(own_rows)),
                               "{{EXAMPLES}}": table(examples), "{{SCALES}}": table(pd.DataFrame({"項目": run["numeric_columns"], "尺度": run["background_IQR_div_1_349"]}))}.items():
        notes = notes.replace(token, replacement)
    (output/"説明.md").write_text(notes, encoding="utf-8")
    if output != HERE:
        # 説明中の相対リンクが同じ出力フォルダでたどれるようにする。
        import shutil
        for name in ["分析.py", "検証.py", "共通設定.py", "レポート.py", "候補選定.json", "入力ハッシュ.json", "説明テンプレート.md"]:
            shutil.copy2(HERE/name, output/name)
    intro = """# 本戦 MIA希少候補分析

2026-09-28 / 自チーム28 / チーム内共有用。

**希少24行からAIA標的13行を除いた11行について、22チームのCを比較した。候補242組の所属はすべて未確定。**
まず確認したい組はチーム17・ID264とチーム9・ID462。3設定すべてで、各チームの11候補中の距離・固有性とも1位だった。ただし属性の不一致もあり、確定会員とはしていない。

## 読む順番

1. [説明・計算式・チーム別判断](説明.md)
2. [チーム別候補](集計/チーム別候補.csv) / [候補242組](集計/候補242組.csv)
3. [計算プログラム](分析.py) / [独立検証プログラム](検証.py) / [検証結果](検証結果.json)

`候補選定.json` は会話で示した暫定判断の記録。プログラムが計算する順位と分けて管理している。
入力データ原本・B_self・トークン・提出用回答は同梱していない。チーム28の手元のB_selfが再計算に必要。

## チーム別の調査候補

""" + table(display) + "\n\n再計算のコマンド、必要な入力、列の意味は[説明.md](説明.md)を参照。\n"
    (output/"README.md").write_text(intro, encoding="utf-8")
    draw(frames["候補242組.csv"], output)
