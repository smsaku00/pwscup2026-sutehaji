"""公開Cの特徴と同一P01方式の第3回成績を記述する図。方式選択には使わない。"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "集計/本戦第3回の方式と公開加工特徴_記述用.csv"
OUT = HERE / "図"
OUT.mkdir(exist_ok=True)
FONT = FontProperties(family="Hiragino Sans")
plt.rcParams.update({"font.family": FONT.get_name(), "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "path", "figure.facecolor": "white"})
data = pd.read_csv(SOURCE)
data = data[data["P01"].eq(True)].sort_values("team")
assert len(data) == 15 and data["final_positive"].sum() == 7
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
for ax, y, label in zip(axes, ["boundary_included_span", "round3_final"],
                       ["C近傍の time 幅（中央値・年）\n5位の同点を含む", "第3回 AIA 対象別の最終評価"]):
    for positive, color, face, name in [(True, "#117d81", "#117d81", "最終評価が正（7対象）"),
                                      (False, "#62686d", "white", "最終評価が0（8対象）")]:
        sub = data[data["final_positive"].eq(positive)]
        ax.scatter(sub["q_C_full_nn_median"], sub[y], s=70, color=color,
                   facecolors=face, linewidths=1.4, label=name, zorder=3)
        for row in sub.itertuples(index=False):
            # Close points at zero are separated vertically in their labels only.
            offsets = ({1: (0, -16), 2: (-20, 12), 13: (12, 20), 12: (8, -16),
                        27: (3, -16), 22: (-15, -20), 26: (14, -12),
                        5: (14, 10), 25: (-20, 14)} if y == "boundary_included_span"
                       else {1: (-12, -16), 27: (3, -16), 8: (-42, 14),
                             12: (-20, -12), 22: (10, 18), 26: (38, -13),
                             5: (14, 8), 25: (-18, -16)})
            offset = offsets.get(row.team, (6, 7))
            ax.annotate(str(row.team), (row.q_C_full_nn_median, getattr(row, y)),
                        xytext=offset, textcoords="offset points", fontsize=9, color=color,
                        arrowprops={"arrowstyle": "-", "lw": .55, "color": color}
                        if max(abs(v) for v in offset) >= 18 else None)
    ax.set_xlabel("課題人物からCへの5属性距離（中央値）")
    ax.set_ylabel(label)
    ax.grid(alpha=.18)
    ax.margins(x=.1, y=.15)
axes[0].set_yscale("symlog", linthresh=.05)
axes[0].set_yticks([0, .05, .2, 1, 5, 10])
axes[0].set_yticklabels(["0", "0.05", "0.2", "1", "5", "10"])
axes[1].set_ylim(-.015, max(data.round3_final) * 1.25)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .89),
           ncol=2, frameon=False, fontsize=10)
fig.suptitle("同じP01方式を使った本戦15対象：公開Cの特徴と第3回成績", fontsize=16, y=.97)
fig.text(.5, .015, "点の数字はチーム番号。結果を見た後の記述比較であり、方式選択の閾値や本戦での改善を示す図ではない。\n"
         "最終評価には公開されていない φ の影響もある。左の縦軸は0付近を線形、それ以外を対数表示。",
         ha="center", va="bottom", fontsize=9, color="#555")
fig.tight_layout(rect=[0, .1, 1, .91])
for ext in ("png", "svg"):
    fig.savefig(OUT / f"本戦P01_公開特徴と第3回成績.{ext}", dpi=170)
plt.close(fig)
receipt = {"source": str(SOURCE.relative_to(HERE)),
           "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           "purpose": "posthoc descriptive only, no threshold fitting",
           "teams": [int(x) for x in data.team],
           "positive_teams": [int(x) for x in data[data.final_positive].team],
           "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(OUT.glob("本戦P01_公開特徴と第3回成績.*"))}}
(OUT / "作図記録.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"n": len(data), "positive": int(data.final_positive.sum()), "out": str(OUT)}, ensure_ascii=False))
