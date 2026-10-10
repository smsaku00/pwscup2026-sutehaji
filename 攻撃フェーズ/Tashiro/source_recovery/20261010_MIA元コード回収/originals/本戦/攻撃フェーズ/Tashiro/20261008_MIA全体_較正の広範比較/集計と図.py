"""保存済みの2回の実験結果から共有表と図を作る。実験結果は変更しない。"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR","/private/tmp/pws_calibration_mpl_20261008")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"共有集計"
# 共有用の表・図は再描画可能。保存済みの実験履歴は読み取りだけ。
OUT.mkdir(exist_ok=True)
font_manager.fontManager.addfont("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")
plt.rcParams.update({"font.family":font_manager.FontProperties(fname="/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc").get_name(),
                    "font.size":10,"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
first=pd.read_csv(HERE/"結果/頑健性比較.csv")
second=pd.read_csv(HERE/"追加混合/結果/頑健性比較.csv")
all_models=pd.concat([first,second[second.family=="mixed"]],ignore_index=True)
assert len(all_models)==89 and all_models.method.is_unique
all_models.to_csv(OUT/"全89方式_頑健性比較.csv",index=False,float_format="%.17g")
names={"flat_pref_sex_s500":"県・性別500（基準）","flat_pref_sex_s100":"県・性別100",
       "flat_pref_sex_s200":"県・性別200","flat_sex_age_s100":"性別・年齢100",
       "mix_county200_age100_w0.1":"県・性別200を90%＋性別・年齢100を10%"}
selected=all_models.set_index("method").loc[list(names)].copy()
selected.insert(0,"label",[names[x] for x in selected.index])
selected.to_csv(OUT/"主要候補の比較.csv",float_format="%.17g")

grid=first[first.family=="flat_pref_sex"].sort_values("strength")
reference=all_models[all_models.method=="flat_pref_sex_s500"].iloc[0]
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,column,title in zip(axes,["full_other18","sample6_mean_other18"],["元の全背景","背景抽出6条件の平均"]):
    ax.plot(grid.strength,grid[column]*100,"o-",color="#2563eb",linewidth=2,markersize=5)
    ax.axhline(reference[column]*100,color="#64748b",linestyle="--",linewidth=1,label="較正500")
    ax.set_xscale("symlog",linthresh=25)
    ax.set_xlim(-3,6500)
    ax.set_xticks([0,100,500,5000])
    ax.set_xticklabels(["0","100","500","5000"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("較正の強さ（小さいほど群別補正が強い）")
    ax.set_ylabel("高得点2対象を除く18対象のTPR（%）")
    ax.set_title(title,pad=12)
    ax.grid(axis="y",alpha=.2)
    ax.legend(frameon=False,loc="best")
fig.suptitle("県・性別の較正：数値を変えた結果",fontsize=15,y=.98)
fig.text(.5,.01,"予備戦公開正解による探索。FPR1%。背景抽出は独立したコホートの検証ではない。",ha="center",fontsize=9,color="#475569")
fig.tight_layout(rect=[0,.055,1,.94])
fig.savefig(OUT/"較正の強さ比較.png",dpi=170)
plt.close(fig)

fig,ax=plt.subplots(figsize=(9,6.5))
groups={"県・性別系":all_models.family.isin(["flat_pref_sex","adaptive_pref_sex","hier_pref_sex"]),
        "年齢帯を使う":all_models.family.isin(["flat_sex_age","hier_sex_age"]),
        "混合":all_models.family=="mixed", "その他":all_models.family.isin(["raw","flat_sex"])}
for (label,mask),color in zip(groups.items(),["#2563eb","#d97706","#059669","#94a3b8"]):
    rows=all_models[mask]
    ax.scatter(rows.full_other18*100,rows.sample6_mean_other18*100,s=44,color=color,alpha=.65,label=label)
ax.axvline(reference.full_other18*100,color="#64748b",linestyle="--",linewidth=1)
ax.axhline(reference.sample6_mean_other18*100,color="#64748b",linestyle="--",linewidth=1)
ax.scatter([reference.full_other18*100],[reference.sample6_mean_other18*100],marker="*",s=240,color="#0f172a",zorder=5)
annotations={"flat_pref_sex_s500":("基準500",(-85,-35)),"flat_pref_sex_s100":("100",(22,-35)),
             "flat_pref_sex_s200":("200",(22,-16)),"flat_sex_age_s100":("性別・年齢100",(-110,22)),
             "mix_county200_age100_w0.1":("90%＋10%の混合",(16,10))}
for name,(label,offset) in annotations.items():
    row=all_models[all_models.method==name].iloc[0]
    ax.annotate(label,xy=(row.full_other18*100,row.sample6_mean_other18*100),xytext=offset,textcoords="offset points",
                arrowprops={"arrowstyle":"-","color":"#475569","lw":.8},fontsize=10,color="#0f172a",
                bbox={"boxstyle":"round,pad=.18","facecolor":"white","edgecolor":"none","alpha":.8})
ax.set_xlabel("高得点2対象を除くTPR：元の全背景（%）",labelpad=10)
ax.set_ylabel("同TPR：背景抽出6条件の平均（%）",labelpad=10)
ax.set_title("89方式の比較：全背景と背景変化の両方を見る",fontsize=14,pad=18)
ax.grid(alpha=.16)
ax.legend(frameon=False,loc="lower right")
fig.text(.5,.018,"破線は較正500。右上ほど両平均が高い。ただし全背景での改善は最大1対象に限られる。",ha="center",fontsize=9,color="#475569")
fig.tight_layout(rect=[0,.055,1,1])
fig.savefig(OUT/"全89方式の比較.png",dpi=170)
plt.close(fig)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
record={"code_sha256":sha(Path(__file__)),"sources_sha256":{str(p.relative_to(HERE)):sha(p) for p in [HERE/"結果/頑健性比較.csv",HERE/"追加混合/結果/頑健性比較.csv"]},
        "outputs_sha256":{p.name:sha(p) for p in OUT.iterdir() if p.name!="集計記録.json"},"uses_saved_results_only":True}
(OUT/"集計記録.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
print(selected[["label","full_other18","full_wins_other18","full_losses_other18","sample6_mean_other18","sample6_wins","sample6_losses"]].to_string())
