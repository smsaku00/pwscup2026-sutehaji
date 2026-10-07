"""既存集計だけを使い、背景参照への厳密一致が改善を説明するか確認する。"""
from pathlib import Path
import json,hashlib
import pandas as pd
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=HERE/"集計/各臨床列の値再利用と課題一致.csv";sp=HERE/"集計/予備戦_対象別と20反復null.csv"
    a=pd.read_csv(ap)
    assert a.Q_match_BG_count.eq(0).all()
    # 各属性の一致セルが全て0。どのqも臨床4列の一致数0なのでQI7同一行一致も不可能。
    public=a[a.attribute=="SBP"].groupby(["phase","team","context"]).Q_rows.sum().reset_index()
    public["BG_clinical_match_count"]=0;public["QI7_in_BG_reference"]=False
    assert len(public)==88 and public.Q_rows.eq(326).all()
    public.to_csv(HERE/"集計/BG一致区分_公開件数.csv",index=False)
    methods=["baseline_partial2_matching","column_all4_mean","column_top2_mean"]
    scores=pd.read_csv(sp);scores=scores[(scores.source=="real_C")&scores.method.isin(methods)]
    b=scores[scores.method==methods[0]].set_index(["team","context"])
    rows=[]
    for x in scores.itertuples():
        for label,column in [("member","hit_m"),("control","hit_c")]:
            base=int(b.loc[(x.team,x.context),column]);hit=int(getattr(x,column))
            rows.append(dict(team=x.team,context=x.context,method=x.method,label=label,
                BG_clinical_match_count=0,QI7_in_BG_reference=False,Q_rows=163,hits=hit,
                baseline_hits=base,hit_difference_vs_baseline=hit-base))
    t=pd.DataFrame(rows);t.to_csv(HERE/"集計/BG一致区分_予備対象別の的中差.csv",index=False)
    agg=[]
    for group,excluded in [("prelim20",[5]),("processed18",[1,2,5])]:
        d=t[~t.team.isin(excluded)]
        z=d.groupby(["context","method","label","BG_clinical_match_count","QI7_in_BG_reference"])[["Q_rows","hits","baseline_hits","hit_difference_vs_baseline"]].sum().reset_index()
        z.insert(0,"group",group);agg.append(z)
    pd.concat(agg).to_csv(HERE/"集計/BG一致区分_的中差合計.csv",index=False)
    rec=dict(analysis="既存公開集計からの後付け診断・追加方式なし",code_sha256=digest(__file__),
       sources_sha256={str(ap.relative_to(HERE)):digest(ap),str(sp.relative_to(HERE)):digest(sp)},
       maximum_same_sex_BG_reference_clinical_match_cells=0,challenge_rows_per_C=326,C_count=44,
       contexts=["split_independent","full_reference_overlap"],
       QI7_absence_proof="QI7一致なら同じ性別のBG行への各臨床列一致が必要。臨床一致セル総数0なのでQI7一致は0。",
       individual_rows_saved=False,main_truth_used=False,B_self_used=False,
       limitation="1e-8以内の一致が0という診断であり、BG参照への近接や周辺分布の情報が寄与しないとは言えない")
    (HERE/"BG一致区分の確認.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n")
    print(pd.concat(agg).query("group == 'processed18'").to_string(index=False))
if __name__=="__main__":main()

