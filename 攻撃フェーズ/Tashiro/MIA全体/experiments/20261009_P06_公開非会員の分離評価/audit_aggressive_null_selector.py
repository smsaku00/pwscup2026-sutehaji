"""Supplemental P06 risk-selection stress audit on already frozen cohort outputs.

No new individual scores. Read only previous P06 heldout-public negative risk
counts and previously published OFFICIAL preliminary aggregate 17-team TPR.
Evaluation follows choice. P01 v KDE-A only; KDE-B kept, P03 excluded from
this ablation because its 20-target full per-team outcomes were not retained.
Selection rule variants conceived after prelim TPR exposure: POSTHOC diagnostic.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook

HERE="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_P06_公開非会員の分離評価/results"
OLD="攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_非学習方式と加工特徴_背景耐性/固定方式_全背景評価.csv"
ALL20="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/results/by_target.csv"
OFFICIAL="公式データ/予備戦/結果/公開用結果表_20260909.xlsx"
BASE="Q_prefecture_sex_shrink500_cross01"
KDEA="q13_median_iqr_sex0_kde_ratio_h2__raw__cross0.1"
KDEB="q7_robust_adaptivek15_h1__age100__cross0.1"
POLICIES=("P04","min_public_null99","min_public_null95","risk99_P01_margin002")
NAME_TO_SOURCE={"P01":BASE,"KDEA":KDEA,"KDEB":KDEB}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(root,out):
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    res=root/HERE
    risk=pd.read_csv(res/"nonmember_risk_panel.csv")
    select=pd.read_csv(res/"selection_frozen_before_truth.csv")
    prob=risk[(risk.source=="A_bg_20pct_heldout")&
              risk.method.isin(["P01","KDEA"])]
    means=prob.groupby(["team","method"])[["risk99","risk95"]].mean()
    prior=select[select.policy=="P04"].set_index("team")["chosen_family"].to_dict()
    assert len(prior)==17 and list(sorted(t for t,m in prior.items() if m=="KDEA"))==[3,7,8,13,23]
    archived=pd.read_csv(root/OLD,encoding="utf-8-sig")
    archived=archived[(archived.condition=="full")&np.isclose(archived.fpr,.01)]
    tpr=archived.set_index(["team","method"]).tpr
    src20=pd.read_csv(root/ALL20)
    row=src20[(src20.background=="full")&
              (src20.inference=="all20")&
              (src20.method=="P01")&
              (src20.team==12)]
    assert len(row)==1
    team12=float(row.iloc[0]["tpr_0.01"])
    wb=load_workbook(root/OFFICIAL,read_only=True,data_only=True)
    mia={}
    for x in list(wb["攻撃_標的別_MIA"].values)[1:]:
        if x[0]==5:mia[int(x[1])]=(float(x[4] or 0),float(x[5] or 0))
    aia={}
    for x in list(wb["攻撃_標的別_AIA"].values)[1:]:
        if x[0]==5:aia[int(x[1])]=float(x[9] or 0)
    oldscore=sum(max(*mia[t])+aia[t] for t in mia)/20
    assert abs(oldscore-.3297953270354882)<1e-12

    metrics=[];bysubject=[]
    for policy in POLICIES:
        assignment=dict(prior)
        for t in prior:
            if prior[t]!="KDEA":continue
            r99_k=float(means.loc[(t,"KDEA"),"risk99"])
            r99_p=float(means.loc[(t,"P01"),"risk99"])
            r95_k=float(means.loc[(t,"KDEA"),"risk95"])
            r95_p=float(means.loc[(t,"P01"),"risk95"])
            if policy=="min_public_null99":change=(r99_p<r99_k-1e-12)
            elif policy=="min_public_null95":change=(r95_p<r95_k-1e-12)
            elif policy=="risk99_P01_margin002":change=(r99_k-r99_p>=.002)
            else:change=False
            if change:assignment[t]="P01"
        selected={}
        for t in mia:
            if t in (1,2):s=mia[t][0]
            elif t==12:s=team12
            else:s=float(tpr.loc[(t,NAME_TO_SOURCE[assignment[t]])])
            selected[t]=s
        strength=sum(max(v,mia[t][1])+aia[t] for t,v in selected.items())/20
        metrics.append({"policy":policy,"hypothetical_attack":strength,
                  "gap_to_prelim_third":.352845805596271-strength,
                  "delta_vs_P04":strength-.34919703643719757,
                  "TPR010_count":sum(v>=.1-1e-12 for v in selected.values()),
                  "changed_KDEA_targets":sum(prior[t]!=assignment[t] for t in prior),
                  "changed_team_ids":";".join(map(str,t for t in prior if prior[t]!=assignment[t]))})
        for t in prior:
            if prior[t]!="KDEA":continue
            bysubject.append({"team":t,"policy":policy,"original":"KDEA","selected":assignment[t],
              "risk99_P01":float(means.loc[(t,"P01"),"risk99"]),
              "risk99_KDEA":float(means.loc[(t,"KDEA"),"risk99"]),
              "risk95_P01":float(means.loc[(t,"P01"),"risk95"]),
              "risk95_KDEA":float(means.loc[(t,"KDEA"),"risk95"]),
              "P01_TPR":float(tpr.loc[(t,BASE)]),
              "KDEA_TPR":float(tpr.loc[(t,KDEA)]),
              "chosen_TPR":selected[t],
              "delta_TPR":selected[t]-float(tpr.loc[(t,KDEA)])})
    m=pd.DataFrame(metrics)
    assert len(m)==4
    assert abs(m[m.policy=="P04"].hypothetical_attack.item()-.34919703643719757)<1e-12
    m.to_csv(out/"unconstrained_null_selection.csv",index=False,float_format="%.17g")
    pd.DataFrame(bysubject).to_csv(out/"unconstrained_null_targets.csv",index=False,float_format="%.17g")
    json_record={
      "status":"explanatory exploratory risk-selection ablation, NOT independent label-free generalization",
      "candidate_method_scope":"P01 vs archived KDE-A on five existing P04 KDE-A teams; other cohorts fixed",
      "policy_method_definitions_seen_past_preliminary_truth":True,
      "negative_rate_SOURCE":"heldout public background A, 1028 rows x 3 seeds, not official FPR",
      "Team1_2_official_and_Team12_P01_frozen":True,
      "official_AIA_rare_MIA_frozen":True,
      "no_main_submission":True,
      "source_sha256":{str(p):sha(root/p) for p in [OLD,ALL20,OFFICIAL,
             HERE+"/nonmember_risk_panel.csv",HERE+"/selection_frozen_before_truth.csv"]}}
    (out/"protocol.json").write_text(json.dumps(json_record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(m.to_string(index=False))
    print(pd.DataFrame(bysubject)[lambda x:x.delta_TPR!=0].to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    x=p.parse_args();run(x.root.resolve(),x.out.resolve())
