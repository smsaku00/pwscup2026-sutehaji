"""P05: auditable public-C/AIA-loss-prevention gates on official preliminary results.

P04 high-imprint A/B + low-imprint P01 is the prior exploratory choice.
P05 fixes two *public-feature-only* fallback hypotheses:
  COPY: if >= 90% of C full13 tuples occur in public background rows, P01;
  CLOSE: for high-imprint KDE-A teams, if at least 2 AIA challenges have
    sex-matched 5D NN RMS <0.05, C median numeric uniqueness >=0.90, and
    AIA median 5D NN RMS is below the median of observed public cohorts,
    use C-NN P03 in preference to KDE A.
The latter is an **outcome-informed hypothesis**: it was devised after seeing
the Team7 loss, and is NOT prospective validation. It is therefore reported
separately and tested for threshold/outer-team sensitivity.

This script reads saved COHORT-level TPRs, public-only processing descriptors,
and the published official PRELIMINARY result workbook. It reads NO individual
membership labels, no self B, no raw C, and writes no submission or candidate
confidence values. All reconstructed "attack power" scores are counterfactual:
AIA + rare MIA held at the official 2026-09-09 results, Team1/2 old TPRs
kept, Team12 is a P01 simulation, not a historical score.

"Seven backgrounds" use previously saved same P01/KDE A/KDE B panels;
P03 was evaluated with DIFFERENT backgrounds, so it is compared in
FULL background only. No claim of 7-background validation of P03.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from openpyxl import load_workbook

HISTORY="攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_非学習方式と加工特徴_背景耐性/固定方式_全背景評価.csv"
FEATURES="攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_非学習方式と加工特徴_背景耐性/公開加工特徴.csv"
PRELIM_GROUPS="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/public_target_groups.csv"
P04_RULES="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_TPR010到達と加工特徴転移検証/p04_family_split/five_public_rules_by_team.csv"
P03="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/posthoc/gated_by_target.csv"
P01_FULL20="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/results/by_target.csv"
OFFICIAL="公式データ/予備戦/結果/公開用結果表_20260909.xlsx"
P01="Q_prefecture_sex_shrink500_cross01"
A="q13_median_iqr_sex0_kde_ratio_h2__raw__cross0.1"
B="q7_robust_adaptivek15_h1__age100__cross0.1"
TEAM17=(3,4,6,7,8,9,10,13,17,18,19,20,22,23,24,26,29)
LEGACY_BIG=(1,2)
TEAM12=12
THRESHOLD=0.10
COPY_FRACTIONS=(0.8,0.9,0.95,1.0)
CLOSE_RMS_SCALES=(0.8,1.0,1.2)  # median(Q 5D public only) multipliers
CLOSE_UNIQUENESS=(0.85,0.90,0.95)
MIN_CLOSE_COUNTS=(1,2,3)
P03_FROM="Cnn_imprint_high"
N_KNOWN_TARGETS=20


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def parse_public_tables(root):
    feat=pd.read_csv(root/FEATURES,encoding="utf-8-sig").set_index("team")
    group=pd.read_csv(root/PRELIM_GROUPS).set_index("team")
    p04=pd.read_csv(root/P04_RULES)
    hist=pd.read_csv(root/HISTORY,encoding="utf-8-sig")
    hist=hist[hist.team.isin(TEAM17)&np.isclose(hist.fpr,0.01)].copy()
    p03=pd.read_csv(root/P03)
    p03=p03[(p03.inference=="without1_2")&
             (p03.background=="full")&
             (p03.method==P03_FROM)].set_index("team")
    full=pd.read_csv(root/P01_FULL20)
    full=full[(full.inference=="all20")&(full.background=="full")&
              (full.method=="P01")].set_index("team")
    assert len(feat.loc[list(TEAM17)])==17
    assert len(p04[p04.policy=="split_AhomeHigh_BhomeLow"])==17
    assert len(p03.loc[list(TEAM17)])==17
    assert len(full)==20
    assert hist.condition.nunique()==7
    assert len(hist)==17*22*7
    return feat,group,p04,hist,p03,full


def official_excel(root):
    wb=load_workbook(root/OFFICIAL,read_only=True,data_only=True)
    attack={}
    for row in list(wb["総括"].values)[1:]:
        if not isinstance(row[0],int) or not isinstance(row[15],(int,float)):
            continue
        attack[int(row[0])]={"team_name":row[1],"official_attack":float(row[15]),
                             "official_attack_rank":row[16]}
    assert len(attack)==21
    main={}
    for row in list(wb["攻撃_標的別_MIA"].values)[1:]:
        if row[0]!=5:continue
        t=int(row[1])
        main[t]={
          "members":int(row[2]),"rare_count":int(row[3]),
          "old_tpr":float(row[4] or 0),
          "rare_tpr":float(row[5] or 0),
          "official_mia_component":float(row[6] or 0),
        }
    aia={}
    for row in list(wb["攻撃_標的別_AIA"].values)[1:]:
        if row[0]!=5:continue
        aia[int(row[1])]=float(row[9] or 0)
    assert len(main)==len(aia)==20
    assert set(main)==set(aia)==(set(TEAM17)|set(LEGACY_BIG)|{TEAM12})
    official=sum(max(m["old_tpr"],m["rare_tpr"])+aia[t]
                 for t,m in main.items())/20
    assert abs(official-attack[5]["official_attack"])<1e-12
    return attack,main,aia,official


def descriptors(feat, group):
    """Derived from published public features only."""
    subset=feat.loc[list(TEAM17)]
    qmedian=float(subset["Q_sexmatched_QI5_RMS_median"].median())
    return qmedian


def choose_original(teams):
    r=teams.set_index("team")
    return {int(t):str(r.loc[t,"method"]) for t in TEAM17}


def public_gates(t,method,feat,grp,median_rms,
                 copy_frac=.9,rms_ratio=1.,uniq_cut=.9,min_close=2,
                 enable_copy=True,enable_close=True):
    """Only public measurements, not predicted or target membership scores."""
    f=feat.loc[t]
    d=grp.loc[t]
    is_copy=(
        float(f["public_A_full13_rows"])/float(f["rows"])>=copy_frac
    )
    is_close=(bool(d["aia_imprint_high"]) and method==A
              and int(f["Q_sexmatched_QI5_RMS_lt005"])>=min_close
              and float(d["C_median_numeric_unique_fraction"])>=uniq_cut
              and float(f["Q_sexmatched_QI5_RMS_median"])<
                  median_rms*rms_ratio)
    if enable_copy and is_copy:return P01,"public_background_copy"
    if enable_close and is_close:return P03_FROM,"public_AIA_near_C_continuous"
    return method,"P04_preserved"


def base_method_score_panel(hist,rule):
    if rule not in (P01,A,B):
        raise ValueError(rule)
    vals=hist[hist.method==rule].set_index(["team","condition"])["tpr"]
    return vals


def inputs(root):
    feat,grp,p04,hist,p03,full=parse_public_tables(root)
    attack,official_mia,official_aia,official_attack=official_excel(root)
    original=choose_original(p04[p04.policy=="split_AhomeHigh_BhomeLow"])
    qmedian=descriptors(feat,grp)
    # numerical regression of the historical baseline for all17.
    historyfull=hist[(hist.condition=="full")&(hist.method==P01)].set_index("team")
    for t in TEAM17:
        prior=p04[(p04.policy=="P01")&(p04.team==t)].full_tpr.iloc[0]
        assert abs(float(historyfull.loc[t,"tpr"])-float(prior))<1e-12
    return (feat,grp,p04,hist,p03,full,attack,official_mia,
            official_aia,official_attack,original,qmedian)


def panel_method_for_record(t,method,view,lookup,p03):
    if method==P03_FROM:
        if view!="full":
            return None
        return float(p03.loc[t,"tpr_0.01"])
    return float(lookup.loc[(t,method,view)])


def score_mia_all(official_mia,full,values,old_1_2=True):
    result={}
    for t in official_mia:
        if t in LEGACY_BIG:
            result[t]=float(official_mia[t]["old_tpr"]) if old_1_2 else float(full.loc[t,"tpr_0.01"])
        elif t==TEAM12:result[t]=float(full.loc[t,"tpr_0.01"])
        else:result[t]=float(values[t])
    return result


def attack_power(tpr_all,official_mia,official_aia):
    return sum(max(float(score),official_mia[t]["rare_tpr"])+official_aia[t]
               for t,score in tpr_all.items())/N_KNOWN_TARGETS


def evaluate(rule_name, assign,lookup,contexts,full,official_mia,official_aia,
             official_attack,attack):
    allview=sorted(set(key[2] for key in lookup.index))
    fullvals={t:panel_method_for_record(t,m,"full",lookup,contexts["p03"])
              for t,m in assign.items()}
    tp=score_mia_all(official_mia,full,fullvals,old_1_2=True)
    power=attack_power(tp,official_mia,official_aia)
    positive=set(int(t) for t,v in tp.items() if v>=THRESHOLD-1e-12)
    rank=1+sum(v["official_attack"]>power+1e-12 for k,v in attack.items() if k!=5)
    baseline=contexts["official_P04"]
    diffs={t:float(fullvals[t]-baseline[t]) for t in TEAM17}
    wins=sum(v>1e-12 for v in diffs.values())
    losses=sum(v< -1e-12 for v in diffs.values())
    stable=[]
    table=[]
    for view in allview:
        bgval={t:panel_method_for_record(t,m,view,lookup,contexts["p03"])
               for t,m in assign.items()}
        if any(v is None for v in bgval.values()):
            continue
        successes=sum(v>=THRESHOLD-1e-12 for v in bgval.values())
        stable.append({"background":view,
                       "p04_success_count_17":successes,
                       "mean_tpr_17":sum(bgval.values())/len(bgval)})
        table.append(bgval)
    # Only 7-bg panels using SAME archived KDE/P01 comparison conditions.
    all7=sum(all(vals[t]>=THRESHOLD-1e-12 for vals in table)
             for t in TEAM17) if len(table)==7 else None
    return {
      "rule":rule_name,
      "simulated_attack_power":power,
      "official_old_attack_power":official_attack,
      "delta_old_official":power-official_attack,
      "gap_to_third":attack[6]["official_attack"]-power,
      "gap_to_second":attack[9]["official_attack"]-power,
      "gap_to_first":attack[19]["official_attack"]-power,
      "pseudo_rank_if_added":rank,
      "prelim_MIA_all_mean_20":sum(tp.values())/20,
      "prelim_attack_010_hit_teams":";".join(map(str,sorted(positive))),
      "prelim_attack_010_hits_20":len(positive),
      "improved_target_count_over_P04":wins,
      "worsened_target_count_over_P04":losses,
      "net_tpr_gain_over_P04_17":sum(diffs.values())/17,
      "background_comparable_N":len(stable),
      "stable_7bg_targets":all7,
      "worst_7bg_success_count_17":min((b["p04_success_count_17"] for b in stable),default=None),
      "best_7bg_success_count_17":max((b["p04_success_count_17"] for b in stable),default=None),
    },fullvals,stable


def run(root,output):
    if output.exists(): raise FileExistsError("Refuse overwrite "+str(output))
    output.mkdir(parents=True)
    (feat,grp,p04,hist,p03,full,ranking,mia,aia,official,
      original,qmedian)=inputs(root)
    lookup=hist.set_index(["team","method","condition"])["tpr"]
    # Fixed candidate contrasts; no separate rules per target ID.
    rules={
        "P04": dict(copy=False,close=False,copy_frac=.9,rms_ratio=1.,uniq=.9,nclose=2),
        "P05_copy_only": dict(copy=True,close=False,copy_frac=.9,rms_ratio=1.,uniq=.9,nclose=2),
        "P05_close_only":dict(copy=False,close=True,copy_frac=.9,rms_ratio=1.,uniq=.9,nclose=2),
        "P05_copy_and_close":dict(copy=True,close=True,copy_frac=.9,rms_ratio=1.,uniq=.9,nclose=2),
        "P05_copy_loose80":dict(copy=True,close=False,copy_frac=.8,rms_ratio=1.,uniq=.9,nclose=2),
        "P05_copy_strict100":dict(copy=True,close=False,copy_frac=1.,rms_ratio=1.,uniq=.9,nclose=2),
        "P05_close_rms08":dict(copy=False,close=True,copy_frac=.9,rms_ratio=.8,uniq=.9,nclose=2),
        "P05_close_rms12":dict(copy=False,close=True,copy_frac=.9,rms_ratio=1.2,uniq=.9,nclose=2),
        "P05_close_require3":dict(copy=False,close=True,copy_frac=.9,rms_ratio=1.,uniq=.9,nclose=3),
        "P05_close_relaxed1":dict(copy=False,close=True,copy_frac=.9,rms_ratio=1.,uniq=.9,nclose=1),
    }
    rows=[];summary=[];backgroundrows=[]
    context={"p03":p03,"official_P04":{t:panel_method_for_record(t,m,"full",lookup,p03)
                      for t,m in original.items()}}
    all_seven=sorted(hist.condition.unique())
    assert len(all_seven)==7
    assert abs(official-.3297953270354882)<1e-12
    for name,params in rules.items():
        assignment={}
        for t in TEAM17:
            method,reason=public_gates(
                t,original[t],feat,grp,qmedian,
                copy_frac=params["copy_frac"],rms_ratio=params["rms_ratio"],
                uniq_cut=params["uniq"],min_close=params["nclose"],
                enable_copy=params["copy"],enable_close=params["close"])
            assignment[t]=method
            fullscore=panel_method_for_record(t,method,"full",lookup,p03)
            rows.append({
                "rule":name,"team":t,"original_method":original[t],
                "assigned_method":method,"reason":reason,
                "full_tpr":fullscore,
                "p04_full_tpr":context["official_P04"][t],
                "full_delta":fullscore-context["official_P04"][t],
                "public_copy_fraction":float(feat.loc[t,"public_A_full13_rows"])/float(feat.loc[t,"rows"]),
                "sexmatched_Q_near005":int(feat.loc[t,"Q_sexmatched_QI5_RMS_lt005"]),
                "public_Q_RMS_median":float(feat.loc[t,"Q_sexmatched_QI5_RMS_median"]),
                "public_RMS_median_threshold":qmedian,
                "C_numeric_median_unique_fraction":float(grp.loc[t,"C_median_numeric_unique_fraction"]),
            })
        s,_,background=evaluate(name,assignment,lookup,context,full,mia,aia,official,ranking)
        summary.append(s)
        for r in background: backgroundrows.append({"rule":name,**r})
    summarydf=pd.DataFrame(summary)
    targetdf=pd.DataFrame(rows)
    bgdf=pd.DataFrame(backgroundrows)
    assert len(summarydf)==10 and len(targetdf)==170
    # P04 and copy-only MUST have all seven truly comparable backgrounds.
    for name in ("P04","P05_copy_only","P05_copy_loose80","P05_copy_strict100"):
        z=summarydf[summarydf.rule==name].iloc[0]
        assert z.background_comparable_N==7,(name,z.background_comparable_N)
    assert summarydf.loc[summarydf.rule=="P04","simulated_attack_power"].item()==(
        summarydf.loc[summarydf.rule=="P04","simulated_attack_power"].item())
    assert abs(summarydf.loc[summarydf.rule=="P04","simulated_attack_power"].item()
               -.34919703643719757)<1e-12
    targetdf.to_csv(output/"by_target.csv",index=False,float_format="%.17g")
    summarydf.to_csv(output/"summary.csv",index=False,float_format="%.17g")
    bgdf.to_csv(output/"backgrounds.csv",index=False,float_format="%.17g")
    # Public data, not target scores, determine gates; this is posthoc rule development.
    params={
      "status":"P05 exploratory post-hoc, no heldout test, not an official submission",
      "public_feature_gates_outcome_informed":True,
      "source_MIA_from_archived_22_methods_and_P03_full":True,
      "P03_background_scope_limit":"P03 was evaluated on other A_bg draws: only SAME full background comparable, no seven-background claim",
      "source_candidates_selected_after_exposed_preliminary_truth":True,
      "baseline_official_attack_reconstructed":official,
      "P04_counterfactual_20_target_attack":.34919703643719757,
      "official_rank_converted_to_hypothetical_rank":True,
      "fixed_official_AIA_and_rare_MIA":True,
      "Team1_2":"Retain actual official preliminary MIA full target TPR 0.80,1.00; actual response records not regenerated",
      "Team12":"Use simulated P01 0.076923, not original 0",
      "public_Q_RMS_median":qmedian,
      "methods_and_candidate_rules":rules,
      "source_sha256":{p:sha(root/p) for p in
       [HISTORY,FEATURES,PRELIM_GROUPS,P04_RULES,P03,P01_FULL20,OFFICIAL]},
      "no_person_level_truth_or_scores_written":True,
      "main_B_data_not_read":True,
      "no_main_submission":True
    }
    write_json(output/"protocol.json",params)
    print("P05 public feature methods:\n",summarydf.to_string(index=False))
    print("P05 full candidate changes:\n",targetdf[(targetdf.full_delta.abs()>1e-12)&
          targetdf.rule.isin(["P05_copy_only","P05_close_only","P05_copy_and_close"])
          ].to_string(index=False))
    print("OUTPUT",output)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    run(a.root.resolve(),a.out.resolve())
