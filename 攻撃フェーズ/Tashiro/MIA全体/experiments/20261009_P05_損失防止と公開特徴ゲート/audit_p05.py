"""Stress-test P05 post-hoc selector with leave-one-team-out and thresholds.

All assessment is cohort-level and preliminary. Public Q/C features and
historical aggregate full TPR inputs have been investigated repeatedly.
Removing the held-out row from POLICY SELECTION does not undo the historical
selection of rule definitions. These are *audits*, not unbiased generalization.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
import run_p05 as base

POLICIES=("P04","P05_copy_only","P05_close_only","P05_copy_and_close")
TARGETS=base.TEAM17


def lto_choose(rows,target):
    """Choose best policy on other 16 target outcomes; no held-out outcome used."""
    train=rows[(rows.team!=target)&rows.rule.isin(POLICIES)]
    ref=train[train.rule=="P04"].set_index("team")["full_tpr"]
    record=[]
    for i,policy in enumerate(POLICIES):
        cur=train[train.rule==policy].set_index("team")["full_tpr"]
        diff=cur-ref.reindex(cur.index)
        # Truth is used only for the OTHER cohorts; prefer no complexity if tie.
        hit=sum(cur>=0.10-1e-12)
        gained=sum((cur>=0.10-1e-12)&(ref<0.10-1e-12))
        lost=sum((cur<0.10-1e-12)&(ref>=0.10-1e-12))
        quality=(int(hit-lost),int(gained),int(-lost),float(diff.mean()),-i)
        record.append((quality,policy))
    return max(record,key=lambda x:x[0])[1]


def audit(root,out):
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    saved=root/"攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_P05_損失防止と公開特徴ゲート/results"
    rows=pd.read_csv(saved/"by_target.csv")
    summary=pd.read_csv(saved/"summary.csv")
    assert len(rows)==170
    p04=rows[rows.rule=="P04"].set_index("team")
    chosen=[]
    for t in TARGETS:
        policy=lto_choose(rows,t)
        row=rows[(rows.rule==policy)&(rows.team==t)].iloc[0]
        before=p04.loc[t]
        chosen.append({
            "heldout_team":t,"selected_policy_16_others":policy,
            "heldout_selected_method":row.assigned_method,
            "heldout_P04_method":before.assigned_method,
            "heldout_tpr":float(row.full_tpr),
            "heldout_P04_tpr":float(before.full_tpr),
            "heldout_gain":float(row.full_tpr-before.full_tpr),
            "heldout_success_new_010":int(row.full_tpr>=.1-1e-12 and before.full_tpr<.1-1e-12),
            "heldout_lost_010":int(row.full_tpr<.1-1e-12 and before.full_tpr>=.1-1e-12)
        })
    cv=pd.DataFrame(chosen)
    cv.to_csv(out/"policy_LOOTO.csv",index=False,float_format="%.17g")
    p04_attack=summary.loc[summary.rule=="P04","simulated_attack_power"].item()
    composite_attack=p04_attack+float(cv.heldout_gain.sum())/20
    group={
        "heldout_count":len(TARGETS),
        "mean_increment_target_tpr":float(cv.heldout_gain.mean()),
        "P04_hypothetical_attack":p04_attack,
        "LOOTO_hypothetical_attack":composite_attack,
        "baseline_3rd_official_attack":.352845805596271,
        "third_place_gap":.352845805596271-composite_attack,
        "LOOTO_improved_targets":cv.loc[cv.heldout_gain>1e-12,"heldout_team"].astype(int).tolist(),
        "LOOTO_worsened_targets":cv.loc[cv.heldout_gain< -1e-12,"heldout_team"].astype(int).tolist(),
        "LOOTO_selected_policy_counts":cv.selected_policy_16_others.value_counts().to_dict(),
        "rule_selection_target_truth_heldout":True,
        "historical_policy_definitions_not_independent":True
    }
    (out/"policy_LOOTO_summary.json").write_text(
        json.dumps(group,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # Copy gate intensity: fractions 0.80..1.00 are all predeclared; compare 7 common A resamples.
    copy=summary[summary.rule.isin(["P04","P05_copy_loose80","P05_copy_only","P05_copy_strict100"])]
    bg=pd.read_csv(saved/"backgrounds.csv")
    copy_bg=bg[bg.rule.isin(copy.rule)]
    assert copy_bg.groupby("rule").background.nunique().eq(7).all()
    pivot=copy_bg.pivot(index="background",columns="rule",values="mean_tpr_17")
    pivot["copy_delta_vs_P04"]=pivot["P05_copy_only"]-pivot["P04"]
    pivot.to_csv(out/"copy_7bg_delta.csv",float_format="%.17g")

    # Relaxed/strict public thresholds: do not infer a margin from score labels.
    (feat,grp,p04table,hist,p03,full,ranking,mia,aia,official,
     original,median_rms)=base.inputs(root)
    lookup=hist.set_index(["team","method","condition"])["tpr"]
    contexts={"p03":p03,"official_P04":{
        t:base.panel_method_for_record(t,m,"full",lookup,p03)
        for t,m in original.items()
    }}
    panel=[]
    for ratio in (0.8,0.9,1.0,1.1,1.2):
        for uniq in (.85,.90,.95,1.):
            for nclose in (1,2,3):
                assign={}
                for t in TARGETS:
                    choice,reason=base.public_gates(
                        t,original[t],feat,grp,median_rms,
                        copy_frac=.9,rms_ratio=ratio,uniq_cut=uniq,min_close=nclose,
                        enable_copy=True,enable_close=True)
                    assign[t]=choice
                res,_,_ =base.evaluate(
                    "grid",assign,lookup,contexts,full,mia,aia,official,ranking)
                panel.append({
                    "RMS_cutoff_ratio":ratio,
                    "numeric_uniqueness_cutoff":uniq,
                    "Q_close_match_minimum":nclose,
                    "hypothetical_attack_power":res["simulated_attack_power"],
                    "gap_to_official_third":res["gap_to_third"],
                    "pseudo_rank":res["pseudo_rank_if_added"],
                    "changed_target_teams":";".join(
                        map(str,[t for t in TARGETS if assign[t]!=original[t]])),
                    "change_count":sum(assign[t]!=original[t] for t in TARGETS),
                    "hypothetical_full_010_hits":res["prelim_attack_010_hits_20"],
                    "P03_background_comparability":res["background_comparable_N"]
                })
    grid=pd.DataFrame(panel)
    assert len(grid)==60
    grid.to_csv(out/"public_threshold_60_sensitivity.csv",index=False,float_format="%.17g")
    report={
        "status":"posthoc preliminary gate stress test only",
        "criteria_LOOTO":"maximize remaining16 full >=.10 hits, new hits, avoid losses, mean TPR gain, then fewest complexity",
        "candidate_policies":POLICIES,
        "policy_LOOTO_learned_from_other16_only":True,
        "rule_invention_used_prelim_target_TPR_before_this_audit":True,
        "copy7BG_reuses_same_preliminary_A_bg":True,
        "P03_7BG_comparison_unavailable":True,
        "grid_conditions":len(grid),
        "pass_third_among_60":int((grid.gap_to_official_third<0).sum()),
        "P05_team_7_close_gate_stability_count":int(grid.changed_target_teams.str.contains("7").sum()),
        "no_submission_generated":True
    }
    (out/"audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(group,ensure_ascii=False,indent=2))
    print("public-threshold sensitivity",json.dumps(report,ensure_ascii=False))
    print(pivot.to_string())


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    audit(a.root.resolve(),a.out.resolve())
