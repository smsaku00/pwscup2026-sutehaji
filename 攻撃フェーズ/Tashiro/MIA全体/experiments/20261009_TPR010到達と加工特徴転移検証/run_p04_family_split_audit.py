"""Exploratory P04 AIA-imprint x public-home two-family KDE split sensitivity.

Strictly compare already archived cohort/target/algorithm/7-background TPR
aggregates; no private person labels or membership scores are read.
KDE A vs KDE B orientation was chosen AFTER inspection of the very same
preliminary truth: this post-hoc diagnostic CANNOT estimate main generalization.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd

import run_transfer as base

A="q13_median_iqr_sex0_kde_ratio_h2__raw__cross0.1"
B="q7_robust_adaptivek15_h1__age100__cross0.1"
P01=base.BASELINE
G_REL="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/public_target_groups.csv"
T_REL="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/group_thresholds.json"
HOME_OFFSETS=(-.03,-.01,-.005,0.,.005,.01,.03)
IMPRINT_OFFSETS=(-.05,-.02,0.,.02,.05)
POLICIES=("P01","KDEA_high","KDEB_high","split_AhomeHigh_BhomeLow","reverse_BhomeHigh_AhomeLow")


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def choose(team,rule,features,cuts):
    r=features.loc[team]
    high=bool(r.AIA_imprint_log_BG_to_C_median_Q_NN>=cuts[0])
    homehigh=bool(r.C_home_fraction>=cuts[1])
    if rule=="P01" or not high:
        return P01
    if rule=="KDEA_high":return A
    if rule=="KDEB_high":return B
    if rule=="split_AhomeHigh_BhomeLow":
        return A if homehigh else B
    if rule=="reverse_BhomeHigh_AhomeLow":
        return B if homehigh else A
    raise ValueError(rule)


def evaluate(rule,features,cuts,values,bgs,targets):
    result=[]
    for t in targets:
        method=choose(t,rule,features,cuts)
        arr=np.array([values.loc[(t,method,b)] for b in bgs],float)
        result.append({
            "team":t,"method":method,
            "full_tpr":float(arr[bgs.index("full")]),
            "min_tpr_7":float(arr.min()),
            "mean_tpr_7":float(arr.mean()),
            "hit_full":int(arr[bgs.index("full")]>=.1-1e-12),
            "hit_all7":int((arr>=.1-1e-12).all()),
            "hit_bg_count":int((arr>=.1-1e-12).sum()),
            "imprint_margin":float(features.loc[t,"AIA_imprint_log_BG_to_C_median_Q_NN"]-cuts[0]),
            "home_margin":float(features.loc[t,"C_home_fraction"]-cuts[1]),
        })
    return pd.DataFrame(result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    root,out=a.root.resolve(),a.out.resolve()
    if out.exists():raise FileExistsError("Refuse overwrite "+str(out))
    out.mkdir(parents=True)
    feats,df,methods,bgs,h=base.read_inputs(root)
    group=pd.read_csv(root/G_REL).set_index("team")
    cut=json.loads((root/T_REL).read_text())
    source_cut=(float(cut["imprint_median"]),float(cut["c_home_median"]))
    targets=base.TARGET_SET
    sel=df[df.method.isin([A,B,P01])]
    values=sel.set_index(["team","method","condition"])["tpr"]
    assert len(values)==len(targets)*3*len(bgs)
    assert len(targets)==17 and len(bgs)==7
    high=[t for t in targets if group.loc[t,"AIA_imprint_log_BG_to_C_median_Q_NN"]>=source_cut[0]]
    assert high==[3,7,8,9,13,18,20,23,26] or sorted(high)==[3,7,8,9,13,18,20,23,26]
    rows=[]
    summary=[]
    for method in POLICIES:
        r=evaluate(method,group,source_cut,values,bgs,targets)
        r.insert(0,"policy",method)
        rows.append(r)
        summary.append({
            "policy":method,
            "hit_full_ge010":int(r.hit_full.sum()),
            "hit_all7_ge010":int(r.hit_all7.sum()),
            "mean_full_tpr":float(r.full_tpr.mean()),
            "full_success_teams":";".join(map(str,r.loc[r.hit_full==1,"team"])),
            "all7_success_teams":";".join(map(str,r.loc[r.hit_all7==1,"team"])),
            "applied_A":int((r.method==A).sum()),
            "applied_B":int((r.method==B).sum()),
            "applied_P01":int((r.method==P01).sum()),
        })
    by=pd.concat(rows,ignore_index=True)
    summary=pd.DataFrame(summary)
    ref=summary[summary.policy=="P01"].iloc[0]
    contrast=summary[summary.policy=="split_AhomeHigh_BhomeLow"].iloc[0]
    assert ref.hit_full_ge010==2 and ref.hit_all7_ge010==0
    assert contrast.hit_full_ge010==4 and contrast.hit_all7_ge010==2

    sensitivities=[]
    for dI in IMPRINT_OFFSETS:
        for dH in HOME_OFFSETS:
            chosen=evaluate(
                "split_AhomeHigh_BhomeLow",group,
                (source_cut[0]+dI,source_cut[1]+dH),
                values,bgs,targets)
            sensitivities.append({
                "imprint_cutoff_shift":dI,"home_cutoff_shift":dH,
                "hit_full":int(chosen.hit_full.sum()),
                "hit_all7":int(chosen.hit_all7.sum()),
                "hit_full_teams":";".join(map(str,chosen.loc[chosen.hit_full==1,"team"])),
                "hit_all7_teams":";".join(map(str,chosen.loc[chosen.hit_all7==1,"team"])),
                "mean_full_tpr":float(chosen.full_tpr.mean()),
            })

    # Public-only leave-one-out thresholds. Only metadata of other 17 for
    # each target; no membership outcome is used to choose a split.
    full18=sorted(set(group.index) - {1,2})
    loo=[]
    for t in targets:
        peers=[u for u in full18 if u!=t]
        tmp=(float(group.loc[peers,"AIA_imprint_log_BG_to_C_median_Q_NN"].median()),
             float(group.loc[peers,"C_home_fraction"].median()))
        r=evaluate("split_AhomeHigh_BhomeLow",group,tmp,values,bgs,[t]).iloc[0]
        orig=by[(by.policy=="split_AhomeHigh_BhomeLow")&(by.team==t)].iloc[0]
        loo.append({
            "team":t,"chosen_full_public_split":orig.method,
            "chosen_LOO_public_split":r.method,
            "changed":int(orig.method!=r.method),
            "full_tpr_LOO_public":float(r.full_tpr),
            "hit_full_LOO_public":int(r.hit_full),
            "hit_all7_LOO_public":int(r.hit_all7),
            "thresholds_public_imprint_LOO":tmp[0],
            "thresholds_public_home_LOO":tmp[1],
        })
    loo_df=pd.DataFrame(loo)
    loo_df.to_csv(out/"public_threshold_LOO.csv",index=False,float_format="%.17g")

    # Training selection: for target t, compare the full/seven-view success of
    # each policy on OTHER 16 targets, never target t. This CV cannot remove
    # the posthoc KDE-A/B policy family's prior dependence on exposed labels.
    fold=[]
    for t in targets:
        candidates=[]
        for policy in POLICIES:
            m=by[(by.policy==policy)&(by.team!=t)]
            candidates.append({
                "policy":policy,
                "all7_hits":int(m.hit_all7.sum()),
                "full_hits":int(m.hit_full.sum()),
                "full_mean":float(m.full_tpr.mean()),
            })
        best=max(candidates,key=lambda x:(x["all7_hits"],x["full_hits"],
                                           x["full_mean"],-POLICIES.index(x["policy"])))
        selected=by[(by.policy==best["policy"])&(by.team==t)].iloc[0]
        fold.append({
            "team":t,"selected_training_policy":best["policy"],
            "selected_method":selected.method,
            "training16_stable_hits":best["all7_hits"],
            "training16_full_hits":best["full_hits"],
            "heldout_full_tpr":float(selected.full_tpr),
            "heldout_hit_full":int(selected.hit_full),
            "heldout_hit_all7":int(selected.hit_all7),
        })
    fold_df=pd.DataFrame(fold)
    fold_df.to_csv(out/"policy_choice_leave_one_team_out.csv",
                   index=False,float_format="%.17g")
    summary.to_csv(out/"five_public_rules.csv",index=False,float_format="%.17g")
    by.to_csv(out/"five_public_rules_by_team.csv",index=False,float_format="%.17g")
    pd.DataFrame(sensitivities).to_csv(out/"threshold_grid_sensitivity.csv",
                                      index=False,float_format="%.17g")
    report={
        "scope":"17 preliminary targets, 22 fixed archived methods subset: P01, KDE A, KDE B",
        "criterion":"TPR@FPR=0.01>=0.10; all7 robustness",
        "rule_selected_after_observing_prelim_labels":True,
        "independent_test":False,
        "thresholds_selected_from_public_features_only":{"imprint":source_cut[0],"C_home_fraction":source_cut[1]},
        "method_orientation_selected_posthoc":"KDE A if imprint-high and C_home-high; KDE B if imprint-high and C_home-low",
        "public_LOO_changed_teams":list(loo_df.loc[loo_df.changed==1,"team"]),
        "public_LOO_full_hits":int(loo_df.hit_full_LOO_public.sum()),
        "public_LOO_all7_hits":int(loo_df.hit_all7_LOO_public.sum()),
        "policy_LOOTO_full_hits":int(fold_df.heldout_hit_full.sum()),
        "policy_LOOTO_all7_hits":int(fold_df.heldout_hit_all7.sum()),
        "threshold_sensitivity_grid_n":len(sensitivities),
        "prelim_C_home_team13_margin":float(group.loc[13,"C_home_fraction"]-source_cut[1]),
        "precisely_same_preliminary_truth_reused_in_panel_and_design":True,
        "data_source_sha256":{**h,G_REL:base.hashfile(root/G_REL),T_REL:base.hashfile(root/T_REL)},
        "no_person_scores_or_truth_written":True,
        "main_round_not_used":True,"submission_unchanged":True
    }
    (out/"audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(summary.to_string(index=False))
    print("AUDIT",json.dumps({
      k:report[k] for k in ("public_LOO_changed_teams","public_LOO_full_hits",
                           "policy_LOOTO_full_hits","policy_LOOTO_all7_hits",
                           "prelim_C_home_team13_margin")},ensure_ascii=False))
    print("DONE",out)


if __name__=="__main__":
    main()
