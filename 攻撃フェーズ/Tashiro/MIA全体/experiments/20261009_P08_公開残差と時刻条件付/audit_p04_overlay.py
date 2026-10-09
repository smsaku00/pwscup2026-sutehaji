"""Post-hoc P04 + P08 official A_bg public-copy ablation crossover.

Use previously frozen P08 exact-C-duplicate mask and SCORE output for one
public feature gate, NOT hardcoded Team18/29 rules:
  if exact13 C copy fraction >= cutoff:
     use P08 strict-decontaminated P01 if residual >=10 rows,
     else fall back to P01 original with unmodified C
  else retain P04's chosen KDE-A/KDE-B/P01 full-target TPR.

The method family & cutoff 50/80/90% thresholds are posthoc choices on same
released preliminary truth, so no independent performance claim.
Official prelim old rare MIA + AIA, old Team1/2 MIA full scores preserved,
Team12 replaced with P01 baseline. All comparisons are synthetic cohort-
metric composites; NO per-person submission CSV, no ability to validate
cross-method score covariance or true prospective generalization.

Outer leave-one-target-out threshold selection uses 16 other cohorts'
released PRELIM TPR, never held-out outcome; source panel had been selected
based on prior publicly released truth, so still selection-biased.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT="攻撃フェーズ/Tashiro/MIA全体/experiments/"
P08=ROOT+"20261009_P08_公開残差と時刻条件付/results/"
P04=ROOT+"20261009_TPR010到達と加工特徴転移検証/p04_family_split/five_public_rules_by_team.csv"
OFFICIAL="公式データ/予備戦/結果/公開用結果表_20260909.xlsx"
MAIN17=(3,4,6,7,8,9,10,13,17,18,19,20,22,23,24,26,29)
THRESHOLDS=(.5,.8,.9)
THIRD_SCORE=.352845805596271


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inputs(root):
    copies=pd.read_csv(root/(P08+"public_copy_cohorts.csv")).set_index("team")
    details=pd.read_csv(root/(P08+"by_target.csv"))
    panel=details[(details.inference=="all20")&(details.background=="full")]
    kp=panel.pivot(index="team",columns="method",values="tpr_0.01")
    p04=pd.read_csv(root/P04)
    p04=p04[p04.policy=="split_AhomeHigh_BhomeLow"].set_index("team")
    assert len(p04)==17 and set(MAIN17)==set(p04.index)
    assert len(kp)==20
    assert int(copies.loc[18,"full_public_A_copy_rows"])==1049
    assert int(copies.loc[29,"full_public_A_copy_rows"])==885
    wb=load_workbook(root/OFFICIAL,read_only=True,data_only=True)
    old={};aia={}
    for row in list(wb["攻撃_標的別_MIA"].values)[1:]:
        if row[0]==5:old[int(row[1])]=(float(row[4] or 0),float(row[5] or 0))
    for row in list(wb["攻撃_標的別_AIA"].values)[1:]:
        if row[0]==5:aia[int(row[1])]=float(row[9] or 0)
    assert len(old)==20 and set(aia)==set(old)
    off=sum(max(*old[t])+aia[t] for t in old)/20
    assert abs(off-.3297953270354882)<1e-12
    return copies,kp,p04,old,aia,off


def replacement(t,cut,copies,kp,p04):
    base=float(p04.loc[t,"full_tpr"])
    share=float(copies.loc[t,"full_public_A_copy_rows"])/float(copies.loc[t,"C_rows"])
    if cut is None or share<cut:return base,"P04_untouched"
    if int(copies.loc[t,"strict_fallback"]):
        return float(kp.loc[t,"P01"]),"fallback_P01_all_C_public"
    return float(kp.loc[t,"A_full_public_remove"]),"P08_A_public_residual_P01"


def aggregate(assign, kp, old, aia):
    values={t:(old[t][0] if t in (1,2) else
                float(kp.loc[t,"P01"]) if t==12 else assign[t]) for t in old}
    power=sum(max(v,old[t][1])+aia[t] for t,v in values.items())/20
    return power,values


def evaluate(root,out):
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    copies,kp,p04,old,aia,official=inputs(root)
    detail=[];sumrows=[]
    candidates=(("P04",None),("P08_A_majority50",.5),
                ("P08_A_majority80",.8),("P08_A_high90",.9))
    for name,cut in candidates:
        chosen={}
        for t in MAIN17:
            result,reason=replacement(t,cut,copies,kp,p04)
            chosen[t]=result
            detail.append({
              "policy":name,"team":t,"P04_tpr":float(p04.loc[t,"full_tpr"]),
              "P08_A_input_p01":float(kp.loc[t,"A_full_public_remove"]),
              "assigned_tpr":result,"delta_vs_P04":result-float(p04.loc[t,"full_tpr"]),
              "copy_share":float(copies.loc[t,"full_public_A_copy_rows"])/float(copies.loc[t,"C_rows"]),
              "public_residual_rows":int(copies.loc[t,"noncopy_rows"]),
              "fallback_due_to_empty_residual":int(copies.loc[t,"strict_fallback"]),
              "rule":reason
            })
        score,m=aggregate(chosen,kp,old,aia)
        sumrows.append({
          "policy":name,"public_match_fraction_cutoff":cut,
          "hypothetical_prelim_attack":score,
          "improvement_over_P04":score-.34919703643719757,
          "gap_to_official_3rd":THIRD_SCORE-score,
          "would_beat_official_3rd":int(score>THIRD_SCORE),
          "TPR_ge010_20":int(sum(v>=.1-1e-12 for v in m.values())),
          "changed_vs_P04":int(sum(abs(chosen[t]-float(p04.loc[t,"full_tpr"]))>1e-12 for t in MAIN17))
        })
    summary=pd.DataFrame(sumrows)
    df=pd.DataFrame(detail)
    summary.to_csv(out/"overlay_counterfactual.csv",index=False,float_format="%.17g")
    df.to_csv(out/"overlay_by_target.csv",index=False,float_format="%.17g")
    assert abs(summary.loc[summary.policy=="P04","hypothetical_prelim_attack"].iloc[0]-.34919703643719757)<1e-12
    assert abs(summary.loc[summary.policy=="P08_A_majority50","hypothetical_prelim_attack"].iloc[0]-.35312011336027455)<1e-12
    # Leave one out threshold CHOICE. Conservatively prefer largest cutoff
    # on ties. For an excluded target, do not access its TPR while choosing.
    outer=[]
    training_per_target={t:df[(df.team==t)] for t in MAIN17}
    for t in MAIN17:
        scored=[]
        for name,cut in candidates:
            # On the 16 training cohorts, rank by gain over P04,
            # reject any TPR 0.10 loss; choose largest cutoff if tie.
            training=df[(df.team!=t)&(df.policy==name)]
            old_hits=int(sum(training.P04_tpr>=.1-1e-12))
            new_hits=int(sum(training.assigned_tpr>=.1-1e-12))
            criterion=(int(new_hits-old_hits),float(training.delta_vs_P04.mean()),
                       -int((training.delta_vs_P04<-1e-12).sum()),
                       (2. if cut is None else cut))
            scored.append((criterion,name,cut))
        criterion,selected,cut=max(scored,key=lambda z:z[0])
        val,reason=replacement(t,cut,copies,kp,p04)
        oldval=float(p04.loc[t,"full_tpr"])
        outer.append({
          "heldout_team":t,"rule_selected_without_heldout_TPR":selected,
          "public_copy_share":float(copies.loc[t,"full_public_A_copy_rows"])/float(copies.loc[t,"C_rows"]),
          "heldout_P04_TPR":oldval,"heldout_P08_TPR":val,
          "heldout_delta":val-oldval,"rule":reason,
        })
    looto=pd.DataFrame(outer)
    looto.to_csv(out/"LOOTO_threshold_choice.csv",index=False,float_format="%.17g")
    chosen={int(z.heldout_team):float(z.heldout_P08_TPR) for z in looto.itertuples()}
    loo_score,_=aggregate(chosen,kp,old,aia)
    diagnostics={
        "scope":"20 target official preliminary; fixed old Team1/2, AIA and rare; Team12=P01",
        "status":"POST-HOC published prelim, NOT independent prospective performance",
        "P08_A_majority_cutoffs":THRESHOLDS,
        "reference_public_only":"official A_bg full13 exact row matching",
        "source_TPR_score_scope":"P08 infer20 full / P04 archived full; no per-record hybrid output",
        "new_method_construction_does_not_select_specific_team_ids":True,
        "historical_success_target_selection_bias":True,
        "full_new_TPR010_target_count":int(summary[summary.policy=="P08_A_majority50"].TPR_ge010_20.item()),
        "outer_LOOTO_selection":"choose cutoff among P04,.5,.8,.9 based on other16 TPR, prefer largest cutoff on ties",
        "LOOTO_counterfactual_attack":loo_score,
        "LOOTO_improvement_over_P04":loo_score-.34919703643719757,
        "LOOTO_gap_third":THIRD_SCORE-loo_score,
        "LOOTO_heldout_improved_teams":looto[looto.heldout_delta>1e-12].heldout_team.astype(int).tolist(),
        "LOOTO_rule_counts":looto.rule_selected_without_heldout_TPR.value_counts().to_dict(),
        "source_sha256":{p:sha(root/p) for p in
               [P08+"by_target.csv",P08+"public_copy_cohorts.csv",P04,OFFICIAL]},
        "actual_main_TPR_unobserved":True,"no_main_submission":True,
        "rare_5bit_unchanged":True,
        "not_same_background_P04_and_P08_4views":True,
    }
    (out/"LOOTO_summary.json").write_text(json.dumps(diagnostics,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("P08-A / P04 public-match overlay comparison")
    print(summary.to_string(index=False))
    print("LOOTO",json.dumps({k:diagnostics[k] for k in [
      "LOOTO_counterfactual_attack","LOOTO_improvement_over_P04",
      "LOOTO_heldout_improved_teams","LOOTO_gap_third"]},ensure_ascii=False))
    print(df[(df.policy=="P08_A_majority50")&df.delta_vs_P04.ne(0)].to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    v=p.parse_args()
    evaluate(v.root.resolve(),v.out.resolve())
