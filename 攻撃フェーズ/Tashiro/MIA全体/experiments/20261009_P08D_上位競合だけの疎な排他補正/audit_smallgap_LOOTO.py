"""Post-hoc leave-one-target-out audit of sparse P08D score-family choice.

Choose one GLOBAL K/margin/penalty method using 17 other preliminary
target cohorts only, evaluate the 18th target's 4-view frozen score results.
This is not independent external validation: the entire candidate family was
devised in a historically label-exposed preliminary contest. The "small gap"
panel followed an earlier first panel that showed no TPR changes.
"""
from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

SRC="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_P08D_上位競合だけの疎な排他補正/small_gap/by_target.csv"
BG=("full","sub80_seed20261009","sub80_seed20261010","sub80_seed20261011")
MODES=("P01","strict_TPR010","four_bg_guard","best_full_mean")
THRESH=.10


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_panel(root):
    df=pd.read_csv(root/SRC)
    df=df[df.inference=="without1_2"].copy()
    assert len(df)==4*18*19
    assert df.team.nunique()==18 and df.method.nunique()==19
    assert set(df.background)==set(BG)
    return df


def choose(train,mode,methods):
    """Train is prefiltered to 17 OTHER target rows, never the heldout target."""
    if mode=="P01":return "P01",{"selected_on":"fixed_P01"}
    wide=train.pivot_table(index=["team","background"],columns="method",values="tpr_0.01",aggfunc="first")
    teams=sorted(train.team.unique())
    assert len(teams)==17
    full=wide.xs("full",level="background").loc[teams]
    baseline=full["P01"]
    stable=[]
    for method in methods:
        curr=full[method]
        new=int(((curr>=THRESH-1e-12)&(baseline<THRESH-1e-12)).sum())
        lost=int(((curr<THRESH-1e-12)&(baseline>=THRESH-1e-12)).sum())
        gained=int((curr-baseline>1e-12).sum())
        delta=float((curr-baseline).mean())
        all_delta=[float((wide.xs(bg,level="background")[method]-
                          wide.xs(bg,level="background")["P01"]).mean()) for bg in BG]
        stable.append((method,new,lost,gained,delta,min(all_delta),np.mean(all_delta)))
    allowed=[]
    for method,new,lost,gained,full_delta,worst_bg,mean_bg in stable:
        if method=="P01":continue
        if mode=="strict_TPR010":
            valid=(new>=1 and lost==0)
            quality=(new,mean_bg,worst_bg,full_delta)
        elif mode=="four_bg_guard":
            valid=(worst_bg>=-1e-12 and gained>=2 and lost==0 and mean_bg>=.001)
            quality=(worst_bg,mean_bg,gained,full_delta)
        elif mode=="best_full_mean":
            valid=full_delta>1e-12
            quality=(full_delta,mean_bg,-lost,gained)
        else:raise ValueError(mode)
        if valid:allowed.append((quality,method,new,lost,mean_bg,worst_bg))
    if not allowed:return "P01",{"selected_on":"no_valid_peer_improvement"}
    # stable rule tie resolution preserves lexical method order, independent
    # of heldout outcome.
    best=max(allowed,key=lambda v:(v[0],-methods.index(v[1])))
    return best[1],{"selected_on":"other17_only",
          "peer_new_hits":best[2],"peer_lost_hits":best[3],
          "peer_mean4_delta":best[4],"peer_worst_background_mean_delta":best[5]}


def run(root,out):
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    df=source_panel(root)
    methods=list(df.method.drop_duplicates())
    assert methods[0]=="P01",methods[0]
    cohorts=sorted(df.team.unique())
    out_rows=[]
    for mode in MODES:
        for test in cohorts:
            other=df[df.team!=test]
            method,meta=choose(other,mode,methods)
            assert len(other.team.unique())==17
            test_rows=df[(df.team==test)&(df.method==method)].set_index("background")
            baseline=df[(df.team==test)&(df.method=="P01")].set_index("background")
            assert set(test_rows.index)==set(BG)
            selected=float(test_rows.loc["full","tpr_0.01"])
            before=float(baseline.loc["full","tpr_0.01"])
            score4=np.array([test_rows.loc[bg,"tpr_0.01"] for bg in BG])
            base4=np.array([baseline.loc[bg,"tpr_0.01"] for bg in BG])
            out_rows.append({
                "policy":mode,"target":test,
                "model_chosen_without_target_truth":method,
                "heldout_full_TPR":selected,
                "heldout_P01_TPR":before,
                "delta_TPR":selected-before,
                "new_TPR010":int(selected>=THRESH-1e-12 and before<THRESH-1e-12),
                "lost_TPR010":int(selected<THRESH-1e-12 and before>=THRESH-1e-12),
                "all4_ge010":int((score4>=THRESH-1e-12).all()),
                "worst4_delta":float((score4-base4).min()),
                "mean4_delta":float((score4-base4).mean()),
                "peer_validated":int(method!="P01"),
                **meta
            })
    output=pd.DataFrame(out_rows)
    output.to_csv(out/"LOOTO_by_target.csv",index=False,float_format="%.17g")
    sums=[]
    for mode,t in output.groupby("policy",sort=False):
        sums.append({
          "policy":mode,"targets":len(t),
          "TPR010_hits":int((t.heldout_full_TPR>=THRESH-1e-12).sum()),
          "new_TPR010_hits":int(t.new_TPR010.sum()),
          "lost_TPR010_hits":int(t.lost_TPR010.sum()),
          "mean_full_TPR":float(t.heldout_full_TPR.mean()),
          "mean_delta_P01":float(t.delta_TPR.mean()),
          "improved_targets":int((t.delta_TPR>1e-12).sum()),
          "worsened_targets":int((t.delta_TPR<-1e-12).sum()),
          "targets_with_changed_method":int(t.peer_validated.sum()),
          "methods_selected":";".join(f"{key}:{count}" for key,count in t.model_chosen_without_target_truth.value_counts().items()),
        })
    summary=pd.DataFrame(sums)
    assert abs(summary[summary.policy=="P01"].mean_full_TPR.item()-.025495409939854385)<1e-12
    summary.to_csv(out/"LOOTO_summary.csv",index=False,float_format="%.17g")
    (out/"protocol.json").write_text(json.dumps({
      "scope":"official preliminary P01/P08D small-gap frozen method panel; 18 target cohorts excluding 1,2",
      "policy_candidates":MODES,
      "input_method_count":len(methods),
      "inner_selection":"For heldout target t, inspect other17 released preliminary full and 4 background score rows only",
      "inner_holdout_target_label_access":False,
      "P08D_family_still_devised_after_repeated_preliminary_label_exposure":True,
      "backgrounds_not_independent":True,
      "no_new_predictions":True,
      "not_independent_external_validation":True,
      "no_main_submission":True,
      "source_sha256":{SRC:sha(root/SRC)}
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(summary.to_string(index=False))
    print("Chosen methods outside P01")
    print(output[output.model_chosen_without_target_truth!="P01"][
       ["policy","target","model_chosen_without_target_truth",
        "heldout_full_TPR","heldout_P01_TPR","new_TPR010","lost_TPR010"]].to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.root.resolve(),a.out.resolve())
