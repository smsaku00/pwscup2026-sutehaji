"""Conservative nested LOOTO selection with an inner threshold-hit safety filter.

An outer cohort is not used for method or policy selection. The safety filter
accepts a policy on the other sixteen only if inner CV preserves old P01 hits
and adds a newly successful target. A relaxed variant allows loss of one old
hit only with >=2 new targets. Based on earlier exploratory results, NOT
independent validation. Prevents a mean-TPR criterion from eliminating proven
P01 successes merely to improve weak 0.01 background scores.
"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

import run_transfer as old
import run_nested_holdout as nested


def get_data(root):
    feat,tab,methods,views,hashes=old.read_inputs(root)
    d,_=old.public_distances(feat)
    tensor=old.make_tensor(tab,methods,views)
    high,_=nested.get_group(root)
    return methods,views,d,tensor,high,hashes


def choose_safely(tensor,methods,views,dist,high,outer,risk_budget):
    """All calculations depend only on other sixteen target score rows."""
    train=[i for i in range(len(nested.TGT)) if i!=outer]
    base=methods.index(nested.BASELINE)
    full=views.index("full")
    baseline_hit=sum(tensor[t,base,full]>=.1-1e-12 for t in train)
    baseline_stable=sum((tensor[t,base,:]>=.1-1e-12).all() for t in train)
    selected=None
    for spec in nested.METHOD_SPECS:
        fit=nested.heldout_predictions(tensor,methods,views,dist,high,train,spec)
        new_hits=fit["hits"]-baseline_hit+fit["lost"]
        if fit["lost"]>risk_budget or new_hits<=risk_budget:
            continue
        if fit["hits"]<=baseline_hit:
            continue
        value=(fit["hits"]-baseline_hit,
               fit["stable"]-baseline_stable,
               -fit["lost"],fit["mean"],-fit["changed"])
        if selected is None or value>selected[0]:
            selected=(value,spec,fit)
    if selected is None:
        return base,None,baseline_hit
    j=nested.selector(tensor,methods,views,dist,high,outer,train,selected[1])
    return j,selected[1],baseline_hit


def run(root,out):
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    methods,views,d,tensor,high,hashes=get_data(root)
    baseline=methods.index(nested.BASELINE)
    full=views.index("full")
    metrics=[]
    for budget in (0,1):
        for i,team in enumerate(nested.TGT):
            j,spec,training_hit=choose_safely(tensor,methods,views,d,high,i,budget)
            v=tensor[i,j,:]
            b=tensor[i,baseline,:]
            metrics.append({
                "team":team,"safety_budget":budget,
                "selected_method":methods[j],
                "selected_policy": "/".join(map(str,spec)) if spec is not None else "safe_fallback_P01",
                "train16_baseline_success_count":int(training_hit),
                "full_tpr":float(v[full]),
                "baseline_tpr":float(b[full]),
                "full_ge010":int(v[full]>=.1-1e-12),
                "P01_full_ge010":int(b[full]>=.1-1e-12),
                "new_ge010":int(v[full]>=.1-1e-12 and b[full]<.1-1e-12),
                "lost_ge010":int(v[full]<.1-1e-12 and b[full]>=.1-1e-12),
                "stable_all7":int((v>=.1-1e-12).all()),
                "delta_P01":float(v[full]-b[full]),
            })
    df=pd.DataFrame(metrics)
    summary=[]
    for budget,p in df.groupby("safety_budget",sort=False):
        summary.append({
            "safety_budget":budget,"total":len(p),
            "full_ge010":int(p.full_ge010.sum()),
            "new_ge010":int(p.new_ge010.sum()),
            "lost_ge010":int(p.lost_ge010.sum()),
            "stable_all7":int(p.stable_all7.sum()),
            "mean_full_tpr":float(p.full_tpr.mean()),
            "wins":int((p.delta_P01>1e-12).sum()),
            "losses":int((p.delta_P01< -1e-12).sum()),
            "P01_fallback_teams":int(p.selected_policy.eq("safe_fallback_P01").sum()),
        })
    assert len(df)==34 and (df.P01_full_ge010.sum()==4)
    df.to_csv(out/"safe_nested_by_target.csv",index=False,float_format="%.17g")
    pd.DataFrame(summary).to_csv(out/"safe_nested_summary.csv",index=False,float_format="%.17g")
    (out/"protocol.json").write_text(json.dumps({
        "status":"post-hoc conservative nested target holdout within previously explored preliminary panel",
        "train_set":"outer target excluded entirely for model/policy/hyperparameter selection",
        "candidate_policies":len(nested.METHOD_SPECS),
        "risk_budgets":[0,1],
        "inner_safety":"require >baseline inner full0.10 hits and new hits > budget, losses <= budget",
        "score_panel_already_tuned_on_preliminary_labels":True,
        "source_sha256":hashes,"target_ids":nested.TGT,
        "main_submission":False
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    print(df.to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    run(a.root.resolve(),a.out.resolve())
