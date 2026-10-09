"""Nested target-holdout test of archived MIA method transfer for TPR >= 0.10.

Outer target t is NEVER queried by the model/policy selection functions.
All target-truth-sensitive selections are nested inside the remaining sixteen.
Candidate method panel was curated on the same preliminary competition truth in
earlier research: this is exploratory nested CV, NOT independent validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

import run_transfer as legacy

TGT=legacy.TARGET_SET
BASELINE=legacy.BASELINE
KS=(1,3,5,7,9,16)
KINDS=("hit","robust","gain","mean","hit_plus_robust","shrunken")
SCOPES=("all","high_only")
THRESH=.10
SOURCE="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/public_target_groups.csv"
METHOD_SPECS=[(scope,k,kind) for scope in SCOPES for k in KS for kind in KINDS]


def get_group(root):
    p=root/SOURCE
    df=pd.read_csv(p)
    high=set(df.loc[df.aia_imprint_high==1,"team"])
    return [int(t in high) for t in TGT],p


def selector(tensor,method_names,views,dist,high_flags,query,train_idxs,spec):
    """Return choice: NEVER read tensor[query]."""
    scope,k,kind=spec
    base=method_names.index(BASELINE)
    if scope=="high_only" and not high_flags[query]:
        return base
    avail=[int(i) for i in train_idxs if int(i)!=query and (
        scope=="all" or high_flags[int(i)])]
    if not avail:
        return base
    distances=dist.iloc[query].to_numpy(float)
    peers=sorted(avail,key=lambda i:(distances[i],i))[:k]
    full=views.index("full")
    mat=tensor[peers,:,:]
    scores=mat[:,:,full]  # peer x method
    hits=(scores>=THRESH-1e-12).sum(axis=0)
    robust=(mat>=THRESH-1e-12).all(axis=2).sum(axis=0)
    ref=scores[:,base]
    gains=scores-ref[:,None]
    mu=scores.mean(axis=0)
    gained=(gains>1e-12).sum(axis=0)
    if kind=="hit":
        value=2*hits+robust+0.01*mu
    elif kind=="robust":
        value=2*robust+hits+0.01*mu
    elif kind=="gain":
        value=gains.mean(axis=0)+0.01*gained+1e-4*mu
    elif kind=="mean":
        value=mu
    elif kind=="hit_plus_robust":
        value=hits+0.35*robust+0.01*mu
    elif kind=="shrunken":
        denom=len(peers)+3
        value=hits/denom+0.4*robust/denom+0.07*mu
    else:
        raise ValueError(kind)
    # A baseline tie is resolved conservatively.
    top=float(np.max(value))
    if abs(value[base]-top)<1e-12:
        return base
    return int(np.where(np.abs(value-top)<1e-12)[0][0])


def heldout_predictions(tensor,methods,views,dist,group,outer_train,spec):
    """All predictions for team indices in outer_train using their other 15 peers."""
    base=methods.index(BASELINE)
    v=views.index("full")
    picks={}
    for held in outer_train:
        known=[x for x in outer_train if x!=held]
        j=selector(tensor,methods,views,dist,group,held,known,spec)
        picks[held]=j
    hits=sum(tensor[t,j,v]>=THRESH-1e-12 for t,j in picks.items())
    stable=sum((tensor[t,j,:]>=THRESH-1e-12).all() for t,j in picks.items())
    lost=sum(tensor[t,base,v]>=THRESH-1e-12 and
             tensor[t,j,v]<THRESH-1e-12 for t,j in picks.items())
    changed=sum(j!=base for j in picks.values())
    avg=np.mean([tensor[t,j,v] for t,j in picks.items()])
    return dict(
      hits=int(hits),stable=int(stable),lost=int(lost),
      changed=int(changed),mean=float(avg),picks=picks
    )


def nested_outer_select(tensor,methods,views,dist,groups,outer):
    train=[i for i in range(len(TGT)) if i!=outer]
    scored=[]
    for spec in METHOD_SPECS:
        result=heldout_predictions(tensor,methods,views,dist,groups,train,spec)
        # Lexicographic rule frozen before evaluating any outer fold.
        quality=(result["stable"],result["hits"],-result["lost"],
                 result["mean"],-result["changed"])
        scored.append((quality,spec,result))
    # Fixed lexical tie-breaking; selecting among candidate models uses no
    # scored outcome for 'outer'.
    best=max(scored,key=lambda x:x[0])
    return best[1],best[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    root,out=a.root.resolve(),a.out.resolve()
    if out.exists():
        raise FileExistsError("Output directory exists: "+str(out))
    out.mkdir(parents=True)
    feat,records,methods,views,source_hash=legacy.read_inputs(root)
    distances,dist_spec=legacy.public_distances(feat)
    high,high_file=get_group(root)
    tensor=legacy.make_tensor(records,methods,views)
    v=views.index("full")
    base=methods.index(BASELINE)
    records_out=[]
    for index,team in enumerate(TGT):
        spec,ins=nested_outer_select(
            tensor,methods,views,distances,high,index)
        method_index=selector(
            tensor,methods,views,distances,high,index,
            [j for j in range(len(TGT)) if j!=index],spec)
        values=tensor[index,method_index]
        baseline=tensor[index,base]
        records_out.append({
            "team":team,"high_AIA_group":high[index],
            "selected_policy_scope":spec[0],
            "selected_policy_k":spec[1],
            "selected_policy_kind":spec[2],
            "selected_method":methods[method_index],
            "full_tpr":float(values[v]),
            "P01_full_tpr":float(baseline[v]),
            "new_ge010":int(values[v]>=THRESH-1e-12 and baseline[v]<THRESH-1e-12),
            "lost_ge010":int(values[v]<THRESH-1e-12 and baseline[v]>=THRESH-1e-12),
            "full_ge010":int(values[v]>=THRESH-1e-12),
            "full_delta_P01":float(values[v]-baseline[v]),
            "all7_ge010":int((values>=THRESH-1e-12).all()),
            "best_nested_train_16_hits":ins["hits"],
            "best_nested_train_16_stable":ins["stable"],
            "best_nested_train_16_mean_tpr":ins["mean"],
            "known_peer_count":16,
        })
    df=pd.DataFrame(records_out)
    df.to_csv(out/"nested_target_holdout.csv",index=False,float_format="%.17g")
    # One table of 72 fixed-policy screening results, but NOT used to
    # reselect policies for the nested folds.
    policy_rows=[]
    for spec in METHOD_SPECS:
        scores=[]
        for i in range(len(TGT)):
            peer=[j for j in range(len(TGT)) if j!=i]
            j=selector(tensor,methods,views,distances,high,i,peer,spec)
            scores.append((i,j))
        hits=sum(tensor[i,j,v]>=THRESH-1e-12 for i,j in scores)
        stable=sum((tensor[i,j,:]>=THRESH-1e-12).all() for i,j in scores)
        lost=sum(tensor[i,base,v]>=THRESH-1e-12 and
            tensor[i,j,v]<THRESH-1e-12 for i,j in scores)
        new=sum(tensor[i,base,v]<THRESH-1e-12 and
            tensor[i,j,v]>=THRESH-1e-12 for i,j in scores)
        policy_rows.append({
            "scope":spec[0],"k":spec[1],"kind":spec[2],
            "hit_ge010":int(hits),"stable_all7_ge010":int(stable),
            "new_hits":int(new),"lost_hits":int(lost),
            "mean_full_tpr":float(np.mean([tensor[i,j,v] for i,j in scores])),
            "changed_methods":int(sum(j!=base for i,j in scores))
        })
    pd.DataFrame(policy_rows).to_csv(out/"policy_screen_72.csv",
                                   index=False,float_format="%.17g")
    summary={
        "baseline":{"full_ge010":int((tensor[:,base,v]>=THRESH-1e-12).sum()),
                    "stable_all7":int((tensor[:,base,:]>=THRESH-1e-12).all(axis=1).sum()),
                    "full_mean_tpr":float(tensor[:,base,v].mean())},
        "nested":{
            "full_ge010":int(df.full_ge010.sum()),
            "stable_all7_ge010":int(df.all7_ge010.sum()),
            "new_hits":int(df.new_ge010.sum()),"lost_hits":int(df.lost_ge010.sum()),
            "mean_full_tpr":float(df.full_tpr.mean()),
            "improved_teams":int((df.full_delta_P01>1e-12).sum()),
            "worsened_teams":int((df.full_delta_P01< -1e-12).sum()),
            "unchanged_teams":int((abs(df.full_delta_P01)<=1e-12).sum()),
            "selected_method_counts":df.selected_method.value_counts().to_dict(),
            "selected_policy_counts":df[["selected_policy_scope","selected_policy_k","selected_policy_kind"]].value_counts().to_dict() if False else [
                 {"scope":sc,"k":int(k),"kind":ki,"count":len(group)}
                 for (sc,k,ki),group in df.groupby(
                     ["selected_policy_scope","selected_policy_k","selected_policy_kind"])
            ],
        }
    }
    assert len(df)==17 and summary["baseline"]["full_ge010"]==2
    (out/"summary.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    audit={
        "dataset":"Published preliminary 17 target x 22 fixed methods x 7 backgrounds",
        "method_panel_preselected_using_exposed_preliminary_truth":True,
        "outer_target_label_used_for_training_or_hyperparameter_selection":False,
        "public_features_transductive_across_targets":True,
        "inner_CV_training_target_count":16,
        "inner_choice_heldout_fold_count":16,
        "inner_selection_criterion":"max stable_hits, full_hits, -lost_hits, mean, -method_changes",
        "policy_candidates":len(METHOD_SPECS),
        "FPR":.01,"TPR_success_threshold":THRESH,
        "background_views":views,
        "team1_2_12_excluded":True,
        "target_ids":TGT,"high_public_group":sorted(t for t,vv in zip(TGT,high) if vv),
        "input_sha256":{**source_hash,str(high_file.relative_to(root)):legacy.hashfile(high_file)},
        "no_main_data":True,"no_submission":True,
        "caveat":"Outer LOOTO does not repair historical reuse of same preliminary labels for method-family selection"
    }
    (out/"protocol.json").write_text(
        json.dumps(audit,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("nested summary",json.dumps(summary,ensure_ascii=False))
    print(df[["team","selected_method","full_tpr","full_ge010",
              "new_ge010","lost_ge010"]].to_string(index=False))
    print("out",out)


if __name__=="__main__":
    main()
