"""Assess 0.10 threshold and cross-target transfer using archived non-training MIA methods.

Input: 22 fixed non-learning algorithms x seven reference-backgrounds x 17 teams,
plus public C processing features. Models are chosen on *other* cohort outcomes
only for LOOTO, never from the evaluation cohort's membership labels.
The existing method panel itself was historically selected after inspecting
the preliminary labels: results are exploratory, NOT an independent validation.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

SUBDIR="攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_非学習方式と加工特徴_背景耐性"
TARGET_SET=[3,4,6,7,8,9,10,13,17,18,19,20,22,23,24,26,29]
BASELINE="Q_prefecture_sex_shrink500_cross01"
METHODS_N=22
THRESHOLD=0.10
BG_N=7
# Public features. Hand-selected from previously documented processing hypotheses.
# No target membership labels or evaluation scores enter distances or standardization.
FEATURES=[
  "numeric7_mean_abs_spearman","QI5_mean_abs_spearman",
  "Q_sexmatched_QI5_RMS_median","A1_sexmatched_QI5_RMS_median",
  "Q_sexmatched_QI5_RMS_lt005","Q_sexmatched_QI5_RMS_lt001",
  "public_A_clinical6_joint_rows","public_A_full13_rows","A2_full13_rows",
  "A2_except_prefecture_rows","full13_duplicates","BMI_unique",
  "TG_unique","ALT_unique","time_unique",
  "BMI_public_value_fraction","FPG_public_value_fraction",
]
LOG_FEATURES={
  "Q_sexmatched_QI5_RMS_lt005","Q_sexmatched_QI5_RMS_lt001",
  "public_A_clinical6_joint_rows","public_A_full13_rows",
  "A2_full13_rows","A2_except_prefecture_rows","full13_duplicates",
  "BMI_unique","TG_unique","ALT_unique","time_unique"
}
# Models fixed before running the held-out team score comparisons.
POLICIES=[
  {"name":"nearest3_fullsuccess","neighbors":3,"criterion":"full_hit"},
  {"name":"nearest5_fullsuccess","neighbors":5,"criterion":"full_hit"},
  {"name":"nearest7_fullsuccess","neighbors":7,"criterion":"full_hit"},
  {"name":"nearest3_stable","neighbors":3,"criterion":"stable_hit"},
  {"name":"nearest5_stable","neighbors":5,"criterion":"stable_hit"},
  {"name":"nearest7_stable","neighbors":7,"criterion":"stable_hit"},
  {"name":"nearest3_mean_gain","neighbors":3,"criterion":"gain"},
  {"name":"nearest5_mean_gain","neighbors":5,"criterion":"gain"},
  {"name":"nearest7_mean_gain","neighbors":7,"criterion":"gain"},
  {"name":"random5_fullsuccess","neighbors":5,"criterion":"full_hit","shuffle_neighbors":True}
]


def hashfile(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def read_inputs(root):
    base=root/SUBDIR
    features=base/"公開加工特徴.csv"
    scores=base/"固定方式_全背景評価.csv"
    feat=pd.read_csv(features,encoding="utf-8-sig")
    tbl=pd.read_csv(scores,encoding="utf-8-sig")
    tbl=tbl[np.isclose(tbl.fpr,0.01) & tbl.team.isin(TARGET_SET)].copy()
    assert len(feat[feat.team.isin(TARGET_SET)])==len(TARGET_SET)
    assert tbl.groupby(["team","method","condition"]).size().eq(1).all()
    methods=sorted(tbl.method.unique())
    backgrounds=sorted(tbl.condition.unique())
    assert len(methods)==METHODS_N,(len(methods),methods)
    assert len(backgrounds)==BG_N,backgrounds
    assert "full" in backgrounds and BASELINE in methods
    assert len(tbl)==len(TARGET_SET)*METHODS_N*BG_N
    return feat, tbl, methods, backgrounds, {str(p.relative_to(root)):hashfile(p) for p in [features,scores]}


def public_distances(feat):
    f=feat.set_index("team").loc[TARGET_SET,FEATURES].astype(float).copy()
    for col in FEATURES:
        if col in LOG_FEATURES:
            f[col]=np.log1p(np.maximum(0,f[col]))
    a=f.to_numpy(dtype=float)
    assert np.isfinite(a).all()
    # No label-dependent feature weighting.
    mean=a.mean(axis=0)
    std=a.std(axis=0,ddof=0)
    a=(a-mean)/np.where(std>1e-9,std,1)
    # Avoid almost-identical feature dimensions making an extreme one dominate.
    a=np.clip(a,-3.0,3.0)
    d=np.sqrt(((a[:,None,:]-a[None,:,:])**2).mean(axis=2))
    np.fill_diagonal(d,np.inf)
    return pd.DataFrame(d,index=TARGET_SET,columns=TARGET_SET),{
      "feature_names":FEATURES,
      "log1p_features":sorted(LOG_FEATURES),
      "distance":"Euclidean squared z-score mean over 17 features; clip z to ±3",
      "public_feature_standardization":"all 17 unlabeled target metadata; no truth",
      "identical_zero_variance_feature":"set scale 1",
    }


def make_tensor(scores,methods,backgrounds):
    pivot=scores.pivot_table(index="team",columns=["method","condition"],values="tpr",aggfunc="first")
    pivot=pivot.loc[TARGET_SET]
    tensor=np.stack([
      np.stack([pivot[(m,bg)].to_numpy(float) for bg in backgrounds],axis=1)
      for m in methods],axis=1)
    assert tensor.shape==(len(TARGET_SET),len(methods),len(backgrounds))
    assert np.all(np.isfinite(tensor))
    return tensor


def neighbor_indices(d, test_index, k, randomized=False):
    values=d.iloc[test_index].to_numpy()
    ids=np.argsort(values,kind="stable")[:k]
    if not randomized:
        return ids
    # Use a deterministic permutation with no labels or test-specific outcomes.
    r=np.random.default_rng(20261009+test_index).permutation(len(TARGET_SET))
    return [j for j in r if j!=test_index][:k]


def selected_method(tensor, methods, backgrounds, neighbors, criterion):
    bi=methods.index(BASELINE)
    ix=backgrounds.index("full")
    train=tensor[neighbors]  # shape (k,methods,bg)
    ref=train[:,bi,:]
    full=train[:,:,ix]
    good_full=full>=THRESHOLD-1e-12
    # Per-neighbor method stable if >=.10 in 7/7 backgrounds.
    stable=(train>=THRESHOLD-1e-12).all(axis=2)
    gained=(train[:,:,ix]-ref[:,ix,None]) if False else (full - ref[:,ix][:,None])

    # Avoid accepting an idiosyncratic one-neighbor method.
    # Require >=2 neighboring cohort successes for threshold-oriented methods.
    num_full=good_full.sum(axis=0)
    num_stable=stable.sum(axis=0)
    num_gain=(gained>1e-12).sum(axis=0)
    if criterion=="full_hit":
        valid=num_full>=2
        objective=2.0*num_full+num_stable+0.01*full.mean(axis=0)
    elif criterion=="stable_hit":
        valid=num_stable>=2
        objective=2.0*num_stable+num_full+0.01*full.mean(axis=0)
    elif criterion=="gain":
        valid=(num_gain>=2)&(gained.mean(axis=0)>0)
        objective=gained.mean(axis=0)+0.01*num_gain+0.0001*full.mean(axis=0)
    else:
        raise ValueError(criterion)
    valid[bi]=True
    if criterion=="gain":
        objective[bi]=0
    # For threshold choices, prefer baseline in ties.
    baseline_min=objective[bi]
    for j in range(len(methods)):
        if not valid[j] or j==bi:
            continue
        if objective[j]<=baseline_min+1e-12:
            valid[j]=False
    if not valid.any():return bi
    scores=np.where(valid,objective,-np.inf)
    return int(np.argmax(scores))


def evaluation_table(scores, feats, methods, backgrounds, tensor, d):
    bi=methods.index(BASELINE)
    fi=backgrounds.index("full")
    per=[]
    all_specs=[{"name":"P01_fixed","neighbors":0,"criterion":"baseline"}]+POLICIES
    for spec in all_specs:
        policy=spec["name"]
        for i,team in enumerate(TARGET_SET):
            if policy=="P01_fixed":j=bi; train_ids=[]
            else:
                neighbors=neighbor_indices(d,i,spec["neighbors"],spec.get("shuffle_neighbors",False))
                train_ids=[TARGET_SET[n] for n in neighbors]
                assert i not in neighbors
                j=selected_method(tensor,methods,backgrounds,neighbors,spec["criterion"])
            is_stable=bool(np.all(tensor[i,j]>=THRESHOLD-1e-12))
            vals=tensor[i,j]
            row={
                "team":team,"policy":policy,"selected_method":methods[j],
                "training_neighbors":";".join(map(str,train_ids)),
                "full_tpr":float(vals[fi]),
                "P01_full_tpr":float(tensor[i,bi,fi]),
                "full_hit010":int(vals[fi]>=THRESHOLD-1e-12),
                "P01_full_hit010":int(tensor[i,bi,fi]>=THRESHOLD-1e-12),
                "hit_backgrounds":int((vals>=THRESHOLD-1e-12).sum()),
                "P01_hit_backgrounds":int((tensor[i,bi]>=THRESHOLD-1e-12).sum()),
                "all7_hit010":int(is_stable),
                "P01_all7_hit010":int(np.all(tensor[i,bi]>=THRESHOLD-1e-12)),
                "min7_tpr":float(vals.min()),"max7_tpr":float(vals.max()),
                "mean7_tpr":float(vals.mean()),
                "mean7_delta_P01":float((vals-tensor[i,bi]).mean()),
                "full_delta_P01":float(vals[fi]-tensor[i,bi,fi]),
            }
            per.append(row)
    records=pd.DataFrame(per)
    agg=[]
    for policy,g in records.groupby("policy",sort=False):
        agg.append({
            "policy":policy,"targets":len(g),
            "full_ge010":int(g.full_hit010.sum()),
            "all7_ge010":int(g.all7_hit010.sum()),
            "full_mean_tpr":float(g.full_tpr.mean()),
            "mean_min7_tpr":float(g.min7_tpr.mean()),
            "mean_delta_full_P01":float(g.full_delta_P01.mean()),
            "full_improved":int((g.full_delta_P01>1e-12).sum()),
            "full_tied":int((g.full_delta_P01.abs()<=1e-12).sum()),
            "full_worsened":int((g.full_delta_P01< -1e-12).sum()),
            "selected_method_count":int(g.selected_method.nunique()),
            "common_selected_methods_ge2":int((g.selected_method.value_counts()>=2).sum()),
        })
    summary=pd.DataFrame(agg)
    return records,summary


def panel_audit(tensor,methods,backgrounds):
    fi=backgrounds.index("full")
    hist=[]
    for i,team in enumerate(TARGET_SET):
        allrow=tensor[i]
        full=allrow[:,fi]>=THRESHOLD-1e-12
        robust=(allrow>=THRESHOLD-1e-12).all(axis=1)
        hist.append({
            "team":team,
            "fixed_method_hits_full":int(full.sum()),
            "fixed_method_hits_all7":int(robust.sum()),
            "oracle_best_full_tpr":float(allrow[:,fi].max()),
            "oracle_best_min7_tpr":float(allrow.min(axis=1).max()),
            "P01_full_tpr":float(allrow[methods.index(BASELINE),fi]),
            "P01_min7_tpr":float(allrow[methods.index(BASELINE)].min()),
            "oracle_fixed22_ge010_full":int(full.any()),
            "oracle_fixed22_ge010_all7":int(robust.any()),
        })
    return pd.DataFrame(hist)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    root=a.root.resolve();out=a.out.resolve()
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    feat, scores, methods, backgrounds, hashes=read_inputs(root)
    public_d, feature_spec=public_distances(feat)
    public_d.to_csv(out/"public_feature_distances.csv",float_format="%.17g")
    tensor=make_tensor(scores,methods,backgrounds)
    panel=panel_audit(tensor,methods,backgrounds)
    panel.to_csv(out/"historical_fixed22_targets.csv",index=False,float_format="%.17g")
    rec, summary=evaluation_table(scores,feat,methods,backgrounds,tensor,public_d)
    rec.to_csv(out/"transfer_looto_by_target.csv",index=False,float_format="%.17g")
    summary.to_csv(out/"transfer_looto_summary.csv",index=False,float_format="%.17g")
    protocol={
        "scope":"official preliminary, nontraining fixed22 x 7 backgrounds x 17 eligible target cohorts",
        "excluded_target_ids":[1,2,12],
        "threshold_tpr_at_fpr001":THRESHOLD,
        "strict_holdout":"method choice for team t uses official truth of other k teams only, never target t",
        "group_provenance":"public C observed characteristics, never target t truth",
        "model_selection_bias":"fixed22 candidate pool historically selected and reported using these same preliminary labels; LOOTO alone cannot remove upstream post-selection bias",
        "target_metric_truth_used_only_after_target_method_choice":True,
        "features":feature_spec,
        "fixed_rules":POLICIES,"baseline":BASELINE,
        "views":backgrounds,
        "source_hashes":hashes,
        "synthetic_round":False,
        "main_submission_generated":False,
        "main_TPR_computed":False,
    }
    safe_json(out/"protocol.json",protocol)
    assert len(panel)==17 and len(rec)==17*(len(POLICIES)+1)
    assert all((rec.P01_full_tpr>=THRESHOLD-1e-12).astype(int)==rec.P01_full_hit010)
    assert summary[summary.policy=="P01_fixed"].full_ge010.item()==2
    print("FIXED METHODS historical full >=0.10",int(panel.oracle_fixed22_ge010_full.sum()),"of 17")
    print("FIXED METHODS historical all7 >=0.10",int(panel.oracle_fixed22_ge010_all7.sum()),"of 17")
    print("TRANSFER summary")
    print(summary.to_string(index=False))
    print("OUTPUT:",out)


if __name__=="__main__":
    main()
