"""Reconstruct the archived 13-attribute median/IQR aligned KDE ratio method.

Provenance: 20261008 public KDE geometry source:
- NUM13 = (age,BMI,SBP,TG,HDL,ALT,FPG,time), CAT13 = (sex,prefecture,smoking,onset,death);
- Gaussian numeric RBF scaled to public A_bg, bandwidth h0*2;
- categorical smoothed equality log(.98 * eq + .02 * empirical-public-background-prob);
- median/IQR-align C numeric marginals to A_bg; raw rank normal875;
- subtract 0.1 median rank of other target scores.

Regression-check archived full A_bg 22-method KPI for all 17 comparable teams.
Then evaluate fixed mixing ratios with P01: all scores and rule parameters
frozen BEFORE any published prelim pool_ids truth is opened. Only aggregate
cohort-level outputs are written. Historical method- and group-selection has
been influenced by known prelim truth: this is NOT independent validation.
"""
from __future__ import annotations

import argparse,hashlib,importlib.util,json,platform,time,warnings
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import rankdata

BASE_REL="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py"
BASE_SHA="28b6a7c6106d89c9bcca6ca138c0aaae23c0b7c03e8dc299c34efd98c9bd4cb3"
GROUP_REL="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/public_target_groups.csv"
HIST_REL="攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_非学習方式と加工特徴_背景耐性/固定方式_全背景評価.csv"
HIST_METHOD="q13_median_iqr_sex0_kde_ratio_h2__raw__cross0.1"
NUM=("age","BMI","SBP","TG","HDL","ALT","FPG","time")
CAT=("sex","prefecture","smoking","onset","death")
WEIGHTS=(0.,0.25,0.50,0.75,1.0)
VIEWS=("full","sub80_seed20261009","sub80_seed20261010","sub80_seed20261011")
T18=(3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29)
BASELINE20=(0.08079772079772081,0.07895156695156695,
            0.07702279202279202,0.07687464387464388)
BASELINE18=(0.025495409939854385,0.02327951883507439,
            0.02780310224754669,0.023358657803102246)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")


def import_base(root):
    p=root/BASE_REL
    if sha(p)!=BASE_SHA:raise RuntimeError("Pinned P01 source changed")
    spec=importlib.util.spec_from_file_location("repro_p01",p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def numeric(common,frame):
    return common.numbers(frame,NUM)


def ref_transform(common,a):
    z=numeric(common,a)
    scale=(np.quantile(z,.75,axis=0)-np.quantile(z,.25,axis=0))/1.349
    scale=np.where(scale>1e-8,scale,np.maximum(np.std(z,axis=0),1e-6))
    center=np.median(z,axis=0)
    assert np.isfinite(scale).all() and np.all(scale>0)
    return center,scale,z


def align_numeric(common,a_numeric,c):
    """Original median/IQR per numeric dimension without by-sex split."""
    cx=numeric(common,c)
    result=cx.copy()
    center_a=np.median(a_numeric,axis=0)
    center_c=np.median(cx,axis=0)
    scale_a=np.quantile(a_numeric,.75,axis=0)-np.quantile(a_numeric,.25,axis=0)
    scale_c=np.quantile(cx,.75,axis=0)-np.quantile(cx,.25,axis=0)
    for j in range(len(NUM)):
        if scale_c[j]<1e-8 or scale_a[j]<1e-8:
            result[:,j]=cx[:,j]-center_c[j]+center_a[j]
        else:
            result[:,j]=(cx[:,j]-center_c[j])*scale_a[j]/scale_c[j]+center_a[j]
    return result


def mixed_logs(distance,query,reference,background,h):
    """Canonical smoothed categorical equality kernel, no membership labels."""
    total=-.5*distance/(h*h)
    for name in CAT:
        levels=sorted(set(background[name])|set(reference[name])|set(query[name]))
        count=background[name].value_counts().reindex(levels,fill_value=0)
        probs=(count+1.)/(len(background)+len(levels))
        p=query[name].map(probs).to_numpy(float)[:,None]
        match=query[name].to_numpy()[:,None]==reference[name].to_numpy()[None,:]
        total+=np.log(.98*match+.02*p)
    ans=logsumexp(total,axis=1)-np.log(len(reference))
    assert np.isfinite(ans).all()
    return ans


def scores_KDEA(common,b,a,cs,teams):
    center,scale,an=ref_transform(common,a)
    x=(numeric(common,b)-center)/scale
    ar=(an-center)/scale
    h0=(4/(len(NUM)+2)/len(a))**(1/(len(NUM)+4))
    h=h0*2.
    la=mixed_logs(cdist(x,ar,"sqeuclidean"),b,a,a,h)
    by_team={}
    for index,t in enumerate(teams):
        c=cs[t]
        cn=align_numeric(common,an,c)
        cr=(cn-center)/scale
        lc=mixed_logs(cdist(x,cr,"sqeuclidean"),b,c,a,h)
        by_team[t]=lc-la
        print("exact-KDEA-score",index+1,len(teams),"team",t,flush=True)
    return np.column_stack([by_team[t] for t in teams]),h


def ranks(raw,normal):
    return np.column_stack([
        rankdata(raw[normal,j],method="average")/int(normal.sum())
        for j in range(raw.shape[1])
    ])


def scores_by_scope(base,cal,kde,normal,teams,high):
    groups=[]
    for scope,ids in [
        ("infer18",[t for t in teams if t not in (1,2)]),
        ("infer20",list(teams))
    ]:
        idx=[teams.index(t) for t in ids]
        r=base.ranks(cal[normal][:,idx])
        p01=r-.1*base.rivals(r,np.median)
        k=ranks(kde[:,idx],normal)
        kaligned=k-.1*base.rivals(k,np.median)
        gate=np.array([int(t in high) for t in ids],float)[None,:]
        assert sum(int(t in high) for t in ids)==9 if scope=="infer18" else True
        scores={}
        for a in WEIGHTS:
            score=np.where(gate>0,(1-a)*p01+a*kaligned,p01)
            name="P01" if a==0 else f"originalKDEA_mix{a:g}"
            scores[name]=score
        groups.append((scope,ids,scores))
    # Separate full20 reference for original geometry-score regression.
    return groups


def metrics(base,truth,teams,mat,normal,private,rare,ids):
    confidence=base.confidence(mat,normal,private,rare)
    assert np.all(confidence[private]==0) and np.all(confidence[rare & ~private]==.4)
    return [
        base.metrics(confidence[:,j],truth[:,teams.index(t)])
        for j,t in enumerate(ids)
    ]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--full-only",action="store_true")
    args=parser.parse_args();root,out=args.root.resolve(),args.output_dir.resolve()
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    start=time.perf_counter()
    base=import_base(root)
    common,calib,basedir,paths,b,a,qs,cs,teams,private,rare,normal=base.load_inputs(root)
    assert len(teams)==20 and tuple(t for t in teams if t not in (1,2))==T18
    gp=root/GROUP_REL
    group=pd.read_csv(gp)
    high=set(group.loc[group.aia_imprint_high==1,"team"]).intersection(T18)
    assert len(high)==9
    backgrounds=VIEWS[:1] if args.full_only else VIEWS
    bundles=[]
    hashes=[]
    raw_kde_full=None
    for view in backgrounds:
        if view=="full":bg=a
        else:
            rng=np.random.default_rng(int(view.split("seed")[1]))
            sel=np.sort(rng.choice(len(a),int(.8*len(a)),replace=False))
            bg=a.iloc[sel]
        kde,h=scores_KDEA(common,b,bg,cs,teams)
        raw,cal=base.raw_scores(common,calib,b,bg,qs,cs,teams)
        if view=="full":raw_kde_full=kde
        group_scopes=scores_by_scope(base,cal,kde,normal,teams,high)
        bundles.append((view,group_scopes))
        for scope,ids,scores in group_scopes:
            for name,mat in scores.items():
                hashes.append({
                    "background":view,"scope":scope,"method":name,
                    "shape":list(mat.shape),
                    "sha256":hashlib.sha256(
                        np.ascontiguousarray(mat,dtype="<f8").tobytes()).hexdigest()
                })
    write_json(out/"prediction_hashes.json",hashes)
    frozen=datetime.now(timezone.utc).isoformat()
    print("FROZEN original KDE A and P01 scores. Official preliminary labels read only now.",flush=True)
    y,audit,truthpaths=base.audit_truth(basedir,b,private,rare,teams)
    result=[]
    for view,sets in bundles:
        for scope,ids,models in sets:
            for method,mat in models.items():
                res=metrics(base,y,teams,mat,normal,private,rare,ids)
                result.extend({
                    "background":view,"inference":scope,"team":t,
                    "method":method,**res[j]
                }for j,t in enumerate(ids))
    table=pd.DataFrame(result)
    table.to_csv(out/"by_target.csv",index=False,float_format="%.17g")
    # Reproduce the 20-target crossmedian archived original KDE (full only).
    exact_rank=ranks(raw_kde_full,normal)
    exact_cross=exact_rank-.1*base.rivals(exact_rank,np.median)
    original_metrics=metrics(base,y,teams,exact_cross,normal,private,rare,teams)
    h=pd.read_csv(root/HIST_REL,encoding="utf-8-sig")
    h=h[(h.method==HIST_METHOD) &
        np.isclose(h.fpr,.01) & (h.condition=="full") &
        h.team.isin(T18)]
    observed={t:original_metrics[i]["tpr_0.01"] for i,t in enumerate(teams)}
    maxerr=max(abs(observed[int(r.team)]-float(r.tpr)) for r in h.itertuples())
    # Require exact historical official TPR@1% on all 18 overlapping target cohorts.
    if len(h)!=17:
        raise RuntimeError("Expected 17 published historical team KPI entries")
    if maxerr>1e-10:
        raise RuntimeError(f"KDEA historical regression mismatch = {maxerr}")
    write_json(out/"archived_KDEA_regression.json",{
        "source_method":HIST_METHOD,
        "result":"reproduced all 17 target TPR@FPR=.01 values within tolerance",
        "targets":len(h),"max_abs_error":maxerr,
        "scope":"20-target joint crossmedian then exclude Teams1/2/12 only for historical 17-team reporting"
    })
    summary=[]
    for (view,scope),g in table.groupby(["background","inference"],sort=False):
        targetids=list(g[g.method=="P01"].team)
        comparator=g[g.method=="P01"].set_index("team")["tpr_0.01"]
        for evalname,ignore in [("without1_2",(1,2)),("without1_2_12",(1,2,12))]:
            allowed=[t for t in targetids if t not in ignore]
            for method,x in g[g.team.isin(allowed)].groupby("method",sort=False):
                val=x.set_index("team")["tpr_0.01"].reindex(allowed)
                baseline=comparator.reindex(allowed)
                diff=val-baseline
                summary.append({
                    "background":view,"inference":scope,"evaluation":evalname,
                    "method":method,"targets":len(allowed),
                    "mean_tpr":float(val.mean()),
                    "mean_delta_P01":float(diff.mean()),
                    "full_reached_010":int((val>=.1-1e-12).sum()),
                    "new_reached_010":int(((val>=.1-1e-12)&(baseline<.1-1e-12)).sum()),
                    "lost_reached_010":int(((val<.1-1e-12)&(baseline>=.1-1e-12)).sum()),
                    "wins":int((diff>1e-12).sum()),
                    "losses":int((diff< -1e-12).sum()),
                    "band2plus":int((x["band"]>=2).sum())
                })
    S=pd.DataFrame(summary)
    S.to_csv(out/"summary.csv",index=False,float_format="%.17g")
    for i,view in enumerate(backgrounds):
        for scope,ref in [("infer18",BASELINE18[i]),
                          ("infer20",BASELINE20[i])]:
            observed=S[(S.background==view)&(S.inference==scope)&
                       (S.evaluation=="without1_2")&(S.method=="P01")]
            expected=ref if scope=="infer18" else (
                .029775245330800885 if i==0 else None)
            if expected is not None and abs(observed.mean_tpr.item()-expected)>1e-12:
                raise RuntimeError(f"P01 baseline mismatch: {scope} {view} {observed.mean_tpr.item()}")
    # KPI for robust 0.10 across views.
    v=table[table.inference=="infer18"].copy()
    counts=(v.assign(hit=v["tpr_0.01"].ge(.1-1e-12))
            .groupby(["method","team"],as_index=False)
            .agg(hit_bg=("hit","sum"),
                 min_tpr=("tpr_0.01","min"),mean_tpr=("tpr_0.01","mean")))
    counts["all_view_hit"]=counts.hit_bg.eq(len(backgrounds))
    counts.to_csv(out/"threshold_stability.csv",index=False,float_format="%.17g")
    write_json(out/"protocol.json",{
        "algorithm":"archived aligned q13 KDE A exact formula + P01 convex rank mixtures on public AIA-imprint 9-group",
        "mix_weights":WEIGHTS,"numeric":NUM,"categorical":CAT,
        "views":backgrounds,
        "two_inference_scopes":["infer18","infer20"],
        "posthoc":True,"historical_regression_max_abs_error":maxerr,
        "pool_truth_only_after_prediction_freeze":True,
        "frozen_utc":frozen,
        "caveat":"Method KDE A and AIA-imprint group originally identified in previously exposed preliminary labels",
        "input_hashes":{str(p.relative_to(root)):sha(p) for p in
                        paths+[root/BASE_REL,gp,root/HIST_REL,Path(__file__).resolve()]},
        "truth_hashes":{str(p.relative_to(root)):sha(p) for p in truthpaths},
        "main_submission":False,"main_truth":False,
        "rare_5bit_unmodified":True,
    })
    print("KDE A archival full TPR regression error",maxerr)
    print(S[(S.background=="full")&(S.inference=="infer18")&
            (S.evaluation=="without1_2")].to_string(index=False))
    print("OUTPUT",out,"seconds",time.perf_counter()-start)


if __name__=="__main__":
    main()
