"""Fresh numeric+categorical 13-attribute KDE score fusion with P01, targeting TPR >= .10.

This is NOT a reconstruction of the earlier archived KDE A/B methods: it is
a new explicitly specified kernel and a fixed P01 mixture. P03 AIA-imprint
group was selected following prior exploratory prelim labels. Methods are
frozen and hashed before the official published pool_ids truth is first read.
"""
from __future__ import annotations

import argparse,hashlib,importlib.util,json,platform,time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

BASEPATH="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py"
BASESHA="28b6a7c6106d89c9bcca6ca138c0aaae23c0b7c03e8dc299c34efd98c9bd4cb3"
GROUPPATH="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/public_target_groups.csv"
TEAM18=[3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
NUM=("age","BMI","SBP","TG","HDL","ALT","FPG","time")
CAT={"sex":1.5,"prefecture":0.75,"smoking":0.5,"onset":0.5,"death":0.5}
BANDWIDTH=(0.75,1.5)
ALIGN=("raw","median_iqr")
WEIGHTS=(0.10,0.25,0.50)
VIEWS=("full","sub80_seed20261009","sub80_seed20261010","sub80_seed20261011")
REF_SAMPLE_SIZE=2048
SEED=20261009
BASELINES=(.025495409939854385,.02327951883507439,
           .02780310224754669,.023358657803102246)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def joint(root):
    p=root/BASEPATH
    if sha(p)!=BASESHA:raise RuntimeError("Pinned joint baseline changed")
    spec=importlib.util.spec_from_file_location("pinned_joint_for_KDE",p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def transform(common,frame,scale):
    return common.numbers(frame,NUM)/scale


def median_iqr_align(a,c):
    """Align public C to background marginal medians/IQRs, never using B labels."""
    am=np.median(a,axis=0)
    cm=np.median(c,axis=0)
    ai=np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0)
    ci=np.quantile(c,.75,axis=0)-np.quantile(c,.25,axis=0)
    # Constrain median/IQR factors under discrete and very narrow C supports.
    factor=np.clip(ai/np.maximum(ci,0.05*ai+1e-12),0.1,10.0)
    return (c-cm)*factor+am


def sq_distance(b,r,query,cats):
    d2=cdist(b,r,metric="sqeuclidean")
    for name,weight in cats.items():
        d2+=weight*(query[name].to_numpy()[:,None]!=
                    r[name].to_numpy()[None,:])
    return d2


def sq_distance_aligned(b,scaled_reference,query,original_reference,cats):
    d2=cdist(b,scaled_reference,metric="sqeuclidean")
    for name,weight in cats.items():
        d2+=weight*(query[name].to_numpy()[:,None]!=
                    original_reference[name].to_numpy()[None,:])
    return d2


def log_density(distance,bandwidth,reference_n):
    # Fixed-size public KDE reference, positive and normalized.
    value=logsumexp(-0.5*distance/(bandwidth**2),axis=1)-np.log(reference_n)
    if not np.isfinite(value).all():
        raise RuntimeError("Nonfinite kernel density")
    return value


def public_kde(common,b,a,cs,targets):
    """Return log(C density/A density) with public-only references, no pool_ids."""
    scale=common.scales(a,NUM)
    x=transform(common,b,scale)
    ax=transform(common,a,scale)
    rng=np.random.default_rng(SEED)
    chosen=np.sort(rng.choice(len(a),size=min(REF_SAMPLE_SIZE,len(a)),replace=False))
    aa=a.iloc[chosen]
    aa_x=ax[chosen]
    denom_dist=sq_distance_aligned(x,aa_x,b,aa,CAT)
    denominators={h:log_density(denom_dist,h,len(aa)) for h in BANDWIDTH}
    del denom_dist
    results={}
    for j,t in enumerate(targets):
        c=cs[t]
        cx=transform(common,c,scale)
        for kind in ALIGN:
            cc=cx if kind=="raw" else median_iqr_align(ax,cx)
            dis=sq_distance_aligned(x,cc,b,c,CAT)
            for h in BANDWIDTH:
                numerator=log_density(dis,h,len(c))
                results[(t,kind,h)]=numerator-denominators[h]
            del dis
        print(f"KDE public reference {j+1}/{len(targets)} target={t}",flush=True)
    return results, {"scale":scale.tolist(),"bg_sample_seed":SEED,
                     "bg_ref_rows":len(aa),"categories":CAT,"NUM":NUM}


def predictors(joint,cal,normal,indices,targets,grouped,signals):
    p0=joint.ranks(cal[normal][:,indices])
    p01=p0-0.10*joint.rivals(p0,np.median)
    choices={"P01":p01}
    mask=np.array([int(t in grouped) for t in targets])[None,:]
    assert int(mask.sum())==9
    for align in ALIGN:
        for h in BANDWIDTH:
            matrix=np.column_stack([signals[(t,align,h)] for t in targets])
            scores=joint.ranks(matrix[normal])
            for w in WEIGHTS:
                name=f"kde_{align}_h{h:g}_mix{w:g}"
                choices[name]=p01+w*scores*mask
    assert len(choices)==1+len(ALIGN)*len(BANDWIDTH)*len(WEIGHTS)
    return choices


def summarize(df):
    rec=[]
    p01=df[df.method=="P01"].set_index(["background","team"])["tpr_0.01"]
    for (bg,method),g in df.groupby(["background","method"],sort=False):
        baseline=np.array([p01.loc[(bg,int(t))] for t in g.team],dtype=float)
        values=g["tpr_0.01"].to_numpy(float)
        diff=values-baseline
        rec.append({
            "background":bg,"method":method,"teams":len(g),
            "mean_tpr":float(values.mean()),
            "mean_delta_p01":float(diff.mean()),
            "reached_010":int((values>=.10-1e-12).sum()),
            "new_reached_010":int(((values>=.10-1e-12)&(baseline<.10-1e-12)).sum()),
            "lost_reached_010":int(((values<.10-1e-12)&(baseline>=.10-1e-12)).sum()),
            "improved":int((diff>1e-12).sum()),
            "worsened":int((diff< -1e-12).sum()),
            "band2plus":int((g["band"]>=2).sum()),
            "mean_tpr_005":float(g["tpr_0.005"].mean()),
            "mean_tpr_020":float(g["tpr_0.02"].mean()),
        })
    return pd.DataFrame(rec)


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--root",type=Path,required=True)
    arg.add_argument("--output-dir",type=Path,required=True)
    arg.add_argument("--full-only",action="store_true")
    args=arg.parse_args();root,out=args.root.resolve(),args.output_dir.resolve()
    if out.exists():raise FileExistsError("Will not overwrite: "+str(out))
    out.mkdir(parents=True)
    began=time.perf_counter()
    base=joint(root)
    common,calib,basedir,paths,b,a,qs,cs,teams,private,rare,normal=base.load_inputs(root)
    assert len(teams)==20 and len(b)==1049 and int(normal.sum())==875
    assert TEAM18==[t for t in teams if t not in (1,2)]
    indices=[teams.index(t) for t in TEAM18]
    gf=root/GROUPPATH
    group=pd.read_csv(gf)
    high=set(group.loc[group.aia_imprint_high==1,"team"]).intersection(TEAM18)
    assert len(high)==9

    hashes=[];frames=[];predictions=[]
    views=VIEWS[:1] if args.full_only else VIEWS
    for view in views:
        if view=="full":bg=a
        else:
            rng=np.random.default_rng(int(view.split("seed")[1]))
            sel=np.sort(rng.choice(len(a),int(.8*len(a)),replace=False))
            bg=a.iloc[sel]
        signals,params=public_kde(common,b,bg,cs,TEAM18)
        raw,cal=base.raw_scores(common,calib,b,bg,qs,cs,teams)
        preds=predictors(base,cal,normal,indices,TEAM18,high,signals)
        predictions.append((view,preds))
        for method,array in preds.items():
            hashes.append({
                "view":view,"method":method,
                "sha256":hashlib.sha256(np.ascontiguousarray(array,dtype="<f8").tobytes()).hexdigest(),
                "shape":list(array.shape)
            })
    dump(out/"prediction_hashes.json",hashes)
    frozen=datetime.now(timezone.utc).isoformat()
    print("ALL PREDICTIONS FROZEN; reading prelim membership truth only now",flush=True)
    truth,audit,truthpaths=base.audit_truth(basedir,b,private,rare,teams)
    rows=[]
    for view,preds in predictions:
        for method,mat in preds.items():
            conf=base.confidence(mat,normal,private,rare)
            assert np.all(conf[private]==0) and np.all(conf[rare & ~private]==0.4)
            for j,target in enumerate(TEAM18):
                result=base.metrics(conf[:,j],truth[:,indices[j]])
                rows.append({"background":view,"team":target,"method":method,**result})
    df=pd.DataFrame(rows)
    agg=summarize(df)
    for i,bg in enumerate(views):
        y=agg[(agg.background==bg)&(agg.method=="P01")]
        assert len(y)==1
        if abs(float(y.iloc[0].mean_tpr)-BASELINES[i])>1e-12:
            raise RuntimeError("P01 regression mismatch "+str((bg,y)))
    df.to_csv(out/"by_target.csv",index=False,float_format="%.17g")
    agg.to_csv(out/"summary.csv",index=False,float_format="%.17g")
    stability=(df.assign(success=df["tpr_0.01"].ge(.10-1e-12))
               .groupby(["team","method"],as_index=False)
               .agg(full_or_any_hits=("success","sum"),
                    min_tpr=("tpr_0.01","min"),mean_tpr=("tpr_0.01","mean")))
    stability["stable_all_backgrounds"]=stability.full_or_any_hits.eq(len(views))
    stability.to_csv(out/"threshold_stability.csv",index=False,float_format="%.17g")
    dump(out/"protocol.json",{
        "method":"Numerical+categorical 13-attribute Gaussian KDE-ratio rank + P01",
        "not_exact_archived_KDE":True,
        "data":"official preliminary, own B + public C,AIA,A_bg only before label freeze",
        "group_provenance":"AIA imprint high 9 targets chosen in earlier posthoc exploratory preliminary test",
        "TEAM18":TEAM18,"high_group":sorted(high),
        "numeric":NUM,"categorical_penalties":CAT,
        "alignment":ALIGN,"bandwidth":BANDWIDTH,"rank_mix_weights":WEIGHTS,
        "backgrounds":views,"num_reference_bg":REF_SAMPLE_SIZE,
        "TPR_goal":.10,"FPR_primary":.01,
        "input_sha256":{str(p.relative_to(root)):sha(p) for p in paths+[root/BASEPATH,gf,Path(__file__).resolve()]},
        "truth_sha256_after_freeze":{str(p.relative_to(root)):sha(p) for p in truthpaths},
        "freeze_at":frozen,
        "leakage_warning":"Method family and group selected after repeated public preliminary truth inspection",
        "main_submission_modified":False,"rare_5bit_modified":False
    })
    print(agg[agg.background=="full"].to_string(index=False))
    print("OUTPUT:",out,"elapsed_sec",time.perf_counter()-began)


if __name__=="__main__":
    main()
