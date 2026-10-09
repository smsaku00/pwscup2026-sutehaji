"""TPR 0.10 threshold screen of P03 AIA-imprint-gated C-neighbor corrections.

Frozen before truth: 18-target P01 plus predeclared weight and relative-neighbor
variants, with the original AIA-imprint grouping obtained solely from public
C and AIA. This reuses a group hypothesis selected post hoc after prior runs:
the exploration is NOT independent. Official preliminary truth is read only
after all score arrays are frozen. Main submission/rare bit-code unchanged.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,platform,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

JOINT_SHA="28b6a7c6106d89c9bcca6ca138c0aaae23c0b7c03e8dc299c34efd98c9bd4cb3"
JOINT="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py"
AIA_DIR="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正"
VIEW_NAMES=["full","sub80_seed20261009","sub80_seed20261010","sub80_seed20261011"]
NUM=("age","SBP","TG","HDL","ALT")
WEIGHTS=[0.05,0.10,0.15,0.20,0.30]
OTHER_VARIANTS=[("rel_med",0.10),("rel_med",0.20),
                ("two",0.10),("two",0.20),
                ("top20",0.15),("top20",0.30)]
TARGET_18=[3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
TRUTH_FPR=0.01
THRESHOLD=0.10
BASELINE_FULL=0.025495409939854385


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def writej(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def base_module(root):
    file=root/JOINT
    if sha(file)!=JOINT_SHA:raise RuntimeError("Baseline code drift")
    spec=importlib.util.spec_from_file_location("joint_pinned",file)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj


def load_group(root,teams):
    group_file=root/AIA_DIR/"results/public_target_groups.csv"
    x=pd.read_csv(group_file)
    assert x.team.is_unique and len(x)==20
    selected=set(x.loc[x.aia_imprint_high==1,"team"])
    active=np.array([team in selected for team in teams],dtype=bool)
    count18=sum(team in selected for team in TARGET_18)
    assert count18==9, (count18,sorted(selected))
    return active,group_file,sorted(selected.intersection(TARGET_18))


def nearest_features(common,b,a,cs,teams):
    scale=common.scales(a,NUM)
    bx=common.numbers(b,NUM)/scale
    one=[];two=[]
    for team in teams:
        cx=common.numbers(cs[team],NUM)/scale
        d=cdist(bx,cx,"sqeuclidean")
        nn=np.partition(d,1,axis=1)
        one.append(nn[:,0])
        two.append(nn[:,:2].mean(axis=1))
    return np.column_stack(one),np.column_stack(two)


def candidate_scores(base,cal,normal,ids,selected,one,two,active):
    assert len(ids)==18
    ranks=base.ranks(cal[normal][:,selected])
    p01=ranks-0.10*base.rivals(ranks,np.median)
    n1=base.ranks(-one[normal][:,selected])
    n2=base.ranks(-two[normal][:,selected])
    mask=active[selected][None,:].astype(float)
    other=base.rivals(n1,np.median)
    p01_rank=base.ranks(p01)
    variants={"P01":p01}
    for w in WEIGHTS:variants[f"P03_nn1_w{w:g}"]=p01+mask*w*n1
    for name,w in OTHER_VARIANTS:
        if name=="rel_med":feature=n1-other
        elif name=="two":feature=n2
        elif name=="top20":feature=n1*(p01_rank>=.80)
        else:raise ValueError(name)
        variants[f"P03_{name}_w{w:g}"]=p01+mask*w*feature
    assert len(variants)==1+len(WEIGHTS)+len(OTHER_VARIANTS)
    assert all(x.shape==(875,18) for x in variants.values())
    return variants


def summarize(results):
    summary=[]
    base=results[results.method=="P01"].set_index(["background","team"])["tpr_0.01"]
    for (bg,method),g in results.groupby(["background","method"],sort=False):
        b=base.loc[[(bg,int(t)) for t in g.team]].to_numpy(float)
        arr=g["tpr_0.01"].to_numpy(float)
        d=arr-b
        summary.append({
            "background":bg,"method":method,"targets":len(g),
            "mean_tpr":float(arr.mean()),
            "mean_delta_P01":float(d.mean()),
            "hit_ge010":int((arr>=THRESHOLD-1e-12).sum()),
            "new_hits_over_P01":int(((arr>=THRESHOLD-1e-12)&(b<THRESHOLD-1e-12)).sum()),
            "lost_hits_vs_P01":int(((arr<THRESHOLD-1e-12)&(b>=THRESHOLD-1e-12)).sum()),
            "wins":int((d>1e-12).sum()),"ties":int((np.abs(d)<=1e-12).sum()),
            "losses":int((d<-1e-12).sum()),
            "mean_tpr_fpr005":float(g["tpr_0.005"].mean()),
            "mean_tpr_fpr020":float(g["tpr_0.02"].mean()),
            "band2plus":int((g.band>=2).sum())
        })
    return pd.DataFrame(summary)


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--root",type=Path,required=True)
    arg.add_argument("--output-dir",type=Path,required=True)
    a=arg.parse_args();root=a.root.resolve();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    started=time.perf_counter()
    old=base_module(root)
    common,calib,base,paths,b,aa,qs,cs,teams,private,rare,normal=old.load_inputs(root)
    assert len(teams)==20 and int(normal.sum())==875
    assert TARGET_18==[t for t in teams if t not in (1,2)]
    ix=[teams.index(t) for t in TARGET_18]
    active,groupfile,high=load_group(root,teams)
    cnear,ctwo=nearest_features(common,b,aa,cs,teams)
    methods=[]
    hashes=[]
    backgrounds=VIEW_NAMES
    for view in backgrounds:
        if view=="full":abg=aa
        else:
            rng=np.random.default_rng(int(view.split("seed")[1]))
            abg=aa.iloc[np.sort(rng.choice(len(aa),int(.8*len(aa)),replace=False))]
        raw,cal=old.raw_scores(common,calib,b,abg,qs,cs,teams)
        score=candidate_scores(old,cal,normal,TARGET_18,ix,cnear,ctwo,active)
        methods.append((view,score))
        for name,mat in score.items():
            hashes.append({"background":view,"method":name,
                           "sha256":hashlib.sha256(np.ascontiguousarray(mat,dtype="<f8").tobytes()).hexdigest(),
                           "shape":list(mat.shape)})
    writej(out/"prediction_hashes.json",hashes)
    frozen=datetime.now(timezone.utc).isoformat()
    print("PREDICTION FREEZE; now reading official released preliminary labels",flush=True)
    y,audit,truthpaths=old.audit_truth(base,b,private,rare,teams)
    rows=[]
    for view,variants in methods:
        for name,mat in variants.items():
            conf=old.confidence(mat,normal,private,rare)
            assert np.all(conf[private]==0) and np.all(conf[rare & ~private]==.4)
            for i,t in enumerate(TARGET_18):
                m=old.metrics(conf[:,i],y[:,ix[i]])
                rows.append({"background":view,"team":t,"method":name,**m})
    df=pd.DataFrame(rows)
    summary=summarize(df)
    df.to_csv(out/"by_target.csv",index=False,float_format="%.17g")
    summary.to_csv(out/"summary.csv",index=False,float_format="%.17g")
    ref=summary[(summary.background=="full")&(summary.method=="P01")].mean_tpr.item()
    if abs(ref-BASELINE_FULL)>1e-12:raise RuntimeError(f"P01 regression changed: {ref}")
    # New hits in any background and stable background success reported separately.
    h=df.assign(hit=df["tpr_0.01"]>=THRESHOLD-1e-12).groupby(
        ["team","method"],as_index=False).agg(
            hit_backgrounds=("hit","sum"),
            worst_tpr=("tpr_0.01","min"),mean_tpr=("tpr_0.01","mean"))
    h["stable_all4_ge010"]=h.hit_backgrounds.eq(4)
    h.to_csv(out/"threshold_stability.csv",index=False,float_format="%.17g")
    writej(out/"protocol.json",{
        "scope":"18-target preliminary (Team1/2 both excluded from cross-target scores and evaluation)",
        "P03_group":"previously selected post-hoc, public Q/C-only, not independent validation",
        "high_group":high,"comparison":"P01, nearest1 weights, nearest2, relative median, top20 gate",
        "weights":WEIGHTS,"additional_variants":OTHER_VARIANTS,
        "FPR":TRUTH_FPR,"target_TPR":THRESHOLD,
        "truth_frozen_after":frozen,
        "input_hashes":{str(p.relative_to(root)):sha(p) for p in paths+[root/JOINT,groupfile,Path(__file__).resolve()]},
        "truth_paths_sha256":{str(p.relative_to(root)):sha(p) for p in truthpaths},
        "main_submission":False,"main_truth":False
    })
    print(summary[summary.background=="full"].to_string(index=False))
    print("MAIN TARGET HIT TABLE:\n",h[h.hit_backgrounds>0].sort_values(
        ["team","method"]).to_string(index=False))
    print("elapsed_sec",time.perf_counter()-started)


if __name__=="__main__":
    main()
