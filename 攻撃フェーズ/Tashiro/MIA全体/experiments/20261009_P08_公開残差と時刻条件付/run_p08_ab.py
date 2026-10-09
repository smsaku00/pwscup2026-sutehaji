"""P08 experiments A/B, official preliminary B_self against 20 submitted C.

A: decontaminate each public C by exact 13-field matches with OFFICIAL A_bg
   using rounding=8. Full removal or fixed deterministic 1/5 retention of
   public matches, then recompute original GenLRA7 Q500 + cross-median P01.
   If full removal leaves <10 C rows, revert to unchanged C; mark fallback.
B: fixed, shared-parameter, age/sex or sex/prefecture conditional onset/time
   evidence contrasting C vs A_bg, fused with existing P01 rank. Additionally
   evaluate exact time/onset matching; not original Team4 legacy event method.

All method formulae and score hashes frozen before reading official pool_ids.
Numerical FPR=.005/.01/.02 uses the pinned official-compatible sklearn ROC
implementation. Report all 20 and primary 18 (excluding Teams1,2 from
INFERENCE and EVALUATION) separately, plus four matched A_bg views.
Cross-team public Q in P01 is unchanged; A decontamination uses full official
A_bg as a fixed public source, not A_bg resample to classify exact matches.
No A2 legacy starter-kit data in repo: C rows copied from A2 cannot be
identified here, so NO assertion about removing all public-derived rows.
Prior methods and hypotheses repeatedly examined published preliminary
labels: NOT unbiased out-of-sample validation. No main B/self/submissions.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from openpyxl import load_workbook

BASE="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py"
JOINT_SHA="28b6a7c6106d89c9bcca6ca138c0aaae23c0b7c03e8dc299c34efd98c9bd4cb3"
OFFICIAL="公式データ/予備戦/結果/公開用結果表_20260909.xlsx"
C_MATCH_COLS=("age","sex","prefecture","BMI","SBP","TG","HDL","ALT",
              "smoking","FPG","time","onset","death")
VIEWS=("full","sub80_seed20261009","sub80_seed20261010","sub80_seed20261011")
EXCLUDED=(1,2)
TIME_CHANNELS=("time_cond_sex_age10_year1","time_cond_sex_pref_year1","time_exact_age10")
WEIGHTS=(.15,.35)
METH=["P01","A_full_public_remove","A_public_keep20"]
for channel in TIME_CHANNELS:
    for w in WEIGHTS:METH.append(f"B_{channel}_w{w:g}")
METH.append("B_time_exact_age10_standalone")
assert len(METH)==10


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,doc):Path(p).write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def load_joint(root):
    p=root/BASE
    if sha(p)!=JOINT_SHA:raise RuntimeError("Pinned P01 source changed")
    spec=importlib.util.spec_from_file_location("p08_base_joint",p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def match_public(C,A):
    keysA=pd.MultiIndex.from_frame(A[list(C_MATCH_COLS)].round(8))
    keysC=pd.MultiIndex.from_frame(C[list(C_MATCH_COLS)].round(8))
    return keysC.isin(keysA)


def clean_public_C(cs,a,teams):
    out={"A_full_public_remove":{},"A_public_keep20":{}}
    stats=[]
    for t in teams:
        c=cs[t]; match=match_public(c,a)
        mask=np.asarray(match,bool)
        n=int(mask.sum())
        strict=c.loc[~mask].reset_index(drop=True)
        # If all C is public A (e.g. Team18) there is no residual density:
        # retain baseline score as explicit honest fallback, not fabricate C.
        fallback=len(strict)<10
        out["A_full_public_remove"][t]=(c if fallback else strict)
        # 20% soft contribution is approximated by stable *row thinning*, not
        # claimed an exact weighted-kernel subtraction.
        stablehash=pd.util.hash_pandas_object(c[list(C_MATCH_COLS)],index=False).to_numpy()
        keep=(~mask)|(stablehash%5==0)
        soft=c.loc[keep].reset_index(drop=True)
        assert len(soft)>=10
        out["A_public_keep20"][t]=soft
        stats.append(dict(team=t,C_rows=len(c),full_public_A_copy_rows=n,
            noncopy_rows=len(strict),strict_fallback=int(fallback),
            soft_retained=int(len(soft)),soft_copied_retained=int((keep&mask).sum())))
    return out,pd.DataFrame(stats)


def recalibrate_changed(common,calib,b,aa,qs,cs,teams,baseline_cal,variant):
    cal=baseline_cal.copy()
    touched=[]
    for j,t in enumerate(teams):
        if variant[t] is cs[t] or len(variant[t])==len(cs[t]):
            # If count equal no exact A1 matches to remove, or strict fallback,
            # no numerical score change; exact identity is sufficient.
            continue
        controls=pd.concat([qs[k][list(common.QI7)] for k in sorted(qs)
                            if k!=t],ignore_index=True)
        query=pd.concat([b[list(common.QI7)],controls],ignore_index=True)
        score=common.core(aa,query,variant[t],common.NUM7,(10,))[10]
        vals,null=score[:len(b)],score[len(b):]
        cal[:,j]=calib.calibrate(vals,b,null,controls,["prefecture","sex"],500)
        touched.append(t)
    return cal,touched


def cond_frame(data,time_width,age_span,mode):
    tmp=pd.DataFrame(index=np.arange(len(data)))
    tmp["sex"]=data["sex"].to_numpy()
    tmp["prefecture"]=data["prefecture"].astype(str).to_numpy()
    tmp["age_group"]=np.floor(data.age.to_numpy(float)/age_span).astype(int)
    tmp["onset"]=data.onset.to_numpy().astype(int)
    tmp["time_bin"]=np.floor(data.time.to_numpy(float)/time_width).astype(int)
    if mode=="sex_age":
        return tmp,["sex","age_group"],["onset","time_bin"]
    if mode=="sex_pref":
        return tmp,["sex","prefecture"],["onset","time_bin"]
    raise ValueError(mode)


def joint_counts(x,cols,target):
    groups=x.groupby(cols,dropna=False).size()
    return groups.reindex(pd.MultiIndex.from_frame(target[cols]),fill_value=0).to_numpy(dtype=float)


def time_conditional_signal(b,a,c,mode):
    assert mode in TIME_CHANNELS
    if mode=="time_exact_age10":
        # Exact/near-exact (10^-6) event-time witness, not a generic
        # time-bin KDE. No signal when neither C nor A has the tuple.
        cols=("sex","age_group","onset","time_bin")
        def trans(df):
            return pd.DataFrame({
              "sex":df.sex.to_numpy(),
              "age_group":np.floor(df.age.to_numpy(float)/10).astype(int),
              "onset":df.onset.to_numpy().astype(int),
              "time_bin":np.rint(df.time.to_numpy(float)*1e6).astype(np.int64)
            })
        bx,ax,cx=(trans(df) for df in (b,a,c))
        nc=joint_counts(cx,list(cols),bx)
        na=joint_counts(ax,list(cols),bx)
        # Count-ratio shrinkage avoids "no match" artifacts; add sample
        # size normalization only to genuine matches, no posterior training.
        val=np.log1p(4*nc)-np.log1p(4*na)
        assert np.isfinite(val).all()
        return val
    width=1.0
    kind="sex_age" if mode=="time_cond_sex_age10_year1" else "sex_pref"
    bx,groups,evt=cond_frame(b,width,10,kind)
    ax,_,_=cond_frame(a,width,10,kind)
    cx,_,_=cond_frame(c,width,10,kind)
    # Public-only fixed event/time alphabet size. Include both outcomes
    # onset in {0,1} and all time year bins observed among public sources.
    high=max(ax.time_bin.max(),cx.time_bin.max())
    low=min(ax.time_bin.min(),cx.time_bin.min())
    nbin=int((high-low+3)*2)
    assert nbin>0 and nbin<400
    ac=joint_counts(ax,groups+evt,bx)
    cc=joint_counts(cx,groups+evt,bx)
    ag=joint_counts(ax,groups,bx)
    cg=joint_counts(cx,groups,bx)
    alpha=.25
    pA=(ac+alpha)/(ag+alpha*nbin)
    pC=(cc+alpha)/(cg+alpha*nbin)
    val=np.clip(np.log(pC)-np.log(pA),-6,6)
    assert np.isfinite(val).all()
    return val


def create_scores(base,baseline,copy_models,event,normal,subset,teams):
    indices=[teams.index(t) for t in subset]
    def p01(cal):
        r=base.ranks(cal[normal][:,indices])
        return r-.1*base.rivals(r,np.median)
    ref=p01(baseline)
    methods={"P01":ref}
    for kind in ("A_full_public_remove","A_public_keep20"):
        methods[kind]=p01(copy_models[kind])
    for kind in TIME_CHANNELS:
        mat=event[kind][normal][:,indices]
        signal=base.ranks(mat)
        for w in WEIGHTS:methods[f"B_{kind}_w{w:g}"]=ref+w*signal
        if kind=="time_exact_age10":
            methods["B_time_exact_age10_standalone"]=signal
    assert set(methods)==set(METH)
    return methods


def official_parts(root):
    wb=load_workbook(root/OFFICIAL,read_only=True,data_only=True)
    mi={};aia={}
    for row in list(wb["攻撃_標的別_MIA"].values)[1:]:
        if row[0]==5:mi[int(row[1])]=(float(row[4] or 0),float(row[5] or 0))
    for row in list(wb["攻撃_標的別_AIA"].values)[1:]:
        if row[0]==5:aia[int(row[1])]=float(row[9] or 0)
    assert len(mi)==len(aia)==20
    old=sum(max(*mi[t])+aia[t] for t in mi)/20
    assert abs(old-.3297953270354882)<1e-12
    return mi,aia,old


def calc(root,out):
    if out.exists():raise FileExistsError("Refuse overwrite: "+str(out))
    out.mkdir(parents=True)
    begin=time.perf_counter()
    base=load_joint(root)
    common,calib,basedir,files,b,a,qs,cs,teams,private,rare,normal=base.load_inputs(root)
    assert len(teams)==20 and int(normal.sum())==875
    primary=[t for t in teams if t not in EXCLUDED]
    assert len(primary)==18
    copy_variant,copy_records=clean_public_C(cs,a,teams)
    copy_records.to_csv(out/"public_copy_cohorts.csv",index=False)
    # P08-A full official A rows define the match masks even on A 80% views.
    changed_by_method={q:[t for t in teams if len(copy_variant[q][t])!=len(cs[t])]
                       for q in copy_variant}
    paths=[f for f in files]+[root/BASE,Path(__file__).resolve()]
    hashes=[];frozen=[];views=[]
    for view in VIEWS:
        if view=="full":bg=a
        else:
            rng=np.random.default_rng(int(view.split("seed")[1]))
            ix=np.sort(rng.choice(len(a),int(.8*len(a)),replace=False))
            bg=a.iloc[ix]
        _,cal=base.raw_scores(common,calib,b,bg,qs,cs,teams)
        copy_cal={}
        for kind in copy_variant:
            copy_cal[kind],actual=recalibrate_changed(common,calib,b,bg,qs,cs,teams,cal,copy_variant[kind])
            assert actual==changed_by_method[kind]
        event={kind:np.column_stack([time_conditional_signal(b,bg,cs[t],kind)
                      for t in teams]) for kind in TIME_CHANNELS}
        scope={}
        for name,subset in [("all20",teams),("without1_2",primary)]:
            score=create_scores(base,cal,copy_cal,event,normal,subset,teams)
            scope[name]=(subset,score)
            for method,matrix in score.items():
                hashes.append({"background":view,"inference":name,"method":method,
                    "shape":list(matrix.shape),
                    "sha256":hashlib.sha256(np.ascontiguousarray(matrix,dtype="<f8").tobytes()).hexdigest()})
        frozen.append((view,scope))
        views.append(view)
        print("FROZEN SCORES",view,"models",len(METH),"elapsed",round(time.perf_counter()-begin,2),flush=True)
    writej(out/"prediction_hashes.json",hashes)
    freeze=datetime.now(timezone.utc).isoformat()
    print("READING OFFICIAL pool_ids ONLY AFTER ALL METHOD/PREDICTION HASHES FROZEN",flush=True)
    y,audit,truthfiles=base.audit_truth(basedir,b,private,rare,teams)
    score_rows=[]
    for view,scopes in frozen:
        for inf,(subset,methods) in scopes.items():
            for method,raw in methods.items():
                conf=base.confidence(raw,normal,private,rare)
                assert np.all(conf[private]==0) and np.all(conf[rare&~private]==.4)
                for j,t in enumerate(subset):
                    score_rows.append({
                       "background":view,"inference":inf,"team":t,
                       "method":method,**base.metrics(conf[:,j],y[:,teams.index(t)])})
    scores=pd.DataFrame(score_rows)
    scores.to_csv(out/"by_target.csv",index=False,float_format="%.17g")
    summary=[]
    for (view,inf),panel in scores.groupby(["background","inference"],sort=False):
        baseline=panel[panel.method=="P01"].set_index("team")["tpr_0.01"]
        for method,group in panel.groupby("method",sort=False):
            z=group.set_index("team")["tpr_0.01"];old=baseline.reindex(z.index)
            diff=z-old
            summary.append({
              "background":view,"inference":inf,"method":method,"targets":len(group),
              "mean_tpr_001":float(z.mean()),
              "mean_delta_P01":float(diff.mean()),
              "TPR_ge010":int((z>=.10-1e-12).sum()),
              "new_ge010":int(((z>=.10-1e-12)&(old<.10-1e-12)).sum()),
              "lost_ge010":int(((z<.10-1e-12)&(old>=.10-1e-12)).sum()),
              "wins":int((diff>1e-12).sum()),
              "losses":int((diff< -1e-12).sum()),
              "band2_plus":int((group.band>=2).sum()),
              "mean_tpr_0005":float(group["tpr_0.005"].mean()),
              "mean_tpr_002":float(group["tpr_0.02"].mean())
            })
    outsummary=pd.DataFrame(summary)
    outsummary.to_csv(out/"summary.csv",index=False,float_format="%.17g")
    expected={"all20":.08079772079772081,"without1_2":.025495409939854385}
    for inf,ref in expected.items():
        row=outsummary[(outsummary.background=="full")&
                       (outsummary.inference==inf)&(outsummary.method=="P01")]
        if abs(row.mean_tpr_001.item()-ref)>1e-12:
            raise RuntimeError(f"P01 baseline regression wrong: {inf}, {row}")
    official,aia,oldattack=official_parts(root)
    virtual=[]
    panel=scores[(scores.background=="full")&(scores.inference=="all20")]
    for method,m in panel.groupby("method"):
        m=m.set_index("team")["tpr_0.01"]
        values={t:(official[t][0] if t in EXCLUDED else float(m.loc[t]))
                for t in official}
        score=sum(max(values[t],official[t][1])+aia[t] for t in official)/20
        virtual.append({"method":method,"prelim_hypothetical_attack":float(score),
          "baseline_official_attack":oldattack,
          "delta_from_official":float(score-oldattack),
          "gap_to_official_third":.352845805596271-score,
          "hits_ge010":int(sum(v>=.1-1e-12 for v in values.values())),
          "MIA_all_mean20":float(np.mean(list(values.values()))),
          "old_Team1_2_retained":True,
          "not_equivalent_to_P04_hybrid":True
        })
    virt=pd.DataFrame(virtual)
    virt.to_csv(out/"virtual_attack_vs_P01.csv",index=False,float_format="%.17g")
    assert len(virt)==len(METH)
    stable=(scores[scores.inference=="without1_2"].assign(hit=lambda x:x["tpr_0.01"]>=.1-1e-12)
             .groupby(["team","method"],as_index=False).agg(
                hits_in_4backgrounds=("hit","sum"),
                min_tpr=("tpr_0.01","min"),mean_tpr=("tpr_0.01","mean")))
    stable["hit_all4"]=stable.hits_in_4backgrounds.eq(len(VIEWS))
    stable.to_csv(out/"threshold_stability.csv",index=False,float_format="%.17g")
    writej(out/"protocol.json",{
      "scope":"preliminary Team5; P01 baseline, P08 A full13 public-copy ablation, P08 B time/onset conditional evidence; no hidden personal labels in predictions",
      "candidate_methods_pre_frozen":METH,
      "P08_A_public_reference":"official 2026 preliminary A_bg only; old starter kit A2 absent",
      "P08_A_copy_round":8,
      "A_strict_empty_fallback":"use unchanged C if fewer than 10 rows remain",
      "A_soft_not_exact_reweight":"retain every noncopy and hash-modulo-5 subset of public copy rows",
      "changed_targets_by_public_copy_method":changed_by_method,
      "P08_B_time_channels":TIME_CHANNELS,"fixed_weights":WEIGHTS,
      "B_time_bins_public_only":True,
      "background_views":VIEWS,"primary_inference":"without1_2, excluded from both median and evaluation",
      "secondary_inference":"all20, with official old Team1/2 TPR only for virtual score",
      "official_old_MIA_rare_AIA_frozen":True,
      "inference_vs_P04_virtual":"NOT comparable as a delta; no KDE A/B hybrid",
      "original_P01_full_regression_expected":expected,
      "predictions_frozen_before_label_read":freeze,
      "prior_repeated_preliminary_truth_reuse":True,
      "no_independent_main_test":True,"no_main_data_read":True,
      "no_competition_submission":True,"no_person_scores_saved":True,
      "input_sha256":{str(p.relative_to(root)):sha(p) for p in paths+[root/OFFICIAL]},
      "truth_sha256_after_freeze":{str(p.relative_to(root)):sha(p) for p in truthfiles},
      "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},
    })
    print("P08 PRIMARY FULL (18 EXCLUDING TEAMS1,2)")
    print(outsummary[(outsummary.background=="full")&(outsummary.inference=="without1_2")].to_string(index=False))
    print("P08 VIRTUAL (P01 BASE, NOT P04)")
    print(virt.sort_values("prelim_hypothetical_attack",ascending=False).to_string(index=False))
    print("P08 PUBLIC COPY COHORTS")
    print(copy_records[copy_records.full_public_A_copy_rows>0].to_string(index=False))
    print("seconds",time.perf_counter()-begin)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    calc(args.root.resolve(),args.out.resolve())
