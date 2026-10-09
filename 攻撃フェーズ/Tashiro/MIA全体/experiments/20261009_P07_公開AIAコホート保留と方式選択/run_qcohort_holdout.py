"""P07: 7-attribute P01/P03 only, public AIA Q cohort-level holdout test.

For target t != holdout h: all 326 Q_h queries are provisionally nonmembers
of B_t under the official AIA nonshared/member-control construction.
Q_h and any identical QI7 rows in other Q cohorts are excluded from P01
calibration. P01 and P03 can both score the seven public AIA columns; KDE A cannot
because Q_h lacks six of its thirteen features. Therefore KDE A/B remain
FIXED. Only P04-P01 targets are eligible for a P03 switch. Candidates
are B_self's 875 ordinary non-private rows. Decisions are frozen before
pool_ids truth. All target scopes use same full20 score construction.

All data used BEFORE truth: own B_self, A_bg, public C, public Q, P04 public
imprint/home thresholds, official workbook's *fixed* old MIA-rare/AIA score
(retrieved only after prediction freeze for additional clarity). No target
membership values used for model scoring or choice. The P04 group, method
family, and policies were originally derived post-hoc from this public
preliminary contest; the present test is NOT independent out-of-sample.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from openpyxl import load_workbook

BASE="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py"
KDEA_SRC="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_TPR010到達と加工特徴転移検証/run_archived_kdeA_mix.py"
HIST="攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_非学習方式と加工特徴_背景耐性/固定方式_全背景評価.csv"
GROUP="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/public_target_groups.csv"
THRESH="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/group_thresholds.json"
OFFICIAL="公式データ/予備戦/結果/公開用結果表_20260909.xlsx"
P01="Q_prefecture_sex_shrink500_cross01"
KDEA="q13_median_iqr_sex0_kde_ratio_h2__raw__cross0.1"
KDEB="q7_robust_adaptivek15_h1__age100__cross0.1"
TEAM17=(3,4,6,7,8,9,10,13,17,18,19,20,22,23,24,26,29)
NUM5=("age","SBP","TG","HDL","ALT")
METHODS=("P01","P03_Cnn")
POLICIES=("P04","Q99_delta005","Q95_delta010","Q99_all_positive","Q99_naive_min")
SEED=20261009
N_Q_OTHER=5
REF_SCORE=.34919703643719757

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savej(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def loadmod(p,name):
    s=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def percentile(normal_b,other,normal):
    good=np.asarray(normal_b[normal],float)
    n=len(good)
    rank=rankdata(good,method="average")/n
    sorted_good=np.sort(good)
    alt=np.asarray(other,float)
    query=(np.searchsorted(sorted_good,alt,"left")+
           np.searchsorted(sorted_good,alt,"right"))/(2*n)
    return rank,query

def two_matrices(base,b_signals,q_signals,normal):
    """Same 7-column P01 and P03 scores for held-out public Q and own B."""
    n=len(normal)
    r={},{}
    for kind in METHODS:
        vb=np.asarray(b_signals[kind],float)
        vq=np.asarray(q_signals[kind],float)
        assert vb.shape[0]==n and vq.shape[1]==vb.shape[1]
        rankb=[];rankq=[]
        for j in range(vb.shape[1]):
            a,z=percentile(vb[:,j],vq[:,j],normal)
            rankb.append(a);rankq.append(z)
        br=np.column_stack(rankb);qr=np.column_stack(rankq)
        if kind=="P01":
            br=br-.1*base.rivals(br,np.median)
            qr=qr-.1*base.rivals(qr,np.median)
        # P03 distance scores will be combined with P01 separately.
        r[0][kind],r[1][kind]=br,qr
    r[0]["P03_Cnn"]=r[0]["P01"]+.1*r[0]["P03_Cnn"]
    r[1]["P03_Cnn"]=r[1]["P01"]+.1*r[1]["P03_Cnn"]
    return r

def nearest_both(common,frame,A,cs,teams):
    scale=common.scales(A,NUM5)
    q=common.numbers(frame,NUM5)/scale
    return np.column_stack([
        -cdist(q,common.numbers(cs[t],NUM5)/scale,"sqeuclidean").min(axis=1)
        for t in teams])

def assign_p04(root):
    g=pd.read_csv(root/GROUP).set_index("team")
    threshold=float(json.loads((root/THRESH).read_text())["c_home_median"])
    assign={}
    for t in TEAM17:
        z=g.loc[t]
        assign[t]="P01" if int(z.aia_imprint_high)!=1 else (
            "KDEA" if float(z.C_home_fraction)>=threshold else "KDEB")
    assert sorted(t for t,m in assign.items() if m=="KDEA")==[3,7,8,13,23]
    return assign

def evaluate_q_risk(b,q,normal,targets,hold):
    out=[]
    for m in METHODS:
        for j,t in enumerate(targets):
            if t==hold:continue
            c=np.asarray(b[m][:,j]);alt=np.asarray(q[m][:,j])
            threshold99=float(np.quantile(c,.99,method="higher"))
            threshold95=float(np.quantile(c,.95,method="higher"))
            up99=int(np.count_nonzero(alt>threshold99))
            up95=int(np.count_nonzero(alt>threshold95))
            out.append({
                "heldout_Q_cohort":hold,"target":t,"method":m,
                "holdout_nonmember_proxy_n":len(alt),
                "risk99":up99/len(alt),"risk95":up95/len(alt),
                "nonmember_rank_mean":float(alt.mean()),
                "FP99":up99,"FP95":up95,
            })
    return out

def choose(risk,team,policy):
    if policy=="P04":return "P01"
    dd=risk[risk.target.eq(team)].copy()
    a=dd[dd.method=="P01"].set_index("heldout_Q_cohort")
    z=dd[dd.method=="P03_Cnn"].set_index("heldout_Q_cohort").reindex(a.index)
    assert len(a)>=4 and len(z)==len(a)
    d99=a.risk99-z.risk99
    d95=a.risk95-z.risk95
    if policy=="Q99_delta005":
        allowed=float(d99.mean())>=.005 and float(d95.mean())>=-.005
    elif policy=="Q95_delta010":
        allowed=float(d95.mean())>=.010 and float(d99.mean())>=-.003
    elif policy=="Q99_all_positive":
        allowed=bool((d99>=.001).all() and (d95>=-.015).all())
    elif policy=="Q99_naive_min":
        allowed=float(d99.mean())>1e-12
    else:raise ValueError(policy)
    return "P03_Cnn" if allowed else "P01"

def run(root,out):
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    ts=time.perf_counter()
    archive=loadmod(root/KDEA_SRC,"p07_archived")
    base=archive.import_base(root)
    common,calib,basefolder,paths,b,a,qs,cs,teams,private,rare,normal=base.load_inputs(root)
    assert len(teams)==20 and len(qs)==21 and int(normal.sum())==875
    qsids=sorted(qs)
    other=[t for t in qsids if t!=5]
    chosen=sorted(np.random.default_rng(SEED).choice(other,size=N_Q_OTHER,replace=False).tolist())
    probes=sorted({5,*chosen})
    assert len(probes)==6
    print("Q COHORT HOLDOUTS fixed by seed",probes,flush=True)
    qi=list(common.QI7)
    qbank=pd.concat([qs[k][qi] for k in qsids],ignore_index=True)
    qidx={};position=len(b)
    for h in qsids:
        qidx[h]=slice(position,position+len(qs[h]))
        position+=len(qs[h])
    whole=pd.concat([b[qi],qbank],ignore_index=True)
    assert len(whole)==position
    # Each row of GenLRA is a deterministic per-candidate function of A/C,
    # independent of other candidates batched with it.
    rawB=[];rawQ={}
    for j,t in enumerate(teams):
        raw=common.core(a,whole,cs[t],common.NUM7,(10,))[10]
        rawB.append(raw[:len(b)])
        rawQ[t]={h:raw[s] for h,s in qidx.items()}
        print("full-public-Q raw",j+1,"/",len(teams),"target",t,flush=True)
    rawB=np.column_stack(rawB)
    del raw
    selectframe=pd.concat([b[qi]]+[qs[k][qi] for k in probes],ignore_index=True)
    # KDE A needs BMI/FPG/time/etc missing from public Q: compute on B only.
    kdeB,h=archive.scores_KDEA(common,b,a,cs,teams)
    nn=nearest_both(common,selectframe,a,cs,teams)
    nnB=nn[:len(b)]
    nnProbe={}
    offset=len(b)
    for h in probes:
        n=len(qs[h])
        nnProbe[h]=nn[offset:offset+n]
        offset+=n
    assert offset==len(selectframe)
    fullCal=[]
    leaveCal={}
    discarded=[]
    # Full P01 calibration includes all other public Q cohorts except Q_t.
    # Heldout h is excluded at COHORT level including overlapping identical
    # QI7 records in other controls. This is the primary leakage boundary.
    for j,t in enumerate(teams):
        ctrlids=[k for k in qsids if k!=t]
        controls=pd.concat([qs[k][qi] for k in ctrlids],ignore_index=True)
        nulls=np.concatenate([rawQ[t][k] for k in ctrlids])
        fullCal.append(calib.calibrate(rawB[:,j],b,nulls,controls,["prefecture","sex"],500))
        for h in probes:
            # Even when h == t, calibrate the Q_h column for use in the
            # OTHER-target median correction. h==t is never evaluated as a
            # negative; leaving it raw would corrupt other targets' median.
            leaveids=[k for k in ctrlids if k!=h]
            frame=pd.concat([qs[k][qi] for k in leaveids],ignore_index=True)
            values=np.concatenate([rawQ[t][k] for k in leaveids])
            qkeys=pd.MultiIndex.from_frame(qs[h][qi])
            keep=~pd.MultiIndex.from_frame(frame).isin(qkeys)
            n_removed=int((~keep).sum())
            discarded.append({"target":t,"heldout":h,"cross_cohort_duplicate_QI7_removed":n_removed,
                            "null_train_cohorts":len(leaveids),
                            "null_train_rows":int(keep.sum())})
            frame=frame.iloc[np.flatnonzero(keep)]
            values=values[keep]
            if len(values)<5000: raise RuntimeError("Too few null controls")
            bcal=calib.calibrate(rawB[:,j],b,values,frame,["prefecture","sex"],500)
            hcal=calib.calibrate(rawQ[t][h],qs[h],values,frame,["prefecture","sex"],500)
            leaveCal[t,h]=(bcal,hcal)
        print("completed P01 calibration",j+1,"/",len(teams),"target",t,flush=True)
    fullCal=np.column_stack(fullCal)
    core_methods={}
    baseb={
      "P01":fullCal,
      "P03_Cnn":nnB
    }
    # private probes for in-sample diagnostic not used in policies.
    privateq={
      "P01":fullCal[private],
      "P03_Cnn":nnB[private]
    }
    fullB,priv=two_matrices(base,baseb,privateq,normal)
    kderanks=base.ranks(kdeB[normal])
    fullB["KDEA"]=kderanks-.1*base.rivals(kderanks,np.median)
    risks=[]
    choices_by_hold={}
    for h in probes:
        bcal=np.column_stack([leaveCal[t,h][0] for t in teams])
        hcal=np.column_stack([leaveCal[t,h][1] for t in teams])
        bsignal={"P01":bcal,"P03_Cnn":nnB}
        hsignal={"P01":hcal,"P03_Cnn":nnProbe[h]}
        rv,rn=two_matrices(base,bsignal,hsignal,normal)
        risks.extend(evaluate_q_risk(rv,rn,normal,teams,h))
        choices_by_hold[h]=len(rn["P01"])
    risks_df=pd.DataFrame(risks)
    assert (risks_df.groupby(["heldout_Q_cohort","target"]).method.nunique()==2).all()
    risks_df.to_csv(out/"cohort_heldout_null_risk.csv",index=False,float_format="%.17g")
    pd.DataFrame(discarded).to_csv(out/"Q_cross_cohort_duplicate_audit.csv",index=False)
    assignments=assign_p04(root)
    choices=[]
    for policy in POLICIES:
        current=dict(assignments)
        for t in TEAM17:
            if assignments[t]=="P01":current[t]=choose(risks_df,t,policy)
        choices.append((policy,current))
    pd.DataFrame([{"policy":name,"team":t,
        "P04_family":assignments[t],"selected":sel[t],
        "method_changed":int(assignments[t]!=sel[t])}
        for name,sel in choices for t in TEAM17]).to_csv(
        out/"frozen_cohort_decisions.csv",index=False)
    print("FROZEN public Q cohorts scores and choices. NOW read official pool_ids.",flush=True)
    frozen=datetime.now(timezone.utc).isoformat()

    # Official truth post-hoc only.
    y,audit,truthpaths=base.audit_truth(basefolder,b,private,rare,teams)
    normalScores={}
    for m,values in fullB.items():
        conf=base.confidence(values,normal,private,rare)
        normalScores[m]={t:float(base.metrics(conf[:,j],y[:,j])["tpr_0.01"])
                         for j,t in enumerate(teams)}
    hist=pd.read_csv(root/HIST,encoding="utf-8-sig")
    old=hist[(hist.condition=="full")&np.isclose(hist.fpr,.01)&
             hist.team.isin(TEAM17)].set_index(["team","method"]).tpr
    diff=max(abs(normalScores[m][t]-old.loc[(t,arch)])
             for t in TEAM17 for m,arch in (("P01",P01),("KDEA",KDEA)))
    if diff>1e-12:raise RuntimeError("P01/KDE A full score regression mismatch "+str(diff))
    archiveB={t:float(old.loc[(t,KDEB)]) for t in TEAM17 if assignments[t]=="KDEB"}

    book=load_workbook(root/OFFICIAL,read_only=True,data_only=True)
    m={};aia={}
    for r in list(book["攻撃_標的別_MIA"].values)[1:]:
        if r[0]==5:m[int(r[1])]=(float(r[4] or 0),float(r[5] or 0))
    for r in list(book["攻撃_標的別_AIA"].values)[1:]:
        if r[0]==5:aia[int(r[1])]=float(r[9] or 0)
    assert len(m)==len(aia)==20
    official=sum(max(*m[t])+aia[t] for t in m)/20
    assert abs(official-.3297953270354882)<1e-12

    summary=[];bytarget=[]
    for policy,sel in choices:
        vals={}
        for t in m:
            if t in (1,2):vals[t]=m[t][0]
            elif t==12:vals[t]=normalScores["P01"][t]
            elif sel[t]=="KDEB":vals[t]=archiveB[t]
            else:vals[t]=normalScores[sel[t]][t]
        power=sum(max(vals[t],m[t][1])+aia[t] for t in m)/20
        summary.append({
          "policy":policy,"hypothetical_prelim_attack":power,
          "delta_vs_P04":power-REF_SCORE,
          "gap_to_official_prelim_third":.352845805596271-power,
          "MIA_full_avg20":sum(vals.values())/20,
          "TPR010_hits20":sum(v>=.1-1e-12 for v in vals.values()),
          "number_KDEA_family_changes":sum(sel[t]!=assignments[t] for t in TEAM17)
        })
        for t in TEAM17:
            before=archiveB[t] if assignments[t]=="KDEB" else normalScores[assignments[t]][t]
            after=archiveB[t] if sel[t]=="KDEB" else normalScores[sel[t]][t]
            bytarget.append({"team":t,"policy":policy,"original_P04":assignments[t],
                             "selected":sel[t],"P04_tpr":before,"new_tpr":after,
                             "delta":after-before})
    s=pd.DataFrame(summary);bys=pd.DataFrame(bytarget)
    assert abs(s.loc[s.policy=="P04","hypothetical_prelim_attack"].item()-REF_SCORE)<1e-12
    s.to_csv(out/"counterfactual_summary.csv",index=False,float_format="%.17g")
    bys.to_csv(out/"target_results.csv",index=False,float_format="%.17g")

    source_sha={str(p.relative_to(root)):sha(p) for p in paths+[root/BASE,root/KDEA_SRC,
       root/HIST,root/GROUP,root/THRESH,root/OFFICIAL,Path(__file__).resolve()]}
    savej(out/"protocol.json",{
        "status":"exploratory public prelim; P04 family and P07 policy set post-hoc",
        "scope":"20 original target scores; KDE-B fixed prior aggregate",
        "Q_cohorts_available":qsids,"Q_heldout_selected":probes,
        "Q_heldout_random_seed":SEED,"n_other_random_cohorts":N_Q_OTHER,
        "heldout_QI7_duplicate_removal":True,
        "risk_metric":"heldout-Q exceed top1pct/top5pct among 875 normal self candidates, NOT official FPR",
        "Q_target_h_excluded_from_test":True,
        "score_models_Q_public":METHODS,"KDEA_Q_risk_not_evaluable_missing_columns":True,
        "P04_P01_only_policy":True,
        "KDEB_not_score_recomputed":True,
        "prediction_and_choices_frozen_utc":frozen,
        "released_pool_ids_read_after_freeze":True,
        "max_official_P01_KDEA_regression_abs":diff,
        "official_team1_team2_TPR_retained":True,
        "team12_uses_P01_model":True,
        "prior_MIA_rare_AIA_scores_frozen":True,
        "no_main_inputs":True,"no_submissions":True,
        "no_person_scores_saved":True,
        "input_sha256":source_sha,
        "published_truth_sha256_after_freeze":{str(p.relative_to(root)):sha(p) for p in truthpaths}
    })
    print(s.to_string(index=False))
    print(pd.DataFrame(discarded).groupby("heldout").cross_cohort_duplicate_QI7_removed.agg(["sum","max"]).to_string())
    print(pd.DataFrame([{"policy":name,"P01_changes":{t:sel[t] for t in TEAM17 if assignments[t]=="P01"}} for name,sel in choices]).to_string(index=False))
    print("seconds",time.perf_counter()-ts)

if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--root",type=Path,required=True)
    arg.add_argument("--out",type=Path,required=True)
    a=arg.parse_args()
    run(a.root.resolve(),a.out.resolve())
