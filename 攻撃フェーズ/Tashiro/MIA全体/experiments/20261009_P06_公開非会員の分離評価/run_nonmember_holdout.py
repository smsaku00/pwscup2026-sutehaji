"""P06: Public-background 80/20 negative holdout for P01, KDE-A, and P03.

Compare method false-alarm rates on 1,028 strictly HELDOUT A_bg public
records against the top 1% and 5% of 875 ordinary B_self candidate scores.
P01 calibration is trained using other teams' public AIA challenges; heldout
A_bg never appears in the model's KDE/GenLRA background reference. A separate
163-Qself-known-nonmember diagnostic is IN-SAMPLE for P01 and NOT used for
decisions. Policies are defined and predictions frozen before pool_ids truth
is read. P04 group + these policies were conceived after examining preliminary
results; LOOTO alone cannot cancel this upstream selection bias.

KDE-B full scores are inaccessible in the retained repository: keep its
previously published official-prelim aggregate fixed; do not claim we checked
its false positive rate. The full virtual attack score also freezes official
preliminary Team1/2, rare MIA, AIA, and substitutes P01 for Team12.
No individual membership scores, output confidence or submissions saved.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from openpyxl import load_workbook

BASE="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py"
KDEA_SRC="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_TPR010到達と加工特徴転移検証/run_archived_kdeA_mix.py"
PRELIM_METHODS="攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_非学習方式と加工特徴_背景耐性/固定方式_全背景評価.csv"
PUBLIC_GROUP="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/public_target_groups.csv"
PUBLIC_CUTS="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/group_thresholds.json"
OFFICIAL="公式データ/予備戦/結果/公開用結果表_20260909.xlsx"
P01_ARCH="Q_prefecture_sex_shrink500_cross01"
KDEA_ARCH="q13_median_iqr_sex0_kde_ratio_h2__raw__cross0.1"
KDEB_ARCH="q7_robust_adaptivek15_h1__age100__cross0.1"
MODELS=("P01","KDEA","P03_Cnn")
POLICIES=("P04","risk99_0005","risk95_0010","risk99_all3")
TARGET17=(3,4,6,7,8,9,10,13,17,18,19,20,22,23,24,26,29)
SEEDS=(20261009,20261010,20261011)
NUM5=("age","SBP","TG","HDL","ALT")
P04_EXPECTED=.34919703643719757

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def dyn(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def percentiles(scores_b,scores_hold,normal):
    """Rank candidate training scores; compare heldout score on same scale."""
    nr=int(normal.sum())
    rb=np.empty((nr,scores_b.shape[1]));rn=np.empty((len(scores_hold),scores_b.shape[1]))
    for j in range(scores_b.shape[1]):
        x=scores_b[normal,j]
        rb[:,j]=rankdata(x,method="average")/nr
        ordered=np.sort(x);v=scores_hold[:,j]
        rn[:,j]=(np.searchsorted(ordered,v,"left")+
                 np.searchsorted(ordered,v,"right"))/(2*nr)
    return rb,rn


def formula(base,calb,calnull,kdeb,kdenull,distb,distnull,normal):
    xr,xn=percentiles(calb,calnull,normal)
    kr,kn=percentiles(kdeb,kdenull,normal)
    nr,nn=percentiles(-distb,-distnull,normal)
    p01=xr-.1*base.rivals(xr,np.median)
    p01n=xn-.1*base.rivals(xn,np.median)
    ka=kr-.1*base.rivals(kr,np.median)
    kan=kn-.1*base.rivals(kn,np.median)
    return (
       {"P01":p01,"KDEA":ka,"P03_Cnn":p01+.1*nr},
       {"P01":p01n,"KDEA":kan,"P03_Cnn":p01n+.1*nn})


def dist_to_C(common,query,a,cs,teams):
    scale=common.scales(a,NUM5)
    q=common.numbers(query,NUM5)/scale
    return np.column_stack([
        cdist(q,common.numbers(cs[t],NUM5)/scale,"sqeuclidean").min(axis=1)
        for t in teams])


def p01_public_holdout(common,calib,b,heldout,a,qs,cs,teams):
    """B and holdout A evaluated with the same P01 Q-control calibration."""
    n=len(b);h=len(heldout)
    xb=[];xn=[]
    controls_per_target=[pd.concat([
        qs[k][list(common.QI7)] for k in sorted(qs) if k!=t
        ],ignore_index=True) for t in teams]
    for j,t in enumerate(teams):
        controls=controls_per_target[j]
        whole=pd.concat([b[list(common.QI7)],heldout[list(common.QI7)],controls],ignore_index=True)
        z=common.core(a,whole,cs[t],common.NUM7,(10,))[10]
        assert len(z)==n+h+len(controls)
        null=z[n+h:]
        xb.append(calib.calibrate(z[:n],b,null,controls,["prefecture","sex"],500))
        xn.append(calib.calibrate(z[n:n+h],heldout,null,controls,["prefecture","sex"],500))
        if (j+1)%5==0:print("public-heldout-P01",j+1,"of",len(teams),flush=True)
    return np.column_stack(xb),np.column_stack(xn)


def score_risk(scores,nulls,view,scope,teams):
    r=[]
    for m in MODELS:
        for j,t in enumerate(teams):
            b=scores[m][:,j];n=nulls[m][:,j]
            top99=np.quantile(b,.99,method="higher")
            top95=np.quantile(b,.95,method="higher")
            fp99=int((n>top99).sum());fp95=int((n>top95).sum())
            r.append({
              "view":view,"source":scope,"team":t,"method":m,
              "null_count":len(n),"candidate_count":len(b),
              "top99_null_count":fp99,"top95_null_count":fp95,
              "risk99":fp99/len(n),"risk95":fp95/len(n),
              "null_mean_rank":float(n.mean())})
    return pd.DataFrame(r)


def determine_p04(root):
    g=pd.read_csv(root/PUBLIC_GROUP).set_index("team")
    c=json.loads((root/PUBLIC_CUTS).read_text())
    cutoff=float(c["c_home_median"])
    assignment={}
    for t in TARGET17:
        r=g.loc[t]
        if int(r["aia_imprint_high"])==0:assignment[t]="P01"
        elif r["C_home_fraction"]>=cutoff:assignment[t]="KDEA"
        else:assignment[t]="KDEB"
    assert [t for t in TARGET17 if assignment[t]=="KDEA"]==[3,7,8,13,23]
    return assignment


def choose(risks,t,policy):
    """Public-A_BG-only decision; no preliminary target-membership labels."""
    if policy=="P04":return "KDEA"
    sub=risks[(risks.team==t)&(risks.source=="A_bg_20pct_heldout")]
    a=sub[sub.method=="KDEA"].set_index("view")
    assert len(a)==3
    selected=[]
    for method in ("P01","P03_Cnn"):
        z=sub[sub.method==method].set_index("view").loc[a.index]
        delta99=a.risk99-z.risk99
        delta95=a.risk95-z.risk95
        if policy=="risk99_0005":
            good=float(delta99.mean())>=.005 and float(delta95.mean())>=-.005
            rank=(delta99.mean(),delta95.mean())
        elif policy=="risk95_0010":
            good=float(delta95.mean())>=.01 and float(delta99.mean())>=-.003
            rank=(delta95.mean(),delta99.mean())
        elif policy=="risk99_all3":
            good=bool((delta99>=.001).all() and (delta95>=-.015).all())
            rank=(delta99.mean(),delta95.mean())
        else:raise ValueError(policy)
        if good:selected.append((tuple(map(float,rank)),method))
    return max(selected,key=lambda x:x[0])[1] if selected else "KDEA"


def run(root,out):
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    started=time.perf_counter()
    archive=dyn(root/KDEA_SRC,"p06_archival_kdeA")
    base=archive.import_base(root)
    common,calib,basefolder,paths,b,a,qs,cs,teams,private,rare,normal=base.load_inputs(root)
    assert len(teams)==20 and int(normal.sum())==875
    assert int(private.sum())==163 and int((private&~rare).sum())==150
    p04=determine_p04(root)
    riskpanels=[];folds=[];digest=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        held=np.sort(rng.choice(len(a),len(a)//5,replace=False))
        train=np.setdiff1d(np.arange(len(a)),held)
        train_a=a.iloc[train].copy();test_a=a.iloc[held].copy()
        calb,caln=p01_public_holdout(common,calib,b,test_a,train_a,qs,cs,teams)
        combo=pd.concat([b,test_a],ignore_index=True)
        kde,h=archive.scores_KDEA(common,combo,train_a,cs,teams)
        ds=dist_to_C(common,combo,train_a,cs,teams)
        m,n=formula(base,calb,caln,kde[:len(b)],kde[len(b):],
                    ds[:len(b)],ds[len(b):],normal)
        key=f"split80_seed{seed}"
        riskpanels.append(score_risk(m,n,key,"A_bg_20pct_heldout",teams))
        folds.append((key,m))
        for kind,score in m.items():
            digest.append({"view":key,"method":kind,
               "sha256":hashlib.sha256(np.ascontiguousarray(score,dtype="<f8").tobytes()).hexdigest()})
        print("BACKGROUND-SPLIT-COMPLETED",key,"duration",round(time.perf_counter()-started,2),flush=True)

    # Full model: official-score-equivalent P01/KDE A, P03-nearest in all20 scope.
    _,fullcal=base.raw_scores(common,calib,b,a,qs,cs,teams)
    fullkde,h=archive.scores_KDEA(common,b,a,cs,teams)
    fulld=dist_to_C(common,b,a,cs,teams)
    # Entire B under full calibration; private Q-self 163 rows are in the Q
    # calibration bank, so this is an IN-SAMPLE diagnostic only.
    ascore,privscore=formula(base,fullcal,fullcal[private],
                            fullkde,fullkde[private],
                            fulld,fulld[private],normal) if False else ({},{})
    # percentile normalization of a mixed B+null query must use B[normal].
    # Passing full B+private data to formula with full-score matrix and
    # a separate private-only rank calculation avoids double-normal indexing.
    p01n,p01priv=percentiles(fullcal,fullcal[private],normal)
    kan,kapriv=percentiles(fullkde,fullkde[private],normal)
    nn,npriv=percentiles(-fulld,-fulld[private],normal)
    ascore={"P01":p01n-.1*base.rivals(p01n,np.median),
            "KDEA":kan-.1*base.rivals(kan,np.median)}
    privscore={"P01":p01priv-.1*base.rivals(p01priv,np.median),
               "KDEA":kapriv-.1*base.rivals(kapriv,np.median)}
    ascore["P03_Cnn"]=ascore["P01"]+.1*nn
    privscore["P03_Cnn"]=privscore["P01"]+.1*npriv
    riskpanels.append(score_risk(ascore,privscore,"full",
                                 "own_AIA_Q5_163_IN_CALIBRATION",teams))
    for method,values in ascore.items():
        digest.append({"view":"full","method":method,
            "sha256":hashlib.sha256(np.ascontiguousarray(values,dtype="<f8").tobytes()).hexdigest()})
    riskdf=pd.concat(riskpanels,ignore_index=True)
    riskdf.to_csv(out/"nonmember_risk_panel.csv",index=False,float_format="%.17g")
    # Fixed before published pool_ids truth.
    choices=[]
    for policy in POLICIES:
        selection=dict(p04)
        for t in TARGET17:
            if p04[t]=="KDEA":
                selection[t]=choose(riskdf,t,policy)
        choices.append((policy,selection))
    selected=[{"policy":name,"team":t,"base_P04_family":p04[t],
               "chosen_family":sel[t],"switched":int(p04[t]!=sel[t])}
              for name,sel in choices for t in TARGET17]
    pd.DataFrame(selected).to_csv(out/"selection_frozen_before_truth.csv",index=False)
    writej(out/"prediction_hashes.json",digest)
    freeze=datetime.now(timezone.utc).isoformat()
    print("FROZEN predictions and A_bg-only policy choices; NOW evaluating released prelim truth",flush=True)

    y,aud,truthpaths=base.audit_truth(basefolder,b,private,rare,teams)
    assert int(y[private].sum())==0
    hist=pd.read_csv(root/PRELIM_METHODS,encoding="utf-8-sig")
    hist=hist[(hist.condition=="full")&np.isclose(hist.fpr,.01)&
              hist.team.isin(TARGET17)].set_index(["team","method"]).tpr
    result={}
    for method,values in ascore.items():
        confidence=base.confidence(values,normal,private,rare)
        for j,t in enumerate(teams):
            result[t,method]=float(base.metrics(confidence[:,j],y[:,j])["tpr_0.01"])
    drift=0
    for t in TARGET17:
        for m,hname in (("P01",P01_ARCH),("KDEA",KDEA_ARCH)):
            drift=max(drift,abs(result[t,m]-hist.loc[(t,hname)]))
    if drift>1e-12:raise RuntimeError(f"Archived full P01 KDE A regression mismatch {drift}")
    for t in TARGET17:
        if p04[t]=="KDEB":result[t,"KDEB"]=float(hist.loc[(t,KDEB_ARCH)])

    wb=load_workbook(root/OFFICIAL,read_only=True,data_only=True)
    mia={};aia={}
    for row in list(wb["攻撃_標的別_MIA"].values)[1:]:
        if row[0]==5:mia[int(row[1])]=(float(row[4] or 0),float(row[5] or 0))
    for row in list(wb["攻撃_標的別_AIA"].values)[1:]:
        if row[0]==5:aia[int(row[1])]=float(row[9] or 0)
    assert set(mia)==set(aia)==(set(TARGET17)|{1,2,12})
    old=sum(max(*mia[t])+aia[t] for t in mia)/20
    assert abs(old-.3297953270354882)<1e-12

    summary=[];detail=[]
    for name,sel in choices:
        all_tpr={t:(mia[t][0] if t in (1,2) else
                    result[t,"P01"] if t==12 else result[t,sel[t]])
                 for t in mia}
        attack=sum(max(all_tpr[t],mia[t][1])+aia[t] for t in mia)/20
        summary.append({
            "policy":name,"hypothetical_attack_power":attack,
            "gap_to_prelim_official_third":.352845805596271-attack,
            "delta_from_old_official":attack-old,
            "hits_ge010_20":sum(v>=.1-1e-12 for v in all_tpr.values()),
            "changed_from_P04":sum(sel[t]!=p04[t] for t in TARGET17),
            "mean_MIA_full_20":float(np.mean(list(all_tpr.values())))
        })
        for t in TARGET17:
            detail.append({"team":t,"policy":name,"method":sel[t],
                           "P04_method":p04[t],"tpr":result[t,sel[t]],
                           "P04_tpr":result[t,p04[t]],
                           "delta":result[t,sel[t]]-result[t,p04[t]]})
    sf=pd.DataFrame(summary)
    if abs(sf.loc[sf.policy=="P04","hypothetical_attack_power"].item()
           -.34919703643719757)>1e-12:
        raise RuntimeError("P04 official-prelim virtual score regression changed")
    sf.to_csv(out/"counterfactual_summary.csv",index=False,float_format="%.17g")
    pd.DataFrame(detail).to_csv(out/"by_target.csv",index=False,float_format="%.17g")

    frows=[]
    for view,method_scores in folds:
        pred={m:base.confidence(v,normal,private,rare) for m,v in method_scores.items()}
        for name,sel in choices:
            for t in TARGET17:
                if p04[t]!="KDEA":continue
                j=teams.index(t);chosen=base.metrics(pred[sel[t]][:,j],y[:,j])
                ref=base.metrics(pred["KDEA"][:,j],y[:,j])
                frows.append({"view":view,"policy":name,"team":t,
                        "selected_tpr":chosen["tpr_0.01"],
                        "KDEA_tpr":ref["tpr_0.01"],
                        "difference":chosen["tpr_0.01"]-ref["tpr_0.01"]})
    foldsdf=pd.DataFrame(frows)
    foldsdf.to_csv(out/"matched_background_refits.csv",index=False,float_format="%.17g")
    foldagg=foldsdf.groupby(["policy","view"]).agg(
        mean_delta=("difference","mean"),wins=("difference",lambda x:int((x>1e-12).sum())),
        losses=("difference",lambda x:int((x< -1e-12).sum())),
        baseline_mean=("KDEA_tpr","mean"),selected_mean=("selected_tpr","mean")
    ).reset_index()
    foldagg.to_csv(out/"matched_background_summary.csv",index=False,float_format="%.17g")
    writej(out/"protocol.json",{
        "scope":"preliminary; P01 KDE-A P03 risk on public A_bg 80/20 disjoint folds; original KDE-B fixed aggregate",
        "background_split_seeds":SEEDS,
        "public_null_sample_each_split":len(a)//5,
        "self_AIA_Q5_private_null_n":int(private.sum()),
        "self_AIA_P01_calibration_in_sample":True,
        "Q5_probe_NOT_USED_FOR_SELECTION":True,
        "KDE_B_risk_not_measured":True,
        "selection_only_in_P04_KDEA_group":[t for t in TARGET17 if p04[t]=="KDEA"],
        "policies":POLICIES,
        "predictions_and_selection_frozen_utc":freeze,
        "official_pool_ids_read_only_after_freeze":True,
        "P01_and_KDEA_archived_regression_max_abs_TPR":drift,
        "P04_counterfactual_regression":float(sf.loc[sf.policy=="P04","hypothetical_attack_power"].item()),
        "official_prior_attack":old,
        "method_family_choice_influenced_by_past_preliminary_results":True,
        "Team1_2_assumed_old_TPRs":True,
        "Team12_simulated_P01":True,
        "AIA_rare_old_official_values_frozen":True,
        "main_inputs_used":False,"main_submission_changed":False,
        "person_level_scores_saved":False,
        "input_sha256":{str(p.relative_to(root)):sha(p) for p in
          paths+[root/BASE,root/KDEA_SRC,root/PRELIM_METHODS,root/PUBLIC_GROUP,
                 root/PUBLIC_CUTS,root/OFFICIAL,Path(__file__).resolve()]},
        "truth_sha256_after_freeze":{str(p.relative_to(root)):sha(p) for p in truthpaths}
    })
    print(sf.to_string(index=False))
    print(foldagg.to_string(index=False))
    print(pd.DataFrame(selected)[lambda x:x.base_P04_family=="KDEA"].to_string(index=False))
    print("TIME_SEC",round(time.perf_counter()-started,2))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    arg=ap.parse_args()
    run(arg.root.resolve(),arg.out.resolve())
