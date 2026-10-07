"""64仮Cの経験順位と同じ帰無分布を、非復元抽出の超幾何式で計算する。
既存5候補の結果は変更しない。公開予備truthは全44Cの予測保存後に採点。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
from pathlib import Path
import sys,json,time,hashlib,itertools,math
from fractions import Fraction
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.spatial.distance import cdist
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
import 人物較正 as original
core=original.core;stable=original.stable
distance=original.distance
ROOT=original.ROOT
CFG=json.loads((PARENT/"固定条件.json").read_text())
B=CFG["calibration_repeats"]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def probability(K,N,n):
    K=np.asarray(K,dtype=int);assert N>=n>0 and ((K>=0)&(K<=N)).all()
    eligible=N-K>=n;lognone=np.full(K.shape,-np.inf)
    logden=gammaln(N+1)-gammaln(n+1)-gammaln(N-n+1)
    lognone[eligible]=gammaln(N-K[eligible]+1)-gammaln(n+1)-gammaln(N-K[eligible]-n+1)-logden
    # K=0の確率は厳密に0。丸めが正のlog確率を作らないよう制限。
    lognone[K==0]=0.
    lognone=np.minimum(lognone,0.)
    return np.clip(-np.expm1(lognone),0.,1.)
def check_enumeration():
    rows=[]
    for distances in ([0.,0.,.2,.2,1.3,1.3],[0.,.1,.1,.5,.5,1.,2.]):
        N=len(distances)
        for n in [1,2,N//2,N]:
            subsets=list(itertools.combinations(range(N),n))
            for d in [-1.,0.,.1,.2,.5,1.3,2.,3.]:
                K=sum(x<=d for x in distances)
                empirical=Fraction(sum(min(distances[i] for i in ix)<=d for ix in subsets),len(subsets))
                exact=Fraction(1,1)-Fraction(math.comb(N-K,n) if N-K>=n else 0,math.comb(N,n))
                p=float(probability(np.array([K]),N,n)[0])
                assert empirical==exact and abs(p-float(empirical))<1e-12
                rows.append(dict(N=N,n=n,K=K,distance_boundary=d,subsets=len(subsets),enumerated_fraction=str(empirical),formula_fraction=str(exact),maximum_float_error=abs(p-float(empirical))))
    large=[]
    for N,n in [(1200,500),(1300,620),(2571,1049)]:
        for K in [0,1,10,N-n,N-n+1,N]:
            exact=Fraction(1)-Fraction(math.comb(N-K,n) if N-K>=n else 0,math.comb(N,n))
            p=float(probability(np.array([K]),N,n)[0])
            large.append(dict(N=N,n=n,K=K,absolute_error=abs(p-float(exact))))
    assert max(x["absolute_error"] for x in large)<5e-11
    writej(HERE/"組合せ全列挙との照合.json",dict(small_pool=rows,large_integer_combination=large,
        small_max_error=max(x["maximum_float_error"] for x in rows),large_max_error=max(x["absolute_error"] for x in large)))
def pair_from_pool(dc,dpool,qs,cs,poolsex):
    result=np.empty_like(dc);ks=np.empty_like(dc,dtype=int);metadata=[]
    for row in range(len(dc)):
        sex=qs[row];finite=np.sort(dpool[row,np.isfinite(dpool[row])]);N=len(finite);n=int((cs==sex).sum())
        K=np.searchsorted(finite,dc[row],side="right");p=probability(K,N,n)
        result[row]=-np.log((1+B*p)/(B+1));result[row,~np.isfinite(dc[row])]=-1e4
        ks[row]=K
    for sex in np.unique(qs):
        a=qs==sex;b=cs==sex
        metadata.append(dict(sex=int(sex),N_pool=int((poolsex==sex).sum()),n_C=int(b.sum()),
            q_count=int(a.sum()),K_min=int(ks[a][:,b].min()),K_max=int(ks[a][:,b].max())))
    assert np.isfinite(result).all()
    return result,metadata
def direct_empirical(dc,dpool,qs,calix):
    nullmin=np.stack([dpool[:,ix].min(axis=1) for ix in calix],axis=1)
    result=np.empty_like(dc)
    for row in range(len(dc)):
        count=np.searchsorted(np.sort(nullmin[row]),dc[row],side="right")
        result[row]=-np.log((1+count)/(len(calix)+1))
    result[~np.isfinite(dc)]=-1e4
    return result
def predict():
    started=time.time();check_enumeration()
    work=HERE/"作業用";work.mkdir(exist_ok=True);(HERE/"集計").mkdir(exist_ok=True)
    entries=[];perms=[];equal=[];meta=[]
    parent=json.loads((PARENT/"予測実行記録.json").read_text())
    parent_entries={(x["phase"],x["team"]):x for x in parent["predictions"]}
    for phase,folder,bgp in original.SETS:
        bg=core.read(bgp);raw=core.raw(bg);scale=np.maximum(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0),1e-5)
        order=np.random.default_rng(CFG["split_seed"]+(0 if phase=="prelim" else 1)).permutation(len(bg))
        pool=bg.iloc[order[:len(bg)//2]].reset_index(drop=True);ps=pool.sex.to_numpy()
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem[2:])):
            k=int(cp.stem[2:]);q=core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True);c=core.read(cp)
            qs,cs=q.sex.to_numpy(),c.sex.to_numpy()
            dc=distance(q,c,scale);dp=distance(q,pool,scale)
            tie=CFG["fixed_choices"]["tie_epsilon"]*np.minimum(cdist(core.raw(q)/scale,core.raw(c)/scale,"sqeuclidean")/5,1000.)
            exact,metadata=pair_from_pool(dc,dp,qs,cs,ps)
            chosen=stable.select(exact-tie,dc,qs,cs,"sex_quota")
            seed=CFG["seed"]+k*1000+(0 if phase=="prelim" else 100000)
            calix=original.samples(ps,cs,B,seed)
            empirical=direct_empirical(dc,dp,qs,calix)
            emchosen=stable.select(empirical-tie,dc,qs,cs,"sex_quota")
            prev=parent_entries[(phase,k)];assert digest(PARENT/prev["path"])==prev["sha256"]
            old=np.load(PARENT/prev["path"]);mi=original.METHODS.index("empirical_upper_rank")
            diff=int((old["real"][0,mi]!=emchosen).sum());diff2=int((old["real"][1,mi]!=emchosen).sum())
            assert diff==0 and diff2==0
            equal.append(dict(phase=phase,team=k,split_CDF_vs_direct_rank_changes=diff,full_CDF_vs_direct_rank_changes=diff2,
              exact_vs_64_repeat_rank_changes=int((chosen!=emchosen).sum()),selected=int(chosen.sum())))
            meta.extend([dict(phase=phase,team=k,**m) for m in metadata])
            for repeat in range(CFG["permutation_repeats"]):
                rng=np.random.default_rng(CFG["seed"]+700000+k*100+repeat);a,b=rng.permutation(len(q)),rng.permutation(len(c))
                perm=stable.select((exact-tie)[a][:,b],dc[a][:,b],qs[a],cs[b],"sex_quota")[np.argsort(a)]
                perms.append(dict(phase=phase,team=k,repeat=repeat,selected_changed=int((chosen!=perm).sum())))
            null=[]
            if phase=="prelim":
                for ix in original.samples(ps,cs,CFG["null_evaluation_repeats"],seed+500000):
                    dnull=dp[:,ix];snull,_=pair_from_pool(dnull,dp,qs,ps[ix],ps)
                    ntie=CFG["fixed_choices"]["tie_epsilon"]*np.minimum(cdist(core.raw(q)/scale,core.raw(pool.iloc[ix])/scale,"sqeuclidean")/5,1000.)
                    null.append(stable.select(snull-ntie,dnull,qs,ps[ix],"sex_quota"))
            saved=work/f"{phase}_{k}.npz";np.savez_compressed(saved,real=chosen,null=np.stack(null) if null else np.empty((0,)),empirical64=emchosen)
            entries.append(dict(phase=phase,team=k,path=str(saved.relative_to(HERE)),sha256=digest(saved)))
            print(json.dumps(dict(stage="exact_predict",phase=phase,team=k,seconds=round(time.time()-started,1))),flush=True)
    pd.DataFrame(perms).to_csv(HERE/"集計/厳密版_行順変更.csv",index=False)
    pd.DataFrame(equal).to_csv(HERE/"集計/CDF経由経験順位と距離順位の一致.csv",index=False)
    pd.DataFrame(meta).to_csv(HERE/"集計/sex群のNとn.csv",index=False)
    assert pd.DataFrame(perms).selected_changed.max()==0
    writej(HERE/"厳密予測実行記録.json",dict(code_sha256=digest(__file__),additional_conditions_sha256=digest(HERE/"追加固定条件.json"),
        parent_fixed_conditions_sha256=digest(PARENT/"固定条件.json"),parent_prediction_receipt_sha256=digest(PARENT/"予測実行記録.json"),
        prediction_entries=entries,input_sha256=dict(core.HASHES),seconds=time.time()-started,
        truth_used_for_predictions=False,main_truth_used=False,is_new_independent_validation=False,submission_created=False))
def score():
    receipt=json.loads((HERE/"厳密予測実行記録.json").read_text());assert receipt["code_sha256"]==digest(__file__)
    td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"));rows=[];truthhash={}
    for x in receipt["prediction_entries"]:
        p=HERE/x["path"];assert digest(p)==x["sha256"]
        if x["phase"]!="prelim":continue
        k=x["team"];tp=td/f"aia_truth_{k}.csv";truth=core.read(tp).sort_values("challenge_row_id").reset_index(drop=True);truthhash[str(tp)]=digest(tp)
        data=np.load(p)
        rows.append(dict(team=k,method="exact_hypergeometric",source="real_C",repeat=-1,**core.scores(np.where(data["real"],9.5,0),truth)))
        for repeat,pred in enumerate(data["null"]):
            rows.append(dict(team=k,method="exact_hypergeometric",source="BG_null_C",repeat=repeat,**core.scores(np.where(pred,9.5,0),truth)))
    table=pd.DataFrame(rows);table.to_csv(HERE/"集計/厳密版_対象別とnull反復.csv",index=False)
    originaltable=pd.read_csv(PARENT/"集計/予備戦_対象別とnull反復.csv")
    combined=[]
    for context in original.CONTEXTS:
        extra=table.copy();extra["context"]=context
        combined.append(extra)
    allrows=pd.concat([originaltable,pd.concat(combined)],ignore_index=True)
    aggs=[]
    for name,exclude in [("prelim20",[5]),("processed18",[1,2,5])]:
        g=allrows[~allrows.team.isin(exclude)]
        agg=g.groupby(["context","method","source"])[original.METRICS].mean().reset_index();agg.insert(0,"group",name);aggs.append(agg)
    pd.concat(aggs).to_csv(HERE/"集計/厳密版を追加した6候補平均.csv",index=False)
    loo=[]
    for context in original.CONTEXTS:
        real=allrows[(allrows.context==context)&allrows.source.eq("real_C")&~allrows.team.isin([1,2,5])]
        methods=original.METHODS+["exact_hypergeometric"];pivot=real.pivot(index="team",columns="method",values="r_all").reindex(columns=methods)
        for held in pivot.index:
            train=pivot.drop(index=held).mean();selected=methods[int(np.argmax(train.to_numpy()))]
            loo.append(dict(context=context,held_team=held,selected_method=selected,held_R=float(pivot.loc[held,selected]),held_baseline_R=float(pivot.loc[held,"baseline"])))
    pd.DataFrame(loo).to_csv(HERE/"集計/6候補_leave_team_out方式選択.csv",index=False)
    writej(HERE/"厳密採点実行記録.json",dict(code_sha256=digest(__file__),prediction_receipt_sha256=digest(HERE/"厳密予測実行記録.json"),
       public_prelim_truth_sha256=truthhash,additional_conditions_sha256=digest(HERE/"追加固定条件.json"),
       main_truth_used=False,is_new_independent_validation=False,after_observing_initial5_results=True,submission_created=False))
    print(pd.concat(aggs).query("group=='processed18' and method=='exact_hypergeometric'").to_string(index=False),flush=True)
    print(pd.DataFrame(loo).groupby("context")[["held_R","held_baseline_R"]].mean().to_string(),flush=True)
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["predict","score","all"],default="all");args=parser.parse_args()
    if args.stage in ["predict","all"]:predict()
    if args.stage in ["score","all"]:score()
