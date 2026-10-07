"""固定条件による人物別の偶然近接較正。公開予備正解は予測保存後に採点。
本戦truth・B_self・token不使用。F/ZIP生成・CodaBench操作なし。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import hashlib,json,time,sys,argparse,platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
HERE=Path(__file__).resolve().parent
ROOT=Path("/Users/taisei1105/pws2026-local")
OLD=ROOT/"本戦/攻撃フェーズ/Tashiro/20261007_AIA全体向上の分析"
sys.path.insert(0,str(OLD))
import 同点に依存しない対応 as stable
import 分析 as core
from 対応競合の頑健性 import distance,pair_score
CFG=json.loads((HERE/"固定条件.json").read_text())
METHODS=CFG["methods"]
CONTEXTS=["split_independent","full_reference_overlap"]
EPS=CFG["fixed_choices"]["tie_epsilon"]
METRICS=["r_all","r_rare","r_max","p_m","p_c","hit_m","hit_c"]
SETS=[
("prelim",ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack",ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
("main",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv"),
]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def samples(poolsex,csex,n,seed):
    rng=np.random.default_rng(seed);nsex={sex:int((csex==sex).sum()) for sex in np.unique(csex)}
    answer=[]
    for repeat in range(n):
        ix=[]
        for sex,nc in nsex.items():
            eligible=np.flatnonzero(poolsex==sex);assert len(eligible)>=nc
            ix.extend(rng.choice(eligible,nc,replace=False).tolist())
        answer.append(np.sort(np.asarray(ix,dtype=int)))
    return answer
def stats_from_null(scores,ixs,qs):
    # scores(q,pool) の各仮C内最大値。各qに同じ64仮Cを比較する。
    null=np.stack([scores[:,ix].max(axis=1) for ix in ixs],axis=1)
    mu=null.mean(axis=1);med=np.median(null,axis=1);sd=null.std(axis=1,ddof=1)
    floor=np.empty(len(qs))
    for sex in np.unique(qs):
        a=qs==sex; floor[a]=max(float(np.median(sd[a]))*.25,1e-8)
    return dict(mean=mu,median=med,sd=np.maximum(sd,floor),raw_sd=sd,sd_floor=floor,sorted=np.sort(null,axis=1),null=null)
def modify(raw,stat,method):
    finite=raw>-1e4
    if method=="baseline":result=raw.copy()
    elif method=="subtract_mean":result=raw-stat["mean"][:,None]
    elif method=="subtract_median":result=raw-stat["median"][:,None]
    elif method=="standardize":result=(raw-stat["mean"][:,None])/stat["sd"][:,None]
    elif method=="empirical_upper_rank":
        result=np.empty_like(raw);n=stat["sorted"].shape[1]
        for row in range(len(raw)):
            upper=n-np.searchsorted(stat["sorted"][row],raw[row],side="left")
            result[row]=-np.log((1+upper)/(n+1))
    else:raise ValueError(method)
    result[~finite]=-1e4
    return result
def evaluate(q,c,pair,dc,stat,context,phase,k,out,perm):
    qs,cs=q.sex.to_numpy(),c.sex.to_numpy()
    b=core.raw(c);scale=out["scale"]
    full=cdist(core.raw(q)/scale,b/scale,"sqeuclidean")/5
    tie=EPS*np.minimum(full,1000.)
    chosen=np.stack([stable.select(modify(pair,stat,m)-tie,dc,qs,cs,"sex_quota") for m in METHODS])
    for i,m in enumerate(METHODS):
        changed=out["baseline_full"]!=chosen[i]
        out["diagnostics"].append(dict(phase=phase,team=k,context=context,method=m,
          selected=int(chosen[i].sum()),changed_vs_full_baseline=int(changed.sum()),
          selected_male=int(chosen[i,qs==1].sum()),selected_female=int(chosen[i,qs==0].sum()),
          null_nearest_mu_mean=float(stat["mean"].mean()),null_nearest_sd_mean=float(stat["raw_sd"].mean()),
          sd_floor_applied=int((stat["raw_sd"]<stat["sd_floor"]).sum())))
        for repeat in range(CFG["permutation_repeats"]):
            rng=np.random.default_rng(CFG["seed"]+700000+k*100+repeat)
            a,bp=rng.permutation(len(q)),rng.permutation(len(c))
            raw=modify(pair,stat,m)-tie
            same=stable.select(raw[a][:,bp],dc[a][:,bp],qs[a],cs[bp],"sex_quota")[np.argsort(a)]
            perm.append(dict(phase=phase,team=k,context=context,method=m,repeat=repeat,changed=int((same!=chosen[i]).sum())))
    return chosen
def predict():
    start=time.time();work=HERE/"作業用";work.mkdir(exist_ok=True)
    diag=[];perms=[];repro=[];entry=[]
    for phase,folder,bgp in SETS:
        bg=core.read(bgp);b=core.raw(bg)
        scale=np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0),1e-5)
        order=np.random.default_rng(CFG["split_seed"]+(0 if phase=="prelim" else 1)).permutation(len(bg))
        pool=bg.iloc[order[:len(bg)//2]].reset_index(drop=True)
        ref=bg.iloc[order[len(bg)//2:]].reset_index(drop=True)
        assert set(order[:len(bg)//2]).isdisjoint(set(order[len(bg)//2:]))
        poolsex=pool.sex.to_numpy()
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem[2:])):
            k=int(cp.stem[2:]);q=core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True);c=core.read(cp)
            qs,cs=q.sex.to_numpy(),c.sex.to_numpy()
            oldpair,dc=stable.matrices(q,c,bg);oldchosen=stable.select(oldpair,dc,qs,cs,"sex_quota")
            dpool=distance(q,pool,scale);dbref=distance(q,ref,scale);dbfull=distance(q,bg,scale)
            rawreal=[pair_score(dc,dbref)["continuous"],pair_score(dc,dbfull)["continuous"]]
            rawpool=[pair_score(dpool,dbref)["continuous"],pair_score(dpool,dbfull)["continuous"]]
            seed=CFG["seed"]+k*1000+(0 if phase=="prelim" else 100000)
            calix=samples(poolsex,cs,CFG["calibration_repeats"],seed)
            evalix=samples(poolsex,cs,CFG["null_evaluation_repeats"],seed+500000)
            realchosen=[];nullchosen=[];statistics=[]
            local=dict(scale=scale,diagnostics=diag,baseline_full=oldchosen)
            for ix,context in enumerate(CONTEXTS):
                stat=stats_from_null(rawpool[ix],calix,qs);statistics.append(stat)
                selected=evaluate(q,c,rawreal[ix],dc,stat,context,phase,k,local,perms);realchosen.append(selected)
                if context=="full_reference_overlap":
                    repro.append(dict(phase=phase,team=k,baseline_selection_changed=int((selected[0]!=oldchosen).sum())))
                if phase=="prelim":
                    nullsets=[]
                    for repeat,indices in enumerate(evalix):
                        dnull=dpool[:,indices];snull=rawpool[ix][:,indices];nullc=pool.iloc[indices]
                        tie=EPS*np.minimum(cdist(core.raw(q)/scale,core.raw(nullc)/scale,"sqeuclidean")/5,1000.)
                        nullsets.append(np.stack([stable.select(modify(snull,stat,m)-tie,dnull,qs,poolsex[indices],"sex_quota") for m in METHODS]))
                    nullchosen.append(np.stack(nullsets))
            saved=work/f"{phase}_{k}.npz"
            np.savez_compressed(saved,real=np.stack(realchosen),null=np.stack(nullchosen) if nullchosen else np.empty((0,)),
                old_full=oldchosen,statistics=np.stack([np.stack([s["mean"],s["median"],s["sd"]]) for s in statistics]))
            entry.append(dict(phase=phase,team=k,path=str(saved.relative_to(HERE)),sha256=digest(saved)))
            print(json.dumps(dict(stage="predict",phase=phase,team=k,seconds=round(time.time()-start,1))),flush=True)
    out=HERE/"集計";out.mkdir(exist_ok=True)
    pd.DataFrame(diag).to_csv(out/"公開入力の動作.csv",index=False)
    pd.DataFrame(perms).to_csv(out/"行順変更.csv",index=False)
    pd.DataFrame(repro).to_csv(out/"全44C基準予測の一致.csv",index=False)
    assert pd.DataFrame(repro).baseline_selection_changed.max()==0
    receipt=dict(start_time=time.strftime("%Y-%m-%dT%H:%M:%S%z",time.localtime(start)),seconds=time.time()-start,
        code_sha256=digest(__file__),config_sha256=digest(HERE/"固定条件.json"),
        imports={p.name:digest(p) for p in [OLD/"同点に依存しない対応.py",OLD/"対応競合の分析.py",OLD/"対応競合の頑健性.py",OLD/"分析.py"]},
        input_sha256=dict(core.HASHES),predictions=entry,truth_used_for_predictions=False,main_truth_used=False,
        background_pool_and_reference_disjoint_primary=True,submission_created=False,
        permutations_max=int(pd.DataFrame(perms).changed.max()),baseline_changed_max=0,
        python=sys.executable,python_version=platform.python_version())
    writej(HERE/"予測実行記録.json",receipt)
    return receipt
def score():
    # 予測記録と予測ハッシュを確認してから、公開予備真値を初めて読む。
    rec=json.loads((HERE/"予測実行記録.json").read_text());assert rec["config_sha256"]==digest(HERE/"固定条件.json");assert rec["code_sha256"]==digest(__file__)
    td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    rows=[];checks=[];truthhash={}
    saved=pd.read_csv(OLD/"集計/同点修正_対象別.csv")
    saved=saved[saved.method.eq("stable_matching/sex_quota/long75")].set_index("team")
    for x in rec["predictions"]:
        p=HERE/x["path"];assert digest(p)==x["sha256"]
        if x["phase"]!="prelim":continue
        k=x["team"];tpath=td/f"aia_truth_{k}.csv";truth=core.read(tpath).sort_values("challenge_row_id").reset_index(drop=True);truthhash[str(tpath)]=digest(tpath)
        data=np.load(p);baseline=core.scores(np.where(data["old_full"],9.5,0),truth)
        diff=max(abs(baseline[m]-float(saved.loc[k,m])) for m in METRICS);checks.append(dict(team=k,maximum_saved_metric_difference=diff));assert diff<1e-12
        for ci,context in enumerate(CONTEXTS):
            for mi,method in enumerate(METHODS):
                rows.append(dict(team=k,context=context,method=method,source="real_C",repeat=-1,**core.scores(np.where(data["real"][ci,mi],9.5,0),truth)))
                for repeat in range(CFG["null_evaluation_repeats"]):
                    rows.append(dict(team=k,context=context,method=method,source="BG_null_C",repeat=repeat,**core.scores(np.where(data["null"][ci,repeat,mi],9.5,0),truth)))
    table=pd.DataFrame(rows);out=HERE/"集計"
    table.to_csv(out/"予備戦_対象別とnull反復.csv",index=False)
    aggregates=[]
    for name,excluded in [("prelim20",[5]),("processed18",[1,2,5]),("all21_diagnostic",[])]:
        g=table[~table.team.isin(excluded)]
        agg=g.groupby(["context","method","source"])[METRICS].mean().reset_index();agg.insert(0,"group",name);aggregates.append(agg)
    pd.concat(aggregates).to_csv(out/"予備戦_平均.csv",index=False)
    actual=table[table.source.eq("real_C")];null=table[table.source.eq("BG_null_C")]
    means=null.groupby(["team","context","method"])[METRICS].mean().reset_index()
    comp=actual.merge(means,on=["team","context","method"],suffixes=("_real","_null"));comp["excess_R_vs_null_mean"]=comp.r_all_real-comp.r_all_null
    comp.to_csv(out/"実Cと背景Cの差.csv",index=False)
    loo=[]
    for context in CONTEXTS:
        piv=actual[(actual.context==context)&~actual.team.isin([1,2,5])].pivot(index="team",columns="method",values="r_all").reindex(columns=METHODS)
        for held in piv.index:
            train=piv.drop(index=held).mean();selected=METHODS[int(np.argmax(train.to_numpy()))]
            loo.append(dict(context=context,held_team=held,selected_method=selected,held_R=float(piv.loc[held,selected]),held_baseline_R=float(piv.loc[held,"baseline"]),train_R=float(train[selected])))
    pd.DataFrame(loo).to_csv(out/"leave_team_out方式選択.csv",index=False)
    pd.DataFrame(checks).to_csv(out/"保存済み基準採点の一致.csv",index=False)
    writej(HERE/"採点実行記録.json",dict(prediction_receipt_sha256=digest(HERE/"予測実行記録.json"),public_prelim_truth_sha256=truthhash,
        code_sha256=digest(__file__),config_sha256=digest(HERE/"固定条件.json"),main_truth_used=False,
        is_official_phi=False,selection_candidates_fixed_before_scoring=True,
        candidate_design_independent_of_past_prelim_results=False,submission_created=False))
    print(pd.concat(aggregates).query("group == 'processed18'").to_string(index=False),flush=True)
    print(pd.DataFrame(loo).groupby("context")[["held_R","held_baseline_R"]].mean().to_string(),flush=True)
    return table
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["predict","score","all"],default="all");args=parser.parse_args()
    HERE.mkdir(parents=True,exist_ok=True)
    if args.stage in ["predict","all"]:predict()
    if args.stage in ["score","all"]:score()
