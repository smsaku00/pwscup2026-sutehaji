"""各臨床列の値集合を使う固定3候補。公開44Cの予測完了後に予備真値のみ採点。
B_self、本戦真値、token、提出F/ZIP、CodaBench、GitHub操作は扱わない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import sys,json,hashlib,argparse,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
ROOT=Path("/Users/taisei1105/pws2026-local")
HERE=Path(__file__).resolve().parent
OLD=ROOT/"本戦/攻撃フェーズ/Tashiro/20261007_AIA全体向上の分析"
sys.path.insert(0,str(OLD))
import 同点に依存しない対応 as stable
import 分析 as core
from 対応競合の頑健性 import distance,pair_score
CFG=json.loads((HERE/"固定条件.json").read_text())
METHODS=CFG["methods"];COLS=CFG["clinical_columns"];CONTEXTS=CFG["contexts"]
EPS=CFG["ratio_epsilon"];TOL=CFG["raw_match_tolerance"]
METRICS=["r_all","r_rare","r_max","p_m","p_c","hit_m","hit_c"]
SETS=[
("prelim",ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack",ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
("main",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def nearest1d(q,values):
    values=np.sort(np.asarray(values,dtype=float));q=np.asarray(q,dtype=float)
    assert len(values)>0 and np.isfinite(values).all() and np.isfinite(q).all()
    ix=np.searchsorted(values,q)
    return np.minimum(np.abs(q-values[np.clip(ix,0,len(values)-1)]),np.abs(q-values[np.clip(ix-1,0,len(values)-1)]))
def nearest_columns(q,c):
    out=np.empty((len(q),len(COLS)))
    for sex in np.unique(q.sex):
        a=q.sex.to_numpy()==sex;b=c.sex.to_numpy()==sex
        assert b.any()
        for j,name in enumerate(COLS):out[a,j]=nearest1d(q.loc[a,name],c.loc[b,name])
    return out
def evidence(q,c,ref):
    dc=nearest_columns(q,c);db=nearest_columns(q,ref)
    ratio=np.log((db+EPS)/(dc+EPS))
    all4=ratio.mean(axis=1);top2=np.sort(ratio,axis=1)[:,-2:].mean(axis=1)
    exact=(dc<=TOL).sum(axis=1)
    assert np.isfinite(ratio).all()
    return dict(dc=dc,db=db,ratio=ratio,all4=all4,top2=top2,exact=exact)
def independent_select(q,ev,method):
    result=np.zeros(len(q),dtype=bool);ids=q.challenge_row_id.to_numpy()
    assert len(np.unique(ids))==len(q)
    for sex in np.unique(q.sex):
        a=np.flatnonzero(q.sex.to_numpy()==sex);n=int(np.ceil(len(a)*.75))
        if method=="column_all4_mean":order=np.lexsort((ids[a],-ev["all4"][a]))
        elif method=="column_top2_mean":order=np.lexsort((ids[a],-ev["top2"][a]))
        elif method=="column_exact_count_then_all4":order=np.lexsort((ids[a],-ev["all4"][a],-ev["exact"][a]))
        else:raise ValueError(method)
        result[a[order[:n]]]=True
    return result
def baseline_pair(q,c,ref,scale):
    dc=distance(q,c,scale)
    raw=pair_score(dc,distance(q,ref,scale))["continuous"]
    full=cdist(core.raw(q)/scale,core.raw(c)/scale,"sqeuclidean")/5
    return raw-stable.EPSILON*np.minimum(full,1000),dc
def baseline_select(q,c,ref,scale):
    pair,dc=baseline_pair(q,c,ref,scale)
    return stable.select(pair,dc,q.sex.to_numpy(),c.sex.to_numpy(),"sex_quota"),pair,dc
def masks(q,c,ref,scale):
    baseline,pair,dc=baseline_select(q,c,ref,scale)
    ev=evidence(q,c,ref)
    return np.stack([baseline]+[independent_select(q,ev,m) for m in METHODS[1:]]),ev,pair,dc
def sample_indices(pool,c,repeats,seed):
    rng=np.random.default_rng(seed);out=[]
    for repeat in range(repeats):
        ix=[]
        for sex in np.unique(c.sex):
            eligible=np.flatnonzero(pool.sex.to_numpy()==sex)
            n=int((c.sex==sex).sum());assert n<=len(eligible)
            ix.extend(rng.choice(eligible,n,replace=False).tolist())
        out.append(np.sort(np.array(ix,dtype=int)))
    return out
def independent_nearest_check(q,c,phase,k,rows):
    sorted_result=nearest_columns(q,c)
    brute=np.empty_like(sorted_result)
    for sex in np.unique(q.sex):
        a=q.sex.to_numpy()==sex;b=c.sex.to_numpy()==sex
        for j,col in enumerate(COLS):
            brute[a,j]=np.abs(q.loc[a,col].to_numpy()[:,None]-c.loc[b,col].to_numpy()[None,:]).min(axis=1)
    difference=float(np.max(np.abs(sorted_result-brute)))
    changed=int(np.count_nonzero((sorted_result<=TOL)!=(brute<=TOL)))
    rows.append(dict(phase=phase,team=k,maximum_distance_difference=difference,tolerance_decision_changed=changed))
    assert difference==0 and changed==0
def fixtures():
    cases=[
      ([-3.,-2.,-1.,0.,.5,1.,2.,3.],[-1.,-1.,1.]),
      ([0.,1e-8,np.nextafter(1e-8,np.inf),-1e-8,np.nextafter(-1e-8,-np.inf)],[0.]),
      ([90.,110.,130.],[100.,120.]),
      ([30.,40.,50.],[40.,40.,40.])]
    answers=[]
    for q,c in cases:
        a=nearest1d(q,c);b=np.abs(np.asarray(q)[:,None]-np.asarray(c)[None,:]).min(axis=1)
        assert np.array_equal(a,b)
        answers.append(dict(q=list(q),c=list(c),distances=a.tolist(),within_1e8=(a<=TOL).tolist()))
    assert answers[1]["within_1e8"]==[True,True,False,True,False]
    return dict(implementation_a="sorted searchsorted nearest two neighbours",implementation_b="all-pair absolute differences and min",cases=answers,max_difference=0,tolerance_inclusive=True)
def diagnostics(q,c,ref,ev,phase,k,rows):
    for sex in np.unique(q.sex):
        a=q.sex.to_numpy()==sex;b=c.sex.to_numpy()==sex
        for j,col in enumerate(COLS):
            cv=c.loc[b,col].to_numpy();sv=np.sort(cv)
            near=np.full(len(sv),np.inf)
            if len(sv)>1:
                step=np.diff(sv);near[:-1]=np.minimum(near[:-1],step);near[1:]=np.minimum(near[1:],step)
            rows.append(dict(phase=phase,team=k,sex=int(sex),attribute=col,C_rows=len(cv),C_unique_exact=len(np.unique(cv)),
              C_exact_reused_rows=len(cv)-len(np.unique(cv)),C_rows_with_another_value_within_tolerance=int((near<=TOL).sum()),
              Q_rows=int(a.sum()),Q_match_C_count=int((ev["dc"][a,j]<=TOL).sum()),Q_match_BG_count=int((ev["db"][a,j]<=TOL).sum()),
              Q_nearest_C_median=float(np.median(ev["dc"][a,j])),Q_nearest_BG_median=float(np.median(ev["db"][a,j])),
              log_distance_ratio_median=float(np.median(ev["ratio"][a,j]))))
def cross_C_table(datasets):
    rows=[]
    for i,(ki,ci) in enumerate(datasets):
        for kj,cj in datasets[i+1:]:
            for sex in [0,1]:
                for name in COLS:
                    a=np.unique(ci.loc[ci.sex==sex,name].to_numpy());b=np.unique(cj.loc[cj.sex==sex,name].to_numpy())
                    if len(a)==0 or len(b)==0:continue
                    da=nearest1d(a,b);db=nearest1d(b,a)
                    rows.append(dict(C_a=ki,C_b=kj,sex=sex,attribute=name,unique_a=len(a),unique_b=len(b),
                        shared_unique_a_within_tolerance=int((da<=TOL).sum()),shared_unique_b_within_tolerance=int((db<=TOL).sum()),
                        shared_fraction_a=float((da<=TOL).mean()),shared_fraction_b=float((db<=TOL).mean())))
    return rows
def predict():
    start=time.time();work=HERE/"作業用";out=HERE/"集計";work.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    writej(HERE/"1D別実装との照合.json",fixtures())
    entries=[];perms=[];repro=[];checks=[];attribute=[];counts=[];selectdiag=[];mainsets=[]
    for phase,folder,bgp in SETS:
        bg=core.read(bgp);raw=core.raw(bg);scale=np.maximum(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0),1e-5)
        order=np.random.default_rng(CFG["split_seed"]+(phase=="main")).permutation(len(bg))
        midpoint=len(bg)//2;pool=bg.iloc[order[:midpoint]].reset_index(drop=True);ref=bg.iloc[order[midpoint:]].reset_index(drop=True)
        assert set(order[:midpoint]).isdisjoint(set(order[midpoint:]))
        references=[ref,bg]
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem[2:])):
            k=int(cp.stem[2:]);c=core.read(cp);q=core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            assert len(c)==1049 and len(q)==326
            if phase=="main":mainsets.append((k,c))
            independent_nearest_check(q,c,phase,k,checks)
            oldpair,olddc=stable.matrices(q,c,bg);old=stable.select(oldpair,olddc,q.sex.to_numpy(),c.sex.to_numpy(),"sex_quota")
            actual=[];null=[]
            nullix=sample_indices(pool,c,CFG["null_repeats"],CFG["null_seed"]+1000*k+(phase=="main")*100000)
            for ri,context in enumerate(CONTEXTS):
                selected,ev,pair,dc=masks(q,c,references[ri],scale);actual.append(selected)
                if context=="full_reference_overlap":
                    changed=int((selected[0]!=old).sum());assert changed==0
                    repro.append(dict(phase=phase,team=k,changed_baseline=changed))
                diagnostics(q,c,references[ri],ev,phase,k,attribute)
                for n in range(5):
                    counts.append(dict(phase=phase,team=k,context=context,matching_clinical_columns=n,Q_rows=int((ev["exact"]==n).sum())))
                for mi,method in enumerate(METHODS):
                    for sex in [0,1]:
                        a=q.sex.to_numpy()==sex
                        selectdiag.append(dict(phase=phase,team=k,context=context,method=method,sex=sex,Q_rows=int(a.sum()),
                           selected=int(selected[mi,a].sum()),changed_vs_full_baseline=int((selected[mi,a]!=old[a]).sum())))
                for repeat in range(CFG["permutation_repeats"]):
                    rng=np.random.default_rng(CFG["permutation_seed"]+100*k+repeat+(phase=="main")*100000)
                    pi=rng.permutation(len(q));pj=rng.permutation(len(c));qp=q.iloc[pi].reset_index(drop=True);cp2=c.iloc[pj].reset_index(drop=True)
                    e2=evidence(qp,cp2,references[ri])
                    s2=np.stack([stable.select(pair[pi][:,pj],dc[pi][:,pj],qp.sex.to_numpy(),cp2.sex.to_numpy(),"sex_quota")]+[independent_select(qp,e2,m) for m in METHODS[1:]])[:,np.argsort(pi)]
                    for mi,method in enumerate(METHODS):
                        change=int((selected[mi]!=s2[mi]).sum());assert change==0
                        perms.append(dict(phase=phase,team=k,context=context,method=method,repeat=repeat,changed=change))
                if phase=="prelim":
                    nullsets=[]
                    for repeat,ix in enumerate(nullix):
                        nc=pool.iloc[ix].reset_index(drop=True)
                        s,ne,_,_=masks(q,nc,references[ri],scale);nullsets.append(s)
                    null.append(np.stack(nullsets))
            cache=work/f"{phase}_{k}.npz"
            np.savez_compressed(cache,real=np.stack(actual),null=np.stack(null) if null else np.empty((0,)),old_full=old)
            entries.append(dict(phase=phase,team=k,path=str(cache.relative_to(HERE)),sha256=digest(cache)))
            print(json.dumps(dict(stage="predict",phase=phase,team=k,seconds=round(time.time()-start,1))),flush=True)
    # 診断は集計のみ。本戦正解は使わず、周辺値の重なりを測る。
    at=pd.DataFrame(attribute)
    # diagnosticsではcontextを後付けでなく正確な行順で対応。各context8行固定。
    assert len(attribute)==len(entries)*len(CONTEXTS)*2*len(COLS)
    at["context"]=[ctx for x in entries for ctx in CONTEXTS for _ in range(2*len(COLS))]
    at.to_csv(out/"各臨床列の値再利用と課題一致.csv",index=False)
    pd.DataFrame(cross_C_table(mainsets)).to_csv(out/"本戦C間の周辺値共通.csv",index=False)
    pd.DataFrame(counts).to_csv(out/"課題の一致属性数分布.csv",index=False)
    pd.DataFrame(selectdiag).to_csv(out/"性別内選別と基準変更数.csv",index=False)
    pd.DataFrame(checks).to_csv(out/"全44C_1D別実装照合.csv",index=False)
    pd.DataFrame(perms).to_csv(out/"行順変更.csv",index=False)
    pd.DataFrame(repro).to_csv(out/"全44C基準再現.csv",index=False)
    receipt=dict(start_time=time.strftime("%Y-%m-%dT%H:%M:%S%z",time.localtime(start)),seconds=time.time()-start,
       code_sha256=digest(__file__),config_sha256=digest(HERE/"固定条件.json"),
       imports={p.name:digest(p) for p in [OLD/"同点に依存しない対応.py",OLD/"対応競合の頑健性.py",OLD/"対応競合の分析.py",OLD/"分析.py"]},
       input_sha256=dict(core.HASHES),predictions=entries,actual_C_count=len(entries),null_repeats=CFG["null_repeats"],
       truth_used_for_predictions=False,main_truth_used=False,B_self_used=False,submission_created=False,
       primary_pool_reference_disjoint=True,permutation_tests=len(perms),permutation_max=0,
       independent_1D_checks=len(checks),independent_1D_max_difference=0,
       python=sys.executable,python_version=platform.python_version())
    assert len(entries)==44 and len(repro)==44
    writej(HERE/"予測実行記録.json",receipt)
def score():
    rec=json.loads((HERE/"予測実行記録.json").read_text())
    assert rec["code_sha256"]==digest(__file__) and rec["config_sha256"]==digest(HERE/"固定条件.json")
    # 全44予測のハッシュ確認を先に行い、その後で公開予備真値に進む。
    for item in rec["predictions"]:assert digest(HERE/item["path"])==item["sha256"]
    td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"));truthhash={};rows=[];checks=[]
    saved=pd.read_csv(OLD/"集計/同点修正_対象別.csv")
    saved=saved[saved.method=="stable_matching/sex_quota/long75"].set_index("team")
    for item in rec["predictions"]:
        if item["phase"]!="prelim":continue
        k=item["team"];p=td/f"aia_truth_{k}.csv";truth=core.read(p).sort_values("challenge_row_id").reset_index(drop=True);truthhash[str(p)]=digest(p)
        data=np.load(HERE/item["path"]);baseline=core.scores(np.where(data["old_full"],9.5,0),truth)
        difference=max(abs(baseline[m]-float(saved.loc[k,m])) for m in METRICS);assert difference<1e-12
        checks.append(dict(team=k,maximum_saved_score_difference=difference))
        for ci,context in enumerate(CONTEXTS):
            for mi,method in enumerate(METHODS):
                rows.append(dict(team=k,context=context,method=method,source="real_C",repeat=-1,**core.scores(np.where(data["real"][ci,mi],9.5,0),truth)))
                for repeat in range(CFG["null_repeats"]):
                    rows.append(dict(team=k,context=context,method=method,source="BG_null_C",repeat=repeat,**core.scores(np.where(data["null"][ci,repeat,mi],9.5,0),truth)))
    table=pd.DataFrame(rows);out=HERE/"集計";table.to_csv(out/"予備戦_対象別と20反復null.csv",index=False)
    aggregates=[]
    for name,excluded in [("prelim20",[5]),("processed18",[1,2,5]),("all21_diagnostic",[])]:
        g=table[~table.team.isin(excluded)]
        a=g.groupby(["context","method","source"])[METRICS].mean().reset_index();a.insert(0,"group",name);aggregates.append(a)
    mean=pd.concat(aggregates);mean.to_csv(out/"予備戦_平均.csv",index=False)
    actual=table[table.source=="real_C"];null=table[table.source=="BG_null_C"].groupby(["team","context","method"])[METRICS].mean().reset_index()
    comparison=actual.merge(null,on=["team","context","method"],suffixes=("_real","_null"));comparison["excess_R_vs_null_mean"]=comparison.r_all_real-comparison.r_all_null
    comparison.to_csv(out/"実Cと背景nullの差.csv",index=False)
    loo=[]
    for context in CONTEXTS:
        piv=actual[(actual.context==context)&~actual.team.isin([1,2,5])].pivot(index="team",columns="method",values="r_all").reindex(columns=METHODS)
        for held in piv.index:
            train=piv.drop(index=held).mean();m=METHODS[int(np.argmax(train.to_numpy()))]
            loo.append(dict(context=context,held_team=held,selected_method=m,held_R=float(piv.loc[held,m]),baseline_R=float(piv.loc[held,METHODS[0]]),train_R=float(train[m])))
    pd.DataFrame(loo).to_csv(out/"leave_team_out方式選択.csv",index=False)
    pd.DataFrame(checks).to_csv(out/"保存済み採点との一致.csv",index=False)
    writej(HERE/"採点実行記録.json",dict(prediction_receipt_sha256=digest(HERE/"予測実行記録.json"),code_sha256=digest(__file__),
      config_sha256=digest(HERE/"固定条件.json"),public_prelim_truth_sha256=truthhash,
      all44_predictions_completed_before_truth_read=True,main_truth_used=False,B_self_used=False,submission_created=False,
      R_all_is_not_official_phi_corrected_max=True,independent_unseen_validation=False))
    print(mean.query("group == 'processed18'").to_string(index=False),flush=True)
    print(pd.DataFrame(loo).groupby("context")[["held_R","baseline_R"]].mean().to_string(),flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["predict","score"],required=True);args=parser.parse_args()
    if args.stage=="predict":predict()
    else:score()
