"""有望なk1列値集合の結果閲覧後の固定追加探索。39候補pool外。

k=3,8、all4/top2の4案だけ。旧Hungarianとk1を含む固定7方式。
全44Cと予備20の20反復nullを保存後、公開予備正解だけを採点する。
本戦正解、B_self、提出、CodaBench、GitHubは扱わない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse,hashlib,importlib.util,io,json,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
FIRST=HERE.parent/"属性値集合による選別"
OLD=HERE.parents[1]/"20261007_AIA全体向上の分析"
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location("original_column_family",FIRST/"分析.py")
column=importlib.util.module_from_spec(spec)
spec.loader.exec_module(column)
core,stable=column.core,column.stable
CFG=json.loads((HERE/"固定条件.json").read_text())
KS=CFG["neighbour_k"]
METHODS=CFG["methods"]
COLS=CFG["clinical_columns"]
CONTEXTS=CFG["contexts"]
EPS=CFG["ratio_epsilon"]
OUT=HERE/"集計"
WORK=HERE/"作業用"
HASHES={}
METRICS=["r_all","r_rare","r_max","p_m","p_c","hit_m","hit_c","hit_m_rare","hit_c_rare","signed_hit_difference"]
PHASES=[("prelim",ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack",ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
        ("main",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    raw=Path(path).read_bytes()
    HASHES[str(path)]=hashlib.sha256(raw).hexdigest()
    return pd.read_csv(io.BytesIO(raw),float_precision="round_trip")


def writej(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def imports():
    files=[FIRST/"分析.py",FIRST/"固定条件.json"]+[OLD/name for name in ["分析.py","同点に依存しない対応.py","対応競合の頑健性.py","対応競合の分析.py"]]
    return {str(path):sha(path) for path in files}


def kth1d(q,values,ks=KS):
    """k左+k右の窓に最近k行は必ず含まれる。範囲外はinf、重複は行として数える。"""
    q=np.asarray(q,float)
    values=np.sort(np.asarray(values,float))
    assert np.isfinite(q).all() and np.isfinite(values).all() and len(values)>=max(ks)
    location=np.searchsorted(values,q,side="left")
    radius=max(ks)
    indices=location[:,None]+np.arange(-radius,radius)[None,:]
    valid=(indices>=0)&(indices<len(values))
    distance=np.where(valid,np.abs(q[:,None]-values[np.clip(indices,0,len(values)-1)]),np.inf)
    ordered=np.sort(distance,axis=1)
    answer=ordered[:,np.array(ks)-1]
    assert np.isfinite(answer).all()
    return answer


def nearest_columns(q,c):
    out=np.empty((len(KS),len(q),len(COLS)))
    qs,cs=q.sex.to_numpy(),c.sex.to_numpy()
    for sex in np.unique(qs):
        a,b=qs==sex,cs==sex
        assert b.sum()>=max(KS)
        for j,name in enumerate(COLS):
            out[:,a,j]=kth1d(q.loc[a,name].to_numpy(),c.loc[b,name].to_numpy()).T
    return out


def select_score(q,score):
    selected=np.zeros(len(q),bool)
    ids=q.challenge_row_id.to_numpy()
    sex=q.sex.to_numpy()
    assert len(np.unique(ids))==len(q) and np.isfinite(score).all()
    for value in np.unique(sex):
        indices=np.flatnonzero(sex==value)
        count=int(np.ceil(len(indices)*.75))
        order=np.lexsort((ids[indices],-score[indices]))
        selected[indices[order[:count]]]=True
    return selected


def predictions(q,c,ref,scale,dc=None,db=None):
    baseline,pair,partial=column.baseline_select(q,c,ref,scale)
    dc=nearest_columns(q,c) if dc is None else dc
    db=nearest_columns(q,ref) if db is None else db
    ratio=np.log((db+EPS)/(dc+EPS))
    scores=[]
    for i in range(len(KS)):
        scores.extend([ratio[i].mean(axis=1),np.sort(ratio[i],axis=1)[:,-2:].mean(axis=1)])
    selected=np.stack([baseline]+[select_score(q,score) for score in scores])
    assert np.isfinite(ratio).all()
    return selected,dict(dc=dc,db=db,ratio=ratio,scores=scores),pair,partial


def check_distances(q,c,phase,team,reference,checks):
    fast=nearest_columns(q,c)
    brute=np.empty_like(fast)
    for sex in np.unique(q.sex):
        a=q.sex.to_numpy()==sex
        b=c.sex.to_numpy()==sex
        for j,name in enumerate(COLS):
            matrix=np.abs(q.loc[a,name].to_numpy()[:,None]-c.loc[b,name].to_numpy()[None,:])
            ordered=np.sort(matrix,axis=1)
            brute[:,a,j]=ordered[:,np.array(KS)-1].T
    assert np.array_equal(fast,brute)
    for i,k in enumerate(KS):
        checks.append(dict(phase=phase,team=team,reference=reference,k=k,compared_distances=int(fast[i].size),
                           maximum_distance_difference=float(np.max(np.abs(fast[i]-brute[i])))))
    return fast


def fixtures():
    cases=[
      ([-100.,-3.,-1.,0.,.5,1.,3.,100.],[-4.,-3.,-2.,-1.,1.,2.,3.,4.]),
      ([0.,1.,1e-8,np.nextafter(1e-8,np.inf)],[-1.,0.,0.,0.,0.,0.,0.,0.,1.,1.]),
      ([10.,20.,30.],[20.]*8),
      ([0.,5.,20.],list(np.arange(8,dtype=float)))
    ]
    result=[]
    for q,c in cases:
        fast=kth1d(q,c)
        brute=np.sort(np.abs(np.asarray(q)[:,None]-np.asarray(c)[None,:]),axis=1)[:,np.array(KS)-1]
        assert np.array_equal(fast,brute)
        result.append(dict(q=q,reference=c,k=KS,kth_distances=fast.tolist(),maximum_difference=0.))
    return dict(implementation_a="sorted insertion window",implementation_b="all-pair absolute distances then full sort",duplicate_values_count_as_rows=True,cases=result)


def diagnostics(q,c,ref,ev,phase,team,context,selected,old_full,rows,attributes):
    qs,cs,bs=q.sex.to_numpy(),c.sex.to_numpy(),ref.sex.to_numpy()
    for sex in np.unique(qs):
        a=qs==sex
        for ki,k in enumerate(KS):
            for j,name in enumerate(COLS):
                dc,db=ev["dc"][ki,a,j],ev["db"][ki,a,j]
                attributes.append(dict(phase=phase,team=team,context=context,sex=int(sex),k=k,attribute=name,
                    q_n=int(a.sum()),C_n=int((cs==sex).sum()),BG_n=int((bs==sex).sum()),
                    C_unique_values=int(c.loc[cs==sex,name].nunique()),BG_unique_values=int(ref.loc[bs==sex,name].nunique()),
                    C_distance_median=float(np.median(dc)),BG_distance_median=float(np.median(db)),
                    log_ratio_median=float(np.median(ev["ratio"][ki,a,j])),
                    C_distance_le_1e8=int((dc<=EPS).sum()),BG_distance_le_1e8=int((db<=EPS).sum())))
        for mi,method in enumerate(METHODS):
            base=selected[1 if mi in [1,3,5] else 2,a] if mi else old_full[a]
            value=selected[mi,a]
            union=(base|value).sum()
            score=ev["scores"][mi-1][a] if mi else None
            cutoff=np.sort(score)[-int(value.sum())] if mi else np.nan
            rows.append(dict(phase=phase,team=team,context=context,method=method,sex=int(sex),q_n=int(a.sum()),
                selected=int(value.sum()),changed_vs_full_baseline=int((value!=old_full[a]).sum()),
                changed_vs_corresponding_k1=int((value!=base).sum()),
                jaccard_vs_corresponding_k1=float((value&base).sum()/union),
                score_cutoff=float(cutoff) if mi else np.nan,
                tied_at_cutoff=int((score==cutoff).sum()) if mi else 0))


def predict():
    started=time.time()
    OUT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(exist_ok=True)
    writej(HERE/"小型_距離別実装照合.json",fixtures())
    entries,checks,repro,permutations,rows,attributes,sampling=[],[],[],[],[],[],[]
    for phase,folder,bgpath in PHASES:
        bg=read(bgpath)
        assert len(bg)==5143
        raw=core.raw(bg)
        scale=np.maximum(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0),1e-5)
        order=np.random.default_rng(CFG["split_seed"]+(phase=="main")).permutation(len(bg))
        middle=len(bg)//2
        pool=bg.iloc[order[:middle]].reset_index(drop=True)
        ref=bg.iloc[order[middle:]].reset_index(drop=True)
        assert len(pool)==2571 and len(ref)==2572 and set(order[:middle]).isdisjoint(order[middle:])
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem[2:])):
            team=int(cp.stem[2:])
            c=read(cp)
            q=read(folder/f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            assert len(q)==326 and len(c)==1049
            dc=check_distances(q,c,phase,team,"actual_C",checks)
            old_pair,old_dc=stable.matrices(q,c,bg)
            old_full=stable.select(old_pair,old_dc,q.sex.to_numpy(),c.sex.to_numpy(),"sex_quota")
            null_indices=column.sample_indices(pool,c,CFG["null_repeats"],CFG["null_seed"]+1000*team+(phase=="main")*100000) if phase=="prelim" and team!=5 else []
            actual,null=[],[]
            for context,reference in zip(CONTEXTS,[ref,bg]):
                db=check_distances(q,reference,phase,team,context,checks)
                selected,ev,pair,partial=predictions(q,c,reference,scale,dc,db)
                actual.append(selected)
                original_ev=column.evidence(q,c,reference)
                assert np.array_equal(ev["dc"][0],original_ev["dc"])
                assert np.array_equal(ev["db"][0],original_ev["db"])
                for mi,method in enumerate(METHODS[1:3],1):
                    original=column.independent_select(q,original_ev,method)
                    assert np.array_equal(selected[mi],original)
                    repro.append(dict(phase=phase,team=team,context=context,method=method,changed=0))
                if context=="full_reference_overlap":
                    assert np.array_equal(selected[0],old_full)
                    repro.append(dict(phase=phase,team=team,context=context,method=METHODS[0],changed=0))
                diagnostics(q,c,reference,ev,phase,team,context,selected,old_full,rows,attributes)
                for repeat in range(CFG["permutation_repeats"]):
                    rng=np.random.default_rng(CFG["permutation_seed"]+100*team+repeat+(phase=="main")*100000)
                    pi,pj=rng.permutation(len(q)),rng.permutation(len(c))
                    qp,cp2=q.iloc[pi].reset_index(drop=True),c.iloc[pj].reset_index(drop=True)
                    cperm=nearest_columns(qp,cp2)
                    bperm=db[:,pi]
                    ratio=np.log((bperm+EPS)/(cperm+EPS))
                    scores=[]
                    for ki in range(len(KS)):
                        scores.extend([ratio[ki].mean(1),np.sort(ratio[ki],axis=1)[:,-2:].mean(1)])
                    baseperm=stable.select(pair[pi][:,pj],partial[pi][:,pj],qp.sex.to_numpy(),cp2.sex.to_numpy(),"sex_quota")
                    permselected=np.stack([baseperm]+[select_score(qp,score) for score in scores])[:,np.argsort(pi)]
                    for mi,method in enumerate(METHODS):
                        change=int((selected[mi]!=permselected[mi]).sum())
                        assert change==0
                        permutations.append(dict(phase=phase,team=team,context=context,method=method,repeat=repeat,changed=change))
                if null_indices:
                    nullsets=[]
                    for repeat,indices in enumerate(null_indices):
                        nc=pool.iloc[indices].reset_index(drop=True)
                        assert np.array_equal(np.bincount(nc.sex.to_numpy().astype(int)),np.bincount(c.sex.to_numpy().astype(int)))
                        assert len(indices)==1049 and len(np.unique(indices))==1049
                        s,_,_,_=predictions(q,nc,reference,scale)
                        nullsets.append(s)
                        original_bg_indices=order[:middle][indices]
                        sampling.append(dict(team=team,context=context,repeat=repeat,n=len(indices),same_sex_counts=True,
                            pool_ref_disjoint=True,original_BG_index_sha256=hashlib.sha256(original_bg_indices.tobytes()).hexdigest()))
                    null.append(np.stack(nullsets))
            path=WORK/f"{phase}_{team}.npz"
            np.savez_compressed(path,real=np.stack(actual),null=np.stack(null) if null else np.empty((0,)),
                                old_full=old_full,challenge_row_ids=q.challenge_row_id.to_numpy(),sex=q.sex.to_numpy())
            entries.append(dict(phase=phase,team=team,path=str(path.relative_to(HERE)),sha256=sha(path)))
            print(json.dumps(dict(stage="predict_before_truth",phase=phase,team=team,seconds=round(time.time()-started,1))),flush=True)
    assert len(entries)==44 and len(sampling)==20*20*2
    pd.DataFrame(checks).to_csv(OUT/"全44CとBG参照_最近k距離の別実装照合.csv",index=False)
    pd.DataFrame(repro).to_csv(OUT/"既存k1と旧全BG基準の予測一致.csv",index=False)
    pd.DataFrame(permutations).to_csv(OUT/"行順変更.csv",index=False)
    pd.DataFrame(rows).to_csv(OUT/"選択変更と同点の対象別性別.csv",index=False)
    pd.DataFrame(attributes).to_csv(OUT/"第k近傍距離と重複の対象別属性別.csv",index=False)
    pd.DataFrame(sampling).to_csv(OUT/"初回条件の同性別null抽出監査.csv",index=False)
    record=dict(code_sha256=sha(__file__),config_sha256=sha(HERE/"固定条件.json"),imports_sha256=imports(),
        input_sha256=HASHES,predictions=entries,actual_C_count=44,contexts=CONTEXTS,methods=METHODS,
        real_prediction_cases=44*2*7,null_prediction_cases=20*20*2*7,
        nearest_check_cases=len(checks),maximum_nearest_distance_difference=0.,
        permutation_cases=len(permutations),maximum_permutation_changes=0,
        primary_pool_reference_disjoint=True,all_predictions_saved_before_truth_read=True,
        main_truth_used=False,B_self_used=False,in_first39_frozen_pool=False,posthoc_inspiration=True,
        started_JST=time.strftime("%Y-%m-%dT%H:%M:%S%z",time.localtime(started)),seconds=time.time()-started,
        python=sys.executable,python_version=platform.python_version())
    writej(HERE/"予測実行記録.json",record)


def scores(selection,truth):
    result=core.scores(np.where(selection,9.5,0.),truth)
    result["signed_hit_difference"]=result["hit_m"]-result["hit_c"]
    return result


def frozen_receipt():
    record=json.loads((HERE/"予測実行記録.json").read_text())
    assert record["code_sha256"]==sha(__file__) and record["config_sha256"]==sha(HERE/"固定条件.json")
    assert all(sha(path)==expected for path,expected in record["input_sha256"].items())
    assert all(sha(path)==expected for path,expected in record["imports_sha256"].items())
    assert len(record["predictions"])==44
    assert all(sha(HERE/item["path"])==item["sha256"] for item in record["predictions"])
    return record


def score():
    record=frozen_receipt()
    td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    truth_hashes={}
    rows=[]
    for item in record["predictions"]:
        if item["phase"]!="prelim":
            continue
        team=item["team"]
        path=td/f"aia_truth_{team}.csv"
        truth=read(path).sort_values("challenge_row_id").reset_index(drop=True)
        truth_hashes[str(path)]=sha(path)
        prediction=np.load(HERE/item["path"],allow_pickle=False)
        assert np.array_equal(prediction["challenge_row_ids"],truth.challenge_row_id.to_numpy())
        for ci,context in enumerate(CONTEXTS):
            for mi,method in enumerate(METHODS):
                rows.append(dict(team=team,context=context,method=method,source="real_C",repeat=-1,**scores(prediction["real"][ci,mi],truth)))
                if team!=5:
                    for repeat in range(CFG["null_repeats"]):
                        rows.append(dict(team=team,context=context,method=method,source="BG_null_C",repeat=repeat,**scores(prediction["null"][ci,repeat,mi],truth)))
    table=pd.DataFrame(rows)
    table.to_csv(OUT/"予備戦_対象別と20反復null.csv",index=False)
    original=read(FIRST/"集計/予備戦_対象別と20反復null.csv")
    checks=[]
    for row in table[table.method.isin(METHODS[:3])].itertuples(index=False):
        saved=original[(original.team==row.team)&(original.context==row.context)&(original.method==row.method)&(original.source==row.source)&(original.repeat==row.repeat)]
        assert len(saved)==1
        error=max(abs(getattr(row,key)-saved.iloc[0][key]) for key in METRICS if key!="signed_hit_difference")
        assert error<1e-12
        checks.append(dict(team=row.team,context=row.context,method=row.method,source=row.source,repeat=row.repeat,maximum_score_difference=error))
    pd.DataFrame(checks).to_csv(OUT/"初回3方式との全実Cとnull採点一致.csv",index=False)
    old=read(OLD/"集計/同点修正_対象別.csv")
    old=old[old.method.eq("stable_matching/sex_quota/long75")].set_index("team")
    saved=table[(table.context=="full_reference_overlap")&(table.method==METHODS[0])&(table.source=="real_C")].set_index("team")
    for key in METRICS:
        if key!="signed_hit_difference":
            assert np.max(np.abs(saved[key]-old.loc[saved.index,key]))<1e-12
    null=table[table.source=="BG_null_C"].groupby(["team","context","method"])[METRICS].mean().reset_index()
    real=table[table.source=="real_C"]
    paired=real[real.team!=5].merge(null,on=["team","context","method"],suffixes=("_real","_null"))
    paired["r_excess"]=paired.r_all_real-paired.r_all_null
    paired["signed_hit_excess"]=paired.signed_hit_difference_real-paired.signed_hit_difference_null
    paired.to_csv(OUT/"実Cと同性別背景nullの差.csv",index=False)
    average,loto,comparisons=[],[],[]
    for scope,excluded in [("prelim20",[5]),("processed18",[1,2,5])]:
        use=table[~table.team.isin(excluded)]
        for context in CONTEXTS:
            for method in METHODS:
                one=use[(use.context==context)&(use.method==method)]
                re=one[one.source=="real_C"]
                nu=one[one.source=="BG_null_C"]
                average.append(dict(scope=scope,context=context,method=method,targets=len(re),null_repeats=20,
                    real_r_all=float(re.r_all.mean()),null_r_all=float(nu.r_all.mean()),r_excess=float(re.r_all.mean()-nu.r_all.mean()),
                    real_signed_hit_difference=float(re.signed_hit_difference.mean()),null_signed_hit_difference=float(nu.signed_hit_difference.mean()),
                    signed_hit_excess=float(re.signed_hit_difference.mean()-nu.signed_hit_difference.mean()),
                    real_r_rare=float(re.r_rare.mean()),real_r_max=float(re.r_max.mean())))
            matrix=real[(real.context==context)&~real.team.isin(excluded)].pivot(index="team",columns="method",values="r_all").reindex(columns=METHODS)
            for team in matrix.index:
                train=matrix.drop(index=team).mean().to_numpy()
                chosen=int(np.flatnonzero(train>=train.max()-1e-12)[0])
                loto.append(dict(scope=scope,context=context,held_team=int(team),selected_method=METHODS[chosen],
                    train_R=float(train[chosen]),held_R=float(matrix.loc[team,METHODS[chosen]]),baseline_R=float(matrix.loc[team,METHODS[0]]),
                    k1_all4_R=float(matrix.loc[team,METHODS[1]]),k1_top2_R=float(matrix.loc[team,METHODS[2]]),
                    unseen_independent_validation=False,in_first39_frozen_pool=False))
            bymethod=paired[(paired.context==context)&~paired.team.isin(excluded)].set_index(["method","team"])
            for method in METHODS[3:]:
                anchor=METHODS[1] if "all4" in method else METHODS[2]
                for metric in ["r_all_real","r_excess","signed_hit_difference_real","signed_hit_excess"]:
                    delta=bymethod.loc[method,metric]-bymethod.loc[anchor,metric]
                    comparisons.append(dict(scope=scope,context=context,method=method,k1_anchor=anchor,metric=metric,
                        mean_difference=float(delta.mean()),improved_targets=int((delta>1e-12).sum()),
                        worsened_targets=int((delta<-1e-12).sum()),equal_targets=int((np.abs(delta)<=1e-12).sum())))
    pd.DataFrame(average).to_csv(OUT/"20対象と18対象の平均.csv",index=False)
    lt=pd.DataFrame(loto)
    lt.to_csv(OUT/"固定7候補_LOTO対象別.csv",index=False)
    lt.groupby(["scope","context"])[["held_R","baseline_R","k1_all4_R","k1_top2_R"]].mean().to_csv(OUT/"固定7候補_LOTO平均.csv")
    pd.DataFrame(comparisons).to_csv(OUT/"同じ集約のk1との差.csv",index=False)
    writej(HERE/"採点実行記録.json",dict(code_sha256=sha(__file__),config_sha256=sha(HERE/"固定条件.json"),
        prediction_receipt_sha256=sha(HERE/"予測実行記録.json"),public_prelim_truth_sha256=truth_hashes,
        initial_aggregate_sha256=sha(FIRST/"集計/予備戦_対象別と20反復null.csv"),old_baseline_aggregate_sha256=sha(OLD/"集計/同点修正_対象別.csv"),
        all44_predictions_saved_before_truth_read=True,main_truth_used=False,B_self_used=False,
        scored_real_cases=21*2*7,scored_null_cases=20*20*2*7,matched_initial_score_cases=len(checks),
        old_full21_scores_match=True,in_first39_frozen_pool=False,posthoc_inspiration=True,
        submission_created=False,GitHub_published=False,LOTO_rows=len(lt),independent_validation=False))
    print(pd.DataFrame(average).query("scope=='processed18'").to_string(index=False),flush=True)
    print(lt.groupby(["scope","context"])[["held_R","baseline_R","k1_all4_R","k1_top2_R"]].mean().to_string(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["predict","score"],required=True)
    stage=parser.parse_args().stage
    {"predict":predict,"score":score}[stage]()
