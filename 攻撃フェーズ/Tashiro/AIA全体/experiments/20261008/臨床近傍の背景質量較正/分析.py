"""最近1点の背景質量の固定2候補。結果閲覧後の最後の追加探索、39pool外。
全44Cと予備20の20nullを保存後、公開予備正解だけを採点する。
人口仮説の属性別近傍出現確率であり、真会員確率や4列結合p値ではない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse,hashlib,importlib.util,io,itertools,json,math,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
FIRST=HERE.parent/"属性値集合による選別"
SOURCE=HERE.parent/"臨床値集合の複数近傍"
OLD=HERE.parents[1]/"20261007_AIA全体向上の分析"
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location("original_column_family_mass",FIRST/"分析.py")
column=importlib.util.module_from_spec(spec)
spec.loader.exec_module(column)
core,stable=column.core,column.stable
CFG=json.loads((HERE/"固定条件.json").read_text())
COLS=CFG["clinical_columns"]
METHODS=CFG["methods"]
CONTEXTS=CFG["contexts"]
OUT=HERE/"集計"
WORK=HERE/"作業用"
HASHES={}
METRICS=["r_all","r_rare","r_max","p_m","p_c","hit_m","hit_c","hit_m_rare","hit_c_rare","signed_hit_difference"]
PHASES=[("prelim",ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack",ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
        ("main",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    content=Path(path).read_bytes()
    HASHES[str(path)]=hashlib.sha256(content).hexdigest()
    return pd.read_csv(io.BytesIO(content),float_precision="round_trip")


def writej(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def import_hashes():
    paths=[FIRST/"分析.py",FIRST/"固定条件.json"]+[OLD/name for name in ["分析.py","同点に依存しない対応.py","対応競合の頑健性.py","対応競合の分析.py"]]
    return {str(path):sha(path) for path in paths}


def near_probability_table(N,n):
    """厳密な有限人口式の浮動小数評価。Kごとの無出現確率を安定に累積する。"""
    assert isinstance(N,(int,np.integer)) and isinstance(n,(int,np.integer)) and 0<=n<=N
    if n==0:
        return np.zeros(N+1)
    probability=np.ones(N+1)
    probability[0]=0.
    last=N-n
    if last:
        j=np.arange(last,dtype=float)
        log_survival=np.cumsum(np.log1p(-n/(N-j)))
        probability[1:last+1]=-np.expm1(log_survival)
    assert np.isfinite(probability).all() and (np.diff(probability)>=-1e-15).all()
    return np.clip(probability,0,1)


def small_checks():
    max_comb_error=0.
    cases=0
    enum_cases=0
    max_enum_error=0.
    for N in range(1,15):
        for n in range(N+1):
            table=near_probability_table(N,n)
            for K in range(N+1):
                exact=1-(math.comb(N-K,n)/math.comb(N,n) if N-K>=n else 0.)
                max_comb_error=max(max_comb_error,abs(table[K]-exact))
                cases+=1
                if N<=8:
                    subsets=list(itertools.combinations(range(N),n))
                    hits=sum(any(value<K for value in subset) for subset in subsets)
                    enumerated=hits/len(subsets)
                    max_enum_error=max(max_enum_error,abs(table[K]-enumerated))
                    enum_cases+=1
    assert max_comb_error<1e-12 and max_enum_error<1e-12
    # 差から得た距離を直接使い、境界を含む。逆算のq+-dは使わない。
    q=np.array([0.,.1,1.,2.])
    bg=np.array([-1.,0.,.1,.2,1.,2.])
    radii=np.array([0.,.1,1.,np.nextafter(1.,0.)])
    distances=np.abs(q[:,None]-bg[None,:])
    sorted_distance=np.sort(distances,axis=1)
    fast=np.array([np.searchsorted(row,radius,side="right") for row,radius in zip(sorted_distance,radii)])
    brute=(distances<=radii[:,None]).sum(1)
    assert np.array_equal(fast,brute)
    return dict(math_comb_cases=cases,full_subset_enumeration_cases=enum_cases,maximum_comb_error=max_comb_error,
                maximum_enumeration_error=max_enum_error,inclusive_distance_boundary_check_counts=fast.tolist(),
                probability_is_analytic_not_Monte_Carlo=True)


def setup_background(q,c,ref):
    qs,cs,bs=q.sex.to_numpy(),c.sex.to_numpy(),ref.sex.to_numpy()
    background_rows=[[None]*len(q) for _ in COLS]
    tables={}
    populations={}
    for sex in np.unique(qs):
        a,b=qs==sex,bs==sex
        N,n=int(b.sum()),int((cs==sex).sum())
        assert 0<n<=N
        tables[int(sex)]=near_probability_table(N,n)
        populations[int(sex)]=(N,n)
        indices=np.flatnonzero(a)
        for j,name in enumerate(COLS):
            ordered=np.sort(np.abs(q.loc[a,name].to_numpy()[:,None]-ref.loc[b,name].to_numpy()[None,:]),axis=1)
            for i,row in zip(indices,ordered):
                background_rows[j][i]=row
    return dict(rows=background_rows,tables=tables,populations=populations)


def evidence(q,c,setup,original_indices=None):
    dc=column.nearest_columns(q,c)
    qs=q.sex.to_numpy()
    original_indices=np.arange(len(q)) if original_indices is None else original_indices
    K=np.empty((len(q),4),int)
    probability=np.empty_like(dc)
    for i,original_i in enumerate(original_indices):
        table=setup["tables"][int(qs[i])]
        for j in range(4):
            count=int(np.searchsorted(setup["rows"][j][original_i],dc[i,j],side="right"))
            K[i,j]=count
            probability[i,j]=table[count]
    e=-np.log((1+CFG["smoothing"]*probability)/(1+CFG["smoothing"]))
    assert np.isfinite(e).all() and (e>=-1e-15).all() and (e<=np.log(65)+1e-15).all()
    return dict(dc=dc,K=K,probability=probability,e=e,all4=e.mean(1),top2=np.sort(e,axis=1)[:,-2:].mean(1))


def choose(q,ev):
    return np.stack([column.independent_select(q,ev,name) for name in ["column_all4_mean","column_top2_mean"]])


def independent_count_check(q,c,ref,ev,phase,team,context):
    K=np.empty_like(ev["K"])
    distance=np.empty_like(ev["dc"])
    for sex in np.unique(q.sex):
        a=q.sex.to_numpy()==sex
        cb,bb=c.sex.to_numpy()==sex,ref.sex.to_numpy()==sex
        for j,name in enumerate(COLS):
            d=np.abs(q.loc[a,name].to_numpy()[:,None]-c.loc[cb,name].to_numpy()[None,:]).min(1)
            distance[a,j]=d
            bg_distance=np.abs(q.loc[a,name].to_numpy()[:,None]-ref.loc[bb,name].to_numpy()[None,:])
            K[a,j]=(bg_distance<=d[:,None]).sum(1)
    assert np.array_equal(distance,ev["dc"]) and np.array_equal(K,ev["K"])
    return dict(phase=phase,team=team,context=context,compared_q_attribute_combinations=int(K.size),
                maximum_C_distance_difference=0.,maximum_BG_count_difference=0)


def source_receipt():
    record=json.loads((SOURCE/"予測実行記録.json").read_text())
    assert record["code_sha256"]==sha(SOURCE/"分析.py") and record["config_sha256"]==sha(SOURCE/"固定条件.json")
    assert all(sha(path)==h for path,h in record["input_sha256"].items())
    assert all(sha(path)==h for path,h in record["imports_sha256"].items())
    assert len(record["predictions"])==44
    return record


def predict():
    started=time.time()
    OUT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(exist_ok=True)
    writej(HERE/"小型_有限人口確率の別実装照合.json",small_checks())
    source=source_receipt()
    source_by={(item["phase"],item["team"]):item for item in source["predictions"]}
    entries,counts,perms,diagnostic,sampling,repro=[],[],[],[],[],[]
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
            q=read(folder/f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c=read(cp)
            item=source_by[(phase,team)]
            assert sha(SOURCE/item["path"])==item["sha256"]
            prior=np.load(SOURCE/item["path"],allow_pickle=False)
            assert np.array_equal(prior["challenge_row_ids"],q.challenge_row_id.to_numpy())
            null_indices=column.sample_indices(pool,c,20,CFG["null_seed"]+1000*team) if phase=="prelim" and team!=5 else []
            actual,null=[],[]
            for ci,(context,reference) in enumerate(zip(CONTEXTS,[ref,bg])):
                setup=setup_background(q,c,reference)
                ev=evidence(q,c,setup)
                selected=choose(q,ev)
                actual.append(np.concatenate([prior["real"][ci,:3],selected]))
                counts.append(independent_count_check(q,c,reference,ev,phase,team,context))
                k1=column.evidence(q,c,reference)
                k1selected=np.stack([column.independent_select(q,k1,m) for m in METHODS[1:3]])
                base,pair,partial=column.baseline_select(q,c,reference,scale)
                assert np.array_equal(k1selected,prior["real"][ci,1:3])
                assert np.array_equal(base,prior["real"][ci,0])
                repro.append(dict(phase=phase,team=team,context=context,changed_existing3_predictions=0))
                for sex in np.unique(q.sex):
                    a=q.sex.to_numpy()==sex
                    N,n=setup["populations"][int(sex)]
                    for j,name in enumerate(COLS):
                        diagnostic.append(dict(phase=phase,team=team,context=context,sex=int(sex),attribute=name,q_n=int(a.sum()),N=N,n=n,
                            K_min=int(ev["K"][a,j].min()),K_median=float(np.median(ev["K"][a,j])),K_max=int(ev["K"][a,j].max()),
                            K_zero_count=int((ev["K"][a,j]==0).sum()),p_mean=float(ev["probability"][a,j].mean()),
                            e_mean=float(ev["e"][a,j].mean()),e_max=float(ev["e"][a,j].max())))
                for repeat in range(CFG["permutation_repeats"]):
                    rng=np.random.default_rng(CFG["permutation_seed"]+100*team+repeat+(phase=="main")*100000)
                    pi,pj=rng.permutation(len(q)),rng.permutation(len(c))
                    qp,cp2=q.iloc[pi].reset_index(drop=True),c.iloc[pj].reset_index(drop=True)
                    eperm=evidence(qp,cp2,setup,pi)
                    massperm=choose(qp,eperm)
                    originalperm=column.evidence(qp,cp2,reference)
                    k1perm=np.stack([column.independent_select(qp,originalperm,m) for m in METHODS[1:3]])
                    baseperm=stable.select(pair[pi][:,pj],partial[pi][:,pj],qp.sex.to_numpy(),cp2.sex.to_numpy(),"sex_quota")
                    allperm=np.concatenate([baseperm[None,:],k1perm,massperm])[:,np.argsort(pi)]
                    for mi,method in enumerate(METHODS):
                        changed=int((actual[-1][mi]!=allperm[mi]).sum())
                        assert changed==0
                        perms.append(dict(phase=phase,team=team,context=context,method=method,repeat=repeat,changed=changed))
                if null_indices:
                    nullsets=[]
                    for repeat,indices in enumerate(null_indices):
                        nc=pool.iloc[indices].reset_index(drop=True)
                        assert np.array_equal(np.bincount(c.sex.to_numpy().astype(int)),np.bincount(nc.sex.to_numpy().astype(int)))
                        ns=choose(q,evidence(q,nc,setup))
                        nullsets.append(np.concatenate([prior["null"][ci,repeat,:3],ns]))
                        sampling.append(dict(team=team,context=context,repeat=repeat,n=len(indices),same_sex_counts=True,
                            split_pool_ref_disjoint=True,null_sample_contained_in_current_reference=context=="full_reference_overlap",
                            original_BG_index_sha256=hashlib.sha256(order[:middle][indices].tobytes()).hexdigest()))
                    null.append(np.stack(nullsets))
            path=WORK/f"{phase}_{team}.npz"
            np.savez_compressed(path,real=np.stack(actual),null=np.stack(null) if null else np.empty((0,)),
                challenge_row_ids=q.challenge_row_id.to_numpy(),sex=q.sex.to_numpy())
            entries.append(dict(phase=phase,team=team,path=str(path.relative_to(HERE)),sha256=sha(path)))
            print(json.dumps(dict(stage="predict_before_truth",phase=phase,team=team,seconds=round(time.time()-started,1))),flush=True)
    pd.DataFrame(counts).to_csv(OUT/"全44C_最小距離と背景人数の別実装照合.csv",index=False)
    pd.DataFrame(perms).to_csv(OUT/"行順変更.csv",index=False)
    pd.DataFrame(diagnostic).to_csv(OUT/"属性別背景人数と近傍質量.csv",index=False)
    pd.DataFrame(sampling).to_csv(OUT/"初回条件の同性別null抽出監査.csv",index=False)
    pd.DataFrame(repro).to_csv(OUT/"既存3方式の予測再現.csv",index=False)
    assert len(entries)==44 and len(counts)==88 and len(sampling)==800
    writej(HERE/"予測実行記録.json",dict(code_sha256=sha(__file__),config_sha256=sha(HERE/"固定条件.json"),
        input_sha256=HASHES,imports_sha256=import_hashes(),source_prediction_receipt_sha256=sha(SOURCE/"予測実行記録.json"),
        source_prediction_hashes={item["path"]:item["sha256"] for item in source["predictions"]},predictions=entries,
        actual_C_count=44,real_prediction_cases=44*2*5,null_prediction_cases=20*20*2*5,
        count_check_cases=len(counts),maximum_count_difference=0,permutation_cases=len(perms),maximum_permutation_changes=0,
        all_predictions_saved_before_truth_read=True,main_truth_used=False,B_self_used=False,
        in_first39_frozen_pool=False,posthoc_inspiration=True,last_method_addition=True,
        new_candidate_count=2,hyperparameter_search=False,seconds=time.time()-started,
        started_JST=time.strftime("%Y-%m-%dT%H:%M:%S%z",time.localtime(started)),
        python=sys.executable,python_version=platform.python_version()))


def scores(selected,truth):
    result=core.scores(np.where(selected,9.5,0.),truth)
    result["signed_hit_difference"]=result["hit_m"]-result["hit_c"]
    return result


def score():
    record=json.loads((HERE/"予測実行記録.json").read_text())
    assert record["code_sha256"]==sha(__file__) and record["config_sha256"]==sha(HERE/"固定条件.json")
    assert all(sha(path)==h for path,h in record["input_sha256"].items())
    assert all(sha(path)==h for path,h in record["imports_sha256"].items())
    assert len(record["predictions"])==44 and all(sha(HERE/item["path"])==item["sha256"] for item in record["predictions"])
    td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    rows=[]
    truth_hashes={}
    for item in record["predictions"]:
        if item["phase"]!="prelim":
            continue
        team=item["team"]
        path=td/f"aia_truth_{team}.csv"
        truth=read(path).sort_values("challenge_row_id").reset_index(drop=True)
        truth_hashes[str(path)]=sha(path)
        data=np.load(HERE/item["path"],allow_pickle=False)
        assert np.array_equal(truth.challenge_row_id,data["challenge_row_ids"])
        for ci,context in enumerate(CONTEXTS):
            for mi,method in enumerate(METHODS):
                rows.append(dict(team=team,context=context,method=method,source="real_C",repeat=-1,**scores(data["real"][ci,mi],truth)))
                if team!=5:
                    for repeat in range(20):
                        rows.append(dict(team=team,context=context,method=method,source="BG_null_C",repeat=repeat,**scores(data["null"][ci,repeat,mi],truth)))
    table=pd.DataFrame(rows)
    table.to_csv(OUT/"予備戦_固定5候補と20null対象別.csv",index=False)
    initial=read(FIRST/"集計/予備戦_対象別と20反復null.csv")
    checks=[]
    for row in table[table.method.isin(METHODS[:3])].itertuples(index=False):
        before=initial[(initial.team==row.team)&(initial.context==row.context)&(initial.method==row.method)&(initial.source==row.source)&(initial.repeat==row.repeat)]
        assert len(before)==1
        error=max(abs(getattr(row,key)-before.iloc[0][key]) for key in METRICS if key!="signed_hit_difference")
        assert error<1e-12
        checks.append(dict(team=row.team,context=row.context,method=row.method,source=row.source,repeat=row.repeat,maximum_score_difference=error))
    pd.DataFrame(checks).to_csv(OUT/"初回3方式との実Cとnull採点一致.csv",index=False)
    null=table[table.source.eq("BG_null_C")].groupby(["team","context","method"])[METRICS].mean().reset_index()
    real=table[table.source.eq("real_C")]
    paired=real[real.team!=5].merge(null,on=["team","context","method"],suffixes=("_real","_null"))
    paired["r_excess"]=paired.r_all_real-paired.r_all_null
    paired["signed_hit_excess"]=paired.signed_hit_difference_real-paired.signed_hit_difference_null
    paired.to_csv(OUT/"実Cと同性別背景nullの差.csv",index=False)
    means,loto,comparisons=[],[],[]
    for scope,excluded in [("prelim20",[5]),("processed18",[1,2,5])]:
        for context in CONTEXTS:
            for method in METHODS:
                use=table[~table.team.isin(excluded)&table.context.eq(context)&table.method.eq(method)]
                re,nu=use[use.source.eq("real_C")],use[use.source.eq("BG_null_C")]
                means.append(dict(scope=scope,context=context,method=method,targets=len(re),null_repeats=20,
                    real_r_all=float(re.r_all.mean()),null_r_all=float(nu.r_all.mean()),r_excess=float(re.r_all.mean()-nu.r_all.mean()),
                    real_signed_hit_difference=float(re.signed_hit_difference.mean()),null_signed_hit_difference=float(nu.signed_hit_difference.mean()),
                    signed_hit_excess=float(re.signed_hit_difference.mean()-nu.signed_hit_difference.mean()),
                    real_r_rare=float(re.r_rare.mean()),real_r_max=float(re.r_max.mean())))
            matrix=real[~real.team.isin(excluded)&real.context.eq(context)].pivot(index="team",columns="method",values="r_all").reindex(columns=METHODS)
            for held in matrix.index:
                train=matrix.drop(index=held).mean().to_numpy()
                chosen=int(np.flatnonzero(train>=train.max()-1e-12)[0])
                loto.append(dict(scope=scope,context=context,held_team=int(held),selected_method=METHODS[chosen],train_R=float(train[chosen]),
                    held_R=float(matrix.loc[held,METHODS[chosen]]),baseline_R=float(matrix.loc[held,METHODS[0]]),
                    k1_all4_R=float(matrix.loc[held,METHODS[1]]),k1_top2_R=float(matrix.loc[held,METHODS[2]]),
                    posthoc_inspiration=True,in_first39_frozen_pool=False,independent_validation=False))
            local=paired[~paired.team.isin(excluded)&paired.context.eq(context)].set_index(["method","team"])
            for method,anchor in zip(METHODS[3:],METHODS[1:3]):
                for metric in ["r_all_real","r_excess","signed_hit_difference_real","signed_hit_excess"]:
                    delta=local.loc[method,metric]-local.loc[anchor,metric]
                    comparisons.append(dict(scope=scope,context=context,method=method,k1_anchor=anchor,metric=metric,mean_difference=float(delta.mean()),
                        improved_targets=int((delta>1e-12).sum()),worsened_targets=int((delta<-1e-12).sum()),equal_targets=int((np.abs(delta)<=1e-12).sum())))
    pd.DataFrame(means).to_csv(OUT/"20対象と18対象の平均.csv",index=False)
    lt=pd.DataFrame(loto)
    lt.to_csv(OUT/"固定5候補_LOTO対象別.csv",index=False)
    lt.groupby(["scope","context"])[["held_R","baseline_R","k1_all4_R","k1_top2_R"]].mean().to_csv(OUT/"固定5候補_LOTO平均.csv")
    pd.DataFrame(comparisons).to_csv(OUT/"対応するk1との差.csv",index=False)
    writej(HERE/"採点実行記録.json",dict(code_sha256=sha(__file__),config_sha256=sha(HERE/"固定条件.json"),
        prediction_receipt_sha256=sha(HERE/"予測実行記録.json"),public_prelim_truth_sha256=truth_hashes,
        initial_aggregate_sha256=sha(FIRST/"集計/予備戦_対象別と20反復null.csv"),
        all44_predictions_saved_before_truth_read=True,scored_real_cases=21*2*5,scored_null_cases=20*20*2*5,
        matched_initial_score_cases=len(checks),main_truth_used=False,B_self_used=False,in_first39_frozen_pool=False,
        posthoc_inspiration=True,last_method_addition=True,hyperparameter_search=False,LOTO_rows=len(lt),independent_validation=False))
    print(pd.DataFrame(means).query("scope=='processed18'").to_string(index=False),flush=True)
    print(lt.groupby(["scope","context"])[["held_R","baseline_R","k1_all4_R","k1_top2_R"]].mean().to_string(),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["predict","score"],required=True)
    stage=parser.parse_args().stage
    {"predict":predict,"score":score}[stage]()
