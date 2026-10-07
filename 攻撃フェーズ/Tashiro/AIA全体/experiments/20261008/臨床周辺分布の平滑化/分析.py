"""固定帯域の臨床KDEと最近1点の比較。結果閲覧後の2候補、39pool外。"""
from pathlib import Path
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
import sys,json,hashlib,time,argparse,importlib.util,math,platform
import numpy as np
import pandas as pd
from scipy.special import logsumexp
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/"属性値集合による選別"
spec=importlib.util.spec_from_file_location("frozen_nearest_column_methods",SOURCE/"分析.py")
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
CFG=json.loads((HERE/"固定条件.json").read_text());METHODS=CFG["methods"];NEWMETHODS=METHODS[-2:]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def log_density(q,values,h):
    values=np.sort(np.asarray(values,float));q=np.asarray(q,float)
    log_kernel=-.5*((q[:,None]-values[None,:])/h)**2
    return logsumexp(log_kernel,axis=1)-np.log(len(values))-np.log(h*np.sqrt(2*np.pi))
def reference_setup(q,c,ref):
    setups={};db=np.empty((len(q),4));qs=q.sex.to_numpy();cs=c.sex.to_numpy()
    for sex in np.unique(qs):
        a=np.flatnonzero(qs==sex);b=cs==sex;rr=ref[ref.sex==sex]
        n=int(b.sum());assert n>0 and len(rr)>0
        raw=rr[exp.COLS].to_numpy();center=np.median(raw,axis=0)
        scale=np.maximum(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0),CFG["IQR_floor"])
        h=n**(-1/5);x=(q.iloc[a][exp.COLS].to_numpy()-center)/scale;rb=(raw-center)/scale
        for j in range(4):db[a,j]=log_density(x[:,j],rb[:,j],h)
        setups[int(sex)]=dict(a=a,center=center,scale=scale,h=h,nC=n,q=x,reference_n=len(rr),IQR_floor_applied=int((scale==CFG["IQR_floor"]).sum()))
    return setups,db
def kde_evidence(q,c,setups,db):
    lc=np.empty_like(db)
    for sex,setup in setups.items():
        cc=c[c.sex==sex];assert len(cc)==setup["nC"]
        raw=(cc[exp.COLS].to_numpy()-setup["center"])/setup["scale"]
        for j in range(4):lc[setup["a"],j]=log_density(setup["q"][:,j],raw[:,j],setup["h"])
    ratio=lc-db;assert np.isfinite(ratio).all()
    return dict(ratio=ratio,all4=ratio.mean(axis=1),top2=np.sort(ratio,axis=1)[:,-2:].mean(axis=1),log_C=lc,log_BG=db)
def choose(q,ev):return np.stack([exp.independent_select(q,ev,m) for m in ["column_all4_mean","column_top2_mean"]])
def independent_small_fixture():
    q=np.array([-2.,0.,1.5,7.]);c=np.array([-1.,0.,2.]);bg=np.array([-3.,-.5,1.,4.,6.])
    h=len(c)**(-1/5)
    a=log_density(q,c,h)-log_density(q,bg,h);b=[]
    for x in q:
        def density(values):
            return sum(math.exp(-.5*((float(x)-float(y))/h)**2)/(h*math.sqrt(2*math.pi)) for y in values)/len(values)
        b.append(math.log(density(c))-math.log(density(bg)))
    err=float(np.max(np.abs(a-b)));assert err<1e-12
    tail=log_density(np.array([1e6]),c,h);assert np.isfinite(tail).all()
    return dict(q=q.tolist(),C=c.tolist(),BG=bg.tolist(),h=h,logsumexp_ratio=a.tolist(),direct_Gaussian_sum_ratio=b,
      maximum_difference=err,extreme_tail_log_density_is_finite=True,same_bandwidth_both_densities=True)
def predict():
    start=time.time();work=HERE/"作業用";out=HERE/"集計";work.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    exp.writej(HERE/"小型Gaussian和の独立照合.json",independent_small_fixture())
    old=json.loads((SOURCE/"予測実行記録.json").read_text())
    assert old["code_sha256"]==digest(SOURCE/"分析.py") and old["config_sha256"]==digest(SOURCE/"固定条件.json")
    for p,h in old["input_sha256"].items():assert digest(p)==h
    entries=[];perms=[];affine=[];diagnostics=[];parameters=[];repro=[]
    for phase,folder,bgp in exp.SETS:
        bg=exp.core.read(bgp);order=np.random.default_rng(CFG["split_seed"]+(phase=="main")).permutation(len(bg));midpoint=len(bg)//2
        pool=bg.iloc[order[:midpoint]].reset_index(drop=True);ref=bg.iloc[order[midpoint:]].reset_index(drop=True)
        assert set(order[:midpoint]).isdisjoint(set(order[midpoint:]))
        for item in [x for x in old["predictions"] if x["phase"]==phase]:
            k=item["team"];q=exp.core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True);c=exp.core.read(folder/f"C_{k}.csv")
            sourcecache=SOURCE/item["path"];assert digest(sourcecache)==item["sha256"];prior=np.load(sourcecache)
            nullix=exp.sample_indices(pool,c,CFG["null_repeats"],CFG["null_seed"]+1000*k+(phase=="main")*100000)
            actual=[];null=[]
            for ci,context in enumerate(CFG["contexts"]):
                reference=ref if ci==0 else bg;setup,db=reference_setup(q,c,reference);ev=kde_evidence(q,c,setup,db);selected=choose(q,ev)
                actual.append(np.concatenate([prior["real"][ci,:3],selected],axis=0))
                for sex,v in setup.items():parameters.append(dict(phase=phase,team=k,context=context,sex=sex,nC=v["nC"],nBG=v["reference_n"],h=v["h"],
                    IQR_floor_applied=v["IQR_floor_applied"],same_h_for_C_BG=True))
                # 最近1点2対照は、同じC/refから再計算して凍結予測と一致。
                nearest=exp.evidence(q,c,reference);same=choose(q,nearest)
                assert np.array_equal(same,prior["real"][ci,[1,2]])
                repro.append(dict(phase=phase,team=k,context=context,k1_control_mask_changed=0))
                qs=q.sex.to_numpy()
                for mi,m in enumerate(NEWMETHODS):
                    for sex in [0,1]:
                        a=qs==sex;assert int(selected[mi,a].sum())==int(np.ceil(int(a.sum())*.75))
                        diagnostics.append(dict(phase=phase,team=k,context=context,method=m,sex=sex,Q_rows=int(a.sum()),selected=int(selected[mi,a].sum()),
                            changed_vs_k1=int((selected[mi,a]!=same[mi,a]).sum()),log_ratio_mean=float(ev["ratio"][a].mean()),
                            log_density_ratio_min=float(ev["ratio"][a].min()),log_density_ratio_max=float(ev["ratio"][a].max())))
                for repeat in range(exp.CFG["permutation_repeats"]):
                    rng=np.random.default_rng(exp.CFG["permutation_seed"]+100*k+repeat+(phase=="main")*100000)
                    pi=rng.permutation(len(q));pj=rng.permutation(len(c));pr=rng.permutation(len(reference))
                    qp=q.iloc[pi].reset_index(drop=True);cp=c.iloc[pj].reset_index(drop=True);rp=reference.iloc[pr].reset_index(drop=True)
                    sp,dp=reference_setup(qp,cp,rp);ep=kde_evidence(qp,cp,sp,dp)
                    permuted=choose(qp,ep)[:,np.argsort(pi)]
                    for mi,m in enumerate(NEWMETHODS):
                        diff=int((permuted[mi]!=selected[mi]).sum());assert diff==0
                        perms.append(dict(phase=phase,team=k,context=context,method=m,repeat=repeat,changed=diff))
                # 列ごとの正の尺度変換と平行移動。IQR下限に触れない場合の不変性。
                factor=np.array([.1,2.,10.,.5]);shift=np.array([100.,-30.,50.,-10.])
                qa=q.copy();ca=c.copy();ra=reference.copy()
                for frame in [qa,ca,ra]:frame[exp.COLS]=frame[exp.COLS].to_numpy()*factor+shift
                sa,da=reference_setup(qa,ca,ra);ea=kde_evidence(qa,ca,sa,da)
                error=float(np.max(np.abs(ea["ratio"]-ev["ratio"])))
                relative_error=float(np.max(np.abs(ea["ratio"]-ev["ratio"])/np.maximum(1.,np.abs(ev["ratio"]))))
                assert np.allclose(ea["ratio"],ev["ratio"],rtol=1e-11,atol=1e-10)
                mask=choose(qa,ea);changed=int((mask!=selected).sum());assert changed==0
                affine.append(dict(phase=phase,team=k,context=context,max_log_ratio_difference=error,max_scaled_difference=relative_error,selection_changed=changed,
                    IQR_floor_inactive=all(v["IQR_floor_applied"]==0 for v in setup.values())))
                if phase=="prelim":
                    nsets=[]
                    for repeat,ix in enumerate(nullix):
                        nc=pool.iloc[ix].reset_index(drop=True);ne=kde_evidence(q,nc,setup,db);ns=choose(q,ne)
                        nsets.append(np.concatenate([prior["null"][ci,repeat,:3],ns],axis=0))
                    null.append(np.stack(nsets))
            saved=work/f"{phase}_{k}.npz";np.savez_compressed(saved,real=np.stack(actual),null=np.stack(null) if null else np.empty(0))
            entries.append(dict(phase=phase,team=k,path=str(saved.relative_to(HERE)),sha256=digest(saved)))
            print(json.dumps(dict(stage="all_predictions_before_truth",phase=phase,team=k,seconds=round(time.time()-start,1))),flush=True)
    pd.DataFrame(perms).to_csv(out/"行順変更.csv",index=False)
    pd.DataFrame(affine).to_csv(out/"原尺度の変換不変.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"公開課題の選別とlog密度比.csv",index=False)
    pd.DataFrame(parameters).to_csv(out/"固定帯域と参照人数.csv",index=False)
    pd.DataFrame(repro).to_csv(out/"凍結最近1点対照の再現.csv",index=False)
    exp.writej(HERE/"予測実行記録.json",dict(start_time=time.strftime("%Y-%m-%dT%H:%M:%S%z",time.localtime(start)),seconds=time.time()-start,
        code_sha256=digest(__file__),config_sha256=digest(HERE/"固定条件.json"),source_code_sha256=digest(SOURCE/"分析.py"),
        source_protocol_sha256=digest(SOURCE/"固定条件.json"),source_prediction_receipt_sha256=digest(SOURCE/"予測実行記録.json"),
        input_sha256=dict(exp.core.HASHES),predictions=entries,all44_predictions_completed=len(entries)==44,unfinished_C=[],
        all_predictions_before_truth_read=True,main_truth_used=False,B_self_used=False,
        post_result_design=True,excluded_from_39_pool=True,new_methods=2,hyperparameter_search=False,
        permutation_tests=len(perms),permutation_changed_max=0,python=sys.executable,python_version=platform.python_version()))
def score():
    rec=json.loads((HERE/"予測実行記録.json").read_text());assert rec["code_sha256"]==digest(__file__) and rec["config_sha256"]==digest(HERE/"固定条件.json")
    for item in rec["predictions"]:assert digest(HERE/item["path"])==item["sha256"]
    td=next((exp.ROOT/"予備戦/配布データ/結果").glob("正解*"));rows=[];truthhash={}
    for item in rec["predictions"]:
        if item["phase"]!="prelim":continue
        k=item["team"];tp=td/f"aia_truth_{k}.csv";truth=exp.core.read(tp).sort_values("challenge_row_id").reset_index(drop=True);truthhash[str(tp)]=digest(tp)
        data=np.load(HERE/item["path"])
        for ci,context in enumerate(CFG["contexts"]):
            for mi,m in enumerate(METHODS):
                r=exp.core.scores(np.where(data["real"][ci,mi],9.5,0),truth);r["signed_hit_difference"]=r["hit_m"]-r["hit_c"]
                rows.append(dict(team=k,context=context,method=m,source="real_C",repeat=-1,**r))
                for repeat in range(CFG["null_repeats"]):
                    r=exp.core.scores(np.where(data["null"][ci,repeat,mi],9.5,0),truth);r["signed_hit_difference"]=r["hit_m"]-r["hit_c"]
                    rows.append(dict(team=k,context=context,method=m,source="BG_null_C",repeat=repeat,**r))
    table=pd.DataFrame(rows);out=HERE/"集計";table.to_csv(out/"予備戦_固定5候補と20null対象別.csv",index=False)
    metrics=exp.METRICS+["signed_hit_difference"];aggregates=[]
    for group,excluded in [("prelim20",[5]),("processed18",[1,2,5]),("all21_diagnostic",[])]:
        a=table[~table.team.isin(excluded)].groupby(["context","method","source"])[metrics].mean().reset_index();a.insert(0,"group",group);aggregates.append(a)
    mean=pd.concat(aggregates);mean.to_csv(out/"予備戦_平均.csv",index=False)
    actual=table[table.source=="real_C"];null=table[table.source=="BG_null_C"].groupby(["team","context","method"])[metrics].mean().reset_index()
    paired=actual.merge(null,on=["team","context","method"],suffixes=("_real","_null"));paired["r_excess"]=paired.r_all_real-paired.r_all_null
    paired["signed_hit_excess"]=paired.signed_hit_difference_real-paired.signed_hit_difference_null
    paired.to_csv(out/"実Cとnull超過_対象別.csv",index=False)
    loo=[]
    for context in CFG["contexts"]:
        for group,excluded in [("prelim20",[5]),("processed18",[1,2,5])]:
            piv=actual[(actual.context==context)&~actual.team.isin(excluded)].pivot(index="team",columns="method",values="r_all").reindex(columns=METHODS)
            for held in piv.index:
                train=piv.drop(index=held).mean();m=METHODS[int(np.argmax(train.to_numpy()))]
                loo.append(dict(context=context,group=group,held_team=held,selected_method=m,held_R=float(piv.loc[held,m]),
                    baseline_R=float(piv.loc[held,METHODS[0]]),train_R=float(train[m]),post_result_candidate_family=True))
    pd.DataFrame(loo).to_csv(out/"固定5候補_leave_team_out.csv",index=False)
    exp.writej(HERE/"採点実行記録.json",dict(prediction_receipt_sha256=digest(HERE/"予測実行記録.json"),code_sha256=digest(__file__),
        public_prelim_truth_sha256=truthhash,all44_predictions_completed_before_truth_read=True,main_truth_used=False,B_self_used=False,
        post_result_design=True,excluded_from_39_pool=True,bandwidth_or_coefficients_selected_from_scores=False))
    print(mean.query("group == 'processed18'").to_string(index=False),flush=True)
    print(pd.DataFrame(loo).groupby(["context","group"])[["held_R","baseline_R"]].mean().to_string(),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["predict","score"],required=True);args=p.parse_args()
    predict() if args.stage=="predict" else score()
