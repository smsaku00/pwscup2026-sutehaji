"""固定値集合2方式の32背景参照感度。結果閲覧後の監査・方式選択対象外。"""
from pathlib import Path
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[key]="1"
import hashlib,json,sys,time,argparse,importlib.util,platform
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/"属性値集合による選別"
BASELINE=HERE.parent/"基準方式の背景感度"
spec=importlib.util.spec_from_file_location("frozen_column_value_methods",SOURCE/"分析.py")
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
spec2=importlib.util.spec_from_file_location("baseline_background_sensitivity",BASELINE/"分析.py")
bgsetup=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(bgsetup)
CFG=json.loads((HERE/"固定プロトコル.json").read_text());METHODS=CFG["methods"];SEEDS=CFG["seeds"]
assert bgsetup.SEEDS==SEEDS and bgsetup.REPEATS==CFG["null_repeats"]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_digest(x):return hashlib.sha256(np.asarray(x,dtype=np.int64).tobytes()).hexdigest()
def ev_from_distances(dc,db):
    r=np.log((db+exp.EPS)/(dc+exp.EPS))
    return dict(dc=dc,db=db,ratio=r,all4=r.mean(axis=1),top2=np.sort(r,axis=1)[:,-2:].mean(axis=1),exact=(dc<=exp.TOL).sum(axis=1))
def choose(q,ev):return np.stack([exp.independent_select(q,ev,m) for m in METHODS])
def stability(full,chosen,qsex,phase,team,seed,method):
    rows=[]
    for sex in ["all",0,1]:
        a=np.ones(len(full),bool) if sex=="all" else qsex==sex
        f,c=full[a],chosen[a];u=int((f|c).sum())
        rows.append(dict(phase=phase,team=team,seed=seed,method=method,sex=str(sex),q_n=int(a.sum()),
          selected_full=int(f.sum()),selected_reference=int(c.sum()),changed=int((f!=c).sum()),
          jaccard=float((f&c).sum()/u) if u else 1.))
    return rows
def frequency(qsex,full,real,team):
    summaries=[];hist=[]
    for mi,method in enumerate(METHODS):
        counts=real[:,mi].sum(axis=0)
        for sex in ["all",0,1]:
            a=np.ones(len(full[mi]),bool) if sex=="all" else qsex==sex
            n=counts[a];f=n/len(SEEDS);quant=np.quantile(f,[0,.05,.25,.5,.75,.95,1])
            summaries.append(dict(team=team,method=method,sex=str(sex),q_n=int(a.sum()),selected_full=int(full[mi,a].sum()),
                frequency_mean=float(f.mean()),minimum=float(quant[0]),p05=float(quant[1]),q25=float(quant[2]),median=float(quant[3]),
                q75=float(quant[4]),p95=float(quant[5]),maximum=float(quant[6]),always_selected=int((n==len(SEEDS)).sum()),
                never_selected=int((n==0).sum()),sometimes_selected=int(((n>0)&(n<len(SEEDS))).sum()),
                not_membership_probability=True))
            for k in range(len(SEEDS)+1):hist.append(dict(team=team,method=method,sex=str(sex),selected_in_32_ref=k,q_rows=int((n==k).sum())))
    return summaries,hist
def predict():
    start=time.time();out=HERE/"集計";work=HERE/"作業用";out.mkdir(exist_ok=True);work.mkdir(exist_ok=True)
    old=json.loads((SOURCE/"予測実行記録.json").read_text())
    assert old["code_sha256"]==digest(SOURCE/"分析.py") and old["config_sha256"]==digest(SOURCE/"固定条件.json")
    baseline_setup_hash=digest(BASELINE/"分析.py")
    for p,h in old["input_sha256"].items():assert digest(p)==h
    entries=[];stab=[];summaries=[];hist=[];sampling=[];reproduce=[];perms=[];public_null=[]
    for phase,folder,bgp in exp.SETS:
        bg=exp.core.read(bgp);_,bgsex,setups=bgsetup.reference_setup(bg)
        assert len(bg)==5143 and len(setups)==32
        for setup in setups:
            # 基準担当のimport関数と別の直接実装を一度照合し、抽出元索引のハッシュを保存。
            order=np.random.default_rng(setup["seed"]).permutation(len(bg));pool=order[:2571];ref=order[2571:]
            assert np.array_equal(pool,setup["pool"]) and np.array_equal(ref,setup["ref"])
            for repeat in range(CFG["null_repeats"]):
                rng=np.random.default_rng(setup["seed"]+400+repeat)
                for sex in [0,1]:
                    independent_order=rng.permutation(pool[bgsex[pool]==sex])
                    assert np.array_equal(independent_order,setup["draw_orders"][repeat][sex])
                    sampling.append(dict(phase=phase,seed=setup["seed"],repeat=repeat,sex=sex,
                        pool_n=len(pool),ref_n=len(ref),pool_reference_overlap=0,draw_order_sha256=array_digest(independent_order),
                        pool_sha256=array_digest(pool),ref_sha256=array_digest(ref),setup_matches_independent_implementation=True))
        for item in [x for x in old["predictions"] if x["phase"]==phase]:
            k=item["team"];q=exp.core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True);c=exp.core.read(folder/f"C_{k}.csv")
            cache=SOURCE/item["path"];assert digest(cache)==item["sha256"];prior=np.load(cache)
            full=choose(q,exp.evidence(q,c,bg));frozen=prior["real"][1,[1,2]]
            assert np.array_equal(full,frozen);reproduce.append(dict(phase=phase,team=k,changed_original_full_BG_masks=0))
            dreal=exp.nearest_columns(q,c);real=np.empty((32,2,len(q)),bool)
            null=np.empty((32,20,2,len(q)),bool) if phase=="prelim" and k!=5 else None
            qs=q.sex.to_numpy();cs=c.sex.to_numpy();n0,n1=int((cs==0).sum()),int((cs==1).sum())
            for si,setup in enumerate(setups):
                ref=bg.iloc[setup["ref"]];db=exp.nearest_columns(q,ref)
                e=ev_from_distances(dreal,db);selected=choose(q,e);real[si]=selected
                for mi,m in enumerate(METHODS):stab+=stability(full[mi],selected[mi],qs,phase,k,setup["seed"],m)
                # 1回の固定q/C行順照合。scoreと最近距離を再計算する。
                rng=np.random.default_rng(setup["seed"]+800+k);pi=rng.permutation(len(q));pj=rng.permutation(len(c))
                qp=q.iloc[pi].reset_index(drop=True);cp=c.iloc[pj].reset_index(drop=True)
                shuffled=choose(qp,exp.evidence(qp,cp,ref))[:,np.argsort(pi)]
                for mi,m in enumerate(METHODS):
                    changed=int((shuffled[mi]!=selected[mi]).sum());assert changed==0
                    perms.append(dict(phase=phase,team=k,seed=setup["seed"],method=m,changed=changed))
                if null is not None:
                    for repeat,orders in enumerate(setup["draw_orders"]):
                        sample=np.sort(np.r_[orders[0][:n0],orders[1][:n1]])
                        assert len(sample)==1049 and len(np.unique(sample))==1049 and set(sample).isdisjoint(setup["ref"])
                        assert int((bgsex[sample]==0).sum())==n0 and int((bgsex[sample]==1).sum())==n1
                        nc=bg.iloc[sample];dn=exp.nearest_columns(q,nc)
                        null[si,repeat]=choose(q,ev_from_distances(dn,db))
                        public_null.append(dict(team=k,seed=setup["seed"],repeat=repeat,sample_n=1049,sex0=n0,sex1=n1,
                           sample_sha256=array_digest(sample),pool_reference_overlap=0))
            if phase=="main":
                ss,hh=frequency(qs,full,real,k);summaries+=ss;hist+=hh
            saved=work/f"{phase}_{k}.npz";np.savez_compressed(saved,full=full,real=real,null=null if null is not None else np.empty(0))
            entries.append(dict(phase=phase,team=k,path=str(saved.relative_to(HERE)),sha256=digest(saved),completed_seeds=SEEDS))
            print(json.dumps(dict(stage="all_predictions_before_truth",phase=phase,team=k,seed_count=32,null_cases=0 if null is None else 640,seconds=round(time.time()-start,1))),flush=True)
    assert len(entries)==44 and digest(BASELINE/"分析.py")==baseline_setup_hash
    pd.DataFrame(stab).to_csv(out/"32参照_全BGからの選別変化_対象別性別.csv",index=False)
    pd.DataFrame(summaries).to_csv(out/"本戦_選択頻度の対象別性別要約.csv",index=False)
    pd.DataFrame(hist).to_csv(out/"本戦_選択頻度の対象別性別分布.csv",index=False)
    pd.DataFrame(sampling).to_csv(out/"BG分割と抽出順の独立実装照合.csv",index=False)
    pd.DataFrame(public_null).to_csv(out/"同性別nullの抽出索引ハッシュ.csv",index=False)
    pd.DataFrame(perms).to_csv(out/"行順変更.csv",index=False)
    pd.DataFrame(reproduce).to_csv(out/"全44C_凍結値集合方式の再現.csv",index=False)
    exp.writej(HERE/"予測実行記録.json",dict(start_time=time.strftime("%Y-%m-%dT%H:%M:%S%z",time.localtime(start)),seconds=time.time()-start,
        code_sha256=digest(__file__),protocol_sha256=digest(HERE/"固定プロトコル.json"),
        source_code_sha256=digest(SOURCE/"分析.py"),source_protocol_sha256=digest(SOURCE/"固定条件.json"),
        matched_reference_setup_code_sha256=baseline_setup_hash,input_sha256=dict(exp.core.HASHES),
        predictions=entries,completed_seeds=SEEDS,unfinished_seeds=[],actual_prediction_cases=44*32*2,null_prediction_cases=20*32*20*2,
        all_predictions_before_truth_read=True,truth_used_for_predictions=False,main_truth_used=False,B_self_used=False,
        selected_best_seed=False,selection_candidate=False,post_result_audit=True,permutation_tests=len(perms),permutation_changed_max=0,
        python=sys.executable,python_version=platform.python_version()))
def score_one(mask,truth):
    s=exp.core.scores(np.where(mask,9.5,0),truth);s["signed_hit_difference"]=s["hit_m"]-s["hit_c"];return s
def score():
    rec=json.loads((HERE/"予測実行記録.json").read_text())
    assert rec["code_sha256"]==digest(__file__) and rec["protocol_sha256"]==digest(HERE/"固定プロトコル.json")
    for item in rec["predictions"]:assert digest(HERE/item["path"])==item["sha256"]
    td=next((exp.ROOT/"予備戦/配布データ/結果").glob("正解*"));truthhash={};fullrows=[];realrows=[];nullrows=[]
    for item in rec["predictions"]:
        if item["phase"]!="prelim":continue
        k=item["team"];tp=td/f"aia_truth_{k}.csv";truth=exp.core.read(tp).sort_values("challenge_row_id").reset_index(drop=True);truthhash[str(tp)]=digest(tp)
        data=np.load(HERE/item["path"])
        for mi,m in enumerate(METHODS):
            fullrows.append(dict(team=k,method=m,**score_one(data["full"][mi],truth)))
            for si,seed in enumerate(SEEDS):
                realrows.append(dict(team=k,seed=seed,method=m,**score_one(data["real"][si,mi],truth)))
                if len(data["null"]):
                    for repeat in range(CFG["null_repeats"]):
                        nullrows.append(dict(team=k,seed=seed,method=m,repeat=repeat,**score_one(data["null"][si,repeat,mi],truth)))
    out=HERE/"集計";full=pd.DataFrame(fullrows);real=pd.DataFrame(realrows);null=pd.DataFrame(nullrows)
    full.to_csv(out/"全BG_凍結値集合方式_予備21対象.csv",index=False)
    real.to_csv(out/"32参照_実C対象別.csv",index=False)
    null.to_csv(out/"32参照_同性別BG_C全反復.csv",index=False)
    keys=["r_all","r_rare","r_max","p_m","p_c","hit_m","hit_c","signed_hit_difference"]
    mean_null=null.groupby(["seed","team","method"])[keys].mean().reset_index()
    paired=real.merge(mean_null,on=["seed","team","method"],suffixes=("_real","_null"))
    paired["r_excess"]=paired.r_all_real-paired.r_all_null
    paired["signed_hit_excess"]=paired.signed_hit_difference_real-paired.signed_hit_difference_null
    paired.to_csv(out/"32参照_実Cと対照差_対象別.csv",index=False)
    st=pd.read_csv(out/"32参照_全BGからの選別変化_対象別性別.csv");rows=[]
    for scope,excluded in [("all20",[5]),("processed18",[1,2,5])]:
        for m in METHODS:
            f=full[(full.method==m)&~full.team.isin(excluded)]
            d=paired[(paired.method==m)&~paired.team.isin(excluded)]
            for seed,z in d.groupby("seed"):
                s=st[(st.phase=="prelim")&(st.sex=="all")&(st.seed==seed)&(st.method==m)&~st.team.isin(excluded)]
                rows.append(dict(scope=scope,method=m,seed=int(seed),targets=len(z),real_r_all=float(z.r_all_real.mean()),
                  null_r_all=float(z.r_all_null.mean()),r_excess=float(z.r_excess.mean()),
                  real_signed_hit_difference=float(z.signed_hit_difference_real.mean()),null_signed_hit_difference=float(z.signed_hit_difference_null.mean()),
                  signed_hit_excess=float(z.signed_hit_excess.mean()),full_BG_r_all=float(f.r_all.mean()),
                  full_BG_signed_hit_difference=float(f.signed_hit_difference.mean()),
                  selected_changed_mean=float(s.changed.mean()),jaccard_mean=float(s.jaccard.mean())))
    summary=pd.DataFrame(rows);summary.to_csv(out/"32参照_20対象と18対象のseed別平均.csv",index=False)
    metrics=["real_r_all","null_r_all","r_excess","real_signed_hit_difference","null_signed_hit_difference","signed_hit_excess","selected_changed_mean","jaccard_mean"]
    distribution=[]
    for (scope,m),z in summary.groupby(["scope","method"]):
        assert len(z)==32
        for key in metrics:
            v=z[key].to_numpy();q=np.quantile(v,[0,.05,.25,.5,.75,.95,1])
            distribution.append(dict(scope=scope,method=m,metric=key,reference_count=32,mean=float(v.mean()),sd=float(v.std(ddof=1)),
                minimum=float(q[0]),p05=float(q[1]),q25=float(q[2]),median=float(q[3]),q75=float(q[4]),p95=float(q[5]),maximum=float(q[6]),
                independent_validation_interval=False))
    pd.DataFrame(distribution).to_csv(out/"32参照_感度分布_独立検証区間ではない.csv",index=False)
    # 同一seed/同一null drawsの旧基準と比較する。方法固有の変換は異なる。
    bp=BASELINE/"集計/32参照_20対象と18対象のseed別平均.csv";baseline=pd.read_csv(bp)
    compare=summary.merge(baseline,on=["scope","seed"],suffixes=("_column","_baseline"))
    compare["real_R_minus_baseline"]=compare.real_r_all_column-compare.real_r_all_baseline
    compare["excess_R_minus_baseline"]=compare.r_excess_column-compare.r_excess_baseline
    compare["signed_excess_minus_baseline"]=compare.signed_hit_excess_column-compare.signed_hit_excess_baseline
    compare.to_csv(out/"同一分割での旧基準比較.csv",index=False)
    exp.writej(HERE/"採点実行記録.json",dict(prediction_receipt_sha256=digest(HERE/"予測実行記録.json"),code_sha256=digest(__file__),
        protocol_sha256=digest(HERE/"固定プロトコル.json"),public_prelim_truth_sha256=truthhash,
        baseline_seed_summary_sha256=digest(bp),all44x32_predictions_completed_before_truth_read=True,main_truth_used=False,B_self_used=False,
        selected_best_seed=False,selection_candidate=False,post_result_audit=True))
    print(pd.DataFrame(distribution).query("scope == 'processed18' and metric in ['real_r_all','null_r_all','r_excess','real_signed_hit_difference','signed_hit_excess','selected_changed_mean','jaccard_mean']").to_string(index=False),flush=True)
    print(compare.query("scope == 'processed18'").groupby("method")[["real_R_minus_baseline","excess_R_minus_baseline","signed_excess_minus_baseline"]].agg(["mean","min","max"]).to_string(),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["predict","score"],required=True);args=p.parse_args()
    predict() if args.stage=="predict" else score()

