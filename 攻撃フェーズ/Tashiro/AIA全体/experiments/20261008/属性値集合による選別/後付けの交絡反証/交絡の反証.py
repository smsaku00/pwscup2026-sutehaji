"""主結果の閲覧後に追加した交絡の反証。方式選択対象外、追加調整なし。"""
from pathlib import Path
import sys,json,time,hashlib,argparse,importlib.util
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location("parent_column_experiment",PARENT/"分析.py")
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
CFG=json.loads((HERE/"追加固定条件.json").read_text());METHODS=CFG["methods"]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def independent(q,scores):
    result=np.zeros(len(q),dtype=bool);ids=q.challenge_row_id.to_numpy()
    for sex in np.unique(q.sex):
        a=np.flatnonzero(q.sex.to_numpy()==sex)
        order=np.lexsort((ids[a],-scores[a]));result[a[order[:int(np.ceil(len(a)*.75))]]]=True
    return result
def control_scores(q,c,ref,scale):
    pair,dc=exp.baseline_pair(q,c,ref,scale)
    boundary=np.partition(dc,11,axis=1)[:,11]
    eligible=np.isfinite(dc)&(dc<=boundary[:,None])
    first=np.where(eligible,pair,-1e4).max(axis=1)
    db=exp.nearest_columns(q,ref)
    r=np.log((db[:,None,:]+exp.EPS)/(np.abs(q[exp.COLS].to_numpy()[:,None,:]-c[exp.COLS].to_numpy()[None,:,:])+exp.EPS))
    all4=r.mean(axis=2)
    top2=np.sort(r,axis=2)[:,:,-2:].mean(axis=2)
    wrong=q.sex.to_numpy()[:,None]!=c.sex.to_numpy()[None,:]
    all4[wrong]=-np.inf;top2[wrong]=-np.inf
    second=all4.max(axis=1);third=top2.max(axis=1)
    assert np.isfinite(np.stack([first,second,third])).all()
    return np.stack([first,second,third]),pair,dc
def predict():
    start=time.time();out=HERE/"集計";work=HERE/"作業用";out.mkdir(exist_ok=True);work.mkdir(exist_ok=True)
    parent=json.loads((PARENT/"予測実行記録.json").read_text())
    assert parent["code_sha256"]==digest(PARENT/"分析.py")
    for p,sha in parent["input_sha256"].items():assert digest(p)==sha
    entries=[];perms=[];repro=[];diagnostics=[];inequalities=[]
    for phase,folder,bgp in exp.SETS:
        bg=exp.core.read(bgp);raw=exp.core.raw(bg);scale=np.maximum(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0),1e-5)
        order=np.random.default_rng(exp.CFG["split_seed"]+(phase=="main")).permutation(len(bg));midpoint=len(bg)//2
        pool=bg.iloc[order[:midpoint]].reset_index(drop=True);ref=bg.iloc[order[midpoint:]].reset_index(drop=True)
        for item in [x for x in parent["predictions"] if x["phase"]==phase]:
            k=item["team"];q=exp.core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True);c=exp.core.read(folder/f"C_{k}.csv")
            parent_cache=PARENT/item["path"];assert digest(parent_cache)==item["sha256"];old=np.load(parent_cache)
            actual=[];null=[]
            ix=exp.sample_indices(pool,c,exp.CFG["null_repeats"],exp.CFG["null_seed"]+1000*k+(phase=="main")*100000)
            for ci,context in enumerate(exp.CONTEXTS):
                reference=ref if ci==0 else bg;scores,pair,dc=control_scores(q,c,reference,scale)
                selected=np.stack([independent(q,s) for s in scores]);actual.append(selected)
                original=exp.stable.select(pair,dc,q.sex.to_numpy(),c.sex.to_numpy(),"sex_quota")
                assert np.array_equal(original,old["real"][ci,0])
                repro.append(dict(phase=phase,team=k,context=context,baseline_changed=0))
                unrestricted=exp.evidence(q,c,reference)
                violations=int((scores[1]>unrestricted["all4"]+1e-12).sum()+(scores[2]>unrestricted["top2"]+1e-12).sum())
                assert violations==0
                inequalities.append(dict(phase=phase,team=k,context=context,same_row_score_exceeds_column_set_score=violations))
                for mi,method in enumerate(METHODS):
                    diagnostics.append(dict(phase=phase,team=k,context=context,method=method,selected=int(selected[mi].sum()),
                        changed_vs_Hungarian=int((selected[mi]!=original).sum()),score_mean=float(scores[mi].mean())))
                for repeat in range(exp.CFG["permutation_repeats"]):
                    rng=np.random.default_rng(exp.CFG["permutation_seed"]+100*k+repeat+(phase=="main")*100000)
                    pi=rng.permutation(len(q));pj=rng.permutation(len(c));qp=q.iloc[pi].reset_index(drop=True);cp=c.iloc[pj].reset_index(drop=True)
                    # old pairは固定距離の行列を並べ替え、同一行対照は再計算で列/行順を照合。
                    boundary2=np.partition(dc[pi][:,pj],11,axis=1)[:,11]
                    eligible2=np.isfinite(dc[pi][:,pj])&(dc[pi][:,pj]<=boundary2[:,None])
                    oldscore=np.where(eligible2,pair[pi][:,pj],-1e4).max(axis=1)
                    db=exp.nearest_columns(qp,reference)
                    rr=np.log((db[:,None,:]+exp.EPS)/(np.abs(qp[exp.COLS].to_numpy()[:,None,:]-cp[exp.COLS].to_numpy()[None,:,:])+exp.EPS))
                    s4=rr.mean(axis=2);s2=np.sort(rr,axis=2)[:,:,-2:].mean(axis=2)
                    wrong=qp.sex.to_numpy()[:,None]!=cp.sex.to_numpy()[None,:];s4[wrong]=-np.inf;s2[wrong]=-np.inf
                    ps=np.stack([oldscore,s4.max(axis=1),s2.max(axis=1)])
                    shuffled=np.stack([independent(qp,s) for s in ps])[:,np.argsort(pi)]
                    for mi,method in enumerate(METHODS):
                        diff=int((shuffled[mi]!=selected[mi]).sum());assert diff==0
                        perms.append(dict(phase=phase,team=k,context=context,method=method,repeat=repeat,changed=diff))
                if phase=="prelim":
                    nullsets=[]
                    for indices in ix:
                        nc=pool.iloc[indices].reset_index(drop=True);s,_,_=control_scores(q,nc,reference,scale)
                        nullsets.append(np.stack([independent(q,v) for v in s]))
                    null.append(np.stack(nullsets))
            cache=work/f"{phase}_{k}.npz";np.savez_compressed(cache,real=np.stack(actual),null=np.stack(null) if null else np.empty(0))
            entries.append(dict(phase=phase,team=k,path=str(cache.relative_to(HERE)),sha256=digest(cache)))
            print(json.dumps(dict(stage="control_predict",phase=phase,team=k,seconds=round(time.time()-start,1))),flush=True)
    pd.DataFrame(perms).to_csv(out/"行順変更.csv",index=False)
    pd.DataFrame(repro).to_csv(out/"既存基準の再現.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(out/"公開課題の選別変更.csv",index=False)
    pd.DataFrame(inequalities).to_csv(out/"同一行制約の数式不等式.csv",index=False)
    exp.writej(HERE/"対照予測実行記録.json",dict(start_time=time.strftime("%Y-%m-%dT%H:%M:%S%z",time.localtime(start)),
       seconds=time.time()-start,code_sha256=digest(__file__),config_sha256=digest(HERE/"追加固定条件.json"),
       parent_config_sha256=digest(PARENT/"固定条件.json"),parent_prediction_receipt_sha256=digest(PARENT/"予測実行記録.json"),
       input_sha256=dict(exp.core.HASHES),predictions=entries,truth_used_for_prediction=False,main_truth_used=False,B_self_used=False,
       selection_candidate=False,post_result_control=True,permutation_tests=len(perms),permutation_max=0,baseline_reproduction_changed_max=0))
def score():
    rec=json.loads((HERE/"対照予測実行記録.json").read_text());assert rec["code_sha256"]==digest(__file__) and rec["config_sha256"]==digest(HERE/"追加固定条件.json")
    for item in rec["predictions"]:assert digest(HERE/item["path"])==item["sha256"]
    td=next((exp.ROOT/"予備戦/配布データ/結果").glob("正解*"));rows=[];truthhash={}
    for item in rec["predictions"]:
        if item["phase"]!="prelim":continue
        k=item["team"];tp=td/f"aia_truth_{k}.csv";truth=exp.core.read(tp).sort_values("challenge_row_id").reset_index(drop=True);truthhash[str(tp)]=digest(tp)
        data=np.load(HERE/item["path"])
        for ci,context in enumerate(exp.CONTEXTS):
            for mi,method in enumerate(METHODS):
                rows.append(dict(team=k,context=context,method=method,source="real_C",repeat=-1,**exp.core.scores(np.where(data["real"][ci,mi],9.5,0),truth)))
                for repeat in range(exp.CFG["null_repeats"]):
                    rows.append(dict(team=k,context=context,method=method,source="BG_null_C",repeat=repeat,**exp.core.scores(np.where(data["null"][ci,repeat,mi],9.5,0),truth)))
    table=pd.DataFrame(rows);out=HERE/"集計";table.to_csv(out/"後付け対照_対象別と20反復null.csv",index=False)
    parent=pd.read_csv(PARENT/"集計/予備戦_対象別と20反復null.csv")
    both=pd.concat([parent,table])
    aggregates=[]
    for group,excluded in [("prelim20",[5]),("processed18",[1,2,5])]:
        a=both[~both.team.isin(excluded)].groupby(["context","method","source"])[exp.METRICS].mean().reset_index()
        a.insert(0,"group",group);a["post_result_control"]=a.method.isin(METHODS);aggregates.append(a)
    avg=pd.concat(aggregates);avg.to_csv(out/"主候補と後付け対照の平均.csv",index=False)
    actual=both[both.source=="real_C"];piv=actual.pivot(index=["team","context"],columns="method",values="r_all").reset_index()
    piv["independent_old_pair_minus_Hungarian"]=piv.old_pair_independent_sex75-piv.baseline_partial2_matching
    piv["all4_column_set_minus_same_row"]=piv.column_all4_mean-piv.same_row_all4_independent_sex75
    piv["top2_column_set_minus_same_row"]=piv.column_top2_mean-piv.same_row_top2_independent_sex75
    piv.to_csv(out/"対象別の交絡差.csv",index=False)
    exp.writej(HERE/"対照採点実行記録.json",dict(prediction_receipt_sha256=digest(HERE/"対照予測実行記録.json"),
      public_prelim_truth_sha256=truthhash,post_result_control=True,selection_candidate=False,
      main_truth_used=False,B_self_used=False,all44_predictions_completed_before_control_truth_scoring=True))
    print(avg[(avg.group=="processed18")&(avg.source=="real_C")].to_string(index=False),flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["predict","score"],required=True);args=parser.parse_args()
    predict() if args.stage=="predict" else score()

