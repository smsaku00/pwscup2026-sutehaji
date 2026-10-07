"""固定P01の背景参照感度。32分割を独立した検証対象数とは扱わない。

32固定seed、pool2571/ref2572、実Cと同数・同性別人数のBG_C20反復。
全44Cとnullの選択を生成後、公開予備戦正解だけを読み採点する。
本戦のq別頻度は作業用NPZに限定し、共有は対象別・sex別の分布だけ。
候補方式を増やさず、best seedを選ばず、提出処理は行わない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key]="1"
import hashlib,importlib.util,io,json,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
OLD=HERE.parents[1]/"20261007_AIA全体向上の分析"
OUT=HERE/"集計"
WORK=HERE/"作業用"
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location("fixed_P01",OLD/"同点に依存しない対応.py")
stable=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable)
core=stable.core
SEEDS=list(range(20261008,20261008+32))
REPEATS=20
HASHES={}
PROTOCOL=dict(seeds=SEEDS,seed_count=32,null_repeats=REPEATS,
              method="fixed partial2/continuous/sex75/Hungarian/9.5-0",new_candidate_methods=0,
              full_BG_n=5143,pool_n=2571,reference_n=2572,
              sample_n=1049,sample_sex_counts="identical to each actual C",
              sample_without_replacement=True,
              per_repeat_sampling="same per-sex shuffled pool orders across target teams; take required counts",
              scale="log1p TG/ALT; IQR from the chosen reference only",
              same_query_adaptive_BG_CDF=True,near_limit=12,boundary_ties_included=True,
              tiny_full5_tie_break=1e-10,main_truth_used=False,
              predictions_before_truth_loading=True,main_row_frequency_work_only=True,
              reference_resamplings_are_independent_validation_units=False,
              choose_best_seed=False,official_phi_reproduction=False,submission_created=False)


def read(path):
    content=Path(path).read_bytes()
    HASHES[str(path)]=hashlib.sha256(content).hexdigest()
    return pd.read_csv(io.BytesIO(content),float_precision="round_trip")


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def smooth_score(dc,sorted_ref,ref_n):
    score=np.empty_like(dc)
    for i,(d,ref) in enumerate(zip(dc,sorted_ref)):
        finite=ref[np.isfinite(ref)]
        count=np.interp(d,np.r_[0.,finite],np.r_[0.,np.arange(1,len(finite)+1)],right=len(finite))
        score[i]=-np.log((count+.25)/(ref_n+1))
    score[~np.isfinite(dc)]=-1e4
    return score


def choose(dc,full_distance,sorted_ref,ref_n,qsex,csex):
    pair=smooth_score(dc,sorted_ref,ref_n)-stable.EPSILON*np.minimum(full_distance,1000.)
    limit=np.partition(dc,11,axis=1)[:,11]
    valid=np.isfinite(dc)&(dc<=limit[:,None])
    limited=np.where(valid,pair,-1e4)
    chosen=np.zeros(len(qsex),bool)
    bad=0
    for sex in np.unique(qsex):
        a,b=qsex==sex,csex==sex
        count=int(np.ceil(a.sum()*.75))
        local=limited[a][:,b]
        selected,assigned=stable.partial_assignment(local,count)
        chosen[a]=selected
        bad+=int((local[np.flatnonzero(selected),assigned[selected]]<=-9999).sum())
    return chosen,bad


def distances(x,y,qsex,ysex):
    sq=(x[:,None]-y[None])**2
    d=np.sort(sq,axis=2)[:,:,:2].mean(2)
    d[qsex[:,None]!=ysex[None]]=np.inf
    return d,cdist(x,y,"sqeuclidean")/5


def reference_setup(bg):
    raw=core.raw(bg)
    sex=bg.sex.to_numpy()
    setups=[]
    for seed in SEEDS:
        order=np.random.default_rng(seed).permutation(len(bg))
        pool,ref=order[:len(bg)//2],order[len(bg)//2:]
        assert len(pool)==2571 and len(ref)==2572 and set(pool).isdisjoint(ref)
        rb=raw[ref]
        scale=np.maximum(np.quantile(rb,.75,axis=0)-np.quantile(rb,.25,axis=0),1e-5)
        permutations=[]
        for repeat in range(REPEATS):
            rng=np.random.default_rng(seed+400+repeat)
            permutations.append({value:rng.permutation(pool[sex[pool]==value]) for value in [0,1]})
        setups.append(dict(seed=seed,pool=pool,ref=ref,scale=scale,norm_BG=raw/scale,draw_orders=permutations))
    return raw,sex,setups


def stability_record(full,chosen,qsex,phase,team,seed,bad):
    records=[]
    for sex in ["all",0,1]:
        rows=np.ones(len(full),bool) if sex=="all" else qsex==sex
        a,b=full[rows],chosen[rows]
        union=(a|b).sum()
        records.append(dict(phase=phase,team=team,seed=seed,sex=str(sex),q_n=int(rows.sum()),
                            selected_full=int(a.sum()),selected_reference=int(b.sum()),
                            changed=int((a!=b).sum()),jaccard=float((a&b).sum()/union) if union else 1.,
                            selected_bad_pairs=bad if sex=="all" else np.nan))
    return records


def frequency_stats(main,team,qsex,full,selection_counts):
    summaries,hist=[] ,[]
    for sex in ["all",0,1]:
        rows=np.ones(len(full),bool) if sex=="all" else qsex==sex
        count=selection_counts[rows]
        freq=count/len(SEEDS)
        f=full[rows]
        quant=np.quantile(freq,[0,.05,.25,.5,.75,.95,1])
        summaries.append(dict(phase=main,team=team,sex=str(sex),q_n=len(count),selected_full=int(f.sum()),
                              frequency_mean=float(freq.mean()),frequency_min=float(quant[0]),frequency_p05=float(quant[1]),
                              frequency_q25=float(quant[2]),frequency_median=float(quant[3]),frequency_q75=float(quant[4]),
                              frequency_p95=float(quant[5]),frequency_max=float(quant[6]),
                              always_selected=int((count==len(SEEDS)).sum()),never_selected=int((count==0).sum()),
                              sometimes_selected=int(((count>0)&(count<len(SEEDS))).sum()),
                              full_selected_frequency_mean=float(freq[f].mean()) if f.any() else np.nan,
                              full_unselected_frequency_mean=float(freq[~f].mean()) if (~f).any() else np.nan,
                              not_membership_probability=True))
        for n in range(len(SEEDS)+1):
            size=int((count==n).sum())
            hist.append(dict(phase=main,team=team,sex=str(sex),selected_in_32_ref=n,q_rows=size,q_share=size/len(count)))
    return summaries,hist


def score_chosen(chosen,truth):
    rec=core.scores(np.where(chosen,9.5,0.),truth)
    rec["signed_hit_difference"]=rec["hit_m"]-rec["hit_c"]
    return rec


def run():
    start=time.time()
    OUT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(exist_ok=True)
    assert len(SEEDS)==32
    dump(HERE/"固定プロトコル.json",PROTOCOL)
    truth_folder=next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    prior_teams=sorted(int(p.stem.rsplit("_",1)[1]) for p in truth_folder.glob("aia_truth_*.csv"))
    assert len(prior_teams)==21
    phases=[("prelim",ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack",ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv",prior_teams),
            ("main",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv",None)]
    pending,stability,freq_summary,freq_hist,main_work,negative_diag=[],[],[],[],[],[]
    full_matches=0
    for phase,folder,bgpath,teams in phases:
        bg=read(bgpath)
        assert len(bg)==5143
        raw_bg,bgsex,setups=reference_setup(bg)
        if teams is None:
            teams=sorted(int(p.stem.rsplit("_",1)[1]) for p in folder.glob("C_*.csv"))
        fullscale=np.maximum(np.quantile(raw_bg,.75,axis=0)-np.quantile(raw_bg,.25,axis=0),1e-5)
        for team in teams:
            q=read(folder/f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c=read(folder/f"C_{team}.csv")
            qr,cr=core.raw(q),core.raw(c)
            qs,cs=q.sex.to_numpy(),c.sex.to_numpy()
            full_dc,full_dist=distances(qr/fullscale,cr/fullscale,qs,cs)
            full_db,_=distances(qr/fullscale,raw_bg/fullscale,qs,bgsex)
            full,bad=choose(full_dc,full_dist,np.sort(full_db,axis=1),len(bg),qs,cs)
            reference_pair,reference_dc=stable.matrices(q,c,bg)
            reference_chosen=stable.select(reference_pair,reference_dc,qs,cs,"sex_quota")
            assert np.array_equal(full,reference_chosen)
            full_matches+=1
            real=np.empty((len(SEEDS),len(q)),bool)
            null=np.empty((len(SEEDS),REPEATS,len(q)),bool) if phase=="prelim" and team!=5 else None
            for i,setup in enumerate(setups):
                x=qr/setup["scale"]
                dc,full_c=distances(x,cr/setup["scale"],qs,cs)
                # BG_CはBGの行なので、正規化済みBGとの距離を一度計算して再利用する。
                dbg,full_bg=distances(x,setup["norm_BG"],qs,bgsex)
                sorted_ref=np.sort(dbg[:,setup["ref"]],axis=1)
                chosen,bad=choose(dc,full_c,sorted_ref,len(setup["ref"]),qs,cs)
                real[i]=chosen
                stability+=stability_record(full,chosen,qs,phase,team,setup["seed"],bad)
                if null is not None:
                    n0,n1=int((cs==0).sum()),int((cs==1).sum())
                    for repeat,orders in enumerate(setup["draw_orders"]):
                        assert len(orders[0])>=n0 and len(orders[1])>=n1
                        sample=np.sort(np.r_[orders[0][:n0],orders[1][:n1]])
                        assert len(sample)==len(c)==1049 and len(np.unique(sample))==len(sample)
                        assert set(sample).isdisjoint(setup["ref"])
                        assert int((bgsex[sample]==0).sum())==n0 and int((bgsex[sample]==1).sum())==n1
                        null_chosen,null_bad=choose(dbg[:,sample],full_bg[:,sample],sorted_ref,len(setup["ref"]),qs,bgsex[sample])
                        null[i,repeat]=null_chosen
                        negative_diag.append(dict(team=team,seed=setup["seed"],repeat=repeat,sample_n=len(sample),
                                                  sample_sex0=n0,sample_sex1=n1,selected=int(null_chosen.sum()),selected_bad_pairs=null_bad,
                                                  pool_reference_disjoint=True,sex_counts_match_actual_C=True))
            if phase=="main":
                counts=real.sum(0).astype(np.uint8)
                summaries,hist=frequency_stats(phase,team,qs,full,counts)
                freq_summary+=summaries
                freq_hist+=hist
                main_work.append(dict(team=team,ids=q.challenge_row_id.to_numpy(),sex=qs,full=full,counts=counts))
            pending.append(dict(phase=phase,team=team,ids=q.challenge_row_id.tolist(),full=full,real=real,null=null))
            print(json.dumps(dict(stage="all_predictions_before_truth",phase=phase,team=team,seed_count=len(SEEDS),
                                  null_cases=0 if null is None else len(SEEDS)*REPEATS,seconds=round(time.time()-start,1))),flush=True)
    assert len(pending)==44 and full_matches==44
    assert sum(rec["null"] is not None for rec in pending)==20
    np.savez_compressed(WORK/"本戦23C_選択頻度_作業用.npz",teams=np.array([rec["team"] for rec in main_work]),
                        challenge_row_ids=np.stack([rec["ids"] for rec in main_work]),sex=np.stack([rec["sex"] for rec in main_work]),
                        full_BG_selected=np.stack([rec["full"] for rec in main_work]),
                        selected_in_32_ref=np.stack([rec["counts"] for rec in main_work]),seeds=np.array(SEEDS))
    # 公開正解の読込みは全実C・対照Cの選択生成後。
    truths={team:read(truth_folder/f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True) for team in prior_teams}
    full_rows,real_rows,null_rows=[],[],[]
    for rec in pending:
        if rec["phase"]!="prelim":
            continue
        team=rec["team"]
        truth=truths[team]
        assert truth.challenge_row_id.tolist()==rec["ids"]
        full_rows.append(dict(team=team,**score_chosen(rec["full"],truth)))
        for i,seed in enumerate(SEEDS):
            real_rows.append(dict(team=team,seed=seed,**score_chosen(rec["real"][i],truth)))
            if rec["null"] is not None:
                for repeat in range(REPEATS):
                    null_rows.append(dict(team=team,seed=seed,repeat=repeat,**score_chosen(rec["null"][i,repeat],truth)))
    full_table,real_table,null_table=map(pd.DataFrame,[full_rows,real_rows,null_rows])
    full_table.to_csv(OUT/"全BG基準_予備戦21対象.csv",index=False)
    real_table.to_csv(OUT/"32参照_実C対象別.csv",index=False)
    null_table.to_csv(OUT/"32参照_同性別BG_C全反復.csv",index=False)
    stability=pd.DataFrame(stability)
    stability.to_csv(OUT/"全BG基準からの選択変化_対象別性別.csv",index=False)
    pd.DataFrame(freq_summary).to_csv(OUT/"本戦_32参照での選択頻度の対象別性別要約.csv",index=False)
    pd.DataFrame(freq_hist).to_csv(OUT/"本戦_32参照での選択頻度の対象別性別分布.csv",index=False)
    pd.DataFrame(negative_diag).to_csv(OUT/"同性別BG_Cの候補制限監査.csv",index=False)
    keys=["r_all","r_rare","r_max","p_m","p_c","signed_hit_difference"]
    mean_null=null_table.groupby(["seed","team"])[keys].mean().reset_index()
    paired=real_table.merge(mean_null,on=["seed","team"],suffixes=("_real","_null"))
    paired["r_excess"]=paired.r_all_real-paired.r_all_null
    paired["signed_hit_excess"]=paired.signed_hit_difference_real-paired.signed_hit_difference_null
    paired.to_csv(OUT/"32参照_実Cと対照差_対象別.csv",index=False)
    summaries=[]
    for scope,excluded in [("all20",[5]),("processed18",[1,2,5])]:
        data=paired[~paired.team.isin(excluded)]
        full_scope=full_table[~full_table.team.isin(excluded)]
        for seed,one in data.groupby("seed"):
            st=stability[(stability.phase=="prelim")&(stability.sex=="all")&(stability.seed==seed)&~stability.team.isin(excluded)]
            summaries.append(dict(scope=scope,seed=int(seed),targets=len(one),
                                  real_r_all=float(one.r_all_real.mean()),null_r_all=float(one.r_all_null.mean()),r_excess=float(one.r_excess.mean()),
                                  real_signed_hit_difference=float(one.signed_hit_difference_real.mean()),
                                  null_signed_hit_difference=float(one.signed_hit_difference_null.mean()),signed_hit_excess=float(one.signed_hit_excess.mean()),
                                  full_BG_r_all=float(full_scope.r_all.mean()),full_BG_signed_hit_difference=float(full_scope.signed_hit_difference.mean()),
                                  selected_changed_mean=float(st.changed.mean()),jaccard_mean=float(st.jaccard.mean())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/"32参照_20対象と18対象のseed別平均.csv",index=False)
    distribution=[]
    metrics=["real_r_all","null_r_all","r_excess","real_signed_hit_difference","null_signed_hit_difference","signed_hit_excess","selected_changed_mean","jaccard_mean"]
    for scope,data in summary.groupby("scope"):
        assert len(data)==32
        for metric in metrics:
            values=data[metric].to_numpy()
            q=np.quantile(values,[0,.05,.25,.5,.75,.95,1])
            distribution.append(dict(scope=scope,metric=metric,reference_resamplings=32,mean=float(values.mean()),sd=float(values.std(ddof=1)),
                                      minimum=float(q[0]),p05=float(q[1]),q25=float(q[2]),median=float(q[3]),q75=float(q[4]),p95=float(q[5]),maximum=float(q[6]),
                                      independent_validation_target_count=len(data.targets.unique())==0))
    pd.DataFrame(distribution).to_csv(OUT/"32参照の感度分布_独立検証区間ではない.csv",index=False)
    st_columns=["changed","jaccard","selected_reference","selected_bad_pairs"]
    stability.groupby(["phase","team","sex"])[st_columns].agg(["mean","std","min","max"]).to_csv(OUT/"選択変化の32参照要約.csv")
    saved=read(OLD/"集計/同点修正_対象別.csv")
    saved=saved[saved.method.eq("stable_matching/sex_quota/long75")].set_index("team")
    full_table=full_table.set_index("team")
    assert np.array_equal(full_table.hit_m,saved.loc[full_table.index,"hit_m"])
    assert np.array_equal(full_table.hit_c,saved.loc[full_table.index,"hit_c"])
    assert len(null_rows)==32*20*20
    dump(HERE/"実行記録.json",dict(PROTOCOL,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         protocol_sha256=hashlib.sha256((HERE/"固定プロトコル.json").read_bytes()).hexdigest(),input_sha256=HASHES,
         imported_code_sha256={str(OLD/name):hashlib.sha256((OLD/name).read_bytes()).hexdigest() for name in ["分析.py","同点に依存しない対応.py","対応競合の分析.py","対応競合の頑健性.py"]},
         python=sys.executable,python_version=platform.python_version(),seconds=time.time()-start,
         full_prediction_matches_frozen_code_cases=full_matches,baseline_full21_hit_counts_match_saved=True,
         null_prediction_cases=len(null_rows),real_reference_prediction_cases=44*32,
         max_real_selected_bad_pairs=int(stability.selected_bad_pairs.max()),max_null_selected_bad_pairs=max(r["selected_bad_pairs"] for r in negative_diag),
         working_main_frequency_file="作業用/本戦23C_選択頻度_作業用.npz",share_individual_frequency=False))
    print(summary.groupby("scope")[metrics].agg(["mean","min","max"]).to_string(),flush=True)


if __name__=="__main__":
    run()
