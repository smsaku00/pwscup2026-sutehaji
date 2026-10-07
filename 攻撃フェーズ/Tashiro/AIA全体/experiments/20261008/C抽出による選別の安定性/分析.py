"""Cの性別内80%抽出を32回繰り返し、選別頻度の追加的な寄与を調べる。

頻度を会員確率とは呼ばない。候補は基準と追加3方式に固定する。
本戦の正解とB_selfを読まず、提出物を作らない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/"AGENTS.md").exists())
OLD=HERE.parent.parent/"20261007_AIA全体向上の分析"
sys.path.insert(0,str(OLD))
import 分析 as core
from 対応競合の頑健性 import distance,pair_score
from 同点に依存しない対応 import select

SEED=20261008
SUBSAMPLES=32
NULL_REPEATS=20
METHODS=["baseline","frequency_quota75","frequency_pairrank_equal_quota75","frequency_ge075"]
PLAN=dict(methods=METHODS,subsamples=SUBSAMPLES,C_sample_fraction=.8,sample_within_sex=True,
    sample_count="floor(0.8*n_sex)",replacement=False,sample_seed_base=SEED+1200,
    canonical_C_order="sorted hash of all public C columns",canonical_query_order="QI then challenge_row_id",
    base="partial2 continuousBG sexquota75 Hungarian long9.5",base_priority="max calibrated pair with original epsilon",
    equal_weight="0.5*frequency + 0.5*within_sex_rank(base_priority)",
    final_ties="max calibrated pair, then canonical query tuple",threshold=.75,
    time_selected=9.5,time_other=0.,main_metric="r_all",null_repeats=NULL_REPEATS,
    null_pool_reference_disjoint=True,frequency_is_membership_probability=False,
    truth_used_for_predictions=False,B_self_read=False,main_truth_read=False,
    official_phi=False,submission_created=False,codabench_submitted=False)


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def scale_for(bg):
    b=core.raw(bg)
    return np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0),1e-5)


def matrices(q,c,bg,scale,db=None):
    dc=distance(q,c,scale)
    if db is None:db=distance(q,bg,scale)
    pair=pair_score(dc,db)["continuous"]
    full=cdist(core.raw(q)/scale,core.raw(c)/scale,"sqeuclidean")/5
    pair-=1e-10*np.minimum(full,1000.)
    return pair,dc


def canonical_C_order(c):
    return np.argsort(pd.util.hash_pandas_object(c,index=False).to_numpy(),kind="stable")


def canonical_q_order(q):
    return q.assign(_position=np.arange(len(q))).sort_values(core.QI+["challenge_row_id"],kind="stable")._position.to_numpy()


def top_sex(q,score,priority):
    chosen=np.zeros(len(q),bool)
    # qはcanonical順。最後の同点もこの同じ入力値順で解消する。
    for sex in np.unique(q.sex):
        positions=np.flatnonzero(q.sex.eq(sex).to_numpy())
        count=int(np.ceil(len(positions)*.75))
        order=np.lexsort((positions,-priority[positions],-score[positions]))
        chosen[positions[order[:count]]]=True
    return chosen


def stability(pair,dc,q,c):
    aq,bp=canonical_q_order(q),canonical_C_order(c)
    sq,sc=q.iloc[aq].reset_index(drop=True),c.iloc[bp].reset_index(drop=True)
    pair,dc=pair[aq][:,bp],dc[aq][:,bp]
    qs,cs=sq.sex.to_numpy(),sc.sex.to_numpy()
    base=select(pair,dc,qs,cs,"sex_quota")
    frequencies=np.zeros(len(q))
    for repeat in range(SUBSAMPLES):
        rng=np.random.default_rng(SEED+1200+repeat)
        draws=[]
        for sex in np.unique(cs):
            population=np.flatnonzero(cs==sex)
            n=int(np.floor(len(population)*.8))
            assert n>=int(np.ceil((qs==sex).sum()*.75)),"80%抽出後のC行数がquotaを下回る"
            draws.append(rng.choice(population,n,False))
        draw=np.sort(np.concatenate(draws))
        chosen=select(pair[:,draw],dc[:,draw],qs,cs[draw],"sex_quota")
        frequencies+=chosen/SUBSAMPLES
    priority=pair.max(1)
    ranks=np.zeros(len(q))
    for sex in np.unique(qs):
        m=qs==sex
        ranks[m]=(rankdata(priority[m],method="average")-1)/max(m.sum()-1,1)
    selections=dict(baseline=base,frequency_quota75=top_sex(sq,frequencies,priority),
        frequency_pairrank_equal_quota75=top_sex(sq,.5*frequencies+.5*ranks,priority),
        frequency_ge075=frequencies>=.75)
    restore=np.argsort(aq)
    predictions={name:np.where(v,9.5,0)[restore] for name,v in selections.items()}
    return predictions,frequencies[restore],priority[restore]


def held_out(table,targets,group):
    records=[]
    for team in targets:
        train=table[table.team.isin([v for v in targets if v!=team])]
        averages=train.groupby("method").r_all.mean().reindex(METHODS)
        method=averages.idxmax()
        actual=table[table.team.eq(team)&table.method.eq(method)].iloc[0]
        records.append(dict(group=group,team=team,method=method,training_R=float(averages.loc[method]),
            **{k:float(actual[k]) for k in ["r_all","r_rare","r_max","p_m","p_c"]}))
    return records


def run():
    started=time.time()
    HERE.mkdir(parents=True,exist_ok=True);out=HERE/"集計";out.mkdir(exist_ok=True)
    planfile=HERE/"事前固定候補.json"
    if planfile.exists():assert json.loads(planfile.read_text())==PLAN
    else:dump(planfile,PLAN)
    prelim=ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack"
    main=ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
    items,null_items,diag,perms=[],[],[],[]
    for phase,folder,bgpath in [("prelim",prelim,ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
                                ("main",main,main/"A_bg.csv")]:
        bg=core.read(bgpath);scale=scale_for(bg)
        # 他C比較と同じ事前固定の独立pool/ref分割を使う。
        order=np.random.default_rng(SEED+41).permutation(len(bg))
        pool=bg.iloc[order[:len(bg)//2]].reset_index(drop=True)
        ref=bg.iloc[order[len(bg)//2:]].reset_index(drop=True)
        split_scale=scale_for(ref)
        draws=[np.sort(np.random.default_rng(SEED+800+r).choice(len(pool),1049,False)) for r in range(NULL_REPEATS)]
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1])):
            team=int(cp.stem.rsplit("_",1)[1])
            q=core.read(folder/f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c=core.read(cp)
            pair,dc=matrices(q,c,bg,scale)
            pred,freq,priority=stability(pair,dc,q,c)
            items.append(dict(phase=phase,team=team,q=q,pred=pred,frequency=freq,priority=priority))
            for name,v in pred.items():
                diag.append(dict(phase=phase,team=team,method=name,selected=int((v==9.5).sum()),
                    changed_vs_baseline=int((v!=pred["baseline"]).sum()),mean_frequency=float(freq.mean()),
                    frequency_min=float(freq.min()),frequency_max=float(freq.max()),
                    frequency_q25=float(np.quantile(freq,.25)),frequency_q75=float(np.quantile(freq,.75))))
            # 行順を変えて、32回の抽出と対応から全て再計算する。
            rng=np.random.default_rng(SEED+team+701)
            aq,bp=rng.permutation(len(q)),rng.permutation(len(c))
            changed,freq_changed,_=stability(pair[aq][:,bp],dc[aq][:,bp],q.iloc[aq].reset_index(drop=True),c.iloc[bp].reset_index(drop=True))
            restore=np.argsort(aq)
            for name,v in changed.items():
                perms.append(dict(phase=phase,team=team,method=name,changed=int((v[restore]!=pred[name]).sum()),
                    max_frequency_difference=float(np.max(np.abs(freq_changed[restore]-freq)))))
            if phase=="prelim" and team!=5:
                dbref=distance(q,ref,split_scale)
                p_real,d_real=matrices(q,c,ref,split_scale,dbref)
                real_pred,_,_=stability(p_real,d_real,q,c)
                null_items += [dict(team=team,source="real_C_split_reference",repeat=-1,method=name,prediction=v) for name,v in real_pred.items()]
                p_pool,d_pool=matrices(q,pool,ref,split_scale,dbref)
                for repeat,draw in enumerate(draws):
                    cb=pool.iloc[draw].reset_index(drop=True)
                    null_pred,_,_=stability(p_pool[:,draw],d_pool[:,draw],q,cb)
                    null_items += [dict(team=team,source="background_C",repeat=repeat,method=name,prediction=v) for name,v in null_pred.items()]
            print(json.dumps(dict(phase=phase,team=team,seconds=round(time.time()-started,1))),flush=True)
    # この時点までは公開truthのファイルを開かない。
    td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    truths,rows,nrows,frequency_diag={},[],[],[]
    for item in items:
        if item["phase"]!="prelim":continue
        team=item["team"]
        truth=core.read(td/f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        assert item["q"].challenge_row_id.tolist()==truth.challenge_row_id.tolist()
        truths[team]=truth
        for name,v in item["pred"].items():rows.append(dict(team=team,method=name,**core.scores(v,truth)))
        member=truth.label.eq("member").to_numpy()
        frequency_diag.append(dict(team=team,frequency_auc=float(roc_auc_score(member,item["frequency"])),
            mean_frequency_member=float(item["frequency"][member].mean()),mean_frequency_control=float(item["frequency"][~member].mean()),
            base_priority_auc=float(roc_auc_score(member,item["priority"])),frequency_is_membership_probability=False))
    for item in null_items:
        nrows.append(dict(team=item["team"],source=item["source"],repeat=item["repeat"],method=item["method"],
            **core.scores(item["prediction"],truths[item["team"]])))
    table=pd.DataFrame(rows);null=pd.DataFrame(nrows)
    table.to_csv(out/"予備戦_対象別.csv",index=False);null.to_csv(out/"独立背景対照_全反復.csv",index=False)
    averages,holdouts,null_means=[],[],[]
    metrics=["r_all","r_rare","r_max","p_m","p_c"]
    for group,excluded in [("20対象_自5除外",[5]),("18対象_1_2_5除外",[1,2,5])]:
        t=table[~table.team.isin(excluded)]
        s=t.groupby("method")[metrics].mean().reset_index();s.insert(0,"group",group);s["targets"]=t.team.nunique();averages.append(s)
        holdouts+=held_out(table,sorted(t.team.unique()),group)
        ns=null[~null.team.isin(excluded)].groupby(["source","method"])[metrics].mean().reset_index();ns.insert(0,"group",group);null_means.append(ns)
    pd.concat(averages).to_csv(out/"予備戦_20対象と18対象平均.csv",index=False)
    h=pd.DataFrame(holdouts);h.to_csv(out/"チーム留保選択_対象別.csv",index=False);h.groupby("group")[metrics].mean().to_csv(out/"チーム留保選択_平均.csv")
    pd.concat(null_means).to_csv(out/"独立背景対照_平均.csv",index=False)
    delta=[]
    for (team,name),d in null.groupby(["team","method"]):
        actual=d[d.source.eq("real_C_split_reference")].iloc[0];b=d[d.source.eq("background_C")]
        delta.append(dict(team=team,method=name,actual_R=actual.r_all,background_mean_R=b.r_all.mean(),
            excess_R=actual.r_all-b.r_all.mean(),background_R_q95=b.r_all.quantile(.95),
            excess_pm=actual.p_m-b.p_m.mean(),excess_pc=actual.p_c-b.p_c.mean()))
    pd.DataFrame(delta).to_csv(out/"独立背景対照_対象別.csv",index=False)
    pd.DataFrame(frequency_diag).to_csv(out/"予備戦_頻度と元スコアの診断.csv",index=False)
    pd.DataFrame(diag).to_csv(out/"予備戦と本戦_選別診断.csv",index=False)
    p=pd.DataFrame(perms);p.to_csv(out/"行順変更.csv",index=False)
    previous=core.read(OLD/"集計/同点修正_対象別.csv")
    oldbase=previous[previous.method.eq("stable_matching/sex_quota/long75")].sort_values("team")
    newbase=table[table.method.eq("baseline")].sort_values("team")
    assert oldbase.team.tolist()==newbase.team.tolist()
    assert np.max(np.abs(oldbase.r_all.to_numpy()-newbase.r_all.to_numpy()))<1e-12
    official=core.validate_scores(truths)
    refs=["分析.py","対応競合の頑健性.py","同点に依存しない対応.py","対応競合の分析.py"]
    dump(HERE/"実行記録.json",dict(plan=PLAN,seed=SEED,python=platform.python_version(),executable=sys.executable,
        seconds=time.time()-started,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_code_sha256={name:hashlib.sha256((OLD/name).read_bytes()).hexdigest() for name in refs},
        input_sha256=core.HASHES,baseline_matches_yesterday=True,official_score_check=official,
        permutation_cases=len(p),changed_rows=int(p.changed.sum()),max_frequency_difference=float(p.max_frequency_difference.max()),
        null_pool_n=2571,null_reference_n=2572,null_splits=1,frequency_is_membership_probability=False,
        B_self_read=False,main_truth_read=False,official_phi=False,submission_created=False,codabench_submitted=False))
    print(pd.concat(averages).to_string(index=False),flush=True)
    print(h.groupby("group")[metrics].mean().to_string(),flush=True)
    print(pd.concat(null_means).to_string(index=False),flush=True)
    print(p.groupby("method").changed.sum().to_string(),flush=True)


if __name__=="__main__":run()
