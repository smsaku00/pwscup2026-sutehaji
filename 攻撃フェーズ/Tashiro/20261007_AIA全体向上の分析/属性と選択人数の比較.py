"""固定部分対応候補の年齢除外、選択人数、長期回答値を比較する。

設定探索の結果を性能保証としない。本戦は公開入力の診断のみ。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from 対応競合の分析 import partial_assignment
from 対応競合の頑健性 import pair_score
from 同点に依存しない対応 import EPSILON
import 分析 as core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
FRACTIONS = [.5, .65, .75, .85, .9]
REPEATS = 20


def matrices(q, c, ref, scale, include_age):
    use = list(range(5)) if include_age else [1,2,3,4]
    x, y, z = [(core.raw(v)/scale)[:,use] for v in [q,c,ref]]
    dc = np.sort((x[:,None]-y[None])**2, axis=2)[:,:,:2].mean(2)
    db = np.sort((x[:,None]-z[None])**2, axis=2)[:,:,:2].mean(2)
    dc[q.sex.to_numpy()[:,None]!=c.sex.to_numpy()[None]] = np.inf
    db[q.sex.to_numpy()[:,None]!=ref.sex.to_numpy()[None]] = np.inf
    pair = pair_score(dc, db)["continuous"]
    pair -= EPSILON*np.minimum(cdist(x,y,"sqeuclidean")/len(use), 1000.)
    return pair, dc


def chosen(pair, dc, qs, cs, fraction):
    limit = np.partition(dc,11,axis=1)[:,11]
    valid = np.isfinite(dc) & (dc<=limit[:,None])
    limited = np.where(valid,pair,-1e4)
    result = np.zeros(len(qs),bool)
    for sex in np.unique(qs):
        a,b=qs==sex,cs==sex
        result[a]=partial_assignment(limited[a][:,b],int(np.ceil(a.sum()*fraction)))[0]
    return result


def evaluate(pair, dc, q, c, truth, attrs):
    result=[]
    for fraction in FRACTIONS:
        selected=chosen(pair,dc,q.sex.to_numpy(),c.sex.to_numpy(),fraction)
        for t in [9.5,10.]:
            result.append(dict(method=f"ablation/{attrs}/sex_quota/gate{fraction}/long{t:g}",
                **core.scores(np.where(selected,t,0),truth)))
    return result


def run():
    start=time.time();folder=ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack"
    bg=core.read(ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
    raw=core.raw(bg);scale=np.maximum(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0),1e-5)
    order=np.random.default_rng(core.SEED+201).permutation(len(bg));cut=len(bg)//3
    pool,ref=[bg.iloc[idx].reset_index(drop=True) for idx in [order[:cut],order[cut:]]]
    samples=[np.sort(np.random.default_rng(core.SEED+600+r).choice(len(pool),1049,False)) for r in range(REPEATS)]
    td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"));records=[]
    for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1])):
        k=int(cp.stem.rsplit("_",1)[1])
        q=core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        c=core.read(cp);truth=core.read(td/f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        for include_age in [True,False]:
            attrs="age_and_clinical5" if include_age else "clinical4_without_age"
            pair,dc=matrices(q,c,bg,scale,include_age)
            records += [dict(team=k,source="real_C_full_reference",repeat=-1,**v) for v in evaluate(pair,dc,q,c,truth,attrs)]
            if k not in [1,2,5]:
                pair,dc=matrices(q,c,ref,scale,include_age)
                records += [dict(team=k,source="real_C_split_reference",repeat=-1,**v) for v in evaluate(pair,dc,q,c,truth,attrs)]
                pp,dp=matrices(q,pool,ref,scale,include_age)
                for r,idx in enumerate(samples):
                    cb=pool.iloc[idx].reset_index(drop=True)
                    records += [dict(team=k,source="background_C",repeat=r,**v) for v in evaluate(pp[:,idx],dp[:,idx],q,cb,truth,attrs)]
        print(json.dumps(dict(team=k,seconds=round(time.time()-start,1))),flush=True)
    out=HERE/"集計";table=pd.DataFrame(records)
    table.to_csv(out/"属性と選択人数_反復.csv",index=False)
    table[~table.team.isin([1,2,5])].groupby(["source","method"])[["r_all","r_rare","r_max","p_m","p_c"]].mean().to_csv(out/"属性と選択人数_加工群平均.csv")
    actual=table[table.source.eq("real_C_full_reference")&~table.team.isin([1,2,5])]
    pivot=actual.pivot(index="team",columns="method",values="r_all");loto=[]
    for k in pivot.index:
        train=pivot.drop(k).mean();name=train.idxmax()
        loto.append(dict(team=k,selected_method=name,r_all=pivot.loc[k,name],train_mean=train[name]))
    pd.DataFrame(loto).to_csv(out/"属性と選択人数_相手を外した設定選択.csv",index=False)
    (HERE/"属性と選択人数_実行記録.json").write_text(json.dumps(dict(seed=core.SEED,fractions=FRACTIONS,repeats=REPEATS,
        pool_rows=len(pool),reference_rows=len(ref),C_rows=1049,is_official_phi=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=core.HASHES,
        seconds=time.time()-start,main_truth_used=False,submission_created=False),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(actual.groupby("method").r_all.mean().sort_values(ascending=False).to_string(),flush=True)
    print("LOTO",float(pd.DataFrame(loto).r_all.mean()),flush=True)


if __name__=="__main__":
    run()
