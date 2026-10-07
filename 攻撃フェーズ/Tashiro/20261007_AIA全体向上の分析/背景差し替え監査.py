"""実Cと、標的人を含まない公開背景標本を使ったAIAの比較。

これは本戦の非公開phiの再現ではなく、新候補に対する別条件の対照実験。
背景を標本用・密度参照用に分割し、同じ行の使い回しによる自己比較を避ける。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import 分析 as core

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
N_REPEAT=100


def distances(q,c,bg,kind):
    if kind=="partial2":
        b=core.raw(bg);scale=np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0),1e-5)
        x,y=core.raw(q)/scale,core.raw(c)/scale
        d=np.sort((x[:,None,:]-y[None,:,:])**2,axis=2)[:,:,:2].mean(2)
        d[q.sex.to_numpy()[:,None]!=c.sex.to_numpy()[None,:]]=np.inf
        return d
    return core.cost(q,c,bg,kind)


def predict(q,c,bg_full,ref,ref_sample,dc,db):
    # 一つの距離条件で個人を完全一致と判定するような処理は行わない。
    first=dc["raw_sd_both"].argmin(1)
    official_style=c.time.to_numpy()[first]
    s4=dc["S4_log_iqr_soft"]
    first=s4.argmin(1);nearest=s4[np.arange(len(q)),first]
    ratio=np.log((db["S4_sample"].min(1)+1e-8)/(nearest+1e-8))
    s4pred=np.where(core.keep(ratio,.75),c.time.to_numpy()[first],0)
    radius=np.partition(dc["partial2"],2,axis=1)[:,2]
    count=(db["partial2"]<=radius[:,None]).sum(1)
    score=np.log(3.5/(len(c)+1))-np.log((count+.5)/(len(ref)+1))
    score[~np.isfinite(radius)]=-8
    return {"reference_NN_style":official_style,"S4_style75":s4pred,
            "partial2_sex_k3_long50":np.where(core.keep(score,.5),9.5,0),
            "partial2_sex_k3_long75":np.where(core.keep(score,.75),9.5,0)}


def run():
    start=time.time();folder=ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack"
    bg=core.read(ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
    order=np.random.default_rng(core.SEED).permutation(len(bg));cut=len(bg)//2
    pool=bg.iloc[order[:cut]].reset_index(drop=True);ref=bg.iloc[order[cut:]].reset_index(drop=True)
    ref_idx=np.sort(np.random.default_rng(core.SEED+1).choice(len(ref),1049,False))
    ref_sample=ref.iloc[ref_idx].reset_index(drop=True)
    td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"));rows=[]
    samples=[np.sort(np.random.default_rng(core.SEED+100+r).choice(len(pool),1049,False)) for r in range(N_REPEAT)]
    for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1])):
        k=int(cp.stem.rsplit("_",1)[1]);q=core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        c=core.read(cp);truth=core.read(td/f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        kinds=["raw_sd_both","S4_log_iqr_soft","partial2"]
        dpool={kind:distances(q,pool,bg,kind) for kind in kinds}
        dreal={kind:distances(q,c,bg,kind) for kind in kinds}
        db={"S4_sample":distances(q,ref_sample,bg,"S4_log_iqr_soft"),"partial2":distances(q,ref,bg,"partial2")}
        real=predict(q,c,bg,ref,ref_sample,dreal,db)
        for method,pred in real.items():
            rows.append(dict(team=k,method=method,source="real_C",repeat=-1,**core.scores(pred,truth)))
        for r,idx in enumerate(samples):
            cb=pool.iloc[idx].reset_index(drop=True)
            d={kind:value[:,idx] for kind,value in dpool.items()}
            for method,pred in predict(q,cb,bg,ref,ref_sample,d,db).items():
                rows.append(dict(team=k,method=method,source="background_C",repeat=r,**core.scores(pred,truth)))
        print(json.dumps(dict(team=k,seconds=round(time.time()-start,1))),flush=True)
    table=pd.DataFrame(rows);out=HERE/"集計";table.to_csv(out/"背景差し替え_全反復.csv",index=False)
    result=[]
    for (team,method),g in table.groupby(["team","method"]):
        real=g[g.source.eq("real_C")].iloc[0];null=g[g.source.eq("background_C")].r_all.to_numpy()
        result.append(dict(team=team,method=method,real_R=real.r_all,background_mean=null.mean(),background_q95=np.quantile(null,.95),excess_vs_background_mean=real.r_all-null.mean(),real_greater_fraction=float((real.r_all>null).mean()),background_p_m=g[g.source.eq("background_C")].p_m.mean(),background_p_c=g[g.source.eq("background_C")].p_c.mean()))
    result=pd.DataFrame(result);result.to_csv(out/"背景差し替え_対象別比較.csv",index=False)
    result[~result.team.isin([1,2,5])].groupby("method").mean(numeric_only=True).to_csv(out/"背景差し替え_加工群平均.csv")
    (HERE/"背景差し替え_実行記録.json").write_text(json.dumps(dict(seed=core.SEED,repeats=N_REPEAT,pool_n=len(pool),reference_n=len(ref),samples_overlap_reference=False,is_official_phi=False,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=core.HASHES,seconds=time.time()-start,submission_created=False),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(result[~result.team.isin([1,2,5])].groupby("method")[["real_R","background_mean","excess_vs_background_mean"]].mean().to_string(),flush=True)


if __name__=="__main__":
    run()
