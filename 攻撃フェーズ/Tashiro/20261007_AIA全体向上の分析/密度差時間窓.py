"""条件付き時間分布だけでなくQI局所密度も含む時間窓の分析。

Cおよび公開背景へ同じ距離・帯域を適用。提出物は生成しない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import 分析 as core

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CENTERS=np.arange(0,10.001,.25)


def predict(q,c,bg):
    raw_bg=core.raw(bg)
    scale=np.maximum(np.quantile(raw_bg,.75,axis=0)-np.quantile(raw_bg,.25,axis=0),1e-5)
    x,y,z=(core.raw(v)/scale for v in (q,c,bg))
    parts_c=np.sort((x[:,None,:]-y[None,:,:])**2,axis=2)
    parts_b=np.sort((x[:,None,:]-z[None,:,:])**2,axis=2)
    mc=q.sex.to_numpy()[:,None]!=c.sex.to_numpy()[None,:]
    mb=q.sex.to_numpy()[:,None]!=bg.sex.to_numpy()[None,:]
    pc=q.prefecture.to_numpy()[:,None]!=c.prefecture.to_numpy()[None,:]
    pb=q.prefecture.to_numpy()[:,None]!=bg.prefecture.to_numpy()[None,:]
    tc=(np.abs(c.time.to_numpy()[:,None]-CENTERS[None,:])<=.5).astype(float)
    tb=(np.abs(bg.time.to_numpy()[:,None]-CENTERS[None,:])<=.5).astype(float)
    preds={}; diagnostics=[]
    for n in [2,4,5]:
        for cat in ["sex","both"]:
            dc,db=parts_c[:,:,:n].mean(2),parts_b[:,:,:n].mean(2)
            dc[mc]=np.inf;db[mb]=np.inf
            if cat=="both":
                dc[pc]=np.inf;db[pb]=np.inf
            for k in [1,5,15]:
                radius=np.partition(dc,k-1,axis=1)[:,k-1]
                first=dc.min(1)
                radius=np.where(np.isfinite(radius),radius,np.where(np.isfinite(first),first,1.))
                h=np.maximum(.35*radius,1e-8)
                wc=np.exp(-np.minimum(dc/h[:,None],700));wc[~np.isfinite(dc)]=0
                wb=np.exp(-np.minimum(db/h[:,None],700));wb[~np.isfinite(db)]=0
                # 質量を条件付きで別々に正規化せず、同じ帯域の強さを残す。
                cm=wc@tc/len(c);bm=wb@tb/len(bg)
                ratio=np.log((wc.sum(1)/len(c)+1e-12)/(wb.sum(1)/len(bg)+1e-12))
                for lam in [.5,.9,1.,1.25]:
                    name=f"joint_partial{n}_{cat}_k{k}_lambda{lam:g}"
                    preds[name]=core.later_best(CENTERS,cm-lam*bm)
                diagnostics.append(dict(n=n,cat=cat,k=k,ratio=ratio))
    return preds,diagnostics


def run():
    start=time.time();rows=[];diags=[];self_rows=[]
    td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    sets=[("prelim",ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack",ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
          ("main",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]
    b=core.read(next((ROOT/"本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    for phase,folder,bgpath in sets:
        bg=core.read(bgpath)
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1])):
            team=int(cp.stem.rsplit("_",1)[1]);q=core.read(folder/f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c=core.read(cp);preds,d=predict(q,c,bg)
            for name,pred in preds.items():
                diags.append(dict(phase=phase,team=team,method=name,zero=int((pred==0).sum()),long=int((pred>=9).sum()),short=int(((pred>0)&(pred<9)).sum())))
            if phase=="prelim":
                truth=core.read(td/f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
                for name,pred in preds.items():
                    rows.append(dict(team=team,method=name,**core.scores(pred,truth)))
            elif team==28:
                joined=q.merge(b[core.QI+["time"]],on=core.QI,how="left",validate="one_to_one")
                member=joined.time.notna().to_numpy();truth=joined.time.to_numpy();assert member.sum()==163
                for name,pred in preds.items():
                    self_rows.append(dict(method=name,p_m=float((np.abs(pred[member]-truth[member])<=.5).mean()),n_member=163,control_truth_known=False))
                for v in d:
                    self_rows.append(dict(method=f"density_auc_partial{v['n']}_{v['cat']}_k{v['k']}",selection_auc=float(roc_auc_score(member,v["ratio"])),n_member=163,control_truth_known=False))
            print(json.dumps(dict(phase=phase,team=team,seconds=round(time.time()-start,1))),flush=True)
    out=HERE/"集計";table=pd.DataFrame(rows)
    table.to_csv(out/"密度差時間窓_対象別.csv",index=False)
    table[~table.team.isin([1,2,5])].groupby("method")[["r_all","r_rare","r_max","p_m","p_c"]].mean().sort_values("r_all",ascending=False).to_csv(out/"密度差時間窓_加工群平均.csv")
    pd.DataFrame(diags).to_csv(out/"密度差時間窓_予測診断.csv",index=False)
    pd.DataFrame(self_rows).to_csv(out/"密度差時間窓_本戦自チーム.csv",index=False)
    (HERE/"密度差時間窓_実行記録.json").write_text(json.dumps(dict(seed=core.SEED,methods=72,seconds=time.time()-start,input_sha256=core.HASHES,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),submission_created=False),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(table[~table.team.isin([1,2,5])].groupby("method")[["r_all","p_m","p_c"]].mean().sort_values("r_all",ascending=False).head(12).to_string(),flush=True)


if __name__=="__main__":
    run()
