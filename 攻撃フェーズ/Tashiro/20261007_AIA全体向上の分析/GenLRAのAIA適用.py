"""既存のMIA用Gen-LRAを、AIA課題の公開QIと時間探索に適用する分析。

元のGen-LRAコード・提出物は変更しない。本戦真値を学習しない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
import 分析 as core

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=HERE.parent/"20261005_MIA初回_GenLRA比較/gen_lra.py"
spec=importlib.util.spec_from_file_location("frozen_genlra_analysis",SOURCE)
gen=importlib.util.module_from_spec(spec);spec.loader.exec_module(gen)
CENTERS=np.arange(.5,10,.9999999999999999)
CENTERS=np.arange(.5,10,1.)


def predict(q,c,bg):
    rank={"mixed7":gen.mixed_scores(bg,q,c,gen.NUM7),
          "ordinal7h5":gen.ordinal_scores(bg,q,c,gen.QI7,5),
          "ordinal7h7":gen.ordinal_scores(bg,q,c,gen.QI7,7)}
    dc=core.cost(q,c,bg,"S4_log_iqr_soft");nn=c.time.to_numpy()[dc.argmin(1)]
    pred={}
    for name,score in rank.items():
        for frac in [.25,.5,.75,.9]:
            selected=core.keep(score,frac)
            for base,value in [("NN",nn),("NNlong95",np.where(nn>=9.5,9.5,nn)),("long95",np.full(len(q),9.5))]:
                pred[f"GenLRA/{name}/{base}/gate{frac}"]=np.where(selected,value,0)
    scanned=pd.concat([q.assign(time=t) for t in CENTERS],ignore_index=True)
    raw=gen.ordinal_scores(bg,scanned,c,gen.QI7+("time",),8).reshape(len(CENTERS),len(q)).T
    # 非重複の1年区間から背景の時間確率を推定し、窓確率を正規化。
    db=core.cost(q,bg,bg,"log_iqr_sex")
    pb=core.window_masses(db,bg.time.to_numpy(),80,CENTERS)
    pb=np.maximum(pb,1e-8);pb/=pb.sum(1,keepdims=True)
    pure=CENTERS[raw.argmax(1)]
    with_prior=CENTERS[(raw+np.log(pb)).argmax(1)]
    pred["GenLRA/scan8/pure"]=pure
    pred["GenLRA/scan8/with_BG_prior"]=with_prior
    posterior=softmax(raw+np.log(pb),axis=1)
    for lam in [.5,.9,1.,1.25]:
        pred[f"GenLRA/scan8/posterior_minus_BG{lam:g}"]=CENTERS[(posterior-lam*pb).argmax(1)]
    for frac in [.5,.75]:
        selected=core.keep(rank["mixed7"],frac)
        pred[f"GenLRA/scan8/pure_gate{frac}"]=np.where(selected,pure,0)
        pred[f"GenLRA/scan8/prior_gate{frac}"]=np.where(selected,with_prior,0)
    assert all(np.isfinite(v).all() for v in pred.values())
    return pred,rank


def run():
    start=time.time();td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"));rows=[];diags=[];aucs=[];ownrows=[]
    sets=[("prelim",ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack",ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
          ("main",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]
    own=core.read(next((ROOT/"本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    for phase,folder,bgpath in sets:
        bg=core.read(bgpath)
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1])):
            k=int(cp.stem.rsplit("_",1)[1]);q=core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True);c=core.read(cp)
            preds,rank=predict(q,c,bg)
            for name,pred in preds.items():diags.append(dict(phase=phase,team=k,method=name,zero=int((pred==0).sum()),long=int((pred>=9).sum()),short=int(((pred>0)&(pred<9)).sum())))
            if phase=="prelim":
                truth=core.read(td/f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
                for name,pred in preds.items():rows.append(dict(team=k,method=name,**core.scores(pred,truth)))
                for name,s in rank.items():aucs.append(dict(team=k,method=name,selection_auc=roc_auc_score(truth.label.eq("member"),s)))
            elif k==28:
                j=q.merge(own[core.QI+["time"]],on=core.QI,how="left",validate="one_to_one");m=j.time.notna().to_numpy();assert m.sum()==163
                for name,s in rank.items():ownrows.append(dict(method=name,selection_auc=float(roc_auc_score(m,s)),control_truth_known=False))
                for name,pred in preds.items():ownrows.append(dict(method=name,p_m=float((np.abs(pred[m]-j.time.to_numpy()[m])<=.5).mean()),control_truth_known=False))
            print(json.dumps(dict(phase=phase,team=k,seconds=round(time.time()-start,1))),flush=True)
    out=HERE/"集計";table=pd.DataFrame(rows);table.to_csv(out/"GenLRA_AIA_対象別.csv",index=False)
    table[~table.team.isin([1,2,5])].groupby("method")[["r_all","r_rare","r_max","p_m","p_c"]].mean().sort_values("r_all",ascending=False).to_csv(out/"GenLRA_AIA_加工群平均.csv")
    pd.DataFrame(diags).to_csv(out/"GenLRA_AIA_予測診断.csv",index=False);pd.DataFrame(aucs).to_csv(out/"GenLRA_AIA_標的選別.csv",index=False);pd.DataFrame(ownrows).to_csv(out/"GenLRA_AIA_本戦自チーム.csv",index=False)
    (HERE/"GenLRA_AIA_実行記録.json").write_text(json.dumps(dict(seed=core.SEED,methods=len(preds),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_genlra_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),input_sha256=core.HASHES,seconds=time.time()-start,main_truth_used_for_training=False,submission_created=False),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(table[~table.team.isin([1,2,5])].groupby("method")[["r_all","p_m","p_c"]].mean().sort_values("r_all",ascending=False).head(15).to_string(),flush=True)


if __name__=="__main__":
    run()
