"""元の行の最近傍以外の、Cと背景の分布識別によるAIA選別を比較。

識別器の学習ラベルはC/背景の出所のみ。本戦の真値は学習へ渡さない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
from scipy.special import logsumexp
import 分析 as core

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SEED=core.SEED


def logpdf(x,mean,cov):
    sign,ld=np.linalg.slogdet(cov);assert sign>0
    v=x-mean;return -.5*(np.einsum("ni,ij,nj->n",v,np.linalg.inv(cov),v)+ld)


def distribution_scores(q,c,bg):
    raw_b=core.raw(bg);med=np.median(raw_b,0);sd=np.maximum(raw_b.std(0),1e-5)
    x,y,z=((core.raw(v)-med)/sd for v in (q,c,bg))
    result={f"{scope}/{name}":np.empty(len(q)) for scope in ["sex","county"] for name in ["linear_gaussian","quadratic_gaussian","logistic_quad_C0.1","logistic_quad_C1","HGB"]}
    for scope in ["sex","county"]:
        for sex in sorted(q.sex.unique()):
            target=q.sex.eq(sex).to_numpy();cm=c.sex.eq(sex).to_numpy();bm=bg.sex.eq(sex).to_numpy()
            if scope=="county":
                pref=q.loc[target,"prefecture"].unique();assert len(pref)==1
                ca=cm&c.prefecture.eq(pref[0]).to_numpy();ba=bm&bg.prefecture.eq(pref[0]).to_numpy()
                if ca.sum()>=25 and ba.sum()>=25:
                    cm,bm=ca,ba
            assert cm.sum()>=5 and bm.sum()>=5
            a,b=y[cm],z[bm];xx=x[target]
            cb=LedoitWolf().fit(b).covariance_+np.eye(5)*1e-4
            ca=LedoitWolf().fit(a).covariance_
            # 小さなCブロックの共分散を背景側へ20%縮める。
            ca=.8*ca+.2*cb+np.eye(5)*1e-4
            result[f"{scope}/linear_gaussian"][target]=logpdf(xx,a.mean(0),cb)-logpdf(xx,b.mean(0),cb)
            result[f"{scope}/quadratic_gaussian"][target]=logpdf(xx,a.mean(0),ca)-logpdf(xx,b.mean(0),cb)
            train=np.vstack([a,b]);labels=np.r_[np.ones(len(a)),np.zeros(len(b))]
            poly=PolynomialFeatures(2,include_bias=False);trainp=poly.fit_transform(np.clip(train,-6,6));testp=poly.transform(np.clip(xx,-6,6))
            for strength in [.1,1.]:
                model=LogisticRegression(C=strength,class_weight="balanced",max_iter=500,random_state=SEED)
                model.fit(trainp,labels);result[f"{scope}/logistic_quad_C{strength:g}"][target]=model.decision_function(testp)
            weight=np.where(labels==1,.5/len(a),.5/len(b))*len(train)
            model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=8,min_samples_leaf=max(10,min(30,len(a)//4)),l2_regularization=1.,learning_rate=.05,early_stopping=False,random_state=SEED)
            model.fit(train,labels,sample_weight=weight)
            result[f"{scope}/HGB"][target]=model.predict_proba(xx)[:,1]
    assert all(np.isfinite(v).all() for v in result.values())
    return result


def group_keep(score,q,fraction):
    # 対照は性別を1対1で合わせている。性別内で同じ割合を選ぶ比較。
    out=np.zeros(len(q),bool)
    for sex in q.sex.unique():
        m=q.sex.eq(sex).to_numpy();out[m]=core.keep(score[m],fraction)
    return out


def run():
    start=time.time();td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"));records=[];diags=[];self_rows=[]
    sets=[("prelim",ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack",ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
          ("main",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]
    own_b=core.read(next((ROOT/"本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    caches=[]
    for phase,folder,bgpath in sets:
        bg=core.read(bgpath)
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1])):
            k=int(cp.stem.rsplit("_",1)[1]);q=core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c=core.read(cp);sc=distribution_scores(q,c,bg)
            dc=core.cost(q,c,bg,"log_iqr_sex");nn=c.time.to_numpy()[dc.argmin(1)]
            preds={}
            for name,s in sc.items():
                for fraction in [.5,.75,.9]:
                    for grouping in ["pooled","sex"]:
                        chosen=core.keep(s,fraction) if grouping=="pooled" else group_keep(s,q,fraction)
                        preds[f"{name}/{grouping}/long{fraction}"]=np.where(chosen,9.5,0)
                        if fraction==.5:
                            preds[f"{name}/{grouping}/NN{fraction}"]=np.where(chosen,nn,0)
            for name,pred in preds.items():
                diags.append(dict(phase=phase,team=k,method=name,zero=int((pred==0).sum()),long=int((pred>=9).sum())))
            if phase=="prelim":
                truth=core.read(td/f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
                for name,pred in preds.items():
                    records.append(dict(team=k,method=name,**core.scores(pred,truth)))
                caches.append(dict(team=k,score=np.column_stack(list(sc.values()))))
            elif k==28:
                joined=q.merge(own_b[core.QI+["time"]],on=core.QI,how="left",validate="one_to_one");m=joined.time.notna().to_numpy();assert m.sum()==163
                for name,s in sc.items():
                    self_rows.append(dict(method=name,selection_auc=float(roc_auc_score(m,s))))
                for name,pred in preds.items():
                    self_rows.append(dict(method=name,p_m=float((np.abs(pred[m]-joined.time.to_numpy()[m])<=.5).mean())))
            print(json.dumps(dict(phase=phase,team=k,seconds=round(time.time()-start,1))),flush=True)
    out=HERE/"集計";table=pd.DataFrame(records);table.to_csv(out/"分布識別_対象別.csv",index=False)
    table[~table.team.isin([1,2,5])].groupby("method")[["r_all","r_rare","r_max","p_m","p_c"]].mean().sort_values("r_all",ascending=False).to_csv(out/"分布識別_加工群平均.csv")
    pd.DataFrame(diags).to_csv(out/"分布識別_予測診断.csv",index=False);pd.DataFrame(self_rows).to_csv(out/"分布識別_自チーム診断.csv",index=False)
    np.savez_compressed(HERE/"作業用/分布識別.npz",prelim_team=np.concatenate([np.full(len(v["score"]),v["team"]) for v in caches]),prelim_score=np.vstack([v["score"] for v in caches]),score_names=np.array(list(sc)))
    (HERE/"分布識別_実行記録.json").write_text(json.dumps(dict(seed=SEED,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=core.HASHES,seconds=time.time()-start,main_truth_used_for_training=False,submission_created=False),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(table[~table.team.isin([1,2,5])].groupby("method")[["r_all","p_m","p_c"]].mean().sort_values("r_all",ascending=False).head(15).to_string(),flush=True)


if __name__=="__main__":
    run()
