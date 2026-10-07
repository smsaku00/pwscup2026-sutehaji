"""QIと候補時間のC対背景密度比を調べるAIA分析。

学習ラベルは公開C/背景の出所。本戦の正解は学習に使わず、提出物も作らない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score
import 分析 as core

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CENTERS=np.arange(0,10.001,.25)


def encode(x,t):
    # 9.5〜10の観察終了値の微小な書式差を出所の手掛かりにしない。
    t=np.where(t>=9.5,9.5,t)
    return np.column_stack([x,t/10,(t>=9.5).astype(float)])


def reverse_search(q,c,bg):
    b=core.raw(bg);med=np.median(b,0);scale=np.maximum(b.std(0),1e-5)
    x,y,z=((core.raw(v)-med)/scale for v in (q,c,bg))
    result={};ratios={}
    for scope in ["sex","county"]:
        modelscores={name:np.empty((len(q),len(CENTERS))) for name in ["logistic","HGB","ET"]}
        for sex in q.sex.unique():
            mask=q.sex.eq(sex).to_numpy();cm=c.sex.eq(sex).to_numpy();bm=bg.sex.eq(sex).to_numpy()
            if scope=="county":
                pref=q.loc[mask,"prefecture"].unique();assert len(pref)==1
                ca=cm&c.prefecture.eq(pref[0]).to_numpy();ba=bm&bg.prefecture.eq(pref[0]).to_numpy()
                if ca.sum()>=25 and ba.sum()>=25:cm,bm=ca,ba
            a=encode(y[cm],c.time.to_numpy()[cm]);btrain=encode(z[bm],bg.time.to_numpy()[bm])
            train=np.vstack([a,btrain]);label=np.r_[np.ones(len(a)),np.zeros(len(btrain))]
            weights=np.where(label==1,.5/len(a),.5/len(btrain))*len(train)
            xx=np.repeat(x[mask],len(CENTERS),axis=0);tt=np.tile(CENTERS,mask.sum());test=encode(xx,tt)
            poly=PolynomialFeatures(2,include_bias=False);trainp=poly.fit_transform(np.clip(train,-6,6));testp=poly.transform(np.clip(test,-6,6))
            for name in modelscores:
                if name=="logistic":
                    model=LogisticRegression(C=.1,max_iter=600,random_state=core.SEED)
                    model.fit(trainp,label,sample_weight=weights);p=model.predict_proba(testp)[:,1]
                elif name=="HGB":
                    model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=8,min_samples_leaf=20,l2_regularization=1.,learning_rate=.05,early_stopping=False,random_state=core.SEED)
                    model.fit(train,label,sample_weight=weights);p=model.predict_proba(test)[:,1]
                else:
                    model=ExtraTreesClassifier(n_estimators=80,min_samples_leaf=24,max_depth=12,max_features=.8,n_jobs=1,random_state=core.SEED)
                    model.fit(train,label,sample_weight=weights);p=model.predict_proba(test)[:,1]
                p=np.clip(p,.025,.975)
                ratio=p/(1-p)
                modelscores[name][mask]=ratio.reshape(mask.sum(),len(CENTERS))
        metric="log_iqr_sex" if scope=="sex" else "log_iqr_both"
        db=core.cost(q,bg,bg,metric);pb=core.window_masses(db,bg.time.to_numpy(),80,CENTERS)
        for name,r in modelscores.items():
            # fn_C(q,t)/fn_BG(q,t)を背景の窓的中確率へ掛け、共同質量の差を比べる。
            for lam in [.5,.9,1.,1.25]:
                result[f"reverse/{scope}/{name}/lambda{lam:g}"]=core.later_best(CENTERS,pb*(r-lam))
            ratios[scope+"/"+name]=r
    return result,ratios


def run():
    start=time.time();td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"));rows=[];diags=[];ownrows=[]
    sets=[("prelim",ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack",ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
          ("main",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]
    own=core.read(next((ROOT/"本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    for phase,folder,bgpath in sets:
        bg=core.read(bgpath)
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1])):
            k=int(cp.stem.rsplit("_",1)[1]);q=core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c=core.read(cp);preds,_=reverse_search(q,c,bg)
            for name,pred in preds.items():
                diags.append(dict(phase=phase,team=k,method=name,zero=int((pred==0).sum()),long=int((pred>=9).sum()),short=int(((pred>0)&(pred<9)).sum())))
            if phase=="prelim":
                truth=core.read(td/f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
                for name,pred in preds.items():rows.append(dict(team=k,method=name,**core.scores(pred,truth)))
            elif k==28:
                j=q.merge(own[core.QI+["time"]],on=core.QI,how="left",validate="one_to_one");m=j.time.notna().to_numpy();assert m.sum()==163
                for name,pred in preds.items():ownrows.append(dict(method=name,p_m=float((np.abs(pred[m]-j.time.to_numpy()[m])<=.5).mean()),control_truth_known=False))
            print(json.dumps(dict(phase=phase,team=k,seconds=round(time.time()-start,1))),flush=True)
    out=HERE/"集計";table=pd.DataFrame(rows);table.to_csv(out/"時間逆探索_対象別.csv",index=False)
    table[~table.team.isin([1,2,5])].groupby("method")[["r_all","r_rare","r_max","p_m","p_c"]].mean().sort_values("r_all",ascending=False).to_csv(out/"時間逆探索_加工群平均.csv")
    pd.DataFrame(diags).to_csv(out/"時間逆探索_予測診断.csv",index=False);pd.DataFrame(ownrows).to_csv(out/"時間逆探索_本戦自チーム.csv",index=False)
    (HERE/"時間逆探索_実行記録.json").write_text(json.dumps(dict(seed=core.SEED,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=core.HASHES,seconds=time.time()-start,main_truth_used_for_training=False,submission_created=False),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(table[~table.team.isin([1,2,5])].groupby("method")[["r_all","p_m","p_c"]].mean().sort_values("r_all",ascending=False).head(15).to_string(),flush=True)


if __name__=="__main__":
    run()
