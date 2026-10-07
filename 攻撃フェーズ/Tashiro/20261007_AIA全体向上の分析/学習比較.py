"""公開済み予備戦の実際の加工・正解を用いるAIA候補選択の研究。

チーム単位の7分割で、評価チームの正解を学習へ混ぜない。
本戦の標的・対照ラベルや正解時間は学習に使わない。
予測はメモリと作業キャッシュのみ。提出CSVは生成しない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

HERE=Path(__file__).resolve().parent
SEED=20261007
LAMBDAS=[.25,.5,.75,.9,1.,1.25]


def evaluate(pred, member, time_, rare):
    hit=np.abs(pred-time_)<=.5
    pm,pc=hit[member].mean(),hit[~member].mean()
    r=float(np.clip((pm-pc)/max(1-pc,1e-9),0,1))
    pmr,pcr=hit[member&rare].mean(),hit[(~member)&rare].mean()
    se=np.sqrt(pmr*(1-pmr)/13+pcr*(1-pcr)/13)
    rr=float(np.clip((pmr-pcr-1.64*se)/max(1-pcr,1e-9),0,1))
    return dict(r_all=r,r_rare=rr,r_max=max(r,rr),p_m=float(pm),p_c=float(pc))


def features(x,pred):
    n,a=pred.shape
    common=np.repeat(x,a,axis=0)
    times=pred.reshape(-1)
    action=np.tile(np.arange(a),n)
    identity=np.eye(a,dtype=np.float32)[action]
    return np.column_stack([common,times/10,(times>=9).astype(float),identity]).astype(np.float32)


def fit(x,y):
    model=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=96,max_depth=16,
                             max_features=.7,random_state=SEED,n_jobs=1)
    return model.fit(x,y)


def main():
    start=time.time();path=HERE/"作業用/比較特徴量.npz"; d=np.load(path)
    names=d["names"].tolist()
    chosen=[i for i,n in enumerate(names) if n in ("constant_0","constant_9.5") or n.endswith(("/nn_all","/window5_lam0","/window5_lam0.25"))]
    action_names=[names[i] for i in chosen]
    p=d["prelim_pred"][:,chosen]; x=d["prelim_x"]; groups=d["prelim_team"]
    member=d["prelim_member"]; truth=d["prelim_time"];rare=d["prelim_rare"]
    hit=(np.abs(p-truth[:,None])<=.5).astype(float)
    y=np.stack([hit*member[:,None],hit*(~member)[:,None]],axis=2).reshape(-1,2)
    xf=features(x,p)
    assert np.isfinite(xf).all() and len(xf)==len(y)
    a=len(chosen); probabilities=np.empty((len(x),a,2)); records=[]; folds=[]
    for fold,(train,test) in enumerate(GroupKFold(n_splits=7).split(x,groups=groups)):
        train_teams=set(groups[train]);test_teams=set(groups[test]);assert not train_teams&test_teams
        train_rows=(train[:,None]*a+np.arange(a)[None,:]).reshape(-1)
        test_rows=(test[:,None]*a+np.arange(a)[None,:]).reshape(-1)
        model=fit(xf[train_rows],y[train_rows])
        probabilities[test]=model.predict(xf[test_rows]).reshape(len(test),a,2)
        folds.append(dict(fold=fold,train_teams=sorted(map(int,train_teams)),test_teams=sorted(map(int,test_teams))))
        print(json.dumps({"fold":fold,"test_teams":sorted(map(int,test_teams)),"seconds":round(time.time()-start,1)}),flush=True)
    for lam in LAMBDAS:
        j=(probabilities[:,:,0]-lam*probabilities[:,:,1]).argmax(1)
        pred=p[np.arange(len(p)),j]
        for k in np.unique(groups):
            m=groups==k
            records.append(dict(team=int(k),method=f"public_truth_cv_lambda{lam:g}",**evaluate(pred[m],member[m],truth[m],rare[m]),zero=int((pred[m]==0).sum()),long=int((pred[m]>=9).sum())))
    out=HERE/"集計";out.mkdir(exist_ok=True)
    table=pd.DataFrame(records);table.to_csv(out/"公開正解学習_チームを分離した結果.csv",index=False)
    table[table.team.ne(5)].groupby("method")[["r_all","r_rare","r_max","p_m","p_c"]].mean().sort_values("r_all",ascending=False).to_csv(out/"公開正解学習_平均.csv")
    model=fit(xf,y)
    pm=d["main_pred"][:,chosen]; xm=d["main_x"];gm=d["main_team"]
    probs=model.predict(features(xm,pm)).reshape(len(xm),a,2)
    diag=[]
    research_pred=[]
    for lam in LAMBDAS:
        j=(probs[:,:,0]-lam*probs[:,:,1]).argmax(1)
        pred=pm[np.arange(len(pm)),j];research_pred.append(pred)
        for k in np.unique(gm):
            m=gm==k;diag.append(dict(team=int(k),lambda_=lam,n=int(m.sum()),zero=int((pred[m]==0).sum()),long=int((pred[m]>=9).sum()),short=int(((pred[m]>0)&(pred[m]<9)).sum()),selected_actions=json.dumps({action_names[int(v)]:int(n) for v,n in zip(*np.unique(j[m],return_counts=True))},ensure_ascii=False)))
    pd.DataFrame(diag).to_csv(out/"公開正解学習_本戦の動作診断.csv",index=False)
    np.savez_compressed(HERE/"作業用/公開正解学習.npz",main_team=gm,main_pred=np.column_stack(research_pred),lambdas=np.array(LAMBDAS),prelim_probabilities=probabilities)
    (HERE/"公開正解学習_実行記録.json").write_text(json.dumps(dict(seed=SEED,seconds=time.time()-start,actions=action_names,lambdas=LAMBDAS,folds=folds,input_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),main_truth_used_for_training=False,submission_created=False),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(table[table.team.ne(5)].groupby("method")[["r_all","r_rare","r_max","p_m","p_c"]].mean().to_string(),flush=True)


if __name__=="__main__":
    main()
