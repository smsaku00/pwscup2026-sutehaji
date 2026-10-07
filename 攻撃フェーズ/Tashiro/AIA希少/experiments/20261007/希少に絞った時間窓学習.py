"""希少の公開正解に絞り、標的への適中と対照への適中を別々に学習する。

外側7分割で評価チームの希少真値・time・標的対照ラベルを学習から除く。
推定希少候補以外は比較元の回答を保持。分析用配列のみで提出ファイルは作らない。
"""
from __future__ import annotations
import 分析 as core
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
import 希少候補の学習 as raremod
import 背景除去と希少照合 as study
import 比較特徴量キャッシュ as cachemod

LAMBDAS=[.5,.75,1.,1.25,1.5]
LEAVES=[32,96]


def fit(x,y,leaf):
    return ExtraTreesRegressor(n_estimators=128,min_samples_leaf=leaf,max_depth=16,
        max_features=.7,random_state=core.SEED,n_jobs=1).fit(x,y)


def action_features(x,p):
    n,a=p.shape
    value=p.reshape(-1)
    return np.column_stack([np.repeat(x,a,axis=0),value/10,(value>=9).astype(float),
        np.eye(a,dtype=np.float32)[np.tile(np.arange(a),n)]]).astype(np.float32)


def chosen_prediction(p,probability,lam):
    j=(probability[:,:,0]-lam*probability[:,:,1]).argmax(axis=1)
    return p[np.arange(len(p)),j]


def run(root):
    start=time.time()
    d,cache=cachemod.load(root);names=d["names"].tolist()
    choices=[i for i,n in enumerate(names) if n in ["constant_0","constant_9.5"] or n.endswith(("/nn_all","/window5_lam0","/window5_lam0.25"))]
    actions=[names[i] for i in choices]+["constant_11"]
    p=np.column_stack([d["prelim_pred"][:,choices],np.full(len(d["prelim_team"]),11.)]);x=d["prelim_x"]
    g=d["prelim_team"];r=d["prelim_rare"].astype(bool);m=d["prelim_member"].astype(bool);t=d["prelim_time"]
    a=len(actions);xf=action_features(x,p)
    hit=(np.abs(p-t[:,None])<=.5).astype(float)
    labels=np.stack([hit*m[:,None],hit*(~m)[:,None]],axis=2).reshape(-1,2)
    pre,bg=core.datasets(root,"prelim")
    assert np.array_equal(g,np.concatenate([np.full(326,v["team"]) for v in pre]))
    qx=np.vstack([raremod.features(v["q"]) for v in pre])
    base=d["prelim_pred"][:,names.index("S4_log_iqr_soft/nn_gate0.75_0")]
    previous=core.read(root/"予備戦/攻撃フェーズ/提出記録/8_20260907_GenLRA_Zhou/F_aia.csv")
    oldbase=base.copy()
    for i,v in enumerate(pre):
        if v["team"]!=5:
            oldbase[i*326:(i+1)*326]=previous[previous.target_k.eq(v["team"])].set_index("challenge_row_id").loc[v["q"].challenge_row_id,"time_hat"].to_numpy()
    rows=[];folds=[];prob_by_leaf={};rareprob=np.zeros(len(g))
    for leaf in LEAVES:
        probability=np.empty((len(g),a,2))
        for fold,(train,test) in enumerate(GroupKFold(n_splits=7).split(x,groups=g)):
            rare_train=train[r[train]]
            assert not set(g[train])&set(g[test])
            tr=(rare_train[:,None]*a+np.arange(a)[None,:]).reshape(-1)
            te=(test[:,None]*a+np.arange(a)[None,:]).reshape(-1)
            model=fit(xf[tr],labels[tr],leaf)
            probability[test]=model.predict(xf[te]).reshape(len(test),a,2)
            if leaf==LEAVES[0]:
                rp=raremod.make_model().fit(qx[train],r[train])
                rareprob[test]=rp.predict_proba(qx[test])[:,1]
                folds.append(dict(fold=fold,train_teams=sorted(map(int,np.unique(g[train]))),test_teams=sorted(map(int,np.unique(g[test]))),rare_training_persons=len(rare_train)))
            print(dict(leaf=leaf,fold=fold,seconds=round(time.time()-start,1)),flush=True)
        prob_by_leaf[leaf]=probability
        for lam in LAMBDAS:
            rawpred=chosen_prediction(p,probability,lam)
            for number in [26,52,82]:
                for prefix,b in [("S4keep",base),("published8keep",oldbase)]:
                    for i,v in enumerate(pre):
                        if prefix=="published8keep" and v["team"]==5:continue
                        sl=slice(i*326,(i+1)*326)
                        pred=np.where(raremod.top(rareprob[sl],number),rawpred[sl],b[sl])
                        rows.append(dict(team=v["team"],method=f"{prefix}/rare_action_leaf{leaf}_lambda{lam:g}_top{number}",**core.score(pred,v["truth"])))
    summary=core.summaries(pd.DataFrame(rows),"希少時間窓学習")
    main,mbg=core.datasets(root,"main")
    gm=d["main_team"];pm=np.column_stack([d["main_pred"][:,choices],np.full(len(gm),11.)]);xm=d["main_x"];fm=action_features(xm,pm)
    assert np.array_equal(gm,np.concatenate([np.full(326,v["team"]) for v in main]))
    rmodel=raremod.make_model().fit(qx,r)
    mq=np.vstack([raremod.features(v["q"]) for v in main])
    mrareprob=rmodel.predict_proba(mq)[:,1]
    current=core.read(root/"本戦/攻撃フェーズ/Tashiro/20261006_第2回提出データ統合/入力/AIA/F_aia.csv")
    mb=d["main_pred"][:,names.index("S4_log_iqr_soft/nn_gate0.75_0")].copy()
    for i,v in enumerate(main):
        if v["team"]!=28:
            mb[i*326:(i+1)*326]=current[current.target_k.eq(v["team"])].set_index("challenge_row_id").loc[v["q"].challenge_row_id,"time_hat"].to_numpy()
    diag=[];selfrows=[];allpred=[];methodnames=[]
    for leaf in LEAVES:
        tr=(np.flatnonzero(r)[:,None]*a+np.arange(a)[None,:]).reshape(-1)
        model=fit(xf[tr],labels[tr],leaf)
        prob=model.predict(fm).reshape(len(gm),a,2)
        for lam in LAMBDAS:
            rawpred=chosen_prediction(pm,prob,lam)
            for number in [26,52,82]:
                final=mb.copy();name=f"rare_action_leaf{leaf}_lambda{lam:g}_top{number}"
                for i,v in enumerate(main):
                    sl=slice(i*326,(i+1)*326)
                    pred=np.where(raremod.top(mrareprob[sl],number),rawpred[sl],mb[sl]);final[sl]=pred
                    diag.append(dict(team=v["team"],method=name,changed=int((np.abs(pred-mb[sl])>1e-10).sum()),zero=int((pred==0).sum()),long=int((pred>=9).sum())))
                    if v["team"]==28:selfrows.append(dict(method=name,**core.self_score(pred,core.self_truth(root,v["q"]))))
                methodnames.append(name);allpred.append(final)
        print(dict(main_leaf=leaf,seconds=round(time.time()-start,1)),flush=True)
    out=core.HERE/"集計"
    pd.DataFrame(diag).to_csv(out/"希少時間窓学習_本戦の動作.csv",index=False)
    pd.DataFrame(selfrows).to_csv(out/"希少時間窓学習_本戦自チーム標的のみ.csv",index=False)
    np.savez_compressed(core.HERE/"作業用/希少時間窓学習.npz",main_team=gm,main_pred=np.column_stack(allpred),method_names=np.array(methodnames),prelim_rareprob=rareprob,**{f"prelim_prob_leaf{leaf}":p for leaf,p in prob_by_leaf.items()})
    core.record("希少時間窓学習",dict(seconds=time.time()-start,folds=folds,actions=actions,leaves=LEAVES,lambdas=LAMBDAS,
        input_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),source_inputs_rechecked=True,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),main_truth_used_to_train=False))
    print(summary[summary.group.eq("弱加工2対象を除く18対象")].head(15).to_string(index=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=core.ROOT)
    run(p.parse_args().root.resolve())
