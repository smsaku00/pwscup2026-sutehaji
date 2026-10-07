"""公開された予備戦のQIから希少らしさを学習し、未学習チームで評価する。

本戦の未知ラベルは使わない。本戦B28の真値は外部評価にだけ使う。
人数26の選別は希少の真値ではなく、公開QIからの予測である。
"""
from __future__ import annotations
import argparse, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import 分析 as core


def features(df):
    return np.column_stack([core.raw(df), df.sex.to_numpy(float)])


def make_model(kind="hgb"):
    if kind=="logistic":
        return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=300))
    return HistGradientBoostingClassifier(max_iter=140,max_leaf_nodes=15,min_samples_leaf=35,
                                         l2_regularization=5,random_state=core.SEED)


def top(probability, n):
    a=np.zeros(len(probability),bool)
    a[np.argsort(-probability,kind="stable")[:n]]=True
    return a


def profile(prob, rare):
    out=dict(auc=float(roc_auc_score(rare,prob)),average_precision=float(average_precision_score(rare,prob)),
             n=len(rare),n_rare=int(rare.sum()))
    for n in [26,52,82]:
        selected=top(prob,n)
        out[f"top{n}_rare_hits"]=int(rare[selected].sum())
        out[f"top{n}_precision"]=float(rare[selected].mean())
        out[f"top{n}_recall"]=float(rare[selected].sum()/rare.sum())
    return out


def run(root):
    start=time.time()
    cases,bg=core.datasets(root,"prelim")
    frames=[v["q"].assign(team=v["team"]) for v in cases]
    q=pd.concat(frames,ignore_index=True)
    rare=np.concatenate([v["truth"].is_rare.to_numpy(bool) for v in cases])
    x=features(q);groups=q.team.to_numpy()
    output=[];folds=[];oof={}
    for kind in ["logistic","hgb"]:
        p=np.zeros(len(q))
        for i,(tr,te) in enumerate(GroupKFold(n_splits=7).split(x,groups=groups)):
            model=make_model(kind).fit(x[tr],rare[tr])
            p[te]=model.predict_proba(x[te])[:,1]
            if kind=="hgb":
                folds.append(dict(fold=i,train_teams=sorted(map(int,np.unique(groups[tr]))),test_teams=sorted(map(int,np.unique(groups[te])))))
        for k in np.unique(groups):
            m=groups==k
            output.append(dict(team=int(k),model=kind,**profile(p[m],rare[m])))
        oof[kind]=p
    out=core.HERE/"集計";out.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(output)
    table.to_csv(out/"希少候補_QIのみのチーム分離評価.csv",index=False)
    table[table.team.ne(5)].groupby("model").mean(numeric_only=True).drop(columns="team").to_csv(out/"希少候補_QIのみの平均.csv")
    model=make_model().fit(x,rare)
    main,mbg=core.datasets(root,"main")
    mainprob=[];diag=[]
    for v in main:
        prob=model.predict_proba(features(v["q"]))[:,1]
        cprob=model.predict_proba(features(v["c"]))[:,1]
        mainprob.append(prob)
        diag.append(dict(team=v["team"],mean_rare_probability=float(prob.mean()),
                         top26_probability_min=float(np.sort(prob)[-26]),q_probability_ge_05=int((prob>=.5).sum()),
                         c_probability_ge_05=int((cprob>=.5).sum()),unknown_rare_truth=True))
    pd.DataFrame(diag).to_csv(out/"希少候補_本戦の動作診断.csv",index=False)
    b=core.read(next((root/"本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    bx=features(b);br=b.is_rare.to_numpy(bool)
    pb=model.predict_proba(bx)[:,1]
    external=profile(pb,br)
    # 外部評価のfeature permutation。学習や本戦の推定のためには使わない。
    rng=np.random.default_rng(core.SEED)
    perms=[]
    for j,name in enumerate(core.NUM+["sex"]):
        perm=bx.copy();perm[:,j]=rng.permutation(perm[:,j])
        pp=model.predict_proba(perm)[:,1]
        perms.append(dict(feature=name,external_auc_after_permutation=float(roc_auc_score(br,pp)),
                          auc_drop=external["auc"]-float(roc_auc_score(br,pp))))
    pd.DataFrame(perms).to_csv(out/"希少候補_本戦自チームでの項目依存.csv",index=False)
    cache=core.HERE/"作業用";cache.mkdir(exist_ok=True)
    # 主催者の本戦正解、自己Bの行、自己Bの真値やモデルを保存しない。
    np.savez_compressed(cache/"希少候補確率.npz",prelim_prob=oof["hgb"],prelim_team=groups,
                        main_prob=np.concatenate(mainprob),main_team=np.concatenate([np.full(326,v["team"]) for v in main]))
    core.record("希少候補",dict(seconds=time.time()-start,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        folds=folds,prelim_total_auc=float(roc_auc_score(rare,oof["hgb"])),prelim_total_ap=float(average_precision_score(rare,oof["hgb"])),
        external_B28=external,external_used_to_train=False,private_B_or_model_uploaded=False))
    print(table[table.team.ne(5)].groupby("model")[["auc","average_precision","top26_recall"]].mean().to_string(),flush=True)
    print("B28 external",external,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=core.ROOT)
    run(p.parse_args().root.resolve())
