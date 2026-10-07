"""希少モデルと攻撃設定の両方から外側評価チームを除いたnested CV。

外側7分割、内側6分割。内側の希少平均だけで設定を選び、外側で採点。
公開済み予備戦での確認であり、本戦の正解を使った性能保証ではない。
"""
from __future__ import annotations
import 分析 as core
import argparse, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import 希少候補の学習 as raremod
import 背景除去と希少照合 as study


def fit_rare(cases,ids):
    x=np.vstack([raremod.features(cases[i]["q"]) for i in ids])
    y=np.concatenate([cases[i]["truth"].is_rare.to_numpy(bool) for i in ids])
    return raremod.make_model().fit(x,y)


def compute(case,bg,model,previous):
    preds,_=study.predictions(case["q"],case["c"],bg,model)
    preds["S4keep/unchanged"]=preds["partial5_soft/all/nn_gate0.75"]
    if case["team"]!=5:
        base=previous[previous.target_k.eq(case["team"])].set_index("challenge_row_id").loc[case["q"].challenge_row_id,"time_hat"].to_numpy()
        preds.update(study.mix_with_baseline(preds,case["q"],model,base,"published8keep"))
    return {name:p for name,p in preds.items() if name.startswith(("S4keep/","published8keep/"))}


def run(root):
    start=time.time();cases,bg=core.datasets(root,"prelim")
    ids=np.arange(len(cases));groups=np.array([v["team"] for v in cases])
    previous=core.read(root/"予備戦/攻撃フェーズ/提出記録/8_20260907_GenLRA_Zhou/F_aia.csv")
    rows=[];selections=[];folds=[];hits=[]
    for outer,(train,test) in enumerate(GroupKFold(n_splits=7).split(ids,groups=groups)):
        train_teams=set(groups[train]);test_teams=set(groups[test]);assert not train_teams&test_teams
        qtrain=set(x for i in train for x in cases[i]["q"][core.QI].itertuples(index=False,name=None))
        qtest=set(x for i in test for x in cases[i]["q"][core.QI].itertuples(index=False,name=None))
        assert not qtrain&qtest,"個人が分割をまたいでいる"
        innerrows=[]
        for inner,(tr,te) in enumerate(GroupKFold(n_splits=6).split(train,groups=groups[train])):
            trids,teids=train[tr],train[te]
            assert not set(groups[trids])&test_teams and not set(groups[trids])&set(groups[teids])
            model=fit_rare(cases,trids)
            for i in teids:
                v=cases[i]
                if v["team"] in [1,2,5]:continue
                for name,p in compute(v,bg,model,previous).items():
                    innerrows.append(dict(team=v["team"],method=name,**core.score(p,v["truth"])))
            print(dict(outer=outer,inner=inner,seconds=round(time.time()-start,1)),flush=True)
        inner=pd.DataFrame(innerrows)
        selected={}
        for family in ["S4keep","published8keep"]:
            rank=inner[inner.method.str.startswith(family+"/")].groupby("method")[["r_rare","r_all"]].mean().reset_index()
            rank=rank.sort_values(["r_rare","r_all","method"],ascending=[False,False,True])
            selected[family]=rank.iloc[0].method
            selections.append(dict(outer_fold=outer,family=family,selected_method=rank.iloc[0].method,
                inner_rare=float(rank.iloc[0].r_rare),inner_all=float(rank.iloc[0].r_all),
                train_teams=sorted(map(int,train_teams)),test_teams=sorted(map(int,test_teams))))
        model=fit_rare(cases,train)
        for i in test:
            v=cases[i]
            if v["team"] in [1,2,5]:continue
            preds=compute(v,bg,model,previous)
            for family,name in selected.items():
                p=preds[name];base=preds[family+"/unchanged"]
                a=core.score(p,v["truth"]);b=core.score(base,v["truth"])
                rows.append(dict(team=v["team"],outer_fold=outer,family=family,method=name,**a,
                    baseline_r_rare=b["r_rare"],baseline_r_all=b["r_all"],rare_gain=a["r_rare"]-b["r_rare"],all_gain=a["r_all"]-b["r_all"]))
                r=v["truth"].is_rare.to_numpy(bool)
                hits.append(dict(team=v["team"],family=family,hit=(np.abs(p[r]-v["truth"].time.to_numpy()[r])<=.5),
                                 member=v["truth"].label.eq("member").to_numpy()[r],onset=v["truth"].onset.to_numpy()[r],
                                 sex=v["q"].sex.to_numpy()[r],age_band=(v["q"].age.to_numpy()[r]//10).astype(int)))
        folds.append(dict(outer_fold=outer,train_teams=sorted(map(int,train_teams)),test_teams=sorted(map(int,test_teams)),shared_persons_across_fold=0))
    out=core.HERE/"集計";table=pd.DataFrame(rows)
    table.to_csv(out/"二重チーム分離_評価対象別.csv",index=False)
    table.groupby("family")[["r_rare","r_all","baseline_r_rare","baseline_r_all","rare_gain","all_gain"]].mean().to_csv(out/"二重チーム分離_平均.csv")
    core.save_json(core.HERE/"二重チーム分離_設定選択.json",selections)
    # 二重CVの結果を対象単位で再標本化する。CIも本戦性能保証ではない。
    rng=np.random.default_rng(core.SEED);ci=[]
    for fam,g in table.groupby("family"):
        index=rng.integers(0,len(g),size=(50000,len(g)))
        for key in ["r_rare","rare_gain","r_all","all_gain"]:
            sim=g[key].to_numpy()[index].mean(axis=1)
            ci.append(dict(family=fam,metric=key,mean=float(g[key].mean()),bootstrap_low=float(np.quantile(sim,.025)),bootstrap_high=float(np.quantile(sim,.975)),targets=len(g)))
    pd.DataFrame(ci).to_csv(out/"二重チーム分離_対象単位の区間.csv",index=False)
    np.savez_compressed(core.HERE/"作業用/二重CV_希少正誤.npz",team=np.array([v["team"] for v in hits]),family=np.array([v["family"] for v in hits]),
        hit=np.stack([v["hit"] for v in hits]),member=np.stack([v["member"] for v in hits]),onset=np.stack([v["onset"] for v in hits]),sex=np.stack([v["sex"] for v in hits]),age_band=np.stack([v["age_band"] for v in hits]))
    core.record("二重チーム分離",dict(seconds=time.time()-start,outer_folds=folds,inner_folds=6,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        predictor_code_sha256=hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
        heldout_team_labels_used_for_any_training_or_choice=False,main_truth_used=False))
    print(table.groupby("family")[["r_rare","r_all","baseline_r_rare","baseline_r_all","rare_gain","all_gain"]].mean().to_string(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=core.ROOT)
    run(p.parse_args().root.resolve())
