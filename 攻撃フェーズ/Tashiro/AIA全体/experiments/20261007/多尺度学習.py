"""Cと全公開背景の局所密度を多尺度・部分属性で比較する分析。

提出物は作らない。予備戦の公開正解によるチーム外評価と本戦診断のみ。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import 分析 as core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
KS = [1, 3, 8, 20]
FRACTIONS = [.25, .5, .75, .9]


def expanded(q, c, bg):
    b = core.raw(bg)
    med = np.median(b, axis=0)
    scale = np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0),1e-5)
    x, y, z = ((core.raw(v)-med)/scale for v in (q,c,bg))
    # 列を個別に最小化せず、同じ候補行の差が小さい2〜5属性を使う。
    parts_c = np.sort((x[:,None,:]-y[None,:,:])**2,axis=2)
    parts_b = np.sort((x[:,None,:]-z[None,:,:])**2,axis=2)
    sexc = q.sex.to_numpy()[:,None] != c.sex.to_numpy()[None,:]
    sexb = q.sex.to_numpy()[:,None] != bg.sex.to_numpy()[None,:]
    prefc = q.prefecture.to_numpy()[:,None] != c.prefecture.to_numpy()[None,:]
    prefb = q.prefecture.to_numpy()[:,None] != bg.prefecture.to_numpy()[None,:]
    f, names, ranks = [], [], {}
    for n in [2,3,4,5]:
        base_c, base_b = parts_c[:,:,:n].mean(2), parts_b[:,:,:n].mean(2)
        for cat in ["soft", "sex", "both"]:
            dc, db = base_c.copy(), base_b.copy()
            if cat == "soft":
                dc += sexc + prefc.astype(float)
                db += sexb + prefb.astype(float)
            else:
                dc[sexc] = np.inf; db[sexb] = np.inf
                if cat == "both":
                    dc[prefc] = np.inf; db[prefb] = np.inf
            name = f"partial{n}_{cat}"
            near = np.partition(dc,[k-1 for k in KS],axis=1)
            for k in KS:
                radius = near[:,k-1]
                inside = (db<=radius[:,None]).sum(1)
                # 同じ探索・部分属性の選択を背景にも適用した経験的確率。
                density = np.log((k+.5)/(len(c)+1))-np.log((inside+.5)/(len(bg)+1))
                density[~np.isfinite(radius)] = -8
                f.extend([np.log1p(np.minimum(radius,1e4)),density])
                names.extend([name+f"_log_radius{k}",name+f"_density{k}"])
                ranks[name+f"_k{k}"] = density
    f.extend(x.T); names.extend(core.NUM)
    f.append(q.sex.to_numpy(float)); names.append("sex")
    prefectures = ["東京都","大阪府","神奈川県","愛知県","福岡県","北海道","宮崎県"]
    for p in prefectures:
        f.append(q.prefecture.eq(p).to_numpy(float)); names.append("pref_"+p)
    # Cの形を知る特徴。評価ラベルは使用しない。
    profile = [c.time.ge(9.5).mean(),c.time.std()/10,c.sex.eq(1).mean()]
    profile += [c[col].nunique()/len(c) for col in core.NUM]
    profile += list(np.std(y,axis=0))
    for i,value in enumerate(profile):
        f.append(np.full(len(q),value)); names.append(f"C_profile_{i}")
    xf = np.column_stack(f).astype(np.float32)
    assert np.isfinite(xf).all()
    return xf,names,ranks


def evaluate_prediction(pred,member,truth,rare):
    from 学習比較 import evaluate
    return evaluate(pred,member,truth,rare)


def run():
    start=time.time(); old=np.load(HERE/"作業用/比較特徴量.npz")
    tasks=[("prelim",ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack",ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
           ("main",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力",ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]
    caches={};unsupervised=[]; td=next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    for phase,folder,bgpath in tasks:
        bg=core.read(bgpath);xlist=[];glist=[];keylist=[]
        for cp in sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1])):
            k=int(cp.stem.rsplit("_",1)[1]);q=core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c=core.read(cp);xf,fn,ranks=expanded(q,c,bg)
            xlist.append(xf);glist.append(np.full(len(q),k));keylist.append(pd.util.hash_pandas_object(q[core.QI],index=False).to_numpy())
            if phase=="prelim":
                truth=core.read(td/f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
                for metric,rank in ranks.items():
                    for frac in [.5,.75]:
                        pred=np.where(core.keep(rank,frac),9.5,0)
                        unsupervised.append(dict(team=k,method=f"density/{metric}/long_gate{frac}",selection_auc=roc_auc_score(truth.label.eq("member"),rank),**core.scores(pred,truth)))
            print(json.dumps(dict(phase=phase,team=k,seconds=round(time.time()-start,1))),flush=True)
        caches[phase]=dict(x=np.vstack(xlist),team=np.concatenate(glist),person_key=np.concatenate(keylist))
        assert np.array_equal(caches[phase]["team"],old[phase+"_team"])
    out=HERE/"集計";pd.DataFrame(unsupervised).to_csv(out/"多尺度密度_対象別.csv",index=False)
    member=old["prelim_member"];truth=old["prelim_time"];rare=old["prelim_rare"]
    groups=caches["prelim"]["team"];x=np.column_stack([caches["prelim"]["x"],old["prelim_x"]]);keys=caches["prelim"]["person_key"]
    prob=np.empty(len(x));folds=[]
    for fold,(train,test) in enumerate(GroupKFold(7).split(x,groups=groups)):
        # 同一QIの人が別チームにいる場合は、その正解も学習から外す。
        overlap=np.isin(keys[train],keys[test]);train=train[~overlap]
        model=ExtraTreesClassifier(n_estimators=128,min_samples_leaf=24,max_depth=14,max_features=.7,n_jobs=1,random_state=core.SEED)
        model.fit(x[train],member[train]);prob[test]=model.predict_proba(x[test])[:,1]
        folds.append(dict(fold=fold,train_teams=sorted(set(map(int,groups[train]))),test_teams=sorted(set(map(int,groups[test]))),excluded_shared_persons=int(overlap.sum())))
        print(json.dumps(dict(fold=fold,seconds=round(time.time()-start,1))),flush=True)
    records=[];names=old["names"].tolist();base_names=["constant_9.5","S4_log_iqr_soft/nn_all","log_iqr_sex/window5_lam0.25","partial4_log_iqr_both/nn_all"]
    for k in np.unique(groups):
        m=groups==k
        for frac in FRACTIONS:
            selected=core.keep(prob[m],frac)
            for name in base_names:
                base=old["prelim_pred"][m,names.index(name)]
                pred=np.where(selected,base,0)
                records.append(dict(team=int(k),method=f"learned_membership/{name}/gate{frac}",selection_auc=roc_auc_score(member[m],prob[m]),**evaluate_prediction(pred,member[m],truth[m],rare[m])))
    table=pd.DataFrame(records);table.to_csv(out/"多尺度選別学習_対象別.csv",index=False)
    table[~table.team.isin([1,2,5])].groupby("method")[["r_all","r_rare","r_max","p_m","p_c","selection_auc"]].mean().sort_values("r_all",ascending=False).to_csv(out/"多尺度選別学習_加工群平均.csv")
    model.fit(x,member);xm=np.column_stack([caches["main"]["x"],old["main_x"]]);pm=model.predict_proba(xm)[:,1];gm=caches["main"]["team"]
    # 自チームのラベルは最終的な診断だけに使う。
    b=core.read(next((ROOT/"本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    q=core.read(tasks[1][1]/"aia_challenge_28.csv").sort_values("challenge_row_id")
    own=q.merge(b[core.QI+["time"]],on=core.QI,how="left",validate="one_to_one").time.notna().to_numpy()
    assert own.sum()==163
    rec=dict(seed=core.SEED,features=fn,folds=folds,seconds=time.time()-start,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=core.HASHES,main28_auc=float(roc_auc_score(own,pm[gm==28])),main_truth_used_for_training=False,submission_created=False)
    (HERE/"多尺度学習_実行記録.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    np.savez_compressed(HERE/"作業用/多尺度学習.npz",prelim_x=x,prelim_probability=prob,main_probability=pm,main_team=gm)
    print(table[~table.team.isin([1,2,5])].groupby("method")[["r_all","p_m","p_c","selection_auc"]].mean().sort_values("r_all",ascending=False).head(10).to_string(),flush=True)
    print("main28_auc",rec["main28_auc"],flush=True)


if __name__=="__main__":
    run()
