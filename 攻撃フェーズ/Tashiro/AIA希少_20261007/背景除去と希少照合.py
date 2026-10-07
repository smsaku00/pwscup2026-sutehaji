"""背景由来の候補除去と、推定した希少候補への別照合を比較する。

各チームの希少候補モデルはそのチームを含まない予備戦で学習。
希少真値は score() と外部評価だけが読み、照合には渡さない。
提出物は生成しない。非公開Bの行、真値、モデルを掲載用に保存しない。
"""
from __future__ import annotations
import argparse, hashlib, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import 分析 as core
import 希少候補の学習 as raremod


def late_best(mass):
    return core.CENTERS[len(core.CENTERS)-1-np.argmax(mass[:,::-1],axis=1)]


def gate(rank,fraction):
    a=np.zeros(len(rank),bool)
    a[np.argsort(-rank,kind="stable")[:int(np.ceil(len(rank)*fraction))]]=True
    return a


def background_origin(c,bg):
    b=core.raw(bg,core.FULL)
    scale=np.maximum(b.std(axis=0,ddof=1),1e-5)
    d=cdist(core.raw(c,core.FULL)/scale,b/scale,"sqeuclidean")/len(core.FULL)
    mismatch=(c.sex.to_numpy()[:,None]!=bg.sex.to_numpy()[None,:])|(c.prefecture.to_numpy()[:,None]!=bg.prefecture.to_numpy()[None,:])
    d[mismatch]=np.inf
    nearest=d.argmin(axis=1)
    rms=np.sqrt(d[np.arange(len(c)),nearest])
    time_agree=np.abs(c.time.to_numpy()-bg.time.to_numpy()[nearest])<=.5
    exact_columns=core.FULL+["sex","prefecture","smoking","time","onset","death"]
    bgkeys=set(bg[exact_columns].itertuples(index=False,name=None))
    exact=np.array([x in bgkeys for x in c[exact_columns].itertuples(index=False,name=None)])
    return dict(exact=exact,near02=(rms<=.02)&time_agree,near10=(rms<=.1)&time_agree),rms


def weights(distance, source_weight, k=5):
    effective=distance/np.maximum(source_weight[None,:],.02)
    effective[:,source_weight==0]=np.inf
    empty=~np.isfinite(effective).any(axis=1)
    effective[empty]=distance[empty]
    radius=np.partition(effective,k-1,axis=1)[:,k-1]
    minimum=effective.min(axis=1)
    radius=np.where(np.isfinite(radius),radius,minimum)
    h=np.maximum(.35*radius,1e-8)
    w=np.exp(-np.minimum(effective/h[:,None],700))
    w[~np.isfinite(effective)]=0
    return w


def predictions(q,c,bg,model):
    pc=model.predict_proba(raremod.features(c))[:,1]
    pb=model.predict_proba(raremod.features(bg))[:,1]
    pq=model.predict_proba(raremod.features(q))[:,1]
    marks,origin_rms=background_origin(c,bg)
    result={};diagnostics={"background_exact":int(marks["exact"].sum()),"background_near02":int(marks["near02"].sum()),
        "background_near10":int(marks["near10"].sum()),"c_rare_probability_ge05":int((pc>=.5).sum()),
        "q_rare_probability_ge05":int((pq>=.5).sum()),"background_distance_median":float(np.median(origin_rms[np.isfinite(origin_rms)]))}
    time_c=c.time.to_numpy();time_b=bg.time.to_numpy()
    tc=(np.abs(time_c[:,None]-core.CENTERS[None,:])<=.5).astype(float)
    tb=(np.abs(time_b[:,None]-core.CENTERS[None,:])<=.5).astype(float)
    rng=np.random.default_rng(core.SEED)
    samples=np.array([rng.choice(len(bg),1049,replace=False) for _ in range(16)])
    restrictions={"all":np.ones(len(c))}
    for name,mask in marks.items():
        keep=(~mask).astype(float)
        restrictions[f"drop_{name}"]=keep if keep.sum()>=40 else np.ones(len(c))
    configs=[(5,"soft"),(5,"sex"),(5,"both"),(4,"sex"),(2,"sex")]
    for n,cat in configs:
        dc=core.cost(q,c,bg,n,cat);db=core.cost(q,bg,bg,n,cat)
        prefix=f"partial{n}_{cat}"
        b1=np.median(np.column_stack([db[:,sample].min(axis=1) for sample in samples]),axis=1)
        for restriction,sw in restrictions.items():
            dr=dc.copy();dr[:,sw==0]=np.inf
            invalid=~np.isfinite(dr).any(axis=1);dr[invalid]=dc[invalid]
            nn=dr.argmin(axis=1);times=time_c[nn]
            rank=np.log((b1+1e-8)/(dr[np.arange(len(q)),nn]+1e-8))
            result[f"{prefix}/{restriction}/nn"]=times
            for fraction in [.25,.5,.75,.9]:
                selected=gate(rank,fraction)
                result[f"{prefix}/{restriction}/nn_gate{fraction:g}"]=np.where(selected,times,0.)
                result[f"{prefix}/{restriction}/nn_gate{fraction:g}_miss11"]=np.where(selected,times,11.)
            wc=weights(dr,np.ones(len(c)))
            wb=weights(db,np.ones(len(bg)),40)
            cm=wc@tc/np.maximum(wc.sum(axis=1,keepdims=True),1e-15)
            bm=wb@tb/np.maximum(wb.sum(axis=1,keepdims=True),1e-15)
            result[f"{prefix}/{restriction}/window5"]=late_best(cm-.25*bm)
        # 元の照合から推定希少候補だけを変更する。全ての行に同じ規則を適用する。
        base=result[f"{prefix}/all/nn_gate0.75"]
        base_window=result[f"{prefix}/all/window5"]
        for label,sw,bw in [
            ("probability",.05+pc,.05+pb),
            ("top24",raremod.top(pc,24).astype(float),raremod.top(pb,int(round(len(bg)*24/1049))).astype(float)),
            ("top48",raremod.top(pc,48).astype(float),raremod.top(pb,int(round(len(bg)*48/1049))).astype(float))]:
            effective=dc/np.maximum(sw[None,:],.02)
            effective[:,sw==0]=np.inf
            # 希少候補が性別/県条件にない場合は元の照合へ戻す。
            empty=~np.isfinite(effective).any(axis=1)
            effective[empty]=dc[empty]
            nn=effective.argmin(axis=1)
            rare_nn=time_c[nn]
            wc=weights(effective,np.ones(len(c)),min(5,int(max(5,(sw>0).sum()))))
            wb=weights(db,bw,40)
            cm=wc@tc/np.maximum(wc.sum(axis=1,keepdims=True),1e-15)
            bm=wb@tb/np.maximum(wb.sum(axis=1,keepdims=True),1e-15)
            rare_window=late_best(cm-.25*bm)
            for number in [26,52,82]:
                select=raremod.top(pq,number)
                result[f"{prefix}/rare_{label}_nn_top{number}"]=np.where(select,rare_nn,base)
                result[f"{prefix}/rare_{label}_window_top{number}"]=np.where(select,rare_window,base_window)
        # 全群モデルではなく希少候補について密度比の順位を取り直す。
        rare_weight=.02+pc;bg_weight=.02+pb
        wc=weights(dc,rare_weight);wb=weights(db,bg_weight)
        density=np.log((wc.sum(axis=1)/len(c)+1e-12)/(wb.sum(axis=1)/len(bg)+1e-12))
        times=time_c[(dc/rare_weight[None,:]).argmin(axis=1)]
        for f in [.25,.5,.75]:
            # 選別順位を希少候補内で作る。人数26/52/82は公開された構成からの推定。
            for number in [26,52]:
                selected=raremod.top(pq,number)
                ids=np.flatnonzero(selected)
                take=np.zeros(len(q),bool);take[ids[gate(density[ids],f)]]=True
                override=np.where(take,times,0.)
                result[f"{prefix}/rare_density_gate{f:g}_top{number}"]=np.where(selected,override,base)
                result[f"{prefix}/rare_density_gate{f:g}_top{number}_miss11"]=np.where(selected,np.where(take,times,11.),base)
    for p in result.values():
        assert np.isfinite(p).all() and (0<=p).all()
    reference=result["partial5_soft/all/nn_gate0.75"]
    for name,pred in list(result.items()):
        if "/rare_" in name:
            number=int(re.search(r"top(\d+)",name).group(1))
            result["S4keep/"+name]=np.where(raremod.top(pq,number),pred,reference)
    return result,diagnostics


def mix_with_baseline(preds,q,model,baseline,prefix):
    pq=model.predict_proba(raremod.features(q))[:,1]
    out={prefix+"/unchanged":baseline}
    for name,pred in preds.items():
        if "/rare_" in name and not name.startswith("S4keep/"):
            number=int(re.search(r"top(\d+)",name).group(1))
            out[prefix+"/"+name]=np.where(raremod.top(pq,number),pred,baseline)
    return out


def run(root):
    start=time.time();pre,bg=core.datasets(root,"prelim")
    X=np.vstack([raremod.features(v["q"]) for v in pre])
    Y=np.concatenate([v["truth"].is_rare.to_numpy(bool) for v in pre])
    groups=np.concatenate([np.full(326,v["team"]) for v in pre]);models={}
    folds=[]
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=7).split(X,groups=groups)):
        model=raremod.make_model().fit(X[tr],Y[tr])
        train=set(groups[tr]);test=set(groups[te]);assert not train&test
        for k in test:models[int(k)]=model
        folds.append(dict(fold=fold,train_teams=sorted(map(int,train)),test_teams=sorted(map(int,test))))
    previous=core.read(root/"予備戦/攻撃フェーズ/提出記録/8_20260907_GenLRA_Zhou/F_aia.csv")
    current=core.read(root/"本戦/攻撃フェーズ/Tashiro/20261006_第2回提出データ統合/入力/AIA/F_aia.csv")
    table=[];diag=[];mains=[];selfrows=[]
    for v in pre:
        preds,d=predictions(v["q"],v["c"],bg,models[v["team"]])
        if v["team"]!=5:
            base=previous[previous.target_k.eq(v["team"])].set_index("challenge_row_id").loc[v["q"].challenge_row_id,"time_hat"].to_numpy()
            preds.update(mix_with_baseline(preds,v["q"],models[v["team"]],base,"published8keep"))
        diag.append(dict(phase="prelim",team=v["team"],**d))
        for name,pred in preds.items():table.append(dict(team=v["team"],method=name,**core.score(pred,v["truth"])))
        print(dict(phase="prelim",team=v["team"],methods=len(preds),seconds=round(time.time()-start,1)),flush=True)
    summary=core.summaries(pd.DataFrame(table),"背景除去と希少照合_v4")
    model=raremod.make_model().fit(X,Y);main,mbg=core.datasets(root,"main")
    for v in main:
        preds,d=predictions(v["q"],v["c"],mbg,model)
        if v["team"]!=28:
            base=current[current.target_k.eq(v["team"])].set_index("challenge_row_id").loc[v["q"].challenge_row_id,"time_hat"].to_numpy()
            preds.update(mix_with_baseline(preds,v["q"],model,base,"current2keep"))
        diag.append(dict(phase="main",team=v["team"],**d))
        base=preds["partial5_soft/all/nn_gate0.75"]
        for name,pred in preds.items():mains.append(dict(team=v["team"],method=name,changed_from_base=int((np.abs(pred-base)>1e-10).sum()),zero=int((pred==0).sum()),long=int((pred>=9).sum())))
        if v["team"]==28:
            truth=core.self_truth(root,v["q"])
            for name,pred in preds.items():selfrows.append(dict(method=name,**core.self_score(pred,truth)))
        print(dict(phase="main",team=v["team"],seconds=round(time.time()-start,1)),flush=True)
    out=core.HERE/"集計"
    pd.DataFrame(diag).to_csv(out/"背景由来候補と希少候補の件数_v4.csv",index=False)
    pd.DataFrame(mains).to_csv(out/"背景除去と希少照合_本戦の動作_v4.csv",index=False)
    pd.DataFrame(selfrows).to_csv(out/"背景除去と希少照合_本戦自チーム標的のみ_v4.csv",index=False)
    core.record("背景除去と希少照合_v4",dict(seconds=time.time()-start,methods=len(preds),folds=folds,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),main_truth_used_to_train=False))
    print(summary[summary.group.eq("弱加工2対象を除く18対象")].head(10).to_string(index=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=core.ROOT)
    run(p.parse_args().root.resolve())
