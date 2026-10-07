"""同じC行を複数候補が使う衝突を、相互NN/一対一割当で比較する。

希少真値は照合に使わず、外側と内側で分離した公開QI分類器から候補を作る。
割当には11年を表すダミー列を加え、全候補を強制的に実在行へ結びつけない。
予備戦公開正解の分析のみ。新しい提出ファイルは生成しない。
"""
from __future__ import annotations
import 分析 as core
import argparse, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold
import 希少候補の学習 as raremod

CONFIGS=[(5,'soft'),(5,'sex'),(4,'sex')]


def distances(v,bg):
    out={};rng=np.random.default_rng(core.SEED)
    samples=np.array([rng.choice(len(bg),1049,False) for _ in range(16)])
    for n,cat in CONFIGS:
        dc=core.cost(v['q'],v['c'],bg,n,cat);db=core.cost(v['q'],bg,bg,n,cat)
        ref=np.median(np.column_stack([db[:,i].min(1) for i in samples]),axis=1)
        ref=np.where(np.isfinite(ref),ref,db.min(1))
        out[f'partial{n}_{cat}']=np.log(dc+1e-8)-np.log(ref[:,None]+1e-8)
    return out


def predictions(v,dist,model,base):
    pq=model.predict_proba(raremod.features(v['q']))[:,1];pc=model.predict_proba(raremod.features(v['c']))[:,1]
    out={'unchanged':base}
    for metric,cost in dist.items():
        for source in ['all','top48']:
            cols=np.arange(len(pc)) if source=='all' else np.flatnonzero(raremod.top(pc,48))
            d=cost[:,cols].copy()
            # 同性の候補を除去して空になる場合に限り全Cへ戻す。
            if not np.isfinite(d).any(1).all():cols=np.arange(len(pc));d=cost.copy()
            for number in [26,52,82]:
                selected=raremod.top(pq,number);ids=np.flatnonzero(selected);small=d[ids]
                for threshold in [0.,-.5]:
                    nn=small.argmin(1);best=small[np.arange(len(ids)),nn]
                    for algorithm in ['nn','mutual','assignment']:
                        if algorithm=='assignment':
                            augmented=np.column_stack([small,np.full((len(ids),len(ids)),threshold)])
                            row,col=linear_sum_assignment(augmented)
                            assert np.array_equal(row,np.arange(len(ids)))
                            accept=col<len(cols);match=np.minimum(col,len(cols)-1)
                            accept&=small[np.arange(len(ids)),match]<threshold
                        else:
                            match=nn;accept=best<threshold
                            if algorithm=='mutual':accept&=(d.argmin(0)[nn]==ids)
                        override=np.where(accept,v['c'].time.to_numpy()[cols[match]],11.)
                        pred=base.copy();pred[ids]=override
                        name=f'{metric}/{source}/{algorithm}/top{number}_threshold{threshold:g}'
                        out[name]=pred
    return out


def fit_rare(cases,ids):
    return raremod.make_model().fit(np.vstack([raremod.features(cases[i]['q']) for i in ids]),np.concatenate([cases[i]['truth'].is_rare.to_numpy(bool) for i in ids]))


def run(root):
    start=time.time();cases,bg=core.datasets(root,'prelim');dist=[distances(v,bg) for v in cases]
    previous=core.read(root/'予備戦/攻撃フェーズ/提出記録/8_20260907_GenLRA_Zhou/F_aia.csv');bases=[]
    for v in cases:
        bases.append(None if v['team']==5 else previous[previous.target_k.eq(v['team'])].set_index('challenge_row_id').loc[v['q'].challenge_row_id,'time_hat'].to_numpy())
    ids=np.arange(len(cases));groups=np.array([v['team'] for v in cases]);rows=[];settings=[];explore=[]
    for outer,(train,test) in enumerate(GroupKFold(n_splits=7).split(ids,groups=groups)):
        innerrows=[]
        for tr,te in GroupKFold(n_splits=6).split(train,groups=groups[train]):
            trids,teids=train[tr],train[te];assert not set(groups[trids])&set(groups[test])
            model=fit_rare(cases,trids)
            for i in teids:
                v=cases[i]
                if v['team'] in [1,2,5]:continue
                for name,p in predictions(v,dist[i],model,bases[i]).items():innerrows.append(dict(team=v['team'],method=name,**core.score(p,v['truth'])))
        mean=pd.DataFrame(innerrows).groupby('method')[['r_rare','r_all']].mean().reset_index();chosen={}
        for objective in ['rare','rare_guard_all']:
            a=mean.copy()
            if objective=='rare_guard_all':a=a[a.r_all>=float(a[a.method.eq('unchanged')].r_all.iloc[0])-1e-12]
            a=a.sort_values(['r_rare','r_all','method'],ascending=[False,False,True]);chosen[objective]=a.iloc[0].method
            settings.append(dict(outer=outer,objective=objective,method=chosen[objective],inner_rare=float(a.iloc[0].r_rare),inner_all=float(a.iloc[0].r_all),train_teams=sorted(map(int,groups[train])),test_teams=sorted(map(int,groups[test]))))
        model=fit_rare(cases,train)
        for i in test:
            v=cases[i]
            if v['team'] in [1,2,5]:continue
            preds=predictions(v,dist[i],model,bases[i])
            for name,p in preds.items():explore.append(dict(team=v['team'],method=name,**core.score(p,v['truth'])))
            b=core.score(bases[i],v['truth'])
            for objective,name in chosen.items():
                a=core.score(preds[name],v['truth']);rows.append(dict(team=v['team'],outer=outer,objective=objective,method=name,**a,
                                baseline_rare=b['r_rare'],baseline_all=b['r_all'],rare_gain=a['r_rare']-b['r_rare'],all_gain=a['r_all']-b['r_all']))
        print(dict(outer=outer,seconds=round(time.time()-start,1)),flush=True)
    out=core.HERE/'集計';table=pd.DataFrame(rows);ex=pd.DataFrame(explore)
    table.to_csv(out/'一対一照合_二重CV_対象別.csv',index=False)
    mean=table.groupby('objective')[['r_rare','r_all','baseline_rare','baseline_all','rare_gain','all_gain']].mean();mean.to_csv(out/'一対一照合_二重CV_平均.csv')
    ex.to_csv(out/'一対一照合_探索対象別.csv',index=False)
    em=ex.groupby('method')[['r_rare','r_all','r_max','hit_m_rare','hit_c_rare']].mean().sort_values('r_rare',ascending=False);em.to_csv(out/'一対一照合_探索平均.csv')
    core.save_json(core.HERE/'一対一照合_設定選択.json',settings)
    core.record('一対一照合',dict(seconds=time.time()-start,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outer_folds=7,inner_folds=6,
        main_truth_used=False,heldout_team_labels_used_for_training_or_choice=False,forced_assignment_of_all_queries=False))
    print(mean.to_string(),flush=True);print(em.head(12).to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=core.ROOT);run(p.parse_args().root.resolve())
