"""希少個人を学習単位にし、標的/対照の正解確率を時間候補ごとに学習。

時間候補ごとに同じ個人を複製する方式と異なり、min_samples_leaf は個人数。
公開予備戦のみで学習し、外側の評価チームを学習・内側設定選択から外す。
分析予測はメモリ上のみ。提出用CSV/ZIPは生成しない。
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
import 比較特徴量キャッシュ as cachemod

LEAVES=[20,40,80]
LAMBDAS=[.5,.75,1.,1.25,1.5]
TOPS=[26,52,82]


def load(root):
    d,cache=cachemod.load(root); names=d['names'].tolist()
    cols=[i for i,name in enumerate(names) if name in ['constant_0','constant_9.5'] or name.endswith(('/nn_all','/window5_lam0','/window5_lam0.25'))]
    actions=[names[i] for i in cols]+['constant_11']
    p=np.column_stack([d['prelim_pred'][:,cols],np.full(len(d['prelim_team']),11.)])
    # 各個人の全候補時刻を特徴量に加える。真値timeは加えない。
    x=np.column_stack([d['prelim_x'],p/10])
    g=d['prelim_team']; rare=d['prelim_rare'].astype(bool);member=d['prelim_member'].astype(bool)
    hit=(np.abs(p-d['prelim_time'][:,None])<=.5).astype(float)
    y=np.concatenate([hit*member[:,None],hit*(~member[:,None])],axis=1)
    cases,bg=core.datasets(root,'prelim')
    assert np.array_equal(g,np.concatenate([np.full(326,v['team']) for v in cases]))
    qx=np.vstack([raremod.features(v['q']) for v in cases])
    base=d['prelim_pred'][:,names.index('S4_log_iqr_soft/nn_gate0.75_0')]
    previous=core.read(root/'予備戦/攻撃フェーズ/提出記録/8_20260907_GenLRA_Zhou/F_aia.csv')
    oldbase=base.copy()
    for i,v in enumerate(cases):
        if v['team']!=5:
            oldbase[i*326:(i+1)*326]=previous[previous.target_k.eq(v['team'])].set_index('challenge_row_id').loc[v['q'].challenge_row_id,'time_hat'].to_numpy()
    return dict(x=x,p=p,y=y,g=g,rare=rare,qx=qx,cases=cases,bases={'S4keep':base,'published8keep':oldbase},actions=actions,
                cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())


def probabilities(d,train,test,leaf):
    tr=train[d['rare'][train]]
    assert not set(d['g'][tr])&set(d['g'][test])
    model=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=leaf,max_depth=12,max_features=.8,
                              n_jobs=1,random_state=core.SEED).fit(d['x'][tr],d['y'][tr])
    return model.predict(d['x'][test])


def choose(p,prob,lam):
    a=p.shape[1]
    delta=prob[:,:a]-lam*prob[:,a:]
    # 11年の適中は必ずゼロ。最大利得0のときは11年を選ぶ。
    j=delta.argmax(axis=1)
    j[delta.max(axis=1)<=1e-12]=a-1
    return p[np.arange(len(p)),j]


def evaluate(d,teams,probs,rareprob):
    rows=[];preds={}
    for leaf,prob in probs.items():
        for lam in LAMBDAS:
            raw=choose(d['p'],prob,lam)
            for top in TOPS:
                for family,base in d['bases'].items():
                    name=f'{family}/person_leaf{leaf}_lambda{lam:g}_top{top}'
                    final=base.copy()
                    for i,v in enumerate(d['cases']):
                        if v['team'] not in teams or (family=='published8keep' and v['team']==5):continue
                        sl=slice(i*326,(i+1)*326)
                        final[sl]=np.where(raremod.top(rareprob[sl],top),raw[sl],base[sl])
                        if v['team'] not in [1,2,5]:
                            rows.append(dict(team=v['team'],method=name,**core.score(final[sl],v['truth'])))
                    preds[name]=final
    for family,base in d['bases'].items():
        name=family+'/unchanged';preds[name]=base
        for i,v in enumerate(d['cases']):
            if v['team'] in teams and v['team'] not in [1,2,5]:
                rows.append(dict(team=v['team'],method=name,**core.score(base[i*326:(i+1)*326],v['truth'])))
    return pd.DataFrame(rows),preds


def run(root,nested):
    started=time.time(); d=load(root)
    ids=np.arange(len(d['g']));groups=d['g'];a=d['p'].shape[1]
    rows=[];settings=[];folds=[];honest_hits=[]
    oof={leaf:np.full(d['y'].shape,np.nan) for leaf in LEAVES}; oofrare=np.zeros(len(ids))
    for outer,(train,test) in enumerate(GroupKFold(n_splits=7).split(ids,groups=groups)):
        trainteams=set(groups[train]);testteams=set(groups[test]);assert not trainteams&testteams
        selected={}
        if nested:
            innerprob={leaf:np.full(d['y'].shape,np.nan) for leaf in LEAVES};innerrp=np.zeros(len(ids))
            for inner,(tr,te) in enumerate(GroupKFold(n_splits=6).split(train,groups=groups[train])):
                trids,teids=train[tr],train[te]
                model=raremod.make_model().fit(d['qx'][trids],d['rare'][trids])
                innerrp[teids]=model.predict_proba(d['qx'][teids])[:,1]
                for leaf in LEAVES:innerprob[leaf][teids]=probabilities(d,trids,teids,leaf)
                print(dict(outer=outer,inner=inner,seconds=round(time.time()-started,1)),flush=True)
            inner,_=evaluate(d,trainteams,innerprob,innerrp)
            for family in d['bases']:
                rank=inner[inner.method.str.startswith(family+'/')].groupby('method')[['r_rare','r_all']].mean().reset_index()
                rank=rank.sort_values(['r_rare','r_all','method'],ascending=[False,False,True])
                selected[family]=rank.iloc[0].method
                settings.append(dict(outer=outer,family=family,selected=rank.iloc[0].method,inner_rare=float(rank.iloc[0].r_rare),inner_all=float(rank.iloc[0].r_all),
                                     train_teams=sorted(map(int,trainteams)),test_teams=sorted(map(int,testteams))))
        rm=raremod.make_model().fit(d['qx'][train],d['rare'][train]);oofrare[test]=rm.predict_proba(d['qx'][test])[:,1]
        for leaf in LEAVES:oof[leaf][test]=probabilities(d,train,test,leaf)
        outertable,preds=evaluate(d,testteams,oof,oofrare)
        if nested:
            for family,name in selected.items():
                for i,v in enumerate(d['cases']):
                    if v['team'] not in testteams or v['team'] in [1,2,5]:continue
                    row=outertable[(outertable.team.eq(v['team']))&(outertable.method.eq(name))].iloc[0].to_dict()
                    base=outertable[(outertable.team.eq(v['team']))&(outertable.method.eq(family+'/unchanged'))].iloc[0]
                    rows.append(dict(**row,family=family,outer=outer,baseline_r_rare=float(base.r_rare),baseline_r_all=float(base.r_all),
                                     rare_gain=row['r_rare']-base.r_rare,all_gain=row['r_all']-base.r_all))
                    sl=slice(i*326,(i+1)*326);t=v['truth'];r=t.is_rare.to_numpy(bool);pred=preds[name][sl]
                    honest_hits.append(dict(team=v['team'],family=family,hit=(np.abs(pred[r]-t.time.to_numpy()[r])<=.5),
                                             member=t.label.eq('member').to_numpy()[r],onset=t.onset.to_numpy()[r],sex=v['q'].sex.to_numpy()[r],age_band=(v['q'].age.to_numpy()[r]//10).astype(int)))
        folds.append(dict(outer=outer,train_teams=sorted(map(int,trainteams)),test_teams=sorted(map(int,testteams))))
        print(dict(outer=outer,completed=True,seconds=round(time.time()-started,1)),flush=True)
    out=core.HERE/'集計'
    fulltable,_=evaluate(d,set(groups),oof,oofrare)
    prefix='個人単位の希少時間学習'+('_二重CV' if nested else '_単一CV')
    fulltable.to_csv(out/(prefix+'_全設定.csv'),index=False)
    fullmean=fulltable.groupby('method')[['r_rare','r_all','r_max','hit_m_rare','hit_c_rare']].mean().sort_values('r_rare',ascending=False)
    fullmean.to_csv(out/(prefix+'_探索平均.csv'))
    if nested:
        table=pd.DataFrame(rows);table.to_csv(out/(prefix+'_対象別.csv'),index=False)
        mean=table.groupby('family')[['r_rare','r_all','baseline_r_rare','baseline_r_all','rare_gain','all_gain']].mean();mean.to_csv(out/(prefix+'_平均.csv'))
        core.save_json(core.HERE/(prefix+'_選択記録.json'),settings)
        rng=np.random.default_rng(core.SEED);cis=[]
        for family,g in table.groupby('family'):
            ix=rng.integers(0,len(g),size=(50000,len(g)))
            for key in ['r_rare','r_all','rare_gain','all_gain']:
                dist=g[key].to_numpy()[ix].mean(axis=1)
                cis.append(dict(family=family,metric=key,mean=float(g[key].mean()),low=float(np.quantile(dist,.025)),high=float(np.quantile(dist,.975))))
        pd.DataFrame(cis).to_csv(out/(prefix+'_対象単位の区間.csv'),index=False)
        np.savez_compressed(core.HERE/'作業用/個人学習二重CV_希少正誤.npz',team=np.array([v['team'] for v in honest_hits]),family=np.array([v['family'] for v in honest_hits]),
                            **{key:np.stack([v[key] for v in honest_hits]) for key in ['hit','member','onset','sex','age_band']})
        print(mean.to_string(),flush=True)
    np.savez_compressed(core.HERE/'作業用'/(prefix+'_予備戦確率.npz'),rareprob=oofrare,**{f'leaf{leaf}':prob for leaf,prob in oof.items()})
    core.record(prefix,dict(seconds=time.time()-started,outer_folds=folds,inner_folds=6 if nested else 0,
                nested=nested,targets_evaluated=18,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cache_sha256=d['cache_sha256'],
                heldout_team_labels_used_for_training_or_setting_selection=False if nested else '設定を探索平均から選ぶ場合は使われる',actions=d['actions'],
                feature_persons_train_unit=True,main_truth_used=False))
    print(fullmean.head(10).to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=core.ROOT);p.add_argument('--nested',action='store_true')
    args=p.parse_args();run(args.root.resolve(),args.nested)
