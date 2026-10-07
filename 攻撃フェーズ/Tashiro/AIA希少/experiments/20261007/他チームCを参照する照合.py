"""非共有AIA標的に対し、他チームの公開Cを参照分布として比較する。

他Cと同一人物を復元する方法ではない。全他CへのNN距離の中央値/10%点を
対象CへのNN距離と比較する。公開QI/C/BGだけで距離を作る。
希少判定モデルと設定選択の両方で外側評価チームの正解を除く二重CV。
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

CONFIGS=[(5,'soft'),(5,'sex'),(5,'both'),(4,'sex')]


def precompute(cases,bg):
    start=time.time();out=[]
    samples=np.array([np.random.default_rng(core.SEED+i).choice(len(bg),1049,False) for i in range(16)])
    for i,v in enumerate(cases):
        row=dict(team=v['team'],pred={},rank={})
        for n,cat in CONFIGS:
            metric=f'partial{n}_{cat}'
            minima=[];nn=None;selfdist=None
            for j,w in enumerate(cases):
                dist=core.cost(v['q'],w['c'],bg,n,cat)
                if i==j:
                    nn=dist.argmin(1);selfdist=dist[np.arange(326),nn]
                else:minima.append(dist.min(1))
            foreign=np.column_stack(minima)
            db=core.cost(v['q'],bg,bg,n,cat)
            medbg=np.median(np.column_stack([db[:,a].min(1) for a in samples]),axis=1)
            # 参照サンプル内の性別/県一致ゼロの場合は、全背景への距離を使う。
            medbg=np.where(np.isfinite(medbg),medbg,db.min(1))
            row['pred'][metric]=v['c'].time.to_numpy()[nn]
            for ref,r in [('bg16',medbg),('other_median',np.median(foreign,axis=1)),('other_q10',np.quantile(foreign,.1,axis=1))]:
                rank=np.log((r+1e-8)/(selfdist+1e-8));assert np.isfinite(rank).all()
                row['rank'][metric+'/'+ref]=rank
        out.append(row)
        print(dict(phase='distance',team=v['team'],seconds=round(time.time()-start,1)),flush=True)
    return out


def predictions(v,raw,model,baseline,family):
    pq=model.predict_proba(raremod.features(v['q']))[:,1]
    out={family+'/unchanged':baseline}
    for metric,times in raw['pred'].items():
        for ref in ['bg16','other_median','other_q10']:
            rank=raw['rank'][metric+'/'+ref]
            for number in [26,52,82]:
                rare=raremod.top(pq,number);ids=np.flatnonzero(rare)
                for f in [.25,.5,.75]:
                    take=np.zeros(326,bool);take[ids[study.gate(rank[ids],f)]]=True
                    out[f'{family}/{metric}/{ref}/rare_top{number}_keep{f:g}']=np.where(rare,np.where(take,times,11.),baseline)
                for threshold in [0.,.5,1.,2.]:
                    take=rank>=threshold
                    out[f'{family}/{metric}/{ref}/rare_top{number}_threshold{threshold:g}']=np.where(rare,np.where(take,times,11.),baseline)
    return out


def fit_rare(cases,ids):
    return raremod.make_model().fit(np.vstack([raremod.features(cases[i]['q']) for i in ids]),
                                  np.concatenate([cases[i]['truth'].is_rare.to_numpy(bool) for i in ids]))


def baselines(cases,raw,previous):
    out=[]
    for v,r in zip(cases,raw):
        # このファイルのbg16サンプルはseed+i方式。他ファイルのS4とは別比較元。
        rank=r['rank']['partial5_soft/bg16'];b=np.where(study.gate(rank,.75),r['pred']['partial5_soft'],0.)
        bases={'publicNN':b}
        if v['team']!=5:
            bases['published8keep']=previous[previous.target_k.eq(v['team'])].set_index('challenge_row_id').loc[v['q'].challenge_row_id,'time_hat'].to_numpy()
        out.append(bases)
    return out


def run(root):
    started=time.time();cases,bg=core.datasets(root,'prelim');raw=precompute(cases,bg)
    previous=core.read(root/'予備戦/攻撃フェーズ/提出記録/8_20260907_GenLRA_Zhou/F_aia.csv')
    bases=baselines(cases,raw,previous)
    ids=np.arange(len(cases));groups=np.array([v['team'] for v in cases]);rows=[];explore=[];select=[];hits=[]
    for outer,(train,test) in enumerate(GroupKFold(n_splits=7).split(ids,groups=groups)):
        innerrows=[]
        for inner,(tr,te) in enumerate(GroupKFold(n_splits=6).split(train,groups=groups[train])):
            trids,teids=train[tr],train[te]
            assert not set(groups[trids])&set(groups[test])
            model=fit_rare(cases,trids)
            for i in teids:
                v=cases[i]
                if v['team'] in [1,2,5]:continue
                for family,b in bases[i].items():
                    for name,p in predictions(v,raw[i],model,b,family).items():
                        innerrows.append(dict(team=v['team'],method=name,**core.score(p,v['truth'])))
        chosen={};inner=pd.DataFrame(innerrows)
        for family in ['publicNN','published8keep']:
            rank=inner[inner.method.str.startswith(family+'/')].groupby('method')[['r_rare','r_all']].mean().reset_index()
            rank=rank.sort_values(['r_rare','r_all','method'],ascending=[False,False,True]);chosen[family]=rank.iloc[0].method
            select.append(dict(outer=outer,family=family,selected=chosen[family],inner_rare=float(rank.iloc[0].r_rare),inner_all=float(rank.iloc[0].r_all),
                               train_teams=sorted(map(int,groups[train])),test_teams=sorted(map(int,groups[test]))))
        model=fit_rare(cases,train)
        for i in test:
            v=cases[i]
            if v['team'] in [1,2,5]:continue
            for family,b in bases[i].items():
                preds=predictions(v,raw[i],model,b,family)
                for name,p in preds.items():explore.append(dict(team=v['team'],method=name,**core.score(p,v['truth'])))
                name=chosen[family];p=preds[name];score=core.score(p,v['truth']);bs=core.score(b,v['truth'])
                rows.append(dict(team=v['team'],outer=outer,family=family,method=name,**score,baseline_rare=bs['r_rare'],baseline_all=bs['r_all'],rare_gain=score['r_rare']-bs['r_rare'],all_gain=score['r_all']-bs['r_all']))
                r=v['truth'].is_rare.to_numpy(bool);t=v['truth']
                hits.append(dict(team=v['team'],family=family,hit=np.abs(p[r]-t.time.to_numpy()[r])<=.5,member=t.label.eq('member').to_numpy()[r],
                                 onset=t.onset.to_numpy()[r],sex=v['q'].sex.to_numpy()[r],age_band=(v['q'].age.to_numpy()[r]//10).astype(int)))
        print(dict(phase='nested',outer=outer,seconds=round(time.time()-started,1)),flush=True)
    out=core.HERE/'集計';table=pd.DataFrame(rows);ex=pd.DataFrame(explore)
    table.to_csv(out/'他チームC参照_二重CV_対象別.csv',index=False)
    mean=table.groupby('family')[['r_rare','r_all','baseline_rare','baseline_all','rare_gain','all_gain']].mean();mean.to_csv(out/'他チームC参照_二重CV_平均.csv')
    ex.to_csv(out/'他チームC参照_探索対象別.csv',index=False)
    exmean=ex.groupby('method')[['r_rare','r_all','r_max','hit_m_rare','hit_c_rare']].mean().sort_values('r_rare',ascending=False);exmean.to_csv(out/'他チームC参照_探索平均.csv')
    core.save_json(core.HERE/'他チームC参照_設定選択.json',select)
    np.savez_compressed(core.HERE/'作業用/他C参照二重CV_希少正誤.npz',team=np.array([v['team'] for v in hits]),family=np.array([v['family'] for v in hits]),
                        **{key:np.stack([v[key] for v in hits]) for key in ['hit','member','onset','sex','age_band']})
    # 本戦は全公開予備戦から学習。真値はなくスコアは計算しない。
    main,mbg=core.datasets(root,'main');mraw=precompute(main,mbg)
    current=core.read(root/'本戦/攻撃フェーズ/Tashiro/20261006_第2回提出データ統合/入力/AIA/F_aia.csv')
    model=fit_rare(cases,ids);diag=[]
    for v,r in zip(main,mraw):
        pq=model.predict_proba(raremod.features(v['q']))[:,1];rare=raremod.top(pq,26)
        for metric in r['pred']:
            for ref in ['bg16','other_median','other_q10']:
                a=r['rank'][metric+'/'+ref];diag.append(dict(team=v['team'],metric=metric,reference=ref,median_rank=float(np.median(a)),
                            rare_candidate_median_rank=float(np.median(a[rare])),rare_candidate_positive_rank=int((a[rare]>0).sum()),rare_truth_known=False))
    pd.DataFrame(diag).to_csv(out/'他チームC参照_本戦の距離診断.csv',index=False)
    core.record('他チームC参照',dict(seconds=time.time()-started,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outer_folds=7,inner_folds=6,main_truth_used=False,heldout_team_labels_used_for_training_or_choice=False,
        reference_other_C_identity_match_assumed=False,public_heldout_C_used_as_unlabelled_reference=True))
    print(mean.to_string(),flush=True);print(exmean.head(12).to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=core.ROOT);run(p.parse_args().root.resolve())
