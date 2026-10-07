"""他C参照の設定選択基準と、探索時の選びすぎを追加検証する。

外側ラベルから外した内側平均で、希少/全体/max/全体非低下条件つき希少を
比較する。固定の全候補群に対するラベル置換は探索結果の診断であって、
本戦の有意性や他チームの成績を判定するものではない。
"""
from __future__ import annotations
import 分析 as core
import argparse, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import 他チームCを参照する照合 as study


def rank_choices(means,family,objective,small):
    a=means[means.method.str.startswith(family+'/')].copy()
    if small:
        keep=a.method.eq(family+'/unchanged') | (a.method.str.contains('/partial5_sex/',regex=False)&a.method.str.contains('/other_median/',regex=False)&a.method.str.endswith(('_threshold0','_threshold0.5')))
        a=a[keep]
    if objective=='rare_guard_all':
        baseline=float(a[a.method.eq(family+'/unchanged')].r_all.iloc[0]);a=a[a.r_all>=baseline-1e-12]
        keys=['r_rare','r_all','method']
    elif objective=='rare':keys=['r_rare','r_all','method']
    elif objective=='all':keys=['r_all','r_rare','method']
    elif objective=='max':keys=['r_max','r_all','method']
    else:raise ValueError(objective)
    return a.sort_values(keys,ascending=[False,False,True]).iloc[0]


def null_exploration(hit,member,onset,names,repetitions,out):
    # hit: 対象×設定×希少26人。time/onsetを固定し、各対象のonset層内で
    # member人数を保存して置換する。各設定の共通した偶然性を同時に保存。
    rng=np.random.default_rng(core.SEED);t,a,n=hit.shape
    lookup=np.array([[core.rare_score(m,c) for c in range(14)] for m in range(14)])
    observed=np.array([lookup[(h*member[i]).sum(1), (h*(~member[i])).sum(1)] for i,h in enumerate(hit)]).mean(0)
    pools={'publicNN':np.array([i for i,name in enumerate(names) if name.startswith('publicNN/')]),
           'published8keep':np.array([i for i,name in enumerate(names) if name.startswith('published8keep/')])}
    sims={family:[] for family in pools};singles={family:[] for family in pools};rows=[]
    for start in range(0,repetitions,512):
        batch=min(512,repetitions-start);score=np.zeros((a,batch))
        for i in range(t):
            m=np.zeros((batch,n),np.float32)
            for value in np.unique(onset[i]):
                ix=np.flatnonzero(onset[i]==value);count=int(member[i,ix].sum())
                chosen=np.argsort(rng.random((batch,len(ix))),axis=1)[:,:count]
                m[np.arange(batch)[:,None],ix[chosen]]=1
            hm=(hit[i].astype(np.float32)@m.T).astype(int)
            hc=hit[i].sum(1)[:,None]-hm
            score+=lookup[hm,hc]/t
        for family,ix in pools.items():
            best=ix[observed[ix].argmax()];sims[family].append(score[ix].max(0));singles[family].append(score[best])
    for family,ix in pools.items():
        best=ix[observed[ix].argmax()];sim=np.concatenate(sims[family]);single=np.concatenate(singles[family]);obs=float(observed[best])
        rows.append(dict(family=family,settings=len(ix),best_method=names[best],observed_rare=obs,targets=t,repetitions=repetitions,
            permutation_mean_of_best=float(sim.mean()),permutation_best_q95=float(np.quantile(sim,.95)),permutation_best_q99=float(np.quantile(sim,.99)),
            exploratory_family_adjusted_p=float((1+(sim>=obs-1e-12).sum())/(1+repetitions)),
            best_setting_unadjusted_p=float((1+(single>=obs-1e-12).sum())/(1+repetitions)),
            selection_of_other_method_families_adjusted=False,main_score_significance_test=False))
    pd.DataFrame(rows).to_csv(out/'他C参照_探索時のラベル置換.csv',index=False)
    return rows


def run(root,repetitions):
    started=time.time();cases,bg=core.datasets(root,'prelim');raw=study.precompute(cases,bg)
    previous=core.read(root/'予備戦/攻撃フェーズ/提出記録/8_20260907_GenLRA_Zhou/F_aia.csv');bases=study.baselines(cases,raw,previous)
    ids=np.arange(len(cases));groups=np.array([v['team'] for v in cases]);rows=[];choices=[];innermeans=[];hitrows=[];members=[];onsets=[];teams=[];names=None
    objectives=['rare','all','max','rare_guard_all']
    for outer,(train,test) in enumerate(GroupKFold(n_splits=7).split(ids,groups=groups)):
        inner=[]
        for tr,te in GroupKFold(n_splits=6).split(train,groups=groups[train]):
            trids,teids=train[tr],train[te];assert not set(groups[trids])&set(groups[test])
            model=study.fit_rare(cases,trids)
            for i in teids:
                v=cases[i]
                if v['team'] in [1,2,5]:continue
                for family,b in bases[i].items():
                    for name,p in study.predictions(v,raw[i],model,b,family).items():
                        inner.append(dict(team=v['team'],method=name,**core.score(p,v['truth'])))
        means=pd.DataFrame(inner).groupby('method')[['r_all','r_rare','r_max']].mean().reset_index();innermeans.append(means.assign(outer=outer))
        selected={}
        for small in [False,True]:
            for objective in objectives:
                for family in ['publicNN','published8keep']:
                    selected[(small,objective,family)]=rank_choices(means,family,objective,small)
        model=study.fit_rare(cases,train)
        for i in test:
            v=cases[i]
            if v['team'] in [1,2,5]:continue
            preds={}
            for family,b in bases[i].items():preds.update(study.predictions(v,raw[i],model,b,family))
            if names is None:names=sorted(preds)
            assert names==sorted(preds)
            t=v['truth'];r=t.is_rare.to_numpy(bool)
            hitrows.append(np.stack([np.abs(preds[name][r]-t.time.to_numpy()[r])<=.5 for name in names]));members.append(t.label.eq('member').to_numpy()[r]);onsets.append(t.onset.to_numpy()[r]);teams.append(v['team'])
            for (small,objective,family),chosen in selected.items():
                score=core.score(preds[chosen.method],t);baseline=core.score(bases[i][family],t)
                rows.append(dict(team=v['team'],outer=outer,family=family,search='small7' if small else 'full253',objective=objective,method=chosen.method,**score,
                    baseline_rare=baseline['r_rare'],baseline_all=baseline['r_all'],rare_gain=score['r_rare']-baseline['r_rare'],all_gain=score['r_all']-baseline['r_all']))
        for (small,objective,family),chosen in selected.items():
            choices.append(dict(outer=outer,family=family,search='small7' if small else 'full253',objective=objective,selected=chosen.method,
                                inner_rare=float(chosen.r_rare),inner_all=float(chosen.r_all),test_teams=sorted(map(int,groups[test]))))
        print(dict(phase='policy',outer=outer,seconds=round(time.time()-started,1)),flush=True)
    out=core.HERE/'集計';table=pd.DataFrame(rows)
    table.to_csv(out/'参照C_選択基準比較_対象別.csv',index=False)
    mean=table.groupby(['family','search','objective'])[['r_rare','r_all','baseline_rare','baseline_all','rare_gain','all_gain']].mean();mean.to_csv(out/'参照C_選択基準比較_平均.csv')
    pd.concat(innermeans,ignore_index=True).to_csv(out/'参照C_内側平均と選択可能設定.csv',index=False)
    core.save_json(core.HERE/'参照C_選択基準比較_設定選択.json',choices)
    null=null_exploration(np.stack(hitrows),np.stack(members),np.stack(onsets),names,repetitions,out)
    core.record('参照C_選択基準と偶然性',dict(seconds=time.time()-started,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        predictor_code_sha256=hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),outer_folds=7,inner_folds=6,permutation_repetitions=repetitions,
        heldout_labels_used_for_nested_choice=False,permutation_stratum='target and rare onset',prediction_model_retrained_for_permutation=False,
        descriptive_exploratory_p_values_only=True,main_truth_used=False))
    print(mean.to_string(),flush=True);print(pd.DataFrame(null).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=core.ROOT);p.add_argument('--repetitions',type=int,default=20000)
    args=p.parse_args();run(args.root.resolve(),args.repetitions)
