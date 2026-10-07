"""少数候補の全体非低下ガードについて、分割安定性と対応置換を検証。

候補群は先行探索から絞った7設定。完全な新規検証データではない。
分割反復を独立データ数として数えない。置換は公開予備戦の希少層のみ、
県・性別・年齢帯・発症有無を固定し、内側設定選択も毎回やり直す。
"""
from __future__ import annotations
import 分析 as core
import argparse, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import 他チームCを参照する照合 as study
import 希少候補の学習 as raremod

NAMES=['unchanged']+[f'top{n}_threshold{t:g}' for n in [26,52,82] for t in [0.,.5]]
NAMES=sorted(NAMES)
BASE=NAMES.index('unchanged');FIXED=NAMES.index('top52_threshold0')


def predict(v,raw,model,base):
    prob=model.predict_proba(raremod.features(v['q']))[:,1]
    rank=raw['rank']['partial5_sex/other_median'];times=raw['pred']['partial5_sex']
    preds={'unchanged':base}
    for n in [26,52,82]:
        selected=raremod.top(prob,n)
        for threshold in [0.,.5]:preds[f'top{n}_threshold{threshold:g}']=np.where(selected,np.where(rank>=threshold,times,11.),base)
    return np.stack([preds[name] for name in NAMES])


def panel(v,p):
    t=v['truth'];rare=t.is_rare.to_numpy(bool);member=t.label.eq('member').to_numpy();hit=np.abs(p-t.time.to_numpy()[None,:])<=.5
    q=v['q'];keys=np.column_stack([q.prefecture.to_numpy()[rare],q.sex.to_numpy()[rare],(q.age.to_numpy()[rare]//10).astype(int),t.onset.to_numpy()[rare]])
    strata=[]
    key_rows=list(map(tuple,keys))
    for key in sorted(set(key_rows)):
        ix=np.array([i for i,value in enumerate(key_rows) if value==key],int);count=int(member[rare][ix].sum())
        assert 2*count==len(ix),'標的と対照の対応層人数が異なる'
        strata.append((ix,count))
    assert sum(count for ix,count in strata)==13
    assert sorted(i for ix,count in strata for i in ix)==list(range(26))
    return dict(team=v['team'],rare_hit=hit[:,rare],member=member[rare],normal_m=hit[:,member&~rare].sum(1),
                normal_c=hit[:,~member&~rare].sum(1),strata=strata)


def score_panel(p,member=None):
    m=p['member'][:,None] if member is None else member
    hm=(p['rare_hit'].astype(np.float32)@m.astype(np.float32)).astype(int);hc=p['rare_hit'].sum(1)[:,None]-hm
    rare=np.array([[core.rare_score(i,j) for j in range(14)] for i in range(14)])[hm,hc]
    am=hm+p['normal_m'][:,None];ac=hc+p['normal_c'][:,None]
    all_=np.clip((am-ac)/np.maximum(163-ac,1e-9),0,1)
    return rare,all_


def decision(rr,aa,objective):
    if objective=='rare_guard_all':
        eligible=aa>=aa[BASE][None,:]-1e-12;r=np.where(eligible,rr,-1.)
        return np.lexsort((np.broadcast_to(np.arange(len(NAMES))[:,None],r.shape),-aa,-r),axis=0)[0]
    if objective=='all':return np.lexsort((np.broadcast_to(np.arange(len(NAMES))[:,None],rr.shape),-rr,-aa),axis=0)[0]
    if objective=='fixed':return np.full(rr.shape[1],FIXED,int)
    if objective=='unchanged':return np.full(rr.shape[1],BASE,int)
    raise ValueError(objective)


def permutation(first_panels,case_panels,observed,count,out):
    rng=np.random.default_rng(core.SEED);objectives=['rare_guard_all','all','fixed'];null={key:[] for key in objectives}
    for start in range(0,count,512):
        b=min(512,count-start);masks={}
        for team,p in case_panels.items():
            m=np.zeros((26,b),np.float32)
            for ix,k in p['strata']:
                j=np.argsort(rng.random((b,len(ix))),axis=1)[:,:k]
                m[ix[j],np.arange(b)[:,None]]=1
            assert (m.sum(0)==13).all();masks[team]=m
        sums={key:np.zeros(b) for key in objectives};ntargets=0
        for fold in first_panels:
            ir=np.zeros((len(NAMES),b));ia=np.zeros_like(ir)
            for p in fold['inner']:
                rr,aa=score_panel(p,masks[p['team']]);ir+=rr/len(fold['inner']);ia+=aa/len(fold['inner'])
            choices={key:decision(ir,ia,key) for key in objectives}
            for p in fold['outer']:
                rr,aa=score_panel(p,masks[p['team']]);ntargets+=1
                for key,choice in choices.items():sums[key]+=rr[choice,np.arange(b)]
        assert ntargets==18
        for key in objectives:null[key].append(sums[key]/18)
    arrays={key:np.concatenate(value) for key,value in null.items()};maxpolicy=np.stack(list(arrays.values())).max(0);rows=[]
    for key,sim in arrays.items():
        obs=float(observed[key]);rows.append(dict(policy=key,observed_rare=obs,permutations=count,targets=18,
            null_mean=float(sim.mean()),null_q95=float(np.quantile(sim,.95)),null_q99=float(np.quantile(sim,.99)),
            conditional_p=float((1+(sim>=obs-1e-12).sum())/(count+1)),
            three_policy_max_adjusted_p=float((1+(maxpolicy>=obs-1e-12).sum())/(count+1)),
            earlier_candidate_pool_exploration_adjusted=False,main_score_test=False))
    pd.DataFrame(rows).to_csv(out/'参照C_対応層を保った内側再選択置換.csv',index=False)
    return rows


def run(root,repeats,permutations):
    start=time.time();cases,bg=core.datasets(root,'prelim');raw=study.precompute(cases,bg)
    previous=core.read(root/'予備戦/攻撃フェーズ/提出記録/8_20260907_GenLRA_Zhou/F_aia.csv')
    bases=[None if v['team']==5 else previous[previous.target_k.eq(v['team'])].set_index('challenge_row_id').loc[v['q'].challenge_row_id,'time_hat'].to_numpy() for v in cases]
    ids=np.arange(len(cases));groups=np.array([v['team'] for v in cases]);rows=[];first=[];case_panels={};rng=np.random.default_rng(core.SEED)
    for repeat in range(repeats):
        if repeat==0:folds=list(GroupKFold(n_splits=7).split(ids,groups=groups))
        else:
            shuffled=rng.permutation(ids);folds=[(np.setdiff1d(ids,test),test) for test in np.array_split(shuffled,7)]
        for outer,(train,test) in enumerate(folds):
            assert not set(groups[train])&set(groups[test]);innerpanels=[];outerpanels=[]
            if repeat==0:innerfolds=[(train[a],train[b]) for a,b in GroupKFold(n_splits=6).split(train,groups=groups[train])]
            else:
                shuffled=rng.permutation(train);innerfolds=[(np.setdiff1d(train,te),te) for te in np.array_split(shuffled,6)]
            for tr,te in innerfolds:
                model=study.fit_rare(cases,tr)
                for i in te:
                    v=cases[i]
                    if v['team'] not in [1,2,5]:innerpanels.append(panel(v,predict(v,raw[i],model,bases[i])))
            scores=[score_panel(p) for p in innerpanels];ir=np.mean([v[0] for v in scores],axis=0);ia=np.mean([v[1] for v in scores],axis=0)
            chosen={key:int(decision(ir,ia,key)[0]) for key in ['rare_guard_all','all','fixed','unchanged']}
            model=study.fit_rare(cases,train)
            for i in test:
                v=cases[i]
                if v['team'] in [1,2,5]:continue
                p=panel(v,predict(v,raw[i],model,bases[i]));rr,aa=score_panel(p);outerpanels.append(p)
                if repeat==0:case_panels[v['team']]=p
                for key,j in chosen.items():rows.append(dict(repeat=repeat,outer=outer,team=v['team'],policy=key,method=NAMES[j],r_rare=float(rr[j,0]),r_all=float(aa[j,0]),rare_gain=float(rr[j,0]-rr[BASE,0]),all_gain=float(aa[j,0]-aa[BASE,0])))
            if repeat==0:first.append(dict(inner=innerpanels,outer=outerpanels))
        print(dict(repeat=repeat,seconds=round(time.time()-start,1)),flush=True)
    out=core.HERE/'集計';table=pd.DataFrame(rows)
    table.to_csv(out/'参照C_反復分割_対象別.csv',index=False)
    rep=table.groupby(['repeat','policy'])[['r_rare','r_all','rare_gain','all_gain']].mean();rep.to_csv(out/'参照C_反復分割_分割別平均.csv')
    stats=[]
    for key,g in rep.reset_index().groupby('policy'):
        for metric in ['r_rare','r_all','rare_gain','all_gain']:
            a=g[metric].to_numpy();stats.append(dict(policy=key,metric=metric,split_mean=float(a.mean()),split_median=float(np.median(a)),split_min=float(a.min()),split_max=float(a.max()),
                                split_q025=float(np.quantile(a,.025)),split_q975=float(np.quantile(a,.975)),repeats=repeats,unique_targets=18,
                                independent_test_sets=False,split_quantiles_are_performance_CI=False))
    pd.DataFrame(stats).to_csv(out/'参照C_反復分割_揺れの集計.csv',index=False)
    average=table.groupby(['policy','team'])[['r_rare','r_all','rare_gain','all_gain']].mean().reset_index();average.to_csv(out/'参照C_反復分割_対象平均.csv',index=False)
    ci=[]
    for key,g in average.groupby('policy'):
        index=rng.integers(0,len(g),size=(50000,len(g)))
        for metric in ['r_rare','r_all','rare_gain','all_gain']:
            a=g[metric].to_numpy();sim=a[index].mean(1);ci.append(dict(policy=key,metric=metric,mean=float(a.mean()),bootstrap_low=float(np.quantile(sim,.025)),bootstrap_high=float(np.quantile(sim,.975)),
                                                remove_highest_target_mean=float(np.sort(a)[:-1].mean()),targets=18,exploratory_CI=True))
    pd.DataFrame(ci).to_csv(out/'参照C_対象単位の区間と最大寄与除外.csv',index=False)
    observed=table[table.repeat.eq(0)].groupby('policy').r_rare.mean().to_dict()
    null=permutation(first,case_panels,observed,permutations,out)
    core.record('参照C_分割安定性と対応置換',dict(seconds=time.time()-start,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        repeats=repeats,unique_evaluation_targets=18,outer_folds=7,inner_folds=6,permutations=permutations,
        permutation_strata=['target','rare','prefecture','sex','age_band_10_years','onset'],inner_choice_recomputed_under_each_permutation=True,
        nonrare_membership_labels_fixed=True,rare_classifier_not_retrained_for_permutation_because_rare_labels_unchanged=True,
        candidate_pool_chosen_after_earlier_exploration=True,main_truth_used=False,full_pipeline_confirmatory_p=False))
    print(pd.DataFrame(stats).to_string(index=False),flush=True);print(pd.DataFrame(null).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=core.ROOT);p.add_argument('--repeats',type=int,default=50);p.add_argument('--permutations',type=int,default=20000)
    args=p.parse_args();run(args.root.resolve(),args.repeats,args.permutations)
