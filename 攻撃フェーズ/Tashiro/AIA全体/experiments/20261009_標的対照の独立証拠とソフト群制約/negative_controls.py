#!/usr/bin/env python3
"""Counterfactual controls. A target's truth is never used to fit a model or construct Cfake."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from experiment import read_data,feature_table,SEED,TEAMS

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).parent/'input');parser.add_argument('--out',type=Path,default=Path(__file__).parent/'output');args=parser.parse_args()
root=args.root;out=args.out;out.mkdir(parents=True,exist_ok=True);bg,D=read_data(root)
F,labels=feature_table(bg,D)
member=['member_near4','member_knn4','member_near5','member_partial2',
        'rank_member_near4','rank_member_near5','rank_member_partial2']+[f'c_{n}' for n in ['sbp','tg','hdl','alt']]
bgfeat=['bg_near4','bg_knn4','bg_near5','rank_bg_near4','rank_bg_knn4']+[f'bg_{n}' for n in ['sbp','tg','hdl','alt']]
ratio=['contrast_near4','contrast_knn4','contrast_near5','col4','col2','col1','col4_min',
       'rank_contrast_near4','rank_col4','rank_col2','rank_col1']+[f'ratio_{n}' for n in ['sbp','tg','hdl','alt']]
columns=member+bgfeat+ratio
fake_bg={};fake_scramble={}
for k,(q,c,t) in D.items():
    rng=np.random.default_rng(SEED+k)
    # Draw a null synthetic C from public background; no target B record is used.
    cnull=bg.iloc[rng.choice(len(bg),size=1049,replace=False)].drop(columns=['record_id']).copy()
    fake_bg[k]=(q,cnull,t)
    cs=c.copy()
    # Within-sex shuffle preserves exact univariate distributions of age/clinical attributes.
    for sex in (0,1):
        idx=np.where(cs.sex.to_numpy()==sex)[0]
        for col in ['age','SBP','TG','HDL','ALT']:
            cs.iloc[idx,cs.columns.get_loc(col)]=rng.permutation(cs.iloc[idx][col].to_numpy())
    fake_scramble[k]=(q,cs,t)

print('Feature calculation for negative controls...',flush=True)
Fbg,_=feature_table(bg,fake_bg)
Fs,_=feature_table(bg,fake_scramble)
rows=[]
for k in TEAMS:
    others=[j for j in TEAMS if j!=k]
    x=pd.concat([F[j][columns] for j in others],ignore_index=True)
    y=np.concatenate([labels[j].label.eq('member').to_numpy(int) for j in others])
    m=make_pipeline(StandardScaler(),LogisticRegression(C=.04,max_iter=1500,solver='lbfgs'))
    m.fit(x.to_numpy(float),y)
    ytest=labels[k].label.eq('member').to_numpy(int)
    for name,dd in [('real_C',F),('shuffled_C',Fs),('BG_sample_as_C',Fbg)]:
        p=m.predict_proba(dd[k][columns].to_numpy(float))[:,1]
        rows.append({'team':k,'intervention':name,'auc':roc_auc_score(ytest,p),'p_mean':p.mean(),'p_std':p.std()})
r=pd.DataFrame(rows);r.to_csv(out/'negative_controls.csv',index=False)
print(r.query('team!=5').groupby('intervention').agg(auc=('auc','mean'),p_mean=('p_mean','mean'),p_std=('p_std','mean')).round(5).to_string(),flush=True)
for mode in ['shuffled_C','BG_sample_as_C']:
    p=r[r.intervention=='real_C'].set_index('team').auc-r[r.intervention==mode].set_index('team').auc
    v=p.drop(index=5).to_numpy()
    gen=np.random.default_rng(SEED);samples=v[gen.integers(0,len(v),(20000,len(v)))].mean(axis=1)
    print('Effect real minus',mode,'mean',v.mean(),'CI',np.quantile(samples,[.025,.975]),'wins',(v>0).sum(),'loss',(v<0).sum(),flush=True)
