#!/usr/bin/env python3
"""Paired bootstrap and strictly labeled posthoc equal-weight ensembles, not model selection proof."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,pandas as pd,json
import model_sweep as ms
OUT=ms.OUT
rngseed=20261009

def pair_stats(a,b,team_ids):
 dif=a.loc[team_ids].to_numpy(float)-b.loc[team_ids].to_numpy(float)
 rng=np.random.default_rng(rngseed)
 boot=rng.integers(0,len(dif),(20000,len(dif)))
 lo,hi=np.quantile(dif[boot].mean(axis=1),[.025,.975])
 return {'difference':float(dif.mean()),'ci025':float(lo),'ci975':float(hi),'better':int((dif>1e-12).sum()),'worse':int((dif<-1e-12).sum()),'same':int((abs(dif)<=1e-12).sum()),'max_abs_change':float(np.max(abs(dif)))}

def main():
 d=pd.concat([pd.read_csv(OUT/'per_team.csv'),pd.read_csv(OUT/'per_team_extra.csv')])
 t,q=({k:pd.read_pickle(ms.CACHE/f'truth_{k}.pkl')for k in ms.TEAMS},{k:pd.read_pickle(ms.CACHE/f'q_{k}.pkl')for k in ms.TEAMS})
 ref={}
 for k in ms.TEAMS:
  f=pd.read_pickle(ms.CACHE/f'features_{k}.pkl'); mask=ms.prior.top75(q[k],f['col2'].to_numpy(float))
  z=ms.prior.metrics(np.where(np.isin(np.arange(len(q[k])),list(mask)),9.5,0.),t[k])
  ref[k]=z['r_all']
 col2=pd.Series(ref)
 pvt=d.pivot(index='team',columns=['features','model'],values='r_all')
 rows=[]
 checks=[('lr_c10_base','c_bg','lr_c10','c_bg','lr_c004'),('xgb_flexible_local','c_bg_local','xgb_flexible','c_bg_local','lr_c004'),('xgb_flexible_base','c_bg','xgb_flexible','c_bg','lr_c004'),('lr_c02_local','c_bg_local','lr_c02','c_bg_local','lr_c004'),('lr_c2_base','c_bg','lr_c2','c_bg','lr_c004')]
 for scope,ids in [('nonself20',[k for k in ms.TEAMS if k!=5]),('processed18',[k for k in ms.TEAMS if k not in [1,2,5]])]:
  for nm,xr,xm,yr,ym in checks:
   vals=pvt[(xr,xm)];base=pvt[(yr,ym)]
   for cmpname,z in [('same_features_lr_c004',base),('column_top2',col2)]:
    rows.append(dict(scope=scope,model=nm,comparison=cmpname,**pair_stats(vals,z,ids)))
 pd.DataFrame(rows).to_csv(OUT/'paired_bootstrap.csv',index=False)
 probs=pd.read_csv(OUT/'public_predictions.csv'); piv=probs.pivot(index=['team','row_id'],columns=['features','model'],values='p_member')
 ensembles={
  'mix_lr004_hgb_base': [('c_bg','lr_c004'),('c_bg','hgb_shallow')],
  'mix_lr2_hgb_base':[('c_bg','lr_c2'),('c_bg','hgb_shallow')],
  'mix_lr2_hgb_local':[('c_bg_local','lr_c2'),('c_bg_local','hgb_shallow')],
  'mix_lr2_base_local':[('c_bg','lr_c2'),('c_bg_local','lr_c2')],
  'mix_lr02_base_local':[('c_bg','lr_c02'),('c_bg_local','lr_c02')],
 }
 # Additional posthoc high-performing models from extended candidates require retraining to store predictions; excluded from unregistered ensembles.
 res=[]
 for name,parts in ensembles.items():
  for k in ms.TEAMS:
   ids=pd.MultiIndex.from_arrays([np.full(len(q[k]),k),q[k].challenge_row_id.to_numpy(int)])
   score=np.mean([piv.loc[ids,c].to_numpy(float)for c in parts],axis=0)
   res.append(dict(team=k,model=name,**ms.scores(score,t[k],q[k])))
 df=pd.DataFrame(res);df.to_csv(OUT/'ensemble_per_team.csv',index=False)
 summaries=[]
 for scope,ex in [('nonself20',[5]),('processed18',[1,2,5])]:
  for name,group in df[~df.team.isin(ex)].groupby('model'):
   summaries.append(dict(scope=scope,model=name,n_teams=len(group),auc=group.auc.mean(),r_all=group.r_all.mean(),r_rare=group.r_rare.mean(),r_max=group.r_max.mean()))
 pd.DataFrame(summaries).to_csv(OUT/'ensemble_summary.csv',index=False)
 print('PAIR_BOOTSTRAP',pd.DataFrame(rows).query("scope=='processed18'").round(5).to_string(index=False),flush=True)
 print('ENSEMBLE_SUMMARY',pd.DataFrame(summaries).to_string(index=False),flush=True)
if __name__=='__main__':main()
