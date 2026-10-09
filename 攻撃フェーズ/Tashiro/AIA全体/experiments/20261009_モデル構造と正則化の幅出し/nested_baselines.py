#!/usr/bin/env python3
"""Fixed-model comparators under exactly the same grouped 7/6-fold outer training as nested selection."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[n]='1'
import numpy as np,pandas as pd
import model_sweep as ms
from nested_selection import folds
from pathlib import Path
CONFIG=[('c_bg','lr_c004'),('c_bg_local','lr_c004'),('c_bg','lr_c2'),('c_bg_local','lr_c2'),('c_bg','hgb_shallow')]
CFG={n:(f,p)for n,f,p in ms.MODES}

def main():
 feats,t,q=ms.load();rows=[]
 for label,ids,nfold in [('all21_outer7',ms.TEAMS,7),('processed18_outer6',[k for k in ms.TEAMS if k not in (1,2,5)],6)]:
  for fold,(train,test) in enumerate(folds(ids,nfold)):
   for rep,name in CONFIG:
    fam,params=CFG[name]; cols=ms.REPR[rep]
    model=ms.estimator(fam,params).fit(np.vstack([feats[j][cols].to_numpy(float)for j in train]),np.concatenate([t[j].label.eq('member').to_numpy(int)for j in train]))
    for k in test:
     p=model.predict_proba(feats[k][cols].to_numpy(float))[:,1]
     rows.append(dict(scope=label,fold=fold,team=k,features=rep,model=name,**ms.scores(p,t[k],q[k])))
   for k in test:
    score=feats[k]['col2'].to_numpy(float)
    mask=ms.prior.top75(q[k],score)
    pred=np.where(np.isin(np.arange(len(q[k])),list(mask)),9.5,0.)
    m=ms.prior.metrics(pred,t[k]); y=t[k].label.eq('member').to_numpy(int); rare=t[k].is_rare.eq(1).to_numpy(bool)
    from sklearn.metrics import roc_auc_score
    rows.append(dict(scope=label,fold=fold,team=k,features='col2_precomputed',model='col2',auc=roc_auc_score(y,score),auc_rare=roc_auc_score(y[rare],score[rare]),r_all=m['r_all'],r_rare=m['r_rare'],r_max=m['r_max'],rare_extra=m['r_max']-m['r_all'],hit_member=m['hit_member'],hit_control=m['hit_control']))
 d=pd.DataFrame(rows);d.to_csv(ms.OUT/'nested_fixed_baselines.csv',index=False)
 summary=[]
 for scope,subset in d.groupby('scope'):
  for label,ss in ([('nonself20',subset[subset.team.ne(5)]),('processed18',subset[~subset.team.isin([1,2,5])])] if scope=='all21_outer7' else [('processed18',subset)]):
   for (rep,name),g in ss.groupby(['features','model']):
    summary.append(dict(scope=scope,group=label,features=rep,model=name,teams=len(g),r_all=g.r_all.mean(),r_rare=g.r_rare.mean(),auc=g.auc.mean()))
 z=pd.DataFrame(summary);z.to_csv(ms.OUT/'nested_fixed_summary.csv',index=False)
 print(z.to_string(index=False),flush=True)
if __name__=='__main__':main()
