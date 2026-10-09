#!/usr/bin/env python3
"""Posthoc-family fixed outer-fold training sensitivity, not independent validation of chosen models."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[n]='1'
import json
import numpy as np,pandas as pd
from xgboost import XGBClassifier
import model_sweep as ms
from nested_selection import folds
from model_extra import PARAMS
CFG={n:(f,p) for n,f,p in PARAMS if n in ['lr_c10','xgb_flexible']}

def main():
 f,t,q=ms.load();rows=[]
 for scope,keys,n in [('all21_outer7',ms.TEAMS,7),('processed18_outer6',[x for x in ms.TEAMS if x not in [1,2,5]],6)]:
  for fold,(tr,te) in enumerate(folds(keys,n)):
   for rep,cols in ms.REPR.items():
    for name,(fam,params) in CFG.items():
     if fam=='lr':model=ms.estimator(fam,params)
     else:model=XGBClassifier(tree_method='hist',n_jobs=1,random_state=ms.SEED,eval_metric='logloss',**params)
     model.fit(np.vstack([f[j][cols].to_numpy(float)for j in tr]),np.concatenate([t[j].label.eq('member').to_numpy(int)for j in tr]))
     for k in te:
      p=model.predict_proba(f[k][cols].to_numpy(float))[:,1]
      rows.append(dict(scope=scope,outer_fold=fold,team=k,features=rep,model=name,**ms.scores(p,t[k],q[k])))
 d=pd.DataFrame(rows);d.to_csv(ms.OUT/'outer_extensions.csv',index=False)
 groups=[]
 for scope,sub in d.groupby('scope'):
  for label, ss in ([('nonself20',sub[sub.team.ne(5)]),('processed18',sub[~sub.team.isin([1,2,5])])]if scope=='all21_outer7' else [('processed18',sub)]):
   for (rep,m),g in ss.groupby(['features','model']):
    groups.append(dict(outer_scope=scope,group=label,features=rep,model=m,teams=len(g),r_all=g.r_all.mean(),auc=g.auc.mean(),r_rare=g.r_rare.mean()))
 z=pd.DataFrame(groups);z.to_csv(ms.OUT/'outer_extensions_summary.csv',index=False)
 print(z.to_string(index=False),flush=True)
if __name__=='__main__':main()
