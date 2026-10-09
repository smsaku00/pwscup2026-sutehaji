#!/usr/bin/env python3
"""Exploratory extension: weak L2 regularization and XGBoost; NOT eligible for frozen nested selector."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[n]='1'
import time,json
from pathlib import Path
import numpy as np,pandas as pd
from xgboost import XGBClassifier
import model_sweep as ms
PARAMS=[
 ('lr_c10','lr',dict(C=10.)),
 ('lr_c100','lr',dict(C=100.)),
 ('xgb_conservative','xgb',dict(n_estimators=90,max_depth=2,min_child_weight=85,reg_lambda=30,learning_rate=.05,subsample=.8,colsample_bytree=.8)),
 ('xgb_flexible','xgb',dict(n_estimators=120,max_depth=4,min_child_weight=35,reg_lambda=10,learning_rate=.05,subsample=.85,colsample_bytree=.85)),
]

def main():
 f,t,q=ms.load();y={k:t[k].label.eq('member').to_numpy(int) for k in ms.TEAMS};results=[]
 for rep,cols in ms.REPR.items():
  for name,fam,par in PARAMS:
   start=time.time()
   for k in ms.TEAMS:
    keys=[j for j in ms.TEAMS if j!=k]
    X=np.vstack([f[j][cols].to_numpy(float) for j in keys]); Y=np.concatenate([y[j] for j in keys])
    m=ms.estimator(fam,par) if fam!='xgb' else XGBClassifier(tree_method='hist',n_jobs=1,random_state=ms.SEED,eval_metric='logloss',**par)
    m.fit(X,Y)
    p=m.predict_proba(f[k][cols].to_numpy(float))[:,1]
    results.append(dict(team=k,features=rep,model=name,**ms.scores(p,t[k],q[k])))
   print('EXTRA_DONE',rep,name,round(time.time()-start,1),flush=True)
 pd.DataFrame(results).to_csv(ms.OUT/'per_team_extra.csv',index=False)
 ms.summary(pd.DataFrame(results)).to_csv(ms.OUT/'summary_extra.csv',index=False)
 (ms.OUT/'extension_protocol.json').write_text(json.dumps({'stage':'post-initial grid extension','nested_selector_eligible':False,'models':PARAMS,'reason':'check whether larger logistic C or a different histogram booster changes initial trend','target_label_leakage':False,'selection_not_tested':True},ensure_ascii=False,indent=2,default=str)+'\n')
if __name__=='__main__':main()
