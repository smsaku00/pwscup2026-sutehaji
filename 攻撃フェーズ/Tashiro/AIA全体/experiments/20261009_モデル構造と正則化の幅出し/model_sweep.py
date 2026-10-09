#!/usr/bin/env python3
"""Frozen candidate-family model sweep, using official prelim Q/C/BG and held-out target truth."""
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[_n]='1'
import sys,json,time,hashlib,argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
sys.path.insert(0,str(Path(__file__).resolve().parent/'prior'))
import experiment as prior

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'cache'
OUT=ROOT/'model_results'
OUT.mkdir(exist_ok=True)
SEED=20261009
TEAMS=prior.TEAMS
MODES=[
 ('lr_c0002','lr',dict(C=.002)),
 ('lr_c001','lr',dict(C=.01)),
 ('lr_c004','lr',dict(C=.04)),
 ('lr_c02','lr',dict(C=.2)),
 ('lr_c2','lr',dict(C=2.)),
 ('hgb_shallow','hgb',dict(max_iter=75,max_leaf_nodes=6,min_samples_leaf=150,l2_regularization=30,learning_rate=.04)),
 ('hgb_original','hgb',dict(max_iter=100,max_leaf_nodes=10,min_samples_leaf=85,l2_regularization=20,learning_rate=.045)),
 ('hgb_medium','hgb',dict(max_iter=125,max_leaf_nodes=15,min_samples_leaf=65,l2_regularization=12,learning_rate=.045)),
 ('hgb_fine','hgb',dict(max_iter=145,max_leaf_nodes=28,min_samples_leaf=35,l2_regularization=5,learning_rate=.045)),
 ('rf_conservative','rf',dict(n_estimators=40,min_samples_leaf=45,max_depth=8,max_features=.7)),
 ('extra_trees','extra',dict(n_estimators=50,min_samples_leaf=25,max_depth=12,max_features=.7)),
 ('lgb_conservative','lgb',dict(n_estimators=100,num_leaves=7,max_depth=5,min_child_samples=140,reg_lambda=40,learning_rate=.045)),
 ('lgb_flexible','lgb',dict(n_estimators=140,num_leaves=18,max_depth=7,min_child_samples=55,reg_lambda=12,learning_rate=.04)),
 ('catboost_regular','cat',dict(iterations=120,depth=4,l2_leaf_reg=20,learning_rate=.055)),
]
MEMBER=['member_near4','member_knn4','member_near5','member_partial2','rank_member_near4','rank_member_near5','rank_member_partial2']+[f'c_{n.lower()}' for n in prior.CLIN]
BG=['bg_near4','bg_knn4','bg_near5','rank_bg_near4','rank_bg_knn4']+[f'bg_{n.lower()}' for n in prior.CLIN]
CONTRAST=['contrast_near4','contrast_knn4','contrast_near5','col4','col2','col1','col4_min','rank_contrast_near4','rank_col4','rank_col2','rank_col1']+[f'ratio_{n.lower()}' for n in prior.CLIN]
BASE=MEMBER+BG+CONTRAST
LOCAL=['knn2_k1','kde4_half']
REPR={'c_bg':BASE,'c_bg_local':BASE+LOCAL}

def estimator(family,p):
 p=dict(p)
 if family=='lr':return make_pipeline(StandardScaler(),LogisticRegression(solver='lbfgs',max_iter=1500,**p))
 if family=='hgb':return HistGradientBoostingClassifier(random_state=SEED,**p)
 if family=='rf':return RandomForestClassifier(random_state=SEED,n_jobs=1,**p)
 if family=='extra':return ExtraTreesClassifier(random_state=SEED,n_jobs=1,**p)
 if family=='lgb':return LGBMClassifier(random_state=SEED,verbosity=-1,n_jobs=1,**p)
 if family=='cat':return CatBoostClassifier(random_seed=SEED,verbose=False,thread_count=1,allow_writing_files=False,**p)
 raise ValueError(family)

def load():
 f={k:pd.read_pickle(CACHE/f'features_{k}.pkl') for k in TEAMS}
 t={k:pd.read_pickle(CACHE/f'truth_{k}.pkl') for k in TEAMS}
 q={k:pd.read_pickle(CACHE/f'q_{k}.pkl') for k in TEAMS}
 density=pd.read_csv(ROOT/'density/results/density_public_scores.csv.gz').set_index(['team','row_id'])
 for k in TEAMS:
  if not np.array_equal(f[k].row_id.to_numpy(int),q[k].challenge_row_id.to_numpy(int)):
   raise ValueError('features/Q mismatch')
  idx=pd.MultiIndex.from_arrays([np.full(len(q[k]),k),q[k].challenge_row_id.to_numpy(int)])
  f[k]['knn2_k1']=density.loc[idx,'knn2_k1'].to_numpy(float)
  f[k]['kde4_half']=density.loc[idx,'kde4_half'].to_numpy(float)
 return f,t,q

def scores(p,t,q):
 y=t.label.eq('member').to_numpy(int)
 r=t.is_rare.eq(1).to_numpy(bool)
 mask=prior.top75(q,p)
 pred=np.where(np.isin(np.arange(len(q)),list(mask)),9.5,0.)
 z=prior.metrics(pred,t)
 pred11=np.where(np.isin(np.arange(len(q)),list(mask)),9.5,11.)
 z11=prior.metrics(pred11,t)
 return dict(auc=float(roc_auc_score(y,p)),auc_rare=float(roc_auc_score(y[r],p[r])),brier=float(brier_score_loss(y,p)),
             r_all=z['r_all'],r_rare=z['r_rare'],r_max=z['r_max'],rare_extra=z['r_max']-z['r_all'],
             hit_member=z['hit_member'],hit_control=z['hit_control'],hit_member_rare=z['hit_member_rare'],hit_control_rare=z['hit_control_rare'],
             r_all11=z11['r_all'],r_max11=z11['r_max'])

def summary(df):
 rows=[]
 for scope,exc in [('nonself20',{5}),('processed18',{1,2,5})]:
  for (rep,mod),part in df[~df.team.isin(exc)].groupby(['features','model']):
   r=part.mean(numeric_only=True)
   rows.append(dict(scope=scope,features=rep,model=mod,n_teams=len(part),**{key:float(r[key]) for key in ('auc','auc_rare','brier','r_all','r_rare','r_max','rare_extra','hit_member','hit_control','hit_member_rare','hit_control_rare','r_all11','r_max11')}))
 return pd.DataFrame(rows)

def run():
 t0=time.time();f,t,q=load()
 assert len(f)==21
 prior_preds=pd.read_csv(ROOT/'prior/output/public_scores.csv.gz').set_index(['team','row_id'])
 prior_den=pd.read_csv(ROOT/'density/results/augment_public_predictions.csv.gz')
 den_piv=prior_den.pivot(index=['team','row_id'],columns='model',values='score')
 for repr,refcol in [('c_bg','c_bg'),('c_bg_local','c_bg_plus_local')]:
  # Baseline reproduction with frozen published exact previous predictions, no labels used.
  refkey='lr_c004'
  columns=REPR[repr]
  errs=[]
  for k in TEAMS:
   Xtr=np.vstack([f[j][columns].to_numpy(float) for j in TEAMS if j!=k]);Y=np.concatenate([t[j].label.eq('member').to_numpy(int) for j in TEAMS if j!=k])
   fit=estimator('lr',dict(C=.04)).fit(Xtr,Y)
   p=fit.predict_proba(f[k][columns].to_numpy(float))[:,1]
   indx=pd.MultiIndex.from_arrays([np.full(len(q[k]),k),q[k].challenge_row_id.to_numpy(int)])
   expected=(prior_preds.loc[indx,refcol].to_numpy(float) if repr=='c_bg' else den_piv.loc[indx,refcol].to_numpy(float))
   errs.append(float(np.max(np.abs(p-expected))))
  print('BASELINE_VERIFY',repr,'maxerr',max(errs),flush=True)
  if max(errs)>2e-10:raise ValueError('Baseline not reproduced')
 whole=[];preds=[]
 for repr,cols in REPR.items():
  X={k:f[k][cols].to_numpy(float) for k in TEAMS}
  Y={k:t[k].label.eq('member').to_numpy(int) for k in TEAMS}
  for name,fam,param in MODES:
   elapsed=time.time()-t0
   print('START',repr,name,'elapsed',round(elapsed,1),flush=True)
   for k in TEAMS:
    train=[j for j in TEAMS if j!=k]
    Xt=np.vstack([X[j] for j in train]); yt=np.concatenate([Y[j] for j in train])
    model=estimator(fam,param)
    model.fit(Xt,yt)
    p=model.predict_proba(X[k])[:,1]
    assert np.isfinite(p).all() and np.min(p)>=0 and np.max(p)<=1
    whole.append(dict(team=k,features=repr,model=name,**scores(p,t[k],q[k])))
    preds.append(pd.DataFrame({'team':k,'row_id':q[k].challenge_row_id,'features':repr,'model':name,'p_member':p}))
   df=pd.DataFrame(whole);df.to_csv(OUT/'per_team_running.csv',index=False)
   print('DONE',repr,name,'total_elapsed',round(time.time()-t0,1),flush=True)
 df=pd.DataFrame(whole); df.to_csv(OUT/'per_team.csv',index=False)
 sm=summary(df);sm.to_csv(OUT/'summary.csv',index=False)
 ps=pd.concat(preds,ignore_index=True)
 ps.to_csv(OUT/'public_predictions.csv',index=False)
 (OUT/'protocol.json').write_text(json.dumps({'seed':SEED,'source':'PWS CUP 2026 official prelim','evaluation':'21 leave-one-target-cohort-out models; 20 nonself and 18 processed','source_before_results':True,'model_specifications':[{'model':name,'family':fam,'params':p} for name,fam,p in MODES], 'feature_specifications':REPR, 'policy':'sex top ceil75% gets 9.5y, others 0y; alternate 11y fixed','selection':'No within-cohort oracle choices; across-model winner is descriptive, requires nested selection', 'rows':int(len(ps)),'time_seconds':round(time.time()-t0,1)},ensure_ascii=False,indent=2)+'\n')
 print('FINISH',round(time.time()-t0,1),flush=True)
 print(sm[sm.scope.eq('processed18')].sort_values('r_all',ascending=False)[['features','model','r_all','auc','r_rare']].head(14).to_string(index=False),flush=True)

if __name__=='__main__':run()
