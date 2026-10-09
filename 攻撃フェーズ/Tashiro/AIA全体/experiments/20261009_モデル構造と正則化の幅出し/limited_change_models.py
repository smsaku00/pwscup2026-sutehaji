#!/usr/bin/env python3
"""Model breadth with a frozen LIMITED-CHANGE policy (cap 10, expected-gain > .02, 9.5->11)."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import time,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
import model_sweep as ms
import prev_calibrated_policy as cp
T=ms.TEAMS
# Same baseline rows for all decision models; model differences cannot alter the incumbent mask.
SPECS=[
 ('reference','c_bg_local','lr_c004'),
 ('LR_C2','c_bg','lr_c2'),
 ('LR_C10','c_bg','lr_c10'),
 ('LR_local_C02','c_bg_local','lr_c02'),
 ('HGB_shallow','c_bg','hgb_shallow'),
 ('XGB_flexible_local','c_bg_local','xgb_flexible'),
]
CFG={n:(fam,p) for n,fam,p in [*ms.MODES]}
CFG['lr_c10']=('lr',dict(C=10.))
CFG['xgb_flexible']=('xgb',dict(n_estimators=120,max_depth=4,min_child_weight=35,reg_lambda=10,learning_rate=.05,subsample=.85,colsample_bytree=.85))


def make(name):
 fam,par=CFG[name]
 return (ms.estimator(fam,par) if fam!='xgb' else XGBClassifier(tree_method='hist',n_jobs=1,random_state=ms.SEED,eval_metric='logloss',**par))


def inner_predict(k,rep,name,features,truth,q,nsplit=4):
 keys=[j for j in T if j!=k]
 X=np.vstack([features[j][ms.REPR[rep]].to_numpy(float) for j in keys]);
 y=np.concatenate([truth[j].label.eq('member').to_numpy(int) for j in keys])
 gids=np.concatenate([np.full(len(truth[j]),j,dtype=int) for j in keys]);
 ans=np.full(len(y),np.nan)
 for tr,va in GroupKFold(n_splits=nsplit).split(X,y,groups=gids):
  ans[va]=make(name).fit(X[tr],y[tr]).predict_proba(X[va])[:,1]
 assert np.isfinite(ans).all()
 return pd.DataFrame(dict(team=gids,score=ans,member=y,
       h9=np.concatenate([(abs(truth[j].time.to_numpy(float)-9.5)<=.5).astype(int)for j in keys]),
       h0=np.concatenate([(truth[j].time.to_numpy(float)<=.5).astype(int) for j in keys]),
       sex=np.concatenate([q[j].sex.to_numpy(int) for j in keys]),
       row_id=np.concatenate([q[j].challenge_row_id.to_numpy(int) for j in keys])))


def main():
 t0=time.time();f,t,q=ms.load()
 r=ms.OUT; r.mkdir(exist_ok=True)
 old=pd.read_csv(r/'public_predictions.csv').set_index(['team','row_id','features','model'])
 all_rows=[];detail=[]
 for k in T:
  prior_inner=inner_predict(k,'c_bg_local','lr_c004',f,t,q,nsplit=4)
  rates=cp.expected_hit_rates(prior_inner)
  lambda_r=cp.expected_lambda(prior_inner)
  if not(0<lambda_r<2):raise ValueError('Invalid lambda')
  keys=[(k,int(i),'c_bg_local','lr_c004') for i in q[k].challenge_row_id]
  incumbent=old.loc[keys,'p_member'].to_numpy(float)
  baseline=cp.top75(q[k],incumbent)
  basevalue=np.where(baseline,9.5,0.)
  base_metric=ms.prior.metrics(basevalue,t[k])
  for label,rep,name in SPECS:
   inner=prior_inner if label=='reference' else inner_predict(k,rep,name,f,t,q,nsplit=4)
   c=cp.fit_calibrator(inner,'platt')
   if label=='reference':p=incumbent
   else:
    other=[j for j in T if j!=k]
    X=np.vstack([f[j][ms.REPR[rep]].to_numpy(float) for j in other]); y=np.concatenate([t[j].label.eq('member').to_numpy(int) for j in other])
    p=make(name).fit(X,y).predict_proba(f[k][ms.REPR[rep]].to_numpy(float))[:,1]
   pcal=np.clip(c(p),.001,.999)
   gain=-(pcal*rates['m9']-lambda_r*(1-pcal)*rates['c9'])
   eligible=np.flatnonzero(baseline & (gain>0.02))
   order=np.lexsort((q[k].challenge_row_id.to_numpy(int)[eligible],-gain[eligible]))
   chosen=eligible[order[:10]]
   edited=basevalue.copy();edited[chosen]=11.
   b=ms.prior.metrics(edited,t[k]);y=t[k].label.eq('member').to_numpy(bool)
   b0=np.abs(basevalue-t[k].time.to_numpy(float))<=.5
   b1=np.abs(edited-t[k].time.to_numpy(float))<=.5
   all_rows.append(dict(team=k,decision_model=label,feature_pack=rep,model=name,changed=len(chosen),available=len(eligible),mean_calibrated_member_prob=float(pcal.mean()),baseline_r_all=base_metric['r_all'],r_all=b['r_all'],r_rare=b['r_rare'],r_max=b['r_max'],rare_extra=b['r_max']-b['r_all'],member_hits=b['hit_member'],control_hits=b['hit_control'],member_lost=int(np.sum(y & b0 & ~b1)),control_hit_removed=int(np.sum((~y) & b0 & ~b1))))
   for idx in chosen:detail.append(dict(team=k,decision_model=label,row_id=int(q[k].challenge_row_id.iloc[idx]),calibrated_p=float(pcal[idx]),expected_gain=float(gain[idx]),member_bool_for_posthoc_only=int(y[idx]),pre_hit=bool(b0[idx]),post_hit=bool(b1[idx])))
  print('LIMITED_DONE',k,'elapsed',round(time.time()-t0,1),flush=True)
 D=pd.DataFrame(all_rows);D.to_csv(r/'limited_model_per_team.csv',index=False)
 pd.DataFrame(detail).to_csv(r/'limited_model_edited_rows.csv',index=False)
 res=[]
 for label,exclude in [('nonself20',[5]),('processed18',[1,2,5])]:
  subset=D[~D.team.isin(exclude)]
  for name,g in subset.groupby('decision_model'):
   res.append(dict(scope=label,decision_model=name,teams=len(g),baseline_r_all=g.baseline_r_all.mean(),r_all=g.r_all.mean(),delta=(g.r_all-g.baseline_r_all).mean(),r_rare=g.r_rare.mean(),r_max=g.r_max.mean(),rare_extra=g.rare_extra.mean(),changed=g.changed.mean(),member_lost=g.member_lost.sum(),control_hit_removed=g.control_hit_removed.sum(),wins=int((g.r_all>g.baseline_r_all+1e-12).sum()),losses=int((g.r_all<g.baseline_r_all-1e-12).sum()),ties=int((abs(g.r_all-g.baseline_r_all)<=1e-12).sum())))
 pd.DataFrame(res).to_csv(r/'limited_model_summary.csv',index=False)
 (r/'limited_protocol.json').write_text(json.dumps({'fixed_incumbent':'c_bg_local logistic C=.04 LOTO sex top75 9.5/0','action':'at most 10 incumbent selected rows to 11 years with leave-one-target-out Platt probability and other-cohort rates/gain threshold .02','inner_training_teams_excludes_target':True,'inner_folds':4,'target_truth_only_for_posthoc_scoring':True,'baseline_training_r_lambda_fixed_to_reference':True,'decision_models':SPECS,'not_prior_registered_candidates':True},ensure_ascii=False,indent=2)+'\n')
 print('LIMITED_SUMMARY\n',pd.DataFrame(res).to_string(index=False),flush=True)
if __name__=='__main__':main()
