#!/usr/bin/env python3
"""True nested cohort CV: candidate selection never reads outer test cohort labels."""
import os
for _n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
 os.environ[_n]='1'
import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import model_sweep as ms
import experiment as ignore_import_source # resolved in prior path inserted by ms

FIXED=['lr_c001','lr_c004','lr_c02','hgb_shallow','hgb_original','hgb_fine','lgb_conservative']
CANDS=[(rep,name,fam,params) for rep in ms.REPR for name,fam,params in ms.MODES if name in FIXED]
assert len(CANDS)==14


def folds(keys,n):
 vals=np.array(keys)
 return [(vals[t].tolist(),vals[v].tolist()) for t,v in GroupKFold(n_splits=n).split(vals,groups=vals)]

def nested(keys,outer_n,label):
 t0=time.time();f,t,q=ms.load()
 X={rep:{k:f[k][cols].to_numpy(float) for k in keys} for rep,cols in ms.REPR.items()}
 Y={k:t[k].label.eq('member').to_numpy(int) for k in keys}
 all_rows=[];chosen=[];selection=[]
 for outer_i,(outer_train,outer_test) in enumerate(folds(keys,outer_n)):
  inner_scores=[]
  for rep,name,fam,params in CANDS:
   inner_rows=[]
   for inner_i,(inner_train,inner_test) in enumerate(folds(outer_train,3)):
    Xt=np.vstack([X[rep][k] for k in inner_train]);yt=np.concatenate([Y[k] for k in inner_train]); model=ms.estimator(fam,params).fit(Xt,yt)
    for k in inner_test:
     p=model.predict_proba(X[rep][k])[:,1]; v=ms.scores(p,t[k],q[k])
     inner_rows.append(dict(team=k,auc=v['auc'],r_all=v['r_all'],r_max=v['r_max']))
   d=pd.DataFrame(inner_rows)
   # Only other attack targets inform meta selection; self5 not evaluated.
   d=d[d.team.ne(5)]
   inner_scores.append(dict(features=rep,model=name,family=fam,params=params,inner_auc=d.auc.mean(),inner_r_all=d.r_all.mean(),inner_r_max=d.r_max.mean()))
  ss=pd.DataFrame(inner_scores);ss['outer_fold']=outer_i
  for obj in ['inner_auc','inner_r_all']:
   # Pre-declared tie ordering in CANDS (stable mergesort); no outer-label choice.
   best=ss.sort_values(obj,ascending=False,kind='stable').iloc[0]
   rep,name,fam,params=best.features,best.model,best.family,best.params
   fit=ms.estimator(fam,params).fit(np.vstack([X[rep][k] for k in outer_train]),np.concatenate([Y[k] for k in outer_train]))
   for k in outer_test:
    p=fit.predict_proba(X[rep][k])[:,1]
    all_rows.append(dict(scope=label,outer_fold=outer_i,team=k,selection=obj,selected_features=rep,selected_model=name,**ms.scores(p,t[k],q[k])))
   chosen.append(dict(scope=label,fold=outer_i,objective=obj,selected_features=rep,selected_model=name,selector_training_teams=';'.join(map(str,outer_train)),selector_test_teams=';'.join(map(str,outer_test)),train_inner_value=float(best[obj])))
  for _,x in ss.iterrows():selection.append(dict(scope=label,outer_fold=outer_i,features=x.features,model=x.model,inner_auc=x.inner_auc,inner_r_all=x.inner_r_all,inner_r_max=x.inner_r_max))
  print('NESTED_DONE',label,outer_i,'outer_test',outer_test,'chosen',chosen[-2:],round(time.time()-t0,1),flush=True)
 return all_rows,chosen,selection


def main():
 s=ROOT/'model_results';s.mkdir(exist_ok=True)
 rows,selection,inner=nested(ms.TEAMS,7,'all21_outer7')
 # Add processed-only second strictly separated nested experiment, not trained on cohorts 1,2,5.
 rr,cc,ii=nested([x for x in ms.TEAMS if x not in (1,2,5)],6,'processed18_outer6')
 rows+=rr;selection+=cc;inner+=ii
 d=pd.DataFrame(rows);d.to_csv(s/'nested_per_team.csv',index=False)
 pd.DataFrame(selection).to_csv(s/'nested_selected_models.csv',index=False)
 pd.DataFrame(inner).to_csv(s/'nested_inner_scores.csv',index=False)
 sm=[]
 for scope,part in d.groupby('scope'):
  for objective,sub in part.groupby('selection'):
   for label,ss in ([('20target',sub[sub.team.ne(5)]),('18target',sub[~sub.team.isin([1,2,5])])] if scope=='all21_outer7' else [('18target',sub)]):
    r=ss.mean(numeric_only=True)
    sm.append(dict(outer_scope=scope,group=label,objective=objective,teams=len(ss),auc=r.auc,auc_rare=r.auc_rare,brier=r.brier,r_all=r.r_all,r_rare=r.r_rare,r_max=r.r_max,rare_extra=r.rare_extra,hit_member=r.hit_member,hit_control=r.hit_control))
 pd.DataFrame(sm).to_csv(s/'nested_summary.csv',index=False)
 (s/'nested_protocol.json').write_text(json.dumps({'source':'official preliminary 21 cohorts', 'seed':ms.SEED,'all21_outer_folds':7,'processed18_outer_folds':6,'inner_folds':3,'candidate_count':len(CANDS),'candidates':[f'{r}:{n}'for r,n,_,_ in CANDS],'selection_objectives':['per-cohort mean r_all','mean AUC'],'test_truth_used_for_selection':False,'nested_is_true_separation':True,'policy':'unchanged 75% 9.5/0, alternative 9.5/11 diagnostics only'},ensure_ascii=False,indent=2)+'\n')
 print('NESTED_SUMMARY\n',pd.DataFrame(sm).to_string(index=False),flush=True)
if __name__=='__main__':main()
