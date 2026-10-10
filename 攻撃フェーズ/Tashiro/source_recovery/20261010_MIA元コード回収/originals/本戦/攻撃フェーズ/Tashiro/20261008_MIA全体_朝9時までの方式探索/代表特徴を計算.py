"""代表12系統を別背景・本戦で計算する。会員ラベルは読み込まない。"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logit
from scipy.stats import rankdata
import 共通 as u
from 距離と分布の探索 import transform,nearest,distances,mixed_logs,gaussian,binned
from 公開データ分類器の探索 import encode,models as source_models
from 密度比の属性と乱数を確認 import one_score,SEEDS
from 別系統特徴の学習 import SIGNALS,GEN

def conditional_rank(value,data,team,county=False):
 n=len(data['b']);v=value[:n]
 if county:
  ii=data['controls'][team];null=value[ii];controls=data['query'].iloc[ii]
  local,count=u.cal.group_stats(v,data['b'],null,controls,['prefecture','sex'])
  transformed=u.cal.blend(local,count,u.cal.empirical_cdf(v,null),500)
  v=.999*transformed+.001*rankdata(v,method='average')/n
 return u.rank(v,data['normal'])

def calculate(data,a,label='full'):
 q=data['query'];b=data['b'];teams=data['teams'];bank={name:{} for _,name in SIGNALS+GEN}
 for filename in ['代表特徴を計算.py','距離と分布の探索.py','公開データ分類器の探索.py','密度比の属性と乱数を確認.py','直接密度比の探索.py','別系統特徴の学習.py']:
  p=u.HERE/filename;u.base.INPUT_HASHES[str(p.resolve())]=u.sha(p)
 ar7,x7=transform(a,[a,q],u.NUM7,'robust')
 da7=cdist(x7,ar7,'sqeuclidean');dn7=da7.copy()
 for col in u.CAT7:dn7+=4*(q[col].to_numpy()[:,None]!=a[col].to_numpy()[None,:])
 na7=nearest(dn7,3).mean(axis=1);del dn7
 h=(4/(len(u.NUM7)+2)/len(a))**(1/(len(u.NUM7)+4))
 log_a=mixed_logs(da7,q,a,u.CAT7,a,h);del da7
 ar13,x13=transform(a,[a,b],u.NUM13,'robust')
 near_a13=nearest(distances(x13,ar13,b,a,u.CAT13,1),10).mean(axis=1)
 for index,(t,c) in enumerate(data['cs'].items(),1):
  _,_,cr7=transform(a,[a,q,c],u.NUM7,'robust')
  dc7=cdist(x7,cr7,'sqeuclidean');nn7=dc7.copy()
  for col in u.CAT7:nn7+=4*(q[col].to_numpy()[:,None]!=c[col].to_numpy()[None,:])
  near_c7=nearest(nn7,3).mean(axis=1);del nn7
  kde=mixed_logs(dc7,q,c,u.CAT7,a,h)-log_a
  g7=gaussian(a,q,c,u.NUM7,u.CAT7)['gaussian_shrink0']
  ga13=gaussian(a,b,c,u.NUM13,u.CAT13)['gaussian_shrink0.5']
  hist13=binned(a,b,c,u.NUM13,u.CAT13,10,100)['marginal']
  _,_,cr13=transform(a,[a,b,c],u.NUM13,'robust')
  nc13=nearest(distances(x13,cr13,b,c,u.CAT13,1),10)
  draws=[one_score(a,c,b,u.NUM13,u.CAT13,s+t) for s in SEEDS]
  raw={
   'q7_cat4_nnratio_k3__raw__cross0':(np.log(na7+1e-12)-np.log(near_c7+1e-12),False),
   'q7_robust_kde_ratio_h1__county500__cross0':(kde,True),
   'q7_gaussian_shrink0__county500__cross0':(g7,True),
   'q13_gaussian_shrink0.5__raw__cross0':(ga13,False),
   'q13_hist10_s100_marginal__raw__cross0':(hist13,False),
   'q13_robust_dcr_k1__raw__cross0':(-np.log(nc13[:,0]+1e-12),False),
   'q13_robust_nnratio_k10__raw__cross0':(np.log(near_a13+1e-12)-np.log(nc13.mean(axis=1)+1e-12),False),
   'q13_robust_ulsif_centers100_h1_a0.1_lambda1__raw__cross0':(draws[0],False),
   'all13_raw_mean__raw__cross0':(np.mean(draws,axis=0),False)}
  # 初回と同じ公開行分類器の設定。source-labelsのみで学習する。
  for name,nums,kind,query,model_name,county in [
   ('q7_robust_extra_leaf1__county500__cross0',u.NUM7,'robust',q,'extra_leaf1',True),
   ('q7_rank_lr_C0.2__raw__cross0',u.NUM7,'rank',q,'lr_C0.2',False),
   ('q9_robust_ocsvmC_gamma0.02__raw__cross0',u.NUM9,'robust',b,'ocsvmC_gamma0.02',False)]:
   ar,cr,x=encode(a,c,query,nums,u.CAT7,kind)
   model=dict((n,m) for n,m,_ in source_models(20261008))[model_name]
   if model_name.startswith('ocsvm'):model.fit(cr);score=model.score_samples(x)
   else:
    model.fit(np.vstack([ar,cr]),np.r_[np.zeros(len(ar),dtype=int),np.ones(len(cr),dtype=int)])
    score=logit(np.clip(model.predict_proba(x)[:,1],1e-8,1-1e-8))
   raw[name]=(score,county)
  gen=u.base.core(a,q,c,u.NUM7,(10,))[10]
  raw['gen7_reference__raw__cross0']=(gen,False)
  raw['gen7_reference__county500__cross0.1']=(gen,True)
  assert set(raw)==set(bank)
  for name,(score,county) in raw.items():bank[name][t]=conditional_rank(score,data,t,county)
  u.log(stage='representative_features',condition=label,completed_targets=index,targets=len(teams))
 result=np.stack([np.column_stack([bank[name][t] for t in teams]) for _,name in SIGNALS+GEN],axis=2)
 # 現行対照だけは他対象補正0.1を加える。その他は学習用の元の順位。
 reference=result[:,:,-1].copy()
 med=np.column_stack([np.median(np.delete(reference,j,axis=1),axis=1) for j in range(len(teams))])
 result[:,:,-1]-=.1*med
 assert np.isfinite(result).all()
 return result

def learning_features(raw,median_excluded=()):
 lower=.5/875;n=len(SIGNALS);valid=np.array([j for j in range(raw.shape[1]) if j not in median_excluded])
 own=logit(np.clip(raw[:,:,:n],lower,1-lower))
 median=np.stack([np.column_stack([np.median(raw[:,valid[valid!=j],k],axis=1) for j in range(raw.shape[1])]) for k in range(n)],axis=2)
 return np.concatenate([own,logit(np.clip(median,lower,1-lower))],axis=2)
