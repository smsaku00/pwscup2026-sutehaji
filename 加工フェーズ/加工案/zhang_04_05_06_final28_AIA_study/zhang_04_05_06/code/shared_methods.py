"""Shared numerical helpers for the three final-cohort schemes. No preliminary data used."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import pandas as pd
from scipy.stats import norm,rankdata
from lifelines import CoxPHFitter
CONT=['age','BMI','SBP','TG','HDL','ALT','FPG']
def public_outliers(b,s):
 r=s['rare_detector'];x=b[r['columns']].to_numpy(float)-np.asarray(r['mu'])
 return np.einsum('ij,jk,ik->i',x,np.linalg.pinv(np.asarray(r['sigma'])),x)>=r['band']['lo']**2

COX=['age','sex','BMI','SBP','log_TG','HDL','log_ALT','FPG','smoking']
def xf(b):
 x=b.copy();x['log_TG']=np.log(x.TG);x['log_ALT']=np.log(x.ALT);return x[COX].astype(float)
def fit(b):
 x=xf(b);x['time']=b.time.to_numpy();x['onset']=b.onset.to_numpy();m=CoxPHFitter();m.fit(x,'time','onset');return m

def covariates(b,a,s,rng,shrink):
 kind="strata_calibrated_main"
 # 通常層は本番A_bg、外れ値層はBから周辺分布と正規順位の相関を学び、独立乱数から新しい組み合わせを生成。
 # whiteningは乱数の偶然の平均・共分散のずれを除く操作。
 # 通常層の順位相関を12回まで合わせる。shrinkで相関を0/5/10%弱める。
 cols=CONT+['sex','smoking'];n=len(b);out=b.copy();flags=public_outliers(b,s)
 group=flags.astype(int)
 for label in np.unique(group):
  idx=np.flatnonzero(group==label);ng=len(idx);source=a.loc[~public_outliers(a,s)] if label==0 else b.iloc[idx];x=source[cols].to_numpy(float)
  marginal=np.column_stack([np.quantile(x[:,j],(np.arange(ng)+.5)/ng) for j in range(len(cols))]) if label==0 else x.copy()
  for j,col in enumerate(cols):
   if col in ['age','sex','smoking']:marginal[:,j]=np.rint(marginal[:,j])
  z=np.column_stack([norm.ppf((rankdata(x[:,j])-.5)/len(x)) for j in range(x.shape[1])])
  corr=np.corrcoef(z,rowvar=False);corr=np.nan_to_num(corr);np.fill_diagonal(corr,1)
  val,vec=np.linalg.eigh(corr);corr=(vec*np.maximum(val,1e-5))@vec.T
  sample=rng.multivariate_normal(np.zeros(len(cols)),corr,ng)
  if 'balanced' in kind or 'calibrated' in kind or 'student' in kind:
   eps=rng.normal(size=(ng,len(cols)));eps-=eps.mean(0)
   ev,eq=np.linalg.eigh(eps.T@eps/ng);eps=eps@(eq*np.maximum(ev,1e-8)**-.5)@eq.T
   target_rank=pd.DataFrame(x).corr(method='spearman').fillna(0).to_numpy();np.fill_diagonal(target_rank,1.)
   
   if label==0:target_rank=(1-shrink)*target_rank+shrink*np.eye(len(cols))
   latent=corr.copy();besterr=np.inf;best_sample=None
   for _ in range(12 if ('calibrated' in kind or 'student' in kind) and (not 'main' in kind or label==0) else 1):
    ev,eq=np.linalg.eigh(latent);latent=(eq*np.maximum(ev,1e-5))@eq.T
    diag=np.sqrt(np.diag(latent));latent/=np.outer(diag,diag)
    ev,eq=np.linalg.eigh(latent);sample=eps@(eq*np.sqrt(np.maximum(ev,1e-8)))@eq.T
    vals=np.empty_like(marginal)
    for j in range(len(cols)): vals[np.argsort(sample[:,j]),j]=np.sort(marginal[:,j])
    current=pd.DataFrame(vals).corr(method='spearman').fillna(0).to_numpy();np.fill_diagonal(current,1.)
    err=float(np.mean((target_rank-current)**2))
    if err<besterr:besterr=err;best_sample=sample.copy()
    latent+=.8*(target_rank-current)
   sample=best_sample
  for j,col in enumerate(cols):
   order=np.argsort(sample[:,j]);v=np.empty(ng);v[order]=np.sort(marginal[:,j]);out.loc[idx,col]=v
 # jitter labs avoids copying isolated values and exact matches, preserves multimodal marginals
 for col in CONT:
  if col=='age':continue
  x=np.log(b[col]) if col in ['TG','ALT'] else b[col]
  v=np.log(out[col]) if col in ['TG','ALT'] else out[col]
  v=v+rng.normal(0,.015*x.std(ddof=0),n)
  if col in ['TG','ALT']:v=np.exp(v)
  out[col]=np.clip(v,*s['ranges'][col])
 out['prefecture']=rng.permutation(b.prefecture.to_numpy())
 return out,group
def outcomes(c,b,group,rng,steps=80):
 # 独立指数乱数から発症時計を作り、各層でBと同じ発症・死亡数を配置。
 # Bの個人別転帰との対応ではなく、層内の時刻分布とCox係数を合わせる。
 # 係数差を発症時計のリスク係数へ戻す。交換探索・公式点数の最大化は行わない。
 target=fit(b);x=xf(c).to_numpy();sd=xf(b).std(ddof=0).to_numpy();mean=xf(b).mean().to_numpy();xx=(x-mean)/sd
 targetbeta=target.params_.to_numpy()*sd;se=target.standard_errors_.to_numpy()*sd
 theta=targetbeta.copy();best=None;bestloss=np.inf
 # independent exponential clocks. same random numbers throughout parameter fitting
 e=-np.log(np.clip(rng.random(len(c)),1e-10,1)); fixed_noise=np.log(e)
 # fixed death assignments based on age (independent of original rows)
 death=np.zeros(len(c),int)
 death_score=xx[:,0]*.5 + rng.gumbel(size=len(c))
 for g in np.unique(group):
  ids=np.flatnonzero(group==g);nd=int(b.iloc[ids].death.sum());death[ids[np.argsort(death_score[ids])[-nd:]]]=1 if nd else 0
 pools={}
 for g in np.unique(group):
  ids=np.flatnonzero(group==g);bg=b.iloc[ids]
  pools[g]=np.sort(np.clip(bg.loc[bg.onset==1,'time'].to_numpy()+rng.normal(0,.15,int(bg.onset.sum())),.003,9.99))
 for it in range(steps):
  c=c.copy();c['onset']=0;c['death']=death;c['time']=10.
  clock=fixed_noise-xx@theta
  for g in np.unique(group):
   ids=np.flatnonzero(group==g);eligible=ids[death[ids]==0];n_on=len(pools[g]);chosen=eligible[np.argsort(clock[eligible])[:n_on]]
   c.loc[chosen,'onset']=1;c.loc[chosen,'time']=pools[g]
   dids=ids[death[ids]==1];bg=b.iloc[ids];dt=np.sort(bg.loc[bg.death==1,'time'].to_numpy());c.loc[dids,'time']=dt
  try:m=fit(c)
  except Exception:break
  coef=m.params_.to_numpy()*sd;err=(targetbeta-coef);loss=float(np.mean((err/se)**2))
  if loss<bestloss:bestloss=loss;best=c.copy()
  # Coefficient loss selects the iterate; official scores are not available here.
  theta+=.6*err
 return best,bestloss

