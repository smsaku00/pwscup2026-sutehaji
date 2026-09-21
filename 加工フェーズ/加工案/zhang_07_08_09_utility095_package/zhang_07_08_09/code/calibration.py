import numpy as np,pandas as pd
from scipy.stats import norm,rankdata
import shared_methods as base

def calibrate_normal(c,b,a,s,g,weight=.5):
 cols=base.CONT+['sex','smoking'];ix=np.flatnonzero(g==0)
 pa=a.loc[~base.public_outliers(a,s),cols];pb=b.loc[~base.public_outliers(b,s),cols]
 q=(np.arange(len(ix))+.5)/len(ix)
 marg=np.column_stack([weight*np.quantile(pb[col],q)+(1-weight)*np.quantile(pa[col],q) for col in cols])
 for j,col in enumerate(cols):
  if col in ['age','sex','smoking']:marg[:,j]=np.rint(marg[:,j])
 target=weight*pb.corr(method='spearman').to_numpy()+(1-weight)*pa.corr(method='spearman').to_numpy()
 x=c.loc[ix,cols].to_numpy(float)
 eps=np.column_stack([norm.ppf((rankdata(x[:,j])-.5)/len(x)) for j in range(x.shape[1])]);eps-=eps.mean(0)
 ev,eq=np.linalg.eigh(eps.T@eps/len(ix));eps=eps@(eq*np.maximum(ev,1e-8)**-.5)@eq.T
 latent=2*np.sin(np.pi*target/6);best=None;besterr=np.inf
 for _ in range(12):
  ev,eq=np.linalg.eigh(latent);latent=(eq*np.maximum(ev,1e-5))@eq.T;diag=np.sqrt(np.diag(latent));latent/=np.outer(diag,diag)
  ev,eq=np.linalg.eigh(latent);sample=eps@(eq*np.sqrt(np.maximum(ev,1e-8)))@eq.T
  vals=np.empty_like(marg)
  for j in range(len(cols)):vals[np.argsort(sample[:,j]),j]=np.sort(marg[:,j])
  current=pd.DataFrame(vals).corr(method='spearman').fillna(0).to_numpy();np.fill_diagonal(current,1)
  err=float(np.mean((target-current)**2))
  if err<besterr:besterr=err;best=vals.copy()
  latent+=.8*(target-current)
 for j,col in enumerate(cols):c.loc[ix,col]=best[:,j]
 return c


def radial_balance(c,b,s,rng,steps=5):
 c=c.copy();c['age']=c['age'].astype(float)
 r=s['rare_detector'];cols=r['columns'];mu=np.asarray(r['mu']);prec=np.linalg.pinv(np.asarray(r['sigma']))
 def dist(x):
  x=x-mu;return np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',x,prec,x)))
 bf=base.public_outliers(b,s);flags=base.public_outliers(c,s);ix=np.flatnonzero(flags);source=b.loc[bf]
 goal_radial=np.quantile(dist(source[cols].to_numpy()),(np.arange(len(ix))+.5)/len(ix))
 for it in range(steps):
  for col in cols:
   v=c.loc[ix,col].to_numpy(float);target=np.quantile(source[col],(np.arange(len(ix))+.5)/len(ix));new=np.empty(len(ix));new[np.argsort(v)]=target
   c.loc[ix,col]=.25*v+.75*new
  x=c.loc[ix,cols].to_numpy();now=dist(x);goal=np.empty(len(ix));goal[np.argsort(now)]=goal_radial
  x=mu+(x-mu)*(goal/np.maximum(now,1e-8))[:,None]
  c.loc[ix,cols]=x
 c.age=np.clip(np.rint(c.age),*s['age_range'])
 for col in s['ranges']:c[col]=np.clip(c[col],*s['ranges'][col])
 return c,[int(base.public_outliers(c,s).sum())]

