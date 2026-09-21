"""本番team28: 04コピュラ、05逐次条件付き、06混合分布。各方式は別の生成機構。
生成入力=B.csv, A_bg.csv, schema.json（すべて本番配布）。予備選・正解ラベルは不使用。
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import norm,rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from previous_shared import CONT,public_outliers,covariates,fit,xf,outcomes
from previous_shared import outcomes as outcomes_Btime
METHODS=('zhang_04_copula_bgtime','zhang_05_conditional_bgtime','zhang_06_mixture_btime')
SEEDS={METHODS[0]:28201,METHODS[1]:28221,METHODS[2]:28261}

def conditional_sampler(b,a,s,rng):
 c,g=covariates(b,a,s,rng,0);ix=np.flatnonzero(g==0);pa=a.loc[~public_outliers(a,s)].reset_index(drop=True)
 cols=CONT+['sex','smoking']
 if True:
  # sequential conditional sampling from population. Fresh random donor chosen for each column.
  z=pa[cols].copy();z[['TG','ALT']]=np.log(z[['TG','ALT']]);z=(z-z.mean())/z.std(ddof=0)
  synth=np.zeros((len(ix),len(cols)))
  for j in range(len(cols)):
   if j<2:
    synth[:,j]=z.iloc[rng.choice(len(z),len(ix),replace=True),j].to_numpy();continue
   keys=list(range(j));dist=NearestNeighbors(n_neighbors=min(80,len(pa))).fit(z.iloc[:,keys].to_numpy())
   inds=dist.kneighbors(synth[:,keys],return_distance=False)
   subsel=inds[np.arange(len(ix)),rng.integers(0,inds.shape[1],len(ix))]
   synth[:,j]=z.iloc[subsel,j].to_numpy()+rng.normal(0,.04,len(ix))
  sub=pd.DataFrame({col:(synth[:,j]*((np.log(pa[col]) if col in ['TG','ALT'] else pa[col]).std(ddof=0))+((np.log(pa[col]) if col in ['TG','ALT'] else pa[col]).mean())) for j,col in enumerate(cols)})
  for col in ['TG','ALT']:sub[col]=np.exp(sub[col])
 # retain approximate population marginals by quantile matching independently
 for col in cols:
  val=sub[col].to_numpy(float);target=np.quantile(pa[col].to_numpy(float),(np.arange(len(ix))+.5)/len(ix));out=np.empty(len(ix))
  out[np.argsort(val)]=np.sort(target)
  if col in ['age','sex','smoking']:out=np.rint(out)
  else:
   v=np.log(out) if col in ['TG','ALT'] else out
   sd=(np.log(pa[col]) if col in ['TG','ALT'] else pa[col]).std(ddof=0)
   v=v+rng.normal(0,.015*sd,len(ix))
   if col in ['TG','ALT']:v=np.exp(v)
  c.loc[ix,col]=np.clip(out,*s['ranges'][col]) if col in s['ranges'] else np.clip(out,*s['age_range']) if col=='age' else out
 return c,g

def mixture(b,a,s,rng):
 c,g=covariates(b,a,s,rng,0);ix=np.flatnonzero(g==0);pa=a.loc[~public_outliers(a,s)].reset_index(drop=True)
 cols=CONT+['sex','smoking'];v=pa[cols].to_numpy(float)
 z=np.column_stack([norm.ppf(np.clip((rankdata(v[:,j])-.5)/len(v),1e-5,1-1e-5)) for j in range(len(cols))]);mdl=GaussianMixture(n_components=3,reg_covar=.003,random_state=28261,n_init=2).fit(z)
 # explicit generator independent of fit seed
 mixture_idx=rng.choice(3,len(ix),p=mdl.weights_)
 latent=np.vstack([rng.multivariate_normal(mdl.means_[k],mdl.covariances_[k],int((mixture_idx==k).sum())) for k in range(3)])
 latent=latent[rng.permutation(len(ix))]
 for j,col in enumerate(cols):
  ords=np.argsort(latent[:,j]);target=np.quantile(pa[col],(np.arange(len(ix))+.5)/len(ix));vals=np.empty(len(ix));vals[ords]=np.sort(target)
  if col in ['age','sex','smoking']:vals=np.rint(vals)
  else:
   v=np.log(vals) if col in ['TG','ALT'] else vals
   v+=rng.normal(0,.015*(np.log(pa[col]) if col in ['TG','ALT'] else pa[col]).std(ddof=0),len(ix))
   if col in ['TG','ALT']:v=np.exp(v)
  c.loc[ix,col]=np.clip(vals,*s['ranges'][col]) if col in s['ranges'] else np.clip(vals,*s['age_range']) if col=='age' else vals
 return c,g
def outcomes_background(c,b,a,s,group,rng,steps=80):
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
  ids=np.flatnonzero(group==g);bg=a.loc[public_outliers(a,s)==bool(g)]
  pool=bg.loc[bg.onset==1,'time'].to_numpy();pools[g]=np.sort(np.clip(rng.choice(pool,int(b.iloc[ids].onset.sum()),replace=True)+rng.normal(0,.35,int(b.iloc[ids].onset.sum())),.003,9.99))
 for it in range(steps):
  c=c.copy();c['onset']=0;c['death']=death;c['time']=10.
  clock=fixed_noise-xx@theta
  for g in np.unique(group):
   ids=np.flatnonzero(group==g);eligible=ids[death[ids]==0];n_on=len(pools[g]);chosen=eligible[np.argsort(clock[eligible])[:n_on]]
   c.loc[chosen,'onset']=1;c.loc[chosen,'time']=pools[g]
   dids=ids[death[ids]==1];bg=b.iloc[ids];src=a.loc[public_outliers(a,s)==bool(g)];tt=src.loc[src.death==1,'time'].to_numpy();dt=np.sort(rng.choice(tt,len(dids),replace=True));c.loc[dids,'time']=dt
  try:m=fit(c)
  except Exception:break
  coef=m.params_.to_numpy()*sd;err=(targetbeta-coef);loss=float(np.mean((err/se)**2))
  if loss<bestloss:bestloss=loss;best=c.copy()
  # Coefficient loss selects the iterate; official scores are not available here.
  theta+=.6*err
 return best,bestloss



def generate(b,a,s,method):
 if method not in METHODS:raise ValueError('unknown method: '+method)
 cols=s['submitted_c_columns']
 if set(b.columns) not in [set(cols),set(cols+['record_id'])]:
  raise ValueError('本番のB.csvを渡してください。B_selfは禁止です')
 if set(a.columns) not in [set(cols),set(cols+['record_id'])]:
  raise ValueError('本番のA_bg.csvを渡してください')
 b=b[cols].reset_index(drop=True).copy();a=a.reset_index(drop=True).copy()
 if b.isna().any().any() or a.isna().any().any() or len(b)<50:
  raise ValueError('欠測値や小標本は非対応')
 rng=np.random.default_rng(SEEDS[method])
 if method==METHODS[0]:
  c,g=covariates(b,a,s,rng,0);c,loss=outcomes_background(c,b,a,s,g,rng)
 elif method==METHODS[1]:
  c,g=conditional_sampler(b,a,s,rng);c,loss=outcomes_background(c,b,a,s,g,rng)
 else:
  c,g=mixture(b,a,s,rng);c,loss=outcomes_Btime(c,b,g,rng)
 if c is None:raise ValueError('Cox係数の較正が失敗')
 c=c[cols].iloc[rng.permutation(len(c))].reset_index(drop=True)
 for column in ('age','sex','smoking','onset','death'):c[column]=c[column].astype(int)
 return c,{'method':method,'seed':SEEDS[method],'cox_scaled_loss':loss}

def csv_bytes(c):
 return c.to_csv(index=False,float_format='%.17g',lineterminator='\n').encode('utf-8')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for arg in ('b','a-bg','schema','out'):parser.add_argument('--'+arg,type=Path,required=True)
 parser.add_argument('--method',choices=METHODS+('all',),default='all');args=parser.parse_args()
 b=pd.read_csv(args.b,float_precision='round_trip');a=pd.read_csv(args.a_bg,float_precision='round_trip');s=json.loads(args.schema.read_text())
 for method in METHODS if args.method=='all' else [args.method]:
  c,meta=generate(b,a,s,method);path=args.out/method;path.mkdir(parents=True,exist_ok=True);data=csv_bytes(c)
  (path/'C.csv').write_bytes(data);meta['sha256']=hashlib.sha256(data).hexdigest();(path/'generation.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
  print(method,'rows',len(c),flush=True)
if __name__=='__main__':main()
