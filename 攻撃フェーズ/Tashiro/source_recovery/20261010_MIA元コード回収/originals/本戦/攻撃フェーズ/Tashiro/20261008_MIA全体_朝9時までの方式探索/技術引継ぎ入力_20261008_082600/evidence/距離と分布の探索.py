"""Gen-LRA以外の距離、点密度、分布差を公開所属正解の前に固定して比較。"""
from __future__ import annotations
import argparse,itertools,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import logsumexp,ndtri
from sklearn.covariance import LedoitWolf
import 共通 as u

KS=(1,3,10,30)

def numeric(frame,cols):
 x=frame[list(cols)].to_numpy(float).copy()
 for col in ('TG','ALT'):
  if col in cols:x[:,cols.index(col)]=np.log1p(x[:,cols.index(col)])
 assert np.isfinite(x).all();return x

def transform(reference,frames,cols,kind):
 r=numeric(reference,cols)
 if kind=='sex':
  result=[]
  for frame in frames:
   x=numeric(frame,cols)
   for sex in sorted(reference.sex.unique()):
    rr=r[reference.sex.to_numpy()==sex]
    scale=(np.quantile(rr,.75,axis=0)-np.quantile(rr,.25,axis=0))/1.349
    scale=np.where(scale>1e-8,scale,np.maximum(rr.std(axis=0),1e-6))
    mask=frame.sex.to_numpy()==sex;x[mask]=(x[mask]-np.median(rr,axis=0))/scale
   result.append(x)
  return result
 if kind=='rank':
  result=[]
  for frame in frames:
   x=numeric(frame,cols)
   for j in range(len(cols)):
    ordered,counts=np.unique(r[:,j],return_counts=True)
    probs=(np.cumsum(counts)-.5*counts)/len(r)
    x[:,j]=ndtri(np.clip(np.interp(x[:,j],ordered,probs,left=0.,right=1.),.001,.999))
   result.append(x)
  return result
 scale=(np.quantile(r,.75,axis=0)-np.quantile(r,.25,axis=0))/1.349
 scale=np.where(scale>1e-8,scale,np.maximum(r.std(axis=0),1e-6))
 result=[(numeric(f,cols)-np.median(r,axis=0))/scale for f in frames]
 if kind=='whiten':
  covariance=LedoitWolf().fit(result[0]).covariance_
  whitening=np.linalg.inv(np.linalg.cholesky(covariance)).T
  result=[x@whitening for x in result]
 return result

def distances(x,y,fx,fy,cats,weight,strict_sex=False):
 d=cdist(x,y,'sqeuclidean')
 for col in cats:d+=weight*(fx[col].to_numpy()[:,None]!=fy[col].to_numpy()[None,:])
 if strict_sex:d[fx.sex.to_numpy()[:,None]!=fy.sex.to_numpy()[None,:]]=1e12
 return d

def nearest(d,k=150):
 return np.sort(np.partition(d,k-1,axis=1)[:,:k],axis=1)

def mixed_logs(distance,query,ref,cats,background,h):
 result=-.5*distance/h**2
 for col in cats:
  levels=sorted(set(background[col])|set(ref[col])|set(query[col]))
  freq=(background[col].value_counts().reindex(levels,fill_value=0)+1)/(len(background)+len(levels))
  eq=query[col].to_numpy()[:,None]==ref[col].to_numpy()[None,:]
  p=query[col].map(freq).to_numpy()[:,None]
  result+=np.log(.98*eq+.02*p)
 return logsumexp(result,axis=1)-np.log(len(ref))

def geometry(a,query,c,nums,cats,kind,weight,ref_near):
 ar,x,cr=transform(a,[a,query,c],nums,kind)
 dnum=cdist(x,cr,'sqeuclidean')
 dc=dnum.copy()
 for col in cats:dc+=weight*(query[col].to_numpy()[:,None]!=c[col].to_numpy()[None,:])
 if kind=='sex':dc[query.sex.to_numpy()[:,None]!=c.sex.to_numpy()[None,:]]=1e12
 nc=nearest(dc,30);result={}
 for k in KS:
  cm=nc[:,:k].mean(axis=1)
  result[f'dcr_k{k}']=-np.log(cm+1e-12)
  result[f'nnratio_k{k}']=np.log(ref_near[:,:k].mean(axis=1)+1e-12)-np.log(cm+1e-12)
  result[f'nnratio_mass_k{k}']=np.log(ref_near[:,:min(150,5*k)].mean(axis=1)+1e-12)-np.log(cm+1e-12)
 # 点密度比は追加前後のGen-LRAと別の式。三つの尺度で確認する。
 if weight==1 and kind in ('robust','rank'):
  da=cdist(x,ar,'sqeuclidean')
  h0=(4/(len(nums)+2)/len(a))**(1/(len(nums)+4))
  for multiplier in (.5,1.,2.):
   h=h0*multiplier
   lc=mixed_logs(dnum,query,c,cats,a,h)
   la=mixed_logs(da,query,a,cats,a,h)
   result[f'kde_ratio_h{multiplier:g}']=lc-la
   result[f'kde_witness_h{multiplier:g}']=np.exp(lc)-np.exp(la)
 return result

def categorical_logratio(a,c,query,cats,alpha):
 answer=np.zeros(len(query))
 for col in cats:
  levels=sorted(set(a[col])|set(c[col])|set(query[col]))
  ar=(a[col].value_counts().reindex(levels,fill_value=0)+1)/(len(a)+len(levels))
  cr=(c[col].value_counts().reindex(levels,fill_value=0)+alpha*ar)/(len(c)+alpha)
  answer+=query[col].map(np.log(cr/ar)).to_numpy()
 return answer

def gaussian(a,query,c,nums,cats):
 ar,x,cr=transform(a,[a,query,c],nums,'robust');result={}
 for shrink in (0.,.5,1.):
  def lp(train):
   cov=LedoitWolf().fit(train).covariance_
   cov=(1-shrink)*cov+shrink*np.diag(np.diag(cov))
   chol=np.linalg.cholesky(cov)
   z=solve_triangular(chol,(x-train.mean(axis=0)).T,lower=True).T
   return -.5*(z*z).sum(axis=1)-np.log(np.diag(chol)).sum()
  result[f'gaussian_shrink{shrink:g}']=lp(cr)-lp(ar)+categorical_logratio(a,c,query,cats,20)
 return result

def binned(a,query,c,nums,cats,bins,alpha):
 frames=[a,query,c];codes=[];sizes=[]
 for col in list(nums)+list(cats):
  if col in nums:
   edge=np.unique(np.quantile(a[col],np.arange(1,bins)/bins))
   codes.append([np.searchsorted(edge,f[col].to_numpy(),side='right') for f in frames]);sizes.append(len(edge)+1)
  else:
   levels=sorted(set(a[col])|set(query[col])|set(c[col]));mapping={v:i for i,v in enumerate(levels)}
   codes.append([f[col].map(mapping).to_numpy() for f in frames]);sizes.append(len(levels))
 def log_ratio(idx):
  size=int(np.prod([sizes[i] for i in idx]));vals=[]
  for f in range(3):
   code=np.zeros(len(frames[f]),dtype=int)
   for i in idx:code=code*sizes[i]+codes[i][f]
   vals.append(code)
  ac=np.bincount(vals[0],minlength=size);cc=np.bincount(vals[2],minlength=size)
  p=(ac+1)/(len(a)+size);q=(cc+alpha*p)/(len(c)+alpha)
  return np.log(q/p)[vals[1]]
 singles=[log_ratio([i]) for i in range(len(codes))]
 marginal=np.sum(singles,axis=0)
 joint=[];interaction=[]
 for i,j in itertools.combinations(range(len(codes)),2):
  score=log_ratio([i,j]);joint.append(score);interaction.append(score-singles[i]-singles[j])
 return {'marginal':marginal,'pairwise':np.mean(joint,axis=0),'interaction':np.mean(interaction,axis=0)}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=u.HERE/'距離と分布結果')
 args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();a=data['a'];b=data['b'];q=data['query'];banks={};specs=[]
 u.base.INPUT_HASHES[str(Path(__file__).resolve())]=u.sha(__file__)
 configurations=[('q7_robust',u.NUM7,u.CAT7,'robust',1),('q7_cat4',u.NUM7,u.CAT7,'robust',4),
                 ('q7_rank',u.NUM7,u.CAT7,'rank',1),('q7_whiten',u.NUM7,u.CAT7,'whiten',1),
                 ('q7_sex',u.NUM7,u.CAT7,'sex',1),('q9_robust',u.NUM9,u.CAT7,'robust',1),
                 ('q13_robust',u.NUM13,u.CAT13,'robust',1),('q13_rank',u.NUM13,u.CAT13,'rank',1)]
 def add(scores,t,calibration,family):
  for name,vector in u.variants(scores,data,t,calibration).items():banks.setdefault(name,{})[t]=vector
  if t==data['teams'][0]:
   for name in scores:specs.append(dict(raw_method=name,family=family,uses_public_null=calibration))
 for label,nums,cats,kind,weight in configurations:
  query=q if label.startswith('q7_') else b;calibration=query is q
  ar,x=transform(a,[a,query],nums,kind)
  ref_near=nearest(distances(x,ar,query,a,cats,weight,kind=='sex'),150)
  for t,c in data['cs'].items():
   scores={label+'_'+name:value for name,value in geometry(a,query,c,nums,cats,kind,weight,ref_near).items()}
   add(scores,t,calibration,'distance_or_density')
  u.log(stage='geometry_complete',configuration=label,raw_scores=len(scores),seconds=time.perf_counter()-started)
 for label,nums,cats in [('q7',u.NUM7,u.CAT7),('q13',u.NUM13,u.CAT13)]:
  query=q if label=='q7' else b;calibration=label=='q7'
  for t,c in data['cs'].items():
   scores={label+'_'+name:v for name,v in gaussian(a,query,c,nums,cats).items()}
   for bins in (5,10):
    for alpha in (20,100):
     scores.update({f'{label}_hist{bins}_s{alpha}_{name}':v for name,v in binned(a,query,c,nums,cats,bins,alpha).items()})
   add(scores,t,calibration,'gaussian_or_histogram')
  u.log(stage='distribution_complete',configuration=label,raw_scores=len(scores),seconds=time.perf_counter()-started)
 # 現行との再現対照。ここでも公開正解は読まない。
 for t,c in data['cs'].items():
  scores={'gen7_reference':u.base.core(a,q,c,u.NUM7,(10,))[10]}
  add(scores,t,True,'reproduction_reference')
 models=u.matrices(banks,data['teams'])
 u.log(stage='predictions_fixed',raw_specs=len(specs),models=len(models),seconds=time.perf_counter()-started)
 table,_=u.save(args.output_dir,data,models,specs,started,{'method_note':'全7属性方式は公開課題非会員で較正可。9・13属性方式はQに不足する属性があるため初回は生順位のみ。'})
 actual=table[(table.method=='gen7_reference__county500__cross0.1')&(table.fpr==.01)].set_index('team').mia_all
 ref=u.base.read(u.CAL/'結果/対象別TPR.csv')
 expected=ref[(ref.condition=='full')&(ref.method=='flat_pref_sex_s500')&(ref.fpr==.01)].set_index('team').mia_all
 err=float(np.max(np.abs(actual-expected)));assert err<1e-12
 u.log(stage='baseline_reproduced',maximum_error=err)
 (args.output_dir/'基準再現.json').write_text(__import__('json').dumps({'maximum_error':err},indent=2)+'\n')

if __name__=='__main__':main()
