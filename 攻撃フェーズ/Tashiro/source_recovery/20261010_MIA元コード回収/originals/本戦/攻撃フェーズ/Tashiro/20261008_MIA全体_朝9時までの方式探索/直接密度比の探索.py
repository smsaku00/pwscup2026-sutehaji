"""uLSIF/RuLSIF型の直接密度比。2つの密度を別々には推定しない。"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import 共通 as u
from 距離と分布の探索 import transform

def kernel_dist(x,y,fx,fy,cats):
 d=cdist(x,y,'sqeuclidean')
 for col in cats:d+=(fx[col].to_numpy()[:,None]!=fy[col].to_numpy()[None,:])
 return d

def ratios(a,c,query,nums,cats,kind,seed):
 ar,cr,x=transform(a,[a,c,query],nums,kind)
 rng=np.random.default_rng(seed);order=rng.permutation(len(c));result={}
 for centers in (100,300):
  ii=order[:centers];f=c.iloc[ii];z=cr[ii]
  da=kernel_dist(ar,z,a,f,cats);dc=kernel_dist(cr,z,c,f,cats);dx=kernel_dist(x,z,query,f,cats)
  for h in (.5,1.,2.,4.):
   ka=np.exp(-.5*da/h**2);kc=np.exp(-.5*dc/h**2);kx=np.exp(-.5*dx/h**2)
   ha=ka.T@ka/len(a);hc=kc.T@kc/len(c);mean=kc.mean(axis=0)
   for relative in (0.,.1,.5):
    matrix=(1-relative)*ha+relative*hc
    eigvec_value,eigvec=np.linalg.eigh(matrix)
    assert eigvec_value.min()>-1e-10
    projected=eigvec.T@mean
    for regularizer in (.0001,.01,1.):
     beta=np.maximum(0,eigvec@(projected/(np.maximum(eigvec_value,0)+regularizer)))
     value=kx@beta
     assert np.isfinite(value).all() and (value>=0).all()
     result[f'centers{centers}_h{h:g}_a{relative:g}_lambda{regularizer:g}']=value
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--seed',type=int,default=20261008)
 parser.add_argument('--output-dir',type=Path,default=u.HERE/'直接密度比結果')
 args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();banks={};specs=[]
 for p in [Path(__file__),u.HERE/'距離と分布の探索.py']:u.base.INPUT_HASHES[str(p.resolve())]=u.sha(p)
 for label,nums,cats,kind in [('q7_robust',u.NUM7,u.CAT7,'robust'),('q7_rank',u.NUM7,u.CAT7,'rank'),('q13_robust',u.NUM13,u.CAT13,'robust')]:
  query=data['query'] if label.startswith('q7_') else data['b'];calibration=label.startswith('q7_')
  for index,(t,c) in enumerate(data['cs'].items(),1):
   scores={label+'_ulsif_'+n:v for n,v in ratios(data['a'],c,query,nums,cats,kind,args.seed+t).items()}
   for name,value in u.variants(scores,data,t,calibration).items():banks.setdefault(name,{})[t]=value
   if index==1:
    for name in scores:specs.append(dict(raw_method=name,family='direct_density_ratio',transform=kind,attributes=len(nums)+len(cats),uses_public_null=calibration,seed=args.seed,
                                      equation='beta=max(0, inv((1-a)*E_A[K K^T]+a*E_C[K K^T]+lambda*I)*E_C[K])'))
   u.log(stage='direct_ratio',configuration=label,completed_targets=index,targets=len(data['cs']),seconds=time.perf_counter()-started)
 models=u.matrices(banks,data['teams'])
 u.log(stage='predictions_fixed',raw_specs=len(specs),models=len(models),seconds=time.perf_counter()-started)
 u.save(args.output_dir,data,models,specs,started,{'seed':args.seed,'source':'https://www.jmlr.org/papers/v10/kanamori09a.html; https://arxiv.org/abs/1106.4729',
                                               'adaptation':'Mixed categorical Gaussian basis, fixed public-data parameter grid; no membership labels in fitting.'})

if __name__=='__main__':main()
