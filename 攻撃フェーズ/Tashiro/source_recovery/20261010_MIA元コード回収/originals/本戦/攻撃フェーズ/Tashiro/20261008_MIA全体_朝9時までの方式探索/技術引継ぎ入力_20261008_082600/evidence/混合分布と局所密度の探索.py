"""混合ガウス・局所帯域の密度比。会員正解は予測確定後の採点のみ。"""
from __future__ import annotations
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.exceptions import ConvergenceWarning
import 共通 as u
from 距離と分布の探索 import transform,categorical_logratio,nearest

def mixture(train,x,ncomp,covariance,regularization,seed,train_sex=None,query_sex=None):
 output=np.empty(len(x));records=[]
 masks=[(None,np.ones(len(train),bool),np.ones(len(x),bool))] if train_sex is None else [(sex,train_sex==sex,query_sex==sex) for sex in sorted(set(train_sex))]
 for sex,mask,querymask in masks:
  fit=GaussianMixture(n_components=ncomp,covariance_type=covariance,reg_covar=regularization,max_iter=200,n_init=2,random_state=seed)
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always',ConvergenceWarning);fit.fit(train[mask])
  output[querymask]=fit.score_samples(x[querymask])
  records.append(dict(sex=sex,components=ncomp,covariance=covariance,regularization=regularization,converged=bool(fit.converged_),iterations=int(fit.n_iter_),bic=float(fit.bic(train[mask])),warnings=';'.join(str(w.message) for w in caught)))
 assert np.isfinite(output).all()
 return output,records

def adaptive_log_density(distance,anchor_distance,query,anchors,a,cats,k,multiplier,dimension):
 # アンカー側の他点までの距離で帯域を決める。自己距離は含めない。
 own=anchor_distance.copy();np.fill_diagonal(own,np.inf)
 scale=np.maximum(nearest(own,k)[:,-1]/dimension,1e-8)*multiplier**2
 lp=-.5*distance/scale[None,:]-.5*dimension*np.log(scale)[None,:]
 for col in cats:
  levels=sorted(set(a[col])|set(anchors[col])|set(query[col]));p=(a[col].value_counts().reindex(levels,fill_value=0)+1)/(len(a)+len(levels))
  lp+=np.log(.98*(query[col].to_numpy()[:,None]==anchors[col].to_numpy()[None,:])+.02*query[col].map(p).to_numpy()[:,None])
 return logsumexp(lp,axis=1)-np.log(len(anchors))

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--family',choices=['mixture','local'],required=True)
 parser.add_argument('--output-dir',type=Path);args=parser.parse_args();out=args.output_dir or u.HERE/('混合分布結果' if args.family=='mixture' else '局所密度結果')
 assert not out.exists();started=time.perf_counter();data=u.setup();a=data['a'];b=data['b'];q=data['query'];banks={};specs=[];fitrecords=[]
 u.base.INPUT_HASHES[str(Path(__file__))]=u.sha(__file__)
 def add(scores,t,calibration):
  for name,values in u.variants(scores,data,t,calibration).items():banks.setdefault(name,{})[t]=values
  if t==data['teams'][0]:specs.extend(dict(raw_method=name,family=args.family,public_null_calibration=calibration) for name in scores)
 for label,nums,cats in [('q7',u.NUM7,u.CAT7),('q9',u.NUM9,u.CAT7),('q13',u.NUM13,u.CAT13)]:
  query=q if label=='q7' else b;calibration=label=='q7'
  for kind in ('robust','rank'):
   ar,x=transform(a,[a,query],nums,kind)
   if args.family=='mixture':
    for conditional in (False,True):
     for covariance in ('full','diag'):
      for reg in (.01,.1):
       reference={};risks={}
       for components in (1,2,4,8,16):
        reference[components],rr=mixture(ar,x,components,covariance,reg,20261008,a.sex.to_numpy() if conditional else None,query.sex.to_numpy() if conditional else None)
        fitrecords.extend(dict(source='A',team=-1,attributes=label,kind=kind,conditional=conditional,**r) for r in rr)
        risks[components]=sum(r['bic'] for r in rr)
       for t,c in data['cs'].items():
        _,_,cr=transform(a,[a,query,c],nums,kind);scores={};crisk={};component_scores={}
        cat=categorical_logratio(a,c,query,cats,20)
        for components in (1,2,4,8,16):
         value,rr=mixture(cr,x,components,covariance,reg,20261008+t,c.sex.to_numpy() if conditional else None,query.sex.to_numpy() if conditional else None)
         component_scores[components]=value;crisk[components]=sum(r['bic'] for r in rr)
         fitrecords.extend(dict(source='C',team=t,attributes=label,kind=kind,conditional=conditional,**r) for r in rr)
         name=f'{label}_{kind}_sex{int(conditional)}_{covariance}_reg{reg:g}_gmm{components}'
         scores[name]=value-reference[components]+cat
        ak=min(risks,key=risks.get);ck=min(crisk,key=crisk.get)
        scores[f'{label}_{kind}_sex{int(conditional)}_{covariance}_reg{reg:g}_bic_selected']=component_scores[ck]-reference[ak]+cat
        add(scores,t,calibration)
       u.log(stage='mixture_group_complete',attributes=label,kind=kind,sex_conditional=conditional,covariance=covariance,regularization=reg,seconds=time.perf_counter()-started)
   else:
    da=cdist(x,ar,'sqeuclidean');daa=cdist(ar,ar,'sqeuclidean');den={}
    for k in (5,15,50):
     ka=min(len(a)-1,round(k*len(a)/1049))
     for multiplier in (.5,1.,2.):den[(k,multiplier)]=adaptive_log_density(da,daa,query,a,a,cats,ka,multiplier,len(nums))
    lof_a={}
    for k in (10,30,100):
     fit=LocalOutlierFactor(n_neighbors=k,novelty=True,n_jobs=2);fit.fit(ar);lof_a[k]=fit.score_samples(x)
    for t,c in data['cs'].items():
     _,_,cr=transform(a,[a,query,c],nums,kind);dc=cdist(x,cr,'sqeuclidean');dcc=cdist(cr,cr,'sqeuclidean');scores={}
     for k in (5,15,50):
      for multiplier in (.5,1.,2.):
       lc=adaptive_log_density(dc,dcc,query,c,a,cats,k,multiplier,len(nums))
       scores[f'{label}_{kind}_adaptivek{k}_h{multiplier:g}']=lc-den[(k,multiplier)]
     for k in (10,30,100):
      fit=LocalOutlierFactor(n_neighbors=k,novelty=True,n_jobs=2);fit.fit(cr);value=fit.score_samples(x)
      scores[f'{label}_{kind}_lofC_k{k}']=value;scores[f'{label}_{kind}_lof_ratio_k{k}']=value-lof_a[k]
     add(scores,t,calibration)
    u.log(stage='local_density_group_complete',attributes=label,kind=kind,seconds=time.perf_counter()-started)
 models=u.matrices(banks,data['teams']);table,agg=u.save(out,data,models,specs,started,{'method_note':'公開A/Cのみで混合成分・局所帯域を推定。BIC選択にも会員ラベルは使わない。局所帯域はC内の他点とA内の他点への距離で定義。','reference_documentation':'https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html'})
 if fitrecords:pd.DataFrame(fitrecords).to_csv(out/'分布モデル適合.csv',index=False,float_format='%.17g')
 record=out/'実行記録.json';r=json.loads(record.read_text());r['output_sha256']={p.name:u.sha(p) for p in out.iterdir() if p.name!='実行記録.json'};record.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
