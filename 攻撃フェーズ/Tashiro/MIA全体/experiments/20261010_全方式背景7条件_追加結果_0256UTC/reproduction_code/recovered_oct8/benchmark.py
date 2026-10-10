from pathlib import Path
import sys,time,importlib,json,resource
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
SRC=HERE/'original/本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索'
sys.path.insert(0,str(SRC));import 共通 as u
INFRA=Path('/workspace/shared/mia_bg7_stage/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_BG7横断比較')
sys.path.insert(0,str(INFRA));import bg7_inputs

def data(bg='starter+prelim+main',teams=None,phase='prelim'):
 d=bg7_inputs.load_phase(phase);b=d['b'].copy();a=bg7_inputs.load_background(bg);qs=d['q'];cs=d['c'];ts=d['targets']
 query=pd.concat([b[list(u.base.QI7)]]+[qs[t][list(u.base.QI7)] for t in sorted(qs)],ignore_index=True)
 owner=np.r_[np.full(len(b),-1),*[np.full(len(qs[t]),t) for t in sorted(qs)]]
 cuts=np.unique(np.quantile(a.age,(.2,.4,.6,.8)));query['age_bin']=np.searchsorted(cuts,query.age,side='right');b['age_bin']=query.age_bin.iloc[:len(b)].to_numpy();a['age_bin']=np.searchsorted(cuts,a.age,side='right')
 if teams is not None:ts=teams;cs={t:cs[t] for t in ts}
 return dict(phase=phase,b=b,a=a,private=d['private'],rare=d['rare'],normal=d['normal'],cs=cs,teams=ts,query=query,controls={t:np.flatnonzero((owner!=-1)&(owner!=t)) for t in ts},age_cuts=cuts,self_id=d['self_id'],qs=qs,targets=Path(bg7_inputs.manifest()['phases'][phase]['attack_dir']))

def main():
 geo=importlib.import_module('距離と分布の探索');mix=importlib.import_module('混合分布と局所密度の探索');hist=importlib.import_module('階層ヒストグラムの探索')
 d=data();a,b,q=d['a'],d['b'],d['query'];c=d['cs'][7];checks=[]
 for name,func in [
  ('histogram_q7_bins40_prefsex_8strengths',lambda:hist.histograms(a,q,c,u.NUM7,u.CAT7,40,['prefecture','sex'])),
  ('geometry_q7_robust_18scores',lambda:geo.geometry(a,q,c,u.NUM7,u.CAT7,'robust',1,geo.nearest(geo.distances(*geo.transform(a,[q,a],u.NUM7,'robust'),q,a,u.CAT7,1),150))),
 ]:
  start=time.perf_counter();result=func();seconds=time.perf_counter()-start
  checks.append({'kernel':name,'seconds':seconds,'score_arrays':len(result),'bg_rows':len(a),'query_rows':len(q),'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss});print(checks[-1],flush=True)
 ar,x,cr=geo.transform(a,[a,q,c],u.NUM7,'robust')
 for source,train,seed in [('A',ar,20261008),('C',cr,20261015)]:
  for ncomp,cov,reg in [(16,'full',.01),(16,'diag',.01),(4,'full',.1)]:
   start=time.perf_counter();_,rr=mix.mixture(train,x,ncomp,cov,reg,seed);seconds=time.perf_counter()-start
   checks.append({'kernel':'GMM','source':source,'components':ncomp,'covariance':cov,'reg':reg,'seconds':seconds,'rows':len(train),'fit_records':rr});print(checks[-1],flush=True)
 (HERE/'benchmark_results.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
