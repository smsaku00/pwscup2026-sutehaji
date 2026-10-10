from pathlib import Path
import time,json,importlib
from benchmark import data,u
import optimized as opt
import numpy as np
from scipy.spatial.distance import cdist
geo=importlib.import_module('距離と分布の探索');mix=importlib.import_module('混合分布と局所密度の探索')
d=data('prelim');a=d['a'];q=d['query'];c=d['cs'][7];checks=[]
for label,nums,cats,kind,query in [('q7',u.NUM7,u.CAT7,'robust',q),('q13',u.NUM13,u.CAT13,'rank',d['b'])]:
 ar,x=geo.transform(a,[a,query],nums,kind);near=geo.nearest(geo.distances(x,ar,query,a,cats,1),150)
 start=time.perf_counter();old=opt.ORIGINAL_GEOMETRY(a,query,c,nums,cats,kind,1,near);elapsed=time.perf_counter()-start
 start=time.perf_counter();new=opt.geometry(a,query,c,nums,cats,kind,1,near);fast=time.perf_counter()-start
 assert old.keys()==new.keys()
 for k in old:assert np.array_equal(old[k],new[k]),(label,k,np.max(abs(old[k]-new[k])))
 checks.append({'test':'geometry_raw_scores_bit_exact','attributes':label,'signals':len(old),'original_seconds':elapsed,'optimized_first_seconds':fast,'passed':True})
ar,x,cr=geo.transform(a,[a,q,c],u.NUM7,'robust');scales=opt.anchor_scales(cr,[5,15,50],len(u.NUM7))
for k in (5,15,50):
 old=mix.adaptive_log_density(cdist(x,cr,'sqeuclidean'),cdist(cr,cr,'sqeuclidean'),q,c,a,u.CAT7,k,2.,len(u.NUM7))
 new=opt.adaptive(x,cr,q,c,a,u.CAT7,scales[k],2.,len(u.NUM7))
 assert np.array_equal(old,new),(k,np.max(abs(old-new)))
 checks.append({'test':'local_adaptive_raw_scores_bit_exact','k':k,'passed':True})
Path(__file__).with_name('optimization_validation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks))
