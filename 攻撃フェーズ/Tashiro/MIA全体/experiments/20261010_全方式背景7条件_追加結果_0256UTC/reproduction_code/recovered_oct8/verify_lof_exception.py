"""Verify the historical LOF exception via untouched source feature functions."""
from pathlib import Path
import hashlib,importlib,json
import numpy as np
from benchmark import data,u,HERE
geo=importlib.import_module('距離と分布の探索');mix=importlib.import_module('混合分布と局所密度の探索')
d=data('prelim');a,b,q=d['a'],d['b'],d['query'];ar,x=geo.transform(a,[a,q],u.NUM7,'robust');banks={};stem='q7_robust_lofC_k10'
for t,c in d['cs'].items():
 _,_,cr=geo.transform(a,[a,q,c],u.NUM7,'robust')
 fit=mix.LocalOutlierFactor(n_neighbors=10,novelty=True,n_jobs=2);fit.fit(cr);value=fit.score_samples(x)
 for name,values in u.variants({stem:value},d,t,True).items():banks.setdefault(name,{})[t]=values
models=u.matrices(banks,d['teams']);z=np.load(HERE/'local/prelim/prelim/scores.npz');names=z['names'].tolist();rows=[]
for name,values in models.items():
 saved=z['scores'][names.index(name)];rows.append({'method':name,'bit_exact':bool(np.array_equal(saved,values)),'max_abs_error':float(np.max(abs(saved-values))),'source_function':'sklearn LocalOutlierFactor called exactly as original mixture/local source; original geometry.transform, u.variants, u.matrices','targets':d['teams']})
assert all(x['bit_exact'] for x in rows)
report={'source_direct_verification_passed':True,'channels':rows,'historical_exception_method':stem+'__age100__cross0.1','interpretation':'All six current-runtime source-direct postprocessed LOF channels are bitwise identical to the saved rerun. Historical catalog mismatch is real; no optimized adaptive/KDE kernel participates in this LOF-C channel. Its historical runtime/cache provenance needs separate investigation; do not claim every historical metric reproduced.'}
(HERE/'local/prelim/lof_source_direct_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
