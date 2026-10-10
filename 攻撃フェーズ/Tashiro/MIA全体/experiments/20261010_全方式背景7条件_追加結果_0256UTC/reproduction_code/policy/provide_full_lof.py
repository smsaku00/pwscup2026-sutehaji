"""Seal the fixed22 q13 LOF channel early for downstream learned-grid reuse."""
import runtime as r
import numpy as np
NAME='q13_robust_lofC_k100__raw__cross0'
for bg in r.BG:
 out=r.HERE/'cache/early_lof'/bg
 if (out/'prediction_record.json').exists():continue
 d=r.data(bg);parent=r.HERE/'cache/fixed22'/bg/'full/histlocal'
 if (parent/'prediction_record.json').exists():models={NAME:r.load_scores(parent)[NAME]}
 else:
  a,b=d['a'],d['b'];x=r.geo.transform(a,[b],r.u.NUM13,'robust')[0];cols=[]
  for t in d['teams']:
   cr=r.geo.transform(a,[d['cs'][t]],r.u.NUM13,'robust')[0]
   fit=r.local.LocalOutlierFactor(n_neighbors=100,novelty=True,n_jobs=1);fit.fit(cr)
   cols.append(r.u.rank(fit.score_samples(x),d['normal']))
  models={NAME:np.column_stack(cols)}
 r.seal_scores(out,models,d,{'source_family':'fixed22 q13 robust LOF-C k100; raw calibration; cross0','background':bg})
 print('LOF_READY',bg,flush=True)
