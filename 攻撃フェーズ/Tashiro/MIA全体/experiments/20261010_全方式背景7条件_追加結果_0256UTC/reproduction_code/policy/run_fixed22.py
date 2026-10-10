"""Narrow original fixed22 on outer source pools and original seven inner views."""
import argparse,time,json,gc
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
import runtime as r

PRIVATE=Path('/workspace/shared/mia_recovered_validation/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/private_inputs')
OUTCOME=Path('/workspace/shared/mia_recovered_sources/攻撃フェーズ/Tashiro/source_recovery/20261010_MIA元コード回収/originals/アーカイブ/予備戦_攻撃フェーズ/work_20260901以前/outcome_mia_redteam.py')

def gen(bg,d,full=False):
 models={};raw=r.cached_raw(bg,d) if full else None
 for name,cols,cal in [('Q_prefecture_sex_shrink500_cross01',r.u.NUM7,'county500'),('drop_ALT_cross01',tuple(x for x in r.u.NUM7 if x!='ALT'),'raw')]:
  values=[]
  for j,t in enumerate(d['teams']):
   x=raw[:,j] if raw is not None and cols==r.u.NUM7 else r.u.base.core(d['a'],d['query'],d['cs'][t],cols,(10,))[10]
   values.append(r.bc.variant(x,d,t,cal))
  models[name]=r.cross(np.column_stack(values))
 return models

def legacy_context(bg,d,historical):
 # Preserve legacy parser and private/public input roles exactly for regression.
 b=pd.read_csv(PRIVATE/'prelim_share_team05_B.csv');practice=pd.read_csv(PRIVATE/'starter_B_practice.csv')
 a=pd.read_csv(f'/workspace/shared/mia_bg7_inputs/bg/subsets/{bg}.csv')
 second=pd.read_csv('/workspace/shared/mia_bg7_inputs/bg/subsets/starter.csv') if historical else a.iloc[:0].copy()
 public=[a,second,practice,practice] if historical else [a,practice,practice]
 cs={t:pd.read_csv(d['targets']/f'C_{t}.csv') for t in d['teams']}
 gates,diag=r.legacy.fixed_public_gates(b,a,second,public,cs,d['teams'])
 return dict(b=b,a=a,second=second,cs=cs,gates=gates,diagnostics=diag,outcome=r.legacy.outcome_functions(OUTCOME))

def legacy_predict(d,context,fraction,seed):
 a=context['a'].copy() if fraction==1 else r.u.cal.stratified_subset(context['a'],fraction,seed)
 background=pd.concat([a,context['second']],ignore_index=True)
 distances=r.legacy.distance_matrix(context['b'],context['cs'],background,d['teams'])
 values=r.legacy.r2_confidence(context['b'],d['rare'],distances,context['gates'],d['teams'])
 return {r.legacy.R2:values,
 r.legacy.EXACT:np.column_stack([context['outcome'].event_time_exact(d['cs'][t],d['a'],d['b']) for t in d['teams']]),
 r.legacy.NEAR:np.column_stack([context['outcome'].event_time_near(d['cs'][t],d['a'],d['b']) for t in d['teams']])}

def run(bg,historical=False):
 out=r.HERE/'cache/fixed22'/('historical_reference_roles' if historical else bg)
 if (out/'complete.json').exists():return
 out.mkdir(parents=True,exist_ok=True);start=time.time();d=r.data(bg)
 context=legacy_context(bg,d,historical);r.dump(out/'legacy_gate_audit.json',context['diagnostics'])
 for view,fraction,seed in r.INNER:
  target=out/view
  if (target/'prediction_record.json').exists():continue
  changed=d.copy();changed['a']=d['a'].copy() if fraction==1 else r.u.cal.stratified_subset(d['a'],fraction,seed)
  models={}
  for family,fn in [('geometry',r.narrow_geometry),('histlocal',r.narrow_histlocal),('gen',lambda z:gen(bg,z,full=fraction==1)),('legacy',lambda z:legacy_predict(z,context,fraction,seed))]:
   # Numeric families are role-independent; exact historical inputs reuse prelim.
   familyout=(r.HERE/'cache/fixed22/prelim'/view/family) if historical and family!='legacy' else target/family
   if (familyout/'prediction_record.json').exists():part=r.load_scores(familyout)
   else:
    t0=time.time();part=fn(changed);r.seal_scores(familyout,part,changed,{'outer_background':bg,'inner_view':view,'family':family,'seconds':time.time()-t0})
   models.update(part)
   print('FIXED22_FAMILY',bg,historical,view,family,len(part),round(time.time()-start,1),flush=True);gc.collect()
  assert len(models)==22,len(models)
  r.seal_scores(target,models,d,{'outer_background':bg,'inner_view':view,'inner_rows':len(changed['a']),'age_bins':'outer full-pool cuts frozen across inner folds, matching original algorithm','two_role_mode':'original prelim+starter' if historical else 'selected unified pool exactly once'})
 r.dump(out/'complete.json',{'outer_background':bg,'historical_roles':historical,'inner_views':r.INNER,'methods':22,'seconds':time.time()-start,'frozen_utc':datetime.now(timezone.utc).isoformat(),'truth_read':False})
 print('FIXED22_COMPLETE',bg,historical,round(time.time()-start,1),flush=True)

def score(bg,historical=False):
 out=r.HERE/'cache/fixed22'/('historical_reference_roles' if historical else bg);assert (out/'complete.json').exists()
 d=r.data(bg);y=r.truth(d);rows=[]
 for view,_,_ in r.INNER:
  for method,x in r.load_scores(out/view).items():rows.extend(r.metrics_rows(x,d,y,background=bg,condition=view,method=method))
 table=pd.DataFrame(rows);table.to_csv(out/'by_target.csv.tmp',index=False,float_format='%.17g');(out/'by_target.csv.tmp').replace(out/'by_target.csv')
 if historical:
  old=pd.read_csv(r.SNAP/'20261008_非学習方式と加工特徴_背景耐性/固定方式_全背景評価.csv');old=old[np.isclose(old.fpr,.01)]
  actual=table[table.population=='historical_native'];joined=actual.merge(old,on=['method','team','condition']);err=float((joined['tpr_0.01']-joined.tpr).abs().max())
  r.dump(out/'historical_regression.json',{'max_TPR_error':err,'matched_rows':len(joined),'expected_rows':len(old),'source_sha256':r.sha(r.SNAP/'20261008_非学習方式と加工特徴_背景耐性/固定方式_全背景評価.csv')})
  assert len(joined)==len(old) and err<1e-12,(len(joined),len(old),err)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--backgrounds',nargs='+',default=list(r.BG));p.add_argument('--historical',action='store_true');p.add_argument('--score',action='store_true');a=p.parse_args()
 for bg in a.backgrounds:
  run(bg,a.historical)
  if a.score:score(bg,a.historical)
