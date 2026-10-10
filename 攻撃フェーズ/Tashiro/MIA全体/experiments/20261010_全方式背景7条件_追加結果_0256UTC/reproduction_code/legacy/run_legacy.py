"""Exact Oct 7 source rerun with seven background substitutions and shared caching.
No repository or external writes. No main labels or invented deployment models.
"""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import argparse,sys,json,time,hashlib,importlib,platform,traceback,types,ast,fcntl
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from joblib import Memory
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source_oct07'))
import 分析 as common
MODULES={'additional':'追加検討','reference':'参照データの検討','bootstrap':'ブートストラップの検討','attributes':'属性部分集合の検討','logistic':'学習による順位補正の検討','qcal':'非会員を使う順位較正の検討','fixed_sensitivity':'固定候補の感度検証','qcal_sensitivity':'非会員較正の感度検証','proxy':'対照を揃えた補助検証'}
BGS=['prelim','starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
class FrozenMain(Exception): pass

def install_cache():
 memory=Memory(BASE/'cache',verbose=0)
 original_core=common.core
 cached_core=memory.cache(original_core)
 def core(reference,candidates,synthetic,columns,ks=(1,3,10),whiten=False):
  # All original k requests are subsets of 1,3,10; canonicalization avoids repeated kernels.
  canonical=tuple(sorted(set((1,3,10))|set(ks)))
  got=cached_core(reference,candidates,synthetic,tuple(columns),canonical,whiten)
  return {k:got[k] for k in ks}
 common.core=core
 for module,names in [('追加検討',['point_scores']),('参照データの検討',['weighted_core','domain_weights']),('ブートストラップの検討',['score_draws']),('非会員を使う順位較正の検討',['background_loo_scores'])]:
  mod=importlib.import_module(module)
  for name in names: setattr(mod,name,memory.cache(getattr(mod,name)))
 # The original scorer is staged byte-exact from the pinned repository.
 sys.path.insert(0,'/workspace/shared/mia_bg7_inputs/official_scoring/src')
 from pwscup2026.scoring import attack_score
 def official_roc(root):
  common.INPUT_HASHES[str(Path(attack_score.__file__).resolve())]=common.sha(Path(attack_score.__file__))
  return attack_score._tpr_at_fpr
 importlib.import_module('低FPRと対象偏りの検討').official_roc=official_roc
 return memory

def geometry(a,b,cs,normal):
 computed={t:common.geometry_scores(a,b,c) for t,c in cs.items()}
 matrices={name:np.column_stack([rankdata(computed[t][name][normal],method='average')/normal.sum() for t in cs]) for name in next(iter(computed.values()))}
 for name in ('global7','global9','sex7','sex9','sex_whiten7'):
  mat=matrices[name]; matrices[name+'_cross025']=mat-.25*np.median(mat,axis=1)[:,None]
 for name,mat in common.candidate_contrast(np.column_stack([computed[t]['global7'] for t in cs]),normal).items(): matrices['sensitivity_'+name]=mat
 return matrices

def summarize(table):
 out=[]
 for method,g in table.groupby('method',sort=False):
  for scope,gg in [('all20',g),('without1_2',g[~g.team.isin([1,2])])]:
   out.append({'method':method,'target_scope':scope,'targets':len(gg),'mia_all':gg.mia_all.mean(),'auc':gg.auc.mean()})
 return pd.DataFrame(out)

def run(bg,family,phase='prelim'):
 if phase=='main' and family in ['logistic','proxy']: raise ValueError('No defined main deployment for this family')
 out=BASE/'results'/phase/bg/family
 if (out/'completion.json').exists(): print('DONE_CACHE',phase,bg,family,flush=True);return
 if out.exists(): raise FileExistsError(f'Incomplete output requires audit before retry: {out}')
 out.parent.mkdir(parents=True,exist_ok=True)
 freeze=BASE/'frozen'/phase/bg/family; freeze.mkdir(parents=True,exist_ok=True)
 started=time.perf_counter(); original_phase=common.phase_inputs; original_eval=common.evaluate_prelim
 bgpath=Path('/workspace/shared/mia_bg7_inputs/bg/subsets')/(bg+'.csv')
 bgframe=common.read(bgpath)
 def inputs(root,requested):
  actual='main' if phase=='main' else requested
  b,a,q,p,r,cs,t,s=original_phase(root,actual)
  return b,bgframe,q,p,r,cs,t,s
 common.phase_inputs=inputs
 captures=[]
 def evaluate(root,b,private,rare,matrices,teams):
  normal=~private&~rare
  assert len(teams)==(20 if phase=='prelim' else 22)
  assert all(np.isfinite(mat).all() for mat in matrices.values())
  # Freeze every method before the original scorer may read labels.
  np.savez_compressed(freeze/'matrices.npz',**matrices)
  hashes={name:hashlib.sha256(np.asarray(mat,dtype='<f8').tobytes()).hexdigest() for name,mat in matrices.items()}
  (freeze/'manifest.json').write_text(json.dumps({'phase':phase,'background':bg,'family':family,'teams':list(map(int,teams)),'methods':list(matrices),'prediction_sha256':hashes,'evaluation_labels_read_after_prediction_freeze':True,'training_labels_used':family=='logistic','training_label_protocol':('person5_target4_joint_exclusion_OOF' if family=='logistic' else 'no_MIA_training_labels'),'n_background':len(bgframe),'normal_rows':int(normal.sum())},indent=2,ensure_ascii=False))
  captures.append(len(matrices))
  if phase=='main':
   out.mkdir(parents=True)
   base=matrices.get('global7',next(iter(matrices.values())))
   rows=[]
   for name,mat in matrices.items():
    for j,team in enumerate(teams):
     x=base[:,j]; y=mat[:,j]; rx=rankdata(x);ry=rankdata(y)
     rows.append({'team':team,'method':name,'top12_overlap_baseline':len(set(np.argsort(-x,kind='stable')[:12])&set(np.argsort(-y,kind='stable')[:12])),'top20_overlap_baseline':len(set(np.argsort(-x,kind='stable')[:20])&set(np.argsort(-y,kind='stable')[:20])),'spearman_baseline':float(np.corrcoef(rx,ry)[0,1]),'mia_all':None,'label_status':'unavailable_not_used'})
   pd.DataFrame(rows).to_csv(out/'main_ranking_diagnostics.csv',index=False)
   raise FrozenMain()
  rows,agg=original_eval(root,b,private,rare,matrices,teams)
  # Additional canonical summaries are separate from historical target-scope outputs.
  table=pd.DataFrame(rows)
  table.to_csv(freeze/'by_target.csv',index=False,float_format='%.17g')
  summarize(table).to_csv(freeze/'summary_scopes.csv',index=False,float_format='%.17g')
  return rows,agg
 common.evaluate_prelim=evaluate
 status='completed'
 try:
  if family=='geometry':
   b,a,q,p,r,cs,t,s=inputs(BASE/'compat_root','prelim')
   mats=geometry(a,b,cs,~p&~r)
   rows,agg=evaluate(BASE/'compat_root',b,p,r,mats,list(cs))
   out.mkdir(parents=True)
   pd.DataFrame(rows).to_csv(out/'予備戦_対象別比較.csv',index=False,float_format='%.17g')
   pd.DataFrame(agg).to_csv(out/'予備戦_集約比較.csv',index=False,float_format='%.17g')
  else:
   module=importlib.import_module(MODULES[family])
   if family in ['fixed_sensitivity','qcal_sensitivity'] and bg not in ['prelim','starter']:
    # Requested source subsets replace old BG-only perturbations. The historical
    # prelim/starter expansions already completed are retained as supplements.
    source=Path(module.__file__).read_text();tree=ast.parse(source)
    for node in tree.body:
     if isinstance(node,ast.FunctionDef) and node.name=='main':
      removed=[v for v in node.body if isinstance(v,ast.AugAssign) and isinstance(v.target,ast.Name) and v.target.id=='references']
      assert len(removed)==1
      node.body=[v for v in node.body if v not in removed]
    adapted=types.ModuleType(family+'_runtime');adapted.__file__=module.__file__
    exec(compile(tree,module.__file__,'exec'),adapted.__dict__);module=adapted
   if family=='logistic':
    # The source's positive-sign baseline equivalence is a historical expectation,
    # not a constraint on newly fitted coefficients under a changed background.
    src=Path(module.__file__).read_text()
    old='assert np.max(np.abs(value - matrices[context + "__baseline"])) < 1e-12'
    new='MODEL_DIAGNOSTICS.append({"context":context,"training":training,"model":model,"baseline_max_abs_difference":float(np.max(np.abs(value - matrices[context + "__baseline"]))),"baseline_equivalent":bool(np.max(np.abs(value - matrices[context + "__baseline"])) < 1e-12)})'
    assert src.count(old)==1
    adapted=types.ModuleType('logistic_runtime');adapted.__file__=module.__file__;adapted.MODEL_DIAGNOSTICS=[]
    exec(compile(src.replace(old,new),module.__file__,'exec'),adapted.__dict__)
    module=adapted
   sys.argv=[str(Path(module.__file__)),'--root',str(BASE/'compat_root'),'--output-dir',str(out)]
   module.main()
   if family=='logistic': (out/'rank1_equivalence_diagnostics.json').write_text(json.dumps(module.MODEL_DIAGNOSTICS,indent=2))
 except FrozenMain: pass
 finally:
  common.phase_inputs=original_phase;common.evaluate_prelim=original_eval
 record={'phase':phase,'background':bg,'family':family,'seconds':time.perf_counter()-started,'methods':captures,'python':platform.python_version(),'background_sha256':common.sha(bgpath),'source_hashes':{p.name:common.sha(p) for p in (BASE/'source_oct07').glob('*.py')},'input_sha256':common.INPUT_HASHES,'wrapper_sha256':common.sha(Path(__file__)),'status':status,'main_membership_truth_used':False,'outputs_only_local':True,'background_only_sensitivity_contexts':('replaced_by_requested_external_background' if family in ['fixed_sensitivity','qcal_sensitivity'] and bg not in ['prelim','starter'] else 'original_contexts_retained')}
 out.mkdir(parents=True,exist_ok=True)
 (out/'completion.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print('COMPLETE',phase,bg,family,record['seconds'],captures,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--background',choices=BGS,required=True);p.add_argument('--family',choices=['geometry']+list(MODULES),required=True);p.add_argument('--phase',choices=['prelim','main'],default='prelim');args=p.parse_args()
 (BASE/'locks').mkdir(exist_ok=True)
 with (BASE/'locks'/f'{args.phase}__{args.background}.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  install_cache();run(args.background,args.family,args.phase)
