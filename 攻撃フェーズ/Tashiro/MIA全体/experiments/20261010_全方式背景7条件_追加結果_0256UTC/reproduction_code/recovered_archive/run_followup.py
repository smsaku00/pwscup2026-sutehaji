"""Six follow-up archive recipes. Numerical source AST only; no archived CLI side effects."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import ast,copy,json,time,types,sys,hashlib,argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import run_archive as ar
ROOT=ar.BASE/'followup'
SOURCE=ar.ORIG/'アーカイブ/予備戦_攻撃フェーズ/work_20260901以前'
METHODS=['legacy_saved_work_20260901以前__prelim_attack__r1__F_mia','legacy_saved_work_20260901以前__prelim_attack2__r1__F_mia']

def run_ast(path,body,env):
 exec(compile(ast.Module(body=copy.deepcopy(body),type_ignores=[]),str(path),'exec'),env)

def generate(bg,version):
 compat=ar.setup(bg);dist=compat/'予備戦/配布データ/PWSCup2026_prelim_attack'
 bpath=next((compat/'予備戦/配布データ').glob('PWSCup2026_prelim_team05*/B_self.csv'))
 # Both historical builders use pandas default CSV floating-point parser.
 Bs=pd.read_csv(bpath);A=pd.read_csv(ar.INPUT/'bg/subsets'/f'{bg}.csv')
 path=SOURCE/('build_attack.py' if version==1 else 'build_attack2.py')
 tree=ast.parse(path.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 env=dict(np=np,pd=pd,NearestNeighbors=NearestNeighbors,A=A,Bs=Bs,B=Bs.drop(columns='is_rare'),Bd=Bs.drop(columns='is_rare'),targets=ar.TEAMS,DIST=dist,a=types.SimpleNamespace(round=1,hit=[]),st={'found':{},'ruled':{}},b=types.SimpleNamespace(is_rare=Bs.is_rare.to_numpy(bool)),CONT=ar.NUM7)
 if version==1:
  helper=ast.parse((SOURCE/'submit_audit.py').read_text())
  funcs=[n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='_z']
  defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='nn_dist']
  assigns=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['QI','TOKYO'] for t in n.targets)]
  cols=[n for n in helper.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='COLS13' for t in n.targets)]
  run_ast(path,funcs+defs+assigns+cols,env)
  body=[copy.deepcopy(n) for n in main.body if 43<=n.lineno<=132 and n.lineno!=109]
  for n in body:
   if isinstance(n,ast.For) and n.lineno==60:n.body=[x for x in n.body if x.lineno<=70]
 else:
  funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['zmat','key_index','nn_by_key']]
  assigns=[n for n in tree.body if isinstance(n,ast.Assign) and n.lineno in [24,25,26,27,28]]
  # Select constants by assigned names, including SELF,HOME,CENS tuple.
  assigns=[n for n in tree.body if isinstance(n,ast.Assign) and set(x.id for t in n.targets for x in ast.walk(t) if isinstance(x,ast.Name)) & {'SELF','HOME','CENS','LAB','QI5','QI4','KEY'}]
  run_ast(path,funcs+assigns,env)
  body=[copy.deepcopy(n) for n in main.body if (70<=n.lineno<=155 and n.lineno not in [73]) or 159<=n.lineno<=172 or 181<=n.lineno<=203]
  for n in body:
   if isinstance(n,ast.For) and n.lineno==104:n.body=[x for x in n.body if x.lineno<=121]
 run_ast(path,body,env)
 F=env['F'];assert list(F.columns)==['record_id']+[str(t) for t in ar.TEAMS]
 assert F.shape==(1049,21)
 return F.iloc[:,1:].to_numpy(),dict(source=str(path.relative_to(ar.REC)),source_sha256=ar.sha(path),source_ast_top_level_intervals=[[43,132]] if version==1 else [[70,155],[159,172],[181,203]],excluded=['AIA calculations','submission/token/file writes','state mutation','CLI and historical path loading'],round=1,n_background=len(A),historical_state='not_used' if version==1 else 'empty found/ruled; saved report nomination matches this formula, historical pre-run state unavailable',historical_generation_input_hashes='not recorded; current official input hashes match relocated historical copies',parser='pandas default; original builder behavior',applicable_target_ids=ar.TEAMS)

def evaluate(bg,mats,dest):
 # Evaluation truth first accessed only after frozen array and manifest were written.
 audit=ar.module('_followup_audit',ar.WRAPPERS/'official_audit.py');truth=audit.load_truth(ar.setup(bg))
 sys.path.insert(0,str(ar.INPUT/'official_scoring/src'))
 from pwscup2026.scoring.attack_score import _tpr_at_fpr
 rows=[]
 for method,mat in mats.items():
  assert mat.shape==(1049,20) and np.isfinite(mat).all()
  for j,k in enumerate(ar.TEAMS):
   tpr,auc=_tpr_at_fpr(mat[:,j],truth[k],.01)
   rows.append(dict(background=bg,method=method,team=k,target_phase='prelim',tpr=tpr,auc=auc,members=int(truth[k].sum())))
 table=pd.DataFrame(rows);table.to_csv(dest/'by_target.csv',index=False,float_format='%.17g')
 summary=[]
 for method,g in table.groupby('method'):
  for scope,gg in [('all_available',g),('without1_2',g[~g.team.isin([1,2])]),('historical_without12',g[g.team!=12])]:
   summary.append(dict(background=bg,method=method,target_phase='prelim',scope=scope,n_targets=len(gg),tpr=gg.tpr.mean(),auc=gg.auc.mean()))
 pd.DataFrame(summary).to_csv(dest/'summary.csv',index=False,float_format='%.17g')

def run(bg):
 dest=ROOT/'results'/bg/'r1';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'completion.json').exists():return
 start=time.perf_counter();mats={};meta={}
 with threadpool_limits(limits=1):
  for v,method in enumerate(METHODS,1):mats[method],meta[method]=generate(bg,v)
 np.savez_compressed(dest/'frozen.npz',**mats)
 manifest=dict(background=bg,family='r1_followup',methods=list(mats),teams=ar.TEAMS,seconds=time.perf_counter()-start,freeze_sha256=ar.sha(dest/'frozen.npz'),predictions_frozen_before_evaluation_labels=True,membership_training_labels_used=False,metadata=meta)
 (dest/'prediction_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 evaluate(bg,mats,dest)
 (dest/'completion.json').write_text(json.dumps({k:v for k,v in manifest.items() if k!='metadata'},ensure_ascii=False,indent=2)+'\n')
 print('COMPLETED FOLLOWUP',bg,manifest['seconds'],flush=True)

def aggregate():
 from scipy.stats import rankdata
 completions=list((ROOT/'results').glob('*/*/completion.json'));tables=[];summ=[]
 for p in completions:tables.append(pd.read_csv(p.parent/'by_target.csv'));summ.append(pd.read_csv(p.parent/'summary.csv'))
 if not tables:return
 tab=pd.concat(tables,ignore_index=True);tab.to_csv(ROOT/'by_target.csv',index=False,float_format='%.17g');pd.concat(summ,ignore_index=True).to_csv(ROOT/'summary.csv',index=False,float_format='%.17g')
 hist=pd.DataFrame(json.loads((ar.WRAPPERS/'historical_cache.json').read_text())['records'])
 base=tab[tab.background=='prelim'].merge(hist[['method','team','tpr','auc']],on=['method','team'],suffixes=('_rerun','_saved'),validate='one_to_one')
 for c in ['tpr','auc']:base['abs_'+c+'_error']=(base[c+'_rerun']-base[c+'_saved']).abs()
 base.to_csv(ROOT/'baseline_by_target.csv',index=False,float_format='%.17g')
 reports=[]
 for method,g in base.groupby('method'):reports.append(dict(method=method,n_targets=len(g),historical_auc_available_rows=int(g.auc_saved.notna().sum()),exact_tpr_rows=int((g.abs_tpr_error<1e-12).sum()),exact_auc_rows=int((g.abs_auc_error<1e-12).sum()),max_tpr_error=g.abs_tpr_error.max(),max_auc_error=g.abs_auc_error.max()))
 pd.DataFrame(reports).to_csv(ROOT/'baseline_validation.csv',index=False,float_format='%.17g')
 (ROOT/'status.json').write_text(json.dumps(dict(completed_jobs=len(completions),completed_method_background_conditions=len(tab.groupby(['method','background'])),target_rows=len(tab),blocked_composites=0,source_missing_recipes=3),indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bg',choices=ar.BGS);p.add_argument('--all',action='store_true');p.add_argument('--wait-primary',action='store_true');a=p.parse_args()
 ROOT.mkdir(exist_ok=True)
 if a.wait_primary:
  while len(list((ar.BASE/'results').glob('*/*/completion.json')))<77:time.sleep(30)
 for bg in ar.BGS if a.all else [a.bg]:
  if bg:run(bg);aggregate()
