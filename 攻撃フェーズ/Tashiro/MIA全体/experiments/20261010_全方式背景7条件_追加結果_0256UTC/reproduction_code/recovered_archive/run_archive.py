"""Recovered original numerical recipes, public inputs only, no archived CLI execution.
AST slices retain original numeric expressions; adapters only route inputs and freeze outputs.
"""
from pathlib import Path
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[_k]='1'
import sys, ast, types, importlib.util, json, hashlib, time, argparse, traceback, platform
sys.dont_write_bytecode=True
sys.path.append('/workspace/shared/mia_archive_dependencies')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
BASE=Path(__file__).resolve().parent
REC=Path('/workspace/shared/mia_recovered_sources/攻撃フェーズ/Tashiro/source_recovery/20261010_MIA元コード回収')
ORIG=REC/'originals'
WRAPPERS=Path('/workspace/shared/mia_all_background_inventory/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/source_snapshot/攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_帯別横断検証')
LEGACY=BASE.parent/'legacy'
INPUT=Path('/workspace/shared/mia_bg7_inputs')
BGS=['prelim','starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
TEAMS=[1,2,3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
NUM7=['age','BMI','SBP','TG','HDL','ALT','FPG']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m
 if Path(path).name=='historical_virtual.py':
  tree=ast.parse(Path(path).read_text())
  for node in ast.walk(tree):
   if isinstance(node,ast.Call):
    for kw in node.keywords:
     if kw.arg=='max_workers':kw.value=ast.copy_location(ast.Constant(value=1),kw.value)
  exec(compile(tree,str(path),'exec'),m.__dict__)
 else:s.loader.exec_module(m)
 return m

def setup(bg):
 root=BASE/'compat'/bg
 root.mkdir(parents=True,exist_ok=True)
 for p in ORIG.iterdir():
  if p.name=='予備戦':continue
  q=root/p.name
  if not q.exists():q.symlink_to(p,target_is_directory=True)
 pdest=root/'予備戦/配布データ';pdest.mkdir(parents=True,exist_ok=True)
 src=LEGACY/'compat_root/予備戦/配布データ'
 for p in src.iterdir():
  q=pdest/p.name
  if p.name=='PWSCup2026_prelim_common':
   q.mkdir(exist_ok=True);a=q/'A_bg.csv'
   if not a.exists():a.symlink_to(INPUT/'bg/subsets'/f'{bg}.csv')
  elif not q.exists():q.symlink_to(p,target_is_directory=p.is_dir())
 wp=root/'wrappers';wp.mkdir(exist_ok=True)
 for p in WRAPPERS.glob('*.py'):
  q=wp/p.name
  if not q.exists():q.write_bytes(p.read_bytes())
 return root

class TargetToken:
 """Team identifier, not membership labels. Used only by intercepted evaluation calls."""
 def __init__(self,team):self.team=team
 def sum(self):return 0

class Capture:
 def __init__(self,root,teams):
  self.root=root;self.teams=teams;self.mats={};self.meta={};self.queue=[]
  self.audit=module('_archive_audit',root/'wrappers/official_audit.py')
  self.real_load_inputs=self.audit.load_inputs
  self.audit.load_truth=lambda root:{k:TargetToken(k) for k in teams}
  self.audit.load_official=lambda root:types.SimpleNamespace(_tpr_at_fpr=self.capture_score,_source_paths=[])
 def capture_score(self,scores,token,fpr):
  assert isinstance(token,TargetToken) and fpr==.01
  self.queue.append((token.team,np.asarray(scores).copy()))
  return 0.,0. # placeholders are never published; official evaluation follows freeze.
 def add_matrix(self,name,scores,**meta):
  assert name not in self.mats
  self.mats[name]=np.asarray(scores).copy();self.meta[name]=meta
 def load(self,name,path):
  if Path(path).name=='official_audit.py':return self.audit
  return module(name,path)
 def collect_records(self,records):
  assert len(records)==len(self.queue),(len(records),len(self.queue))
  for r,(k,s) in zip(records,self.queue):
   assert r['team']==k
   name=r['method'];self.mats.setdefault(name,np.full((len(s),len(self.teams)),np.nan))[:,self.teams.index(k)]=s
   self.meta.setdefault(name,{key:v for key,v in r.items() if key not in ['team','tpr','auc','members','rank_signature']})

# Source line intervals refer to the hash-pinned wrappers. These delimit original
# self-contained numeric recipe sections and exclude saved-only file loading.
RANGES={'library':(237,253),'cross':(254,280),'density':(283,339),'redteam':(346,414),'outcome':(416,421),'aia2mia':(425,450)}

def run_block(family,root,cap):
 path=root/'wrappers/historical_results.py';tree=ast.parse(path.read_text())
 wrapper=module('_archive_history',path);env=wrapper.__dict__.copy()
 b,private,rare,normal,dist=cap.real_load_inputs(root)
 bgpath=root/'予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv';bg=pd.read_csv(bgpath,float_precision='round_trip')
 cs={k:pd.read_csv(dist/f'C_{k}.csv',float_precision='round_trip') for k in cap.teams}
 old=root/'アーカイブ/予備戦_攻撃フェーズ';attack=module('_archive_attacks',old/'work_20260901以前/attacks.py')
 env.update(root=root,archive=old,b=b,bg=bg,ids=b.record_id.to_numpy(),rare=rare,private=private,dist=dist,cs=cs,teams=cap.teams,
  mu=bg[NUM7].mean().to_numpy(),sd=bg[NUM7].std(ddof=0).to_numpy(),attack=attack,source_paths=set(),add_matrix=cap.add_matrix,
  note=lambda *args:cap.meta.setdefault('_notes',[]).append(args),_module=cap.load)
 lo,hi=RANGES[family];body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gather').body
 selected=[n for n in body if lo<=n.lineno<=hi]
 if family=='outcome':
  # Reuse the original pure AST extractor definition, independent of Cox modules.
  selected=[n for n in body if isinstance(n,ast.FunctionDef) and n.name=='extracted']+selected
  from sklearn.neighbors import NearestNeighbors
  env['env']=dict(np=np,pd=pd,NearestNeighbors=NearestNeighbors)
 assert selected,(family,lo,hi)
 
 for node in ast.walk(ast.Module(body=selected,type_ignores=[])):
  if isinstance(node,ast.Call):
   for kw in node.keywords:
    if kw.arg=='max_workers':kw.value=ast.copy_location(ast.Constant(value=1),kw.value)
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec',flags=__import__('__future__').annotations.compiler_flag),env)


def run_gen(root,cap):
 gen=module('_archive_initial_gen',root/'本戦/攻撃フェーズ/Tashiro/20261005_MIA初回_GenLRA比較/gen_lra.py')
 b,p,r,n,dist=cap.real_load_inputs(root);a=pd.read_csv(root/'予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv',float_precision='round_trip')
 mats={name:np.empty((len(b),len(cap.teams))) for name in gen.VARIANTS}
 for j,k in enumerate(cap.teams):
  c=pd.read_csv(dist/f'C_{k}.csv',float_precision='round_trip');raw,out=gen.calculate(a,b.drop(columns='is_rare'),c,r,p)
  for name,s in out.items():mats[name][:,j]=s
  print('initial_gen',j+1,len(cap.teams),flush=True)
 for name,s in mats.items():cap.add_matrix('initial_20261005_'+name,s,params=gen.VARIANTS[name],source='gen_lra.py#calculate')


def extract(path,names,environment,assignments=()):
 tree=ast.parse(path.read_text());selected=[]
 for n in tree.body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:selected.append(n)
  elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignments for t in n.targets):selected.append(n)
 assert {n.name for n in selected if isinstance(n,(ast.FunctionDef,ast.ClassDef))}==set(names)
 m=types.ModuleType('_pure_'+path.parent.name);m.__dict__.update(environment);sys.modules[m.__name__]=m
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec',flags=__import__('__future__').annotations.compiler_flag),m.__dict__);return m

def run_raw_gen(root,cap):
 import math
 folder=ORIG/'予備戦/攻撃フェーズ/分析'
 gen=extract(folder/'20_Team20希少のGenLRA検証/分析.py',['numeric','distance2','logsumexp','category_frequencies','log_category','log_kernel','influence'],dict(np=np,pd=pd,math=math,RHO=.98))
 a21=extract(folder/'21_MIA全体と希少のGenLRA比較/分析.py',['compute'],dict(np=np,pd=pd,math=math,gen=gen))
 num5=['age','SBP','TG','HDL','ALT'];num7=['age','BMI','SBP','TG','HDL','ALT','FPG'];cats=['sex','prefecture']
 a22=extract(folder/'22_GenLRAの7属性9属性10属性比較/分析.py',['smoke_decomposition'],dict(np=np,pd=pd,math=math,gen=gen,NUM9=num7,CAT9=cats))
 b,p,r,n,dist=cap.real_load_inputs(root);bg=pd.read_csv(root/'予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv',float_precision='round_trip');mats={}
 for j,k in enumerate(cap.teams):
  c=pd.read_csv(dist/f'C_{k}.csv',float_precision='round_trip');vals={}
  for prefix,nums,cc in [('qi7',num5,cats),('attrs10',num7,cats+['smoking'])]:
   o=a21.compute(bg,c,b,nums,cc)
   for name in ['gen','distance','point']:vals[prefix+'_'+name]=o[name]
  o=a22.smoke_decomposition(bg,c,b)
  for name in ['attrs9','smoking_kernel_only','smoking_neighbors_only']:vals[name]=o[name]
  for name,s in vals.items():mats.setdefault(name,np.empty((len(b),len(cap.teams))))[:,j]=s
  print('raw_gen',j+1,len(cap.teams),flush=True)
 for name,s in mats.items():cap.add_matrix('legacy_raw_'+name,s,source='analysis21/22 pure functions')

def run_domias(root,cap):
 from scipy.special import expit
 mod=module('_archive_domias',root/'アーカイブ/予備戦_攻撃フェーズ/過去のコード/mia_domias/build_F_mia_domias.py')
 b,p,r,n,dist=cap.real_load_inputs(root);a=pd.read_csv(root/'予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv',float_precision='round_trip')
 p_self=mod.identify_nonshared(b,pd.read_csv(dist/'aia_challenge_5.csv',float_precision='round_trip'));assert np.array_equal(p_self,p)
 home=(b.prefecture==b.loc[p_self,'prefecture'].mode().iloc[0]).to_numpy();rate=(int(home.sum())-313)/int((home&~p_self).sum())
 z=mod.zscaler(a,mod.CONTINUOUS);query=z(b);ref=mod.fit_kde(z(a),1.);assert ref is not None;lref=mod.logpdf(ref,query)
 mats={name:np.empty((len(b),len(cap.teams))) for name in ['out','out_prior','out_w07']}
 for j,k in enumerate(cap.teams):
  c=pd.read_csv(dist/f'C_{k}.csv',float_precision='round_trip');syn=mod.fit_kde(z(c),1.)
  base=mod.logpdf(syn,query)-lref if syn is not None else None
  for name,w in [('out',0.),('out_prior',1.),('out_w07',.7)]:
   conf=expit(base+np.where(home&~p_self,w*np.log(rate),0.)) if base is not None else np.full(len(b),.5)
   mats[name][:,j]=np.where(p_self,0.,conf)
 for name,s in mats.items():cap.add_matrix('legacy_domias_'+name,s,params={'bw':1.,'prior_weight':{'out':0.,'out_prior':1.,'out_w07':.7}[name],'zeroed':True,'home_share_rate':rate,'n_nonshared':313},source='build_F_mia_domias.py')

def _run(bg,family,bench=False):
 if family=='gen_composites_available':
  import run_composites,run_followup
  for selected in BGS:
   if (BASE/'results'/selected/'raw_gen/completion.json').exists():run_composites.run(selected)
  run_followup.aggregate();return
 if family in ['r1_followup','r1_followup_all']:
  import run_followup
  for selected in BGS if family=='r1_followup_all' else [bg]:run_followup.run(selected)
  run_followup.aggregate();return
 dest=BASE/('benchmarks' if bench else 'results')/bg/family
 if (dest/'completion.json').exists():print('CACHED',bg,family,flush=True);return
 dest.mkdir(parents=True,exist_ok=True)
 root=setup(bg);teams=[7] if bench else TEAMS;cap=Capture(root,teams);start=time.perf_counter()
 if family in ['redteam','virtual']:
  from lifelines import CoxPHFitter
  original_cox_fit=CoxPHFitter.fit
  def audited_cox_fit(self,*args,**kwargs):
   try:return original_cox_fit(self,*args,**kwargs)
   except Exception as exc:
    cap.meta.setdefault('_cox_source_fallback_exceptions',[]).append({'type':type(exc).__name__,'message':str(exc),'n_rows':len(args[0]) if args else None})
    raise
  CoxPHFitter.fit=audited_cox_fit
 with threadpool_limits(limits=1):
  if family in RANGES:run_block(family,root,cap)
  elif family in ['atk','virtual']:
   if bench and family=='atk':raise ValueError('atk benchmark needs original 24/26/29 RNG order')
   m=module('_archive_'+family,root/'wrappers'/f'historical_{family}.py');m._module=cap.load;m._sha=lambda path:'provenance_recorded_separately_without_reading_truth'
   records,notes=m.gather(root,use_cache=False);cap.collect_records(records);cap.meta['_notes']=notes
  elif family=='initial_gen':run_gen(root,cap)
  elif family=='raw_gen':run_raw_gen(root,cap)
  elif family=='domias':run_domias(root,cap)
  else:raise ValueError(family)
 freeze=dest/'frozen.npz';np.savez_compressed(freeze,**cap.mats)
 elapsed=time.perf_counter()-start
 manifest={'background':bg,'family':family,'teams':teams,'methods':list(cap.mats),'n_methods':len(cap.mats),'seconds':elapsed,
  'applicable_target_ids':{name:[teams[j] for j in range(len(teams)) if np.isfinite(mat[:,j]).all()] for name,mat in cap.mats.items()},'n_background':5143*len(bg.split('+')),'freeze_sha256':sha(freeze),'predictions_frozen_before_evaluation_labels':True,'membership_training_labels_used':False,
  'numpy':np.__version__,'pandas':pd.__version__,'scipy':__import__('scipy').__version__,'sklearn':__import__('sklearn').__version__,'lifelines':__import__('lifelines').__version__,'python':platform.python_version(),'metadata':cap.meta}
 for _name,_meta in cap.meta.items():
  if isinstance(_meta,dict):
   if 'background' in _meta:_meta['source_historical_background_label']=_meta.pop('background')
   _meta['background_condition']=bg;_meta['n_selected_background']=manifest['n_background']
 (dest/'prediction_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str)+'\n')
 assert cap.mats
 # Only here are actual preliminary evaluation labels read.
 audit=module('_archive_real_eval',root/'wrappers/official_audit.py');truth=audit.load_truth(root)
 sys.path.insert(0,str(INPUT/'official_scoring/src'))
 from pwscup2026.scoring.attack_score import _tpr_at_fpr
 rows=[]
 for name,mat in cap.mats.items():
  assert mat.shape==(1049,len(teams))
  for j,k in enumerate(teams):
   if np.isnan(mat[:,j]).all():continue # original 3-target split only
   assert np.isfinite(mat[:,j]).all(),(name,k)
   tpr,auc=_tpr_at_fpr(mat[:,j],truth[k],.01)
   rows.append(dict(background=bg,family=family,method=name,team=k,tpr=tpr,auc=auc,members=int(truth[k].sum())))
 table=pd.DataFrame(rows);table.to_csv(dest/'by_target.csv',index=False,float_format='%.17g')
 summary=[]
 for name,g in table.groupby('method'):
  for scope,gg in [('all_available',g),('without1_2',g[~g.team.isin([1,2])]),('historical_without12',g[g.team!=12])]:
   if len(gg):summary.append(dict(background=bg,family=family,method=name,scope=scope,n_targets=len(gg),tpr=gg.tpr.mean(),auc=gg.auc.mean()))
 pd.DataFrame(summary).to_csv(dest/'summary.csv',index=False,float_format='%.17g')
 (dest/'completion.json').write_text(json.dumps({k:v for k,v in manifest.items() if k!='metadata'},ensure_ascii=False,indent=2)+'\n')
 print('COMPLETED',bg,family,len(cap.mats),'seconds',elapsed,flush=True)
 if family=='raw_gen':
  import run_composites,run_followup
  run_composites.run(bg);run_followup.aggregate()

def run(bg,family,bench=False):
 import fcntl
 folder=BASE/'locks';folder.mkdir(exist_ok=True)
 with (folder/(('bench_' if bench else '')+bg+'__'+family+'.lock')).open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
  return _run(bg,family,bench)

def take_priority_if_available():
 import fcntl
 request=BASE/'priority_request.json'
 if not request.exists():return
 folder=BASE/'locks';folder.mkdir(exist_ok=True)
 with (folder/'priority_dispatch.lock').open('a') as lock:
  try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  state=json.loads(request.read_text())
  if state['status']!='pending':return
  state.update(status='running',started_at=time.time());request.write_text(json.dumps(state,indent=2)+'\n')
  print('PRIORITY_START',state['background'],state['family'],flush=True)
  try:run(state['background'],state['family'])
  except Exception as exc:
   state.update(status='failed',error=type(exc).__name__+': '+str(exc),finished_at=time.time());request.write_text(json.dumps(state,indent=2)+'\n');raise
  state.update(status='completed',finished_at=time.time());request.write_text(json.dumps(state,indent=2)+'\n')
  print('PRIORITY_COMPLETED',state['background'],state['family'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bg',required=True,choices=BGS);p.add_argument('--family',required=True);p.add_argument('--bench',action='store_true');a=p.parse_args();take_priority_if_available() if not a.bench else None;run(a.bg,a.family,a.bench)
