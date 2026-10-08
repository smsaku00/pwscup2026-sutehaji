"""Fixed preliminary MIA experiment. No main data or competition submission.
Run at repository root with --output-dir NEW_DIRECTORY. Labels are read only
by audit_truth(), after all predictions and parameters have been frozen.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, platform, sys, time
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix, vstack
from scipy.special import logsumexp, xlogy
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, roc_auc_score

CONFIG = {
 'protocol':'fixed-before-current-label-evaluation-v1','phase':'prelim','self_id':5,
 'N':30,'shared_total':736,'pair_lower':25,'pair_upper':27,
 'primary_inference':'all20','primary_baseline':'P01','primary_soft':'soft_box_0.05',
 'max_penalties':[0.1,0.25,0.5],'soft_temperatures':[0.02,0.05,0.1],
 'dummy_score':0.5,'cross_median':0.1,'shrink':500,
 'backgrounds':['full','sub80_seed20261009','sub80_seed20261010','sub80_seed20261011'],
 'inference_sets':['all20','without1_2','without1_2_12'],
 'evaluation_sets':['all','without12','without1_2','without1_2_12'],
 'alpha':[0.005,0.01,0.02],'top_k':[10,12,25],
 'non_normal_output':'private=0; non-private rare=0.4; normal=(0.6,0.9]',
 'latent_assignment':'all 886 unknown rows, including 11 rare; only normal outputs change',
 'labels':'no individual labels, rare-member counts or exact observed totals used by prediction',
 'scope_limit':'one previously explored preliminary cohort; not independent validation'}
SOURCE_HASHES = {'分析.py': '25cdf97bee3adbda9a5fd8b0793fa0c3f98102366bd1944386fb619d04569aad', '非会員を使う順位較正の検討.py': '9643ac8d512147c7926e6cd6ff816d2adbbb12b8ef8c9e46f31b22ce2b0cb833'}

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def load_inputs(root):
 src=root/'攻撃フェーズ/Tashiro/MIA全体/experiments/20261007'
 for name,h in SOURCE_HASHES.items():
  if digest(src/name)!=h: raise RuntimeError(f'Baseline code changed: {name}')
 sys.path.insert(0,str(src)); common=importlib.import_module('分析'); calib=importlib.import_module('非会員を使う順位較正の検討')
 base=root/'公式データ/予備戦'; own=list(base.glob('PWSCup2026_prelim_team05_*/B_self.csv'))
 if len(own)!=1: raise RuntimeError('Expected one preliminary B_self')
 td=base/'PWSCup2026_prelim_attack'
 paths=[own[0],base/'PWSCup2026_prelim_common/A_bg.csv']+sorted(td.glob('C_*.csv'))+sorted(td.glob('aia_challenge_*.csv'))
 paths += [td/'mia_columns.json',src/'分析.py',src/'非会員を使う順位較正の検討.py']
 read=lambda p:pd.read_csv(p,float_precision='round_trip')
 b,a=read(paths[0]),read(paths[1]); qs={int(p.stem.rsplit('_',1)[1]):read(p) for p in td.glob('aia_challenge_*.csv')}
 cs={int(p.stem.rsplit('_',1)[1]):read(p) for p in td.glob('C_*.csv') if p.stem!='C_5'}; teams=sorted(cs)
 assert len(teams)==20 and len(b)==1049 and b.record_id.is_unique
 assert not b.duplicated(list(common.QI7)).any() and not qs[5].duplicated(list(common.QI7)).any()
 private=pd.MultiIndex.from_frame(b[list(common.QI7)]).isin(pd.MultiIndex.from_frame(qs[5][list(common.QI7)]))
 rare=b.is_rare.to_numpy(bool); normal=~private & ~rare
 assert (int(private.sum()),int(rare.sum()),int((private & rare).sum()),int(normal.sum()))==(163,24,13,875)
 return common,calib,base,paths,b,a,qs,cs,teams,private,rare,normal

def raw_scores(common,calib,b,a,qs,cs,teams):
 raw,cal=[],[]
 for t in teams:
  controls=pd.concat([qs[k][list(common.QI7)] for k in sorted(qs) if k!=t],ignore_index=True)
  query=pd.concat([b[list(common.QI7)],controls],ignore_index=True)
  combined=common.core(a,query,cs[t],common.NUM7,(10,))[10]; values,null=combined[:len(b)],combined[len(b):]
  raw.append(values); cal.append(calib.calibrate(values,b,null,controls,['prefecture','sex'],500))
  print(f'scored target {t}',flush=True)
 return np.column_stack(raw),np.column_stack(cal)

def ranks(x): return np.column_stack([rankdata(v,method='average')/len(v) for v in np.asarray(x).T])
def rivals(x,stat): return np.column_stack([stat(np.delete(x,j,axis=1),axis=1) for j in range(x.shape[1])])

def extend_ranks(values,normal,unknown):
 """Normal scores equal average-rank/n; extend ranks to unknown rare candidates."""
 out=[]
 for v in values.T:
  ordered=np.sort(v[normal]); q=v[unknown]
  score=(np.searchsorted(ordered,q,'left')+np.searchsorted(ordered,q,'right')+1)/(2*len(ordered))
  out.append(np.clip(score,0,1))
 return np.column_stack(out)

def capacity_bounds(n,k):
 missing=CONFIG['N']-1-k
 low=max(k*25,736-missing*27); high=min(k*27,736-missing*25)
 lo=np.r_[np.repeat(25.,k),n-high]; hi=np.r_[np.repeat(27.,k),n-low]
 if np.any(lo<0) or np.any(lo>hi) or lo.sum()>n or hi.sum()<n: raise ValueError('Infeasible capacities')
 return lo,hi

def feasibility(p,lo,hi):
 mass=p.sum(0)
 return max(float(np.abs(p.sum(1)-1).max()),float(np.maximum(lo-mass,0).max()),float(np.maximum(mass-hi,0).max()))

def soft_assignment(scores,lo,hi,tau):
 """Entropy-regularized transport, convex dual over box-capacity multipliers.
 Rows sum to 1 by softmax. No labels or exact target member counts enter.
 """
 n,m=scores.shape
 def fun(x):
  upper,lower=x[:m],x[m:]; a=(scores-upper+lower)/tau; lse=logsumexp(a,axis=1)
  p=np.exp(a-lse[:,None]); mass=p.sum(0)
  return tau*lse.sum()+upper@hi-lower@lo,np.r_[hi-mass,mass-lo]
 opt=minimize(fun,np.zeros(2*m),jac=True,method='L-BFGS-B',bounds=[(0,None)]*(2*m),
              options={'ftol':1e-15,'gtol':1e-9,'maxiter':10000,'maxls':100,'maxcor':30})
 upper,lower=opt.x[:m],opt.x[m:]; logits=(scores-upper+lower)/tau
 p=np.exp(logits-logsumexp(logits,axis=1)[:,None]); error=feasibility(p,lo,hi)
 primal=float((scores*p).sum()-tau*xlogy(p,p).sum()); gap=float(opt.fun-primal)
 comp=max(float(np.abs(upper*(hi-p.sum(0))).max()),float(np.abs(lower*(p.sum(0)-lo)).max()))
 if error>5e-5 or abs(gap)>1e-4 or comp>1e-4: raise RuntimeError(f'Soft optimizer: {opt.message}; error={error}; gap={gap}')
 return p,{'success':bool(opt.success),'iterations':int(opt.nit),'feasibility':error,'dual_gap':gap,'complementarity':comp,'column_mass':p.sum(0).tolist()}

def hard_assignment(scores,lo,hi):
 """LP over an integral capacitated transportation polytope."""
 n,m=scores.shape; idx=np.arange(n*m)
 row=coo_matrix((np.ones(n*m),(idx//m,idx)),shape=(n,n*m)).tocsr()
 col=coo_matrix((np.ones(n*m),(idx%m,idx)),shape=(m,n*m)).tocsr()
 opt=linprog(-scores.ravel(),A_ub=vstack([col,-col]),b_ub=np.r_[hi,-lo],A_eq=row,b_eq=np.ones(n),bounds=(0,1),method='highs-ds')
 if not opt.success: raise RuntimeError(opt.message)
 p=opt.x.reshape(n,m); integ=float(np.abs(p-np.round(p)).max())
 if integ>1e-7 or feasibility(p,lo,hi)>1e-7: raise RuntimeError('Nonintegral/infeasible flow')
 return np.round(p),{'success':True,'integrality':integ,'feasibility':feasibility(p,lo,hi),'objective':float(-opt.fun),'column_mass':p.sum(0).tolist()}

def predict(raw,cal,normal,private,indices):
 unknown=~private; r=ranks(cal[normal][:,indices]); g=ranks(raw[normal][:,indices])
 median=rivals(r,np.median); maximum=rivals(r,np.max)
 methods={'GenLRA7':g,'Q500':r,'P01':r-.1*median}
 for w in CONFIG['max_penalties']: methods[f'max_{w:g}']=r-w*maximum
 methods['P01_plus_max_0.1']=methods['P01']-.1*maximum
 latent=extend_ranks(cal[:,indices],normal,unknown)
 assert np.allclose(latent[normal[unknown]],r,atol=0,rtol=0)
 costs=np.column_stack([latent,np.full(unknown.sum(),CONFIG['dummy_score'])]); lo,hi=capacity_bounds(len(costs),len(indices)); ds={}
 for tau in CONFIG['soft_temperatures']:
  p,d=soft_assignment(costs,lo,hi,tau); methods[f'soft_box_{tau:g}']=p[normal[unknown],:-1]; ds[f'soft_box_{tau:g}']=d
 p,d=hard_assignment(costs,lo,hi); methods['hard_rerank']=p[normal[unknown],:-1]+.001*r; ds['hard_rerank']=d
 methods['hard_binary_diagnostic']=p[normal[unknown],:-1]
 return methods,ds

def confidence(values,normal,private,rare):
 result=np.zeros((len(normal),values.shape[1]))
 for j in range(values.shape[1]): result[normal,j]=.6+.3*rankdata(values[:,j],method='min')/normal.sum()
 result[rare & ~private,:]=.4
 return result

def audit_truth(base,b,private,rare,teams):
 paths=sorted((base/'結果/正解データ').glob('pool_ids_*.csv')); pools={int(p.stem.rsplit('_',1)[1]):pd.read_csv(p) for p in paths}
 for p in pools.values(): assert len(p)==1049 and p.record_id.is_unique and p.pool_id.is_unique
 own=pools[5].set_index('record_id').loc[b.record_id].pool_id
 truth_all=np.column_stack([own.isin(pools[k].pool_id).to_numpy(int) for k in sorted(pools) if k!=5])
 y=np.column_stack([own.isin(pools[k].pool_id).to_numpy(int) for k in teams]); hist=Counter(v for p in pools.values() for v in p.pool_id)
 audit={'available_pool_files':len(pools),'self_records':len(b),'known_private':int(private.sum()),'normal_candidates':int((~private & ~rare).sum()),'latent_unknown':int((~private).sum()),
 'max_cohort_multiplicity_in_available_truth':max(hist.values()),'self_max_other_cohorts':int(truth_all.sum(1).max()),
 'self_rows_in_multiple_targets':int((y.sum(1)>1).sum()),'self_rows_in_multiple_available_other_cohorts':int((truth_all.sum(1)>1).sum()),
 'private_positive_cells':int(y[private].sum()),'observed_target_count':len(teams),'observed_total_members':int(y.sum()),
 'observed_rare_members':int(y[rare].sum()),'observed_normal_members':int(y[~rare].sum()),
 'members_by_target':{str(t):int(y[:,j].sum()) for j,t in enumerate(teams)},
 'normal_members_by_target':{str(t):int(y[~rare,j].sum()) for j,t in enumerate(teams)},
 'source_limit':'Checks available preliminary truth only; not all 30 cohorts nor main truth.'}
 assert audit['max_cohort_multiplicity_in_available_truth']<=2
 assert audit['self_rows_in_multiple_available_other_cohorts']==0 and audit['private_positive_cells']==0
 lo,hi=capacity_bounds(1049,len(teams)); assert all(25<=v<=27 for v in audit['members_by_target'].values())
 assert 1049-hi[-1]<=y.sum()<=1049-lo[-1]
 return y,audit,paths

def metrics(s,y):
 f,t,_=roc_curve(y,s,drop_intermediate=False); out={f'tpr_{a:g}':float(np.interp(a,f,t)) for a in CONFIG['alpha']}; out['auc']=float(roc_auc_score(y,s))
 order=np.argsort(-s,kind='stable'); ss=s[order]; yy=y[order]; end=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
 fp=np.r_[0,np.cumsum(1-yy)[end]/(len(y)-y.sum())]; tp=np.r_[0,np.cumsum(yy)[end]/y.sum()]
 assert abs(float(np.interp(.01,fp,tp))-out['tpr_0.01'])<1e-12
 for k in CONFIG['top_k']:
  threshold=np.sort(s)[-min(k,len(s))]; gt=s>threshold; eq=s==threshold
  out[f'top{k}_expected_hits']=float(y[gt].sum()+(min(k,len(s))-gt.sum())*y[eq].sum()/eq.sum())
 v=out['tpr_0.01']; out['band']=1+int(v>.25)+int(v>.5)+int(v>.75)
 return out

def evaluate(predictions,y,teams,normal,private,rare):
 rows=[]
 for view,infset,targets,methods in predictions:
  for method,values in methods.items():
   scores=confidence(values,normal,private,rare)
   assert np.all(scores[private]==0) and np.all(scores[rare & ~private]==.4)
   for j,t in enumerate(targets):
    truth=y[:,teams.index(t)]; rows.append({'background':view,'inference':infset,'team':t,'method':method,'members':int(truth.sum()),**metrics(scores[:,j],truth)})
 table=pd.DataFrame(rows); summaries=[]
 excluded={'all':[],'without12':[12],'without1_2':[1,2],'without1_2_12':[1,2,12]}
 for (view,infset),panel in table.groupby(['background','inference'],sort=False):
  baseline=panel[panel.method=='P01'].set_index('team')['tpr_0.01']
  for evset,drop in excluded.items():
   for method,group in panel[~panel.team.isin(drop)].groupby('method',sort=False):
    delta=group.set_index('team')['tpr_0.01']-baseline.loc[group.team]
    summaries.append({'background':view,'inference':infset,'evaluation':evset,'method':method,'targets':len(group),
      'mean_tpr':float(group['tpr_0.01'].mean()),'mean_delta_P01':float(delta.mean()),'wins':int((delta>1e-12).sum()),
      'ties':int((abs(delta)<=1e-12).sum()),'losses':int((delta< -1e-12).sum()),'worst_delta':float(delta.min()),'best_delta':float(delta.max()),'band2plus':int((group.band>=2).sum())})
 return table,pd.DataFrame(summaries)

def main():
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',type=Path,default=Path.cwd()); parser.add_argument('--output-dir',type=Path,required=True)
 parser.add_argument('--full-only',action='store_true',help='Smoke test only; final comparison uses all four backgrounds')
 args=parser.parse_args(); root=args.root.resolve(); out=args.output_dir; out.mkdir(parents=True,exist_ok=False)
 start=time.perf_counter(); started=datetime.now(timezone.utc).isoformat(); dump(out/'protocol.json',CONFIG)
 c,calib,base,paths,b,a,qs,cs,teams,private,rare,normal=load_inputs(root)
 ih={str(p.relative_to(root)):digest(p) for p in paths}; ih['run_experiment.py']=digest(__file__); dump(out/'input_sha256.json',ih)
 predictions=[]; diagnostics=[]; hashes=[]; views=CONFIG['backgrounds'][:1] if args.full_only else CONFIG['backgrounds']
 for view in views:
  if view=='full': aa=a
  else:
   rng=np.random.default_rng(int(view.split('seed')[1])); selected=np.sort(rng.choice(len(a),int(.8*len(a)),replace=False)); aa=a.iloc[selected]
  print(f'background={view},rows={len(aa)}',flush=True); raw,cal=raw_scores(c,calib,b,aa,qs,cs,teams)
  np.savez_compressed(out/f'_scores_{view}.npz',raw=raw,cal=cal)
  for infset,drop in [('all20',[]),('without1_2',[1,2]),('without1_2_12',[1,2,12])]:
   if view!='full' and infset!='all20': continue
   idx=[j for j,t in enumerate(teams) if t not in drop]; ts=[teams[j] for j in idx]; methods,ds=predict(raw,cal,normal,private,idx)
   predictions.append((view,infset,ts,methods))
   for name,d in ds.items(): diagnostics.append({'background':view,'inference':infset,'method':name,**d})
   for name,mat in methods.items(): hashes.append({'background':view,'inference':infset,'method':name,'shape':list(mat.shape),'sha256':hashlib.sha256(np.ascontiguousarray(mat,dtype='<f8').tobytes()).hexdigest()})
 dump(out/'prediction_hashes.json',hashes); dump(out/'optimizer_diagnostics.json',diagnostics); frozen=datetime.now(timezone.utc).isoformat()
 y,audit,truthpaths=audit_truth(base,b,private,rare,teams); dump(out/'constraint_audit.json',audit); dump(out/'truth_input_sha256.json',{str(p.relative_to(root)):digest(p) for p in truthpaths})
 table,summary=evaluate(predictions,y,teams,normal,private,rare); table.to_csv(out/'by_target.csv',index=False,float_format='%.17g'); summary.to_csv(out/'summary.csv',index=False,float_format='%.17g')
 oldpath=root/'攻撃フェーズ/Tashiro/MIA全体/experiments/20261007/非会員較正結果/非会員較正_対象別比較.csv'; old=pd.read_csv(oldpath); regression={}
 new=table[(table.background=='full')&(table.inference=='all20')]
 for fresh,previous in [('P01','Q_prefecture_sex_shrink500_cross01'),('GenLRA7','global7')]:
  v=new[new.method==fresh].set_index('team')['tpr_0.01']; w=old[old.method==previous].set_index('team').mia_all
  assert set(v.index)==set(w.index); err=float((v-w).abs().max()); regression[fresh]=err
  if err>1e-12: raise RuntimeError(f'Historical baseline differs: {fresh}: {err}')
 dump(out/'run_record.json',{'started_utc':started,'predictions_frozen_utc':frozen,'completed_utc':datetime.now(timezone.utc).isoformat(),
 'seconds':time.perf_counter()-start,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
 'code_sha256':digest(__file__),'baseline_regression_max_error':regression,'full_only':args.full_only,'prediction_conditions':len(hashes),'target_method_rows':len(table),
 'no_main_data':True,'no_competition_submission':True})
 for p in paths: assert digest(p)==ih[str(p.relative_to(root))]
 print(summary[(summary.background=='full')&(summary.inference=='all20')&(summary.evaluation=='all')].to_string(index=False)); print(json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
