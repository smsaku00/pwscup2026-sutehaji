"""P05/P06/P07 decisions use regenerated public features/risk, never truth.

Native scope is17 targets. Historic20 mixes fixed official1/2 TPR and P01
team12; this is reported separately, never as20 regenerated predictions.
"""
import argparse,ast,json,time
from datetime import datetime,timezone
import numpy as np,pandas as pd
import runtime as r

KDEB=r.p05.B
def ensure_kdeb(bg,d):
 p=r.HERE/'cache/kdeb'/bg
 if (p/'prediction_record.json').exists():return r.load_scores(p)[KDEB]
 for source in [r.HERE/'cache/fixed22'/bg/'full/histlocal']:
  if (source/'prediction_record.json').exists():models={KDEB:r.load_scores(source)[KDEB]};break
 else:models=r.narrow_histlocal(d,only_kdeb=True)
 r.seal_scores(p,models,d);return models[KDEB]

def metadata(bg,historical):
 p=r.HERE/'cache/metadata'/('historical_reference_roles' if historical else bg)
 return pd.read_csv(p/'public_features.csv').set_index('team'),pd.read_csv(p/'groups.csv').set_index('team'),json.loads((p/'cuts.json').read_text())

def rules():
 tree=ast.parse((r.HERE/'original/run_p05.py').read_text())
 for node in ast.walk(tree):
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='rules' for t in node.targets):return eval(compile(ast.Expression(node.value),'original_P05_rule_literal','eval'),{'__builtins__':{},'dict':dict})
 raise RuntimeError('Original P05 rules not found')

def summarize(by):
 rows=[]
 for labels,h in by.groupby(['family','policy','view','population'],sort=False):
  v=h['tpr_0.01'];rows.append(dict(zip(['family','policy','view','population'],labels),targets=len(h),mean_tpr=float(v.mean()),TPR_ge010=int((v>=.1-1e-12).sum())))
 return pd.DataFrame(rows)

def run(bg,families,historical=False):
 d=r.data(bg);out=r.HERE/'results_public'/('historical_reference_roles' if historical else bg);out.mkdir(parents=True,exist_ok=True)
 features,groups,cuts=metadata(bg,historical);assign=r.assignments(groups.reset_index(),cuts);full=r.load_scores(r.HERE/'cache/p07'/bg/'full');full['KDEB']=ensure_kdeb(bg,d)
 indices=[j for j,t in enumerate(d['teams']) if t not in (1,2)];cal=r.raw_cal(bg,d);dist=r.p06.dist_to_C(r.u.base,d['b'],d['a'],d['cs'],d['teams'])
 p03=r.cross(r.base.ranks(cal[d['normal']][:,indices]))+.1*r.base.ranks(-dist[d['normal']][:,indices]);p03full=full['P01'].copy();p03full[:,indices]=p03;full['P03_native18']=p03full
 decisions=[];frozen={};risk_info={}
 for family in families:
  if (out/f'{family}_complete.json').exists():continue
  if family=='p05':
   p4long={'P01':r.p05.P01,'KDEA':r.p05.A,'KDEB':r.p05.B};short={r.p05.P01:'P01',r.p05.A:'KDEA',r.p05.B:'KDEB',r.p05.P03_FROM:'P03_native18'}
   qmedian=r.p05.descriptors(features,groups)
   selections={}
   for name,params in rules().items():
    choices={}
    for t in r.p05.TEAM17:
     chosen,reason=r.p05.public_gates(t,p4long[assign[t]],features,groups,qmedian,copy_frac=params['copy_frac'],rms_ratio=params['rms_ratio'],uniq_cut=params['uniq'],min_close=params['nclose'],enable_copy=params['copy'],enable_close=params['close'])
     choices[t]=short[chosen];decisions.append({'family':family,'policy':name,'team':t,'P04_family':assign[t],'selected_family':short[chosen],'reason':reason})
    selections[name]=choices
  elif family=='p06':
   source=r.HERE/'cache/p06'/bg;assert (source/'complete.json').exists();risk=pd.read_csv(source/'risk.csv');selections={}
   for name in r.p06.POLICIES:
    choices={t:(r.p06.choose(risk,t,name) if assign[t]=='KDEA' else assign[t]) for t in r.p06.TARGET17};selections[name]=choices
    decisions.extend({'family':family,'policy':name,'team':t,'P04_family':assign[t],'selected_family':choices[t],'reason':'public A80/20 risk in original KDE-A eligible group only'} for t in choices)
   risk_info[family]=r.sha(source/'risk.csv')
  elif family=='p07':
   source=r.HERE/'cache/p07'/bg;assert (source/'complete.json').exists();risk=pd.read_csv(source/'risk.csv');selections={}
   for name in r.p07.POLICIES:
    choices={t:(r.p07.choose(risk,t,name) if assign[t]=='P01' else assign[t]) for t in r.p07.TEAM17};selections[name]=choices
    decisions.extend({'family':family,'policy':name,'team':t,'P04_family':assign[t],'selected_family':choices[t],'reason':'Q-cohort heldout risk in original P01 eligible group only'} for t in choices)
   risk_info[family]=r.sha(source/'risk.csv')
  else:raise ValueError(family)
  for name,choices in selections.items():
   mat=full['P01'].copy()
   for t,m in choices.items():mat[:,d['teams'].index(t)]=full[m][:,d['teams'].index(t)]
   frozen[family,name]=mat
 if not frozen:return
 pd.DataFrame(decisions).to_csv(out/('selection_'+''.join(sorted(families))+'.csv'),index=False)
 r.dump(out/('freeze_'+''.join(sorted(families))+'.json'),{'frozen_utc':datetime.now(timezone.utc).isoformat(),'decisions_use_target_membership_truth':False,'historically_outcome_informed_policy_family':True,'native_scope':list(r.p05.TEAM17),'public_risk_sha256':risk_info,'prediction_hashes':{f'{f}/{p}':r.msha(v) for (f,p),v in frozen.items()}})
 y=r.truth(d);rows=[]
 for (family,policy),mat in frozen.items():
  rows.extend(x for x in r.metrics_rows(mat,d,y,family=family,policy=policy,view='full',outer_background=bg) if x['team'] in r.p05.TEAM17)
  if family=='p06':
   choices={x['team']:x['selected_family'] for x in decisions if x['family']==family and x['policy']==policy}
   eligible=[t for t in r.p06.TARGET17 if assign[t]=='KDEA']
   for seed in r.p06.SEEDS:
    view=f'split80_seed{seed}';scores=r.load_scores(r.HERE/'cache/p06'/bg/view);combined=scores['P01'].copy()
    for t in eligible:combined[:,d['teams'].index(t)]=scores[choices[t]][:,d['teams'].index(t)]
    rows.extend(x for x in r.metrics_rows(combined,d,y,family=family,policy=policy,view=view,outer_background=bg) if x['team'] in eligible)
 by=pd.DataFrame(rows)
 for family in sorted(set(k[0] for k in frozen)):
  part=by[by.family==family];part.to_csv(out/f'{family}_by_target.csv',index=False,float_format='%.17g');summarize(part).to_csv(out/f'{family}_summary.csv',index=False,float_format='%.17g')
  r.dump(out/f'{family}_complete.json',{'background':bg,'native_targets':17,'frozen_policies':sum(f==family for f,p in frozen),'score_population':['historical_native','all_fixed','normal'],'main_application':'not applicable: original native17 policy only; no invented main rule','historical_roles':historical})
 # Original Q-omission matrices are an additional unselected source grid.
 if 'p07' in families:
  extras=[];source=r.HERE/'cache/p07'/bg;probes=json.loads((source/'complete.json').read_text())['folds']
  for h in probes:
   for name,score in r.load_scores(source/f'hold_{h}').items():extras.extend(r.metrics_rows(score,d,y,family='p07_Q_omission',policy=name,view=f'hold_{h}',outer_background=bg))
  pd.DataFrame(extras).to_csv(out/'p07_omission_by_target.csv',index=False,float_format='%.17g')
 print('PUBLIC_POLICIES_SCORED',bg,historical,','.join(families),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--families',nargs='+',default=['p05','p06','p07']);p.add_argument('--backgrounds',nargs='+',default=list(r.BG));p.add_argument('--historical',action='store_true');a=p.parse_args()
 for bg in a.backgrounds:run(bg,a.families,a.historical)
