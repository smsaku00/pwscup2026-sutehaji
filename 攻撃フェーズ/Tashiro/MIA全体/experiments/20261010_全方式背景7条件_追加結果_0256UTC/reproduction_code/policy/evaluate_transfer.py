"""Replay original target exclusion and nested policy search on regenerated22.

The entire original72-spec nested search is retained. Per-outer-fold repeated
inner predictions are memoized only within the same frozen tensor; no pruning.
"""
import argparse,json,time
from functools import lru_cache
from datetime import datetime,timezone
import numpy as np,pandas as pd
import runtime as r
import run_transfer as tr
import run_nested_holdout as nested
import run_nested_safe as safe

def run(bg,historical=False):
 key='historical_reference_roles' if historical else bg
 inp=r.HERE/'cache/fixed22'/key;out=r.HERE/'results_transfer'/key
 if (out/'complete.json').exists():return
 assert (inp/'complete.json').exists();out.mkdir(parents=True,exist_ok=True);start=time.time()
 tab=pd.read_csv(inp/'by_target.csv');native=tab[(tab.population=='historical_native')&tab.team.isin(tr.TARGET_SET)].copy();native['tpr']=native['tpr_0.01']
 methods=sorted(native.method.unique());views=sorted(native.condition.unique());assert len(methods)==22 and len(views)==7
 metadata=r.HERE/'cache/metadata'/key;feat=pd.read_csv(metadata/'public_features.csv');groups=pd.read_csv(metadata/'groups.csv').set_index('team')
 dist,spec=tr.public_distances(feat);tensor=tr.make_tensor(native,methods,views);high=[int(groups.loc[t,'aia_imprint_high']) for t in tr.TARGET_SET];base=methods.index(tr.BASELINE)
 choices=[]
 for policy in [{'name':'P01_fixed','neighbors':0,'criterion':'baseline'}]+tr.POLICIES:
  for i,t in enumerate(tr.TARGET_SET):
   peers=[] if policy['name']=='P01_fixed' else tr.neighbor_indices(dist,i,policy['neighbors'],policy.get('shuffle_neighbors',False))
   assert i not in peers
   j=base if len(peers)==0 else tr.selected_method(tensor,methods,views,peers,policy['criterion'])
   choices.append({'family':'transfer','policy':policy['name'],'team':t,'selected_method':methods[j],'training_targets':';'.join(str(tr.TARGET_SET[k]) for k in peers),'selection_scope':'other-target outcomes only'})
 original=nested.heldout_predictions
 @lru_cache(None)
 def cached_inner(train,policy):return original(tensor,methods,views,dist,high,list(train),policy)
 def replacement(_tensor,_methods,_views,_dist,_group,train,policy):
  assert _tensor is tensor
  return cached_inner(tuple(train),policy)
 nested.heldout_predictions=replacement
 try:
  for i,t in enumerate(tr.TARGET_SET):
   peer=[k for k in range(17) if k!=i]
   chosen,inside=nested.nested_outer_select(tensor,methods,views,dist,high,i)
   j=nested.selector(tensor,methods,views,dist,high,i,peer,chosen)
   choices.append({'family':'nested','policy':'nested_outer_select','team':t,'selected_method':methods[j],'selected_spec':'/'.join(map(str,chosen)),'training_targets':';'.join(str(tr.TARGET_SET[k]) for k in peer),'selection_scope':'outer target excluded from all inner folds'})
   for budget in (0,1):
    j,chosen,hits=safe.choose_safely(tensor,methods,views,dist,high,i,budget)
    choices.append({'family':'nested_safety','policy':f'safety_budget_{budget}','team':t,'selected_method':methods[j],'selected_spec':'safe_fallback_P01' if chosen is None else '/'.join(map(str,chosen)),'training_targets':';'.join(str(tr.TARGET_SET[k]) for k in peer),'selection_scope':'outer target excluded; original inner safety filter'})
   for policy in nested.METHOD_SPECS:
    j=nested.selector(tensor,methods,views,dist,high,i,peer,policy)
    choices.append({'family':'nested_fixed72','policy':'/'.join(map(str,policy)),'team':t,'selected_method':methods[j],'training_targets':';'.join(str(tr.TARGET_SET[k]) for k in peer),'selection_scope':'fixed-policy LOOTO; full72 screen reported, not outer policy selection'})
 finally:nested.heldout_predictions=original
 choice=pd.DataFrame(choices);assert len(choice)==17*(11+1+2+72)
 for row in choice.itertuples():assert str(row.team) not in str(row.training_targets).split(';')
 choice.to_csv(out/'selection_audit.csv',index=False)
 # Choices are sealed before evaluation rows for each selected target are joined.
 r.dump(out/'selection_freeze.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'choices_sha256':r.sha(out/'selection_audit.csv'),'base_panel_sha256':r.sha(inp/'by_target.csv'),'public_features_sha256':r.sha(metadata/'public_features.csv'),'groups_sha256':r.sha(metadata/'groups.csv'),'every_outer_target_excluded':True,'original_policy_specs':nested.METHOD_SPECS,'method_panel_historically_label_exposed':True,'scope_targets':tr.TARGET_SET,'main_applicability':'not applicable: no original main-trained or main-applied selector'})
 joined=choice.merge(tab,on=['team'],suffixes=('','_panel'));joined=joined[joined.selected_method==joined.method].copy();joined['outer_background']=bg
 assert len(joined)==len(choice)*7*3
 joined.to_csv(out/'by_target.csv',index=False,float_format='%.17g');summaries=[]
 for labels,part in joined.groupby(['family','policy','population','condition'],sort=False):
  vals=part['tpr_0.01'];summaries.append(dict(zip(['family','policy','population','condition'],labels),targets=len(part),mean_tpr=float(vals.mean()),TPR_ge010=int((vals>=.1-1e-12).sum())))
 summary=pd.DataFrame(summaries);summary.to_csv(out/'summary.csv',index=False,float_format='%.17g')
 if historical:
  checks=[]
  old=pd.read_csv(r.SNAP/'20261009_TPR010到達と加工特徴転移検証/results/transfer_looto_summary.csv').set_index('policy')
  actual=summary[(summary.family=='transfer')&(summary.population=='historical_native')&(summary.condition=='full')].set_index('policy')
  for name,x in actual.iterrows():checks.append({'family':'transfer','policy':name,'mean_error':abs(x.mean_tpr-old.loc[name,'full_mean_tpr']),'hits_match':bool(x.TPR_ge010==old.loc[name,'full_ge010'])})
  old=pd.read_csv(r.SNAP/'20261009_TPR010到達と加工特徴転移検証/nested_safety/safe_nested_summary.csv').set_index('safety_budget')
  actual=summary[(summary.family=='nested_safety')&(summary.population=='historical_native')&(summary.condition=='full')].set_index('policy')
  for budget in (0,1):
   x=actual.loc[f'safety_budget_{budget}'];checks.append({'family':'nested_safety','policy':f'safety_budget_{budget}','mean_error':abs(x.mean_tpr-old.loc[budget,'mean_full_tpr']),'hits_match':bool(x.TPR_ge010==old.loc[budget,'full_ge010'])})
  old=json.loads((r.SNAP/'20261009_TPR010到達と加工特徴転移検証/nested_holdout/summary.json').read_text())['nested'];x=summary[(summary.family=='nested')&(summary.population=='historical_native')&(summary.condition=='full')].iloc[0]
  checks.append({'family':'nested','policy':'nested_outer_select','mean_error':abs(x.mean_tpr-old['mean_full_tpr']),'hits_match':bool(x.TPR_ge010==old['full_ge010'])})
  r.dump(out/'historical_policy_regression.json',checks);assert all(x['mean_error']<1e-12 and x['hits_match'] for x in checks),checks
 r.dump(out/'complete.json',{'outer_background':bg,'historical_reference_roles':historical,'policies':86,'native_targets':17,'inner_views':7,'original_specs':len(nested.METHOD_SPECS),'memoized_inner_calls':cached_inner.cache_info().misses,'seconds':time.time()-start})
 print('TRANSFER_COMPLETE',bg,historical,round(time.time()-start,1),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--backgrounds',nargs='+',default=list(r.BG));p.add_argument('--historical',action='store_true');a=p.parse_args()
 for bg in a.backgrounds:run(bg,a.historical)
