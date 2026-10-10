"""Stable historical identities and source-expanded outputs; no best-only filtering."""
import json,hashlib
import pandas as pd
import runtime as r
import run_nested_holdout as nested
REG=r.Path('/workspace/shared/mia_all_background_inventory/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/inventory/config_registry.jsonl')
a=[json.loads(x) for x in REG.read_text().splitlines()];old=[x for x in a if x['family'].startswith('oct09/') and x['status']=='blocked'];assert len(old)==48
rows=[]
for x in old:
 f=x['family'];family='p05' if '/20261009_P05_' in f else 'p06' if '/20261009_P06_' in f else 'p07' if '/20261009_P07_' in f else 'p08d_looto' if '/20261009_P08D_' in f else 'nested_safety' if f.endswith('/nested_safety') else 'transfer'
 rows.append({'config_id':x['config_id'],'family':family,'policy':x['method'],'view':x['parameters'].get('view','full'),'historical_registry':True,'source':x['source'],'parameters':x['parameters']})
for spec in nested.METHOD_SPECS:rows.append({'family':'nested_fixed72','policy':'/'.join(map(str,spec)),'view':'full','source':'original/run_nested_holdout.py','parameters':{'spec':spec}})
rows.append({'family':'nested','policy':'nested_outer_select','view':'full','source':'original/run_nested_holdout.py','parameters':{}})
probes=json.loads((r.HERE/'cache/p07/prelim/complete.json').read_text())['folds']
for h in probes:
 for method in ('P01','P03_Cnn'):rows.append({'family':'p07_Q_omission','policy':method,'view':f'hold_{h}','source':'original/run_qcohort_holdout.py','parameters':{'heldout_Q':h}})
 rows.append({'family':'p07_Q_omission_composite','policy':f'omit_Q_{h}','view':'full','source':'original/run_qcohort_holdout.py','parameters':{'heldout_Q':h}})
assert len(rows)==139
for row in rows:
 row.setdefault('historical_registry',False);row.setdefault('config_id','BG7-policy-extra-'+hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:16])
 row['main_applicability']='No original main score/selector implementation. P04 has public-only family projection, explicitly no candidate scores; no main TPR or rank invented.'
 row['selection_status']='historically outcome-exposed family; original nested/LOOTO exclusions retained where applicable'
 row['semantic_alias_of']=''
p04=[x for x in rows if x['policy']=='P04' and x['view']=='full'];assert len(p04)==3
for x in p04[1:]:x['semantic_alias_of']=p04[0]['config_id']
out=r.HERE/'registry';out.mkdir(exist_ok=True)
(out/'outputs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
coverage=[];frames={}
for row in rows:
 f=row['family'];policy=row['policy'];view=row['view']
 for bg in r.BG:
  if f in ('p05','p06','p07'):p=r.HERE/'results_public'/bg/f'{f}_by_target.csv';fc=None
  elif f in ('transfer','nested','nested_safety','nested_fixed72'):p=r.HERE/'results_transfer'/bg/'by_target.csv';fc=f
  elif f=='p08d_looto':p=r.HERE/'cache/p08d'/bg/'policy_by_target.csv';fc=None
  elif f=='p07_Q_omission':p=r.HERE/'results_public'/bg/'p07_omission_by_target.csv';fc=None
  else:p=r.HERE/'results_public'/bg/'p07_omission_composite_by_target.csv';fc=None
  if p.exists():
   
   if p not in frames:frames[p]=pd.read_csv(p)
   tab=frames[p];z=tab[tab.policy==policy];vcol='condition' if 'condition' in z else 'view' if 'view' in z else 'background';z=z[(z[vcol]==view)&(z.population=='all_fixed')]
   if fc:z=z[z.family==fc]
   count=len(z);status='complete' if count else 'not_applicable_empty_eligible_group'
  else:count=0;status='pending'
  coverage.append({'config_id':row['config_id'],'family':f,'policy':policy,'view':view,'background':bg,'status':status,'native_targets':count,'result_path':str(p.relative_to(r.HERE)),'semantic_alias_of':row['semantic_alias_of']})
pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
r.dump(out/'counts.json',{'historical_ids':48,'source_expanded_additions':91,'registered_outputs':139,'explicit_P04_semantic_aliases':2,'registered_semantic_recipes_before_data_dependent_equivalence':137,'outer_backgrounds':7,'coverage_rows':len(coverage),'note':'P06 split-view historical identities retained; numerical agreement on one dataset does not make different parameters semantic aliases.'})
print('REGISTRY',len(rows),pd.Series([x['status'] for x in coverage]).value_counts().to_dict())
