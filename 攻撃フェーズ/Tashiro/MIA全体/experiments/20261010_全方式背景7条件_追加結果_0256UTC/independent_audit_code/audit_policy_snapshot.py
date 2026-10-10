from pathlib import Path
import json,hashlib,datetime
from functools import lru_cache
import numpy as np,pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent
exec((R/'audit_recovered_calibration.py').read_text().split('results={}')[0])
P=Path('/workspace/shared/mia_all_bg_policy/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/policy')
reg=read(P/'registry/coverage.csv').fillna('');detail=read(P/'report/primary_by_target.csv').fillna('');summary=read(P/'report/primary_summary.csv')
assert reg.config_id.nunique()==139 and len(reg)==973 and (reg.status=='complete').all()
assert not detail.duplicated(['config_id','background_source','team','population']).any()
assert set(map(tuple,detail[['config_id','background_source']].drop_duplicates().to_numpy()))==set(map(tuple,reg[['config_id','background']].to_numpy()))
lookup=detail.set_index(['config_id','background_source','team','population']);errors=[];hashes=0;checked=0;skipped={};families=set();rosters={}
@lru_cache(None)
def bank(rel):
 global hashes
 p=P/rel;m=json.loads((p/'prediction_record.json').read_text());assert sha(p/'scores.npz')==m['file_sha256'];assert not m['truth_read_for_prediction']
 with np.load(p/'scores.npz') as z:
  out={name:z[f'x{i}'] for i,name in enumerate(m['methods'])}
 for name,v in out.items():assert hashlib.sha256(np.ascontiguousarray(v,dtype='<f8').tobytes()).hexdigest()==m['hashes'][name];hashes+=1
 return out,m['teams']
def conf(v):
 out=np.zeros(1049);out[rare&~private]=.4;out[normal]=.6+.3*rankdata(v,method='min')/875;return out
for bg in BG:
 selections=[]
 for p in (P/'results_public'/bg).glob('selection_*.csv'):selections.append(read(p))
 choice=pd.concat(selections,ignore_index=True).drop_duplicates(['family','policy','team'])
 tr=read(P/'results_transfer'/bg/'selection_audit.csv');lo=read(P/'cache/p08d'/bg/'selection_audit.csv')
 for tab in [tr,lo]:
  assert all(str(x.team) not in str(x.training_targets).split(';') for x in tab.itertuples())
 for row in reg[reg.background==bg].itertuples():
  family=row.family;policy=row.policy;view=row.view;actual={}
  expected_teams=set(detail[(detail.config_id==row.config_id)&(detail.background_source==bg)].team)
  if family in ('transfer','nested','nested_safety','nested_fixed72'):
   arrays,teams=bank(f'cache/fixed22/{bg}/full');z=tr[(tr.family==family)&(tr.policy==policy)]
   actual={int(x.team):arrays[x.selected_method][:,teams.index(x.team)] for x in z.itertuples()}
  elif family=='p08d_looto':
   arrays,teams=bank(f'cache/p08d/{bg}/full');small=[t for t in teams if t not in(1,2)];z=lo[lo.policy==policy]
   actual={int(x.team):arrays[x.chosen_method][:,small.index(x.team)] for x in z.itertuples()}
  elif family=='p07_Q_omission':
   arrays,teams=bank(f'cache/p07/{bg}/{view}');actual={t:arrays[policy][:,j] for j,t in enumerate(teams)}
  elif family in ('p05','p06','p07'):
   arrays,teams=bank(f'cache/p06/{bg}/{view}' if family=='p06' and view!='full' else f'cache/p07/{bg}/full')
   if 'KDEB' not in arrays:
    extra,_=bank(f'cache/kdeb/{bg}');arrays={**arrays,'KDEB':extra['q7_robust_adaptivek15_h1__age100__cross0.1']}
   z=choice[(choice.family==family)&(choice.policy==policy)&choice.team.isin(expected_teams)]
   for x in z.itertuples():
    if x.selected_family=='P03_native18':skipped['P03_requires_new_distance_reconstruction']=skipped.get('P03_requires_new_distance_reconstruction',0)+1;continue
    actual[int(x.team)]=arrays[x.selected_family][:,teams.index(x.team)]
  else:
   skipped[family]=skipped.get(family,0)+len(expected_teams);continue
  assert set(actual)<=expected_teams
  if family not in ('p05','p06','p07'):assert set(actual)==expected_teams
  for t,v in actual.items():
   native=v if len(v)==1049 else conf(v);fixed=conf(native[normal])
   for pop,vals,mask in [('historical_native',native,np.ones(1049,bool)),('all_fixed',fixed,np.ones(1049,bool)),('normal',fixed,normal)]:
    expected=lookup.loc[(row.config_id,bg,t,pop)];assert expected.n==mask.sum() and expected.members==ys[t][mask].sum()
    errors.append(abs(roc(vals[mask],ys[t][mask],.01)-expected['tpr_0.01']));checked+=1
   families.add(family)
 # r2 has its own whole-MIA result beyond its use in selected policies.
 arrays,teams=bank(f'cache/fixed22/{bg}/full/legacy');tab=read(P/'cache/fixed22'/bg/'by_target.csv');name='legacy_saved_work_20260901以前__prelim_attack__r2__F_mia'
 z=tab[(tab.condition=='full')&(tab.method==name)&(tab.population=='historical_native')].set_index('team')
 for j,t in enumerate(teams):errors.append(abs(roc(arrays[name][:,j],ys[t],.01)-z.loc[t,'tpr_0.01']))
 print('AUDIT_POLICY',bg,flush=True)
# Every normalized summary is recomputed, including the policy rows omitted from
# the independent frozen-score sample above.
g=detail.groupby(['config_id','family','policy','native_view','background_source','population'])['tpr_0.01'].agg(['size','mean'])
s=summary.set_index(['config_id','family','policy','native_view','background_source','population']);assert set(g.index)==set(s.index) and (g['size']==s.loc[g.index,'targets']).all();errors.extend(abs(g['mean']-s.loc[g.index,'mean_native']).tolist())
assert max(errors)<1e-12
out={'status':'passed','audit_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'registrations':139,'BG_conditions':7,'registry_conditions':len(reg),'all_report_target_population_rows':len(detail),'all_summary_rows_recomputed':len(s),'frozen_primitive_matrix_hashes_checked':hashes,'policy_target_population_TPR_values_recomputed':checked,'r2_full_target_TPR_values_recomputed':140,'all_numeric_values_checked':len(errors),'max_absolute_error':max(errors),'families_with_independent_frozen_score_checks':sorted(families),'explicitly_not_reconstructed_in_this_audit':skipped,'selection_audit_heldout_target_exclusions_checked':True,'method':'Independent tie-group ROC and direct frozen-array selection; no experiment/scorer imports. P03 distance recomputation and Q-omission composite assignment are outside this sample; their full original-source tests are separately supplied.'}
(R/'inventory/policy_independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False))
