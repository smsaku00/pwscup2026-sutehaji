"""End-to-end checks against unchanged source summaries and frozen fold seals."""
import argparse,json
import numpy as np,pandas as pd
import runtime as r

def historical():
 out=r.HERE/'validation';out.mkdir(exist_ok=True);checks=[]
 for p in [r.HERE/'cache/fixed22/historical_reference_roles/historical_regression.json',r.HERE/'results_public/historical_reference_roles/historical_counterfactual_regression.json',r.HERE/'results_transfer/historical_reference_roles/historical_policy_regression.json',r.HERE/'results_public/historical_reference_roles/historical_q_omission_composite_regression.json']:
  assert p.exists();checks.append({'source':str(p.relative_to(r.HERE)),'sha256':r.sha(p)})
 old=pd.read_csv(r.SNAP/'20261009_TPR010到達と加工特徴転移検証/nested_holdout/policy_screen_72.csv');old['policy']=old[['scope','k','kind']].astype(str).agg('/'.join,axis=1)
 new=pd.read_csv(r.HERE/'results_transfer/historical_reference_roles/summary.csv');new=new[(new.family=='nested_fixed72')&(new.population=='historical_native')&(new.condition=='full')]
 joined=new.merge(old,on='policy');err=float((joined.mean_tpr-joined.mean_full_tpr).abs().max());eq=bool((joined.TPR_ge010==joined.hit_ge010).all());assert len(joined)==72 and err<1e-12 and eq
 checks.append({'test':'all_original72_fixed_specs','rows':72,'max_mean_tpr_error':err,'hits_identical':eq})
 path=r.HERE/'cache/p08d/prelim/policy_by_target.csv'
 if path.exists():
  old=pd.read_csv(r.SNAP/'20261009_P08D_上位競合だけの疎な排他補正/looto/LOOTO_by_target.csv').rename(columns={'target':'team'})
  new=pd.read_csv(path);new=new[(new.population=='historical_native')&(new.background=='full')];z=new.merge(old,on=['team','policy']);err=float((z['tpr_0.01']-z.heldout_full_TPR).abs().max());eq=bool((z.selected_method==z.model_chosen_without_target_truth).all());assert len(z)==72 and err<1e-12 and eq
  checks.append({'test':'P08D72_held_target_selections','rows':72,'max_TPR_error':err,'chosen_methods_identical':eq})
 old=pd.read_csv(r.SNAP/'20261009_P07_公開AIAコホート保留と方式選択/results/Q_cohort_omission_TPR.csv').rename(columns={'target':'team','method':'policy'})
 old['view']='hold_'+old.excluded_public_Q_cohort.astype(str)
 new=pd.read_csv(r.HERE/'results_public/prelim/p07_omission_by_target.csv');new=new[new.population=='historical_native']
 z=new.merge(old,on=['team','policy','view']);err=float((z['tpr_0.01']-z.tpr_001).abs().max());assert len(z)==240 and err<1e-12
 checks.append({'test':'P07_all12_primitive_omission_variants','rows':240,'max_TPR_error':err})
 r.dump(out/'historical_extended_tests.json',checks)

def all_complete():
 checks=[]
 for bg in r.BG:
  for family in ('p06','p07','p08d','fixed22'):
   p=r.HERE/'cache'/family/bg;assert (p/'complete.json').exists(),str(p)
  for family in ('p05','p06','p07'):assert (r.HERE/'results_public'/bg/f'{family}_complete.json').exists()
  assert (r.HERE/'results_transfer'/bg/'complete.json').exists()
  # Scores and file SHA checks cover every stored primitive array.
  count=0
  for family in ('p06','p07','p08d','fixed22'):
   for seal in (r.HERE/'cache'/family/bg).rglob('prediction_record.json'):
    models=r.load_scores(seal.parent);assert all(np.isfinite(x).all() for x in models.values());count+=len(models)
  s=pd.read_csv(r.HERE/'results_transfer'/bg/'selection_audit.csv');assert len(s)==1462
  assert all(str(x.team) not in str(x.training_targets).split(';') for x in s.itertuples())
  s=pd.read_csv(r.HERE/'cache/p08d'/bg/'selection_audit.csv');assert len(s)==72
  assert all(str(x.team) not in str(x.training_targets).split(';') for x in s.itertuples())
  # Recreate split membership by row position, then bind to source+record_id.
  ids=pd.read_csv(f'/workspace/shared/mia_bg7_inputs/bg/subsets/{bg}.identity.csv');n=len(ids);assert ids.source_record_key.is_unique
  for seed in r.p06.SEEDS:
   held=np.sort(np.random.default_rng(seed).choice(n,n//5,replace=False));train=np.setdiff1d(np.arange(n),held);assert not set(ids.iloc[held].source_record_key)&set(ids.iloc[train].source_record_key)
  checks.append({'background':bg,'sealed_arrays_verified':count,'transfer_outer_target_exclusion':True,'p08d_outer_target_exclusion':True,'p06_disjoint_source_record_keys':True})
 r.dump(r.HERE/'validation/all_background_tests.json',checks)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');a=p.parse_args();historical()
 if a.all:all_complete()
 print('POLICY_VALIDATION_OK',a.all)
