from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'source_oct07'));sys.path.insert(0,'/workspace/shared/mia_bg7_inputs/official_scoring/src')
import 分析 as common
from scipy.stats import rankdata
from collect_results import MAP
from pwscup2026.scoring.attack_score import _tpr_at_fpr
b,a,q,private,rare,cs,t,s=common.phase_inputs(BASE/'compat_root','prelim');normal=~private&~rare
truthdir=BASE/'compat_root/予備戦/配布データ/結果/正解データ';own=common.read(truthdir/'pool_ids_5.csv').set_index('record_id').loc[b.record_id]
ys={team:own.pool_id.isin(common.read(truthdir/f'pool_ids_{team}.csv').pool_id).to_numpy(int) for team in cs}
coverage=pd.read_csv(BASE/'aggregate/coverage.csv');assert len(coverage)==2212
assert (coverage[coverage.evaluation_role=='primary'].status=='completed').all()
assert (coverage.status=='completed').sum()==1882 and (coverage.status=='replaced_by_requested_external_background').sum()==330
bytarget=pd.read_csv(BASE/'aggregate/by_target_all.csv');assert len(bytarget)==37640
assert len(pd.read_csv(BASE/'aggregate/by_target_primary.csv'))==35000
validation=pd.read_csv(BASE/'aggregate/historical_validation.csv');assert len(validation)==10
checks=[];maximum=0.;aucmax=0.
for manifest in sorted((BASE/'frozen/prelim').glob('*/*/manifest.json')):
 meta=json.loads(manifest.read_text());data=np.load(manifest.with_name('matrices.npz'))
 assert sorted(meta['teams'])==sorted(cs) and len(meta['methods'])==len(data.files)
 detail=pd.read_csv(manifest.with_name('by_target.csv')).set_index(['method','team'])
 for name in data.files:
  mat=data[name];assert mat.shape==(875,20) and np.isfinite(mat).all()
  assert hashlib.sha256(mat.astype('<f8').tobytes()).hexdigest()==meta['prediction_sha256'][name]
  for j,team in enumerate(meta['teams']):
   conf=common.confidence(mat[:,j],normal,rare,private);tpr,auc=_tpr_at_fpr(conf,ys[team],.01);row=detail.loc[(name,team)]
   maximum=max(maximum,abs(tpr-row.mia_all));aucmax=max(aucmax,abs(auc-row.auc))
 checks.append({'background':meta['background'],'family':meta['family'],'methods':len(data.files)})
assert maximum<1e-12 and aucmax<1e-12
assert len(pd.read_csv(BASE/'aggregate/by_target_primary_canonical.csv'))==28840
mapping=json.loads((BASE/'primary_alias_map.json').read_text())['map'];inverse={script:family for family,script in MAP.items()};aliaschecks=[]
for bg in sorted(coverage.background.unique()):
 banks={family:np.load(BASE/'frozen/prelim'/bg/family/'matrices.npz') for family in MAP}
 for item in mapping:
  if not item['is_alias']:continue
  script,name=item['original_config_id'].split(':',1);cscript,cname=item['canonical_config_id'].split(':',1)
  x=banks[inverse[script]][name];y=banks[inverse[cscript]][cname]
  equal=bool(np.array_equal(rankdata(x,axis=0,method='min'),rankdata(y,axis=0,method='min')))
  aliaschecks.append({'background':bg,'original_config_id':item['original_config_id'],'canonical_config_id':item['canonical_config_id'],'confidence_rank_exactly_equal':equal,'raw_matrix_exactly_equal':bool(np.array_equal(x,y))})
  assert equal,(bg,item['original_config_id'],item['canonical_config_id'])
 for bank in banks.values():bank.close()
pd.DataFrame(aliaschecks).to_csv(BASE/'aggregate/alias_execution_audit.csv',index=False)
assert len(aliaschecks)==308
# Source baseline validity is a separate check, never silently treated as completed.
all_historical_valid=bool(validation.pass_tolerance_1e12.all())
rareframes=[];rare_aggregate_frames=[]
for done in sorted((BASE/'rare_results').glob('*/completion.json')):
 bg=done.parent.name;f=pd.read_csv(done.parent/'集計/全設定診断.csv');assert len(f)==1584
 assert f.method.nunique()==3 and f.team.nunique()==22 and f.record_id.nunique()==24
 f['background']=bg;rareframes.append(f)
 aggregate=pd.read_csv(done.parent/'集計/既知非会員比較.csv');aggregate['background']=bg;rare_aggregate_frames.append(aggregate)
assert len(rareframes)==7
rare_by_target=pd.concat(rare_aggregate_frames,ignore_index=True);assert len(rare_by_target)==462
rare_by_target.to_csv(BASE/'aggregate/rare_by_target_aggregate.csv',index=False,float_format='%.17g')
rare_summary=rare_by_target.groupby(['background','method'],as_index=False).mean(numeric_only=True).drop(columns='team');rare_summary['targets']=22;rare_summary['membership_inferred']=False;rare_summary.to_csv(BASE/'aggregate/rare_summary.csv',index=False,float_format='%.17g')
rareall=pd.concat(rareframes,ignore_index=True);rareall.to_csv(BASE/'aggregate/rare_diagnostics_all.csv',index=False,float_format='%.17g')
oldpath=next(Path('/workspace/shared/mia_all_background_inventory').glob('**/source_snapshot/**/MIA希少/analysis/集計/全設定診断.csv'))
old=pd.read_csv(oldpath);new=rareall[rareall.background=='main'];z=old.merge(new,on=['team','record_id','method'],suffixes=['_old','_new']);assert len(z)==1584
rareerrors={}
for col in old.select_dtypes(include='number'):
 if col in ['team','record_id']:continue
 x=z[col+'_old'].to_numpy();y=z[col+'_new'].to_numpy();assert np.array_equal(np.isnan(x),np.isnan(y))
 err=float(np.nanmax(np.abs(x-y))) if np.isfinite(x).any() else 0
 rareerrors[col]=err
assert max(rareerrors.values())<1e-12
# Separate baseline-ordered row diagnostic and BG-induced ranking changes for rare candidates.
base=rareall[rareall.background=='main'].set_index(['team','record_id','method'])
rarechanges=[]
for (bg,method),g in rareall[rareall.role=='candidate'].groupby(['background','method']):
 ref=base.loc[list(zip(g.team,g.record_id,g.method))]
 rarechanges.append({'background':bg,'method':method,'candidate_target_pairs':len(g),'nearest_C_changed':int((g.nearest_C_row0.to_numpy()!=ref.nearest_C_row0.to_numpy()).sum()),'candidate_distance_rank_changed':int((g.candidate_distance_rank.to_numpy()!=ref.candidate_distance_rank.to_numpy()).sum()),'team_rank_for_candidate_changed':int((g.team_distance_rank_for_record.to_numpy()!=ref.team_distance_rank_for_record.to_numpy()).sum()),'mean_abs_d1_change':float(np.mean(np.abs(g.d1.to_numpy()-ref.d1.to_numpy())))})
pd.DataFrame(rarechanges).to_csv(BASE/'aggregate/rare_background_changes.csv',index=False,float_format='%.17g')
result={'primary_config_backgrounds':1750,'primary_target_rows':35000,'historical_bg_sensitivity_supplement_target_rows':2640,'replaced_historical_scenario_backgrounds':330,'primary_family_background_runs':len(checks),'official_scorer_max_abs_tpr':maximum,'official_scorer_max_abs_auc':aucmax,'historical_whole_mia_baselines_all_pass':all_historical_valid,'historical_baseline_families':validation.replace({np.nan:None}).to_dict('records'),'distinct_primary_recipes':206,'distinct_recipe_backgrounds':1442,'alias_pairs_validated':len(aliaschecks),'rare_config_backgrounds':21,'rare_diagnostic_rows':len(rareall),'rare_main_baseline_max_numeric_error':max(rareerrors.values()),'main_membership_truth_used':False,'status':'PASS' if all_historical_valid else 'COMPLETE_WITH_BASELINE_DIFFERENCES'}
(BASE/'aggregate/qa.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,ensure_ascii=False))
