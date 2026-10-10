from pathlib import Path
import hashlib,json,datetime
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
S=Path('/workspace/shared/mia_all_bg_fusion/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_oct8/chow/prelim')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:pd.read_csv(p,float_precision='round_trip')
prov=json.loads((S/'input_provenance.json').read_text());assert sha(Path(prov['manifest_path']))==prov['manifest_sha256']
completion=json.loads((S/'completion.json').read_text());assert completion['complete'] and not completion['membership_scoring'] and completion['source_no_GT_guard_retained']
forms={'independent_pmf','Aref_shared_MI_tree','source_specific_MI_tree'}
metrics=['crps','mean_fold_cdf_above_99','mean_fold_cdf_above_95','averaged_cdf_score_above_99','averaged_cdf_score_above_95','averaged_cdf_score_mean']
errors=[];cells=0;hashes=0;cutschecked=0
def compare(a,b,keys,cols):
 global cells
 a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index();assert a.index.equals(b.index)
 e=np.abs(a[cols].to_numpy()-b[cols].to_numpy());errors.extend(e.ravel());cells+=e.size
for bg,info in prov['subsets'].items():
 assert sha(Path(info['path']))==info['sha256'];a=read(info['path'])
 p=S/bg/'prelim';rec=json.loads((p/'実行記録.json').read_text());pred=json.loads((S/bg/'prediction_record.json').read_text())
 assert set(rec['forms'])==forms and rec['no_GT_guard_installed'] and not rec['membership_labels_used'] and not rec['mia_scoring'] and not rec['form_selection']
 assert rec['owner_target_cases']==1080 and rec['fold_owner_target_cases']==2160 and len(pred['prediction_hashes'])==3 and not pred['labels_used']
 for path,h in rec['support_code_sha256'].items():assert sha(Path(path))==h;hashes+=1
 for name,h in rec['output_sha256'].items():assert sha(p/name)==h;hashes+=1
 order=np.random.default_rng(20261008).permutation(len(a));assignment=np.empty(len(a),int);assignment[order[:len(a)//2]]=0;assignment[order[len(a)//2:]]=1
 assert rec['A_reference_rows']==[len(a)//2,len(a)-len(a)//2]
 for x in json.loads((p/'Arefの固定bin境界.json').read_text()):
  mask=(assignment==x['fold'])&(a.sex.to_numpy()==x['sex']);v=a.loc[mask,x['attribute']].to_numpy(float)
  if x['attribute'] in ['TG','ALT']:v=np.log1p(v)
  c=np.unique(np.quantile(v,np.arange(1,10)/10))
  assert len(v)==x['reference_rows'] and len(c)+1==x['bin_states'] and np.array_equal(c,x['finite_cuts']);cutschecked+=1
 fold=read(p/'fold別の課題所有者保留.csv');outer=read(p/'二fold集約の課題所有者保留.csv')
 assert len(fold)==2160 and len(outer)==1080 and set(fold.form)==forms and not fold.membership_labels_used.any() and not outer.membership_labels_used.any() and not outer.used_for_selection.any()
 assert not (fold.target==fold.heldout_null_owner).any() and set(fold.heldout_rows)=={326} and set(fold.training_Q_rows)=={6194}
 keys=['phase','target','heldout_null_owner','form']
 reduced=fold.groupby(keys,as_index=False)[['crps','cdf_score_mean','fold_cdf_above_99','fold_cdf_above_95']].mean().rename(columns={'cdf_score_mean':'averaged_cdf_score_mean','fold_cdf_above_99':'mean_fold_cdf_above_99','fold_cdf_above_95':'mean_fold_cdf_above_95'})
 compare(reduced,outer,keys,['crps','averaged_cdf_score_mean','mean_fold_cdf_above_99','mean_fold_cdf_above_95'])
 panel=outer.groupby(['phase','heldout_null_owner','form'],as_index=False)[metrics].mean()
 compare(panel,read(p/'Q所有者等重みの診断パネル.csv'),['phase','heldout_null_owner','form'],metrics)
 compare(panel.groupby(['phase','form'],as_index=False)[metrics].mean(),read(p/'固定三形の非会員診断.csv'),['phase','form'],metrics)
 compare(outer.groupby(['phase','target','form'],as_index=False)[metrics].mean(),read(p/'対象別の固定三形診断.csv'),['phase','target','form'],metrics)
 numeric=read(p/'batchとsingleの生スコア検証.csv');assert len(numeric)==40 and (numeric.batch_single_raw_max_error==0).all() and numeric.all_log_PMFS_finite.all()
 struct=read(p/'木構造とPMF正規化.csv');assert (struct.conditional_row_sum_one_error<1e-12).all() and (struct.pref_factor_sum_one_error<1e-12).all()
 print('AUDIT chow',bg,flush=True)
drift=read(S/'background_rank_diagnostics.csv');assert len(drift)==7*3*20 and not drift.duplicated(['background','method','team']).any() and drift.membership_accuracy_unknown.all()
base=drift[drift.background=='prelim'];assert np.allclose(base.spearman,1,atol=1e-14) and (base.top12_tie_inclusive_overlap==base.reference_top12_rows).all()
assert max(errors)<1e-12
out={'status':'passed','method':'Independent selected-BG half-split/bin reconstruction, file hashes, owner/fold aggregation arithmetic and diagnostic scope checks; no membership truth loaded','backgrounds':7,'forms':3,'condition_panels':21,'matrix_hashes_recorded':21,'matrix_hashes_independently_rechecked':0,'matrix_limitation':'Original no-GT guard prohibits NPZ; matrices intentionally not persisted, so this audit verifies public aggregate outputs and provenance rather than rehashing matrices.','source_and_output_hashes_checked':hashes,'BG_half_sex_attribute_bin_sets_recomputed':cutschecked,'aggregate_numeric_cells_recomputed':cells,'maximum_aggregate_error':max(errors),'rank_diagnostic_rows':len(drift),'no_membership_accuracy_claim':True,'audit_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(R/'inventory/recovered_chow_independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False))

