from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import run_archive as ar
import run_followup as f
sys.path.insert(0,str(ar.INPUT/'official_scoring/src'))
from pwscup2026.scoring.attack_score import _tpr_at_fpr
root=f.ROOT
assert all((root/'results'/bg/'r1/completion.json').exists() for bg in ar.BGS)
truth=ar.module('_followup_verify_audit',ar.WRAPPERS/'official_audit.py').load_truth(ar.setup('prelim'))
checked=0;maxerr=0.;matrices=0
for bg in ar.BGS:
 d=root/'results'/bg/'r1';m=json.loads((d/'prediction_manifest.json').read_text());assert m['freeze_sha256']==ar.sha(d/'frozen.npz')
 assert m['predictions_frozen_before_evaluation_labels'] and not m['membership_training_labels_used']
 scores=np.load(d/'frozen.npz');t=pd.read_csv(d/'by_target.csv',float_precision='round_trip').set_index(['method','team']);assert set(scores.files)==set(f.METHODS)
 for method in scores.files:
  matrix=scores[method];assert matrix.shape==(1049,20) and np.isfinite(matrix).all() and (matrix>=0).all() and (matrix<=1).all();matrices+=1
  mm=m['metadata'][method];assert mm['source_sha256']==ar.sha(ar.REC/mm['source']);assert mm['applicable_target_ids']==ar.TEAMS
  for j,k in enumerate(ar.TEAMS):
   actual=_tpr_at_fpr(matrix[:,j],truth[k],.01);saved=t.loc[(method,k),['tpr','auc']].to_numpy(float);err=float(np.max(np.abs(actual-saved)));maxerr=max(maxerr,err);assert err<1e-12;checked+=2
b=pd.read_csv(root/'baseline_validation.csv');assert b.exact_tpr_rows.eq(20).all() and b.exact_auc_rows.eq(20).all()
q=dict(status='PASS',frozen_matrices=matrices,target_rows=checked//2,recomputed_metric_values=checked,max_absolute_error=maxerr,historical_baseline_target_rows=40,historical_baseline_tpr_auc_rows_exact=40,original_score_bytes_not_available=True,historical_input_and_state_uncertainty_preserved=True)
(root/'qa.json').write_text(json.dumps(q,indent=2)+'\n')
t=pd.read_csv(root/'by_target.csv');s=pd.read_csv(root/'summary.csv');assert len(t)==280 and len(t.groupby(['method','background']))==14
base=t[t.background=='prelim'][['method','team','tpr']].rename(columns={'tpr':'prelim_tpr'});c=t.merge(base,on=['method','team'],validate='many_to_one');c['tpr_delta']=c.tpr-c.prelim_tpr;c['newly_at_least_010']=(c.prelim_tpr<.1)&(c.tpr>=.1);c.to_csv(root/'background_changes.csv',index=False,float_format='%.17g');c[~c.team.isin([1,2])].sort_values('tpr_delta',ascending=False).to_csv(root/'non12_same_config_gains.csv',index=False,float_format='%.17g')
manifest=dict(status='COMPLETE_AVAILABLE_R1_FOLLOWUP',background_dependent_recipes=2,method_background_conditions=14,target_rows=280,qa=q,blocked_composite_recipes=4,missing_generator_recipes=3,methods=f.METHODS,backgrounds=ar.BGS)
(root/'completion_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
p=root/'README.md';txt=p.read_text();txt=txt.replace('元20対象、元のCSV読み込み精度、閾値・6種seedを維持する。','元20対象、元のCSV読み込み精度、閾値・6種seedを維持する。2方式×7背景の14条件・280対象行が完了し、560個の公式TPR/AUC値を再計算して一致した。旧prelim背景の40対象行は保存済みTPR/AUCと全件一致した。');p.write_text(txt)
print(json.dumps(q))
