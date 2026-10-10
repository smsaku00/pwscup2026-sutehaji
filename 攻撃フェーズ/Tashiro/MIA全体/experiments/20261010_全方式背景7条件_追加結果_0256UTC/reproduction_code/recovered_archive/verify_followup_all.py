from pathlib import Path
import json,sys,time,argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import run_archive as ar
import run_followup as f
import run_composites as c
import run_submission7 as s7
p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
if args.wait:
 while len(list((f.ROOT/'results').glob('*/*/completion.json')))<21:time.sleep(20)
assert len(list((f.ROOT/'results').glob('*/*/completion.json')))==21
f.aggregate()
sys.path.insert(0,str(ar.INPUT/'official_scoring/src'))
from pwscup2026.scoring.attack_score import _tpr_at_fpr
audit=ar.module('_followup_all_verify_audit',ar.WRAPPERS/'official_audit.py');truth=audit.load_truth(ar.setup('prelim'));b,private,rare,normal,dist=audit.load_inputs(ar.setup('prelim'))
baseline=pd.read_csv(c.BASELINE,float_precision='round_trip').iloc[:,1:].to_numpy();mask=~rare&~private
checked=0;maxerr=0.;matrices=0;composition_values=0
for bg in ar.BGS:
 for family in ['r1','composites','submission7']:
  d=f.ROOT/'results'/bg/family;m=json.loads((d/'prediction_manifest.json').read_text());assert m['freeze_sha256']==ar.sha(d/'frozen.npz')
  assert m['predictions_frozen_before_evaluation_labels'] and not m['membership_training_labels_used']
  scores=np.load(d/'frozen.npz');table=pd.read_csv(d/'by_target.csv',float_precision='round_trip').set_index(['method','team']);assert set(scores.files)==set(f.METHODS if family=='r1' else c.METHODS if family=='composites' else [s7.METHOD])
  if family in ['composites','submission7']:raw=np.load(ar.BASE/'results'/bg/'raw_gen/frozen.npz')
  for method in scores.files:
   matrix=scores[method];assert matrix.shape==(1049,20) and np.isfinite(matrix).all() and (matrix>=0).all() and (matrix<=1).all();matrices+=1
   mm=m['metadata'][method];assert mm['source_sha256']==ar.sha(ar.REC/mm['source']);assert mm['applicable_target_ids']==ar.TEAMS
   if family=='composites':
    key={'legacy_gen7_nonrare_historical_replace':'legacy_raw_qi7_gen','legacy_gen9_nonrare_historical_replace':'legacy_raw_attrs9','legacy_gen10_nonrare_historical_replace':'legacy_raw_attrs10_gen'}[method]
    expected=baseline.copy()
    for j,k in enumerate(ar.TEAMS):
     if k not in [1,2]:expected[mask,j]=.1+.8*rankdata(raw[key][mask,j],method='average')/(mask.sum()+1);expected[~rare&private,j]=0
    assert np.array_equal(matrix,expected);composition_values+=matrix.size
   if family=='submission7':
    settings=json.loads(s7.SETTINGS.read_text());base6=pd.read_csv(s7.BASELINE,float_precision='round_trip');expected=base6.iloc[:,1:].to_numpy().copy();ids=base6.record_id.to_numpy();pos={int(v):i for i,v in enumerate(ids)};known={(v['team'],v['record_id']) for v in settings['known_negative_pairs']}
    for j,k in enumerate(ar.TEAMS):
     cap=expected[rare,j].max()
     if k not in (1,2):expected[mask,j]=cap+(1-cap)*(.1+.8*rankdata(raw['legacy_raw_qi7_gen'][mask,j],method='min')/875)
     expected[rare,j]=0
     order=sorted([v for v in settings['eligible_ids'] if (k,v) not in known],key=lambda v:(-raw['legacy_raw_attrs9'][pos[v],j],v))
     for rank0,v in enumerate(order):expected[pos[v],j]=cap*(.9-.8*rank0/len(order))
    assert np.array_equal(matrix,expected);composition_values+=matrix.size
   for j,k in enumerate(ar.TEAMS):
    actual=_tpr_at_fpr(matrix[:,j],truth[k],.01);saved=table.loc[(method,k),['tpr','auc']].to_numpy(float);err=float(np.max(np.abs(actual-saved)));maxerr=max(maxerr,err);assert err<1e-12;checked+=2
base=pd.read_csv(f.ROOT/'baseline_validation.csv');assert len(base)==6
assert base.exact_tpr_rows.eq(20).all() and base.exact_auc_rows.eq(base.historical_auc_available_rows).all(),base.to_string()
q=dict(status='PASS',frozen_matrices=matrices,target_rows=checked//2,recomputed_metric_values=checked,max_absolute_error=maxerr,independent_composition_values=composition_values,independent_composition_max_error=0.,historical_baseline_target_rows=120,historical_baseline_tpr_rows_exact=120,historical_baseline_auc_rows_available_and_exact=60,historical_baseline_auc_rows_unavailable=60,original_r1_score_bytes_not_available=True,historical_r1_input_and_state_uncertainty_preserved=True,submission8_baseline_sha256=c.BASELINE_SHA)
(f.ROOT/'qa_all.json').write_text(json.dumps(q,indent=2)+'\n')
t=pd.read_csv(f.ROOT/'by_target.csv');assert len(t)==840 and len(t.groupby(['method','background']))==42
base=t[t.background=='prelim'][['method','team','tpr']].rename(columns={'tpr':'prelim_tpr'});changes=t.merge(base,on=['method','team'],validate='many_to_one');changes['tpr_delta']=changes.tpr-changes.prelim_tpr;changes['newly_at_least_010']=(changes.prelim_tpr<.1)&(changes.tpr>=.1);changes.to_csv(f.ROOT/'background_changes.csv',index=False,float_format='%.17g');changes[~changes.team.isin([1,2])].sort_values('tpr_delta',ascending=False).to_csv(f.ROOT/'non12_same_config_gains.csv',index=False,float_format='%.17g')
manifest=dict(status='COMPLETE_AVAILABLE_FOLLOWUP',background_dependent_recipes=6,method_background_conditions=42,target_rows=840,qa=q,blocked_composite_recipes=0,missing_generator_recipes=3,methods=f.METHODS+c.METHODS+[s7.METHOD],backgrounds=ar.BGS)
(f.ROOT/'completion_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
p=f.ROOT/'README.md';txt=p.read_text();txt=txt.replace('固定済みraw scoreと元の合成式で7背景を実行する。','固定済みraw scoreと元の合成式で7背景を実行した。追加6方式×7背景の42条件・840対象行、1,680個の公式TPR/AUC値が全件再採点一致。合成587,440セルは独立した式でも完全一致。6方式120対象行の旧prelim背景TPRと、保存履歴にAUCがあるR1および提出7の60対象行は全件一致。Gen差替3方式の旧AUC60行は保存履歴にないため比較対象外。');p.write_text(txt)
print(json.dumps(q))
