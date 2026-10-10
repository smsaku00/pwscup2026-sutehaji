from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;D=Path('/workspace/shared/mia_bg7_inputs/root/公式データ/予備戦')
A=Path('/workspace/shared/mia_all_bg_legacy/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_archive')
read=lambda p:pd.read_csv(p,float_precision='round_trip');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=read(next(D.glob('PWSCup2026_prelim_team05_*/B_self.csv')))
pools={int(p.stem.rsplit('_',1)[1]):read(p) for p in (D/'結果/正解データ').glob('pool_ids_*.csv')}
assert np.array_equal(pools[5].record_id,b.record_id)
ys={t:pools[5].pool_id.isin(v.pool_id).to_numpy(int) for t,v in pools.items()}
def metrics(s,y):
 ix=np.argsort(-s,kind='stable');ss=s[ix];yy=y[ix];ends=np.r_[np.where(np.diff(ss)!=0)[0],len(s)-1]
 tp=np.r_[0,np.cumsum(yy)[ends]]/y.sum();fp=np.r_[0,(np.arange(len(s))+1-np.cumsum(yy))[ends]]/(len(y)-y.sum())
 return float(np.interp(.01,fp,tp)),float(np.trapezoid(tp,fp))
errors=[];counts=[];rows=0;matrices=0;seen=set();control_hashes={}
for p in sorted((A/'results').glob('*/*/completion.json')):
 rec=json.loads(p.read_text());dest=p.parent
 assert rec['predictions_frozen_before_evaluation_labels'] and not rec['membership_training_labels_used']
 assert sha(dest/'frozen.npz')==rec['freeze_sha256']
 d=read(dest/'by_target.csv');assert not d.duplicated(['method','team']).any();look=d.set_index(['method','team'])
 with np.load(dest/'frozen.npz') as z:
  assert set(z.files)==set(rec['methods']) and len(z.files)==rec['n_methods']
  for name in rec['methods']:
   mat=z[name];assert mat.shape==(1049,len(rec['teams']))
   applicable=[24,26,29] if name.startswith('legacy_atk_split_seed1_original_') else rec['teams']
   for j,t in enumerate(rec['teams']):assert np.isfinite(mat[:,j]).all() if t in applicable else np.isnan(mat[:,j]).all()
   assert (name,rec['background']) not in seen;seen.add((name,rec['background']));matrices+=1
   if name in ['legacy_library_exact','legacy_redteam_event_time_exact']:control_hashes.setdefault(name,set()).add(hashlib.sha256(mat.tobytes()).hexdigest())
   for j,t in enumerate(rec['teams']):
    if t not in applicable:continue
    row=look.loc[(name,t)];assert row.members==ys[t].sum()
    a,b=metrics(mat[:,j],ys[t]);errors.extend([abs(a-row.tpr),abs(b-row.auc)]);rows+=1
  assert len(d)==sum(3 if n.startswith('legacy_atk_split_seed1_original_') else len(rec['teams']) for n in z.files)
 counts.append({'background':rec['background'],'family':rec['family'],'methods':len(rec['methods']),'targets':len(rec['teams']),'freeze_sha256':rec['freeze_sha256']})
assert max(errors)<1e-12 and all(len(x)==1 for x in control_hashes.values())
status=json.loads((A/'aggregate_status.json').read_text())
out={'status':'passed_for_all_completed_family_jobs','audit_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'family_background_jobs_checked':len(counts),'matrix_files_sha256_checked':len(counts),'method_background_conditions_checked':matrices,'all_completed_target_rows_recomputed':rows,'TPR_and_AUC_values_recomputed':len(errors),'maximum_absolute_error':max(errors),'BG_independent_controls_verified_invariant':{k:len(v)==1 for k,v in control_hashes.items()},'method':'Independent tie-group ROC and trapezoidal AUC on native full1049 frozen confidence, no prediction/scorer modules imported','scope':'Exactly completed source-native rosters; three atk split-seed original methods have only teams24/26/29. Completion status is not historical numerical equivalence.','coverage_at_audit':status,'jobs':counts}
(R/'inventory/recovered_archive_independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['jobs','coverage_at_audit']},ensure_ascii=False))

