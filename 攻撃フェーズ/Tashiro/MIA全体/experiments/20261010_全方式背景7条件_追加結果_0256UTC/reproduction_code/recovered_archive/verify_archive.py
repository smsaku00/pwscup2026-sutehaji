from run_archive import *
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(INPUT/'official_scoring/src'))
from pwscup2026.scoring.attack_score import _tpr_at_fpr
root=setup('prelim');audit=module('_verify_archive_truth',root/'wrappers/official_audit.py');truth=audit.load_truth(root)
errors=[];nmat=0;nrows=0;max_tpr=0.;max_auc=0.;frozen={};controls={}
for p in (BASE/'results').glob('*/*/completion.json'):
 meta=json.loads(p.read_text());dest=p.parent;fp=dest/'frozen.npz'
 if sha(fp)!=meta['freeze_sha256']:errors.append(str(fp)+' hash mismatch')
 table=pd.read_csv(dest/'by_target.csv',float_precision='round_trip').set_index(['method','team']);teams=meta['teams'];data=np.load(fp)
 for name in data.files:
  mat=data[name];nmat+=1
  for j,k in enumerate(teams):
   if np.isnan(mat[:,j]).all():
    if 'split_seed1' not in name or k in [24,26,29]:errors.append(f'{dest} {name} unexpected missing{k}')
    continue
   if not np.isfinite(mat[:,j]).all():errors.append(f'{dest} {name} nonfinite{k}');continue
   tpr,auc=_tpr_at_fpr(mat[:,j],truth[k],.01);row=table.loc[name,k];max_tpr=max(max_tpr,abs(tpr-row.tpr));max_auc=max(max_auc,abs(auc-row.auc));nrows+=1
   if abs(tpr-row.tpr)>1e-14 or abs(auc-row.auc)>1e-14:errors.append(f'{dest} {name} scorer mismatch{k}')
  if name in ['legacy_library_exact','legacy_redteam_event_time_exact']:controls.setdefault(name,[]).append((meta['background'],mat))
for name,items in controls.items():
 for bg,mat in items[1:]:
  if not np.array_equal(mat,items[0][1]):errors.append(f'{name} unexpected BG dependence {bg}')
status={'matrices_checked':nmat,'target_metrics_checked':nrows,'max_tpr_error':max_tpr,'max_auc_error':max_auc,'errors':errors,'unaffected_controls_background_counts':{k:len(v) for k,v in controls.items()},'status':'PASS' if not errors else 'FAIL'}
(BASE/'qa.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status))
