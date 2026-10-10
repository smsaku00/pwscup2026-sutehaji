# Reuse only independent audit helpers, never experiment/scorer code.
from pathlib import Path
R=Path(__file__).resolve().parent
exec(compile((R/'audit_recovered_calibration.py').read_text().split('results={}')[0],str(R/'audit_recovered_calibration.py'),'exec'))
S=S/'diagnostics'
errors=[];nhash=0;ntpr=0;rows=0;panels=[];family_sets={}
for p in sorted(S.glob('*/*/execution_record.json')):
 rec=json.loads(p.read_text())
 if not(rec.get('evaluation_complete') and rec.get('phase')=='prelim'):continue
 pred=json.loads((p.parent/'prediction_complete.json').read_text());hp=json.loads((p.parent/'prediction_hashes.json').read_text());hd={x['method']:x for x in hp}
 assert rec['prediction_process_guard'] and not rec['source_membership_labels_used'] and not rec['truth_used_for_selection']
 assert sha(Path(pred['source_path']))==pred['source_sha256'] and sha(Path(pred['cache_path']))==pred['cache_sha256']
 assert sha(Path('/workspace/shared/mia_bg7_inputs/bg/subsets')/(rec['background']+'.csv'))==rec['background_sha256']
 d=read(p.parent/'対象別TPR.csv');assert not d.duplicated(['method','team','fpr']).any();lookup=d.set_index(['method','team','fpr']);fprs=sorted(d.fpr.unique())
 with np.load(pred['cache_path']) as z:
  names=z['names'].tolist();scores=z[pred['cache_key']];teams=rec['teams']
  assert len(names)==rec['models'] and set(names)==set(hd) and scores.shape==(len(names),875,20) and np.isfinite(scores).all()
  for i,n in enumerate(names):
   assert hashlib.sha256(np.ascontiguousarray(scores[i],dtype='<f8').tobytes()).hexdigest()==hd[n]['sha256'];nhash+=1
   # Every target of every completed output, all available outer backgrounds.
   for j in range(len(teams)):
    conf=np.zeros(1049);conf[rare&~private]=.4;conf[normal]=.6+.3*rankdata(scores[i,:,j],method='min')/875
    for a in fprs:
     row=lookup.loc[(n,teams[j],a)];assert row.members==ys[teams[j]].sum();errors.append(abs(roc(conf,ys[teams[j]],a)-row.mia_all));ntpr+=1
  assert len(d)==len(names)*20*len(fprs);rows+=len(d)
  family_sets.setdefault(rec['family'],[]).append((rec['background'],set(names)))
 panels.append({'family':rec['family'],'background':rec['background'],'outputs':rec['models'],'source_sha256':rec['source_sha256'],'cache_sha256':pred['cache_sha256']})
assert max(errors)<1e-12
for f,sets in family_sets.items():assert all(x[1]==sets[0][1] for x in sets)
out={'status':'passed_for_completed_runs','family_BG_jobs':len(panels),'per_family_completed_BG_counts':{f:len(x) for f,x in family_sets.items()},'matrix_hashes_rechecked':nhash,'all_output_sets_consistent_within_family':True,'stored_target_fpr_rows':rows,'all_TPR_values_recomputed':ntpr,'max_abs_TPR_error':max(errors),'method':'Independent tie-group ROC; hash/input/source verification; prediction/scoring process separation checked from frozen records','audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'panels':panels}
(R/'inventory/recovered_public_diagnostics_independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='panels'},ensure_ascii=False))

