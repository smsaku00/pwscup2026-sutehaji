from pathlib import Path
import json,hashlib,datetime,gc
import numpy as np,pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent
exec((R/'audit_recovered_calibration.py').read_text().split('results={}')[0])
results={}
for family,folder,expected,versions in [('attribute_seeds','密度比_属性乱数結果',320,('2.3.5','1.8.0')),('stable_centers','中心配置の安定化結果',240,('2.4.6','1.9.0')),('classifiers','公開データ分類器結果',272,('2.4.6','1.9.0'))]:
 sets=[];errors=[];nhash=0;nrows=0;nsample=0;backgrounds=[];records=[]
 for bg in BG:
  p=S/family/bg
  if not (p/'実行記録.json').exists():continue
  rec=json.loads((p/'実行記録.json').read_text())
  if not rec.get('complete'):continue
  if family=='classifiers' and rec.get('forest_prediction_execution')!='fixed_tree_order_serial_predict_v1':continue
  initial_record_hash=sha(p/'実行記録.json')
  hs=json.loads((p/'prediction_hashes.json').read_text());hd={x['method']:x for x in hs}
  assert rec['models']==expected and sha(SRC/rec['source'])==rec['source_sha256']
  assert (rec['numpy'],rec['sklearn'])==versions
  assert rec['background']==bg and rec['phase']=='prelim'
  assert sha(Path('/workspace/shared/mia_bg7_inputs/bg/subsets')/(bg+'.csv'))==rec['background_sha256']
  d=read(p/'対象別TPR.csv');assert not d.duplicated(['method','team','fpr']).any();look=d.set_index(['method','team','fpr']);fprs=sorted(d.fpr.unique())
  with np.load(W/'work'/bg/folder/'内部順位キャッシュ.npz') as z:
   names=z['names'].tolist();scores=z['scores'];teams=z['teams'].tolist();assert z['record_ids'].tolist()==b.loc[normal,'record_id'].tolist()
   assert scores.shape==(expected,875,20) and len(names)==len(hs) and set(names)==set(hd) and teams==rec['team_order'] and np.isfinite(scores).all()
   assert len(d)==expected*20*len(fprs) and set(d.method)==set(names)
   for i,n in enumerate(names):assert hashlib.sha256(np.ascontiguousarray(scores[i],dtype='<f8').tobytes()).hexdigest()==hd[n]['sha256'];nhash+=1
   for i in sorted(set([0,expected-1]+list(range(0,expected,23)))):
    for j in [0,19]:
     conf=np.zeros(1049);conf[rare&~private]=.4;conf[normal]=.6+.3*rankdata(scores[i,:,j],method='min')/875
     for a in fprs:
      row=look.loc[(names[i],teams[j],a)];assert row.members==ys[teams[j]].sum()
      errors.append(abs(roc(conf,ys[teams[j]],a)-row.mia_all));nsample+=1
   sets.append(set(names));nrows+=len(d)
  assert sha(p/'実行記録.json')==initial_record_hash
  backgrounds.append(bg);records.append({'background':bg,'record_sha256':initial_record_hash,'source_sha256':rec['source_sha256'],'numpy':rec['numpy'],'sklearn':rec['sklearn'],'forest_prediction_execution':rec.get('forest_prediction_execution')})
  del scores,d,look;gc.collect();print('AUDIT',family,bg,flush=True)
 assert sets and all(s==sets[0] for s in sets) and max(errors)<1e-12
 results[family]={'status':'passed_completed_primary_backgrounds','configurations_per_BG':expected,'completed_backgrounds':backgrounds,'all7_complete':len(backgrounds)==7,'all_matrix_hashes_checked':nhash,'same_exact_config_set':True,'target_fpr_rows':nrows,'representative_TPR_values_recomputed':nsample,'max_abs_TPR_error':max(errors),'records':records}
results['audit_metadata']={'method':'Independent tie-group ROC; no source prediction/scorer modules imported. Exact primary folders only; earlier parallel/cloud controls excluded.','audit_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'historical_reproduction':'Historical HGB/forest/KMeans differences are not repaired by this current-score audit.'}
(R/'inventory/recovered_classifier_features_independent_audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:({kk:vv for kk,vv in v.items() if kk!='records'} if isinstance(v,dict) else v) for k,v in results.items()},ensure_ascii=False))
