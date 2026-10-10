from pathlib import Path
import json,hashlib,datetime,gc
import numpy as np,pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent
D=Path('/workspace/shared/mia_bg7_inputs/root/公式データ/予備戦')
H=Path('/workspace/shared/mia_all_bg_fusion/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_oct8/alignment/prelim')
read=lambda p:pd.read_csv(p,float_precision='round_trip')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=read(next(D.glob('PWSCup2026_prelim_team05_*/B_self.csv')));q=read(D/'PWSCup2026_prelim_attack/aia_challenge_5.csv')
QI=['age','sex','prefecture','SBP','TG','HDL','ALT']
private=pd.MultiIndex.from_frame(b[QI]).isin(pd.MultiIndex.from_frame(q[QI])); rare=b.is_rare.to_numpy(bool);normal=~private&~rare
pools={int(p.stem.rsplit('_',1)[1]):read(p) for p in (D/'結果/正解データ').glob('pool_ids_*.csv')}
assert np.array_equal(pools[5].record_id,b.record_id)
ys={t:pools[5].pool_id.isin(v.pool_id).to_numpy(int) for t,v in pools.items()}
def metrics(s,y):
 ix=np.argsort(-s,kind='stable');ss=s[ix];yy=y[ix];ends=np.r_[np.where(np.diff(ss)!=0)[0],len(s)-1]
 tp=np.r_[0,np.cumsum(yy)[ends]]/y.sum();fp=np.r_[0,(np.arange(len(s))+1-np.cumsum(yy))[ends]]/(len(y)-y.sum())
 return [float(np.interp(a,fp,tp)) for a in [.005,.01,.02,.05]]+[float(np.trapezoid(tp,fp))]
BG=['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
report={'status':'running','backgrounds':{},'method':'independent tie-group ROC and trapezoidal AUC; no experiment methods imported','sampling':'First and last method, then every 33rd sorted source output; first and last target; all three populations; every BG'}
sets=[];errors=[];count=0;totalrows=0
assert (H/'completion.json').exists(), 'Family is not sealed yet'
protocol=json.loads((H/'fixed_protocol.json').read_text())
for bg in BG:
 p=H/bg;rec=json.loads((p/'prediction_record.json').read_text()); hashes=json.loads((p/'prediction_hashes.json').read_text())
 assert sha(p/'scores.npz')==rec['scores_sha256']
 assert sha(Path('/workspace/shared/mia_bg7_inputs/bg/subsets')/(bg+'.csv'))==protocol['BG_inputs'][bg]
 assert rec['truth_read'] is False and rec['methods']==750
 by=read(p/'by_target.csv');assert len(by)==750*20*3
 assert not by.duplicated(['method','team','population']).any();lookup=by.set_index(['method','team','population'])
 with np.load(p/'scores.npz') as z:
  names=z['names'].tolist();scores=z['scores'];teams=z['teams'].tolist()
  assert names==[x['method'] for x in hashes] and scores.shape==(750,875,20)
  assert teams==rec['targets'] and np.isfinite(scores).all()
  for i,x in enumerate(scores):assert hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()==hashes[i]['sha256']
  for i in sorted(set([0,749]+list(range(0,750,33)))):
   for j in [0,19]:
    conf=np.zeros(1049);conf[rare&~private]=.4;conf[normal]=.6+.3*rankdata(scores[i,:,j],method='min')/875
    for pop,mask in [('all_fixed',np.ones(1049,bool)),('nonrare',~rare),('normal',normal)]:
     row=lookup.loc[(names[i],teams[j],pop)]
     assert row.n==mask.sum() and row.members==ys[teams[j]][mask].sum()
     vals=metrics(conf[mask],ys[teams[j]][mask]);expected=[row['tpr_'+str(a)] for a in [.005,.01,.02,.05]]+[row.auc]
     errors.extend(abs(x-y) for x,y in zip(vals,expected));count+=1
 sets.append(set(names));totalrows+=len(by)
 report['backgrounds'][bg]={'settings':len(names),'targets':len(teams),'target_population_rows':len(by),'all_prediction_hashes_passed':True,'score_file_sha256':rec['scores_sha256'],'prediction_frozen_at':rec['frozen_at']}
 del scores,by,lookup;gc.collect()
 print('AUDIT',bg,flush=True)
assert all(x==sets[0] for x in sets) and max(errors)<1e-12
report.update(status='passed',settings_per_background=750,backgrounds_complete=7,condition_panels=5250,stored_target_population_rows=totalrows,same_exact_config_set_across_backgrounds=True,prediction_matrix_hashes_checked=5250,representative_target_population_rows_checked=count,metric_values_checked=len(errors),max_absolute_metric_error=max(errors),historical_regression_scope='Historical source/catalog regression is recorded separately by the source runner; current independently checked all20 inference/evaluation scores do not establish historical byte reproduction.',audited_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(R/'inventory/recovered_alignment_independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='backgrounds'},ensure_ascii=False))

