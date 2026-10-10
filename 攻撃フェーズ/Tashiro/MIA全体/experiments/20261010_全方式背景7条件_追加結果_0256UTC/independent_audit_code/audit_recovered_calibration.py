from pathlib import Path
import json,hashlib,datetime,gc
import numpy as np,pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent
D=Path('/workspace/shared/mia_bg7_inputs/root/公式データ/予備戦')
S=Path('/workspace/shared/mia_all_bg_recovered_calibration/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_calibration')
W=Path('/workspace/shared/mia_recovered_calibration')
SRC=Path('/workspace/shared/mia_recovered_sources/攻撃フェーズ/Tashiro/source_recovery/20261010_MIA元コード回収/originals/本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索')
read=lambda p:pd.read_csv(p,float_precision='round_trip')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=read(next(D.glob('PWSCup2026_prelim_team05_*/B_self.csv')));q=read(D/'PWSCup2026_prelim_attack/aia_challenge_5.csv')
QI=['age','sex','prefecture','SBP','TG','HDL','ALT'];private=pd.MultiIndex.from_frame(b[QI]).isin(pd.MultiIndex.from_frame(q[QI]));rare=b.is_rare.to_numpy(bool);normal=~private&~rare
pools={int(p.stem.rsplit('_',1)[1]):read(p) for p in (D/'結果/正解データ').glob('pool_ids_*.csv')}
assert np.array_equal(pools[5].record_id,b.record_id)
ys={t:pools[5].pool_id.isin(v.pool_id).to_numpy(int) for t,v in pools.items()}
def roc(s,y,a):
 ix=np.argsort(-s,kind='stable');ss=s[ix];yy=y[ix];ends=np.r_[np.where(np.diff(ss)!=0)[0],len(s)-1]
 tp=np.r_[0,np.cumsum(yy)[ends]]/y.sum();fp=np.r_[0,(np.arange(len(s))+1-np.cumsum(yy))[ends]]/(len(y)-y.sum())
 return float(np.interp(a,fp,tp))
BG=['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
results={}
for family,folder,expected in [('direct','直接密度比結果',1008)]:
 sets=[];errors=[];nhash=0;nrows=0;nsample=0
 for bg in BG:
  p=S/family/bg;rec=json.loads((p/'実行記録.json').read_text());hs=json.loads((p/'prediction_hashes.json').read_text());hd={x['method']:x for x in hs}
  assert rec['complete'] and rec['models']==expected and sha(SRC/rec['source'])==rec['source_sha256']
  assert sha(Path('/workspace/shared/mia_bg7_inputs/bg/subsets')/(bg+'.csv'))==rec['background_sha256']
  d=read(p/'対象別TPR.csv');assert not d.duplicated(['method','team','fpr']).any();look=d.set_index(['method','team','fpr']);fprs=sorted(d.fpr.unique())
  with np.load(W/'work'/bg/folder/'内部順位キャッシュ.npz') as z:
   names=z['names'].tolist();scores=z['scores'];teams=z['teams'].tolist();assert z['record_ids'].tolist()==b.loc[normal,'record_id'].tolist()
   assert scores.shape==(expected,875,20) and len(names)==len(hs) and set(names)==set(hd) and teams==rec['team_order'] and np.isfinite(scores).all()
   assert len(d)==expected*20*len(fprs) and set(d.method)==set(names)
   for i,n in enumerate(names):assert hashlib.sha256(np.ascontiguousarray(scores[i],dtype='<f8').tobytes()).hexdigest()==hd[n]['sha256'];nhash+=1
   for i in sorted(set([0,expected-1]+list(range(0,expected,64)))):
    for j in [0,19]:
     conf=np.zeros(1049);conf[rare&~private]=.4;conf[normal]=.6+.3*rankdata(scores[i,:,j],method='min')/875
     for a in fprs:
      row=look.loc[(names[i],teams[j],a)];assert row.members==ys[teams[j]].sum()
      errors.append(abs(roc(conf,ys[teams[j]],a)-row.mia_all));nsample+=1
   sets.append(set(names));nrows+=len(d)
  del scores,d,look;gc.collect();print('AUDIT',family,bg,flush=True)
 assert all(s==sets[0] for s in sets) and max(errors)<1e-12
 results[family]={'status':'passed','configurations_per_BG':expected,'backgrounds':7,'all_matrix_hashes_checked':nhash,'same_exact_config_set':True,'target_fpr_rows':nrows,'representative_TPR_values_recomputed':nsample,'max_abs_TPR_error':max(errors)}
# Broad calibration is separately namespaced, retaining7 source-proven duplicated controls.
errors=[];sets=[];nhash=0;nsample=0
d=read(S/'broad/prelim/by_target.csv');assert not d.duplicated(['background','family','method','team']).any();look=d.set_index(['background','family','method','team'])
teams=json.loads((S/'direct/prelim/実行記録.json').read_text())['team_order'];assert len(teams)==20 and set(teams)==set(d.team)
for phase in ['prelim','main']:
 phasesets=[]
 for bg in BG:
  p=S/'broad'/phase/bg;hs=json.loads((p/'prediction_hashes.json').read_text());rec=json.loads((p/'record.json').read_text())
  assert len(hs)==96 and rec['methods']==96 and rec['labels_used_for_prediction'] is False
  assert rec['bg_sha256']==sha(Path('/workspace/shared/mia_bg7_inputs/bg/subsets')/(bg+'.csv'))
  with np.load(W/'cache/broad'/f'{phase}_{bg}.npz') as z:
   assert set(z.files)=={x['family']+'::'+x['method'] for x in hs};phasesets.append(set(z.files))
   for h in hs:
    n=h['family']+'::'+h['method'];x=z[n];assert np.isfinite(x).all() and list(x.shape)==h['shape'] and x.shape[0]==875
    assert hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()==h['sha256'];nhash+=1
    if phase=='prelim':
     for j in [0,19]:
      conf=np.zeros(1049);conf[rare&~private]=.4;conf[normal]=.6+.3*rankdata(x[:,j],method='min')/875
      row=look.loc[(bg,h['family'],h['method'],teams[j])];assert row.members==ys[teams[j]].sum()
      for a in [.005,.01,.02,.05]:errors.append(abs(roc(conf,ys[teams[j]],a)-row[f'tpr_{a:g}']));nsample+=1
  print('AUDIT broad',phase,bg,flush=True)
 assert all(s==phasesets[0] for s in phasesets);sets.append(phasesets[0])
assert sets[0]==sets[1] and max(errors)<1e-12 and len(d)==96*7*20
results['broad']={'status':'passed','source_registrations_per_BG':96,'source_proven_distinct_per_BG':89,'backgrounds':7,'phases':['prelim','main_rank_only'],'all_matrix_hashes_checked':nhash,'same_exact_config_set':True,'prelim_target_rows':len(d),'representative_TPR_values_recomputed':nsample,'max_abs_TPR_error':max(errors)}
results['audit_metadata']={'method':'Independent tie-group ROC; source methods not imported','audit_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(R/'inventory/recovered_calibration_independent_audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(results,ensure_ascii=False))

