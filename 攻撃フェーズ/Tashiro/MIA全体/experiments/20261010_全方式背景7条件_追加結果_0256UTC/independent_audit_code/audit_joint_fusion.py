from pathlib import Path
import json,hashlib,collections
import numpy as np,pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent;D=Path('/workspace/shared/mia_bg7_inputs/root/公式データ/予備戦')
read=lambda p:pd.read_csv(p,float_precision='round_trip');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=read(next(D.glob('PWSCup2026_prelim_team05_*/B_self.csv')));q=read(D/'PWSCup2026_prelim_attack/aia_challenge_5.csv');QI=['age','sex','prefecture','SBP','TG','HDL','ALT']
private=pd.MultiIndex.from_frame(b[QI]).isin(pd.MultiIndex.from_frame(q[QI]));rare=b.is_rare.to_numpy(bool);normal=~private&~rare
pools={int(p.stem.rsplit('_',1)[1]):read(p) for p in (D/'結果/正解データ').glob('pool_ids_*.csv')};assert np.array_equal(pools[5].record_id,b.record_id);ys={t:pools[5].pool_id.isin(v.pool_id).to_numpy(int) for t,v in pools.items()}
def roc(s,y,a):
 ix=np.argsort(-s,kind='stable');ss=s[ix];yy=y[ix];ends=np.r_[np.where(np.diff(ss)!=0)[0],len(s)-1];tp=np.r_[0,np.cumsum(yy)[ends]];fp=np.r_[0,(np.arange(len(s))+1-np.cumsum(yy))[ends]]
 return float(np.interp(a,fp/(len(y)-y.sum()),tp/y.sum()))
BG=('starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main');reports={}
for kind in ['joint','fusion']:
 stage=Path('/workspace/shared/mia_all_bg_'+kind)/'攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件'/kind
 table=read(stage/('prelim/by_target.csv' if kind=='joint' else 'prelim_by_target.csv'))
 keys=['background','family','inference','method','team','population'];assert not table.duplicated(keys).any();lookup=table.set_index(keys)
 errors=[];sample_rows=0;hashes=0;coverage={};expected_prediction_target_rows=0;configuration_sets={}
 for phase in ['prelim','main']:
  counts=[];sets=[]
  for bg in BG:
   records=json.loads((stage/phase/bg/'prediction_hashes.json').read_text());counts.append(len(records));sets.append({(x['family'],x['inference'],x['method']) for x in records})
   cache=Path('/workspace/shared/mia_joint_cache' if kind=='joint' else '/workspace/shared/mia_fusion_cache')/f'{phase}_{bg}.npz'
   with np.load(cache) as z:
    assert len(z.files)==len(records)
    for i,item in enumerate(records):
     x=z[str(i) if kind=='joint' else item['key']];assert x.shape==(875,len(item['teams'])) and np.isfinite(x).all()
     assert hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()==item['sha256'];hashes+=1
     if phase!='prelim':continue
     expected_prediction_target_rows+=len(item['teams'])*(2 if kind=='joint' else 3)
     # First and last target for every method/inference/config, in every BG.
     for j in sorted(set([0,len(item['teams'])-1])):
      t=item['teams'][j];conf=np.zeros(1049);conf[rare&~private]=.4;conf[normal]=.6+.3*rankdata(x[:,j],method='min')/875
      populations=[('all_fixed',np.ones(1049,bool)),('normal',normal)] if kind=='joint' else [('all_fixed',np.ones(1049,bool)),('nonrare',~rare),('normal',normal)]
      for pop,mask in populations:
       if kind=='fusion' and (bg,item['family'],item['inference'],item['method'],t,pop) not in lookup.index:pop='normal_unknown_nonrare'
       row=lookup.loc[(bg,item['family'],item['inference'],item['method'],t,pop)]
       assert int(row.n)==mask.sum() and int(row.members)==ys[t][mask].sum()
       for a in [.005,.01,.02]:errors.append(abs(roc(conf[mask],ys[t][mask],a)-row[f'tpr_{a:g}']))
       sample_rows+=1
  assert len(set(counts))==1 and all(s==sets[0] for s in sets)
  coverage[phase]={'backgrounds':7,'configurations_per_background':counts[0],'configurations_total':sum(counts),'same_exact_config_set_across_backgrounds':True}
  configuration_sets[phase]=[{'family':f,'inference':inf,'method':m} for f,inf,m in sorted(sets[0])]
 assert len(table)==expected_prediction_target_rows
 assert max(errors)<1e-12
 reports[kind]={'status':'passed','coverage':coverage,'stored_target_rows':len(table),'sample_rows_recomputed':sample_rows,'TPR_values_recomputed':len(errors),'max_abs_TPR_error':max(errors),'prediction_hashes_checked':hashes,'population_policy':'full1049 official-style plus explicitly labeled normal875; fusion also nonrare1025','method':'independent tie-group ROC; experiment methods not imported','config_sets':configuration_sets}
 if kind=='joint':
  assert coverage['prelim']['configurations_per_background']==203 and coverage['main']['configurations_per_background']==82
  assert not any(x['family']=='p02' or x['method'].startswith(('soft_box','hard_')) for x in configuration_sets['main'])
 else:assert coverage['prelim']['configurations_per_background']==60 and coverage['main']['configurations_per_background']==30
(R/'inventory/joint_fusion_independent_audit.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:{a:b for a,b in v.items() if a!='config_sets'} for k,v in reports.items()},ensure_ascii=False))
