from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
from scipy.stats import rankdata
R=Path(__file__).resolve().parent
A=Path('/workspace/shared/mia_all_bg_available/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/available')
D=Path('/workspace/shared/mia_bg7_inputs/root/公式データ/予備戦');BG=Path('/workspace/shared/mia_bg7_inputs/bg/subsets')
read=lambda p:pd.read_csv(p,float_precision='round_trip')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=read(next(D.glob('PWSCup2026_prelim_team05_*/B_self.csv')));q=read(D/'PWSCup2026_prelim_attack/aia_challenge_5.csv')
QI=['age','sex','prefecture','SBP','TG','HDL','ALT'];NUM=['age','SBP','TG','HDL','ALT']
private=pd.MultiIndex.from_frame(b[QI]).isin(pd.MultiIndex.from_frame(q[QI]));rare=b.is_rare.to_numpy(bool);normal=~private&~rare
assert (len(b),private.sum(),rare.sum(),normal.sum())==(1049,163,24,875)
pools={int(p.stem.rsplit('_',1)[1]):read(p) for p in (D/'結果/正解データ').glob('pool_ids_*.csv')}
assert np.array_equal(pools[5].record_id,b.record_id)
ys={t:pools[5].pool_id.isin(v.pool_id).to_numpy(int) for t,v in pools.items()}
def roc(s,y,alpha):
 ix=np.argsort(-s,kind='stable');ss=s[ix];yy=y[ix];ends=np.r_[np.where(np.diff(ss)!=0)[0],len(s)-1]
 tp=np.r_[0,np.cumsum(yy)[ends]];fp=np.r_[0,(np.arange(len(s))+1-np.cumsum(yy))[ends]]
 return float(np.interp(alpha,fp/(len(y)-y.sum()),tp/y.sum()))
checks=[]
for folder in ['results_prelim_primary','results_prelim']:
 P=A/folder;proto=json.loads((P/'fixed_protocol.json').read_text());by=read(P/'by_target.csv');errors=[];n=0
 expected=len(proto['backgrounds'])*len(proto['methods'])*sum(map(len,proto['inference_targets'].values()))*2
 assert len(by)==expected==3192
 assert not by.duplicated(['background','inference','method','team','population']).any()
 for bg in proto['backgrounds']:
  for scope,teams in proto['inference_targets'].items():
   with np.load(P/f'predictions_{bg}_{scope}.npz') as z:
    assert set(z.files)==set(proto['methods'])
    for method in proto['methods']:
     arr=z[method];assert arr.shape==(875,len(teams)) and np.isfinite(arr).all()
     for j,t in enumerate(teams):
      conf=np.zeros(1049);conf[rare&~private]=.4;conf[normal]=.6+.3*rankdata(arr[:,j],method='min')/875
      for pop,mask in [('all_fixed',np.ones(1049,bool)),('normal',normal)]:
       row=by[(by.background==bg)&(by.inference==scope)&(by.method==method)&(by.team==t)&(by.population==pop)].iloc[0]
       assert row.n==mask.sum() and row.members==ys[t][mask].sum()
       for alpha in [.005,.01,.02]:errors.append(abs(roc(conf[mask],ys[t][mask],alpha)-row[f'tpr_{alpha:g}']))
       n+=1
 assert max(errors)<1e-12
 checks.append({'folder':folder,'prediction_rows_checked':n,'roc_values_recomputed':len(errors),'max_abs_TPR_error':max(errors),'score_implementation':'independent tie-group ROC and full1049 confidence embedding; no experiment function imported'})
# Verify exact percentile edges from each selected BG and all matrix hash metadata.
hashcount=0;edgecount=0
for phase in ['prelim','main']:
 P=A/f'results_{phase}_primary';proto=json.loads((P/'fixed_protocol.json').read_text());edges=json.loads((P/'bin_edges.json').read_text())
 for bg in proto['backgrounds']:
  aa=read(BG/f'{bg}.csv')
  for col in NUM:
   assert np.array_equal(np.unique(np.quantile(aa[col],np.arange(.1,1,.1))),edges[bg][col]);edgecount+=1
 for h in json.loads((P/'prediction_hashes.json').read_text()):
  with np.load(P/f"predictions_{h['background']}_{h['inference']}.npz") as z:
   assert hashlib.sha256(np.ascontiguousarray(z[h['method']],dtype='<f8').tobytes()).hexdigest()==h['sha256'];hashcount+=1
 # Main reports rank comparisons only; no by-target membership accuracy artifact.
 if phase=='main':assert not (P/'by_target.csv').exists()
# Ensure old phase-fixed bin outputs are correctly named as controls.
control=json.loads((A/'results_prelim/fixed_protocol.json').read_text());primary=json.loads((A/'results_prelim_primary/fixed_protocol.json').read_text())
assert 'fixed across' in control['bin_edges_source'] and 'refit' in primary['bin_edges_source']
summary={'status':'passed','scope':'six source-complete recent methods only; not exhaustive historical MIA','checks':checks,'primary_bin_edges_checked':edgecount,'primary_prediction_hashes_checked':hashcount,'prelim_inference_targets':primary['inference_targets'],'normal_mask_counts':{'private':int(private.sum()),'rare':int(rare.sum()),'normal':int(normal.sum())},'full_pipeline_vs_control':{'primary':'bin edges recomputed from each selected BG; verified all numeric edges','control':'phase-fixed original bins explicitly labeled separately'},'main_TPR':'not_computed_no_truth','oracle_selection':'no method or background selected by this audit','truth_source_hashes':{str(p):sha(p) for p in (D/'結果/正解データ').glob('pool_ids_*.csv')}}
(R/'inventory/available_independent_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='truth_source_hashes'},ensure_ascii=False))
