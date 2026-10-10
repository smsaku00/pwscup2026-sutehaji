from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parent;D=Path('/workspace/shared/mia_bg7_inputs/root/公式データ/予備戦')
A=Path('/workspace/shared/mia_all_bg_legacy/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_archive')
read=lambda p:pd.read_csv(p,float_precision='round_trip');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=read(next(D.glob('PWSCup2026_prelim_team05_*/B_self.csv')));q=read(D/'PWSCup2026_prelim_attack/aia_challenge_5.csv');QI=['age','sex','prefecture','SBP','TG','HDL','ALT']
private=pd.MultiIndex.from_frame(b[QI]).isin(pd.MultiIndex.from_frame(q[QI]));rare=b.is_rare.to_numpy(bool);rr=b.loc[rare].sort_values('record_id')
eligible=~rr.record_id.isin(b.loc[private&rare,'record_id']).to_numpy()
# Source-explicit historical two exclusions retained solely as a diagnostic scope; no posterior/labels reconstructed.
remaining=eligible&~rr.record_id.isin([201,238]).to_numpy()
masks={'all24':np.ones(24,bool),'eligible11':eligible,'remaining9_historical':remaining};assert [v.sum() for v in masks.values()]==[24,11,9]
summary=read(A/'rare_secondary_summary.csv').set_index(['background','method','scope']);change=read(A/'rare_secondary_background_changes.csv').set_index(['background','method','scope'])
assert len(summary)==len(change)==26*7*3 and summary.index.is_unique and change.index.is_unique
errors=[];checked=0;sets=[];h=0
with np.load(A/'rare_secondary/prelim/frozen.npz') as z:base={n:z[n].copy() for n in z.files}
for p in sorted((A/'rare_secondary').glob('*/completion.json')):
 bg=p.parent.name;rec=json.loads(p.read_text());assert rec['prediction_only'] and not rec['membership_truth_read'] and sha(p.parent/'frozen.npz')==rec['frozen_sha256'];h+=1
 with np.load(p.parent/'frozen.npz') as z:
  assert len(z.files)==26;sets.append(set(z.files))
  for n in z.files:
   v=z[n];assert v.shape==(24,) and np.isfinite(v).all()
   for scope,mask in masks.items():
    x=v[mask];ref=base[n][mask];row=summary.loc[(bg,n,scope)];drift=change.loc[(bg,n,scope)]
    assert row.n_candidates==len(x) and row.n_top_ties==(x==x.max()).sum()
    values={'score_min':x.min(),'score_max':x.max(),'score_mean':x.mean(),'score_std':x.std()}
    for key,val in values.items():errors.append(abs(val-row[key]));checked+=1
    corr=float(np.corrcoef(rankdata(x),rankdata(ref))[0,1]);errors.extend([abs(corr-drift.spearman_prelim),abs(np.max(np.abs(x-ref))-drift.max_abs_score_change)]);checked+=2
    assert bool(drift.top_tied_set_equal)==np.array_equal(x==x.max(),ref==ref.max())
assert len(sets)==7 and all(s==sets[0] for s in sets) and max(errors)<1e-12
out={'status':'passed','source_score_settings':26,'backgrounds':7,'panels':182,'aggregate_rows_per_table':546,'independent_numeric_comparisons':checked,'max_abs_error':max(errors),'frozen_file_hashes_checked':h,'same_exact_method_set':True,'truth_loaded':False,'historical_remaining_scope':'source-explicit9-person subset; not an independently verified posterior','historical_posterior_recomputed':False,'audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(R/'inventory/recovered_rare_secondary_independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out))

