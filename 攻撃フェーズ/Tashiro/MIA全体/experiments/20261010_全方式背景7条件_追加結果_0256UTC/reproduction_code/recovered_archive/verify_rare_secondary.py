from run_archive import *
from scipy.stats import rankdata
root=setup('prelim');audit=module('_rare_verify_inputs',root/'wrappers/official_audit.py');b,p,r,n,dist=audit.load_inputs(root)
rr=b.loc[r].sort_values('record_id');ids=rr.record_id.astype(int).tolist();priv=set(b.loc[p&r,'record_id']);eligible=[j for j,rid in enumerate(ids) if rid not in priv];remaining=[j for j in eligible if ids[j] not in [201,238]]
scopes={'all24':list(range(24)),'eligible11':eligible,'remaining9_historical':remaining};table=pd.read_csv(BASE/'rare_secondary_summary.csv',float_precision='round_trip').set_index(['background','method','scope']);change=pd.read_csv(BASE/'rare_secondary_background_changes.csv',float_precision='round_trip').set_index(['background','method','scope'])
base=np.load(BASE/'rare_secondary/prelim/frozen.npz');rows=0;maxerr=0.;errors=[];cross=[]
for bg in BGS:
 folder=BASE/'rare_secondary'/bg;meta=json.loads((folder/'completion.json').read_text());assert sha(folder/'frozen.npz')==meta['frozen_sha256'];data=np.load(folder/'frozen.npz');assert len(data.files)==26
 for name in data.files:
  for scope,ix in scopes.items():
   s=data[name][ix];orig=base[name][ix];exp=dict(n_candidates=len(s),score_min=s.min(),score_max=s.max(),score_mean=s.mean(),n_top_ties=int((s==s.max()).sum()),score_std=s.std());saved=table.loc[bg,name,scope]
   for key,value in exp.items():maxerr=max(maxerr,abs(saved[key]-value));rows+=1
   oldrank=rankdata(orig);newrank=rankdata(s);sp=np.corrcoef(oldrank,newrank)[0,1];record=change.loc[bg,name,scope]
   if np.isfinite(sp):maxerr=max(maxerr,abs(record.spearman_prelim-sp));rows+=1
   assert bool(record.top_tied_set_equal)==np.array_equal(orig==orig.max(),s==s.max())
   maxerr=max(maxerr,abs(record.max_abs_score_change-np.max(np.abs(s-orig))));rows+=1
 # Six shared score definitions independently computed through raw whole-MIA source21.
 full=BASE/'results'/bg/'raw_gen'
 if (full/'completion.json').exists():
  raw=np.load(full/'frozen.npz');j=TEAMS.index(20);ix=b.set_index('record_id').reset_index().index # full candidates retain original row order
  positions=[int(np.flatnonzero(b.record_id.to_numpy()==rid)[0]) for rid in ids]
  for a,z in [('qi7_h1_k10','legacy_raw_qi7_gen'),('attrs10_h1_k10','legacy_raw_attrs10_gen'),('qi7_DCR','legacy_raw_qi7_distance'),('attrs10_DCR','legacy_raw_attrs10_distance'),('qi7_point_h1','legacy_raw_qi7_point'),('attrs10_point_h1','legacy_raw_attrs10_point')]:
   err=float(np.max(np.abs(data[a]-raw[z][positions,j])));cross.append({'background':bg,'rare_method':a,'whole_mia_method':z,'max_abs_error':err})
   if err>1e-12:errors.append((bg,a,err))
assert maxerr<1e-12,maxerr
res={'status':'PASS' if not errors else 'FAIL','aggregate_numeric_values_checked':rows,'max_abs_aggregate_error':float(maxerr),'cross_source_scores_checked':len(cross)*24,'cross_source_max_abs_error':max([v['max_abs_error'] for v in cross],default=0),'errors':errors,'cross_source_details':cross,'membership_truth_read':False}
(BASE/'rare_secondary_qa.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='cross_source_details'}))
