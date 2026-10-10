from run_archive import *
expected={'historical_cache.json':'8a94631cb670a9a4d7fe1c17c9cb5b11c6274760','historical_atk_cache.json':'e2e033a5fa4d2e0f9cefe60c348285ad46a6e3c3','historical_virtual_cache.json':'6ea22519213ccf21380734002a887887d70dae41'}
records=[];missing=[];sources={}
for file,digest in expected.items():
 p=WRAPPERS/file
 if not p.exists():missing.append(file);continue
 data=p.read_bytes();gitsha=hashlib.sha1(b'blob '+str(len(data)).encode()+bytes([0])+data).hexdigest();assert gitsha==digest,(file,gitsha)
 recs=json.loads(data)['records']
 for r in recs:r['historical_cache']=file
 records+=recs;sources[file]={'git_blob':gitsha,'sha256':sha(p),'records':len(recs)}
old=pd.DataFrame(records)
cur=pd.read_csv(BASE/'by_target.csv',float_precision='round_trip');cur=cur[cur.background=='prelim']
m=cur.merge(old,on=['method','team'],suffixes=('_current','_historical'),how='inner',validate='one_to_one')
m['tpr_error']=m.tpr_current-m.tpr_historical
m['auc_error']=m.auc_current-m.auc_historical
m['tpr_match']=np.abs(m.tpr_error)<1e-12;m['auc_match']=np.abs(m.auc_error)<1e-12
cols=['method','team','historical_cache','tpr_historical','tpr_current','tpr_error','tpr_match','auc_historical','auc_current','auc_error','auc_match']
m[cols].to_csv(BASE/'baseline_target_validation.csv',index=False,float_format='%.17g')
summary=m.groupby('method').agg(targets=('team','size'),tpr_mismatches=('tpr_match',lambda x:int((~x).sum())),auc_mismatches=('auc_match',lambda x:int((~x).sum())),max_abs_tpr_error=('tpr_error',lambda x:float(x.abs().max())),max_abs_auc_error=('auc_error',lambda x:float(x.abs().max()))).reset_index()
summary.to_csv(BASE/'baseline_method_validation.csv',index=False,float_format='%.17g')
status={'historical_files_missing':missing,'sources':sources,'methods_checked':len(summary),'target_rows_checked':len(m),'methods_tpr_exact':int((summary.tpr_mismatches==0).sum()),'methods_tpr_and_auc_exact':int(((summary.tpr_mismatches==0)&(summary.auc_mismatches==0)).sum()),'tpr_mismatched_rows':int((~m.tpr_match).sum()),'auc_mismatched_rows':int((~m.auc_match).sum()),'max_abs_tpr_error':float(m.tpr_error.abs().max()),'max_abs_auc_error':float(m.auc_error.abs().max()),'note':'Same-source current environment rerun; exact historical exceptions retained, not tuned away.'}
(BASE/'baseline_target_validation.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status));print(summary[(summary.tpr_mismatches>0)|(summary.auc_mismatches>0)].to_string(index=False))
