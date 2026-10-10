from run_archive import *
import warnings
from lifelines import CoxPHFitter
p=ORIG/'アーカイブ/予備戦_攻撃フェーズ/work_20260901以前/defense.py'
d=extract(p,['_design','fit_cox_raw'],dict(np=np,pd=pd,warnings=warnings,CoxPHFitter=CoxPHFitter,COX_COLS=['age','sex','BMI','SBP','log_TG','HDL','log_ALT','FPG','smoking']))
root=setup('prelim');a=module('_cox_inputs',root/'wrappers/official_audit.py');b,_,_,_,dist=a.load_inputs(root)
frames=[('C',str(k),pd.read_csv(dist/f'C_{k}.csv',float_precision='round_trip')) for k in TEAMS]
frames += [('background',bg,pd.read_csv(INPUT/'bg/subsets'/f'{bg}.csv',float_precision='round_trip')) for bg in BGS]
rows=[]
with threadpool_limits(limits=1):
 for kind,name,frame in frames:
  for cause in ['onset','death']:
   r={'input_type':kind,'input':name,'cause':cause,'rows':len(frame)}
   try:
    fit=d.fit_cox_raw(frame,cause);beta=fit.params_.to_numpy(float);r.update(status='fit_succeeded',finite=bool(np.isfinite(beta).all()),max_abs_beta=float(np.max(np.abs(beta))))
   except Exception as exc:r.update(status='original_zero_beta_fallback',exception_type=type(exc).__name__,exception_message=str(exc))
   rows.append(r)
res={'fits':len(rows),'fit_succeeded':sum(r['status']=='fit_succeeded' for r in rows),'source_fallbacks':sum(r['status']!='fit_succeeded' for r in rows),'rows':rows,'membership_labels_read':False,'source_sha256':sha(p)}
(BASE/'cox_fit_qa.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='rows'}))
