"""Test grouped ROC evaluator against official scorer on synthetic ties."""
from pathlib import Path
import sys,json
import numpy as np
from sklearn.metrics import roc_auc_score
from run_grid import metrics,HERE
sys.path.insert(0,'/workspace/shared/mia_bg7_inputs/official_scoring/src')
from pwscup2026.scoring.attack_score import _tpr_at_fpr
rng=np.random.default_rng(20261010);checks=[]
for size in (875,1025,1049):
 for members in (1,7,25,100):
  for kind in ('continuous','integer_ties','all_tied'):
   y=np.r_[np.ones(members,int),np.zeros(size-members,int)];rng.shuffle(y)
   s=rng.random(size) if kind=='continuous' else rng.integers(0,8,size).astype(float) if kind=='integer_ties' else np.zeros(size)
   v=metrics(s,y);official=[_tpr_at_fpr(s,y,a)[0] for a in (.005,.01,.02,.05)];err=max(abs(np.array(v[:4])-official));ae=abs(v[4]-roc_auc_score(y,s));assert err<1e-12 and ae<1e-12
   checks.append({'size':size,'members':members,'kind':kind,'tpr_max_error':float(err),'auc_error':float(ae)})
report={'passed':True,'synthetic_cases':len(checks),'official_tpr_max_error':max(x['tpr_max_error'] for x in checks),'auc_max_error':max(x['auc_error'] for x in checks),'checks':checks}
(HERE/'scoring_validation.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='checks'})
