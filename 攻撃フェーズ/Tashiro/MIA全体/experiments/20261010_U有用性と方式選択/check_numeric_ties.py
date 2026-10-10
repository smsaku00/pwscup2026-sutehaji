"""Post-result numerical stability check: deterministic 1e-12 tie rule, no feature tuning."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import run_utility_experiment as r
m,cat,c,u,_=r.inputs();m=m[:,2:];c=c[2:];u=u[2:];teams=r.TEAMS[2:]
TOL=1e-12
def firstmax(p):return int(np.flatnonzero(p>=np.max(p)-TOL)[0])
def choose(tr,t,view,independent=False):
    d=r.distances(c,tr,t) if view=='C' else r.distances(u,tr,t) if view=='U' else .5*r.distances(c,tr,t)+.5*r.distances(u,tr,t)
    nn=np.array(tr)[np.argsort(d,kind='stable')]
    avg=m[:,tr].mean(axis=1);cum=np.cumsum(m[:,nn[:9]],axis=1);ids=[]
    for k,a in r.OPTIONS:
        if independent:
            weights=np.zeros(18);weights[tr]=(1-a)/len(tr);weights[nn[:k]]+=a/k
            p=m@weights
        else:p=avg if a==0 else (1-a)*avg+a*cum[:,k-1]/k
        ids.append(firstmax(p))
    return ids
rows=[];audit=[]
for t in range(18):
    tr=[i for i in range(18) if i!=t]
    for view in ['C','U','CU']:
        ids=choose(tr,t,view);indep=choose(tr,t,view,True);assert ids==indep
        inner=[]
        for v in tr:
            it=[i for i in tr if i!=v];i=choose(it,v,view);ii=choose(it,v,view,True);assert i==ii
            inner.append(m[i,v])
        obj=np.mean(inner,axis=0);best=firstmax(obj)
        policies={'fixed':ids[r.OPTIONS.index((3,.5))],'nested':ids[best],
                  'p01_guard_nested':r.P01 if m[r.P01,tr].mean()>=obj[best]-TOL else ids[best]}
        for policy,i in policies.items():rows.append({'team':teams[t],'strategy':view+'_'+policy,'candidate_index':i,'test_tpr':m[i,t]})
        audit.append({'team':teams[t],'view':view,'all_outer_and_inner_argmax_match_independent_weighted_dot':True})
out=pd.DataFrame(rows);out.to_csv(r.HERE/'stable_tie_by_target.csv',index=False,float_format='%.17g')
old=r.read(r.HERE/'heldout_selection_by_target.csv');old=old[old.scope=='strict18'].set_index(['team','strategy'])
summ=[]
for name,g in out.groupby('strategy'):
    prev=old.xs(name,level='strategy').loc[g.team];new=g.set_index('team')
    summ.append({'strategy':name,'stable_tie_mean_tpr':g.test_tpr.mean(),'original_mean_tpr':prev.test_tpr.mean(),
      'delta_due_to_tie_rule':g.test_tpr.mean()-prev.test_tpr.mean(),'changed_candidate_targets':int((new.candidate_index!=prev.candidate_index).sum()),
      'changed_tpr_targets':int((np.abs(new.test_tpr-prev.test_tpr)>1e-14).sum()),'targets_gt_0.10':int((g.test_tpr>.1).sum())})
sd=pd.DataFrame(summ);sd.to_csv(r.HERE/'stable_tie_summary.csv',index=False,float_format='%.17g')
r.dump(r.HERE/'stable_tie_algebra_audit.json',{'after_primary_result':True,'absolute_tie_tolerance':TOL,
 'tie_rule':'first candidate or option within 1e-12 of maximum; P01 wins guarded-policy ties',
 'passed':True,'outer_views':len(audit),'outer_plus_inner_policy_selections_checked':len(audit)*18*13,
 'checks':audit})
print(sd.to_string(index=False))
