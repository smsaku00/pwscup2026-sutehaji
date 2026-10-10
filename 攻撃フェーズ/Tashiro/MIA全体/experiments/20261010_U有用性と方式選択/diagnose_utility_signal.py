"""Post-result diagnostics, not confirmatory and not used to tune any selector."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,time
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
import run_utility_experiment as r
start=time.time();m,cat,c,u,util=r.inputs();m=m[:,2:];c=c[2:];u=u[2:];N=18
trains=[[j for j in range(N) if j!=i] for i in range(N)]
avgs=[m[:,tr].mean(axis=1) for tr in trains]
def calc(ux,view):
    result=[]
    for t,tr in enumerate(trains):
        d=r.distances(ux,tr,t)
        if view=='CU':d=.5*d+.5*r.distances(c,tr,t)
        nn=np.array(tr)[np.argsort(d,kind='stable')[:3]]
        idx=np.argmax(.5*avgs[t]+.5*m[:,nn].mean(axis=1));result.append(m[idx,t])
    return np.array(result)
rng=np.random.default_rng(20261010);observed={v:calc(u,v) for v in ['U','CU']};rows=[]
for k in range(199):
    perm=rng.permutation(N)
    for view in ['U','CU']:
        vals=calc(u[perm],view)
        rows.append({'permutation':k,'view':view,'mean_tpr':vals.mean(),'mapping':','.join(str(r.TEAMS[2:][j]) for j in perm)})
df=pd.DataFrame(rows);df.to_csv(r.HERE/'utility_permutation_diagnostic.csv',index=False,float_format='%.17g')
res=[]
for view in ['U','CU']:
    z=df[df.view==view].mean_tpr;v=observed[view].mean()
    res.append({'view':view,'observed_mean_tpr':v,'shuffle_mean_tpr':z.mean(),'shuffle_median_tpr':z.median(),
       'shuffle_p025':z.quantile(.025),'shuffle_p975':z.quantile(.975),
       'shuffles_ge_observed':int((z>=v-1e-12).sum()),
       'exploratory_upper_tail_fraction':float((1+(z>=v-1e-12).sum())/200)})
r.dump(r.HERE/'utility_permutation_diagnostic.json',{'after_primary_results':True,'permutations':199,
   'shuffle':'permute only the 4U block jointly across strict18 target identities; C stays fixed in CU',
   'selector':'fixed k=3 local weight=.5; no nested retuning or policy change',
   'result':res,'elapsed_seconds':time.time()-start,
   'limitations':'not confirmatory p-values; targets nonexchangeable and bank historically adapted to prelim labels'})
p=r.read(r.HERE/'strict18_target_comparison.csv').set_index('team')
boot_rng=np.random.default_rng(20261011);boots=boot_rng.integers(0,N,size=(20000,N));pairs=[]
for a,b in [('U_fixed','C_fixed'),('U_nested','C_nested'),('U_nested','fixed_P01'),('CU_fixed','C_fixed'),('CU_nested','C_nested')]:
    d=(p[a]-p[b]).to_numpy();bm=d[boots].mean(axis=1)
    pairs.append({'strategy':a,'comparator':b,'mean_delta':d.mean(),'bootstrap95_low':float(np.quantile(bm,.025)),
     'bootstrap95_high':float(np.quantile(bm,.975)),'wins':int((d>1e-12).sum()),'losses':int((d< -1e-12).sum()),
     'ties':int((np.abs(d)<=1e-12).sum()),'interpretation':'descriptive paired target bootstrap; does not remove grid-development bias'})
pd.DataFrame(pairs).to_csv(r.HERE/'paired_target_diagnostics.csv',index=False,float_format='%.17g')
print(json.dumps(res,indent=2));print(pd.DataFrame(pairs).to_string(index=False))
