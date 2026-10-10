"""Post-result diagnostic: shuffle only C-to-target correspondence, never tune a gate.
This is exploratory; target exchangeability and an untouched candidate grid are not guaranteed.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,time
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent));import run_experiment as r
start=time.time();x=pd.read_csv(r.HERE/'C_only_features.csv',float_precision='round_trip').drop(columns='team').to_numpy()[2:];m=np.load(r.HERE/'private_cache/candidate_metrics.npy')[:,2:]
trains=[[j for j in range(18) if j!=t] for t in range(18)];global_means=[m[:,tr].mean(axis=1) for tr in trains]
def calc(features):
    out=[]
    for t,tr in enumerate(trains):
        nn=r.neighbors(features,tr,t)[:3];pred=.5*global_means[t]+.5*m[:,nn].mean(axis=1);i=int(np.argmax(pred));out.append(m[i,t])
    return np.array(out)
actual=calc(x);rng=np.random.default_rng(20261010);rows=[]
for n in range(199):
    perm=rng.permutation(18);v=calc(x[perm]);rows.append({'permutation':n,'mean_tpr':v.mean(),'TPR_gt010':int((v>.1+1e-12).sum()),'mapping':','.join(str(r.TEAMS[2:][j]) for j in perm)})
d=pd.DataFrame(rows);d.to_csv(r.HERE/'C_feature_permutation_control.csv',index=False,float_format='%.17g')
result={'added_after_primary_results':True,'scope':'18 targets, excluding1_2 in both training and evaluation','strategy':'fixed k3 localweight.5; no nested retuning','shuffled':'C feature vectors assigned to target identities','permutations':199,'seed':20261010,'observed_mean_tpr':actual.mean(),'permutation_mean_tpr_mean':d.mean_tpr.mean(),'permutation_mean_tpr_median':d.mean_tpr.median(),'permutation_p025':d.mean_tpr.quantile(.025),'permutation_p975':d.mean_tpr.quantile(.975),'permutations_ge_observed':int((d.mean_tpr>=actual.mean()-1e-12).sum()),'exploratory_one_sided_tail_fraction':float((1+(d.mean_tpr>=actual.mean()-1e-12).sum())/200),'not_confirmatory_p_value':True,'limitation':'Candidate bank historical adaptation and target nonexchangeability remain; this diagnostic cannot establish null or alternative scientifically','elapsed_seconds':time.time()-start}
r.dump(r.HERE/'C_feature_permutation_control.json',result);print(result)
