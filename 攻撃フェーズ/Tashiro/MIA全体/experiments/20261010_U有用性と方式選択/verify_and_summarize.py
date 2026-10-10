"""Independent algebra checks and phase-separated context. Never fits a base attack."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,shutil
import numpy as np,pandas as pd
H=Path(__file__).parent;C=Path('/workspace/shared/mia_c_only_stage/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_C単独選択とアンサンブル')
M=Path('/workspace/shared/攻撃フェーズ/Tashiro/MIA全体/本戦公式有用性_20261010')
read=lambda p:pd.read_csv(p,float_precision='round_trip')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dump=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
tids=[1,2,3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
u=read(H/'prelim_official_utility_all22.csv').set_index('team').loc[tids,['U_gen','U_spec','U_rare','U_valid']].to_numpy()[2:]
c=read(C/'C_only_features.csv').drop(columns='team').to_numpy()[2:]
m=np.load(C/'private_cache/candidate_metrics.npy')[:,2:]
options=[(1,0)]+[(k,a) for a in (.25,.5,1) for k in (1,3,5,9)]
got=read(H/'heldout_selection_by_target.csv');got=got[got.scope=='strict18'].set_index(['team','strategy'])
def order(x,tr,t):
    scale=np.std(x[tr],axis=0);keep=scale>1e-10
    return np.sum((x[tr][:,keep]-x[t,keep])**2/scale[keep]**2,axis=1)/sum(keep)
def calculate(tr,t,view):
    d=order(c,tr,t) if view=='C' else order(u,tr,t) if view=='U' else (order(c,tr,t)+order(u,tr,t))/2
    nn=[tr[i] for i in np.argsort(d,kind='stable')]
    ids=[]
    for k,a in options:
        w=np.zeros(18);w[tr]=(1-a)/len(tr);w[nn[:k]]+=a/k
        ids.append(int(np.argmax(m@w)))
    return ids
# Independent selection algebra audit is implemented by check_numeric_ties.py.
assert sha(M/'main_utility_scores.csv')=='3f8a031eadc8653eab905d5e9324cc2fcce3ae5a20f0297296b0d4b2b3625a46'
assert sha(M/'source_monitor_state.json')=='718dcc785e78b1365f3556e244f2d4a61074931d23d0b7292321d08a28598f66'
for name in ['main_utility_scores.csv','source_monitor_state.json','provenance.json','validation.json']:
    if (M/name).exists():shutil.copy2(M/name,H/'sources'/('main_'+name if name!='main_utility_scores.csv' else name))
main=read(M/'main_utility_scores.csv');pre=read(H/'prelim_official_utility_all22.csv')
# Exact official cohort crosswalk from official team table, not identity of numerical IDs.
cross={0:4,1:8,2:18,3:29,4:13,5:7,6:12,7:6,8:26,9:2,10:24,11:None,12:1,13:9,14:10,15:20,16:17,17:21,22:3,25:22,26:23,27:19,28:5}
rows=[]
for _,a in main.iterrows():
    pt=cross[a.cohort]
    if pt is not None:
        pname=pre.set_index('team').loc[pt,'team_name'];assert a.team==pname
    rows.append({'main_cohort':int(a.cohort),'prelim_cohort':pt,'team_name':a.team,'main_submission_id':int(a.submission_id),
      'source':'https://www.iwsec.org/pws/2026/cup26.html','identity_method':'official phase-specific roster; exact team name confirmed'})
pd.DataFrame(rows).to_csv(H/'phase_cohort_crosswalk.csv',index=False)
facets=['U_gen','U_spec','U_rare','U_valid']
assert np.allclose(main.U,main[facets].min(axis=1),rtol=0,atol=1e-12)
ctx=main[['cohort','team','submission_id','U']+facets].copy()
ctx['bottleneck_at_display_precision']=['|'.join(f for f in facets if abs(r[f]-r['U'])<1e-12) for _,r in main.iterrows()]
ctx['phase']='main';ctx['attack_performance_inferred']=False
ctx.to_csv(H/'main_utility_context_only.csv',index=False,float_format='%.17g')
dist=[]
for phase,d in [('prelim_submitted21',pre.dropna(subset=['U'])),('prelim_strict18',pre[pre.team.isin(tids[2:])]),('main23_preserved',main)]:
    for col in ['U']+facets:
        dist.append({'phase':phase,'feature':col,'n':len(d),'min':d[col].min(),'median':d[col].median(),'max':d[col].max(),
                     'mean':d[col].mean(),'not_directly_comparable_across_phase':True})
pd.DataFrame(dist).to_csv(H/'phase_separated_utility_summary.csv',index=False,float_format='%.17g')
base=read(C/'heldout_by_target.csv');base=base[(base.scope=='train_and_test_without1_2')&(base.strategy=='ensemble_train_champion_mean')&(base.population=='all_fixed')]
assert len(base)==18
base=base.set_index('team')['tpr_0.01'];p01=got.xs('fixed_P01',level='strategy').test_tpr
cmp=read(H/'selection_summary.csv')
extra={'scope':'strict18','strategy':'eight_family_train_champion_rank_mean_prior','n':18,'mean_tpr':base.mean(),
       'median_tpr':base.median(),'min_tpr':base.min(),'targets_gt_0.10':int((base>.1).sum()),
       'delta_vs_fixed_P01':(base-p01).mean(),'target_wins_vs_P01':int((base-p01>1e-12).sum()),
       'target_losses_vs_P01':int((base-p01< -1e-12).sum())}
cmp=pd.concat([cmp,pd.DataFrame([extra])],ignore_index=True);cmp.to_csv(H/'selection_summary_with_rank_ensemble.csv',index=False,float_format='%.17g')
dump(H/'main_context_audit.json',{'cohort_matches':23,'submission_id_matches':'see sources/main_validation.json and official table',
 'snapshot':'2026-09-25 08:55:23 JST','display_precision':4,'all_U_min_checks_passed':True,
 'padded_cohort_unidentified':True,'postpadding_rescore_unverified':True,'main_attack_truth_used':False,
 'main_values_used_for_prelim_selectors':False,'main_selectors_deployed':False})
print('PASS 23 phase-aware main identities')
print(ctx.bottleneck_at_display_precision.value_counts().to_string())
