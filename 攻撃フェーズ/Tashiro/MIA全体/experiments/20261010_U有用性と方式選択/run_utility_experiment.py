"""Utility-guided target-held-out selection. Predeclared protocol.json is authoritative.
All features come from C or the published defense utility ledger.
No attacker scores, Anon, rank, protected B, or target IDs enter feature construction.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import sys, json, hashlib, datetime
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import openpyxl
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent
CBASE=Path('/workspace/shared/mia_c_only_stage/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_C単独選択とアンサンブル')
WBASE=Path('/workspace/shared/mia_recovered_validation/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件')
WB=WBASE/'private_inputs/公開用結果表_20260909.xlsx'
MANIFEST=Path('/workspace/shared/mia_bg7_inputs/manifest.json')
TEAMS=[1,2,3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
UNAMES=['U_gen','U_spec','U_rare','U_valid']
OPTIONS=[(1,0.)]+[(k,a) for a in (.25,.5,1.) for k in (1,3,5,9)]
P01=14109
read=lambda p:pd.read_csv(p,float_precision='round_trip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def inputs():
    prov=json.loads((WBASE/'official_workbook_provenance.json').read_text())
    assert sha(WB)==prov['sha256']
    w=openpyxl.load_workbook(WB,read_only=True,data_only=True)
    rows=list(w['総括'].values);df=pd.DataFrame(rows[1:],columns=rows[0])
    utility=df[['コホート番号','チーム名','U']+UNAMES].rename(columns={'コホート番号':'team','チーム名':'team_name'})
    meta=dict(list(w['メタ'].values)[1:]);utility=utility[utility.team.notna()].copy();utility.team=utility.team.astype(int)
    # Explicit whitelist above excludes all Anon/attack/rank columns.
    assert utility.team.is_unique and len(utility)==22
    valid=utility[utility.team.isin(TEAMS)]
    assert len(valid)==20 and np.isfinite(valid[['U']+UNAMES].to_numpy(float)).all()
    assert np.allclose(valid.U,valid[UNAMES].min(axis=1),atol=1e-10,rtol=0)
    mfest=json.loads(MANIFEST.read_text());files={r['repo_path']:r for r in mfest['files']}
    audits=[]
    for t in TEAMS:
        key=f'公式データ/予備戦/PWSCup2026_prelim_attack/C_{t}.csv'
        f=files[key];h=sha(f['path']);assert h==f['sha256']==meta[f'sha256.defense.C_{t}.csv']
        audits.append({'phase':'prelim','team':t,'C_sha256':h,'official_submission_id':str(meta[f'ledger.公式C_提出ID.{t}']),'source_C':f['source_url']})
    utility['phase']='prelim'
    utility['source_url']='https://drive.google.com/file/d/1BLiwflR7FJ9tp5lCeSkP-kJga4ByHRnB/view'
    utility['value_semantics']='published 0..1 higher-is-better facets; U=min(facets)'
    utility.to_csv(HERE/'prelim_official_utility_all22.csv',index=False,float_format='%.17g')
    pd.DataFrame(audits).to_csv(HERE/'utility_C_binding_audit.csv',index=False)
    u=utility.set_index('team').loc[TEAMS,UNAMES].to_numpy(float)
    cdf=read(CBASE/'C_only_features.csv');assert cdf.team.tolist()==TEAMS
    c=cdf.drop(columns='team').to_numpy(float)
    bank=np.load(CBASE/'private_cache/candidate_metrics.npy')
    catalog=read(CBASE/'candidate_catalog.csv')
    assert bank.shape==(43148,20) and catalog.candidate_index.tolist()==list(range(len(bank)))
    assert catalog.iloc[P01]['family']=='geometry' and catalog.iloc[P01]['background']=='prelim'
    assert catalog.iloc[P01]['method']=='gen7_reference__county500__cross0.1'
    p01=read(CBASE/'fixed_P01_verified_target_tpr.csv')
    dump(HERE/'input_audit.json',{'workbook':prov,'U_ledger_note':meta['U'],'utility_feature_columns':UNAMES,'C_feature_count':c.shape[1],
       'bank_sha256':sha(CBASE/'private_cache/candidate_metrics.npy'),'catalog_sha256':sha(CBASE/'candidate_catalog.csv'),
       'C_features_sha256':sha(CBASE/'C_only_features.csv'),'phase_join':'same-phase cohort + exact C SHA256 + official submission ID',
       'all20_team_order':TEAMS,'four_facets_all_in_0_1':bool(((u>=0)&(u<=1)).all()),
       'per_facet_nunique_primary18':dict(zip(UNAMES,[int(len(np.unique(u[2:,i]))) for i in range(4)])),
       'utility_inputs_have_no_attack_metrics':True,'main_utility_values_used_in_prelim_selection':False})
    return bank,catalog,c,u,utility

def distances(x,train,test):
    tr=x[train];sd=tr.std(axis=0);keep=sd>1e-10
    if not keep.any():return np.zeros(len(train))
    return np.mean(((tr[:,keep]-x[test,keep])/sd[keep])**2,axis=1)

def choose(m,c,u,train,test,view):
    if view=='C':d=distances(c,train,test)
    elif view=='U':d=distances(u,train,test)
    elif view=='CU':d=.5*distances(c,train,test)+.5*distances(u,train,test)
    else:raise ValueError(view)
    nn=np.array(train)[np.argsort(d,kind='stable')]
    avg=m[:,train].mean(axis=1);local=np.cumsum(m[:,nn[:9]],axis=1)
    ids=[]
    for k,a in OPTIONS:
        k=min(k,len(nn));pred=avg if a==0 else (1-a)*avg+a*local[:,k-1]/k
        ids.append(int(np.argmax(pred)))
    return ids,nn

def fold(m,c,u,active,test,view):
    tr=[j for j in active if j!=test];ids,nn=choose(m,c,u,tr,test,view)
    inner=[]
    for v in tr:
        it=[j for j in tr if j!=v];v_ids,_=choose(m,c,u,it,v,view)
        inner.append(m[v_ids,v])
    obj=np.mean(inner,axis=0);best=int(np.argmax(obj))
    fixed=OPTIONS.index((3,.5))
    p01_obj=m[P01,tr].mean()
    guard=P01 if p01_obj>=obj[best] else ids[best]
    return {'global':ids[0],'fixed':ids[fixed],'nested':ids[best],'p01_guard_nested':guard},obj,best,nn,p01_obj

def run():
    m,cat,c,u,util=inputs();rows=[];inner_rows=[];leak=[];families=[];summaries=[]
    for scope,active in [('strict18',list(range(2,20))),('all20',list(range(20)))]:
        for t in active:
            tr=[j for j in active if j!=t]
            for fam in cat.family.unique():
                ix=cat.index[cat.family==fam].to_numpy()
                champ=int(ix[np.argmax(m[ix][:,tr].mean(axis=1))])
                families.append({'scope':scope,'team':TEAMS[t],'family':fam,'representative_index':champ,'heldout_tpr':m[champ,t],
                     'grid_mean_tpr':m[ix,t].mean(),'selection_train_ids':','.join(str(TEAMS[j]) for j in tr)})
            rows.append({'scope':scope,'team':TEAMS[t],'strategy':'fixed_P01','candidate_index':P01,'test_tpr':m[P01,t]})
            for view in ['C','U','CU']:
                selections,obj,best,nn,p01_obj=fold(m,c,u,active,t,view)
                for policy,idx in selections.items():
                    if policy=='global' and view!='C':continue
                    rows.append({'scope':scope,'team':TEAMS[t],'strategy':'global_train_best' if policy=='global' else view+'_'+policy,
                        'candidate_index':idx,'test_tpr':m[idx,t],'family':cat.iloc[idx]['family'],'background':cat.iloc[idx]['background'],
                        'method':cat.iloc[idx]['method'],'selected_k':OPTIONS[best][0] if 'nested' in policy else 3,
                        'selected_local_weight':OPTIONS[best][1] if 'nested' in policy else .5,
                        'nearest3_train_ids':','.join(str(TEAMS[j]) for j in nn[:3]),'inner_best_objective':float(obj[best]),
                        'inner_P01_objective':float(p01_obj)})
                for j,value in enumerate(obj):
                    inner_rows.append({'scope':scope,'team':TEAMS[t],'view':view,'k':OPTIONS[j][0],'local_weight':OPTIONS[j][1],
                        'inner_loto_tpr':value,'selected':j==best})
                # Perturb heldout target labels/scores only. Rerun entire nested procedure.
                altered=m.copy();altered[:,t]=np.linspace(10,20,len(m))
                altered_sel,altered_obj,altered_best,_,_=fold(altered,c,u,active,t,view)
                assert selections==altered_sel and best==altered_best and np.array_equal(obj,altered_obj)
                leak.append({'scope':scope,'team':TEAMS[t],'view':view,'heldout_perturbation_selection_invariant':True,
                             'heldout_perturbation_inner_objective_invariant':True})
            print('COMPLETE',scope,TEAMS[t],flush=True)
    df=pd.DataFrame(rows);df.to_csv(HERE/'heldout_selection_by_target.csv',index=False,float_format='%.17g')
    pd.DataFrame(inner_rows).to_csv(HERE/'nested_inner_validation.csv',index=False,float_format='%.17g')
    pd.DataFrame(families).to_csv(HERE/'family_representative_heldout.csv',index=False,float_format='%.17g')
    dump(HERE/'heldout_exclusion_audit.json',leak)
    # Match existing C-only experiment exactly before interpreting U.
    old=read(CBASE/'selected_methods_by_target.csv');checks=[]
    mapping={'strict18':'train_and_test_without1_2','all20':'all20'}
    for scope,os in mapping.items():
        for ours,theirs in [('C_fixed','C_gate_fixed_k3_weight05'),('C_nested','C_gate_nested'),('global_train_best','global_train_best')]:
            a=df[(df.scope==scope)&(df.strategy==ours)].sort_values('team')
            b=old[(old.scope==os)&(old.strategy==theirs)].sort_values('team')
            assert a.team.tolist()==b.team.tolist() and np.array_equal(a.candidate_index,b.candidate_index)
            assert np.allclose(a.test_tpr,b.test_tpr,atol=1e-14,rtol=0)
            checks.append({'scope':scope,'strategy':ours,'matches_prior_selection_and_tpr':True})
    dump(HERE/'C_baseline_reproduction_audit.json',checks)
    oldheld=read(CBASE/'heldout_by_target.csv')
    for (scope,strategy),part in df.groupby(['scope','strategy'],sort=False):
        base=df[(df.scope==scope)&(df.strategy=='fixed_P01')].set_index('team').test_tpr
        v=part.set_index('team').test_tpr;delta=v-base
        summaries.append({'scope':scope,'strategy':strategy,'n':len(v),'mean_tpr':v.mean(),'median_tpr':v.median(),
          'min_tpr':v.min(),'targets_gt_0.10':int((v>.10).sum()),'delta_vs_fixed_P01':delta.mean(),
          'target_wins_vs_P01':int((delta>1e-12).sum()),'target_losses_vs_P01':int((delta< -1e-12).sum())})
    summary=pd.DataFrame(summaries);summary.to_csv(HERE/'selection_summary.csv',index=False,float_format='%.17g')
    for scope in ['strict18','all20']:
        pivot=df[df.scope==scope].pivot(index='team',columns='strategy',values='test_tpr')
        pivot.to_csv(HERE/f'{scope}_target_comparison.csv',float_format='%.17g')
    fam=pd.DataFrame(families);corr=[]
    # Primary exploratory panel, 9 outcomes x5 utility measures =45 comparisons.
    for feat in ['U']+UNAMES:
        ux=util.set_index('team').loc[TEAMS[2:],feat].to_numpy(float)
        outcomes={'fixed_P01':m[P01,2:]}
        for f in cat.family.unique():outcomes[f]=fam[(fam.scope=='strict18')&(fam.family==f)].set_index('team').loc[TEAMS[2:],'heldout_tpr'].to_numpy()
        for name,y in outcomes.items():
            if len(np.unique(ux))<2 or len(np.unique(y))<2:r=p=np.nan
            else:r,p=spearmanr(ux,y)
            corr.append({'utility':feat,'outcome':name,'n':18,'spearman_r':r,'p_uncorrected':p,'interpretation':'descriptive; candidate grid historically posthoc'})
    cd=pd.DataFrame(corr);ok=cd.p_uncorrected.notna();ps=cd.loc[ok,'p_uncorrected'].to_numpy();order=np.argsort(ps)
    q=np.empty(len(ps));q[order]=np.minimum.accumulate((ps[order]*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1]
    cd.loc[ok,'q_BH_exploratory']=np.clip(q,0,1)
    cd.to_csv(HERE/'utility_family_spearman_primary18.csv',index=False,float_format='%.17g')
    print(summary[summary.scope=='strict18'].to_string(index=False),flush=True)
    print('Top correlations',cd.reindex(cd.spearman_r.abs().sort_values(ascending=False).index).head(8).to_string(index=False),flush=True)
    dump(HERE/'completion.json',{'complete':True,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'primary_target_count':18,'secondary_target_count':20,'candidate_count':len(m),'heldout_perturbation_checks':len(leak),
         'all_C_baseline_reproductions_passed':True,'main_truth_read':False,'new_base_models_fit':0})
if __name__=='__main__':run()
