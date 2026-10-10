"""Aggregate the completed BG7 grids without changing or selecting predictions."""
from pathlib import Path
import argparse, hashlib, json, re
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent

def dump(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def summarize():
    d=pd.read_csv(HERE/'prelim/by_target.csv',float_precision='round_trip')
    s=pd.read_csv(HERE/'prelim/summary.csv',float_precision='round_trip')
    keys=['family','inference','method','population','team']
    old=d[d.background=='prelim'][keys+['tpr_0.01']].rename(columns={'tpr_0.01':'original_prelimBG_tpr'})
    v=d.merge(old,on=keys,validate='many_to_one')
    v['delta_prelimBG']=v['tpr_0.01']-v.original_prelimBG_tpr
    v['new_ge010']=(v['tpr_0.01']>=.1-1e-12)&(v.original_prelimBG_tpr<.1-1e-12)
    v['lost_ge010']=(v['tpr_0.01']<.1-1e-12)&(v.original_prelimBG_tpr>=.1-1e-12)
    v['comparison_scope']='Same Oct9 family, method, inference, population and target; not complete historical best'
    v.to_csv(HERE/'prelim/background_delta_by_target.csv',index=False,float_format='%.17g')
    v[(v.population=='all_fixed')&v.new_ge010&~v.diagnostic_only].to_csv(HERE/'prelim/new_tpr010_crossings.csv',index=False,float_format='%.17g')
    for inf in ('all20','without1_2','without1_2_12'):
        q=d[(d.population=='all_fixed')&(d.inference==inf)&~d.diagnostic_only]
        records=[]
        for team,p in q.groupby('team'):
            old=p[p.background=='prelim'];a=old['tpr_0.01'].max();b=p['tpr_0.01'].max()
            oa=old[old['tpr_0.01']>=a-1e-12];nb=p[p['tpr_0.01']>=b-1e-12]
            records.append({'inference':inf,'team':int(team),'best_original_prelimBG_tpr':a,'best_all7BG_tpr':b,'delta':b-a,'new_ge010_from_old_best':bool(a<.1-1e-12 and b>=.1-1e-12),'old_best_configs':json.dumps(oa[['family','method']].to_dict('records'),ensure_ascii=False),'new_best_configs':json.dumps(nb[['background','family','method']].to_dict('records'),ensure_ascii=False),'comparison_scope':'Only the reproduced Oct9 joint/correction panel; not the complete historical method set','interpretation':'Retrospective best over observed methods/backgrounds; diagnostic, not selected generalizing attack'})
        pd.DataFrame(records).to_csv(HERE/'prelim'/f'best_observed_per_target_{inf}.csv',index=False,float_format='%.17g')
    q=s[(s.population=='all_fixed')&(s.evaluation=='all')&~s.diagnostic_only]
    keys=['family','inference','method']
    old=q[q.background=='prelim'][keys+['mean_tpr']].rename(columns={'mean_tpr':'prelimBG_mean_tpr'})
    z=q.merge(old,on=keys,validate='many_to_one');z['delta_prelimBG']=z.mean_tpr-z.prelimBG_mean_tpr
    z.to_csv(HERE/'prelim/background_delta_summary.csv',index=False,float_format='%.17g')
    z.groupby(['inference','background'],sort=False).agg(configurations=('method','size'),mean_tpr_across_configurations=('mean_tpr','mean'),mean_delta_originalBG=('delta_prelimBG','mean'),worst_config_delta=('delta_prelimBG','min'),improved_configurations=('delta_prelimBG',lambda x:int((x>1e-12).sum())),worsened_configurations=('delta_prelimBG',lambda x:int((x<-1e-12).sum()))).reset_index().to_csv(HERE/'prelim/background_configuration_diagnostics.csv',index=False,float_format='%.17g')
    robust=q.groupby(['family','inference','method']).agg(backgrounds=('background','size'),mean_tpr_across_BG=('mean_tpr','mean'),minimum_tpr_BG=('mean_tpr','min'),maximum_tpr_BG=('mean_tpr','max'),mean_delta_P01=('mean_delta_P01','mean'),minimum_delta_P01_BG=('mean_delta_P01','min'),maximum_delta_P01_BG=('mean_delta_P01','max')).reset_index()
    robust['independent_replicates']=False
    robust.to_csv(HERE/'prelim/configuration_background_sensitivity.csv',index=False,float_format='%.17g')

def audit(private_cache):
    checks=[];expected={'prelim':203,'main':82}
    for phase in expected:
        folders=[p for p in (HERE/phase).iterdir() if (p/'prediction_record.json').exists()]
        assert len(folders)==7
        for folder in folders:
            rec=json.loads((folder/'prediction_hashes.json').read_text())
            assert len(rec)==expected[phase]
            with np.load(private_cache/f'{phase}_{folder.name}.npz') as z:
                for i,row in enumerate(rec):
                    matrix=z[str(i)]
                    checksum=hashlib.sha256(np.ascontiguousarray(matrix,dtype='<f8').tobytes()).hexdigest()
                    assert checksum==row['sha256'] and np.isfinite(matrix).all()
                    assert matrix.shape==(875,len(row['teams']))
            checks.append({'phase':phase,'background':folder.name,'conditions':len(rec),'all_finite_and_hash_valid':True})
    d=pd.read_csv(HERE/'prelim/by_target.csv')
    assert len(d)==53382 and not d.duplicated(['background','family','inference','method','team','population']).any()
    assert not d[['tpr_0.005','tpr_0.01','tpr_0.02','auc']].isna().any().any()
    assert d['tpr_0.005'].between(0,1).all() and d['tpr_0.02'].between(0,1).all()
    r=pd.read_csv(HERE/'main/rank_by_target.csv')
    assert len(r)==12628 and not r.isna().any().any()
    tests=json.loads((HERE/'original_unit_tests.json').read_text())
    unit_count=sum(int(re.search(r'Ran (\d+) tests',t['output']).group(1)) for t in tests)
    assert unit_count==39 and all(t['returncode']==0 for t in tests)
    dump(HERE/'completeness_audit.json',{'completed':True,'prediction_panels':checks,'prelim_conditions':1421,'prelim_target_population_rows':len(d),'main_conditions':574,'main_target_rank_rows':len(r),'original_unit_tests_passed':unit_count,'historical_fullBG_hash_comparisons':187,'historical_summary_comparisons':327,'posthoc_summary_comparisons':16,'main_truth_read':False,'submission_created':False,'candidate_scores_in_report_tree':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private-cache',type=Path,default=Path('/workspace/shared/mia_joint_cache'));a=p.parse_args();summarize();audit(a.private_cache)
