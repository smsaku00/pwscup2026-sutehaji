"""Audit a pre-existing fixed P01/prelim-BG baseline, without rerunning any model.
No target outcome is used to choose a BG, a P01 variant or the candidate ensemble.
"""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import run_experiment as r
sys.path.insert(0,'/workspace/shared/mia_bg7_inputs/official_scoring/src')
from pwscup2026.scoring.attack_score import _tpr_at_fpr
SOURCE=Path('/workspace/shared/mia_all_bg_available/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/available/results_prelim_primary')

def main():
    protocol=json.loads((SOURCE/'fixed_protocol.json').read_text());teams=protocol['inference_targets']['all20'];assert teams==r.TEAMS
    record=json.loads((SOURCE/'freeze_record.json').read_text())
    p=SOURCE/'predictions_prelim_all20.npz'
    with np.load(p) as z:scores=z['P01'];assert scores.shape==(875,20) and np.isfinite(scores).all()
    h=[h for h in json.loads((SOURCE/'prediction_hashes.json').read_text()) if h['background']=='prelim' and h['inference']=='all20' and h['method']=='P01'];assert len(h)==1
    assert r.msha(scores)==h[0]['sha256']
    meta=pd.read_csv(HERE/'candidate_catalog.csv');matching=meta[(meta.family=='geometry')&(meta.background=='prelim')&(meta.method=='gen7_reference__county500__cross0.1')]
    assert len(matching)==1 and int(matching.iloc[0].candidate_index)==14109
    bankpath=r.OCT/'geometry/prelim/prelim/scores.npz'
    with np.load(bankpath) as z:
        assert z['teams'].tolist()==teams
        bank=z['scores'][z['names'].tolist().index('gen7_reference__county500__cross0.1')]
    assert np.array_equal(scores,bank)
    binding={'candidate_index':14109,'family':'geometry','background':'prelim','method':'gen7_reference__county500__cross0.1','source_score_file':str(bankpath),'exact_prediction_array_match':True,'max_abs_score_error':float(np.max(np.abs(scores-bank))),'matrix_sha256':r.msha(bank)}
    r.dump(HERE/'fixed_P01_bank_binding.json',binding)
    source=pd.read_csv(SOURCE/'by_target.csv',float_precision='round_trip');source=source[(source.background=='prelim')&(source.inference=='all20')&(source.method=='P01')&(source.population=='all_fixed')].set_index('team').loc[teams]
    b,private,rare,normal,y=r.load_eval();values=[];errors=[]
    mapping=pd.read_csv(r.OCT/'prelim_score_row_mapping.csv');schema=json.loads((r.OCT/'cache_schema.json').read_text())
    assert r.sha(r.OCT/'prelim_score_row_mapping.csv')==schema['row_mapping_sha256']
    assert np.array_equal(mapping.record_id,b.loc[normal,'record_id']) and np.array_equal(mapping.B_row,np.flatnonzero(normal))
    own=pd.read_csv(r.ROOT/'結果/正解データ/pool_ids_5.csv');assert own.record_id.is_unique
    assert set(b.record_id).issubset(set(own.record_id))
    alignment={'inference_team_order':teams,'primary_evaluation_teams':teams[2:],'official_FPR':.01,'evaluation_population':'all_fixed1049','normal875_score_row_to_B_record_id_verified':True,'row_mapping_sha256':schema['row_mapping_sha256'],'own_pool_labels_joined_by_record_id_before_target_membership_lookup':True,'target_labels_column_order_matches_scores':True,'normal_record_ids_sha256':hashlib.sha256(b.loc[normal,'record_id'].to_numpy(dtype='<i8').tobytes()).hexdigest()}

    for j,t in enumerate(teams):
        conf=np.zeros(len(b));conf[rare&~private]=.4;conf[normal]=.6+.3*rankdata(scores[:,j],method='min')/normal.sum()
        value=float(_tpr_at_fpr(conf,y[:,j],.01)[0]);values.append(value);errors.append(abs(value-source.loc[t,'tpr_0.01']))
    assert max(errors)<1e-12
    baseline=pd.Series(values,index=teams);outputs=pd.read_csv(HERE/'heldout_by_target.csv',float_precision='round_trip');summary=[];by=[]
    views=[('primary18_train_and_test_without1_2','train_and_test_without1_2',teams[2:]),('all20_loto','all20',teams),('all20_trained_summary_without1_2','all20',teams[2:])]
    for view,scope,active in views:
        panel=outputs[(outputs.scope==scope)&outputs.team.isin(active)];fixed=baseline.loc[active]
        summary.append({'view':view,'strategy':'fixed_P01_prelimBG_all20inference','targets':len(active),'mean_tpr':fixed.mean(),'TPR_ge010':int((fixed>=.1-1e-12).sum()),'TPR_gt010':int((fixed>.1+1e-12).sum()),'fixed_P01_mean_tpr':fixed.mean(),'delta_fixed_P01':0.,'wins':0,'ties':len(active),'losses':0,'candidate_strategy_label_use':'fixed pre-existing baseline; no membership-label training'})
        for name,g in panel.groupby('strategy',sort=False):
            vv=g.set_index('team')['tpr_0.01'].loc[active];delta=vv-fixed
            summary.append({'view':view,'strategy':name,'targets':len(active),'mean_tpr':vv.mean(),'TPR_ge010':int((vv>=.1-1e-12).sum()),'TPR_gt010':int((vv>.1+1e-12).sum()),'fixed_P01_mean_tpr':fixed.mean(),'delta_fixed_P01':delta.mean(),'wins':int((delta>1e-12).sum()),'ties':int((abs(delta)<=1e-12).sum()),'losses':int((delta< -1e-12).sum()),'candidate_strategy_label_use':'oracle not deployable' if name=='oracle_infeasible' else 'unchanged original frozen experimental strategy'})
            for t in active:by.append({'view':view,'team':t,'strategy':name,'candidate_tpr':vv[t],'fixed_P01_tpr':fixed[t],'delta':delta[t]})
    pd.DataFrame(summary).to_csv(HERE/'fixed_P01_baseline_comparison.csv',index=False,float_format='%.17g')
    pd.DataFrame(by).to_csv(HERE/'fixed_P01_baseline_by_target.csv',index=False,float_format='%.17g')
    table=pd.DataFrame({'team':teams,'fixed_P01_prelimBG_all20inference_tpr':values});table.to_csv(HERE/'fixed_P01_verified_target_tpr.csv',index=False,float_format='%.17g')
    result={'passed':True,'chosen_before_comparison':{'method':'P01','background':'prelim','base_inference':'all20','evaluation_population':'all_fixed1049','normal_rank_population':875},'primary_comparison_scope':'Team1/2 removed from evaluation. New selector/ensemble training excludes1/2, but frozen base predictions on both sides retain all20 inference. P01 has no selector training step.','baseline_not_selected_by_best_test_target_TPR':True,'source':str(SOURCE),'source_prediction_matrix_sha256':r.msha(scores),'source_hashes':{n:r.sha(SOURCE/n) for n in ['fixed_protocol.json','freeze_record.json','prediction_hashes.json','by_target.csv','predictions_prelim_all20.npz']},'original_TPR_comparison_count':len(errors),'max_official_TPR_error':max(errors),'fixed_P01_mean18':float(baseline.loc[teams[2:]].mean()),'additional_base_models_run':0,'same_bank_binding':binding,'alignment_audit':alignment,'scope_limit':'classifier family completed after original 8-family bank freeze and is not included','freeze_record':record}
    r.dump(HERE/'fixed_P01_baseline_audit.json',result)
    print(pd.DataFrame(summary).query("view == 'primary18_train_and_test_without1_2'").to_string(index=False))
if __name__=='__main__':main()
