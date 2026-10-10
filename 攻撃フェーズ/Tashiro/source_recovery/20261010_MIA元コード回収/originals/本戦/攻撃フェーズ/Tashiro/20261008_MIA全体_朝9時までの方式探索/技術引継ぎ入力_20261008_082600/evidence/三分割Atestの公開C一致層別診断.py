"""凍結三分割A診断のAtestを、同phase・同対象の公開C支持との一致で記述する。

4群は all13 / QI7だがall13なし / clinical3だがQI7なし / 両方なし。
公開一致は人や加工法の証明ではない。個人行・ID・順位・NPZを保存しない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import 共通 as u

a3=u.module('Cmatch_frozen_A3',u.HERE/'公開背景三分割の独立非会員診断.py')
gauss,split,extension=a3.gauss,a3.split,a3.extension
public=u.module('Cmatch_frozen_public_attributes',u.HERE/'公開背景との属性単位一致.py')
ALL13=public.ALL13
QI7=public.QI7
TRIPLE=('SBP','HDL','FPG')
GROUPS=('all13','QI7_without_all13','triple_without_QI7','neither_QI7_nor_triple')
OLD=u.HERE/'公開背景三分割の独立非会員診断結果'


def flags_and_groups(test,c):
    assert not test[list(ALL13)].isna().any().any() and not c[list(ALL13)].isna().any().any()
    flags=[]
    for columns in (ALL13,QI7,TRIPLE):
        observed=pd.MultiIndex.from_frame(test[list(columns)]).isin(pd.MultiIndex.from_frame(c[list(columns)]))
        support=set(c[list(columns)].itertuples(index=False,name=None))
        independent=np.array([row in support for row in test[list(columns)].itertuples(index=False,name=None)])
        assert np.array_equal(observed,independent)
        multiplicity=Counter(test[list(columns)].itertuples(index=False,name=None))
        assert int(observed.sum())==sum(n for row,n in multiplicity.items() if row in support)
        flags.append(observed)
    full,qi,triple=flags
    assert not (full&~qi).any() and not (full&~triple).any()
    group=np.full(len(test),3,int);group[triple&~qi]=2;group[qi&~full]=1;group[full]=0
    assert np.array_equal(group==3,~qi&~triple)
    return np.column_stack(flags),group


def numerical_qa():
    # Deliberate repeated A/C rows, QI7+clinical3 partial overlap, and disjoint patterns.
    c=pd.DataFrame([{column:float(i*100+j) for j,column in enumerate(ALL13)} for i in range(3)])
    test=pd.concat([c.iloc[[0,0,1,2]]],ignore_index=True)
    test.loc[2,'BMI']+=.125
    test.loc[3,'age']+=.125
    extra=c.iloc[[0]].copy();extra.loc[:,'SBP']+=.125
    test=pd.concat([test,extra],ignore_index=True);c=pd.concat([c,c.iloc[[0,0]]],ignore_index=True)
    flags,groups=flags_and_groups(test,c)
    assert groups.tolist()==[0,0,1,2,3]
    assert flags[:,2].tolist()==[True,True,True,True,False]
    return dict(duplicate_test_rows=True,duplicate_C_rows=True,priority_group_cases=5,Counter_pandas_max_count_error=0,
                QI7_and_clinical3_are_not_nested=True,all13_implies_both=True)


def frozen_hashes():
    return {str(p.resolve()):u.sha(p) for p in sorted(OLD.rglob('*')) if p.is_file()}


def support_hashes():
    result=a3.loaded_sources();result[str(Path(a3.__file__).resolve())]=u.sha(a3.__file__)
    result.pop(str(Path(__file__).resolve()),None)
    return result


def rates(record,names,n):
    for name in names:record[name+'_fraction']=record[name]/n if n else np.nan
    return record


def phase_run(phase,out,plan):
    started=time.perf_counter();data=u.setup(phase);owner,_,duplicates=split.wide.pub.public_folds(data)
    assignment,parts=a3.partition_a(data['a']);n=len(data['b']);a_start=len(data['query'])
    primary=[t for t in data['teams'] if phase!='prelim' or t not in (1,2)]
    olddiag=pd.read_csv(OLD/phase/'Atestの対象fold別CRPSと単CDF診断.csv',float_precision='round_trip').set_index(['fold','target','form','config'])
    oldhead=pd.read_csv(OLD/phase/'Atestの対象fold別通常順位閾値超過.csv',float_precision='round_trip').set_index(['fold','target','form','config','k'])
    cdfrows=[];headrows=[];supportrows=[];checks=[];fullflags={};foldflags={t:np.zeros(8,int) for t in data['teams']}
    max_loss_error=max_cdf_error=max_scale_error=0.;baseline_diag_cases=baseline_head_cases=0
    for target,c in data['cs'].items():
        flags,groups=flags_and_groups(data['a'],c)
        bit=flags[:,0].astype(int)+2*flags[:,1].astype(int)+4*flags[:,2].astype(int)
        fullflags[target]=np.bincount(bit,minlength=8)
    for fold in range(3):
        ref,cal,test=parts[fold],parts[(fold+1)%3],parts[(fold+2)%3]
        query=pd.concat([data['query'][list(u.base.QI7)+['age_bin']],cal[list(u.base.QI7)+['age_bin']],test[list(u.base.QI7)+['age_bin']]],ignore_index=True)
        test_start=a_start+len(cal);test_frame=pd.concat([data['b'],test],ignore_index=True)
        coordinate=gauss.contexts(ref);afit,_=gauss.fit_source(ref,coordinate);aback=gauss.density(afit,query,coordinate)[1]
        banks={form+'::'+cfg['name']:{} for form in a3.FORMS for cfg in split.CONFIGS};testbanks={name:{} for name in banks};groupbank={}
        for target,c in data['cs'].items():
            flags,groups=flags_and_groups(test,c);groupbank[target]=groups
            bit=flags[:,0].astype(int)+2*flags[:,1].astype(int)+4*flags[:,2].astype(int);bitcounts=np.bincount(bit,minlength=8);foldflags[target]+=bitcounts
            for bits,count in enumerate(bitcounts):
                supportrows.append(dict(phase=phase,fold=fold,target=target,primary_target=target in primary,test_A_rows=len(test),
                    all13_match=bool(bits&1),QI7_match=bool(bits&2),clinical3_match=bool(bits&4),matching_A_rows=int(count),
                    clinical3_columns='SBP+HDL+FPG',direction='A_test_rows_in_same_target_C_unique_tuple_support'))
            fitted,_=gauss.fit_source(c,coordinate)
            raws=(u.base.core(ref,query,c,u.NUM7,(10,))[10],gauss.density(fitted,query,coordinate)[1]-aback)
            qidx=np.flatnonzero((owner!=-1)&(owner!=target));qframe=data['query'].iloc[qidx]
            assert len(qidx)==(len(data['qs'])-1)*326 and qidx.max()<a_start
            ids=np.unique(np.r_[0,n-1,a_start,test_start,len(query)-1])
            controls=(u.base.core(ref,query.iloc[ids],c,u.NUM7,(10,))[10],
                gauss.density(fitted,query.iloc[ids],coordinate)[1]-gauss.density(afit,query.iloc[ids],coordinate)[1])
            for form,raw,control in zip(a3.FORMS,raws,controls):
                assert np.isfinite(raw).all();raw_error=float(np.max(np.abs(raw[ids]-control)));assert raw_error<1e-10
                qvalues=raw[qidx];avalues=raw[a_start:test_start];tvalues=raw[test_start:]
                calibrated=a3.calibrate_all(np.r_[raw[:n],tvalues],test_frame,qvalues,qframe,avalues,cal)
                frozen,_,_=split.predict_cdf(raw[:test_start],query.iloc[:test_start],data,target,owner,a_start)
                assert np.array_equal(calibrated[:,:n],frozen)
                evaluation=a3.evaluate_atest(tvalues,test,qvalues,qframe,avalues,cal)
                pit_error=float(np.max(np.abs(evaluation['pits'].T-calibrated[:,n:])));assert pit_error<1e-12
                for ci,config in enumerate(split.CONFIGS):
                    reference=calibrated[ci,:n][data['normal']];extended=extension.rank_extend(calibrated[ci],reference)
                    assert np.array_equal(extended,extension.counter_rank_extend(calibrated[ci],reference))
                    brank=extended[:n][data['normal']];assert np.array_equal(brank,rankdata(reference,method='average')/875)
                    name=form+'::'+config['name'];banks[name][target]=brank;testbanks[name][target]=extended[n:]
                    if target not in primary:continue
                    pit=evaluation['pits'][:,ci];loss=evaluation['losses'][:,ci];old=olddiag.loc[(fold,target,form,config['name'])]
                    error_loss=abs(float(loss.mean())-float(old.crps));error_cdf=abs(float(pit.mean())-float(old.single_fold_cdf_mean))
                    error_scale=abs(evaluation['scale']-float(old.training_Q_scale))
                    assert error_loss<1e-12 and error_cdf<1e-12 and error_scale<1e-12
                    assert len(test)==old.test_A_rows and len(qidx)==old.training_Q_rows and evaluation['rule']==old.scale_rule
                    totals=(int((pit>.99).sum()),int((pit>.95).sum()))
                    assert totals==tuple(round(float(old[key])*len(test)) for key in ('single_fold_cdf_above_99','single_fold_cdf_above_95'))
                    max_loss_error=max(max_loss_error,error_loss);max_cdf_error=max(max_cdf_error,error_cdf);max_scale_error=max(max_scale_error,error_scale);baseline_diag_cases+=1
                    records=[]
                    for gi,label in enumerate(GROUPS):
                        mask=groups==gi;gn=int(mask.sum());r=dict(phase=phase,fold=fold,target=target,form=form,config=config['name'],group=label,
                            group_A_rows=gn,test_A_rows=len(test),empty_group=gn==0,descriptive_small_group_lt20=0<gn<20,
                            single_fold_CDF_above_99_rows=int((pit[mask]>.99).sum()),single_fold_CDF_above_95_rows=int((pit[mask]>.95).sum()),
                            single_fold_CDF_sum=float(pit[mask].sum()),CRPS_sum=float(loss[mask].sum()),
                            single_fold_CDF_mean=float(pit[mask].mean()) if gn else np.nan,CRPS_mean=float(loss[mask].mean()) if gn else np.nan,
                            training_Q_scale=evaluation['scale'],post_discovery_description=True,official_FPR=False)
                        records.append(rates(r,('single_fold_CDF_above_99_rows','single_fold_CDF_above_95_rows'),gn))
                    assert sum(r['group_A_rows'] for r in records)==len(test)
                    assert tuple(sum(r[key] for r in records) for key in ('single_fold_CDF_above_99_rows','single_fold_CDF_above_95_rows'))==totals
                    assert abs(sum(r['CRPS_sum'] for r in records)-float(loss.sum()))<1e-8
                    assert abs(sum(r['single_fold_CDF_sum'] for r in records)-float(pit.sum()))<1e-9
                    cdfrows.extend(records)
                checks.append(dict(phase=phase,fold=fold,target=target,form=form,single_query_raw_max_error=raw_error,
                    frozen_six_CDF_B_max_error=0.,scaled_vs_unscaled_CDF_max_error=pit_error,rank_counter_max_error=0.,all_finite=True))
        bmatrices=u.matrices(banks,data['teams'],cross=(.1,))
        for name,bytarget in testbanks.items():
            bscore=bmatrices[name+'__cross0.1'];base_b=np.column_stack([banks[name][t] for t in data['teams']]);assert np.array_equal(bscore,extension.cross(base_b))
            trank=np.column_stack([bytarget[t] for t in data['teams']]);tscore=extension.cross(trank);form,config=name.split('::')
            for j,target in enumerate(data['teams']):
                if target not in primary:continue
                groups=groupbank[target]
                for k in a3.HEADS:
                    threshold=float(np.sort(bscore[:,j])[-k]);bge=int((bscore[:,j]>=threshold).sum());bgt=int((bscore[:,j]>threshold).sum())
                    ge=tscore[:,j]>=threshold;gt=tscore[:,j]>threshold;old=oldhead.loc[(fold,target,form,config,k)]
                    assert threshold==old.b_threshold and bge==old.b_head_rows_ge and bgt==old.b_head_rows_gt
                    assert int(ge.sum())==old.A_test_exceed_rows_ge and int(gt.sum())==old.A_test_exceed_rows_gt
                    assert int((trank[:,j]>1).sum())==old.A_test_rank_above_one_rows and float(trank[:,j].max())==old.A_test_rank_max
                    records=[]
                    for gi,label in enumerate(GROUPS):
                        mask=groups==gi;gn=int(mask.sum());r=dict(phase=phase,fold=fold,target=target,form=form,config=config,k=k,group=label,
                            group_A_rows=gn,test_A_rows=len(test),empty_group=gn==0,descriptive_small_group_lt20=0<gn<20,
                            A_test_exceed_rows_ge=int(ge[mask].sum()),A_test_exceed_rows_gt=int(gt[mask].sum()),
                            b_threshold=threshold,b_head_rows_ge=bge,b_head_rows_gt=bgt,normal_reference_rows=875,
                            post_discovery_description=True,official_FPR=False,independent_patient_exposures=False)
                        records.append(rates(r,('A_test_exceed_rows_ge','A_test_exceed_rows_gt'),gn))
                    assert sum(r['group_A_rows'] for r in records)==len(test)
                    assert sum(r['A_test_exceed_rows_ge'] for r in records)==int(ge.sum()) and sum(r['A_test_exceed_rows_gt'] for r in records)==int(gt.sum())
                    headrows.extend(records);baseline_head_cases+=1
        u.log(stage='Cmatch_A3_fold_complete',phase=phase,fold=fold,seconds=time.perf_counter()-started)
    assert baseline_diag_cases==len(olddiag) and baseline_head_cases==len(oldhead)
    assert all(np.array_equal(foldflags[t],fullflags[t]) for t in data['teams'])
    dest=out/phase;dest.mkdir();cdf=pd.DataFrame(cdfrows);head=pd.DataFrame(headrows)
    summaries=[]
    for keys,g in head.groupby(['phase','form','config','k','group'],sort=True):
        nonempty=g[g.group_A_rows>0];exposures=int(g.group_A_rows.sum());ge=int(g.A_test_exceed_rows_ge.sum());gt=int(g.A_test_exceed_rows_gt.sum())
        summaries.append(dict(zip(['phase','form','config','k','group'],keys),A_test_target_exposures=exposures,
            exceed_target_exposures_ge=ge,exceed_target_exposures_gt=gt,exposure_weighted_fraction_ge=ge/exposures if exposures else np.nan,
            exposure_weighted_fraction_gt=gt/exposures if exposures else np.nan,eligible_target_fold_count=len(nonempty),
            eligible_target_count=int(nonempty.target.nunique()),eligible_fold_count=int(nonempty.fold.nunique()),empty_target_fold_count=int((g.group_A_rows==0).sum()),
            equal_nonempty_target_fold_mean_ge=float(nonempty.A_test_exceed_rows_ge_fraction.mean()) if len(nonempty) else np.nan,
            equal_nonempty_target_fold_mean_gt=float(nonempty.A_test_exceed_rows_gt_fraction.mean()) if len(nonempty) else np.nan,
            independent_patient_exposures=False,post_discovery_description=True,official_FPR=False))
    outputs=[(pd.DataFrame(supportrows),'公開C支持へのAtest一致三flag組合せ件数.csv'),(cdf,'一致層別の単CDFとCRPS集約.csv'),
        (head,'一致層別の通常順位閾値超過.csv'),(pd.DataFrame(summaries),'一致層別の三fold対象exposure集約.csv'),(pd.DataFrame(checks),'元三分割対照と同一参照数値QA.csv')]
    for table,name in outputs:table.to_csv(dest/name,index=False,float_format='%.17g')
    assert frozen_hashes()==plan['frozen_A3_output_sha256']
    for path,expected in plan['input_sha256'].items():assert u.sha(path)==expected
    for path,expected in plan['support_code_sha256'].items():assert u.sha(path)==expected
    assert u.sha(__file__)==plan['code_sha256']
    record=dict(completed_jst=a3.stamp(),phase=phase,seconds=time.perf_counter()-started,code_sha256=u.sha(__file__),
        support_code_sha256=plan['support_code_sha256'],input_sha256=dict(u.base.INPUT_HASHES),output_sha256={p.name:u.sha(p) for p in dest.iterdir()},
        frozen_A3_output_sha256=plan['frozen_A3_output_sha256'],partition_assignment_sha256=hashlib.sha256(assignment.astype('<i8').tobytes()).hexdigest(),
        primary_targets=primary,all_targets_for_cross=data['teams'],baseline_diagnostic_cases_reproduced=baseline_diag_cases,
        baseline_head_cases_reproduced=baseline_head_cases,baseline_CRPS_max_error=max_loss_error,baseline_CDF_mean_max_error=max_cdf_error,
        baseline_training_Q_scale_max_error=max_scale_error,baseline_CDF_tail_count_max_error=0,baseline_head_count_max_error=0,
        baseline_threshold_max_error=0.,stratum_count_and_exceed_sum_max_error=0,full_A_vs_threefold_flag_count_max_error=0,
        matching_direction='A rows in same-phase same-target C unique tuple supports',public_Q_duplicates=duplicates,
        group_definition=list(GROUPS),QI7_group_may_also_match_clinical3=True,all13_implies_QI7_and_clinical3=True,
        full_A_age_bins_include_test_covariates=True,Atest_scores_in_calibration_fit=False,untouched_entire_development_test=False,
        C_construction_independence_guaranteed=False,post_discovery_description=True,membership_labels_used=False,mia_scoring=False,
        method_selection=False,individual_rows_or_ranks_exported=False,NPZ_created=False,official_FPR=False,
        final_prose_or_GitHub_created=False,python=sys.executable,python_version=platform.python_version())
    a3.write_json(dest/'実行記録.json',record);u.log(stage='Cmatch_A3_phase_complete',phase=phase,seconds=record['seconds'],baseline_diag=baseline_diag_cases,baseline_head=baseline_head_cases)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-plan',action='store_true');parser.add_argument('--output-dir',type=Path,default=u.HERE/'三分割Atestの公開C一致層別診断結果');args=parser.parse_args()
    assert sys.dont_write_bytecode and Path(sys.executable).absolute()==u.ROOT/'.venv/bin/python'
    out=args.output_dir
    if args.write_plan:
        assert not out.exists();qa=numerical_qa();out.mkdir()
        plan=dict(created_jst=a3.stamp(),code_sha256=u.sha(__file__),support_code_sha256=support_hashes(),input_sha256={str(p):u.sha(p) for p in a3.input_paths()},
            frozen_A3_output_sha256=frozen_hashes(),seed=a3.SEED,forms=a3.FORMS,configs=split.CONFIGS,heads=a3.HEADS,partition_sizes=[1714,1714,1715],
            match_attributes={'all13':list(ALL13),'QI7':list(QI7),'clinical3':list(TRIPLE)},groups=list(GROUPS),
            match_direction='Each Atest observed tuple is looked up in same-phase same-target public C unique support; A multiplicity retained.',
            priority='all13, then QI7 and not all13 (clinical3 overlap allowed), then clinical3 and not QI7, then neither QI7 nor clinical3',
            exact_matching='Same round-trip parsed observed numeric equality, no tolerance; no identity or processing inference.',
            strata_source='post-discovery descriptive follow-up; no changes to raw, CDF, rank, cross or target selection',
            baseline_barrier='Original A3 per-case CRPS/CDF means, Q scale, CDF tail counts, B threshold/head and Atest exceed counts; frozen source/output hashes unchanged.',
            empty_groups='n=0 count=0 rates=CSV blank; not zero percent',small_group_flag='fixed descriptive 0<n<20, not a selection threshold',
            aggregation='Target/fold rows; exposure-weighted aggregate plus equal nonempty target-fold rates with eligible counts; not independent patients.',
            numerical_qa=qa,no_GT_guard=True,full_A_age_bins_include_test_covariates=True,Atest_scores_in_calibration_fit=False,
            untouched_entire_development_test=False,C_construction_independence_guaranteed=False,membership_labels_used=False,mia_scoring=False,
            method_selection=False,individual_rows_or_ranks_exported=False,NPZ_created=False,official_FPR=False,final_prose_or_GitHub_created=False,
            limits=['Public tuple equality does not prove a person, copy or processing algorithm.',
                'These strata condition on C support after public-pattern discovery and are not an independent test or causal comparison.',
                'Clinical3 includes FPG, which is a public descriptor only and is not added to the frozen five numeric raw-score features.',
                'Known same-phase A nonmembership does not guarantee B-negative representativeness or independence from C construction.',
                'Full-A fixed age cuts include test covariates; test scores do not enter density, CDF or IQR fitting.',
                'CRPS only describes six calibrations within each frozen raw form; cross-form or historical CRPS performance ranking is not justified.',
                'Rank extension can exceed one, singlefold CDF is not guaranteed uniform, and threshold exposure rates are not official FPR.',
                'No method, filtering or calibration is selected from this descriptive follow-up.'])
        a3.write_json(out/'事前固定条件.json',plan);u.log(stage='Cmatch_A3_plan_fixed',code_sha256=plan['code_sha256'],support_sources=len(plan['support_code_sha256']),qa=qa);return
    assert out.is_dir() and {p.name for p in out.iterdir()}=={'事前固定条件.json'}
    plan=json.loads((out/'事前固定条件.json').read_text());assert u.sha(__file__)==plan['code_sha256'] and frozen_hashes()==plan['frozen_A3_output_sha256']
    for phase in ('prelim','main'):phase_run(phase,out,plan)
    a3.write_json(out/'完了記録.json',dict(completed_jst=a3.stamp(),code_sha256=u.sha(__file__),plan_sha256=u.sha(out/'事前固定条件.json'),
        phase_record_sha256={phase:u.sha(out/phase/'実行記録.json') for phase in ('prelim','main')},post_discovery_description=True,
        membership_labels_used=False,mia_scoring=False,method_selection=False,individual_rows_or_ranks_exported=False,final_prose_or_GitHub_created=False))


if __name__=='__main__':main()
