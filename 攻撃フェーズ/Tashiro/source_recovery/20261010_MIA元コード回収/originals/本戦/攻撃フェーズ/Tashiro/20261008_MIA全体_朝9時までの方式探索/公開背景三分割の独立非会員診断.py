"""公開Aの参照・較正・テストを分離する固定三循環の非会員診断。

Gen7k10と共有probit-LWを、各形内の固定六CDFで診断する。
Atestは較正/尺度fitへ入れず、固定normal875順位参照の拡張と共同cross.1を通す。
個人行/ID/順位配列/提出形式を保存せず、所属真値・公式採点・方式選択を使わない。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def no_gt_guard(event,args):
    if event!='open' or not args or isinstance(args[0],int):return
    path=unicodedata.normalize('NFC',str(args[0]))
    if any(v in path for v in ('pool_ids','/正解データ/','/アーカイブ/','.npz','F_mia.csv','F_aia.csv','attack_score.py','対象別TPR.csv','result.txt')):
        raise PermissionError(f'三分割診断では参照禁止: {path}')


sys.addaudithook(no_gt_guard)
import 共通 as u

gauss=u.module('triple_A_frozen_gaussian',u.HERE/'共有中順位座標の密度比.py')
split=gauss.split
extension=u.module('triple_A_frozen_extension',u.HERE/'通常順位を通した非会員診断.py')
SEED=20261008
FORMS=('gen7k10','shared_probit_gaussian_lw')
HEADS=(10,12)


def stamp():return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def write_json(path,value):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def loaded_sources():
    paths=set()
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name:continue
        p=Path(name).resolve()
        if p.suffix=='.py' and p.is_relative_to(u.ROOT/'本戦') and p!=Path(__file__).resolve():paths.add(p)
    return {str(p):u.sha(p) for p in sorted(paths)}


def input_paths():
    result=[]
    for phase in ('prelim','main'):
        distributed=u.ROOT/('予備戦' if phase=='prelim' else '本戦')/'配布データ'
        own=next(distributed.glob('PWSCup2026_prelim_team05_*' if phase=='prelim' else 'PWSCup2026_final_team28_*'))
        targets=distributed/'PWSCup2026_prelim_attack' if phase=='prelim' else u.HERE.parent/'20260928_配布データ基本分析/入力'
        a=distributed/'PWSCup2026_prelim_common/A_bg.csv' if phase=='prelim' else own/'A_bg.csv'
        result.extend([a,own/'B_self.csv',*targets.glob('C_*.csv'),*targets.glob('aia_challenge_*.csv')])
    return sorted(set(p.resolve() for p in result))


def partition_a(a):
    assert len(a)==5143
    order=np.random.default_rng(SEED).permutation(len(a));assignment=np.empty(len(a),int)
    assignment[order[:1714]]=0;assignment[order[1714:3428]]=1;assignment[order[3428:]]=2
    parts=[a.iloc[np.flatnonzero(assignment==fold)].reset_index(drop=True) for fold in range(3)]
    assert np.bincount(assignment).tolist()==[1714,1714,1715]
    for i in range(3):
        for j in range(i):
            assert not pd.MultiIndex.from_frame(parts[i][list(u.base.QI7)]).isin(pd.MultiIndex.from_frame(parts[j][list(u.base.QI7)])).any()
    return assignment,parts


def calibrate_all(values,frame,qvalues,qframe,avalues,aframe):
    qg=u.cal.empirical_cdf(values,qvalues);ag=u.cal.empirical_cdf(values,avalues)
    qc,nqc=u.cal.group_stats(values,frame,qvalues,qframe,['prefecture','sex'])
    qp,nqp=u.cal.group_stats(values,frame,qvalues,qframe,['sex','age_bin'])
    ac,nac=u.cal.group_stats(values,frame,avalues,aframe,['prefecture','sex'])
    ap,nap=u.cal.group_stats(values,frame,avalues,aframe,['sex','age_bin'])
    cdfs=np.column_stack([qg,qc,qp,ag,ac,ap]);out=[]
    for config in split.CONFIGS:
        lq=nqc/(nqc+300);la=nac/(nac+300)
        pq=nqp/(nqp+100) if config['shape']==split.SHAPES[1] else np.zeros(len(values))
        pa=nap/(nap+100) if config['shape']==split.SHAPES[1] else np.zeros(len(values))
        w=np.column_stack([config['q_weight']*(1-lq)*(1-pq),config['q_weight']*lq,config['q_weight']*(1-lq)*pq,
                           config['a_weight']*(1-la)*(1-pa),config['a_weight']*la,config['a_weight']*(1-la)*pa])
        assert np.allclose(w.sum(axis=1),1,atol=1e-15,rtol=0)
        out.append((cdfs*w).sum(axis=1))
    out=np.stack(out);assert np.isfinite(out).all()
    return out


def evaluate_atest(values,frame,qvalues,qframe,avalues,aframe):
    scale=float(np.quantile(qvalues,.75)-np.quantile(qvalues,.25));rule='target_excluded_training_Q_iqr'
    if scale<=np.finfo(float).eps:scale=max(float(np.std(qvalues)),1.);rule='target_excluded_training_Q_std_or_one'
    qvalues=qvalues/scale;avalues=avalues/scale;observed=values/scale
    losses=np.full((len(frame),6),np.nan);pits=np.full_like(losses,np.nan);counts=np.zeros((len(frame),4),int)
    for (pref,sex,age),positions in frame.groupby(['prefecture','sex','age_bin'],sort=True).indices.items():
        qm=split.group_masks(qframe,pref,sex,age);am=split.group_masks(aframe,pref,sex,age)
        samples=[qvalues,qvalues[qm[0]],qvalues[qm[1]],avalues,avalues[am[0]],avalues[am[1]]]
        support=[len(samples[j]) for j in (1,2,4,5)];w=split.mixture_weights(support);samples=split.filled_samples(samples)
        a,b,c=split.wide.crps_basis(samples,observed[positions]);v=w[:,1:]
        losses[positions]=a[:,None]+2*b@v.T+np.einsum('ki,ij,kj->k',v,c,v)[None,:]
        pits[positions]=np.column_stack([u.cal.empirical_cdf(observed[positions],sample) for sample in samples])@w.T
        counts[positions]=support
    assert np.isfinite(losses).all() and np.isfinite(pits).all() and (losses>=-1e-9).all()
    return dict(losses=losses,pits=pits,scale=scale,rule=rule,counts=counts)


def numerical_qa():
    reference=np.array([-3.,0.,0.,1.,5.]);values=np.r_[reference,-10.,.5,10.]
    got=extension.rank_extend(values,reference);independent=extension.counter_rank_extend(values,reference)
    assert np.array_equal(got,independent) and np.array_equal(got[:len(reference)],rankdata(reference,method='average')/len(reference))
    assert got.min()==.5/len(reference) and got.max()==1+.5/len(reference)
    crps=split.numerical_qa();gaussian=gauss.numerical_qa()
    return dict(rank_extension_counter_max_error=0.,rank_extension_reference_scipy_max_error=0.,
                rank_extension_test_queries=len(values),exact_six_CDF_crps=crps,gaussian_source=gaussian)


def phase_run(phase,out,plan):
    started=time.perf_counter();data=u.setup(phase);owner,_,duplicates=split.wide.pub.public_folds(data)
    assignment,parts=partition_a(data['a']);n=len(data['b']);a_start=len(data['query'])
    primary=[t for t in data['teams'] if phase!='prelim' or t not in (1,2)]
    diagnostics=[];heads=[];checks=[];partition_rows=[]
    for fold in range(3):
        ref=parts[fold];cal=parts[(fold+1)%3];test=parts[(fold+2)%3]
        query=pd.concat([data['query'][list(u.base.QI7)+['age_bin']],cal[list(u.base.QI7)+['age_bin']],test[list(u.base.QI7)+['age_bin']]],ignore_index=True)
        test_start=a_start+len(cal);test_frame=pd.concat([data['b'],test],ignore_index=True)
        coordinate=gauss.contexts(ref);afit,_=gauss.fit_source(ref,coordinate);aback=gauss.density(afit,query,coordinate)[1]
        banks={form+'::'+c['name']:{} for form in FORMS for c in split.CONFIGS};testbanks={name:{} for name in banks}
        partition_rows.append(dict(phase=phase,fold=fold,reference_part=fold,calibration_part=(fold+1)%3,test_part=(fold+2)%3,
                                   reference_rows=len(ref),calibration_rows=len(cal),test_rows=len(test),QI7_pairwise_duplicates=0,
                                   each_A_row_test_once=True,A_test_used_in_calibration_or_scale=False))
        for target,c in data['cs'].items():
            fitted,_=gauss.fit_source(c,coordinate)
            raws=(u.base.core(ref,query,c,u.NUM7,(10,))[10],gauss.density(fitted,query,coordinate)[1]-aback)
            qidx=np.flatnonzero((owner!=-1)&(owner!=target));qframe=data['query'].iloc[qidx]
            assert len(qidx)==(len(data['qs'])-1)*326 and qidx.max()<a_start
            ids=np.unique(np.r_[0,n-1,a_start,test_start,len(query)-1])
            controls=(u.base.core(ref,query.iloc[ids],c,u.NUM7,(10,))[10],
                      gauss.density(fitted,query.iloc[ids],coordinate)[1]-gauss.density(afit,query.iloc[ids],coordinate)[1])
            for form,raw,control in zip(FORMS,raws,controls):
                assert np.isfinite(raw).all();raw_error=float(np.max(np.abs(raw[ids]-control)));assert raw_error<1e-10
                qvalues=raw[qidx];avalues=raw[a_start:test_start];tvalues=raw[test_start:]
                values=np.r_[raw[:n],tvalues]
                calibrated=calibrate_all(values,test_frame,qvalues,qframe,avalues,cal)
                frozen,_,_=split.predict_cdf(raw[:test_start],query.iloc[:test_start],data,target,owner,a_start)
                cal_error=float(np.max(np.abs(calibrated[:,:n]-frozen)));assert cal_error==0,cal_error
                evaluation=evaluate_atest(tvalues,test,qvalues,qframe,avalues,cal)
                pit_error=float(np.max(np.abs(evaluation['pits'].T-calibrated[:,n:])));assert pit_error<1e-12,pit_error
                rank_error=0.
                for k,config in enumerate(split.CONFIGS):
                    reference=calibrated[k,:n][data['normal']]
                    extended=extension.rank_extend(calibrated[k],reference)
                    independent=extension.counter_rank_extend(calibrated[k],reference)
                    error=float(np.max(np.abs(extended-independent)));assert error==0;rank_error=max(rank_error,error)
                    brank=extended[:n][data['normal']];assert np.array_equal(brank,rankdata(reference,method='average')/875)
                    name=form+'::'+config['name'];banks[name][target]=brank;testbanks[name][target]=extended[n:]
                    if target in primary:
                        pit=evaluation['pits'][:,k];counts=evaluation['counts']
                        diagnostics.append(dict(phase=phase,fold=fold,target=target,form=form,config=config['name'],reference_rows=len(ref),
                            calibration_A_rows=len(cal),test_A_rows=len(test),training_Q_rows=len(qidx),training_Q_scale=evaluation['scale'],scale_rule=evaluation['rule'],
                            crps=float(evaluation['losses'][:,k].mean()),single_fold_cdf_mean=float(pit.mean()),
                            single_fold_cdf_above_99=float((pit>.99).mean()),single_fold_cdf_above_95=float((pit>.95).mean()),
                            Q_county_sex_min=int(counts[:,0].min()),Q_county_sex_median=float(np.median(counts[:,0])),
                            Q_sex_age_min=int(counts[:,1].min()),Q_sex_age_median=float(np.median(counts[:,1])),
                            A_county_sex_min=int(counts[:,2].min()),A_county_sex_median=float(np.median(counts[:,2])),
                            A_sex_age_min=int(counts[:,3].min()),A_sex_age_median=float(np.median(counts[:,3])),
                            A_test_used_in_calibration_or_scale=False,used_for_method_selection=False,official_FPR=False))
                checks.append(dict(phase=phase,fold=fold,target=target,form=form,single_query_raw_max_error=raw_error,
                                   frozen_six_CDF_B_control_max_error=cal_error,scaled_vs_unscaled_CDF_max_error=pit_error,
                                   rank_extension_counter_max_error=rank_error,all_scores_finite=True,A_test_used_in_calibration_or_scale=False))
        bmatrices=u.matrices(banks,data['teams'],cross=(.1,))
        for name,bytarget in testbanks.items():
            bscore=bmatrices[name+'__cross0.1'];base_b=np.column_stack([banks[name][t] for t in data['teams']])
            assert np.array_equal(bscore,extension.cross(base_b))
            trank=np.column_stack([bytarget[t] for t in data['teams']]);tscore=extension.cross(trank);form,config=name.split('::')
            for j,target in enumerate(data['teams']):
                if target not in primary:continue
                for k in HEADS:
                    threshold=float(np.sort(bscore[:,j])[-k]);bge=int((bscore[:,j]>=threshold).sum());bgt=int((bscore[:,j]>threshold).sum())
                    assert bge>=k and bgt<k
                    ge=int((tscore[:,j]>=threshold).sum());gt=int((tscore[:,j]>threshold).sum())
                    heads.append(dict(phase=phase,fold=fold,target=target,form=form,config=config,k=k,test_A_rows=len(test),normal_reference_rows=875,
                        b_threshold=threshold,b_head_rows_ge=bge,b_head_rows_gt=bgt,A_test_exceed_rows_ge=ge,A_test_exceed_rows_gt=gt,
                        A_test_exceed_fraction_ge=ge/len(test),A_test_exceed_fraction_gt=gt/len(test),A_test_rank_above_one_rows=int((trank[:,j]>1).sum()),
                        A_test_rank_max=float(trank[:,j].max()),normal_branch_only=True,official_FPR=False,membership_accuracy_unknown=True,
                        A_test_exposures_are_unique_patient_count=False,used_for_method_selection=False))
        u.log(stage='triple_A_fold_complete',phase=phase,fold=fold,seconds=time.perf_counter()-started)
    dest=out/phase;dest.mkdir();diag=pd.DataFrame(diagnostics);head=pd.DataFrame(heads)
    dmetrics=['crps','single_fold_cdf_mean','single_fold_cdf_above_99','single_fold_cdf_above_95']
    fold_diag=diag.groupby(['phase','fold','form','config'],as_index=False)[dmetrics].mean();fold_diag['primary_targets']=len(primary)
    hmetrics=['A_test_exceed_fraction_ge','A_test_exceed_fraction_gt','b_head_rows_ge','b_head_rows_gt','A_test_rank_above_one_rows']
    fold_head=head.groupby(['phase','fold','form','config','k'],as_index=False)[hmetrics].mean();fold_head['primary_targets']=len(primary)
    phase_head=fold_head.groupby(['phase','form','config','k'],as_index=False)[hmetrics].mean();phase_head['folds']=3;phase_head['aggregation']='equal folds after equal primary targets; no averaged CDF'
    exposure=head.groupby(['phase','form','config','k'],as_index=False).agg(A_test_target_exposures=('test_A_rows','sum'),A_test_exceed_exposures_ge=('A_test_exceed_rows_ge','sum'),A_test_exceed_exposures_gt=('A_test_exceed_rows_gt','sum'))
    exposure['exposure_weighted_exceed_fraction_ge']=exposure.A_test_exceed_exposures_ge/exposure.A_test_target_exposures
    exposure['exposure_weighted_exceed_fraction_gt']=exposure.A_test_exceed_exposures_gt/exposure.A_test_target_exposures
    phase_head=phase_head.merge(exposure,on=['phase','form','config','k']);phase_head['official_FPR']=False;phase_head['independent_patient_exposures']=False
    outputs=[(pd.DataFrame(partition_rows),'三分割の非重複と役割.csv'),(pd.DataFrame(checks),'同一参照と凍結順位の数値QA.csv'),
             (diag,'Atestの対象fold別CRPSと単CDF診断.csv'),(fold_diag,'Atestのfold別固定六CDF診断.csv'),
             (head,'Atestの対象fold別通常順位閾値超過.csv'),(fold_head,'Atestのfold別通常順位閾値超過.csv'),(phase_head,'Atestの三fold集約通常順位閾値超過.csv')]
    for table,name in outputs:table.to_csv(dest/name,index=False,float_format='%.17g')
    for path,expected in plan['input_sha256'].items():assert u.sha(path)==expected,path
    for path,expected in plan['support_code_sha256'].items():assert u.sha(path)==expected,path
    assert u.sha(__file__)==plan['code_sha256']
    record=dict(completed_jst=stamp(),seconds=time.perf_counter()-started,phase=phase,code_path=str(Path(__file__).resolve()),code_sha256=u.sha(__file__),
                support_code_sha256=plan['support_code_sha256'],input_sha256=dict(u.base.INPUT_HASHES),output_sha256={p.name:u.sha(p) for p in dest.iterdir()},
                forms=FORMS,configs=split.CONFIGS,partition_sizes=[len(p) for p in parts],partition_assignment_sha256=__import__('hashlib').sha256(assignment.astype('<i8').tobytes()).hexdigest(),
                public_Q_duplicates=duplicates,primary_targets=primary,all_targets_for_cross=data['teams'],normal_reference_rows=875,
                diagnosis_cases=len(diag),threshold_cases=len(head),max_single_query_error=float(pd.DataFrame(checks).single_query_raw_max_error.max()),
                maximum_rank_counter_error=0.,A_test_used_in_calibration_or_scale=False,each_A_row_test_once_per_target_form_config=True,
                A_test_target_exposures_per_form_config=int(5143*len(primary)),membership_labels_used=False,mia_scoring=False,method_selection=False,
                individual_rows_or_ranks_exported=False,NPZ_created=False,official_FPR=False,averaged_fold_CDF_created=False,
                final_prose_or_GitHub_created=False,python=sys.executable,python_version=platform.python_version())
    write_json(dest/'実行記録.json',record);u.log(stage='triple_A_phase_complete',phase=phase,seconds=record['seconds'],cases=len(diag),threshold_cases=len(head))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-plan',action='store_true');parser.add_argument('--output-dir',type=Path,default=u.HERE/'公開背景三分割の独立非会員診断結果');args=parser.parse_args()
    assert sys.dont_write_bytecode and Path(sys.executable).absolute()==u.ROOT/'.venv/bin/python'
    out=args.output_dir
    if args.write_plan:
        assert not out.exists();qa=numerical_qa();out.mkdir()
        plan=dict(created_jst=stamp(),code_sha256=u.sha(__file__),support_code_sha256=loaded_sources(),input_sha256={str(p):u.sha(p) for p in input_paths()},
                  seed=SEED,partition_sizes=[1714,1714,1715],partition='permutation split at1714/3428, preserve within-part original row order',
                  roles='foldf ref=A[f],cal=A[(f+1)%3],test=A[(f+2)%3]; each A row is test once',forms=FORMS,configs=split.CONFIGS,
                  age_cuts='same-phase full public A quantiles20/40/60/80%; covariate-only fixed strata, no test raw score fitting',
                  raw='Gen7k10 or same shared Aref probit-LedoitWolf source density ratio; original formulas unchanged',
                  calibration='Q target-owner excluded, A-null only Acal; Atest never enters CDF or IQR fitting',
                  scale='each form-fold target-excluded training Q raw IQR only; fixed std/one fallback at machine precision',
                  rank_extension='R=(#Bref<x+.5#Bref=x+.5)/875; Bref fixed from that target/method calnormal875; at reference points rankdataaverage/875',
                  cross='same .1 other-target median for B and Atest, all20/22 jointly; primary diagnostics prelim18/main22',
                  thresholds='Bnormal kth largest cross.1 score k10/12; Atest >= and > separately, Bhead ties retained',
                  aggregation='perfold target aggregates; no averaged fold CDF; equal folds and exposure-weighted rates labeled separately',
                  numerical_qa=qa,no_GT_guard=True,method_selection=False,membership_labels_used=False,mia_scoring=False,individual_rows_or_ranks_exported=False,
                  limits=['A is same-phase known nonmember, but this does not establish representativeness of B true nonmembers.',
                          'All folds reuse the same public A/C/Q and B normal reference; target/fold/config exposures are not independent patients.',
                          'This is a predeclared internal nonmember diagnostic, not independent external validation or official candidate FPR.',
                          'CRPS comparisons apply only to the six calibrations within the same form/fold raw variable and training-Q scale.',
                          'Two raw forms and older one-half-A reference experiments have different prediction variables; no cross-form/old CRPS ranking.',
                          'No fold-average CDF is formed; singlefold CDF tail rates are diagnostics, not guaranteed uniform PIT.',
                          'Atest covariates affect only fixed full-A strata definitions; its raw scores never enter reference density, CDF fit or IQR fit.'],
                  final_prose_or_GitHub_created=False)
        write_json(out/'事前固定条件.json',plan);u.log(stage='triple_A_plan_fixed',code_sha256=plan['code_sha256'],support_sources=len(plan['support_code_sha256']),numerical_qa=qa);return
    assert out.is_dir() and {p.name for p in out.iterdir()}=={'事前固定条件.json'}
    plan=json.loads((out/'事前固定条件.json').read_text());assert u.sha(__file__)==plan['code_sha256']
    for phase in ('prelim','main'):phase_run(phase,out,plan)
    write_json(out/'完了記録.json',dict(completed_jst=stamp(),code_sha256=u.sha(__file__),phases_completed=['prelim','main'],
                                     phase_record_sha256={phase:u.sha(out/phase/'実行記録.json') for phase in ('prelim','main')},plan_sha256=u.sha(out/'事前固定条件.json'),
                                     membership_labels_used=False,mia_scoring=False,method_selection=False,individual_rows_or_ranks_exported=False,final_prose_or_GitHub_created=False))


if __name__=='__main__':main()
