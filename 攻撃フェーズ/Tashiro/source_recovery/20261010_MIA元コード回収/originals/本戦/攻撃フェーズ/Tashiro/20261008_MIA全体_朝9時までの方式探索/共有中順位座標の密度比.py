"""共有Aref座標での、ラベルなし固定三形Gaussian source密度比。

raw robust座標+LedoitWolf、共有Aref midrank probability mapのprobit+LedoitWolf、
同じprobit+sourceごとの対角Gaussian。sex priorは含めず、県と数値をsex内で積にする。
固定splitA/halfQ-A/県性別300→性別年齢100を使い、形の性能選択や所属採点はしない。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.stats import multivariate_normal,rankdata
from sklearn.covariance import LedoitWolf

import 共通 as u

split=u.module('fixed_split_A_gaussian_source',u.HERE/'公開背景の分割較正.py')
FORMS=('robust_gaussian_lw','probit_gaussian_lw','probit_gaussian_diagonal')
CAL_INDEX=next(k for k,c in enumerate(split.CONFIGS) if c['source']=='Q_A_half' and c['shape']=='county300_sex_age100')
RHO=.98
D=len(u.NUM7)
EPS=np.finfo(float).eps


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


def midrank_map(values):
    """strict step ECDFではなく、unique支持のmidrank確率を線形補間する共有map。"""
    unique,count=np.unique(np.asarray(values,float),return_counts=True)
    probability=(np.cumsum(count)-count/2)/count.sum()
    assert np.isfinite(unique).all() and (probability>0).all() and (probability<1).all()
    return dict(unique=unique,probability=probability,reference_rows=int(count.sum()))


def map_probit(values,maps):
    values=np.asarray(values,float);out=np.empty_like(values)
    for j,mapping in enumerate(maps):
        p=np.interp(values[:,j],mapping['unique'],mapping['probability'])
        out[:,j]=ndtri(p)
    assert np.isfinite(out).all()
    return out


def contexts(reference):
    levels=sorted(reference.prefecture.unique());result={}
    for sex,frame in reference.groupby('sex',sort=True):
        values=u.base.numbers(frame,u.NUM7)
        center=np.median(values,axis=0);scale=(np.quantile(values,.75,axis=0)-np.quantile(values,.25,axis=0))/1.349
        assert (scale>0).all()
        z=(values-center)/scale
        result[int(sex)]=dict(center=center,scale=scale,rows=len(frame),maps=[midrank_map(z[:,j]) for j in range(D)],
                              freq=(frame.prefecture.value_counts().reindex(levels,fill_value=0)+1)/(len(frame)+len(levels)))
    assert set(result)=={0,1}
    return result


def fit_numeric(values,diagonal=False):
    """source meanとLW共分散、またはMLE対角分散。floorは数値rank不足専用の固定eps。"""
    values=np.asarray(values,float);mean=values.mean(axis=0)
    if diagonal:
        variance=np.mean((values-mean)**2,axis=0)
        floor=EPS*max(1.,float(variance.mean()));floored=variance<floor
        used=np.maximum(variance,floor);covariance=np.diag(used)
        inverse=np.diag(1/used);logdet=float(np.log(used).sum())
        return dict(mean=mean,covariance=covariance,inverse=inverse,logdet=logdet,shrinkage=None,
                    numerical_floor=floor,floored_dimensions=int(floored.sum()),unfloored_min=float(variance.min()),
                    variance=variance)
    estimator=LedoitWolf(store_precision=False,assume_centered=False).fit(values)
    covariance=estimator.covariance_;eigen,vector=np.linalg.eigh(covariance)
    floor=EPS*max(1.,float(np.trace(covariance)/D));floored=eigen<floor;used=np.maximum(eigen,floor)
    covariance=(vector*used)@vector.T;inverse=(vector*(1/used))@vector.T;logdet=float(np.log(used).sum())
    assert np.isfinite(inverse).all() and np.isfinite(logdet)
    return dict(mean=mean,covariance=covariance,inverse=inverse,logdet=logdet,
                shrinkage=float(estimator.shrinkage_),numerical_floor=floor,floored_dimensions=int(floored.sum()),
                unfloored_min=float(eigen.min()),variance=None)


def gaussian_logpdf(values,model):
    difference=np.asarray(values,float)-model['mean']
    mahalanobis=np.einsum('ni,ij,nj->n',difference,model['inverse'],difference)
    assert (mahalanobis>=-1e-9).all()
    result=-.5*(D*np.log(2*np.pi)+model['logdet']+mahalanobis)
    assert np.isfinite(result).all()
    return result


def fit_source(source,context):
    fitted={};records=[]
    for sex,coordinate in sorted(context.items()):
        part=source[source.sex==sex];assert len(part)>0
        z=(u.base.numbers(part,u.NUM7)-coordinate['center'])/coordinate['scale'];p=map_probit(z,coordinate['maps'])
        models=[fit_numeric(z),fit_numeric(p),fit_numeric(p,diagonal=True)]
        fitted[sex]=dict(models=models,rows=len(part),pref=part.prefecture.value_counts()/len(part))
        outside=[int(((z[:,j]<coordinate['maps'][j]['unique'][0])|(z[:,j]>coordinate['maps'][j]['unique'][-1])).sum()) for j in range(D)]
        for form,model in zip(FORMS,models):
            records.append(dict(sex=sex,form=form,source_rows=len(part),mean=model['mean'].tolist(),
                                covariance=model['covariance'].tolist(),logdet=model['logdet'],shrinkage=model['shrinkage'],
                                numerical_floor=model['numerical_floor'],floored_dimensions=model['floored_dimensions'],
                                unfloored_min=model['unfloored_min'],probit_outside_reference_support_by_column=outside,
                                diagonal_variance=model['variance'].tolist() if model['variance'] is not None else None))
    return fitted,records


def density(fitted,query,context):
    out=np.full((len(FORMS),len(query)),np.nan)
    for sex,coordinate in sorted(context.items()):
        positions=np.flatnonzero(query.sex.to_numpy()==sex)
        if not len(positions):continue
        frame=query.iloc[positions];source=fitted[sex]
        z=(u.base.numbers(frame,u.NUM7)-coordinate['center'])/coordinate['scale'];p=map_probit(z,coordinate['maps'])
        source_freq=frame.prefecture.map(source['pref']).fillna(0).to_numpy()
        reference_freq=frame.prefecture.map(coordinate['freq']).to_numpy()
        category=RHO*source_freq+(1-RHO)*reference_freq
        assert np.isfinite(category).all() and (category>0).all()
        for k,values in enumerate((z,p,p)):
            out[k,positions]=gaussian_logpdf(values,source['models'][k])+np.log(category)
    assert np.isfinite(out).all()
    return out


def numerical_qa():
    rng=np.random.default_rng(20261008);maximum=0.;comparisons=0
    for values in (rng.normal(size=(67,D)),rng.normal(size=(41,D))*np.arange(1.,D+1)):
        query=rng.normal(size=(13,D))
        for diagonal in (False,True):
            model=fit_numeric(values,diagonal);got=gaussian_logpdf(query,model)
            expected=multivariate_normal.logpdf(query,mean=model['mean'],cov=model['covariance'],allow_singular=False)
            maximum=max(maximum,float(np.max(np.abs(got-expected))));comparisons+=len(query)
    # Singular source floor is explicitly numerical and yields finite density.
    singular=[fit_numeric(np.ones((3,D)),diagonal=v) for v in (False,True)]
    assert all(model['floored_dimensions']==D for model in singular)
    assert all(np.isfinite(gaussian_logpdf(np.vstack([np.ones(D),np.zeros(D)]),model)).all() for model in singular)
    partially_constant=np.column_stack([np.ones(51),rng.normal(size=(51,D-1))])
    constant_diagonal=fit_numeric(partially_constant,diagonal=True)
    assert constant_diagonal['floored_dimensions']==1
    constant_query=rng.normal(size=(13,D));constant_query[:,0]=1.
    analytic=-.5*np.sum(np.log(2*np.pi*np.diag(constant_diagonal['covariance']))+
                        (constant_query-constant_diagonal['mean'])**2/np.diag(constant_diagonal['covariance']),axis=1)
    partial_error=float(np.max(np.abs(gaussian_logpdf(constant_query,constant_diagonal)-analytic)))
    assert partial_error<1e-10
    mapping=midrank_map(np.array([0.,0.,1.,3.]))
    assert np.array_equal(mapping['probability'],np.array([.25,.625,.875]))
    maps=[mapping for _ in range(D)];values=np.repeat(np.array([-1e300,0.,.5,1.,2.,3.,1e300])[:,None],D,axis=1)
    transformed=map_probit(values,maps)
    assert np.isfinite(transformed).all() and (np.diff(transformed[:,0])>=0).all()
    assert np.array_equal(transformed[0],transformed[1]) and np.array_equal(transformed[-1],transformed[-2])
    singleton=[midrank_map(np.ones(3)) for _ in range(D)]
    assert np.array_equal(map_probit(values,singleton),np.zeros_like(values))
    example=pd.DataFrame(dict(age=60+rng.normal(size=80),SBP=130+10*rng.normal(size=80),
                              TG=np.exp(rng.normal(4.,.5,size=80)),HDL=50+5*rng.normal(size=80),
                              ALT=np.exp(rng.normal(3.,.4,size=80)),sex=np.arange(80)%2,prefecture=np.arange(80)%3+1))
    coordinate=contexts(example);reference,_=fit_source(example,coordinate);source,_=fit_source(example.iloc[:48],coordinate)
    query=example.iloc[[0,1,7,12,28,67,75,79]]
    batch=density(source,query,coordinate)-density(reference,query,coordinate)
    singles=np.column_stack([density(source,query.iloc[[j]],coordinate)[:,0]-
                             density(reference,query.iloc[[j]],coordinate)[:,0] for j in range(len(query))])
    single_error=float(np.max(np.abs(batch-singles)));assert single_error<1e-10
    category_error=max(abs((RHO*example.iloc[:48].query('sex==@sex').prefecture.value_counts().reindex(model['freq'].index,fill_value=0)/
                            len(example.iloc[:48].query('sex==@sex'))+(1-RHO)*model['freq']).sum()-1)
                       for sex,model in coordinate.items())
    assert category_error<1e-14
    assert maximum<1e-8,maximum
    return dict(gaussian_independent_scipy_comparisons=comparisons,gaussian_max_log_error=maximum,
                tied_map_probabilities=mapping['probability'].tolist(),probit_finite_extremes=True,
                map_monotone=True,endpoint_clamp_test=True,singleton_map_zero=True,
                singular_numeric_floor_dimensions=[model['floored_dimensions'] for model in singular],
                partially_constant_diagonal_independent_log_error=partial_error,
                source_batch_single_query_max_log_error=single_error,category_probability_sum_error=float(category_error),
                covariance_floor_rule='float eps * max(1, mean raw source variance or mean covariance diagonal)',
                source_mle_diagonal_variance=True)


def phase_run(phase,out,qa,support_hashes):
    started=time.perf_counter();data=u.setup(phase);owner,_,duplicates=split.wide.pub.public_folds(data)
    assignment,parts=split.split_a(data);a_start=len(data['query'])
    queries=[pd.concat([data['query'][list(u.base.QI7)+['age_bin']],parts[1-f][list(u.base.QI7)+['age_bin']]],ignore_index=True) for f in range(2)]
    coordinates=[contexts(part) for part in parts];background=[];models=[];coordinate_rows=[];background_fits=[]
    for fold in range(2):
        fit,rows=fit_source(parts[fold],coordinates[fold]);background_fits.append(fit)
        background.append(density(fit,queries[fold],coordinates[fold]))
        models.extend(dict(phase=phase,fold=fold,target=-1,source='Aref',**row) for row in rows)
        for sex,model in coordinates[fold].items():
            coordinate_rows.append(dict(phase=phase,fold=fold,sex=sex,reference_rows=model['rows'],
                                        center=model['center'].tolist(),scale=model['scale'].tolist(),
                                        pref_levels=len(model['freq']),pref_probability={str(k):float(v) for k,v in model['freq'].items()},
                                        midrank_maps=[dict(column=col,unique_values=len(mp['unique']),
                                                           reference_min=float(mp['unique'][0]),reference_max=float(mp['unique'][-1]),
                                                           endpoint_probability=[float(mp['probability'][0]),float(mp['probability'][-1])])
                                                      for col,mp in zip(u.NUM7,model['maps'])]))
    diagnostic_targets=[t for t in data['teams'] if phase!='prelim' or t not in (1,2)]
    banks={form:{} for form in FORMS};outer=[];fold_rows=[];raw_checks=[]
    for completed,(target,c) in enumerate(data['cs'].items(),1):
        raws=[];predictions=[]
        for fold in range(2):
            fit,rows=fit_source(c,coordinates[fold]);values=density(fit,queries[fold],coordinates[fold]);raw=values-background[fold];raws.append(raw)
            models.extend(dict(phase=phase,fold=fold,target=target,source='C',**row) for row in rows)
            ids=np.unique(np.r_[0,1048,a_start,len(queries[fold])-1])
            cd=density(fit,queries[fold].iloc[ids],coordinates[fold]);ad=density(background_fits[fold],queries[fold].iloc[ids],coordinates[fold])
            error=float(np.max(np.abs((cd-ad)-raw[:,ids])));assert error<1e-10,error
            raw_checks.append(dict(phase=phase,target=target,fold=fold,single_query_raw_max_error=error,
                                   raw_min=float(raw.min()),raw_max=float(raw.max()),all_log_densities_finite=True))
            predictions.append(np.stack([split.predict_cdf(raw[k],queries[fold],data,target,owner,a_start)[0][CAL_INDEX] for k in range(len(FORMS))]))
        average=(predictions[0]+predictions[1])/2
        for k,form in enumerate(FORMS):banks[form][target]=u.rank(average[k],data['normal'])
        if target in diagnostic_targets:
            for heldowner in sorted(data['qs']):
                if heldowner==target:continue
                for k,form in enumerate(FORMS):
                    cases=[split.evaluate_fold(raws[fold][k],queries[fold],data,target,owner,heldowner,a_start) for fold in range(2)]
                    loss=(cases[0]['losses'][:,CAL_INDEX]+cases[1]['losses'][:,CAL_INDEX])/2
                    score=(cases[0]['pits'][:,CAL_INDEX]+cases[1]['pits'][:,CAL_INDEX])/2
                    for fold,case in enumerate(cases):
                        pit=case['pits'][:,CAL_INDEX]
                        fold_rows.append(dict(phase=phase,target=target,heldout_null_owner=heldowner,fold=fold,form=form,
                                              crps=float(case['losses'][:,CAL_INDEX].mean()),cdf_score_mean=float(pit.mean()),
                                              fold_cdf_above_99=float((pit>.99).mean()),fold_cdf_above_95=float((pit>.95).mean()),
                                              training_Q_rows=case['q_training_rows'],training_Q_iqr=case['scale'],scale_rule=case['scale_rule'],
                                              A_null_rows=case['a_null_rows'],heldout_rows=326,
                                              Q_county_support_min=int(case['q_counts'][:,0].min()),A_county_support_min=int(case['a_counts'][:,0].min()),
                                              membership_labels_used=False))
                    outer.append(dict(phase=phase,target=target,heldout_null_owner=heldowner,form=form,crps=float(loss.mean()),
                                      mean_fold_cdf_above_99=float(np.mean([(case['pits'][:,CAL_INDEX]>.99).mean() for case in cases])),
                                      mean_fold_cdf_above_95=float(np.mean([(case['pits'][:,CAL_INDEX]>.95).mean() for case in cases])),
                                      averaged_cdf_score_above_99=float((score>.99).mean()),averaged_cdf_score_above_95=float((score>.95).mean()),
                                      averaged_cdf_score_mean=float(score.mean()),heldout_rows=326,used_for_selection=False,membership_labels_used=False))
        u.log(stage='gaussian_source_target',phase=phase,completed_targets=completed,targets=len(data['teams']),seconds=time.perf_counter()-started)
    matrices=u.matrices(banks,data['teams'],cross=(.1,));table=pd.DataFrame(outer);fold_table=pd.DataFrame(fold_rows)
    metrics=['crps','mean_fold_cdf_above_99','mean_fold_cdf_above_95','averaged_cdf_score_above_99','averaged_cdf_score_above_95','averaged_cdf_score_mean']
    panel=table.groupby(['phase','heldout_null_owner','form'],as_index=False)[metrics].mean()
    summary=panel.groupby(['phase','form'],as_index=False)[metrics].mean();summary['targets']=len(diagnostic_targets);summary['owners']=len(data['qs']);summary['used_for_selection']=False
    target_summary=table.groupby(['phase','target','form'],as_index=False)[metrics].mean()
    reference=matrices[FORMS[0]+'__cross0.1'];drift=[]
    for name,mat in matrices.items():
        for j,target in enumerate(data['teams']):
            old=set(np.flatnonzero(rankdata(-reference[:,j],method='min')<=12));new=set(np.flatnonzero(rankdata(-mat[:,j],method='min')<=12))
            drift.append(dict(phase=phase,target=target,form=name,robust_lw_top_count=len(old),other_top_count=len(new),
                              added=len(new-old),removed=len(old-new),spearman=float(np.corrcoef(rankdata(reference[:,j]),rankdata(mat[:,j]))[0,1]),
                              membership_accuracy_unknown=True))
    dest=out/phase;dest.mkdir()
    pd.DataFrame({'A_row_position':np.arange(len(assignment)),'reference_fold':assignment,'heldout_fold':1-assignment}).to_csv(dest/'Aの固定二分割.csv',index=False)
    write_json(dest/'Arefの共有座標と有限中順位境界.json',coordinate_rows)
    write_json(dest/'source別Gaussian固定fit.json',models)
    floor_table=pd.DataFrame([{k:r[k] for k in ('phase','fold','target','source','sex','form','source_rows','shrinkage','numerical_floor','floored_dimensions','unfloored_min')} for r in models])
    floor_table.to_csv(dest/'数値floorのdataset発生数.csv',index=False,float_format='%.17g')
    pd.DataFrame(raw_checks).to_csv(dest/'同一共有座標の密度比検証.csv',index=False,float_format='%.17g')
    fold_table.to_csv(dest/'fold別の課題所有者保留.csv',index=False,float_format='%.17g')
    table.to_csv(dest/'二fold集約の課題所有者保留.csv',index=False,float_format='%.17g')
    panel.to_csv(dest/'Q所有者等重みの診断パネル.csv',index=False,float_format='%.17g')
    summary.to_csv(dest/'固定三形の非会員診断.csv',index=False,float_format='%.17g')
    target_summary.to_csv(dest/'対象別の固定三形診断.csv',index=False,float_format='%.17g')
    pd.DataFrame(drift).to_csv(dest/'robustLWとの上位12変化.csv',index=False,float_format='%.17g')
    names=sorted(matrices);scores=np.stack([matrices[name] for name in names])
    ranks=np.stack([np.column_stack([rankdata(mat[:,j],method='average')/875 for j in range(len(data['teams']))]) for mat in scores])
    np.savez_compressed(dest/'通常875行の内部順位だけ.npz',names=np.array(names),teams=np.array(data['teams']),cross_scores=scores,ranks=ranks,
                        record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
    for path,hashed in u.base.INPUT_HASHES.items():assert u.sha(path)==hashed,path
    for path,hashed in support_hashes.items():assert u.sha(path)==hashed,path
    assert not any('pool_ids' in path or '正解データ' in path for path in u.base.INPUT_HASHES)
    record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,
                python=sys.executable,python_version=platform.python_version(),code_path=str(Path(__file__).resolve()),code_sha256=u.sha(__file__),
                support_code_sha256=support_hashes,input_sha256=dict(u.base.INPUT_HASHES),output_sha256={p.name:u.sha(p) for p in dest.iterdir()},
                numerical_qa=qa,candidate_targets=data['teams'],diagnostic_targets=diagnostic_targets,forms=FORMS,
                calibration=split.CONFIGS[CAL_INDEX],A_split_seed=20261008,A_each_row_heldout_once=True,public_Q_duplicates=duplicates,
                single_query_raw_max_error=float(max(row['single_query_raw_max_error'] for row in raw_checks)),
                gaussian_source_fit_rows=len(models),floor_fit_cases=int((floor_table.floored_dimensions>0).sum()),
                floor_dimensions=int(floor_table.floored_dimensions.sum()),
                score='sex-conditional transformed-coordinate Gaussian logp(C)-logp(Aref), plus source-average fixed categorical kernel log ratio',
                sex_prior_included=False,numeric_columns=u.NUM7,numeric_coordinates='TG/ALT log1p; shared sex Aref median and IQR/1.349',
                probability_map='shared Aref unique supports with (less+.5equal)/n, linearly interpolated between supports, clamped to finite endpoint midrank probabilities; not a strict step ECDF',
                probit_is_exact_original_coordinate_jacobian_ratio=False,
                covariance_rule='LedoitWolf automatic covariance shrinkage per source; diagonal alternative uses source mean and MLE variance',
                numerical_floor_rule='float eps*max(1,source mean variance or mean covariance diagonal); eigen/variance below floor only',
                pref_kernel='p_s(pref|sex)=.98*(source_pref_count/source_sex_rows)+.02*((Aref_pref_count+1)/(Aref_sex_rows+47)); shared Aref category universe',
                candidate_protocol='two fold CDF mean, normal875 rank, other-target median subtraction0.1; no raw tie term',
                crps_protocol='mean exact per-fold loss / that form-fold outer-training-Q-only IQR; different raw scores cannot establish density-form performance ranking',
                cdf_tail_protocol='individual fold tails and averaged-score tails separate; averaged score is not generally uniform PIT',
                aggregation='equal Q owners after mean across eligible targets; prelim diagnostics exclude1/2',
                membership_labels_used=False,mia_scoring=False,method_selection=False,weight_search=False,C_modified=False,old_or_kit_A_used=False,
                limitations=['Numerical and prefecture factors assume conditional independence given sex.',
                             'Probit endpoint flattening loses tail information; this is a transformed-feature density score, not an exact original-coordinate Jacobian ratio.',
                             'Different raw-score forms have different prediction problems; their normalized CRPS is not a form-performance selection criterion.',
                             'Gaussian and diagonal assumptions describe source distributions and do not reconstruct processing algorithms.',
                             'A/Q known nonmembers need not represent normal875 nonmembers; CDF tails and head changes do not prove candidate FPR or MIA gains.'],
                final_prose_or_github_created=False)
    write_json(dest/'実行記録.json',record)
    u.log(stage='gaussian_source_complete',phase=phase,seconds=record['seconds'],cases=len(table),fold_cases=len(fold_table),output=str(dest))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'共有中順位座標の密度比結果')
    args=parser.parse_args();assert not args.output_dir.exists(),args.output_dir
    assert sys.dont_write_bytecode and Path(sys.executable).absolute()==u.ROOT/'.venv/bin/python'
    supports=loaded_sources();qa=numerical_qa();args.output_dir.mkdir()
    plan=dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),code_sha256=u.sha(__file__),support_code_sha256=supports,
              forms=FORMS,numeric_columns=u.NUM7,log_transform=['TG log1p','ALT log1p'],sex_prior_included=False,
              raw_coordinates='same sex Aref median/IQR1.349 for C/A/query',
              probability_map='Aref unique midrank probability (less+.5equal)/n, linear interpolation, finite endpoint clamp; not strict step ECDF',
              probit_score_scope='transformed-coordinate density source score; endpoint flattening means no exact original-coordinate Jacobian-ratio claim',
              gaussian_formula='-.5*(5log(2pi)+logdet(cov)+(x-mean)T inverse(cov)(x-mean))',
              covariance='source LedoitWolf or source mean+MLE diagonal variance; no membership fitting or grid search',
              numeric_floor='float eps*max(1,mean source variance or mean covariance diagonal), only eigenvalues/variances below floor',
              pref_formula='rho*(source_pref_count/source_sex_rows)+(1-rho)*(Aref_pref_count+1)/(Aref_sex_rows+K_Aref)',rho=RHO,
              pref_numerical_conditional_independence=True,score='logp_C-logp_Aref in shared coordinates, plus log categorical probabilities difference',
              calibration=split.CONFIGS[CAL_INDEX],split_seed=20261008,heldout='foreign-Q-owner out; target and heldowner excluded from Q null and form-fold trainingQ IQR',
              candidate='two fold CDF mean, normal875 rank, cross-other-target median0.1, no raw tie',
              crps_can_rank_forms=False,averaged_cdf_is_uniform_pit=False,numerical_qa=qa,
              only_current_phase_A=True,C_modified=False,method_selection=False,weight_search=False,membership_labels_used=False,mia_scoring=False,
              final_prose_or_github_created=False)
    write_json(args.output_dir/'事前固定条件.json',plan)
    u.log(stage='gaussian_source_plan_fixed',code_sha256=plan['code_sha256'],support_sources=len(supports),numerical_qa=qa)
    for phase in ('prelim','main'):phase_run(phase,args.output_dir,qa,supports)
    write_json(args.output_dir/'完了記録.json',dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),code_sha256=u.sha(__file__),
                                                 phases_completed=['prelim','main'],method_selection=False,membership_labels_used=False,mia_scoring=False,
                                                 final_prose_or_github_created=False))


if __name__=='__main__':main()
