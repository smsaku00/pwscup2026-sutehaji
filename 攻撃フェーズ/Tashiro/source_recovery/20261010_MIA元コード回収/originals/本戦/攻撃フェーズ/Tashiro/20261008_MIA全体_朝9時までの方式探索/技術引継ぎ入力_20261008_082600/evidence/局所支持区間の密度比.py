"""観測支持の間隔を不確かさとみなす、固定三形の条件付きKDE密度比。

Gaussian点 / 年齢のみVoronoi区間 / 数値五項目Voronoi区間を全対象へ共通適用。
TG/ALTはlog1p、sex内Aref尺度・Silverman帯域を固定しC/Aへ同じ規則を使う。
現行A二分割を交換し、Q/A半々＋県性別300→性別年齢100の較正だけを使う。
加工法・会員性能の復元保証はない。方式選択・所属正解・MIA採点・提出形式なし。
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
from scipy.integrate import quad
from scipy.special import log_ndtr,logsumexp
from scipy.stats import norm,rankdata

import 共通 as u

split=u.module('frozen_split_A_for_interval',u.HERE/'公開背景の分割較正.py')
FORMS=('point_gaussian','age_interval_gaussian','all5_interval_gaussian')
CAL_INDEX=next(k for k,c in enumerate(split.CONFIGS) if c['source']=='Q_A_half' and c['shape']=='county300_sex_age100')
RHO=.98
BLOCK=128
TINY_RELATIVE_WIDTH=1e-8
LOG_SQRT_2PI=.5*np.log(2*np.pi)


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


def voronoi_bounds(values):
    """各source/sex/列のunique値の中点。端は隣接間隔の半分だけ延長。"""
    values=np.asarray(values,float);unique=np.unique(values)
    if len(unique)==1:return values.copy(),values.copy(),unique
    middle=(unique[:-1]+unique[1:])/2
    lower=np.r_[unique[0]-(unique[1]-unique[0])/2,middle]
    upper=np.r_[middle,unique[-1]+(unique[-1]-unique[-2])/2]
    assert np.isfinite(lower).all() and np.isfinite(upper).all() and (upper>lower).all()
    index=np.searchsorted(unique,values)
    return lower[index],upper[index],unique


def point_log_kernel(x,center,h):
    return -.5*((x-center)/h)**2-np.log(h)-LOG_SQRT_2PI


def uniform_gaussian_log_kernel(x,center,lower,upper,h):
    """区間uniformとGaussianの畳み込み。正のtailはsurvival symmetry。"""
    x,center,lower,upper=np.broadcast_arrays(np.asarray(x,float),np.asarray(center,float),
                                           np.asarray(lower,float),np.asarray(upper,float))
    width=upper-lower
    assert (width>=0).all() and h>0
    result=point_log_kernel(x,center,h)
    active=width>TINY_RELATIVE_WIDTH*h
    if active.any():
        lo=(lower[active]-x[active])/h;hi=(upper[active]-x[active])/h
        positive=lo>0
        high=np.where(positive,log_ndtr(-lo),log_ndtr(hi))
        low=np.where(positive,log_ndtr(-hi),log_ndtr(lo))
        difference=low-high
        assert (difference<0).all(),'CDF差のfloat精度が不足している'
        result[active]=high+np.log(-np.expm1(difference))-np.log(width[active])
    assert np.isfinite(result).all()
    return result


def log_numeric_kernels(query,anchors,lower,upper,h):
    shape=(len(query),len(anchors));point=np.zeros(shape);interval=np.zeros(shape)
    age_difference=np.zeros(shape)
    for j in range(len(u.NUM7)):
        x=query[:,j,None];center=anchors[None,:,j]
        lp=point_log_kernel(x,center,h)
        li=uniform_gaussian_log_kernel(x,center,lower[None,:,j],upper[None,:,j],h)
        point+=lp;interval+=li
        if j==0:age_difference=li-lp
    return point,point+age_difference,interval


def scaled_quad_log_kernel(x,lower,upper,h):
    """独立quad。tailのunderflowを避け、最大標準Gaussian値を外へ出す。"""
    if upper-lower<=1e-6*h:
        closest=float(np.clip(x,lower,upper))
        peak=-.5*((closest-x)/h)**2-np.log(h)-LOG_SQRT_2PI
        measured=quad(lambda z:np.exp(-.5*((lower+z*(upper-lower))-closest)*
                                     ((lower+z*(upper-lower))+closest-2*x)/h**2),0.,1.,
                      epsabs=1e-12,epsrel=1e-12,limit=300)[0]
        return peak+np.log(measured)
    lo=(lower-x)/h;hi=(upper-x)/h;closest=float(np.clip(0.,lo,hi))
    peak=-.5*closest**2-LOG_SQRT_2PI
    candidates=[lo,hi]
    candidates.extend(v for v in (-12.,-8.,-4.,-1.,0.,1.,4.,8.,12.) if lo<v<hi)
    step=1/max(abs(closest),1.)
    candidates.extend(v for v in (closest-32*step,closest-8*step,closest-step,
                                   closest+step,closest+8*step,closest+32*step) if lo<v<hi)
    knots=np.unique(candidates)
    measured=sum(quad(lambda z:np.exp(-.5*(z-closest)*(z+closest)),a,b,
                      epsabs=1e-12,epsrel=1e-12,limit=300)[0] for a,b in zip(knots[:-1],knots[1:]))
    assert measured>0
    return peak+np.log(measured)-np.log(upper-lower)


def numerical_qa():
    tests=[(0.,-.5,.5,.7),(-1.,2.,3.,.25),(1.,-3.,-2.,.25),
           (3.,-100.,100.,.5),(0.,40.,42.,1.),(0.,-42.,-40.,1.),
           (0.,1000.,1000.001,1.),(1.3,-1e-12,1e-12,1.),(40.,-1e-12,1e-12,1.)]
    for support in (np.arange(50.,85.),np.unique(np.rint(np.linspace(50.,84.,11)))):
        lower,upper,_=voronoi_bounds(support)
        for j in (0,len(support)//2,len(support)-1):
            for x in (48.,68.,86.):tests.append((x,float(lower[j]),float(upper[j]),.75))
    maximum=0.;comparisons=0
    for x,lower,upper,h in tests:
        center=(lower+upper)/2
        got=float(uniform_gaussian_log_kernel(np.array([x]),np.array([center]),np.array([lower]),np.array([upper]),h)[0])
        expected=scaled_quad_log_kernel(x,lower,upper,h)
        maximum=max(maximum,abs(got-expected));comparisons+=1
    assert maximum<2e-7,maximum
    repeated=np.array([2.,2.,2.]);lower,upper,unique=voronoi_bounds(repeated)
    assert len(unique)==1 and np.array_equal(lower,repeated) and np.array_equal(upper,repeated)
    x=np.array([-5.,2.,8.])[:,None]
    got=uniform_gaussian_log_kernel(x,repeated[None,:],lower[None,:],upper[None,:],.6)
    point=point_log_kernel(x,repeated[None,:],.6)
    pointlimit_error=float(np.max(np.abs(got-point)));assert pointlimit_error==0
    source=np.array([[0.,1.,2.,3.,4.],[.5,1.5,1.,2.,3.],[-.5,.2,.3,1.,2.]])
    query=np.array([[.2,.3,.4,.5,.6],[1.,2.,3.,4.,5.]])
    bounds=[voronoi_bounds(source[:,j]) for j in range(5)]
    low=np.column_stack([b[0] for b in bounds]);high=np.column_stack([b[1] for b in bounds]);h=.7
    values=log_numeric_kernels(query,source,low,high,h)
    expected=np.log(np.prod(norm.pdf((query[:,None,:]-source[None,:,:])/h)/h,axis=2))
    point_pdf_error=float(np.max(np.abs(values[0]-expected)));assert point_pdf_error<1e-12
    assert all(np.isfinite(value).all() for value in values)
    assert len(np.unique(np.rint(np.linspace(50.,84.,11))))==11
    return dict(uniform_quad_comparisons=comparisons,uniform_quad_max_log_error=maximum,
                singleton_point_limit_max_error=pointlimit_error,naive_point_pdf_max_error=point_pdf_error,
                tested_support_sizes=[35,11],tested_tails='both signs,40sigma,1000sigma',
                tested_widths='singleton0,2e-12,ordinary,200',tiny_relative_width=TINY_RELATIVE_WIDTH)


def contexts(reference):
    levels=sorted(reference.prefecture.unique());result={}
    for sex,frame in reference.groupby('sex',sort=True):
        values=u.base.numbers(frame,u.NUM7)
        scale=(np.quantile(values,.75,axis=0)-np.quantile(values,.25,axis=0))/1.349
        assert (scale>0).all()
        result[int(sex)]=dict(scale=scale,center=np.median(values,axis=0),rows=len(frame),
                              h=(4/(len(u.NUM7)+2)/len(frame))**(1/(len(u.NUM7)+4)),
                              freq=(frame.prefecture.value_counts().reindex(levels,fill_value=0)+1)/(len(frame)+len(levels)))
    assert set(result)=={0,1}
    return result


def density(source,query,context):
    out=np.full((len(FORMS),len(query)),np.nan);support=[]
    for sex in sorted(context):
        model=context[sex];part=source[source.sex==sex];positions=np.flatnonzero(query.sex.to_numpy()==sex)
        assert len(part)>0
        if len(positions)==0:continue
        anchors=(u.base.numbers(part,u.NUM7)-model['center'])/model['scale']
        q=(u.base.numbers(query.iloc[positions],u.NUM7)-model['center'])/model['scale']
        bounds=[voronoi_bounds(anchors[:,j]) for j in range(len(u.NUM7))]
        lower=np.column_stack([item[0] for item in bounds]);upper=np.column_stack([item[1] for item in bounds])
        pref=part.prefecture.to_numpy();qpref=query.iloc[positions].prefecture.to_numpy()
        probability=query.iloc[positions].prefecture.map(model['freq']).to_numpy()
        assert np.isfinite(probability).all() and (probability>0).all()
        for j,col in enumerate(u.NUM7):
            unique=bounds[j][2];spacing=np.diff(unique);width=upper[:,j]-lower[:,j]
            support.append(dict(sex=sex,column=col,source_rows=len(part),unique_values=len(unique),
                                transformed_scale=float(model['scale'][j]),bandwidth=float(model['h']),
                                spacing_min=float(spacing.min()) if len(spacing) else 0.,
                                spacing_median=float(np.median(spacing)) if len(spacing) else 0.,
                                spacing_max=float(spacing.max()) if len(spacing) else 0.,
                                interval_width_min=float(width.min()),interval_width_median=float(np.median(width)),
                                interval_width_max=float(width.max()),singleton_point_fallback=len(unique)==1,
                                tiny_width_point_rows=int((width<=TINY_RELATIVE_WIDTH*model['h']).sum())))
        for start in range(0,len(positions),BLOCK):
            stop=min(start+BLOCK,len(positions));logs=log_numeric_kernels(q[start:stop],anchors,lower,upper,model['h'])
            category=np.log(RHO*(qpref[start:stop,None]==pref[None,:])+(1-RHO)*probability[start:stop,None])
            for k,values in enumerate(logs):out[k,positions[start:stop]]=logsumexp(values+category,axis=1)-np.log(len(part))
    assert np.isfinite(out).all()
    return out,support


def phase_run(phase,out,qa,support_hashes):
    started=time.perf_counter();data=u.setup(phase);owner,_,duplicates=split.wide.pub.public_folds(data)
    assignment,parts=split.split_a(data);a_start=len(data['query'])
    queries=[pd.concat([data['query'][list(u.base.QI7)+['age_bin']],parts[1-f][list(u.base.QI7)+['age_bin']]],ignore_index=True) for f in range(2)]
    fits=[contexts(part) for part in parts];background=[];supports=[];fitting=[]
    for fold in range(2):
        values,rows=density(parts[fold],queries[fold],fits[fold]);background.append(values)
        supports.extend(dict(phase=phase,fold=fold,target=-1,source='Aref',**row) for row in rows)
        fitting.extend(dict(phase=phase,fold=fold,sex=sex,reference_rows=model['rows'],h=float(model['h']),
                            numeric_scale=model['scale'].tolist(),numeric_center=model['center'].tolist(),
                            category_levels=len(model['freq'])) for sex,model in fits[fold].items())
    diagnostic_targets=[t for t in data['teams'] if phase!='prelim' or t not in (1,2)]
    banks={form:{} for form in FORMS};outer=[];fold_rows=[];raw_checks=[]
    for completed,(target,c) in enumerate(data['cs'].items(),1):
        raws=[];predictions=[]
        for fold in range(2):
            values,rows=density(c,queries[fold],fits[fold]);raw=values-background[fold];raws.append(raw)
            supports.extend(dict(phase=phase,fold=fold,target=target,source='C',**row) for row in rows)
            ids=np.unique(np.r_[0,1048,a_start,len(queries[fold])-1])
            cd,_=density(c,queries[fold].iloc[ids],fits[fold]);ad,_=density(parts[fold],queries[fold].iloc[ids],fits[fold])
            error=float(np.max(np.abs((cd-ad)-raw[:,ids])));assert error==0,error
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
        u.log(stage='interval_density_target',phase=phase,completed_targets=completed,targets=len(data['teams']),seconds=time.perf_counter()-started)
    matrices=u.matrices(banks,data['teams'],cross=(.1,));table=pd.DataFrame(outer);fold_table=pd.DataFrame(fold_rows)
    metrics=['crps','mean_fold_cdf_above_99','mean_fold_cdf_above_95','averaged_cdf_score_above_99','averaged_cdf_score_above_95','averaged_cdf_score_mean']
    panel=table.groupby(['phase','heldout_null_owner','form'],as_index=False)[metrics].mean()
    summary=panel.groupby(['phase','form'],as_index=False)[metrics].mean();summary['targets']=len(diagnostic_targets);summary['owners']=len(data['qs']);summary['used_for_selection']=False
    target_summary=table.groupby(['phase','target','form'],as_index=False)[metrics].mean()
    reference=matrices[FORMS[0]+'__cross0.1'];drift=[]
    for name,mat in matrices.items():
        for j,target in enumerate(data['teams']):
            old=set(np.flatnonzero(rankdata(-reference[:,j],method='min')<=12));new=set(np.flatnonzero(rankdata(-mat[:,j],method='min')<=12))
            drift.append(dict(phase=phase,target=target,form=name,point_top_count=len(old),interval_top_count=len(new),
                              added=len(new-old),removed=len(old-new),spearman=float(np.corrcoef(rankdata(reference[:,j]),rankdata(mat[:,j]))[0,1]),
                              membership_accuracy_unknown=True))
    dest=out/phase;dest.mkdir()
    pd.DataFrame({'A_row_position':np.arange(len(assignment)),'reference_fold':assignment,'heldout_fold':1-assignment}).to_csv(dest/'Aの固定二分割.csv',index=False)
    pd.DataFrame(fitting).to_json(dest/'性別ごとの固定尺度と帯域.json',orient='records',force_ascii=False,indent=2)
    pd.DataFrame(supports).to_csv(dest/'観測支持と仮定した区間幅.csv',index=False,float_format='%.17g')
    pd.DataFrame(raw_checks).to_csv(dest/'同一参照の密度比検証.csv',index=False,float_format='%.17g')
    fold_table.to_csv(dest/'fold別の課題所有者保留.csv',index=False,float_format='%.17g')
    table.to_csv(dest/'二fold集約の課題所有者保留.csv',index=False,float_format='%.17g')
    panel.to_csv(dest/'Q所有者等重みの診断パネル.csv',index=False,float_format='%.17g')
    summary.to_csv(dest/'固定三形の非会員診断.csv',index=False,float_format='%.17g')
    target_summary.to_csv(dest/'対象別の固定三形診断.csv',index=False,float_format='%.17g')
    pd.DataFrame(drift).to_csv(dest/'点Gaussianとの上位12変化.csv',index=False,float_format='%.17g')
    names=sorted(matrices);scores=np.stack([matrices[name] for name in names])
    ranks=np.stack([np.column_stack([rankdata(mat[:,j],method='average')/875 for j in range(len(data['teams']))]) for mat in scores])
    np.savez_compressed(dest/'通常875行の内部順位だけ.npz',names=np.array(names),teams=np.array(data['teams']),cross_scores=scores,ranks=ranks,
                        record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
    for path,hashed in u.base.INPUT_HASHES.items():assert u.sha(path)==hashed,path
    for path,hashed in support_hashes.items():assert u.sha(path)==hashed,path
    record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,
                python=sys.executable,python_version=platform.python_version(),code_path=str(Path(__file__).resolve()),code_sha256=u.sha(__file__),
                support_code_sha256=support_hashes,input_sha256=dict(u.base.INPUT_HASHES),output_sha256={p.name:u.sha(p) for p in dest.iterdir()},
                numerical_qa=qa,candidate_targets=data['teams'],diagnostic_targets=diagnostic_targets,forms=FORMS,
                calibration=split.CONFIGS[CAL_INDEX],A_split_seed=20261008,A_cross_fold_qi7_duplicates=0,A_each_row_heldout_once=True,
                public_Q_duplicates=duplicates,single_query_raw_max_error=float(max(row['single_query_raw_max_error'] for row in raw_checks)),
                score='sex-conditional log KDE(C)-log KDE(Aref); normalized point Gaussian or uniform-Gaussian per numeric dimension',
                numeric_coordinates='TG/ALT log1p, centered and scaled by the same sex Aref median and IQR/1.349',
                interval_rule='Source-specific observed unique values in transformed standardized coordinates; midpoint Voronoi boundaries; endpoint half-gap extension; unique1/tinywidth point fallback.',
                bandwidth='Silverman d5 and sex Aref row count; fixed without search',pref_kernel='rho.98 equal category plus .02 same sex Aref smoothed category frequency; same C/A rule',
                candidate_protocol='average two fold CDF scores, normal875 rank, other-target median0.1; no raw tie term',
                crps_protocol='mean of two exact per-fold losses scaled by that fold outer-training Q-only IQR; not CRPS of a single averaged predictive distribution',
                cdf_tail_protocol='Marginal per-fold CDF tails and averaged-CDF-score threshold tails are stored separately; averaged score is not generally uniform PIT.',
                aggregation='equal Q owners after mean across eligible targets; prelim diagnostic targets exclude1/2',
                known_nonmember_basis='only current phase A; reference and held A split; Q target/heldowner excluded from outer null',
                membership_labels_used=False,mia_scoring=False,method_selection=False,bandwidth_search=False,C_modified=False,old_or_kit_A_used=False,
                limitations=['Sparse observed support can arise from a small cohort or from coarsening; the source of sparsity is unknown.',
                             'Uniform local intervals are a latent-coarsening hypothesis, not reconstructed truth or a proven processing algorithm.',
                             'Voronoi uncertainty is formed after log/scale transform and need not correspond to physical-unit uniform uncertainty.',
                             'A/Q are different sampling strata; known nonmember identity does not establish normal875 distributional representativeness.',
                             'Neither CDF-tail type establishes candidate FPR or MIA performance.'],final_prose_or_github_created=False)
    write_json(dest/'実行記録.json',record)
    u.log(stage='interval_density_complete',phase=phase,seconds=record['seconds'],cases=len(table),fold_cases=len(fold_table),output=str(dest))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'局所支持区間の密度比結果')
    args=parser.parse_args();assert not args.output_dir.exists(),args.output_dir
    assert sys.dont_write_bytecode and Path(sys.executable).absolute()==u.ROOT/'.venv/bin/python'
    supports=loaded_sources();qa=numerical_qa();args.output_dir.mkdir()
    plan=dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),code_sha256=u.sha(__file__),support_code_sha256=supports,
              forms=FORMS,source_rule='same local support interval rule for C and Aref',interval_coordinates='after TG/ALT log1p and sex Aref robust scaling',
              interval_endpoints='unique-value midpoints; first/last extend half adjacent spacing; singleton/tiny width point fallback',
              score='log normalized KDE(C)-log normalized KDE(Aref); sex-specific density, no Gen-LRA increment ratio',
              numeric_columns=u.NUM7,bandwidth='(4/(5+2)/nAref_sex)^(1/(5+4)) fixed',scale='same sex Aref IQR/1.349',rho=RHO,
              pref_frequency='same sex Aref counts+1 over fold Aref category universe for both C and A densities',
              split='same seed20261008 two halves2571/2572; swap Aref and heldA',calibration=split.CONFIGS[CAL_INDEX],
              heldout='Q-owner leave-out; Q target and heldowner excluded; trainingQ-only IQR per form/fold',
              candidate='two fold CDF mean then normal875 rank then cross median0.1; no extra raw tie',
              head_control='point_gaussian, fixed common control; no per-target selection',numerical_qa=qa,
              only_current_phase_A=True,C_modified=False,method_selection=False,bandwidth_search=False,membership_labels_used=False,mia_scoring=False,
              cdf_tail_mean_is_uniform_pit=False,final_prose_or_github_created=False,
              hypothesis='Observed sparse support is treated as uncertainty. It may reflect cohort sampling or coarsening; no processing-algorithm or membership-performance guarantee.')
    write_json(args.output_dir/'事前固定条件.json',plan)
    u.log(stage='interval_density_plan_fixed',code_sha256=plan['code_sha256'],support_sources=len(supports),numerical_qa=qa)
    for phase in ('prelim','main'):phase_run(phase,args.output_dir,qa,supports)
    write_json(args.output_dir/'完了記録.json',dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),code_sha256=u.sha(__file__),
                                                 phases_completed=['prelim','main'],method_selection=False,membership_labels_used=False,mia_scoring=False,
                                                 final_prose_or_github_created=False))


if __name__=='__main__':main()
