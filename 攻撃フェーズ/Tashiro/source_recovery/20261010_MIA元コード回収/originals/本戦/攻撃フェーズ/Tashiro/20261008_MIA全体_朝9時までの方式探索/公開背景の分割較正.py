"""現行Aだけの二分割を使う、会員ラベル無しの固定六較正案。

A参照とA非会員nullは固定seedで分離し、二foldで交換する。
同じ参照でB・全Q・heldAのGen-LRA7 k10を計算。Cは変更しない。
Q/A/50:50 × flat県性別300/県性別300から性別年齢100への退避を比較する。
方式・強度の選択、所属正解、MIA採点、提出形式、最終文章は使用・作成しない。
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


def no_gt_guard(event, args):
    if event!='open' or not args or isinstance(args[0], int): return
    path=unicodedata.normalize('NFC', str(args[0]))
    forbidden=('pool_ids', '/正解データ/', '/アーカイブ/', 'F_mia.csv',
               '対象別TPR.csv', '集約比較.csv', 'result.txt', 'attack_score.py',
               '/participant_data/A_bg.csv')
    if any(term in path for term in forbidden):
        raise PermissionError(f'本実験では参照禁止: {path}')


sys.addaudithook(no_gt_guard)
import 共通 as u

wide=u.module('split_a_exact_crps',u.HERE/'非会員較正の広い集団を保留検証.py')
SEED=20261008
SOURCE_WEIGHTS=(('Q_only',1.,0.),('A_only',0.,1.),('Q_A_half',.5,.5))
SHAPES=('flat_county_sex300','county300_sex_age100')
CONFIGS=tuple(dict(name=source+'__'+shape,source=source,q_weight=wq,a_weight=wa,shape=shape)
              for source,wq,wa in SOURCE_WEIGHTS for shape in SHAPES)
LOCAL=300
PARENT=100
CROSS=.1
TOP=12


def write_json(path, value):
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')


def loaded_project_sources():
    """推移的importを含む、実際に読み込まれた本戦支援ソースを固定する。"""
    paths=set()
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if not name: continue
        p=Path(name).resolve()
        if p.suffix=='.py' and p.is_relative_to(u.ROOT/'本戦') and p!=Path(__file__).resolve():
            paths.add(p)
    return {str(p):u.sha(p) for p in sorted(paths)}


def filled_samples(samples):
    assert len(samples)==6 and len(samples[0]) and len(samples[3])
    return [samples[0],samples[1] if len(samples[1]) else samples[0],
            samples[2] if len(samples[2]) else samples[0],samples[3],
            samples[4] if len(samples[4]) else samples[3],
            samples[5] if len(samples[5]) else samples[3]]


def mixture_weights(counts):
    """six samplesの順＝Q全体/Q県性別/Q性別年齢/A全体/A県性別/A性別年齢。"""
    nq,nqp,na,nap=counts
    rows=[]
    for config in CONFIGS:
        localq=nq/(nq+LOCAL);locala=na/(na+LOCAL)
        parentq=nqp/(nqp+PARENT) if config['shape']==SHAPES[1] else 0.
        parenta=nap/(nap+PARENT) if config['shape']==SHAPES[1] else 0.
        q=np.array([(1-localq)*(1-parentq),localq,(1-localq)*parentq])*config['q_weight']
        a=np.array([(1-locala)*(1-parenta),locala,(1-locala)*parenta])*config['a_weight']
        rows.append(np.r_[q,a])
    result=np.stack(rows)
    assert (result>=0).all() and np.allclose(result.sum(axis=1),1.,atol=1e-15,rtol=0)
    return result


def numerical_qa():
    """任意五差分のCRPSを、全CDF節点と観測値で区切ったGauss積分と独立照合。"""
    rng=np.random.default_rng(SEED);nodes,weights=np.polynomial.legendre.leggauss(3)
    cases=[
        [np.array([0.,0.,1.,3.]),np.array([-1.,0.,2.]),np.array([.5,.5,4.]),
         np.array([-2.,-1.,1.,5.]),np.array([2.,3.]),np.array([-3.,0.,2.])],
        [rng.normal(size=n) for n in (19,8,5,23,11,3)],
        [np.full(n,2.) for n in (3,2,1,4,3,2)],
        [np.array([0.,1.,4.]),np.array([]),np.array([]),np.array([-1.,2.,5.]),np.array([]),np.array([])],
    ]
    maximum=0.;comparisons=0
    for original in cases:
        samples=filled_samples(original)
        observed=np.array([-6.,-1.,0.,.5,2.,7.])
        a,b,c=wide.crps_basis(samples,observed)
        assert b.shape==(len(observed),5) and c.shape==(5,5)
        fixed=mixture_weights([len(original[1]),len(original[2]),len(original[4]),len(original[5])])
        coefficients=np.vstack([fixed,np.eye(6)[[0,1,3,4]],rng.dirichlet(np.ones(6),size=2)])
        for coefficient in coefficients:
            v=coefficient[1:]
            expected=a+2*b@v+v@c@v
            for j,z in enumerate(observed):
                knots=np.unique(np.r_[np.concatenate(samples),z])
                lo=knots[:-1,None];hi=knots[1:,None]
                x=((hi-lo)*nodes+(hi+lo))/2
                f=sum(coefficient[k]*u.cal.empirical_cdf(x.ravel(),samples[k]) for k in range(6))
                integral=float(np.sum((hi-lo)/2*(f.reshape(x.shape)-(x>=z))**2*weights))
                maximum=max(maximum,abs(float(expected[j])-integral));comparisons+=1
    assert maximum<1e-11,maximum
    return dict(cases=len(cases),cdf_samples=6,differences=5,comparisons=comparisons,maximum_absolute_error=maximum,
                independent_integral='three-point Gauss on every interval from all six CDF supports plus observation')


def split_a(data):
    a=data['a'];assert len(a)==5143,len(a)
    order=np.random.default_rng(SEED).permutation(len(a))
    assignment=np.empty(len(a),int);assignment[order[:len(a)//2]]=0;assignment[order[len(a)//2:]]=1
    parts=[a.iloc[np.flatnonzero(assignment==fold)].reset_index(drop=True) for fold in range(2)]
    overlap=pd.MultiIndex.from_frame(parts[0][list(u.base.QI7)]).isin(pd.MultiIndex.from_frame(parts[1][list(u.base.QI7)]))
    assert not overlap.any(),'ArefとAheldのQI7に重複がある'
    assert np.bincount(assignment).tolist()==[2571,2572]
    return assignment,parts


def group_masks(frame,prefecture,sex,age_bin):
    sx=frame.sex.to_numpy()==sex
    return [(frame.prefecture.to_numpy()==prefecture)&sx,sx&(frame.age_bin.to_numpy()==age_bin)]


def evaluate_fold(raw,query,data,target,owner,heldowner,a_start):
    """外側Q所有者はnullにも尺度にも使わない。A-onlyを含め六案の尺度は同じ。"""
    n=len(data['query']);eligible=(owner!=-1)&(owner!=target)&(owner!=heldowner)
    held_indices=np.flatnonzero(owner==heldowner)
    assert heldowner!=target and len(held_indices)==326
    q_indices=np.flatnonzero(eligible);q=data['query'].iloc[q_indices]
    a=query.iloc[a_start:];qvalues=raw[q_indices];avalues=raw[a_start:]
    scale=float(np.quantile(qvalues,.75)-np.quantile(qvalues,.25));scale_rule='outer_training_Q_iqr'
    if scale<=np.finfo(float).eps:
        scale=max(float(np.std(qvalues)),1.);scale_rule='outer_training_Q_std_or_one'
    qvalues=qvalues/scale;avalues=avalues/scale
    observed=raw[held_indices]/scale;heldq=data['query'].iloc[held_indices]
    losses=np.full((len(held_indices),len(CONFIGS)),np.nan);pits=np.full_like(losses,np.nan)
    q_counts=np.zeros((len(held_indices),2),int);a_counts=q_counts.copy()
    for (prefecture,sex,age_bin),positions in heldq.groupby(['prefecture','sex','age_bin'],sort=True).indices.items():
        qm=group_masks(q,prefecture,sex,age_bin);am=group_masks(a,prefecture,sex,age_bin)
        samples=[qvalues,qvalues[qm[0]],qvalues[qm[1]],avalues,avalues[am[0]],avalues[am[1]]]
        counts=[len(samples[j]) for j in (1,2,4,5)];mix=mixture_weights(counts)
        samples=filled_samples(samples)
        base,b,c=wide.crps_basis(samples,observed[positions]);v=mix[:,1:]
        losses[positions]=base[:,None]+2*b@v.T+np.einsum('ki,ij,kj->k',v,c,v)[None,:]
        cdfs=np.column_stack([u.cal.empirical_cdf(observed[positions],sample) for sample in samples])
        pits[positions]=cdfs@mix.T
        q_counts[positions]=counts[:2];a_counts[positions]=counts[2:]
    assert np.isfinite(losses).all() and np.isfinite(pits).all()
    assert (losses>=-1e-9).all() and (pits>=0).all() and (pits<=1+1e-14).all()
    return dict(losses=losses,pits=pits,scale=scale,scale_rule=scale_rule,
                q_training_rows=len(qvalues),a_null_rows=len(avalues),q_counts=q_counts,a_counts=a_counts)


def predict_cdf(raw,query,data,target,owner,a_start):
    """Bとnullのスコアは同じArefに基づく。返すのは順位化前の六CDF。"""
    n=len(data['b']);values=raw[:n]
    qmask=(owner!=-1)&(owner!=target);q=data['query'].iloc[np.flatnonzero(qmask)]
    a=query.iloc[a_start:];qnull=raw[np.flatnonzero(qmask)];anull=raw[a_start:]
    qg=u.cal.empirical_cdf(values,qnull);ag=u.cal.empirical_cdf(values,anull)
    qc,nqc=u.cal.group_stats(values,data['b'],qnull,q,['prefecture','sex'])
    qp,nqp=u.cal.group_stats(values,data['b'],qnull,q,['sex','age_bin'])
    ac,nac=u.cal.group_stats(values,data['b'],anull,a,['prefecture','sex'])
    ap,nap=u.cal.group_stats(values,data['b'],anull,a,['sex','age_bin'])
    cdfs=np.column_stack([qg,qc,qp,ag,ac,ap]);out=np.empty((len(CONFIGS),n))
    for k,config in enumerate(CONFIGS):
        lq=nqc/(nqc+LOCAL);la=nac/(nac+LOCAL)
        pq=nqp/(nqp+PARENT) if config['shape']==SHAPES[1] else np.zeros(n)
        pa=nap/(nap+PARENT) if config['shape']==SHAPES[1] else np.zeros(n)
        weights=np.column_stack([config['q_weight']*(1-lq)*(1-pq),config['q_weight']*lq,config['q_weight']*(1-lq)*pq,
                                 config['a_weight']*(1-la)*(1-pa),config['a_weight']*la,config['a_weight']*(1-la)*pa])
        assert np.allclose(weights.sum(axis=1),1,rtol=0,atol=1e-15)
        out[k]=(cdfs*weights).sum(axis=1)
    # 固定flat Q CDFを既存group_stats/blendの別経路と照合。raw tieは加えない。
    error=float(np.max(np.abs(out[0]-u.cal.blend(qc,nqc,qg,LOCAL))))
    assert error==0,error
    assert np.isfinite(out).all()
    return out,dict(q_county_sex=nqc,q_sex_age=nqp,a_county_sex=nac,a_sex_age=nap),error


def run_phase(phase,out,qa,support_hashes):
    started=time.perf_counter();data=u.setup(phase)
    owner,_,duplicates=wide.pub.public_folds(data)
    assignment,parts=split_a(data)
    queries=[];a_start=len(data['query'])
    for fold in range(2):
        held=parts[1-fold][list(u.base.QI7)+['age_bin']]
        queries.append(pd.concat([data['query'][list(u.base.QI7)+['age_bin']],held],ignore_index=True))
    selection_targets=[t for t in data['teams'] if phase!='prelim' or t not in (1,2)]
    banks={c['name']:{} for c in CONFIGS};outer_rows=[];fold_rows=[];coverage=[];raw_qa=[]
    for index,(target,c) in enumerate(data['cs'].items(),1):
        raw=[];predictions=[]
        for fold in range(2):
            scores=u.base.core(parts[fold],queries[fold],c,u.NUM7,(10,))[10]
            assert np.isfinite(scores).all();raw.append(scores)
            pred,counts,flat_error=predict_cdf(scores,queries[fold],data,target,owner,a_start)
            predictions.append(pred)
            # 別の候補位置を単独queryとして再計算し、同一referenceを使うことを確認。
            ids=np.unique(np.r_[0,1048,a_start,len(scores)-1])
            independently=u.base.core(parts[fold],queries[fold].iloc[ids],c,u.NUM7,(10,))[10]
            source_error=float(np.max(np.abs(independently-scores[ids])));assert source_error==0,source_error
            raw_qa.append(dict(phase=phase,target=target,fold=fold,reference_rows=len(parts[fold]),
                               held_a_rows=len(parts[1-fold]),query_rows=len(queries[fold]),
                               gen_single_query_max_error=source_error,q_flat_cdf_max_error=flat_error))
            for (prefecture,sex,age_bin),positions in data['b'].loc[data['normal']].groupby(['prefecture','sex','age_bin'],sort=True).indices.items():
                normal_pos=np.flatnonzero(data['normal'])[positions]
                first=normal_pos[0]
                coverage.append(dict(phase=phase,target=target,fold=fold,prefecture=prefecture,sex=int(sex),age_bin=int(age_bin),
                                     candidate_rows=len(positions),q_county_sex_rows=int(counts['q_county_sex'][first]),
                                     q_sex_age_rows=int(counts['q_sex_age'][first]),a_county_sex_rows=int(counts['a_county_sex'][first]),
                                     a_sex_age_rows=int(counts['a_sex_age'][first]),
                                     q_county_weight=float(counts['q_county_sex'][first]/(counts['q_county_sex'][first]+LOCAL)),
                                     a_county_weight=float(counts['a_county_sex'][first]/(counts['a_county_sex'][first]+LOCAL)),
                                     membership_accuracy_unknown=True))
        average=(predictions[0]+predictions[1])/2
        for k,config in enumerate(CONFIGS): banks[config['name']][target]=u.rank(average[k],data['normal'])
        if target in selection_targets:
            for heldowner in sorted(data['qs']):
                if heldowner==target: continue
                cases=[evaluate_fold(raw[fold],queries[fold],data,target,owner,heldowner,a_start) for fold in range(2)]
                loss=(cases[0]['losses']+cases[1]['losses'])/2
                pit=(cases[0]['pits']+cases[1]['pits'])/2
                for k,config in enumerate(CONFIGS):
                    for fold,case in enumerate(cases):
                        fold_rows.append(dict(phase=phase,target=target,heldout_null_owner=heldowner,fold=fold,method=config['name'],
                                              crps=float(case['losses'][:,k].mean()),pit_mean=float(case['pits'][:,k].mean()),
                                              pit_above_99=float((case['pits'][:,k]>.99).mean()),pit_above_95=float((case['pits'][:,k]>.95).mean()),
                                              training_Q_rows=case['q_training_rows'],training_iqr_scale=case['scale'],scale_rule=case['scale_rule'],
                                              A_null_rows=case['a_null_rows'],heldout_rows=326,
                                              Q_county_support_min=int(case['q_counts'][:,0].min()),
                                              A_county_support_min=int(case['a_counts'][:,0].min()),
                                              Q_county_support_median=float(np.median(case['q_counts'][:,0])),
                                              A_county_support_median=float(np.median(case['a_counts'][:,0])),membership_labels_used=False))
                    outer_rows.append(dict(phase=phase,target=target,heldout_null_owner=heldowner,method=config['name'],
                                           crps=float(loss[:,k].mean()),pit_mean=float(pit[:,k].mean()),
                                           pit_above_99=float((pit[:,k]>.99).mean()),pit_above_95=float((pit[:,k]>.95).mean()),
                                           mean_fold_pit_above_99=float(np.mean([(case['pits'][:,k]>.99).mean() for case in cases])),
                                           mean_fold_pit_above_95=float(np.mean([(case['pits'][:,k]>.95).mean() for case in cases])),
                                           heldout_rows=326,folds=2,membership_labels_used=False,used_for_selection=False))
        u.log(stage='split_A_calibration_target',phase=phase,completed_targets=index,targets=len(data['teams']),
              diagnostic_targets=len(selection_targets),seconds=time.perf_counter()-started)
    matrices=u.matrices(banks,data['teams'],cross=(CROSS,))
    table=pd.DataFrame(outer_rows);fold_table=pd.DataFrame(fold_rows)
    metrics=['crps','pit_mean','pit_above_99','pit_above_95','mean_fold_pit_above_99','mean_fold_pit_above_95']
    owner_table=table.groupby(['phase','heldout_null_owner','method'],as_index=False)[metrics].mean()
    summary=owner_table.groupby(['phase','method'],as_index=False)[metrics].mean()
    summary['targets']=len(selection_targets);summary['owners']=len(data['qs'])
    summary['aggregation']='equal Q-owner weight after averaging eligible targets within owner'
    summary['used_for_selection']=False
    target_table=table.groupby(['phase','target','method'],as_index=False)[metrics].mean()
    drift=[];reference_name=CONFIGS[0]['name']+'__cross0.1';reference=matrices[reference_name]
    for name,mat in matrices.items():
        for j,target in enumerate(data['teams']):
            old=set(np.flatnonzero(rankdata(-reference[:,j],method='min')<=TOP))
            new=set(np.flatnonzero(rankdata(-mat[:,j],method='min')<=TOP))
            drift.append(dict(phase=phase,target=target,method=name,reference=reference_name,
                              reference_top_count=len(old),current_top_count=len(new),added=len(new-old),removed=len(old-new),
                              spearman=float(np.corrcoef(rankdata(reference[:,j]),rankdata(mat[:,j]))[0,1]),
                              membership_accuracy_unknown=True,reference_A='same two split references, not full-A baseline'))
    dest=out/phase;dest.mkdir()
    pd.DataFrame({'A_row_position':np.arange(len(assignment)),'reference_fold':assignment,'heldout_fold':1-assignment}).to_csv(dest/'Aの固定二分割.csv',index=False)
    fold_table.to_csv(dest/'fold別の課題所有者保留.csv',index=False,float_format='%.17g')
    table.to_csv(dest/'二fold集約の課題所有者保留.csv',index=False,float_format='%.17g')
    owner_table.to_csv(dest/'Q所有者等重みの診断パネル.csv',index=False,float_format='%.17g')
    summary.to_csv(dest/'固定六案の非会員診断.csv',index=False,float_format='%.17g')
    target_table.to_csv(dest/'対象別の固定六案診断.csv',index=False,float_format='%.17g')
    pd.DataFrame(coverage).to_csv(dest/'通常候補のQとAの支持範囲.csv',index=False,float_format='%.17g')
    pd.DataFrame(raw_qa).to_csv(dest/'同一参照の生スコア検証.csv',index=False,float_format='%.17g')
    pd.DataFrame(drift).to_csv(dest/'Qのみflatとの上位12変化.csv',index=False,float_format='%.17g')
    names=sorted(matrices);scores=np.stack([matrices[name] for name in names])
    ranks=np.stack([np.column_stack([rankdata(mat[:,j],method='average')/875 for j in range(len(data['teams']))]) for mat in scores])
    np.savez_compressed(dest/'通常875行の内部順位だけ.npz',names=np.array(names),teams=np.array(data['teams']),
                        ranks=ranks,cross_scores=scores,record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
    for path,hashed in u.base.INPUT_HASHES.items(): assert u.sha(path)==hashed,path
    for path,hashed in support_hashes.items(): assert u.sha(path)==hashed,path
    assert not any('pool_ids' in path or '正解データ' in path or 'participant_data/A_bg.csv' in path for path in u.base.INPUT_HASHES)
    record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,
                python=sys.executable,python_version=platform.python_version(),numpy_version=np.__version__,pandas_version=pd.__version__,
                code_path=str(Path(__file__).resolve()),code_sha256=u.sha(__file__),support_code_sha256=support_hashes,
                input_sha256=dict(u.base.INPUT_HASHES),output_sha256={p.name:u.sha(p) for p in dest.iterdir()},
                numerical_qa=qa,phase=phase,candidate_targets=data['teams'],diagnostic_targets=selection_targets,
                source_weight_configs=CONFIGS,A_reference_rows=[len(x) for x in parts],A_heldout_rows=[len(parts[1-f]) for f in range(2)],
                A_cross_fold_qi7_duplicates=0,A_each_row_heldout_once=True,public_Q_duplicates=duplicates,
                same_reference_gen_max_error=float(max(row['gen_single_query_max_error'] for row in raw_qa)),
                q_only_flat_cdf_max_error=float(max(row['q_flat_cdf_max_error'] for row in raw_qa)),
                no_gt_guard_installed=True,membership_labels_used=False,mia_scoring_performed=False,
                method_selection=False,coefficient_search=False,C_modified=False,old_or_kit_A_used=False,
                candidate_protocol='average two-fold CDF, then normal875 average rank, then other-target median0.1; no extra raw tie term',
                crps_protocol='per-fold exact six-CDF mixture in raw-score coordinates divided by outer-training Q-only IQR, then mean across folds',
                pit_protocol='primary PIT averages two fold CDF values before threshold; per-fold threshold averages also retained',
                aggregation='equal Q-owner weight after averaging eligible targets within owner; prelim excludes diagnostic targets1/2 only',
                known_nonmember_basis='phase A is outside every phase B under official rulebook p2; Q owner != target under p6/p9',
                limitations=['A and Q are distinct sampling strata; neither is guaranteed representative of normal875 candidate nonmembers.',
                             'Halving A changes the Gen-LRA reference and bandwidth; this is not a same-raw-score comparison with full-A baseline.',
                             'CRPS averages two normalized raw-coordinate losses; PIT averages fold CDF values.',
                             'County-sex and sex-age overlap and are not a nested partition.',
                             'No member discrimination or MIA performance is evaluated.'],final_prose_or_github_created=False)
    write_json(dest/'実行記録.json',record)
    u.log(stage='split_A_calibration_complete',phase=phase,seconds=record['seconds'],output=str(dest),
          cases=len(table),fold_cases=len(fold_table),same_reference_gen_max_error=record['same_reference_gen_max_error'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=u.HERE/'公開背景の分割較正結果')
    args=parser.parse_args()
    assert not args.output_dir.exists(),args.output_dir
    assert sys.dont_write_bytecode and Path(sys.executable).absolute()==u.ROOT/'.venv/bin/python',sys.executable
    support_hashes=loaded_project_sources();qa=numerical_qa()
    assert any('非会員を使う順位較正の検討.py' in path for path in support_hashes)
    args.output_dir.mkdir()
    plan=dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),code_sha256=u.sha(__file__),support_code_sha256=support_hashes,
              phases=['prelim','main'],seed=SEED,A_split='permutation first2571 versus remaining2572, preserve original order within each half',
              folds='two swapped reference/null halves; every A row held out once',only_current_phase_A=True,C_modified=False,
              score='fixed existing Gen-LRA7 k10; B+all Q+held A scored with the same fold reference',
              age_bins='fixed full phase A quantiles20/40/60/80%; same edges for both folds and all six methods',
              configs=CONFIGS,local_strength=LOCAL,parent_strength=PARENT,cross_median=CROSS,
              outer_holdout='Q-null excludes target owner and held owner; A-null is reference-disjoint held A',
              scale='each fold uses outer-training Q-null IQR only for all six methods including A-only',
              prediction='average fold CDF, normal875 rank, then cross median0.1; no extra raw tie coefficient',
              aggregation='pre18/main22 diagnostics, equal Q owners after averaging eligible targets within each owner',
              crps='exact six-CDF mixture, five differences; mean fold losses',pit='average fold CDF then threshold; also report mean fold thresholds',
              numerical_qa=qa,no_gt_guard_installed=True,membership_labels_used=False,mia_scoring=False,
              method_selection=False,coefficient_search=False,outputs='only local normal875 internal ranks and diagnostic tables',
              no_submission_or_final_prose=True)
    write_json(args.output_dir/'事前固定条件.json',plan)
    u.log(stage='split_A_plan_fixed',code_sha256=plan['code_sha256'],output=str(args.output_dir),support_sources=len(support_hashes),numerical_qa=qa)
    for phase in ('prelim','main'): run_phase(phase,args.output_dir,qa,support_hashes)
    write_json(args.output_dir/'完了記録.json',dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                                                 phases_completed=['prelim','main'],code_sha256=u.sha(__file__),
                                                 membership_labels_used=False,mia_scoring=False,method_selection=False,
                                                 final_prose_or_github_created=False))


if __name__=='__main__': main()
