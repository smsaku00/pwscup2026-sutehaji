"""固定Gaussian三形の、同じ公開A二分割seedに対する順位感度。

元seed20261008の全対象pipelineを先に完全再現し、固定四seed09/10/11/12を両phaseへ共通適用。
所属正解/TPR/形・seed選択なし。個人CSV/IDなし、内部順位NPZはPro/packetへの添付対象外。
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
import scipy
import sklearn
from scipy.stats import rankdata

import 共通 as u

gauss=u.module('frozen_gaussian_seed_sensitivity',u.HERE/'共有中順位座標の密度比.py')
BASE_SEED=20261008
PERTURBATIONS=(20261009,20261010,20261011,20261012)
FORMS=gauss.FORMS
BASE_OUT=u.HERE/'共有中順位座標の密度比結果'


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


def split_seed(data,seed):
    """元split_aと同じ手順をここで明示。seedだけを変更し、Aを再利用する。"""
    a=data['a'];assert len(a)==5143
    order=np.random.default_rng(seed).permutation(len(a))
    assignment=np.empty(len(a),int)
    assignment[order[:len(a)//2]]=0;assignment[order[len(a)//2:]]=1
    parts=[a.iloc[np.flatnonzero(assignment==fold)].reset_index(drop=True) for fold in range(2)]
    overlap=pd.MultiIndex.from_frame(parts[0][list(u.base.QI7)]).isin(pd.MultiIndex.from_frame(parts[1][list(u.base.QI7)]))
    assert not overlap.any() and np.bincount(assignment).tolist()==[2571,2572]
    return assignment,parts


def source_diagnostics(source,context,fitrows,phase,seed,fold,target,kind,primary):
    result=[]
    for sex,model in sorted(context.items()):
        part=source[source.sex==sex]
        z=(u.base.numbers(part,u.NUM7)-model['center'])/model['scale']
        outside=np.column_stack([(z[:,j]<mapping['unique'][0])|(z[:,j]>mapping['unique'][-1])
                                 for j,mapping in enumerate(model['maps'])])
        for row in fitrows:
            if row['sex']!=sex:continue
            probit=row['form'].startswith('probit_')
            result.append(dict(phase=phase,seed=seed,fold=fold,target=target,source=kind,sex=sex,form=row['form'],
                               source_rows=len(part),primary_target=primary,numeric_dimensions=5,
                               floored_dimensions=row['floored_dimensions'],numerical_floor=row['numerical_floor'],
                               unfloored_min=row['unfloored_min'],shrinkage=row['shrinkage'],
                               probit_map_used=probit,
                               clamped_coordinate_cells=int(outside.sum()) if probit else 0,
                               rows_with_any_clamped_coordinate=int(outside.any(axis=1).sum()) if probit else 0,
                               coordinate_cell_denominator=len(part)*5,
                               clamp_is_strict_outside_Aref_support=True,counts_are_repeated_source_exposures=True))
    return result


def run_seed(data,seed):
    started=time.perf_counter();phase=data['phase'];owner,_,duplicates=gauss.split.wide.pub.public_folds(data)
    assignment,parts=split_seed(data,seed);a_start=len(data['query'])
    queries=[pd.concat([data['query'][list(u.base.QI7)+['age_bin']],parts[1-f][list(u.base.QI7)+['age_bin']]],ignore_index=True) for f in range(2)]
    contexts=[gauss.contexts(part) for part in parts];background=[];fitted_background=[];fits=[];rawchecks=[]
    for fold in range(2):
        fitted,rows=gauss.fit_source(parts[fold],contexts[fold]);fitted_background.append(fitted)
        background.append(gauss.density(fitted,queries[fold],contexts[fold]))
        fits.extend(source_diagnostics(parts[fold],contexts[fold],rows,phase,seed,fold,-1,'Aref',True))
    banks={form:{} for form in FORMS}
    for target,c in data['cs'].items():
        predictions=[]
        primary=phase!='prelim' or target not in (1,2)
        for fold in range(2):
            fitted,rows=gauss.fit_source(c,contexts[fold]);values=gauss.density(fitted,queries[fold],contexts[fold]);raw=values-background[fold]
            fits.extend(source_diagnostics(c,contexts[fold],rows,phase,seed,fold,target,'C',primary))
            ids=np.unique(np.r_[0,1048,a_start,len(queries[fold])-1])
            single=gauss.density(fitted,queries[fold].iloc[ids],contexts[fold])-gauss.density(fitted_background[fold],queries[fold].iloc[ids],contexts[fold])
            error=float(np.max(np.abs(single-raw[:,ids])));assert error<1e-10
            rawchecks.append(dict(phase=phase,seed=seed,target=target,fold=fold,single_query_raw_max_error=error,
                                  all_densities_and_raw_finite=True,primary_target=primary))
            predictions.append(np.stack([gauss.split.predict_cdf(raw[k],queries[fold],data,target,owner,a_start)[0][gauss.CAL_INDEX] for k in range(len(FORMS))]))
        average=(predictions[0]+predictions[1])/2
        for k,form in enumerate(FORMS):banks[form][target]=u.rank(average[k],data['normal'])
    matrices=u.matrices(banks,data['teams'],cross=(.1,))
    u.log(stage='gaussian_seed_complete',phase=phase,seed=seed,seconds=time.perf_counter()-started,
          targets=len(data['teams']),single_query_max_error=max(row['single_query_raw_max_error'] for row in rawchecks))
    return dict(matrices=matrices,fits=fits,rawchecks=rawchecks,assignment=assignment,
                public_Q_duplicates=duplicates,seconds=time.perf_counter()-started)


def heads_and_shifts(matrices,baseline,teams,phase,seed):
    rows=[]
    for name,values in matrices.items():
        form=name.removesuffix('__cross0.1');original=baseline[name]
        for j,target in enumerate(teams):
            oldrank=rankdata(original[:,j],method='average');newrank=rankdata(values[:,j],method='average')
            old=set(np.flatnonzero(rankdata(-original[:,j],method='min')<=12))
            new=set(np.flatnonzero(rankdata(-values[:,j],method='min')<=12))
            common=len(old&new);union=len(old|new);shift=float(np.mean(np.abs(newrank-oldrank)))
            rows.append(dict(phase=phase,seed=seed,form=form,target=target,primary_target=phase!='prelim' or target not in (1,2),
                             normal_rows=875,baseline_head_count=len(old),perturbed_head_count=len(new),
                             head_intersection=common,head_union=union,head_jaccard=common/union,
                             added=len(new-old),removed=len(old-new),
                             spearman=float(np.corrcoef(oldrank,newrank)[0,1]),
                             mean_absrankshift_positions=shift,mean_absrankshift_fraction=shift/875,
                             membership_accuracy_unknown=True,used_for_seed_or_form_selection=False))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'共有GaussianのA分割seed感度結果')
    args=parser.parse_args();out=args.output_dir;assert not out.exists()
    assert sys.dont_write_bytecode and Path(sys.executable).absolute()==u.ROOT/'.venv/bin/python'
    started=time.perf_counter();code_hash=u.sha(__file__);support_hashes=loaded_sources()
    datasets={phase:u.setup(phase) for phase in ('prelim','main')}
    inputs=dict(u.base.INPUT_HASHES);frozen={};baselines={};baseline_ranks={}
    for phase,data in datasets.items():
        for name in ('実行記録.json','通常875行の内部順位だけ.npz'):
            path=BASE_OUT/phase/name;frozen[str(path)]=u.sha(path)
        record=json.loads((BASE_OUT/phase/'実行記録.json').read_text())
        assert record['code_sha256']==u.sha(gauss.__file__)
        for path,expected in record['input_sha256'].items():assert u.sha(path)==expected
        for path,expected in record['support_code_sha256'].items():assert u.sha(path)==expected
        assert u.sha(BASE_OUT/phase/'通常875行の内部順位だけ.npz')==record['output_sha256']['通常875行の内部順位だけ.npz']
        with np.load(BASE_OUT/phase/'通常875行の内部順位だけ.npz') as saved:
            assert np.array_equal(saved['teams'],data['teams'])
            baselines[phase]={str(name):array.copy() for name,array in zip(saved['names'],saved['cross_scores'])}
            baseline_ranks[phase]={str(name):array.copy() for name,array in zip(saved['names'],saved['ranks'])}
    out.mkdir()
    plan=dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),code_sha256=code_hash,support_code_sha256=support_hashes,
              input_sha256=inputs,frozen_original_sha256=frozen,base_seed=BASE_SEED,perturbation_seeds=PERTURBATIONS,
              same_seeds_both_phases=True,forms=FORMS,calibration=gauss.split.CONFIGS[gauss.CAL_INDEX],
              explicit_split='default_rng(seed).permutation(A5143); first2571 referencefold0, rest2572 referencefold1; swap reference/null',
              source_formulas_unchanged=True,baseline_barrier='full pipeline cross_scores and ranks must match saved seed08 for both phases before any perturbation',
              primary_targets='prelim excludes1/2, main all22; internal arrays retain all20/22',
              head_rule='descending minrank<=12, include ties; head size may exceed12',
              shift_rule='average absolute difference of average ranks of final cross0.1 scores, in positions and divided by875',
              intersection_rule='both four perturbations alone and allfive including baseline are stored',
              clamp_rule='source C/Aref coordinate strictly outside own fold Aref support; only probit forms clamp; endpoint-equal anchors not counted',
              independent_cohorts=False,reference_and_null_reuse_same_A=True,
              limitations=['Changing the split perturbs Aref coordinates, reference density fit and held-A null jointly; this does not isolate calibration alone.',
                           'All seeds reuse the same A/C/Q/B cohort data. Seeds are perturbations, not independent cohort validation.',
                           'Rank stability is sensitivity of unknown-membership rankings, not membership accuracy or candidate FPR.',
                           'No seed/form winner is chosen and no CRPS/TPR or membership labels are used.'],
              internal_npz='never attach to root manifest/Pro packet; no record IDs are saved',
              csv_outputs_are_aggregates_only=True,membership_labels_used=False,mia_scoring=False,method_or_seed_selection=False,
              final_prose_or_github_created=False)
    write_json(out/'事前固定条件.json',plan)
    runs={};baseline_qa=[];all_fits=[];all_raw=[]
    # Explicit barrier: both seed08 phases finish and match before any seed09-12 work.
    for phase,data in datasets.items():
        run=run_seed(data,BASE_SEED);runs[phase,BASE_SEED]=run
        assert set(run['matrices'])==set(baselines[phase])
        score_error=max(float(np.max(np.abs(values-baselines[phase][name]))) for name,values in run['matrices'].items())
        rank_errors=[]
        for name,values in run['matrices'].items():
            computed=np.column_stack([rankdata(values[:,j],method='average')/875 for j in range(len(data['teams']))])
            rank_errors.append(float(np.max(np.abs(computed-baseline_ranks[phase][name]))))
        rank_error=max(rank_errors)
        original_assignment,_=gauss.split.split_a(data);assert np.array_equal(original_assignment,run['assignment'])
        assert score_error==0 and rank_error==0,(phase,score_error,rank_error)
        baseline_qa.append(dict(phase=phase,seed=BASE_SEED,score_max_error=score_error,rank_max_error=rank_error,
                                rank_values=3*875*len(data['teams']),A_assignment_equal=True,full_density_cdf_rank_cross_pipeline=True))
        all_fits.extend(run['fits']);all_raw.extend(run['rawchecks'])
    write_json(out/'元seed08の全順位再現QA.json',dict(checks=baseline_qa,all_score_and_rank_errors_zero=True,
                                                    barrier_before_any_perturbation=True,individual_rows_exported=False))
    for seed in PERTURBATIONS:
        for phase,data in datasets.items():
            run=run_seed(data,seed);runs[phase,seed]=run;all_fits.extend(run['fits']);all_raw.extend(run['rawchecks'])
    comparisons=[];intersections=[]
    for phase,data in datasets.items():
        for seed in PERTURBATIONS:comparisons.extend(heads_and_shifts(runs[phase,seed]['matrices'],baselines[phase],data['teams'],phase,seed))
        for name in baselines[phase]:
            form=name.removesuffix('__cross0.1')
            for j,target in enumerate(data['teams']):
                heads=[set(np.flatnonzero(rankdata(-runs[phase,seed]['matrices'][name][:,j],method='min')<=12)) for seed in (BASE_SEED,)+PERTURBATIONS]
                intersections.append(dict(phase=phase,form=form,target=target,primary_target=phase!='prelim' or target not in (1,2),
                                          baseline_head_count=len(heads[0]),perturbation_head_count_min=min(map(len,heads[1:])),
                                          perturbation_head_count_max=max(map(len,heads[1:])),
                                          four_seed_head_intersection=len(set.intersection(*heads[1:])),
                                          four_seed_head_union=len(set.union(*heads[1:])),
                                          baseline_plus_four_head_intersection=len(set.intersection(*heads)),
                                          baseline_plus_four_head_union=len(set.union(*heads)),
                                          membership_accuracy_unknown=True,used_for_selection=False))
    comparison=pd.DataFrame(comparisons);intersection=pd.DataFrame(intersections);fit_table=pd.DataFrame(all_fits)
    metrics=['baseline_head_count','perturbed_head_count','head_intersection','head_union','head_jaccard','added','removed','spearman',
             'mean_absrankshift_positions','mean_absrankshift_fraction']
    primary=comparison[comparison.primary_target]
    perseed=primary.groupby(['phase','form','seed'],as_index=False)[metrics].mean()
    summary=primary.groupby(['phase','form'],as_index=False)[metrics].mean()
    summaries=intersection[intersection.primary_target].groupby(['phase','form'],as_index=False)[[
        'four_seed_head_intersection','four_seed_head_union','baseline_plus_four_head_intersection','baseline_plus_four_head_union']].mean()
    summary=summary.merge(summaries,on=['phase','form']);summary['seeds']=4;summary['membership_accuracy_unknown']=True;summary['used_for_selection']=False
    for table,name in [(comparison,'対象別seed順位感度.csv'),(intersection,'四seedと元seedを含む上位集合の共通部分.csv'),
                       (perseed,'primary対象平均のseed別感度.csv'),(summary,'固定三形の分割感度集約.csv'),
                       (fit_table,'sourceGaussianのfloorとclamp集計.csv'),(pd.DataFrame(all_raw),'単独query一致の集約検証.csv')]:
        table.to_csv(out/name,index=False,float_format='%.17g')
    cfits=fit_table[(fit_table.source=='C')&fit_table.primary_target]
    fit_summary=cfits.groupby(['phase','seed','form'],as_index=False).agg(
        source_fit_cases=('source_rows','size'),source_row_exposures=('source_rows','sum'),
        coordinate_cell_exposures=('coordinate_cell_denominator','sum'),floor_dimensions=('floored_dimensions','sum'),
        clamped_coordinate_cells=('clamped_coordinate_cells','sum'),source_row_exposures_with_any_clamp=('rows_with_any_clamped_coordinate','sum'))
    fit_summary['clamped_coordinate_cell_fraction']=fit_summary.clamped_coordinate_cells/fit_summary.coordinate_cell_exposures
    fit_summary['counts_are_repeated_source_exposures']=True
    fit_summary.to_csv(out/'primaryCsourceのfloorとclamp集約.csv',index=False,float_format='%.17g')
    for phase,data in datasets.items():
        names=sorted(baselines[phase]);seeds=(BASE_SEED,)+PERTURBATIONS
        scores=np.stack([np.stack([runs[phase,seed]['matrices'][name] for name in names]) for seed in seeds])
        ranks=np.stack([np.stack([np.column_stack([rankdata(mat[:,j],method='average')/875 for j in range(len(data['teams']))]) for mat in byform]) for byform in scores])
        np.savez_compressed(out/(phase+'_内部875順位_外部添付禁止.npz'),seeds=np.array(seeds),names=np.array(names),teams=np.array(data['teams']),scores=scores,ranks=ranks)
    for path,expected in inputs.items():assert u.sha(path)==expected
    for path,expected in support_hashes.items():assert u.sha(path)==expected
    for path,expected in frozen.items():assert u.sha(path)==expected
    assert u.sha(__file__)==code_hash
    assert not any('pool_ids' in path or '正解データ' in path for path in inputs)
    record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,
                code_path=str(Path(__file__).resolve()),code_sha256=code_hash,support_code_sha256=support_hashes,
                input_sha256=inputs,frozen_original_sha256=frozen,output_sha256={p.name:u.sha(p) for p in out.iterdir()},
                python=sys.executable,python_version=platform.python_version(),numpy_version=np.__version__,pandas_version=pd.__version__,
                scipy_version=scipy.__version__,sklearn_version=sklearn.__version__,seeds=(BASE_SEED,)+PERTURBATIONS,
                baseline_max_score_error=max(row['score_max_error'] for row in baseline_qa),baseline_max_rank_error=max(row['rank_max_error'] for row in baseline_qa),
                baseline_rank_values=sum(row['rank_values'] for row in baseline_qa),candidate_targets={p:d['teams'] for p,d in datasets.items()},
                primary_target_counts={'prelim':18,'main':22},source_fit_records=len(fit_table),floor_dimensions=int(fit_table.floored_dimensions.sum()),
                single_query_max_error=float(max(row['single_query_raw_max_error'] for row in all_raw)),
                original_artifacts_unchanged=True,csv_individual_rows_exported=False,record_ids_exported=False,
                internal_npz_excluded_from_root_manifest_and_pro_packet=True,
                same_A_cohort_reused=True,independent_cohorts=False,accuracy_claim=False,
                no_CRPS_recomputed=True,membership_labels_used=False,mia_scoring=False,method_or_seed_selection=False,
                final_prose_or_github_created=False)
    write_json(out/'実行記録.json',record)
    u.log(stage='gaussian_seed_sensitivity_completed',seconds=record['seconds'],output=str(out),baseline_rank_values=record['baseline_rank_values'],
          score_error=record['baseline_max_score_error'],rank_error=record['baseline_max_rank_error'])


if __name__=='__main__':main()
