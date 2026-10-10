"""Frozen public-A internal random-allocation diagnostics; no B/Q/GT or individual output."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

SCRIPT = Path(__file__).resolve()
HERE = SCRIPT.parent
ROOT = HERE.parents[3]
OUT = HERE / '公開Aの有限標本p値結果'
ALLOWED_CSV = set()


def public_only_guard(event, args):
    if event != 'open' or not args or isinstance(args[0], int):
        return
    p = Path(str(args[0])).resolve()
    name = unicodedata.normalize('NFC', str(p))
    forbidden = ('B_self.csv', 'B_practice.csv', 'aia_challenge_', 'pool_ids', '/正解データ/',
                 '/アーカイブ/', '.npz', 'F_mia.csv', 'F_aia.csv', 'attack_score.py', '対象別TPR.csv', 'result.txt')
    if any(s in name for s in forbidden):
        raise PermissionError(f'public A/C only: forbidden path {p}')
    if p.suffix == '.csv' and p.is_relative_to(ROOT) and p not in ALLOWED_CSV and not p.is_relative_to(OUT):
        raise PermissionError(f'CSV not in public-A/C allowlist: {p}')


sys.addaudithook(public_only_guard)
import 公開背景三分割の独立非会員診断 as frozen

u, gauss = frozen.u, frozen.gauss
SEED = 20261008
FORMS = ('gen7k10', 'shared_probit_gaussian_lw')
CONFIGS = (('global', ()), ('sex_fullA_age_bin', ('sex', 'age_bin')),
           ('prefecture_sex', ('prefecture', 'sex')))
ALPHAS = (0.005, 0.01, 0.02, 0.05)
FROZEN_A3_SHA = 'f85daa5f24af45951b943441e9a0db9b4e37a12aca670a3f9f407e5a687d4f51'
CSV_NAMES = ('群別の較正標本数と閾値到達可否.csv', '対象fold別の有限標本p値診断.csv',
             'fold別の有限標本p値診断.csv', '三fold集約の有限標本p値診断.csv', '公開生スコアとCounterの数値QA.csv')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write('\n')


def phase_paths(phase):
    distributed = ROOT / ('予備戦' if phase == 'prelim' else '本戦') / '配布データ'
    if phase == 'prelim':
        a = distributed / 'PWSCup2026_prelim_common/A_bg.csv'
        targets = distributed / 'PWSCup2026_prelim_attack'
        self_id = 5
    else:
        own = next(distributed.glob('PWSCup2026_final_team28_*'))
        a = own / 'A_bg.csv'
        targets = HERE.parent / '20260928_配布データ基本分析/入力'
        self_id = 28
    cs = {int(p.stem.split('_')[-1]): p.resolve() for p in sorted(targets.glob('C_*.csv'))
          if int(p.stem.split('_')[-1]) != self_id}
    assert len(cs) == (20 if phase == 'prelim' else 22)
    return a.resolve(), cs


PHASE_PATHS = {phase: phase_paths(phase) for phase in ('prelim', 'main')}
for a, cs in PHASE_PATHS.values():
    ALLOWED_CSV.update([a, *cs.values()])


def support_hashes():
    paths = set()
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if not path:
            continue
        p = Path(path).resolve()
        if p.suffix == '.py' and p.is_relative_to(ROOT / '本戦') and p != SCRIPT:
            paths.add(p)
    return {str(p): sha(p) for p in sorted(paths)}


def split_public_a(a):
    assert len(a) == 5143
    order = np.random.default_rng(SEED).permutation(len(a))
    assignment = np.empty(len(a), dtype=int)
    assignment[order[:1714]] = 0
    assignment[order[1714:3428]] = 1
    assignment[order[3428:]] = 2
    parts = [a.iloc[np.flatnonzero(assignment == k)].reset_index(drop=True) for k in range(3)]
    old_assignment, old_parts = frozen.partition_a(a)
    assert np.array_equal(assignment, old_assignment)
    assert all(p.equals(q) for p, q in zip(parts, old_parts))
    return assignment, parts


def grouped_positions(frame, columns):
    if not columns:
        return {(): np.arange(len(frame), dtype=int)}
    return {tuple(key) if isinstance(key, tuple) else (key,): np.asarray(pos, dtype=int)
            for key, pos in frame.groupby(list(columns), sort=True).indices.items()}


def upper_counts(cal_scores, test_scores):
    """Integer conservative upper ranks: #cal >= query; empty calibration yields numerator1/denominator1."""
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    query = np.asarray(test_scores, dtype=float)
    assert np.isfinite(cal).all() and np.isfinite(query).all()
    numerator = 1 + len(cal) - np.searchsorted(cal, query, side='left')
    denominator = np.full(len(query), len(cal) + 1, dtype=int)
    return numerator.astype(int), denominator


def counter_counts(cal_scores, test_scores):
    """Independent Counter multiplicities and Python bisect, including ties and empty groups."""
    counts = Counter(float(x) for x in cal_scores)
    keys = sorted(counts)
    suffix = [0] * (len(keys) + 1)
    for j in range(len(keys) - 1, -1, -1):
        suffix[j] = suffix[j + 1] + counts[keys[j]]
    numerator = np.array([1 + suffix[bisect_left(keys, float(x))] for x in test_scores], dtype=int)
    return numerator, np.full(len(numerator), len(cal_scores) + 1, dtype=int)


def reject(numerator, denominator, alpha):
    a = Fraction(str(alpha))
    # Exact rational threshold comparison, avoiding p rounding at <= alpha boundaries.
    return numerator * a.denominator <= denominator * a.numerator


def p_toy_qa():
    cal, query = np.array([1., 1., 3.]), np.array([0., 1., 2., 3., 4.])
    num, den = upper_counts(cal, query)
    assert np.array_equal(num, [4, 4, 2, 2, 1]) and np.array_equal(den, [4] * 5)
    alt = counter_counts(cal, query)
    assert np.array_equal(num, alt[0]) and np.array_equal(den, alt[1])
    empty = upper_counts([], query)
    assert np.array_equal(empty[0], np.ones(5, int)) and np.array_equal(empty[1], np.ones(5, int))
    assert not reject(*empty, 0.05).any()
    assert reject(np.array([1, 1, 1, 1]), np.array([199, 200, 99, 100]), .005).tolist() == [False, True, False, False]
    # Full finite-pool random allocation: every query and every calibration subset from remaining rows.
    population = [0., 0., 1., 2.]
    probabilities = []
    for qidx in range(len(population)):
        other = [j for j in range(len(population)) if j != qidx]
        for ids in itertools.combinations(other, 2):
            n, d = upper_counts([population[j] for j in ids], [population[qidx]])
            probabilities.append(Fraction(int(n[0]), int(d[0])))
    for alpha in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(1)):
        assert Fraction(sum(p <= alpha for p in probabilities), len(probabilities)) <= alpha
    return dict(ties_ge_conservative=True,empty_group_p_one=True,Counter_searchsorted_max_error=0,
                exact_threshold_boundary=True,finite_pool_tied_allocation_cases=len(probabilities),
                finite_pool_allocation_superuniform_checks=6)


def independent_raw_qa(reference, source, query, raw):
    """Public-only independent Gen formula and scipy Gaussian normalizing formula."""
    scale = u.base.scales(reference, u.NUM7)
    r, c, q = [u.base.numbers(f, u.NUM7) / scale for f in (reference, source, query)]
    h = (4 / (len(u.NUM7) + 2) / len(r)) ** (1 / (len(u.NUM7) + 4))
    cr = np.sum((c[:, None, :] - r[None, :, :]) ** 2, axis=2)
    cq = np.sum((c[:, None, :] - q[None, :, :]) ** 2, axis=2)
    distance = cq.T.copy()
    log_cr, log_cq = -.5 * cr / h**2, -.5 * cq / h**2
    for column in ('sex', 'prefecture'):
        levels = sorted(set(reference[column]) | set(source[column]) | set(query[column]))
        freq = (reference[column].value_counts().reindex(levels, fill_value=0) + 1) / (len(reference) + len(levels))
        cf = source[column].map(freq).to_numpy()[:, None]
        eq_r = source[column].to_numpy()[:, None] == reference[column].to_numpy()[None, :]
        eq_q = source[column].to_numpy()[:, None] == query[column].to_numpy()[None, :]
        log_cr += np.log(.98 * eq_r + .02 * cf)
        log_cq += np.log(.98 * eq_q + .02 * cf)
        distance += ~eq_q.T
    delta = (np.logaddexp(0., log_cq - logsumexp(log_cr, axis=1)[:, None]) - np.log1p(1 / len(r))).T
    nearest = np.argsort(distance, axis=1, kind='stable')[:, :10]
    expected_gen = np.take_along_axis(delta, nearest, axis=1).sum(axis=1)
    gen_error = float(np.max(np.abs(expected_gen - raw[0])))
    coordinate = gauss.contexts(reference)
    afit, _ = gauss.fit_source(reference, coordinate)
    cfit, _ = gauss.fit_source(source, coordinate)
    def scipy_density(fitted):
        result = np.full(len(query), np.nan)
        for sex, context in coordinate.items():
            positions = np.flatnonzero(query.sex.to_numpy() == sex)
            frame = query.iloc[positions]
            z = (u.base.numbers(frame, u.NUM7) - context['center']) / context['scale']
            transformed = gauss.map_probit(z, context['maps'])
            model = fitted[sex]['models'][1]
            numeric = multivariate_normal.logpdf(transformed, mean=model['mean'], cov=model['covariance'])
            cat = .98 * frame.prefecture.map(fitted[sex]['pref']).fillna(0).to_numpy() + .02 * frame.prefecture.map(context['freq']).to_numpy()
            result[positions] = numeric + np.log(cat)
        return result
    gaussian_error = float(np.max(np.abs((scipy_density(cfit) - scipy_density(afit)) - raw[1])))
    assert gen_error < 1e-8 and gaussian_error < 1e-8
    return gen_error, gaussian_error


def phase_run(phase, plan):
    start = time.perf_counter()
    apath, cpaths = PHASE_PATHS[phase]
    a = pd.read_csv(apath, usecols=list(u.base.QI7), float_precision='round_trip')
    assert len(a) == 5143 and not a.duplicated(list(u.base.QI7)).any()
    cuts = np.unique(np.quantile(a.age, (.2, .4, .6, .8)))
    a['age_bin'] = np.searchsorted(cuts, a.age, side='right')
    assignment, parts = split_public_a(a)
    primary = sorted(t for t in cpaths if phase != 'prelim' or t not in (1, 2))
    sources = {t: pd.read_csv(cpaths[t], usecols=list(u.base.QI7), float_precision='round_trip') for t in primary}
    assert all(len(c) == 1049 for c in sources.values())
    support, cases, controls = [], [], []
    for fold in range(3):
        reference, cal, test = parts[fold], parts[(fold + 1) % 3], parts[(fold + 2) % 3]
        query = pd.concat([cal, test], ignore_index=True)
        coordinate = gauss.contexts(reference)
        afit, _ = gauss.fit_source(reference, coordinate)
        adensity = gauss.density(afit, query, coordinate)[1]
        grouped = {}
        for config, columns in CONFIGS:
            cgroups, tgroups = grouped_positions(cal, columns), grouped_positions(test, columns)
            grouped[config] = (cgroups, tgroups)
            for key in sorted(set(cgroups) | set(tgroups)):
                nc, nt = len(cgroups.get(key, ())), len(tgroups.get(key, ()))
                row = dict(phase=phase,fold=fold,config=config,
                           group='ALL' if not columns else ';'.join(f'{col}={val}' for col,val in zip(columns,key)),
                           calibration_group_rows=nc,test_group_rows=nt,minimum_possible_p=1/(nc+1),
                           empty_calibration_group=(nc==0),reference_rows=len(reference),calibration_rows=len(cal),test_rows=len(test))
                for alpha in ALPHAS:
                    row[f'can_reach_alpha_{alpha:g}'] = bool(reject(np.array([1]),np.array([nc+1]),alpha)[0])
                support.append(row)
        for target, source in sources.items():
            fitted, _ = gauss.fit_source(source, coordinate)
            raws = (u.base.core(reference,query,source,u.NUM7,(10,))[10],
                    gauss.density(fitted,query,coordinate)[1]-adensity)
            ids = np.unique([0,len(cal)-1,len(cal),len(query)-1])
            small = query.iloc[ids]
            singles = (frozen.u.base.core(reference,small,source,u.NUM7,(10,))[10],
                       frozen.gauss.density(fitted,small,coordinate)[1]-frozen.gauss.density(afit,small,coordinate)[1])
            errors = [float(np.max(np.abs(raw[ids]-smallraw))) for raw,smallraw in zip(raws,singles)]
            assert max(errors)<1e-10
            independent = independent_raw_qa(reference,source,small,singles) if fold==0 and target==(10 if phase=='prelim' else 0) else (None,None)
            for form, raw, single_error, independent_error in zip(FORMS,raws,errors,independent):
                assert np.isfinite(raw).all()
                cscore,tvalue = raw[:len(cal)],raw[len(cal):]
                counter_error = 0
                for config,_ in CONFIGS:
                    cgroups,tgroups = grouped[config]
                    numerator = np.zeros(len(test),int)
                    denominator = np.zeros(len(test),int)
                    for key,positions in tgroups.items():
                        sample = cscore[cgroups.get(key,np.array([],int))]
                        n,d = upper_counts(sample,tvalue[positions])
                        alt_n,alt_d = counter_counts(sample,tvalue[positions])
                        counter_error=max(counter_error,int(np.max(np.abs(n-alt_n))),int(np.max(np.abs(d-alt_d))))
                        assert np.array_equal(n,alt_n) and np.array_equal(d,alt_d)
                        numerator[positions],denominator[positions]=n,d
                    assert (numerator>=1).all() and (numerator<=denominator).all()
                    empty=int((denominator==1).sum())
                    for alpha in ALPHAS:
                        reached=reject(numerator,denominator,alpha)
                        resolved=reject(np.ones(len(test),int),denominator,alpha)
                        cases.append(dict(phase=phase,fold=fold,target=target,form=form,config=config,alpha=alpha,
                            reference_rows=len(reference),calibration_rows=len(cal),test_rows=len(test),
                            p_le_alpha_rows=int(reached.sum()),p_le_alpha_fraction=float(reached.mean()),
                            empty_calibration_query_rows=empty,zero_information_fraction=empty/len(test),
                            unresolvable_query_rows=int((~resolved).sum()),unresolvable_query_fraction=float((~resolved).mean()),
                            calibration_group_n_min=int(denominator.min()-1),calibration_group_n_median=float(np.median(denominator-1)),
                            minimum_possible_p_min=float((1/denominator).min()),minimum_possible_p_max=float((1/denominator).max()),
                            minimum_possible_p_query_mean=float((1/denominator).mean()),official_candidate_FPR=False,
                            internal_public_pool_random_allocation_diagnostic=True,used_for_selection=False))
                controls.append(dict(phase=phase,fold=fold,target=target,form=form,
                    original_A3_helper_single_query_max_error=single_error,
                    independent_original_score_formula_max_error=independent_error,
                    Counter_searchsorted_integer_max_error=counter_error,all_raw_scores_finite=True,
                    raw_min=float(raw.min()),raw_max=float(raw.max()),reference_cal_test_disjoint=True))
        print(json.dumps(dict(stage='finite_public_A_fold_complete',phase=phase,fold=fold,seconds=time.perf_counter()-start)),flush=True)
    dest=OUT/phase
    dest.mkdir()
    table=pd.DataFrame(cases)
    foldmean=table.groupby(['phase','fold','form','config','alpha'],as_index=False).agg(
        equal_target_p_le_alpha_fraction=('p_le_alpha_fraction','mean'),
        equal_target_zero_information_fraction=('zero_information_fraction','mean'),
        equal_target_unresolvable_fraction=('unresolvable_query_fraction','mean'),
        test_target_exposures=('test_rows','sum'),p_le_alpha_exposures=('p_le_alpha_rows','sum'))
    foldmean['primary_targets']=len(primary)
    phase_mean=foldmean.groupby(['phase','form','config','alpha'],as_index=False).agg(
        equal_fold_equal_target_p_le_alpha_fraction=('equal_target_p_le_alpha_fraction','mean'),
        equal_fold_zero_information_fraction=('equal_target_zero_information_fraction','mean'),
        equal_fold_unresolvable_fraction=('equal_target_unresolvable_fraction','mean'),
        test_target_exposures=('test_target_exposures','sum'),p_le_alpha_exposures=('p_le_alpha_exposures','sum'))
    phase_mean['exposure_weighted_p_le_alpha_fraction']=phase_mean.p_le_alpha_exposures/phase_mean.test_target_exposures
    phase_mean['each_A_row_test_once_per_target_form_config']=True
    phase_mean['fold_p_values_averaged']=False
    phase_mean['repeated_target_exposures_are_independent_people']=False
    phase_mean['official_candidate_FPR']=False
    outputs=(pd.DataFrame(support),table,foldmean,phase_mean,pd.DataFrame(controls))
    for name,data in zip(CSV_NAMES,outputs):
        data.to_csv(dest/name,index=False,float_format='%.17g')
    for path,expected in plan['input_sha256'].items():assert sha(path)==expected
    for path,expected in plan['support_code_sha256'].items():assert sha(path)==expected
    assert sha(SCRIPT)==plan['code_sha256']
    record=dict(status='completed',completed_jst=stamp(),seconds=time.perf_counter()-start,phase=phase,
        code_sha256=sha(SCRIPT),support_code_sha256=plan['support_code_sha256'],input_sha256=plan['input_sha256'],
        output_sha256={name:sha(dest/name) for name in CSV_NAMES},primary_targets=primary,age_cuts=cuts.tolist(),
        partition_assignment_sha256=hashlib.sha256(assignment.astype('<i8').tobytes()).hexdigest(),
        partition_sizes=[len(p) for p in parts],target_fold_cases=len(table),group_support_rows=len(support),
        maximum_single_query_error=max(r['original_A3_helper_single_query_max_error'] for r in controls),
        maximum_Counter_searchsorted_error=0,three_fold_average_p_value=False,
        each_A_row_test_once=True,private_B_Q_GT_or_official_scores_accessed=False,
        individual_p_values_or_IDs_or_NPZ_saved=False,final_rank_or_cross_created=False,method_or_config_selection=False)
    write_json(dest/'実行記録.json',record)
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--write-plan',action='store_true');args=ap.parse_args()
    assert sys.dont_write_bytecode and Path(sys.executable).absolute()==ROOT/'.venv/bin/python'
    assert sha(Path(frozen.__file__))==FROZEN_A3_SHA
    if args.write_plan:
        assert not OUT.exists();OUT.mkdir()
        plan=dict(created_jst=stamp(),code_sha256=sha(SCRIPT),support_code_sha256=support_hashes(),
            input_sha256={str(p):sha(p) for p in sorted(ALLOWED_CSV)},seed=SEED,partition_sizes=[1714,1714,1715],
            roles='foldf reference=A[f], calibration=A[(f+1)%3], test=A[(f+2)%3]',forms=list(FORMS),
            configs=[dict(name=name,group_columns=list(cols)) for name,cols in CONFIGS],alphas=list(ALPHAS),
            raw='same frozen A3 Gen7k10 and shared Aref midrank-probit Gaussian LW source-ratio helper; no IQR rescaling required for order p',
            p_value='(1 + count(Acal_raw_score >= query_raw_score))/(n_calibration_group + 1)',
            empty_calibration_group='p=1',ties='>= conservative',threshold_comparison='exact rational integer numerator/denominator comparison',
            zero_information_definition='no calibration observation in query group (n_group=0), distinct from merely p=1',
            unresolvable_definition='minimum possible p=1/(n_group+1) exceeds the predeclared alpha',
            grouping_age_cuts='same-phase full public A age quantiles .2/.4/.6/.8; fixed covariate strata include test covariates',
            primary_targets='prelim excludes1/2 from fixed primary diagnostics (18); main22; own C excluded as in frozen A3',
            exchangeability_scope='conditional on fixed public pool/C/reference/source scores and fixed strata, ideal randomized cal/test allocation among remaining public A rows supplies within-group symmetry; one saved seed is one internal realization',
            limits=[
                'This is an internal finite public-A-pool random-allocation diagnostic. Public-A nonmembership does not establish representativeness or exchangeability with unseen B candidate nonmembers.',
                'C may depend on public A; that dependence alone does not negate symmetry of a subsequent independently randomized cal/test allocation when the pool/C/reference scores are fixed.',
                'The fixed seed creates reproducible partitions. The theoretical randomization statement concerns the hypothetical random-allocation law, not a guarantee that every realized empirical rejection rate is <=alpha.',
                'Full-A age cuts include test covariates. Group definitions are fixed before allocation and use no membership labels or test-score tuning.',
                'No shrinkage, Q mixture, fitted scale, CRPS, final rank, cross correction, score-based configuration selection, official candidate FPR, or MIA accuracy is computed.',
                'Each A point is test once in the3role cycle; its p values are never averaged across folds because other folds use that point as reference or calibration.',
                'The same A points are reused across targets/raw forms/configurations/alphas. Aggregate exposures are not independent patient counts.',
                'Empty groups have p=1. Ties count >= and are conservative; integer rational thresholding includes exact p=alpha boundaries.',
            ],numerical_qa=p_toy_qa(),public_only_file_guard=True,individual_p_values_IDs_NPZ_exported=False,
            python=sys.executable,python_version=platform.python_version(),
            membership_truth_or_official_scores=False,method_selection=False,final_prose_or_GitHub=False)
        write_json(OUT/'事前固定条件.json',plan)
        print(json.dumps(dict(stage='finite_public_A_plan_fixed',source_sha256=plan['code_sha256'],support_sources=len(plan['support_code_sha256']),public_inputs=len(plan['input_sha256'])),ensure_ascii=False));return
    assert OUT.is_dir() and {p.name for p in OUT.iterdir()}=={'事前固定条件.json'}
    plan=json.loads((OUT/'事前固定条件.json').read_text());assert sha(SCRIPT)==plan['code_sha256']
    for path,expected in plan['input_sha256'].items():assert sha(path)==expected
    for path,expected in plan['support_code_sha256'].items():assert sha(path)==expected
    records={phase:phase_run(phase,plan) for phase in ('prelim','main')}
    write_json(OUT/'完了記録.json',dict(completed_jst=stamp(),code_sha256=sha(SCRIPT),plan_sha256=sha(OUT/'事前固定条件.json'),
        phase_record_sha256={phase:sha(OUT/phase/'実行記録.json') for phase in records},phases_completed=list(records),
        maximum_single_query_error=max(r['maximum_single_query_error'] for r in records.values()),
        private_B_Q_GT_or_scores_accessed=False,individual_p_ID_NPZ_exported=False,fold_average_p_value=False,
        final_rank_cross_submission_created=False,method_selected=False,final_prose_GitHub_external_action=False))
    print(json.dumps(dict(stage='finite_public_A_completed',phases=list(records),source_sha256=sha(SCRIPT)),ensure_ascii=False))


if __name__=='__main__':main()
