"""Exact integer-resolution expectation using saved public-A group counts only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import unicodedata
from zoneinfo import ZoneInfo

SCRIPT = Path(__file__).resolve()
HERE = SCRIPT.parent
OUT = HERE / '有限p値の整数解像度結果'
SOURCES = tuple(HERE / '公開Aの有限標本p値結果' / phase / '群別の較正標本数と閾値到達可否.csv'
                for phase in ('prelim', 'main'))
ALPHAS = ('0.005', '0.01', '0.02', '0.05')
CONFIGS = ('global', 'sex_fullA_age_bin', 'prefecture_sex')
OUTPUTS = ('群とalpha別の整数解像度.csv', 'phaseとgroup別の期待尾率.csv',
           'foldとgroup別の期待尾率.csv', '整数解像度の数式と制限.json')


def count_only_guard(event, args):
    if event != 'open' or not args or isinstance(args[0], int):
        return
    path = Path(str(args[0])).resolve()
    name = unicodedata.normalize('NFC', str(path))
    if any(x in name for x in ('B_self.csv', 'B_practice.csv', 'A_bg.csv', 'aia_challenge_', 'pool_ids',
                              '/正解データ/', '/アーカイブ/', '.npz', 'F_mia.csv', 'F_aia.csv',
                              'attack_score.py', '対象別TPR.csv', 'result.txt')):
        raise PermissionError(f'count-only mathematical artifact: {path}')
    if path.suffix == '.csv' and path not in SOURCES and not path.is_relative_to(OUT):
        raise PermissionError(f'not a saved aggregate count input: {path}')


sys.addaudithook(count_only_guard)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def json_x(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def csv_x(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def alpha_grid(n, alpha):
    """n+1 equiprobable tie-free ranks; exact boundary for p<=alpha."""
    assert n >= 0 and Fraction(0) < alpha < Fraction(1)
    ranks = 0 if n == 0 else (alpha.numerator * (n + 1)) // alpha.denominator
    return ranks, Fraction(ranks, n + 1)


def verify_counts(ns):
    checked = 0
    for n in sorted(set(ns) | {0, 1, 18, 19, 20, 48, 49, 50, 98, 99, 100, 149, 199, 249, 999, 1714}):
        for text in ALPHAS:
            alpha = Fraction(text)
            ranks, rate = alpha_grid(n, alpha)
            # Independently enumerate the attainable rational p values, without a floor operation.
            enumerated = sum(Fraction(r, n + 1) <= alpha for r in range(1, n + 2))
            assert ranks == enumerated and rate == Fraction(enumerated, n + 1)
            assert rate <= alpha
            checked += 1
    # Independently enumerate query/calibration allocation of distinct and tied finite pools.
    toy_checks = 0
    for population in ([0, 1, 2, 3], [0, 0, 1, 2], [1, 1, 1, 1]):
        for n in range(4):
            ps = []
            for query_idx in range(4):
                others = [i for i in range(4) if i != query_idx]
                for calibration in itertools.combinations(others, n):
                    greater_equal = sum(population[i] >= population[query_idx] for i in calibration)
                    ps.append(Fraction(1 + greater_equal, n + 1))
            for alpha in (Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4)):
                observed_law = Fraction(sum(p <= alpha for p in ps), len(ps))
                _, grid = alpha_grid(n, alpha)
                assert observed_law <= grid
                if len(set(population)) == len(population):
                    assert observed_law == grid
                toy_checks += 1
    return dict(exact_rank_threshold_count_checks=checked,finite_pool_tie_allocation_checks=toy_checks,
                exact_fraction_errors=0,empty_group_p_one=True,ties_no_greater_than_tie_free_grid=True)


def summarize(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    summaries = []
    for key, part in sorted(grouped.items()):
        alpha = Fraction(part[0]['alpha'])
        total = sum(r['test_group_rows'] for r in part)
        assert total > 0
        expected_count = sum((Fraction(r['alpha_grid_numerator'], r['alpha_grid_denominator']) *
                              r['test_group_rows'] for r in part), Fraction(0))
        expected_rate = expected_count / total
        row = {k: v for k, v in zip(keys, key)}
        row.update(
            group_fold_rows=len(part),test_group_exposures=total,
            expected_tail_rows_numerator=expected_count.numerator,expected_tail_rows_denominator=expected_count.denominator,
            tie_free_expected_tail_rows=float(expected_count),
            expected_tail_rate_numerator=expected_rate.numerator,expected_tail_rate_denominator=expected_rate.denominator,
            test_count_weighted_alpha_grid=float(expected_rate),nominal_alpha=float(alpha),
            nominal_minus_grid=float(alpha-expected_rate),grid_over_nominal_alpha=float(expected_rate/alpha),
            empty_calibration_test_rows=sum(r['test_group_rows'] for r in part if r['calibration_group_rows']==0),
            unresolvable_test_rows=sum(r['test_group_rows'] for r in part if r['attainable_rejecting_ranks']==0),
            test_rows_with_grid_equal_alpha=sum(r['test_group_rows'] for r in part if Fraction(r['alpha_grid_numerator'],r['alpha_grid_denominator'])==alpha),
            expectation_under_tie_free_uniform_rank=True,upper_bound_for_every_seed_realization=False,
            observed_p_values_used=False,official_candidate_FPR=False,used_for_selection=False)
        summaries.append(row)
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write-plan', action='store_true')
    args = ap.parse_args()
    assert sys.dont_write_bytecode
    assert Path(sys.executable).absolute() == HERE.parents[3] / '.venv/bin/python'
    if args.write_plan:
        assert not OUT.exists()
        OUT.mkdir()
        plan = dict(created_jst=stamp(),source_sha256=sha(SCRIPT),
                    input_sha256={str(p):sha(p) for p in SOURCES},alphas=list(ALPHAS),configs=list(CONFIGS),
                    source_group_rows=630,group_alpha_rows=2520,phase_config_alpha_rows=24,fold_config_alpha_rows=72,
                    equation='alpha_grid=floor(alpha*(n+1))/(n+1) under uniform tie-free ranks1..n+1; n0 gives p1 and grid0',
                    grouping_weights='saved test_group_rows, sum over3folds within phase/config/alpha; no per-patient p averaging',
                    inputs_only_saved_public_aggregate_support_counts=True,raw_data_or_p_values_read=False,
                    expected_outputs=list(OUTPUTS)+['実行記録.json'],method_or_threshold_selection=False,
                    final_experiment_prose_or_GitHub_or_external_action=False)
        json_x(OUT/'事前固定計画.json',plan)
        print(json.dumps(dict(stage='integer_resolution_plan_fixed',source_sha256=plan['source_sha256'],plan_sha256=sha(OUT/'事前固定計画.json'))));return
    assert OUT.is_dir() and {p.name for p in OUT.iterdir()} == {'事前固定計画.json'}
    plan = json.loads((OUT/'事前固定計画.json').read_text())
    assert sha(SCRIPT) == plan['source_sha256']
    assert list(ALPHAS) == plan['alphas'] and list(CONFIGS) == plan['configs']
    for path, expected in plan['input_sha256'].items():assert sha(path)==expected
    source_rows=[]
    for path in SOURCES:
        with path.open(newline='',encoding='utf-8') as f:
            source_rows.extend(csv.DictReader(f))
    assert len(source_rows)==630
    assert len({(r['phase'],r['fold'],r['config'],r['group']) for r in source_rows})==630
    groups=[]
    ns=[]
    for row in source_rows:
        n,nt=int(row['calibration_group_rows']),int(row['test_group_rows'])
        assert n>=0 and nt>=0
        ns.append(n)
        for text in ALPHAS:
            alpha=Fraction(text);ranks,grid=alpha_grid(n,alpha)
            groups.append(dict(phase=row['phase'],fold=int(row['fold']),config=row['config'],group=row['group'],alpha=text,
                calibration_group_rows=n,test_group_rows=nt,minimum_possible_p=1/(n+1),
                attainable_rejecting_ranks=ranks,alpha_grid_numerator=grid.numerator,alpha_grid_denominator=grid.denominator,
                tie_free_uniform_rank_expected_tail=float(grid),nominal_minus_grid=float(alpha-grid),
                empty_calibration_group=(n==0),can_reach_alpha=(ranks>0),
                expectation_under_uniform_allocation_model=True,realized_seed_rate_upper_bound=False))
    checks=verify_counts(ns)
    phase=summarize(groups,('phase','config','alpha'))
    fold=summarize(groups,('phase','fold','config','alpha'))
    assert (len(groups),len(phase),len(fold))==(2520,24,72)
    assert all(r['test_group_exposures']==5143 for r in phase)
    sex_age_details=[]
    for p in ('prelim','main'):
        part=[r for r in groups if r['phase']==p and r['config']=='sex_fullA_age_bin' and r['alpha']=='0.01']
        summary=next(r for r in phase if r['phase']==p and r['config']=='sex_fullA_age_bin' and r['alpha']=='0.01')
        sex_age_details.append(dict(phase=p,nominal_alpha=.01,test_group_exposures=5143,
            minimum_calibration_group_n=min(r['calibration_group_rows'] for r in part if r['test_group_rows']>0),
            test_rows_n_at_least99=sum(r['test_group_rows'] for r in part if r['calibration_group_rows']>=99),
            test_rows_n_at_least99_but_grid_below_alpha=sum(r['test_group_rows'] for r in part
                if r['calibration_group_rows']>=99 and Fraction(r['alpha_grid_numerator'],r['alpha_grid_denominator'])<Fraction(1,100)),
            test_count_weighted_alpha_grid=summary['test_count_weighted_alpha_grid']))
    math_note=dict(kind='small_mathematical_resolution_artifact_not_final_experiment_prose',
        formula='m=floor(alpha*(n+1)); alpha_grid=m/(n+1)',
        derivation=[
            'For one test draw and n calibration observations in a fixed group, under conditional uniform allocation and tie-free score ranks, p is uniformly distributed on {1/(n+1),...,1}.',
            'Exactly floor(alpha*(n+1)) of those n+1 ranks satisfy p<=alpha. Thus the expected rejection probability is alpha_grid, not necessarily nominal alpha.',
            'For n=0, p=1 and alpha_grid=0 for the fixed alphas<1.',
            'At alpha=.01, n>=99 permits at least one rejecting rank, but alpha_grid equals.01 only if n+1 is a multiple of100. Otherwise the final attainable step below.01 is smaller.',
            'For example n=99 gives1/100=.01; n=100 gives1/101; n=149 gives1/150; n=199 gives2/200=.01.',
            'The saved test-count-weighted expectation is sum_g(n_test_g*alpha_grid_g)/sum_g(n_test_g). Groups and fold counts are pooled as exposures; individual p values are not averaged.',
        ],sex_age_alpha01_details=sex_age_details,
        assumptions_and_limits=[
            'An expectation/decomposition under an ideal uniform random-allocation model, conditional on fixed public pool/C/reference membership/scores and group counts. It is not an upper bound for the realized seed20261008 empirical rate.',
            'This artifact reads only two previously saved public-A group-count CSVs. No public individual A/C rows, raw scores, realized p values, B/Q, membership labels, scoring results, or ROC are read.',
            'Conservative >= ties may make the allocation-law expected rejection probability smaller than or equal to the tie-free alpha_grid. The actual tie distribution is not measured here.',
            'Finite integer resolution may explain a portion of low observed public-null rejection rates. A low observed rate is not evidence of good MIA accuracy or attack suitability.',
            'No configuration/alpha/model is selected, and these counts do not establish exchangeability or representativeness for unseen B nonmembers.',
            'The3roles reuse one finite A pool. Each point is test once perphase/config; no p values are averaged across folds. Groups/configs/phases are not declared independent replications.',
            'Full-A age strata and reference/C dependence retain the original finite-p diagnostic scope; this count calculation adds no new validation of source-score assumptions.',
        ],verification=checks,source_sha256=sha(SCRIPT),plan_sha256=sha(OUT/'事前固定計画.json'))
    csv_x(OUT/OUTPUTS[0],groups);csv_x(OUT/OUTPUTS[1],phase);csv_x(OUT/OUTPUTS[2],fold)
    json_x(OUT/OUTPUTS[3],math_note)
    for path,expected in plan['input_sha256'].items():assert sha(path)==expected
    record=dict(status='completed',completed_jst=stamp(),source_sha256=sha(SCRIPT),plan_sha256=sha(OUT/'事前固定計画.json'),
        input_sha256=plan['input_sha256'],output_sha256={name:sha(OUT/name) for name in OUTPUTS},
        input_group_rows=630,group_alpha_rows=2520,phase_summary_rows=24,fold_summary_rows=72,
        verification=checks,python=sys.executable,python_version=platform.python_version(),
        raw_rows_or_scores_or_p_values_or_B_Q_GT_ROC_read=False,existing_finite_p_source_or_outputs_modified=False,
        individual_rows_or_IDs_or_NPZ_exported=False,method_selected=False,final_prose_GitHub_external_action=False)
    json_x(OUT/'実行記録.json',record)
    print(json.dumps(dict(stage='integer_resolution_complete',source_sha256=sha(SCRIPT),groups=len(groups),phase_summaries=len(phase),verification=checks,sex_age_alpha01=sex_age_details),ensure_ascii=False))


if __name__=='__main__':main()
