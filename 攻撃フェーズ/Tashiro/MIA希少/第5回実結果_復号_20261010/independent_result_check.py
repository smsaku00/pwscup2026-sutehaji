#!/usr/bin/env python3
"""Independent exhaustive decoder for the supplied official round-5 rare MIA score.

Run with /Users/taisei1105/pws2026-local/.venv/bin/python.
Does not read saved legal-configuration lists or change any submission CSV.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import sys
sys.dont_write_bytecode = True

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scorer-src', type=Path,
                    default=Path('/Users/taisei1105/pws2026-local/本戦/配布データ/pwscup2026-kit-main_revised0912/starter/src'),
                    help='Local official main-round starter/src directory. Scorer hash is checked in the report.')
args = parser.parse_args()
SCORER_ROOT = args.scorer_src.resolve()
sys.path.insert(0, str(SCORER_ROOT))
from pwscup2026.scoring.attack_score import score_mia

PARAMETERS = {
    'candidate_ids': [47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675],
    'active_teams': [6, 10, 11, 16, 17, 25],
    'positive_counts': [1, 2, 1, 2, 1, 2],
    'rare_population': 24,
    'high_side_counts': [6, 7, 7, 11, 10, 12],
    'questions_round3_to7': [
        [115, 169, 264, 383, 462],
        [47, 115, 304, 462, 675],
        [47, 90, 264, 462, 495],
        [47, 76, 264, 304, 383],
        [47, 76, 90, 115, 169],
    ],
    'official_observations_round3_to5': [
        '0.017203703703703704',
        '0.010814814814814814',
        '0.011166666666666667',
    ],
    'observation_source': 'Official round-5 mia.tpr_rare supplied by parent; HTML acquisition and source hashing handled separately by parent.',
    'low_fpr': 0.01,
    'mean_denominator_active_teams': 6,
    'shared_membership_exclusive': True,
}


def main() -> None:
    ids = PARAMETERS['candidate_ids']
    teams = PARAMETERS['active_teams']
    ms = PARAMETERS['positive_counts']
    ns = PARAMETERS['high_side_counts']
    qs = PARAMETERS['questions_round3_to7']
    q_masks = [sum(1 << i for i, person in enumerate(ids) if person in q) for q in qs]
    lookup = [
        [Fraction((24 - m) * k, 600 * m * (n - k)) for k in range(m + 1)]
        for m, n in zip(ms, ns)
    ]
    denominator = math.lcm(*(s.denominator for row in lookup for s in row))
    assert denominator == 54000
    score_lookup = [[int(s * denominator) for s in row] for row in lookup]
    obs_exact = [Fraction(s).limit_denominator(10**9) for s in PARAMETERS['official_observations_round3_to5']]
    obs_numerators = tuple(int(s * denominator) for s in obs_exact)
    assert all(Fraction(n, denominator) == s for n, s in zip(obs_numerators, obs_exact))
    assert obs_numerators == (929, 584, 603)

    # Confirm the rational shortcut against the actual official ROC implementation.
    scorer_checks = []
    for ti, (m, n) in enumerate(zip(ms, ns)):
        for k in range(m + 1):
            confidence = np.array([1.0] * n + [0.0] * (24 - n))
            truth = np.zeros(24, dtype=int)
            truth[:k] = 1
            truth[n:n + (m - k)] = 1
            actual = score_mia(
                confidence, truth, {'scoring': {'low_fpr': 0.01}},
                rare_mask=np.ones(24, dtype=bool),
            ).breakdown['tpr_rare']
            expected = float(lookup[ti][k] * 6)
            scorer_checks.append(abs(actual - expected))
    assert max(scorer_checks) < 1e-15

    # Enumerate every unordered membership subset at each team, rejecting reuse
    # of a person at any later team. This visits all 2,494,800 placements.
    choices = []
    for ti, m in enumerate(ms):
        team_choices = []
        for inds in combinations(range(len(ids)), m):
            mask = sum(1 << i for i in inds)
            scores = tuple(score_lookup[ti][(mask & qm).bit_count()] for qm in q_masks)
            team_choices.append((mask, scores, tuple(ids[i] for i in inds)))
        choices.append(team_choices)

    all_count = 0
    after_round3 = 0
    after_round4 = []
    after_round5 = []

    def walk(ti, used, scores, config):
        nonlocal all_count, after_round3
        if ti == len(teams):
            all_count += 1
            if scores[0] != obs_numerators[0]:
                return
            after_round3 += 1
            if scores[1] != obs_numerators[1]:
                return
            after_round4.append((config, scores))
            if scores[2] == obs_numerators[2]:
                after_round5.append((config, scores))
            return
        for mask, row_scores, people in choices[ti]:
            if not used & mask:
                walk(ti + 1, used | mask, tuple(a + b for a, b in zip(scores, row_scores)), config + (people,))

    walk(0, 0, (0, 0, 0, 0, 0), ())
    assert all_count == 2494800
    assert after_round3 == 43200
    assert len(after_round4) == 1296
    assert len(after_round5) == 16
    round5_high_positive_vectors = sorted({
        tuple((sum(1 << ids.index(p) for p in people) & q_masks[2]).bit_count() for people in conf)
        for conf, _ in after_round5
    })
    used_sets = [{p for people in conf for p in people} for conf, _ in after_round5]
    always_shared_ids = sorted(set.intersection(*used_sets))
    never_shared_ids = sorted(set(ids) - set.union(*used_sets))

    def score_object(n):
        return {'numerator': n, 'denominator': denominator, 'fraction': str(Fraction(n, denominator)), 'float': n / denominator}

    def configuration_object(config, scores):
        return {
            'team_members': {str(team): list(people) for team, people in zip(teams, config)},
            'unassigned_candidates': sorted(set(ids) - {p for people in config for p in people}),
            'scores_round3_to7': [score_object(n) for n in scores],
        }

    target_candidates = {}
    for ti, team in enumerate(teams):
        subsets = sorted({tuple(conf[ti]) for conf, _ in after_round5})
        union = sorted({p for subset in subsets for p in subset})
        certain = sorted(set.intersection(*(set(s) for s in subsets)))
        target_candidates[str(team)] = {
            'number_members': ms[ti],
            'possible_member_subsets': [list(s) for s in subsets],
            'possible_member_ids': union,
            'certain_member_ids': certain,
            'certain_nonmember_ids_within_11_candidates': sorted(set(ids) - set(union)),
        }

    round6 = defaultdict(list)
    pairs = defaultdict(list)
    for config, scores in after_round5:
        round6[scores[3]].append((config, scores))
        pairs[scores[3:]].append((config, scores))
    branches = []
    for n6, configs in sorted(round6.items()):
        next_branches = defaultdict(list)
        for config, scores in configs:
            next_branches[scores[4]].append((config, scores))
        branches.append({
            'round6_predicted_score': score_object(n6),
            'remaining_configurations': len(configs),
            'round7_branches': [
                {
                    'round7_predicted_score': score_object(n7),
                    'remaining_configurations': len(items),
                    'decoded_configurations': [configuration_object(c, s) for c, s in items],
                }
                for n7, items in sorted(next_branches.items())
            ],
        })
    precision = {}
    for digits in range(3, 7):
        buckets = Counter(tuple(f'{s / denominator:.{digits}f}' for s in scores[3:]) for _, scores in after_round5)
        precision[str(digits)] = {
            'distinct_round6_round7_observation_pairs': len(buckets),
            'collision_buckets': sum(n > 1 for n in buckets.values()),
            'worst_remaining_configurations': max(buckets.values()),
        }
    report = {
        'parameters': PARAMETERS,
        'parameters_sha256': hashlib.sha256(json.dumps(PARAMETERS, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'reproduction_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scorer_path': str(SCORER_ROOT / 'pwscup2026/scoring/attack_score.py'),
        'scorer_sha256': hashlib.sha256((SCORER_ROOT / 'pwscup2026/scoring/attack_score.py').read_bytes()).hexdigest(),
        'official_scorer_rational_formula_max_abs_error': max(scorer_checks),
        'observation_fractions_round3_to5': [str(x) for x in obs_exact],
        'all_legal_configurations_before_observations': all_count,
        'configurations_after_round3': after_round3,
        'configurations_after_round4': len(after_round4),
        'configurations_after_actual_round5': len(after_round5),
        'round5_high_positive_vectors_in_active_team_order': [list(x) for x in round5_high_positive_vectors],
        'always_shared_ids_across_active_teams': always_shared_ids,
        'never_shared_ids_across_active_teams': never_shared_ids,
        'target_candidates_after_actual_round5': target_candidates,
        'all_16_configurations': [configuration_object(c, s) for c, s in after_round5],
        'round6_distinct_branches': len(round6),
        'round6_worst_remaining_configurations': max(len(x) for x in round6.values()),
        'round6_branch_size_histogram': dict(sorted(Counter(len(x) for x in round6.values()).items())),
        'round6_round7_distinct_pairs': len(pairs),
        'round6_round7_collision_buckets': sum(len(x) > 1 for x in pairs.values()),
        'round7_worst_remaining_configurations': max(len(x) for x in pairs.values()),
        'all_future_branches': branches,
        'future_display_precision_given_exact_past_observations': precision,
        'conclusion': 'With the stated population, exclusivity, candidate and scoring assumptions, the unchanged fixed round-6 and round-7 questions identify all 16 surviving configurations uniquely.',
        'no_submission_csv_read_or_changed': True,
        'no_tournament_submission': True,
        'no_github_mutation': True,
    }
    assert report['round6_round7_collision_buckets'] == 0
    assert report['round7_worst_remaining_configurations'] == 1
    output = Path(__file__).with_name('independent_result_check.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in [
        'all_legal_configurations_before_observations', 'configurations_after_round3',
        'configurations_after_round4', 'configurations_after_actual_round5',
        'target_candidates_after_actual_round5', 'round6_distinct_branches',
        'round6_worst_remaining_configurations', 'round6_branch_size_histogram',
        'round6_round7_distinct_pairs', 'round6_round7_collision_buckets',
        'round7_worst_remaining_configurations',
        'future_display_precision_given_exact_past_observations',
    ]}, ensure_ascii=False, indent=2))
    print('OUTPUT', output)
    print('OUTPUT_SHA256', hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
