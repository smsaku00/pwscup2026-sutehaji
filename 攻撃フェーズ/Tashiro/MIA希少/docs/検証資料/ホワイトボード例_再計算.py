"""ホワイトボードの続きを、仮のA〜Kで再計算する。標準ライブラリのみ。

プロジェクトの .venv/bin/python -B <このファイル> で実行。
結果は画面へ出すだけで、CSV・ZIP・結果ファイルの生成や提出は行わない。
実際の候補ID・個人の正解は使わない。式は本戦の固定24人を前提とする。
"""
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import json
import random

LETTERS = 'ABCDEFGHIJK'
TEAMS = (6, 10, 11, 16, 17, 25)
MEMBERS = {6: 1, 10: 2, 11: 1, 16: 2, 17: 1, 25: 2}
EXAMPLE_TRUTH = {6: 'G', 10: 'AK', 11: 'B', 16: 'CI', 17: 'J', 25: 'DE'}
TOLERANCE = 1e-13
MINIMUM_GAP = Fraction(1, 100000000)


def mask(letters):
    return sum(1 << LETTERS.index(c) for c in letters)


def names(bits):
    return ''.join(c for i, c in enumerate(LETTERS) if bits >> i & 1)


def score(high_count, member_count, high_members):
    assert member_count in (1, 2) and 0 <= high_members <= member_count
    assert high_count > high_members
    return Fraction((24-member_count)*high_members,
                    100*member_count*(high_count-high_members))


@lru_cache(None)
def choose(candidates):
    # 正解を受け取らず、最も多く残る候補群が小さくなる質問を選ぶ。
    best = None
    for selected in range(1, 1 << 10):
        groups = tuple(tuple(c for c in candidates if (c & selected).bit_count() == h)
                       for h in range(3))
        sizes = tuple(map(len, groups))
        key = (max(sizes), sum(n*n for n in sizes), selected.bit_count(), selected)
        if best is None or key < best[0]:
            best = (key, selected, groups)
    assert max(map(len, best[2])) < len(candidates)
    return best[1], best[2]


def make_query(candidates, rng):
    selected = {t: choose(candidates[t])[0] if len(candidates[t]) > 1 else 0 for t in TEAMS}
    options = {t: sorted({(c & selected[t]).bit_count() for c in candidates[t]}) for t in TEAMS}
    outcomes = list(product(*(options[t] for t in TEAMS)))
    for trial in range(1, 10001):
        pads = {t: rng.randint(1, 13) if selected[t] else 1 for t in TEAMS}
        table = {hits: sum((score(selected[t].bit_count()+pads[t], MEMBERS[t], hits[i])
                           for i, t in enumerate(TEAMS)), Fraction(0))/6
                 for hits in outcomes}
        values = sorted(set(table.values()))
        gap = min((b-a for a, b in zip(values, values[1:])), default=Fraction(1))
        if len(values) == len(outcomes) and gap > MINIMUM_GAP:
            return selected, pads, table, gap, trial
    raise ValueError('読み分けられる設定が見つからない')


def mock_score(selected, pads):
    # 仮の採点側。推定側へは平均1つだけ返す。
    values = [score(selected[t].bit_count()+pads[t], MEMBERS[t],
                    (mask(EXAMPLE_TRUTH[t]) & selected[t]).bit_count()) for t in TEAMS]
    return float(sum(values, Fraction(0))/6)


def compute():
    if not __debug__:
        raise RuntimeError('検査を省略する実行は禁止')
    assert len(''.join(EXAMPLE_TRUTH.values())) == len(set(''.join(EXAMPLE_TRUTH.values()))) == 9
    assert all(len(EXAMPLE_TRUTH[t]) == MEMBERS[t] for t in TEAMS)
    candidates = {t: tuple(sum(1 << i for i in pair) for pair in combinations(range(11), MEMBERS[t]))
                  for t in TEAMS}
    initial_pairs = candidates[10]
    fourth_branches = []
    for h, next_selected in ((0, 'EFG'), (1, 'AEFG'), (2, 'AB')):
        remaining = tuple(c for c in initial_pairs if (c & mask('ABCD')).bit_count() == h)
        sizes = [sum((c & mask(next_selected)).bit_count() == k for c in remaining) for k in range(3)]
        fourth_branches.append({'round3_hits': h, 'before': len(remaining),
                                'round4_selected': next_selected, 'after_hits_0_1_2': sizes})
    assert [r['before'] for r in fourth_branches] == [21, 28, 6]
    assert [r['after_hits_0_1_2'] for r in fourth_branches] == [[6, 12, 3], [12, 13, 3], [1, 4, 1]]
    # 異なる係数でも、和が衝突する反例。
    collision = [score(5+p, 1, 1) for p in (6, 11, 2)]
    assert len(set(collision)) == 3 and collision[0]+collision[1] == collision[2]
    rng = random.Random(20261006)
    trace = []
    first_table = []
    first_team_scores = []
    for round_no in range(3, 8):
        selected, pads, table, gap, trials = make_query(candidates, rng)
        if round_no == 3:
            assert [names(selected[t]) for t in TEAMS] == ['ABCDE','ABCD','ABCDE','ABCD','ABCDE','ABCD']
            assert [pads[t] for t in TEAMS] == [8, 2, 11, 5, 5, 9]
            assert len(table) == 216 and gap == Fraction(1, 504000)
            first_table = [{'hits': list(h), 'mean_exact': str(v), 'mean': float(v)} for h, v in table.items()]
            first_team_scores = [{'team': t, 'members': MEMBERS[t], 'selected': names(selected[t]),
                                  'padding': pads[t], 'high_count': selected[t].bit_count()+pads[t],
                                  'scores_by_hits': [str(score(selected[t].bit_count()+pads[t], MEMBERS[t], h))
                                                     for h in range(MEMBERS[t]+1)]} for t in TEAMS]
        observed = mock_score(selected, pads)
        matching = [(hits, exact) for hits, exact in table.items() if abs(float(exact)-observed) <= TOLERANCE]
        assert len(matching) == 1
        hits, exact = matching[0]
        before = candidates
        candidates = {t: tuple(c for c in before[t] if (c & selected[t]).bit_count() == hits[i])
                      for i, t in enumerate(TEAMS)}
        assert all(candidates.values())
        trace.append({'round': round_no, 'mean_exact': str(exact), 'observed_example_mean': observed,
                      'response_patterns': len(table), 'minimum_gap': str(gap), 'local_design_trials': trials,
                      'teams': [{'team': t, 'selected': names(selected[t]), 'padding': pads[t],
                                 'decoded_hits': hits[i], 'before': len(before[t]), 'after': len(candidates[t]),
                                 'remaining': [names(c) for c in candidates[t]]} for i, t in enumerate(TEAMS)]})
    # 答え合わせは最後。正解を見て質問を変更しない。
    assert all(candidates[t] == (mask(EXAMPLE_TRUTH[t]),) for t in TEAMS)
    ten = [next(c for c in row['teams'] if c['team'] == 10) for row in trace]
    assert [c['selected'] for c in ten] == ['ABCD', 'AEFG', 'BCEF', 'AHI', 'J']
    assert [c['decoded_hits'] for c in ten] == [1, 1, 0, 1, 0]
    assert [c['after'] for c in ten] == [28, 13, 5, 2, 1]
    assert trace[0]['mean_exact'] == '241/24000'
    return {'scope': '仮のA〜Kによる説明例。本戦の個人の推定結果でも提出設定の決定でもない。',
            'teams_order': list(TEAMS), 'members': MEMBERS,
            'mock_truth_used_by_scorer_and_final_check_only': EXAMPLE_TRUTH,
            'tolerance': TOLERANCE, 'random_seed': 20261006,
            'first_query': first_team_scores, 'first_query_table_216': first_table,
            'round4_branches': fourth_branches,
            'collision_example': {'padding': [6,11,2], 'scores': list(map(str, collision)),
                                  'first_two_sum_equals_third': True},
            'trace': trace, 'final_answer_matches_mock_truth': True}


if __name__ == '__main__':
    print(json.dumps(compute(), ensure_ascii=False, indent=2))
