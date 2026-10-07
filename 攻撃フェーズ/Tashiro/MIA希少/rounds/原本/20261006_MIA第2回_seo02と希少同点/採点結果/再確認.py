"""保存結果から人数を再確認する。読み取り専用、次回提出物は作らない。

依存: numpy。プロジェクトの .venv/bin/python -B で実行。
人数探索は20261005_MIA希少_2回200通り検証/分析.pyのDecoderを無変更で使用。
元ファイルSHA-256: 4d32855b3537214017c0dbb1af28e9e257fafc8281705a49039395a6e08e2260
今回は実際の第1・2回の設定と点数を入力する。仮想ケースの生成は行わない。
"""
from __future__ import annotations
import csv
import hashlib
import html
import json
import re
import unicodedata
from fractions import Fraction
from math import lcm
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RARE_COUNT = 24
MAX_MEMBERS = 11
FLOAT_GRID_TOLERANCE = 0.0001


def exact_team_score(n, m):
    return Fraction(24-m, 100*(n-m)) if m else Fraction(0)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def report(path):
    objects = []
    for block in re.findall(r'<pre[^>]*>(.*?)</pre>', path.read_text(encoding='utf-8'), re.S):
        try:
            obj = json.loads(html.unescape(block))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get('mia'), dict):
            objects.append(obj)
    if len(objects) != 1:
        raise ValueError('MIAの結果を一意に読めません')
    return objects[0]


class Decoder:
    """各チーム0/1/2人、全体11人以下の全人数配置を対象に探索する。

    22チームを11ずつに分け、それぞれの全配置を足し合わせて照合する。
    入力は同点人数と2回の平均点だけ。正解の人数・共有先の数は受け取らない。
    """
    def __init__(self, sizes, max_members=MAX_MEMBERS):
        self.sizes = sizes
        self.n = len(sizes)
        self.max_members = max_members
        self.denominator = lcm(*(exact_team_score(n, m).denominator
                                 for ns in sizes for n in ns for m in (1, 2)))
        weights = np.array([[[int(exact_team_score(n, m) * self.denominator) for m in range(3)]
                             for n in ns] for ns in sizes], dtype=np.int64)
        midpoint = self.n // 2
        self.halves = []
        for begin, end in ((0, midpoint), (midpoint, self.n)):
            ids = np.arange(3 ** (end - begin), dtype=np.int64)
            states = np.column_stack([(ids // (3 ** j)) % 3 for j in range(end - begin)]).astype(np.int8)
            total = states.sum(axis=1)
            keep = total <= max_members
            states, total = states[keep], total[keep]
            active = (states > 0).sum(axis=1)
            scores = np.zeros((len(states), 2), dtype=np.int64)
            for j in range(end - begin):
                scores += weights[begin + j, :, states[:, j]]
            self.halves.append((states, total, active, scores))
        self.right = {}
        for idx, (a, t, s) in enumerate(zip(self.halves[1][2], self.halves[1][1], self.halves[1][3])):
            self.right.setdefault((int(a), int(s[0]), int(s[1])), []).append((int(t), idx))

    def decode(self, observed):
        if len(observed) != 2 or not all(np.isfinite(observed)):
            raise ValueError('有限な平均点が2個必要です')
        if observed == [0.0, 0.0]:
            return [tuple([0] * self.n)]
        answers = []
        left_states, total, active, scores = self.halves[0]
        for candidate_t in range(1, min(self.n, self.max_members) + 1):
            raw = [v * self.denominator * candidate_t for v in observed]
            goals = [round(v) for v in raw]
            if any(abs(v - g) > FLOAT_GRID_TOLERANCE for v, g in zip(raw, goals)):
                continue
            g0, g1 = goals
            for li, (a, t, s) in enumerate(zip(active, total, scores)):
                if a > candidate_t:
                    continue
                key = (candidate_t - int(a), g0 - int(s[0]), g1 - int(s[1]))
                for rt, ri in self.right.get(key, ()):
                    if int(t) + rt <= self.max_members:
                        answers.append(tuple(int(v) for v in np.r_[left_states[li], self.halves[1][0][ri]]))
        return sorted(answers)



def main(round2_dir=None, round1_dir=None):
    round2_dir = Path(round2_dir) if round2_dir is not None else HERE.parent
    if not __debug__:
        raise RuntimeError('検査を省略する実行オプションは禁止です')
    c = read_json(HERE/'導出条件.json')
    result = read_json(HERE/'MIA結果.json')
    source = read_json(HERE/'出典と照合.json')
    reports = [report(HERE/f'第{i}回_detailed_results.html') for i in (1, 2)]
    for i, obj in enumerate(reports, 1):
        assert obj['mia'] == result['official_mia'][f'round{i}']
        assert obj['constants']['low_fpr'] == c['low_fpr'] == .01
        path = HERE/f'第{i}回_detailed_results.html'
        expected = source['first_report_sha256' if i == 1 else 'second_report_sha256']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert reports[1]['mia_per_target'] == result['mia_per_target_round2']
    assert reports[1]['constants'] == result['observed_constants']
    teams = c['teams']
    observations = [o['mia']['tpr_rare'] for o in reports]
    assert observations == [c['round1_rare_score'], c['round2_rare_score']]
    assert (c['rare_count'], c['unknown_count'], c['known_nonmember_count']) == (24, 11, 13)
    assert c['maximum_members_per_team'] == 2 and c['maximum_members_total'] == 11
    assert teams == list(map(str, [*range(18), 22, 25, 26, 27]))
    assert len(c['round1_sizes']) == len(c['round2_sizes']) == len(teams)
    assert all(11 <= n <= 24 for n in c['round1_sizes']+c['round2_sizes'])
    tolerance = c['score_match_tolerance']
    assert tolerance == 1e-13
    # 第1回だけの場合も22チームの全候補を探索。平均に含めるチーム数は未知。
    first = Decoder(list(zip(c['round1_sizes'], c['round1_sizes']))).decode([observations[0]]*2)
    both = Decoder(list(zip(c['round1_sizes'], c['round2_sizes']))).decode(observations)
    cases = read_json(round2_dir/'結果/2回目の点数と人数102通り.json')
    assert len(first) == len(cases) == 102
    assert {sum(m > 0 for m in row) for row in first} == {6, 7}
    assert set(first) == {tuple(case['counts'][t] for t in teams) for case in cases}
    assert len(both) == 1
    counts = dict(zip(teams, both[0]))
    assert counts == result['derived_rare_members']
    active = sum(m > 0 for m in counts.values())
    total = sum(counts.values())
    assert active == result['active_teams'] == 6
    assert total == result['total_rare_members'] == 9
    exact = [sum((exact_team_score(n, counts[t]) for t, n in zip(teams, c[f'round{i}_sizes'])), Fraction(0))/active for i in (1, 2)]
    assert all(abs(float(x)-v) <= tolerance for x, v in zip(exact, observations))
    assert [str(x) for x in exact] == [result['exact_rare_mean_round1'], result['exact_rare_mean_round2']]
    for case in cases:
        ms = [case['counts'][t] for t in teams]
        assert all(m in (0, 1, 2) for m in ms) and sum(ms) <= 11
        k = sum(m > 0 for m in ms)
        means = [sum((exact_team_score(n, m) for n, m in zip(c[f'round{i}_sizes'], ms)), Fraction(0))/k for i in (1, 2)]
        assert abs(float(means[0])-observations[0]) <= tolerance
        assert str(means[1]) == case['round2_score_exact']
        assert abs(float(means[1])-case['round2_score']) <= tolerance
    hits = [case for case in cases if abs(case['round2_score']-observations[1]) <= tolerance]
    assert len(hits) == 1 and hits[0]['counts'] == counts
    assert hits[0]['case_id'] == result['matched_precomputed_case'] == 'N015'
    assert len({case['round2_score_exact'] for case in cases}) == 102
    ordered = sorted(Fraction(case['round2_score_exact']) for case in cases)
    assert min(b-a for a, b in zip(ordered, ordered[1:])) == Fraction(1, 936000)
    raw_scores = read_json(HERE/'第2回_scores.json')
    assert raw_scores['score_2'] == reports[1]['mia']['tpr_at_low_fpr']
    assert raw_scores['score_3'] == observations[1]

    # GitHubのMIA全体が手元にあれば、公開済みCSVの同一性と実際の同点人数も検査。
    mia_root = round2_dir.parent
    expected_name = '20261005_MIA希少_第1回同点データ'
    round1_dirs = [p for p in mia_root.iterdir() if p.is_dir() and unicodedata.normalize('NFC', p.name) == expected_name]
    if round1_dir is not None:
        round1_dirs = [Path(round1_dir)]
    csv_checked = False
    if len(round1_dirs) == 1:
        paths = [round1_dirs[0]/'結果/F_mia.csv', round2_dir/'結果/F_mia.csv']
        settings = read_json(round2_dir/'結果/設定.json')
        rare = settings['rare_ids']
        unknown = settings['unknown_rare_ids']
        assert len(rare) == 24 and len(unknown) == 11 and set(unknown) <= set(rare)
        for i, path in enumerate(paths):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['mia_csv_sha256'][i]
            with path.open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            frame = {int(row['record_id']): row for row in rows}
            assert len(rows) == len(frame) == 1049
            for t, n in zip(teams, c[f'round{i+1}_sizes']):
                values = [float(frame[r][t]) for r in rare]
                assert all(v in (.1, .4) for v in values)
                assert sum(v == .4 for v in values) == n
                assert all(float(frame[r][t]) == .4 for r in unknown)
        csv_checked = True
    print(json.dumps({'status': 'OK', 'first_candidates': len(first), 'second_candidates': len(both),
        'counts': counts, 'active_teams': active, 'total_members': total,
        'exact_means': list(map(str, exact)), 'csv_checked': csv_checked,
        'note': '共有条件・保存設定を前提とする人数推定。個人のIDは未特定。'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
