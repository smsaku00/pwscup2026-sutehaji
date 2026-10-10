"""MIA希少の制約付き同時確率。提出CSV・ZIPには一切書き込まない。

実行: python3 分析.py                 # NumPyのみ必要。ネット接続不要
      python3 分析.py --no-write      # 再計算し、JSONも更新しない
      python3 分析.py --betas 0.5 1 4 16

計算時間の予想（2026-09-06のこの端末で標準10モデルを実測: 約3.4秒）:
  保存済み距離の読み込みと公式結果の照合: 約0.3秒
  標準10モデルの同時確率・第4回→第5回検証: 約3秒
  合計の目安: 3〜15秒。端末性能と同時実行の負荷で変わる。
  31億通りを列挙せず、患者11人の使用状況2^11通りでまとめる。
  最初のモデルの実測から残り時間を推定し、各モデル完了時に表示する。
  実測時間は「実行記録.json」に保存。距離自体の再計算は含まない。

重要: exp(beta*score)は未校正の重みであり、会員の尤度ではない。
第5回検証の分母は第1〜4回だけで条件付け、第5回の8負例・群条件を使わない。
モデル案自体は第5回の後に考案しているので、検証は回顧的である。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
UPSTREAM = HERE.parent / '14_MIA希少の距離と対応固有性'
SUBMISSIONS = ROOT / '予備戦/攻撃フェーズ/提出記録'
KIT = ROOT / 'スターターキット/改訂版_20260824/starter/src'
sys.path.insert(0, str(KIT))
from pwscup2026.scoring.attack_score import _tpr_at_fpr


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_result(path):
    text = path.read_text(encoding='utf-8')
    start = text.index('{')
    return json.JSONDecoder().raw_decode(text[start:])[0]


def read_scores(path, rare_ids, teams):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == 1049, f'{path}: 1049行でない')
    by_id = {int(row['record_id']): row for row in rows}
    require(len(by_id) == 1049, f'{path}: ID重複')
    return {k: np.array([float(by_id[i][str(k)]) for i in rare_ids]) for k in teams}


def load_inputs():
    facts = read_json(UPSTREAM / '検証結果.json')
    rows = read_json(UPSTREAM / '距離と対応の診断.json')
    ids = facts['eligible_ids']
    rare_ids = sorted(ids + facts['private_rare_ids'])
    require(len(ids) == 11 and len(rare_ids) == 24, '候補・希少人数')
    ix = {rid: i for i, rid in enumerate(ids)}
    base = [r for r in rows if r['metric'] == 'l1_cat1' and r['record_id'] in ix]
    teams = sorted({r['target_k'] for r in base})
    require(len(base) == 220 and len(teams) == 20, '20チーム×11人')
    old = {(r['target_k'], ix[r['record_id']]) for r in base
           if r['status'] == '第1〜3回からの既知非会員'}
    new = {(r['target_k'], ix[r['record_id']]) for r in base
           if r['status'] == '第5回からの既知非会員'}
    require(len(old) == 32 and len(new) == 8 and not old & new, '32+8負例')
    evidence = {(r['target_k'], ix[r['record_id']]): r['round5_mean_evidence'] for r in base}
    specificity = {}
    for k, i in product(teams, range(11)):
        rr = [r for r in rows if r['target_k'] == k and r['record_id'] == ids[i]]
        require(len(rr) == 3, '固有性は3設定を平均')
        specificity[k, i] = sum(r['specificity_log_ratio'] for r in rr) / 3
    require(all(math.isfinite(v) for v in evidence.values()), '距離スコアの欠損')
    names = ['1_20260902_Seo', '2_20260906_Seo', '3_20260906_Tashiro',
             '4_20260906_Seo_2', '5_20260906_Codex']
    score_paths = [SUBMISSIONS / n / ('第4回目/F_mia.csv' if j == 3 else 'F_mia.csv')
                   for j, n in enumerate(names)]
    result_paths = [SUBMISSIONS / n / 'result.txt' for n in names]
    scores = [read_scores(p, rare_ids, teams) for p in score_paths]
    results = [read_result(p) for p in result_paths]
    input_paths = [UPSTREAM / name for name in
                   ['検証結果.json', '距離と対応の診断.json', '入力ファイルSHA256.json']]
    input_paths += score_paths + result_paths + [KIT / 'pwscup2026/scoring/attack_score.py',
                                               KIT / 'pwscup2026/kit/codabench/scoring_io.py']
    # 上流の元データが変わっていれば、古い診断JSONで黙って続けない。
    upstream_hashes = read_json(UPSTREAM / '入力ファイルSHA256.json')
    for rel, expected in upstream_hashes.items():
        require(sha256(ROOT / rel) == expected, f'上流入力の変更: {rel}')
        input_paths.append(ROOT / rel)
    # 既存の提出記録について内容とファイル集合の不変性を検証する。tokenは出力しない。
    protected_paths = [p for n in names for p in (SUBMISSIONS / n).rglob('*') if p.is_file()]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in set(input_paths + protected_paths)}
    return dict(ids=ids, rare_ids=rare_ids, teams=teams, old=old, new=new,
                evidence=evidence, specificity=specificity, scores=scores,
                results=results, hashes=hashes, protected_names=names)


def official_utility(score, members, rare_ids, ids):
    actual = {ids[i] for i in members}
    truth = np.array([int(rid in actual) for rid in rare_ids])
    return _tpr_at_fpr(score, truth, .01)[0] if actual else 0.0


def audit_history(data):
    """既存の結論を提出CSVと公式採点関数から再照合する。"""
    ids, rare, teams = data['ids'], data['rare_ids'], data['teams']
    old, new, scores = data['old'], data['new'], data['scores']
    derived_old = set()
    for j in range(3):
        require(data['results'][j]['mia']['tpr_rare'] == 0, '第1〜3回は希少0')
        for k in teams:
            top = max(scores[j][k])
            derived_old |= {(k, i) for i, rid in enumerate(ids)
                            if scores[j][k][rare.index(rid)] == top}
    require(derived_old == old, '第1〜3回の最高点の和集合と32負例')
    checked = 0
    for k in teams:
        for r in (1, 2):
            for members in combinations(range(11), r):
                observed_zero = all(official_utility(scores[j][k], members, rare, ids) == 0
                                    for j in range(3))
                require(observed_zero == all((k, i) not in old for i in members),
                        '第1〜3回の0点制約と32負例の同値性')
                checked += 1
    # 第4回: 他19チームは11人同点、Team20は第1回と同じ。
    require(np.array_equal(scores[3][20], scores[0][20]), '第4回Team20')
    for k in teams:
        if k != 20:
            vals = [scores[3][k][rare.index(rid)] for rid in ids]
            private_vals = [scores[3][k][j] for j, rid in enumerate(rare) if rid not in ids]
            require(len(set(vals)) == 1 and min(vals) > max(private_vals), '第4回11人同点')
    mean4 = data['results'][3]['mia']['tpr_rare']
    solutions4 = []
    for a, b, r20 in product(range(20), range(20), range(3)):
        count = a + b + int(r20 > 0)
        if not count or a + b > 19 or a + 2*b + r20 > 11:
            continue
        value = (a*.23/10 + b*.22/9) / count
        if abs(value - mean4) < 1e-13:
            solutions4.append((a, b, r20))
    require(solutions4 == [(3, 2, 1), (3, 2, 2)], '第4回人数条件の一意性')
    chosen, groups = set(), {10: [], 9: [], 'chosen': []}
    minimum_hit = []
    for k in teams:
        top = max(scores[4][k])
        tops = [i for i, rid in enumerate(ids) if scores[4][k][rare.index(rid)] == top]
        require(top > 0, '第5回の最高値が正')
        if len(tops) <= 2:
            chosen.update((k, i) for i in tops)
            if k != 20:
                groups['chosen'].append(k)
            possible_values = []
            for r in (1, 2):
                for members in combinations(range(11), r):
                    if set(tops) & set(members) and all((k, i) not in old for i in members):
                        possible_values.append(official_utility(scores[4][k], members, rare, ids))
            minimum_hit.append(min(possible_values))
        else:
            require(len(tops) in (9, 10), '第5回の同点候補数')
            groups[len(tops)].append(k)
            allowed = {i for i in range(11) if (k, i) not in old}
            require(set(tops) == allowed, '同点群は旧負例以外全員')
    require(chosen == new, '第5回選択8組と新負例')
    total5 = data['results'][4]['mia']['tpr_rare'] * 6
    require(min(minimum_hit) > total5, '一組でも当たった場合の下限と矛盾')
    solutions5 = []
    terms = [Fraction(23, 900), Fraction(22, 800), Fraction(23, 800), Fraction(22, 700)]
    for a, b, c, d in product(range(4), range(3), range(4), range(3)):
        if a+c <= 3 and b+d <= 2:
            if abs(float(sum(n*t for n, t in zip((a, b, c, d), terms))) - total5) < 1e-13:
                solutions5.append((a, b, c, d))
    require(solutions5 == [(1, 0, 1, 1)], '第5回群条件の一意性')
    return groups, dict(old_zero_equivalence_cases=checked, count_solutions_round4=solutions4,
                        count_solutions_round5=solutions5, minimum_hit_tpr=min(minimum_hit),
                        rare_mean_round4=mean4, rare_mean_round5=total5/6,
                        derived_old_negative_pairs=32, derived_new_negative_pairs=8,
                        groups=groups)


def options(k, r, allowed, n=11):
    for people in combinations([i for i in range(n) if (k, i) in allowed], r):
        yield sum(1 << i for i in people), tuple((k, i) for i in people)


def make_groups(data, groups, prior):
    allowed = set(product(data['teams'], range(11))) - data['old'] - data['new']
    local = [[(m, e, 1) for k in groups[10] for m, e in options(k, 1, allowed)]]
    for group in [groups[9], groups['chosen']]:
        states = []
        for k1, k2 in permutations(group, 2):
            for m1, e1 in options(k1, 1, allowed):
                for m2, e2 in options(k2, 2, allowed):
                    if not m1 & m2:
                        states.append((m1 | m2, e1 + e2, 1))
        local.append(states)
    # r20=1/2を除外前に半々とするなら、個別割当の事前重みは3:2。
    # ID割当数は11!/(3!2!2!)と11!/(2!2!2!2!)で、後者が1.5倍。
    prior_weights = {1: 3, 2: 2} if prior == 'balanced' else {1: 1, 2: 1}
    local.append([(m, e, prior_weights[r]) for r in (1, 2) for m, e in options(20, r, allowed)])
    return local


def convolve(left, right):
    """同じ患者を使わない群だけ掛け合わせる（整数入力なら整数のまま）。"""
    out = defaultdict(int)
    for m, w in left.items():
        for n, v in right.items():
            if not m & n:
                out[m | n] += w*v
    return dict(out)


def joint_posterior(data, groups, signal, beta, prior):
    local = make_groups(data, groups, prior)
    weighted, polys = [], []
    for states in local:
        terms, pol = [], defaultdict(int)
        for mask, edges, w in states:
            if beta:
                w *= math.exp(beta * sum(signal[e] for e in edges))
            terms.append((mask, edges, w))
            pol[mask] += w
        weighted.append(terms)
        polys.append(dict(pol))
    complete = {0: 1}
    for pol in polys:
        complete = convolve(complete, pol)
    z = sum(complete.values())
    require(z > 0 and math.isfinite(z), '正規化定数')
    subsets = {k: defaultdict(float) for k in data['teams']}
    for g, terms in enumerate(weighted):
        outside = {0: 1}
        for h, pol in enumerate(polys):
            if g != h:
                outside = convolve(outside, pol)
        mass = {m: sum(v for n, v in outside.items() if not m & n) for m in polys[g]}
        check = 0.0
        for mask, edges, weight in terms:
            p = weight * mass[mask] / z
            check += p
            by_team = defaultdict(list)
            for k, i in edges:
                by_team[k].append(i)
            for k, members in by_team.items():
                subsets[k][tuple(sorted(members))] += p
        require(abs(check - 1) < 1e-10, '群の確率和')
    marg = {}
    for k, dist in subsets.items():
        dist[()] = max(0.0, 1 - sum(dist.values()))
        require(abs(sum(dist.values()) - 1) < 1e-10, 'チーム内の確率和')
        for i in range(11):
            marg[k, i] = sum(p for members, p in dist.items() if i in members)
    p20two = sum(p for members, p in subsets[20].items() if len(members) == 2)
    require(abs(sum(1-dist[()] for dist in subsets.values()) - 6) < 1e-9, '共有6チーム')
    require(abs(sum(marg.values()) - 8 - p20two) < 1e-9, '共有8〜9人')
    for i in range(11):
        require(sum(marg[k, i] for k in data['teams']) <= 1+1e-9, '各患者は最大1チーム')
    require(all(marg[e] == 0 for e in data['old'] | data['new']), '既知負例の確率0')
    return z, marg, subsets, [len(x) for x in local]


def round4_partition(data, signal, beta, prior):
    """第4回までの条件のみ。第5回の群や8負例を引数にも受け取らない。

    dp[a,b,mask]: 処理済み対象で1人共有aチーム・2人共有bチームの重み。
    最大a=3,b=2。19チームを順に足し、最後にTeam20の1/2人を足す。
    """
    teams, old = data['teams'], data['old']
    allowed = set(product(teams, range(11))) - old
    masks = np.arange(1 << 11)
    transitions = {}
    for r in (1, 2):
        for indices in combinations(range(11), r):
            mask = sum(1 << i for i in indices)
            src = masks[(masks & mask) == 0]
            transitions[mask] = (src, src | mask)
    dp = np.zeros((4, 3, 1 << 11), dtype=float)
    dp[0, 0, 0] = 1
    for k in teams:
        if k == 20:
            continue
        nxt = dp.copy()  # このチームが0人の選択肢。
        for r in (1, 2):
            for mask, edges in options(k, r, allowed):
                src, dst = transitions[mask]
                w = math.exp(beta * sum(signal[e] for e in edges)) if beta else 1
                if r == 1:
                    nxt[1:][:, :, dst] += dp[:-1][:, :, src] * w
                else:
                    nxt[:, 1:][:, :, dst] += dp[:, :-1][:, :, src] * w
        dp = nxt
    z_by_r = {}
    for r in (1, 2):
        z_by_r[r] = sum(float(dp[3, 2, transitions[mask][0]].sum()) *
                       (math.exp(beta * sum(signal[e] for e in edges)) if beta else 1)
                       for mask, edges in options(20, r, allowed))
    weights = {1: 3, 2: 2} if prior == 'balanced' else {1: 1, 2: 1}
    z = sum(z_by_r[r] * weights[r] for r in (1, 2))
    require(z > 0 and math.isfinite(z), '第4回の正規化')
    return z, z_by_r


def unique_utility(first, second, members):
    """全候補を異なる点数で並べる場合、r<=2では上位2人だけが効く。"""
    if first not in members:
        return 0.0
    return 1.0 if len(members) == 1 or second in members else .5


def evaluate_decisions(data, marg, subsets):
    out = []
    for k in data['teams']:
        candidates = [i for i in range(11) if (k, i) not in data['old'] | data['new']]
        dist = subsets[k]
        pairs = []
        for a, b in permutations(candidates, 2):
            utility = sum(p * unique_utility(a, b, members) for members, p in dist.items())
            pairs.append((utility, a, b))
        best, a, b = min(pairs, key=lambda x: (-x[0], data['ids'][x[1]], data['ids'][x[2]]))
        ranking = sorted(candidates, key=lambda i: (-marg[k, i], data['ids'][i]))
        marginal_u = sum(p * unique_utility(*ranking[:2], members) for members, p in dist.items())
        tied_u = sum(p * .01*(24-len(members))/(len(candidates)-len(members))
                     for members, p in dist.items() if members)
        fifth_u = sum(p * official_utility(data['scores'][4][k], members, data['rare_ids'], data['ids'])
                      for members, p in dist.items())
        uplift = {}
        for all_tpr in (0, .25, .75):
            # 全体TPRが変更前後でこの値に固定されるという仮定の下の差。
            # 希少の変更は全体ROCにも影響するため、実際の増分を予測する値ではない。
            uplift[str(all_tpr)] = sum(p * (max(all_tpr, unique_utility(a, b, members)) -
                                           max(all_tpr, official_utility(data['scores'][4][k], members,
                                                                       data['rare_ids'], data['ids'])))
                                       for members, p in dist.items())
        out.append(dict(target_k=k, p_active=1-dist[()],
                        p_one=sum(p for m, p in dist.items() if len(m) == 1),
                        p_two=sum(p for m, p in dist.items() if len(m) == 2),
                        best_unique_first=data['ids'][a], best_unique_second=data['ids'][b],
                        best_unique_expected_rare_tpr=best,
                        p_both_first_second=dist.get(tuple(sorted((a, b))), 0),
                        marginal_order_first_two=[data['ids'][i] for i in ranking[:2]],
                        marginal_order_expected_rare_tpr=marginal_u,
                        all_remaining_tied_expected_rare_tpr=tied_u,
                        round5_expected_rare_tpr=fifth_u,
                        conditional_uplift_if_all_tpr_fixed=uplift))
    require(abs(sum(r['round5_expected_rare_tpr'] for r in out)/6 -
                data['results'][4]['mia']['tpr_rare']) < 1e-11, '事後で第5回結果が一致')
    return out


def run(betas):
    start = perf_counter()
    print('計算時間の目安: 3〜15秒（保存済み距離を使用）。提出ファイルは変更しません。', flush=True)
    data = load_inputs()
    groups, audit = audit_history(data)
    print(f'入力・過去結果の照合完了: {perf_counter()-start:.2f}秒。32+8負例と群条件が一致。', flush=True)
    specs = [('constraints', None, 0, 'balanced'), ('constraints_uniform', None, 0, 'uniform')]
    for kind in ('evidence', 'specificity'):
        specs += [(f'{kind}_beta{beta:g}', kind, beta, 'balanced') for beta in betas]
    models, probabilities, subset_rows, decisions, timings = [], [], [], [], []
    elapsed_models = 0.0
    for j, (name, kind, beta, prior) in enumerate(specs):
        t = perf_counter()
        signal = data[kind] if kind else {}
        z5, marg, subsets, local_counts = joint_posterior(data, groups, signal, beta, prior)
        z4, by_r = round4_partition(data, signal, beta, prior)
        event_p = z5/z4
        require(0 < event_p <= 1+1e-10, '第5回観測の予測確率')
        choice = evaluate_decisions(data, marg, subsets)
        for row in choice:
            decisions.append(dict(model=name, **row))
        for k, i in product(data['teams'], range(11)):
            status = '第1〜3回の非会員' if (k, i) in data['old'] else (
                '第5回の非会員' if (k, i) in data['new'] else '所属未確定')
            probabilities.append(dict(model=name, target_k=k, record_id=data['ids'][i],
                                      status=status, model_probability=marg[k, i]))
        for k, dist in subsets.items():
            for members, p in sorted(dist.items()):
                subset_rows.append(dict(model=name, target_k=k,
                                        member_ids=[data['ids'][i] for i in members], probability=p))
        models.append(dict(model=name, signal=kind, beta=beta, prior=prior,
                           partition_round4=z4, partition_round5=z5,
                           round4_partition_by_r20_unweighted=by_r,
                           round5_observed_event_predictive_probability=event_p,
                           negative_log_predictive_probability=-math.log(event_p),
                           expected_rare_mean_best_unique=sum(r['best_unique_expected_rare_tpr'] for r in choice)/6,
                           expected_rare_mean_all_tied=sum(r['all_remaining_tied_expected_rare_tpr'] for r in choice)/6,
                           expected_rare_mean_round5=sum(r['round5_expected_rare_tpr'] for r in choice)/6,
                           local_state_counts=local_counts))
        elapsed = perf_counter()-t
        timings.append(dict(model=name, elapsed_seconds=elapsed))
        elapsed_models += elapsed
        eta = elapsed_models/(j+1)*(len(specs)-j-1)
        print(f'[{j+1}/{len(specs)}] {name}: {elapsed:.2f}秒、残り約{eta:.1f}秒。'
              f' 第5回結果のモデル内予測確率={event_p:.6g}', flush=True)
    baseline = models[0]['round5_observed_event_predictive_probability']
    for row in models:
        row['predictive_probability_ratio_to_baseline'] = row['round5_observed_event_predictive_probability']/baseline
    # βの最適化や自動選択はしない。一つの集約観測では校正を保証できない。
    now_paths = [p for name in data['protected_names'] for p in (SUBMISSIONS/name).rglob('*') if p.is_file()]
    protected_set = {str(p.relative_to(ROOT)) for p in now_paths}
    old_protected = {p for p in data['hashes'] if p.startswith('予備戦/攻撃フェーズ/提出記録/')}
    require(protected_set == old_protected, '既存提出記録のファイル集合')
    require(all(sha256(ROOT/p) == h for p, h in data['hashes'].items()), '入力・提出記録の不変性')
    audit.update(known_members=0, new_membership_facts=0, unknown_pairs=180,
                 official_results_reproduced=True, input_and_submissions_unchanged=True,
                 past5_excluded_from_round4_denominator=True,
                 retrospective_not_prospective_validation=True,
                 calibrated_membership_probabilities=False)
    runtime = dict(elapsed_seconds=perf_counter()-start, model_timings=timings,
                   python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                   model_count=len(specs), estimated_seconds_before_run=[3, 15],
                   analysis_code_sha256=sha256(Path(__file__)),
                   excludes_distance_recomputation=True)
    return {'モデル比較.json': models, '各組のモデル内確率.json': probabilities,
            'チーム内の同時確率.json': subset_rows, '期待希少TPRと順位.json': decisions,
            '検証結果.json': audit, '実行記録.json': runtime, '入力ファイルSHA256.json': data['hashes']}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--no-write', action='store_true', help='結果を画面に表示するだけ。ファイル更新なし')
    parser.add_argument('--betas', type=float, nargs='+', default=[.5, 1, 4, 16],
                        help='重みの感度比較。0超16以下。最適化・自動選択ではない')
    args = parser.parse_args()
    require(len(set(args.betas)) == len(args.betas) and all(0 < b <= 16 for b in args.betas), 'βは重複なし、0超16以下')
    outputs = run(args.betas)
    if not args.no_write:
        for name, value in outputs.items():
            (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print('分析フォルダ内のJSON 7件だけを保存しました。提出CSV・ZIPの変更なし。')
    print(f"実測合計: {outputs['実行記録.json']['elapsed_seconds']:.2f}秒")


if __name__ == '__main__':
    main()
