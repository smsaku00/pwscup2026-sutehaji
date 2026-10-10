"""Team20のMIA希少だけを対象に、混合型KDE版Gen-LRAを検証する。

実行: python3 -B 分析.py              # NumPy・pandas。ネット接続不要
      python3 -B 分析.py --no-write   # 計算・照合のみ。既存出力も更新しない

計算時間の予想: 10〜120秒。背景5,143行×C1,049行の密度計算6組と、
13,440通りの共有人数配置の照合を行う。各設定群の終了時に実測からETA表示。
計算はこのフォルダのJSONだけを出力する。提出CSV・ZIP・配布データを変更しない。
既存の分析16から読み取り・履歴監査の純粋関数だけを利用する。

主設定: 公開QI7属性、Silverman係数×1、近傍k=10。
感度分析: QI7/属性10 × 帯域倍率0.5/1/2 × k=1/10/20（全18設定）。
カテゴリは保持重みrho=.98の正規化した混合カーネル。結果を見る前に固定。
論文の順序符号化を再現するのではなく、県などを名義尺度として扱う適用版。
出典: https://arxiv.org/html/2508.21146v2 （式6、4.6節）

Gen-LRA値は参照KDEへの追加の影響であり、会員確率・真の加工の尤度比ではない。
既知負例を下げる設定探索や、未知の候補を正解ラベルとする評価はしない。
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import importlib.util
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = ROOT/'予備戦/配布データ'
DIST = DATA/'PWSCup2026_prelim_attack'
SPEC = importlib.util.spec_from_file_location('history16', HERE.parent/'16_MIA希少の制約付き同時確率/分析.py')
history = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(history)  # mainは実行しない。

FEATURES = [('qi7', ['age', 'SBP', 'TG', 'HDL', 'ALT'], ['sex', 'prefecture']),
            ('attrs10', ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG'], ['sex', 'prefecture', 'smoking'])]
FACTORS = [.5, 1., 2.]
KS = [1, 10, 20]
RHO = .98
PRIMARY = 'qi7_h1_k10'


def read_frame(path, hashes):
    hashes[str(path.relative_to(ROOT))] = history.sha256(path)
    return pd.read_csv(path)


def numeric(frame, cols):
    x = frame[cols].to_numpy(float).copy()
    for col in ['TG', 'ALT']:
        if col in cols:
            x[:, cols.index(col)] = np.log1p(x[:, cols.index(col)])
    assert np.isfinite(x).all()
    return x


def distance2(x, y):
    out = np.zeros((len(x), len(y)))
    for j in range(x.shape[1]):
        out += (x[:, j, None]-y[None, :, j])**2
    return out


def logsumexp(x, axis):
    maximum = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(maximum+np.log(np.exp(x-maximum).sum(axis=axis, keepdims=True)), axis=axis)


def category_frequencies(bg, cols):
    # 背景に全カテゴリが存在することを別途確認。加算1平滑化後も和は1。
    return {col: {str(v): (int(n)+1)/(len(bg)+bg[col].nunique())
                  for v, n in bg[col].value_counts().items()} for col in cols}


def log_category(query, reference, cats, frequencies):
    """log K(s_cat | r_cat)。queryとreferenceの方向は交換できない。"""
    out = np.zeros((len(query), len(reference)))
    for col in cats:
        freq = np.array([frequencies[col][str(v)] for v in query[col]])
        match = query[col].to_numpy()[:, None] == reference[col].to_numpy()[None, :]
        out += np.log(RHO*match+(1-RHO)*freq[:, None])
    return out


def log_kernel(d2, logcat, h, d):
    return -.5*d2/(h*h)-d*math.log(h)-d/2*math.log(2*math.pi)+logcat


def influence(logk, logp, m):
    # 固定カーネルでの p_aug=(m*p+K)/(m+1) を厳密に計算（Taylor近似なし）。
    return np.logaddexp(0., logk-logp-math.log(m))-math.log1p(1/m)


def ordered(ids, scores, allowed=None):
    use = set(ids if allowed is None else allowed)
    return [int(ids[j]) for j in sorted(range(len(ids)), key=lambda j: (-scores[j], ids[j])) if ids[j] in use]


def minimum_rank(ids, scores, rid, allowed):
    value = scores[ids.index(rid)]
    return 1+int(sum(scores[ids.index(a)] > value for a in allowed))


def compute_feature(name, nums, cats, bg, c, rare, ids, eligible, remaining):
    gx = numeric(bg, nums)
    scale = (np.quantile(gx, .75, axis=0)-np.quantile(gx, .25, axis=0))/1.349
    assert (scale > 0).all()
    gx, cx, qx = gx/scale, numeric(c, nums)/scale, numeric(rare, nums)/scale
    for col in cats:
        assert set(c[col]) | set(rare[col]) <= set(bg[col]), f'未知カテゴリ: {col}'
    freq = category_frequencies(bg, cats)
    cg2, cq2 = distance2(cx, gx), distance2(cx, qx)
    qg2 = distance2(qx, gx)
    cgcat, cqcat = log_category(c, bg, cats, freq), log_category(c, rare, cats, freq)
    qgcat, qccat = log_category(rare, bg, cats, freq), log_category(rare, c, cats, freq)
    # 密度とは別に、標準化L2+カテゴリ不一致1の固定距離でCの近傍を選ぶ。
    d2 = cq2.T.copy()
    for col in cats:
        d2 += rare[col].to_numpy()[:, None] != c[col].to_numpy()[None, :]
    neighbors = np.argsort(d2, axis=1, kind='stable')[:, :max(KS)]
    dcr = -np.sqrt(d2.min(axis=1)/(len(nums)+len(cats)))
    base_h = (len(bg)*(len(nums)+2)/4)**(-1/(len(nums)+4))
    config = dict(feature=name, numeric=nums, categorical=cats, scale=scale.tolist(),
                  transformed_log1p=['TG', 'ALT'], silverman_factor=base_h,
                  rho=RHO, category_frequencies=freq,
                  exact_rare_background_matches=len(rare.merge(bg, on=nums+cats)),
                  exact_c_background_matches=len(c.merge(bg, on=nums+cats)))
    rows, settings, contributions, baselines = [], [], [], []
    for factor in FACTORS:
        h = base_h*factor
        logp_c = logsumexp(log_kernel(cg2, cgcat, h, len(nums)), 1)-math.log(len(bg))
        logk_cq = log_kernel(cq2, cqcat, h, len(nums))
        delta = influence(logk_cq, logp_c[:, None], len(bg)).T
        assert np.isfinite(delta).all()
        local = np.take_along_axis(delta, neighbors, axis=1)
        # 点でのKDE密度比。Gen-LRAとは評価位置・集約方法が異なる比較対象。
        logp_q = logsumexp(log_kernel(qg2, qgcat, h, len(nums)), 1)-math.log(len(bg))
        logp_c_at_q = logsumexp(log_kernel(cq2.T, qccat, h, len(nums)), 1)-math.log(len(c))
        point_ratio = logp_c_at_q-logp_q
        for j, rid in enumerate(ids):
            baselines.append(dict(feature=name, bandwidth_factor=factor, record_id=rid,
                                  dcr_score=float(dcr[j]), dcr_rank11=minimum_rank(ids, dcr, rid, eligible),
                                  pointwise_log_density_ratio=float(point_ratio[j]),
                                  pointwise_rank11=minimum_rank(ids, point_ratio, rid, eligible)))
        for k in KS:
            setting = f'{name}_h{factor:g}_k{k}'
            score = local[:, :k].sum(axis=1)
            rank11, rank9 = ordered(ids, score, eligible), ordered(ids, score, remaining)
            private = sorted(set(ids)-set(eligible))
            best9score = score[ids.index(rank9[0])]
            settings.append(dict(setting=setting, feature=name, bandwidth_factor=factor,
                                 bandwidth=h, k=k, primary=setting == PRIMARY,
                                 order11=rank11, order9=rank9,
                                 negative201_rank=minimum_rank(ids, score, 201, eligible),
                                 negative238_rank=minimum_rank(ids, score, 238, eligible),
                                 private_above_best_remaining=[rid for rid in private if score[ids.index(rid)] > best9score],
                                 private_nonnegative=int(sum(score[ids.index(rid)] >= 0 for rid in private))))
            for j, rid in enumerate(ids):
                rows.append(dict(setting=setting, record_id=rid, influence=float(score[j]),
                                 status='専用希少の既知非会員' if rid in private else
                                        ('過去結果からの既知非会員' if rid not in remaining else '未確定候補'),
                                 rank24=minimum_rank(ids, score, rid, ids),
                                 rank11=minimum_rank(ids, score, rid, eligible) if rid in eligible else None,
                                 rank9=minimum_rank(ids, score, rid, remaining) if rid in remaining else None,
                                 positive_contribution_rows=int((local[j, :k] > 0).sum()),
                                 largest_positive_share=(float(max(0, local[j, :k].max())/np.maximum(local[j, :k], 0).sum())
                                                         if (local[j, :k] > 0).any() else None)))
            if setting == PRIMARY:
                for j, rid in enumerate(ids):
                    for order, crow in enumerate(neighbors[j, :k], 1):
                        contributions.append(dict(record_id=rid, neighbor_rank=order, c_row_index=int(crow),
                                                  c_csv_line=int(crow)+2,
                                                  distance=float(np.sqrt(d2[j, crow]/(len(nums)+len(cats)))),
                                                  log_kernel=float(logk_cq[crow, j]),
                                                  log_reference_density=float(logp_c[crow]),
                                                  log_density_gain=float(delta[j, crow])))
    return config, rows, settings, contributions, baselines


def feasible_counts(active, allowed):
    # 各teamのr個のスロットに別々の患者を割り当てられるか、一般化Hall条件。
    for mask in range(1, 1 << len(active)):
        available, required = 0, 0
        for j, (team, r) in enumerate(active):
            if mask & (1 << j):
                available |= allowed[team]
                required += r
        if available.bit_count() < required:
            return False
    return True


def tied_tpr(n, r):
    return Fraction(24-r, 100*(n-r))


def score_bounds(data, groups):
    allowed = {k: sum(1 << i for i in range(11) if (k, i) not in data['old'] | data['new']) for k in data['teams']}
    counts = {k: mask.bit_count() for k, mask in allowed.items()}
    out = []
    for a, (b1, b2), (c1, c2), r20 in product(groups[10], permutations(groups[9], 2),
                                             permutations(groups['chosen'], 2), (1, 2)):
        active = [(a, 1), (b1, 1), (b2, 2), (c1, 1), (c2, 2), (20, r20)]
        if not feasible_counts(active, allowed):
            continue
        other = sum((tied_tpr(counts[k], r) for k, r in active if k != 20), Fraction())/6
        all_tied = other+tied_tpr(counts[20], r20)/6
        out.append((r20, other, all_tied))
    assert len(out) == 13440
    def interval(values):
        lo, hi = min(values), max(values)
        return dict(min=float(lo), max=float(hi), min_exact=str(lo), max_exact=str(hi))
    return dict(remaining_counts=counts, feasible_count_arrangements=len(out), rare_mean_denominator=6,
                round5_observed=data['results'][4]['mia']['tpr_rare'],
                all20_tied=interval([v[2] for v in out]),
                other19_tied=interval([v[1] for v in out]),
                team20_unique_first_false=interval([v[1] for v in out]),
                team20_unique_one_of_two=interval([v[1]+Fraction(1, 12) for v in out if v[0] == 2]),
                team20_unique_all_true=interval([v[1]+Fraction(1, 6) for v in out]),
                caveat='既存の所属条件が正しい場合の数式上の範囲。固定した選択IDに条件付けた両端の達成可能性は未主張。')


def state_utilities(data, settings):
    order = next(s['order9'] for s in settings if s['primary'])
    rare_ids, eligible = data['rare_ids'], data['ids']
    # 提出ではない。順位付け対同点の公式関数による仮の効用比較。
    unique = np.array([(len(order)-order.index(rid))/len(order) if rid in order else 0 for rid in rare_ids])
    tied = np.array([int(rid in order) for rid in rare_ids], float)
    states = []
    for r in (1, 2):
        for members in combinations(sorted(order), r):
            ix = tuple(eligible.index(rid) for rid in members)
            utility = history.official_utility(unique, ix, rare_ids, eligible)
            expected = (int(order[0] in members)+int(r == 2 and set(order[:2]) <= set(members)))/r
            assert abs(utility-expected) < 1e-14
            tie_value = history.official_utility(tied, ix, rare_ids, eligible)
            assert abs(tie_value-float(tied_tpr(9, r))) < 1e-14
            states.append(dict(members=list(members), unique_order_tpr=utility, all9_tied_tpr=tie_value))
    return dict(unique_order=order, states=states,
                note='45状態は正解仮定の列挙。正解ラベルでも、状態の確率でもない。')


def run():
    started = perf_counter()
    print('予想10〜120秒。Team20希少24人、主設定1＋感度分析17。提出物は更新しません。', flush=True)
    data = history.load_inputs()
    groups, audit = history.audit_history(data)
    hashes = data['hashes'].copy()
    for path in [Path(__file__), HERE.parent/'16_MIA希少の制約付き同時確率/分析.py']:
        hashes[str(path.relative_to(ROOT))] = history.sha256(path)
    b = read_frame(DATA/'PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/B_self.csv', hashes)
    bg = read_frame(DATA/'PWSCup2026_prelim_common/A_bg.csv', hashes)
    c = read_frame(DIST/'C_20.csv', hashes)
    ch5 = read_frame(DIST/'aia_challenge_5.csv', hashes)
    assert len(bg) == 5143 and len(c) == len(b) == 1049
    rare = b.loc[b.is_rare.eq(1)].sort_values('record_id').reset_index(drop=True)
    ids = rare.record_id.astype(int).tolist()
    assert ids == data['rare_ids'] and len(ids) == 24
    qi = FEATURES[0][1]+FEATURES[0][2]
    private = b.merge(ch5[qi], on=qi, validate='one_to_one')
    assert len(private) == 163
    private_ids = sorted(private.loc[private.is_rare.eq(1), 'record_id'].astype(int))
    eligible = data['ids']
    assert set(private_ids) == set(ids)-set(eligible) and len(private_ids) == 13
    remaining = [rid for i, rid in enumerate(eligible) if (20, i) not in data['old'] | data['new']]
    assert sorted(set(eligible)-set(remaining)) == [201, 238]
    configs, rows, settings, contributions, baselines = [], [], [], [], []
    setup_time = perf_counter()-started
    for j, (name, nums, cats) in enumerate(FEATURES, 1):
        result = compute_feature(name, nums, cats, bg, c, rare, ids, eligible, remaining)
        conf, rr, ss, cc, bb = result
        configs.append(conf); rows.extend(rr); settings.extend(ss); contributions.extend(cc); baselines.extend(bb)
        elapsed = perf_counter()-started
        eta = (elapsed-setup_time)/j*(len(FEATURES)-j)
        print(f'{name}: {len(ss)}設定完了、経過{elapsed:.1f}秒。密度計算残り約{eta:.1f}秒＋採点条件照合。', flush=True)
    bounds = score_bounds(data, groups)
    utility = state_utilities(data, settings)
    # 既存入力・提出記録の不変性。トークン内容は読み出して表示しない。
    assert all(history.sha256(ROOT/rel) == value for rel, value in hashes.items())
    before_names = {p for p in hashes if any(p.startswith(f'予備戦/攻撃フェーズ/提出記録/{n}/') for n in data['protected_names'])}
    after_names = {str(p.relative_to(ROOT)) for n in data['protected_names'] for p in (history.SUBMISSIONS/n).rglob('*') if p.is_file()}
    assert before_names == after_names
    primary = next(s for s in settings if s['primary'])
    summary = dict(primary=primary, eligible_ids=eligible, remaining_ids=remaining, private_rare_ids=private_ids,
                   n_rare=24, n_reference=len(bg), n_settings=len(settings), n_known_positive_labels=0,
                   top11_counts=dict(Counter(s['order11'][0] for s in settings)),
                   top9_counts=dict(Counter(s['order9'][0] for s in settings)),
                   history_audit=audit, protected_files_unchanged=True,
                   candidate_probabilities_estimated=False, new_submission_performed=False)
    elapsed = perf_counter()-started
    print(json.dumps(dict(primary=primary, top9_counts=summary['top9_counts'], all20_tied=bounds['all20_tied'],
                          elapsed_seconds=elapsed), ensure_ascii=False, indent=2), flush=True)
    return {'固定条件と前処理.json': dict(primary=PRIMARY, factors=FACTORS, ks=KS, features=configs),
            '希少24人の全設定順位.json': rows, '設定比較.json': settings,
            '主設定の近傍別寄与.json': contributions, '同条件の距離と点密度比.json': baselines,
            '希少スコアの範囲.json': bounds, 'Team20の45状態と採点.json': utility,
            '検証結果.json': summary, '入力ファイルSHA256.json': hashes,
            '実行記録.json': dict(elapsed_seconds=elapsed, estimated_range_seconds=[10, 120],
                               python=sys.version, numpy=np.__version__, pandas=pd.__version__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args()
    outputs = run()
    if not args.no_write:
        for name, content in outputs.items():
            (HERE/name).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
