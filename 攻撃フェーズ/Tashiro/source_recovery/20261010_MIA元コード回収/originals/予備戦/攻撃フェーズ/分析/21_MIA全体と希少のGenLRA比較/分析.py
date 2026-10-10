"""MIA全体・希少のGen-LRA比較。提出物を変更せずJSONの分析結果だけ保存する。

実行: Codex同梱Pythonで python3 -B 分析.py --save
再検証: python3 -B 分析.py           # ファイルを生成・更新しない
必要: NumPy, pandas。旧分析の純粋関数と改訂版公式採点関数を利用。
時間予想: 20チーム×1,049人×2属性設定の計算と希少の同時分布に30秒〜3分。
終了チーム数から計算の残り時間を表示。初期予想は実測で更新する。
新規JSONの上書きは拒否。CSV・ZIP・AIA・配布データには書き込まない。

主設定QI7/h1/k10、感度比較attrs10/h1/k10を事前固定。
Gen-LRAは独自混合カーネル適用版。確信度は校正済み確率ではない。
負例を除外する前の診断と除外後の候補を区別する。
自チーム課題にはQI7しかないので10属性の答え合わせを捏造しない。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
from itertools import combinations
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
RECORDS = ROOT / '予備戦/攻撃フェーズ/提出記録'
R6 = RECORDS / '6_20260907_GenLRA_Zhou'
DATA = ROOT / '予備戦/配布データ'
DIST = DATA / 'PWSCup2026_prelim_attack'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = module('gen20', HERE.parent / '20_Team20希少のGenLRA検証/分析.py')
h = gen.history
from pwscup2026.scoring.attack_score import score_mia


def rank(values):
    return pd.Series(values).rank(ascending=False, method='min').to_numpy(int)


def order(values, ids, use):
    return sorted(np.flatnonzero(use), key=lambda i: (-values[i], ids[i]))


def compute(bg, c, q, nums, cats):
    """参照密度の尺度を固定。スカラーの追加密度式でも独立に照合する。"""
    gx = gen.numeric(bg, nums)
    scale = (np.quantile(gx, .75, axis=0)-np.quantile(gx, .25, axis=0))/1.349
    assert (scale > 0).all()
    gx, cx, qx = gx/scale, gen.numeric(c, nums)/scale, gen.numeric(q, nums)/scale
    for col in cats:
        assert set(c[col]) | set(q[col]) <= set(bg[col]), col
    freq = gen.category_frequencies(bg, cats)
    cg, cq = gen.distance2(cx, gx), gen.distance2(cx, qx)
    qg = gen.distance2(qx, gx)
    bandwidth = (len(bg)*(len(nums)+2)/4)**(-1/(len(nums)+4))
    logk_cg = gen.log_kernel(cg, gen.log_category(c, bg, cats, freq), bandwidth, len(nums))
    logp_c = gen.logsumexp(logk_cg, 1)-math.log(len(bg))
    logk_cq = gen.log_kernel(cq, gen.log_category(c, q, cats, freq), bandwidth, len(nums))
    d2 = cq.T.copy()
    for col in cats:
        d2 += q[col].to_numpy()[:, None] != c[col].to_numpy()[None, :]
    near = np.argsort(d2, axis=1, kind='stable')[:, :10]
    delta = gen.influence(logk_cq, logp_c[:, None], len(bg)).T
    local = np.take_along_axis(delta, near, axis=1)
    score = local.sum(axis=1)
    logp_q = gen.logsumexp(gen.log_kernel(qg, gen.log_category(q, bg, cats, freq), bandwidth, len(nums)), 1)-math.log(len(bg))
    logc_q = gen.logsumexp(gen.log_kernel(cq.T, gen.log_category(q, c, cats, freq), bandwidth, len(nums)), 1)-math.log(len(c))
    # 明示的なR+x密度との照合。極端値でもexpの桁落ちを避けるため最大log密度を共通に引く。
    max_error = 0.
    check_rows = sorted(set([0, len(q)-1, int(np.argmax(score)), int(np.argmin(score))]))
    for j in check_rows:
        total = 0.
        for s in near[j]:
            logs = np.r_[logk_cg[s], logk_cq[s, j]]
            shift, before_shift = logs.max(), logs[:-1].max()
            augmented_logp = shift+math.log(np.exp(logs-shift).mean())
            before_logp = before_shift+math.log(np.exp(logs[:-1]-before_shift).mean())
            total += augmented_logp-before_logp
        max_error = max(max_error, abs(total-score[j]))
    assert max_error < 1e-10
    return dict(gen=score, distance=-np.sqrt(d2.min(axis=1)/(len(nums)+len(cats))),
                point=logc_q-logp_q, independent_error=max_error, bandwidth=bandwidth,
                top_positive_rows=(local > 0).sum(axis=1))


def convolve(a, b):
    out = defaultdict(float)
    for (mask, old_u, new_u), w in a.items():
        for (other, x, y), z in b.items():
            if old_u+x <= 1 and not mask & other:
                out[mask | other, old_u+x, new_u+y] += w*z
    return dict(out)


def utility_units(pair, members):
    if pair[0] not in members:
        return 0
    return 2 if len(members) == 1 or pair[1] in members else 1


def rare_analysis(data, groups, old_top, known, raw, plans):
    """第6回の結果で条件付け。患者の重複共有禁止・各群人数を同時に維持する。"""
    models = [('constraints_balanced', None, 0, 'balanced'),
              ('constraints_uniform', None, 0, 'uniform'),
              ('gen7_rank_weak', 'qi7', 1, 'balanced'),
              ('gen7_rank_medium', 'qi7', 2, 'balanced'),
              ('gen10_rank_weak', 'attrs10', 1, 'balanced')]
    result = []
    ranks = {(k, metric): rank(values) for k, features in raw.items() for metric, values in features.items()}
    for name, metric, beta, prior in models:
        local = h.make_groups(data, groups, prior)
        groups_terms = []
        for group in local:
            terms = []
            for mask, edges, w in group:
                if any(e in known for e in edges):
                    continue
                per = defaultdict(list)
                for k, i in edges:
                    per[k].append(i)
                old_units = sum(utility_units(old_top[k], ms) for k, ms in per.items())
                if old_units > 1 or any(utility_units(old_top[k], ms) for k, ms in per.items() if k in (1, 2)):
                    continue
                if metric:
                    w *= math.exp(beta*sum((11-ranks[k, metric][i])/10 for k, i in edges))
                terms.append((mask, old_units, per, w))
            groups_terms.append(terms)
        outputs = {}
        normalizer = None
        for label, pairs in plans.items():
            whole = {(0, 0, 0): 1.}
            for terms in groups_terms:
                poly = defaultdict(float)
                for mask, old_units, per, w in terms:
                    new_units = sum(utility_units(pairs[k], ms) for k, ms in per.items())
                    poly[mask, old_units, new_units] += w
                whole = convolve(whole, poly)
            z = sum(w for (m, old, new), w in whole.items() if old == 1)
            assert z > 0
            if normalizer is None:
                normalizer = z
            assert abs(z/normalizer-1) < 1e-9
            dist = defaultdict(float)
            for (m, old, new), w in whole.items():
                if old == 1:
                    dist[new] += w/z
            assert abs(sum(dist.values())-1) < 1e-9
            outputs[label] = dict(expected=sum(u*p/12 for u, p in dist.items()),
                zero=dist[0], improve=sum(p for u, p in dist.items() if u > 1),
                minimum=min(u for u, p in dist.items() if p > 0)/12,
                maximum=max(u for u, p in dist.items() if p > 0)/12,
                distribution={str(u/12): p for u, p in sorted(dist.items()) if p > 0})
        result.append(dict(model=name, normalizer=normalizer, plans=outputs))
        print(f'希少の同時条件を照合: {name}', flush=True)
    assert abs(result[1]['normalizer']-585179373) < .01
    return result


def run():
    start = perf_counter()
    print('予想30秒〜3分。MIA全体・希少を比較。提出物・AIAは変更しません。', flush=True)
    protected = {str(p.relative_to(ROOT)): h.sha256(p) for p in RECORDS.rglob('*') if p.is_file()}
    data = h.load_inputs()
    groups, audit = h.audit_history(data)
    paths = [DATA/'PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/B_self.csv',
             DATA/'PWSCup2026_prelim_common/A_bg.csv', DIST/'aia_challenge_5.csv',
             HERE/'既検討の希少案.json', Path(__file__), Path(gen.__file__), Path(h.__file__)]
    paths += [DIST/f'C_{k}.csv' for k in [5]+data['teams']]
    hashes = {str(p.relative_to(ROOT)): h.sha256(p) for p in paths}
    b, bg, challenge = [pd.read_csv(p, float_precision='round_trip') for p in paths[:3]]
    submitted = pd.read_csv(R6/'F_mia.csv', float_precision='round_trip')
    round5 = pd.read_csv(RECORDS/'5_20260906_Codex/F_mia.csv', float_precision='round_trip')
    official = h.read_result(R6/'result.txt')
    ids = b.record_id.to_numpy(int)
    qi = gen.FEATURES[0][1]+gen.FEATURES[0][2]
    private_ids = set(b.merge(challenge[qi], on=qi, validate='one_to_one').record_id.astype(int))
    private = np.isin(ids, list(private_ids))
    rare = b.is_rare.to_numpy(bool)
    assert len(b) == len(submitted) == 1049 and len(bg) == 5143
    assert submitted.record_id.tolist() == ids.tolist()
    assert private.sum() == 163 and (private & rare).sum() == 13 and rare.sum() == 24
    assert np.array_equal(round5.loc[~rare].to_numpy(), submitted.loc[~rare].to_numpy())
    assert (submitted.loc[private].iloc[:, 1:].to_numpy() == 0).all()
    eligible = data['ids']
    positions = {int(v): i for i, v in enumerate(ids)}
    old_raw = h.read_json(R6/'MIA/希少24人のスコア.json')
    old_by_pair = {(r['team'], r['record_id']): r for r in old_raw}
    old_top = {}
    for k in data['teams']:
        vals = submitted[str(k)].to_numpy()
        ordered_ids = sorted(eligible, key=lambda rid: (-vals[positions[rid]], rid))
        old_top[k] = tuple(eligible.index(rid) for rid in ordered_ids[:2])
        positive = vals[rare & ~private & (vals > 0)]
        assert len(set(positive)) == len(positive)  # 正の未確定候補は同点なし。0の人数は対象ごとに違う。
    # Team1・2が希少の唯一の的中先なら希少.5<全体.75なので上乗せゼロとなり公式合計と矛盾。
    aia_power = float(np.mean(list(official['aia_per_target'].values())))
    overall = official['mia']['tpr_at_low_fpr']
    assert .2140-.00005-aia_power > overall
    assert official['mia']['tpr_rare'] == 1/12
    assert official['mia_per_target']['1'] == official['mia_per_target']['2'] == 4
    excluded = set(groups[10]) | {1, 2}
    old_known = data['old'] | data['new']
    new_known = {(k, old_top[k][0]) for k in excluded}
    assert len(new_known) == 8 and not old_known & new_known
    known = old_known | new_known
    assert len(known) == 48
    all_results, raw_rare, record_scores = [], {}, {}
    exact_error = 0.
    independent_error = 0.
    for step, k in enumerate(data['teams'], 1):
        c = pd.read_csv(DIST/f'C_{k}.csv', float_precision='round_trip')
        methods = {}
        for feature, nums, cats in gen.FEATURES:
            methods[feature] = compute(bg, c, b, nums, cats)
            independent_error = max(independent_error, methods[feature]['independent_error'])
        for rid in ids[rare]:
            exact_error = max(exact_error, abs(methods['qi7']['gen'][positions[rid]]-old_by_pair[k, rid]['genlra']))
        baseline = submitted[str(k)].to_numpy()
        use = ~private & ~rare
        baseline_order = order(baseline, ids, use)
        summary = dict(team=k, baseline_top20=[int(ids[j]) for j in baseline_order[:20]], methods={})
        record_scores[str(k)] = {'round6': baseline.tolist()}
        raw_rare[k] = {}
        for feature, methods_data in methods.items():
            score = methods_data['gen']
            ordered = order(score, ids, use)
            raw_nonrare = order(score, ids, ~rare)
            private_top = {str(n): int(private[raw_nonrare[:n]].sum()) for n in (10, 20, 40)}
            cutoff_count = {str(n): int((private & ~rare & (score >= score[ordered[n-1]])).sum()) for n in (10, 20, 40)}
            summary['methods'][feature] = dict(top20=[int(ids[j]) for j in ordered[:20]],
                overlap_baseline_top20=len(set(ordered[:20]) & set(baseline_order[:20])),
                private_in_raw_top=private_top, private_at_or_above_candidate_cutoff=cutoff_count,
                raw_nonrare_unique_scores=len(set(score[~rare])),
                raw_top20_private_ids=[int(ids[j]) for j in raw_nonrare[:20] if private[j]],
                positive_gen_count=int((score[use] > 0).sum()))
            raw_rare[k][feature] = np.array([score[positions[rid]] for rid in eligible])
            for metric in ['gen', 'distance', 'point']:
                record_scores[str(k)][f'{feature}_{metric}'] = methods_data[metric].tolist()
        summary['overlap_gen7_gen10_top20'] = len(set(summary['methods']['qi7']['top20']) & set(summary['methods']['attrs10']['top20']))
        all_results.append(summary)
        elapsed = perf_counter()-start
        print(f'全体・希少 {step}/20チーム: 経過{elapsed:.1f}秒、全体計算残り約{elapsed/step*(20-step):.1f}秒', flush=True)
    assert exact_error < 1e-10
    # 自チームの課題はQI7のみ。会員はBへの一意照合、未一致163人を対照とする補助検証。
    labeled = challenge.merge(b[qi+['record_id']], on=qi, how='left', validate='one_to_one')
    truth = labeled.record_id.notna().to_numpy(int)
    assert truth.sum() == 163 and len(truth) == 326
    control_b_matches = int(labeled.loc[truth == 0, 'record_id'].notna().sum())
    assert control_b_matches == 0
    c5 = pd.read_csv(DIST/'C_5.csv', float_precision='round_trip')
    auxiliary = compute(bg, c5, challenge, gen.FEATURES[0][1], gen.FEATURES[0][2])
    aux_metrics = {metric: dict(zip(['tpr_at_fpr_1pct', 'auc'], h._tpr_at_fpr(auxiliary[metric], truth, .01)))
                   for metric in ['gen', 'distance', 'point']}
    # QI7/10の差は生順位から比較。既知負例を下げたことを攻撃精度とは数えない。
    plans = {'round6': old_top, 'gen7_after48': {}, 'gen10_after48': {}, 'previous_joint_proposal': {}}
    previous = h.read_json(HERE/'既検討の希少案.json')['top2']
    rare_rows = []
    checks = 0
    for k in data['teams']:
        remaining = [i for i in range(11) if (k, i) not in known]
        for feature, plan in [('qi7', 'gen7_after48'), ('attrs10', 'gen10_after48')]:
            ix = sorted(remaining, key=lambda i: (-raw_rare[k][feature][i], eligible[i]))
            plans[plan][k] = tuple(ix[:2])
        plans['previous_joint_proposal'][k] = tuple(eligible.index(rid) for rid in previous[str(k)])
        for i, rid in enumerate(eligible):
            rare_rows.append(dict(team=k, record_id=rid, status='x' if (k, i) in known else '?',
                gen7=float(raw_rare[k]['qi7'][i]), gen10=float(raw_rare[k]['attrs10'][i]),
                gen7_rank11=int(rank(raw_rare[k]['qi7'])[i]), gen10_rank11=int(rank(raw_rare[k]['attrs10'])[i])))
        # 全体と希少を同時に変える際、希少24値を維持すれば希少TPRだけは厳密に維持できる。
        original = submitted[str(k)].to_numpy()
        changed = original.copy()
        use = ~private & ~rare
        vals = np.array(record_scores[str(k)]['qi7_gen'])
        ord_idx = order(vals, ids, use)
        rare_cap = float(original[rare].max())
        assert rare_cap < 1
        # 検査用の単調写像。同点は同点のまま。38人以上を希少の上に配置する構成の例。
        ranks = pd.Series(vals[use]).rank(method='min').to_numpy()
        changed[use] = rare_cap+(1-rare_cap)*(.1+.8*ranks/len(ranks))
        if k in (1, 2):
            changed = original.copy()
        assert np.array_equal(changed[rare], original[rare]) and (changed[private] == 0).all()
        assert int((changed[~rare] > changed[rare].max()).sum()) >= 38
        for pair in set(plan[k] for plan in plans.values()):
            allowed_order = list(pair)+[i for i in remaining if i not in pair]
            scores = np.zeros(24)
            for j, i in enumerate(allowed_order):
                scores[data['rare_ids'].index(eligible[i])] = 1-j/24
            for r in (1, 2):
                for ms in combinations(remaining, r):
                    assert abs(h.official_utility(scores, ms, data['rare_ids'], eligible)-utility_units(pair, ms)/2) < 1e-14
                    checks += 1
        # 仮ラベルは採点実装の検査専用。性能評価に使わない。
        for seed in range(12):
            rng = np.random.default_rng(10000*k+seed)
            y = np.zeros(1049, int)
            r = seed % 3
            positives = list(rng.choice([positions[eligible[i]] for i in remaining], r, replace=False))
            positives += list(rng.choice(np.flatnonzero(use), 25+seed % 3-r, replace=False))
            y[positives] = 1
            before = score_mia(original, y, {'scoring': {'low_fpr': .01}}, rare_mask=rare).breakdown
            after = score_mia(changed, y, {'scoring': {'low_fpr': .01}}, rare_mask=rare).breakdown
            assert r == 0 or before['tpr_rare'] == after['tpr_rare']
            if k in (1, 2):
                assert before['tpr_at_low_fpr'] == after['tpr_at_low_fpr']
    posterior = rare_analysis(data, groups, old_top, known, raw_rare, plans)
    for model in posterior:
        assert abs(model['plans']['round6']['expected']-1/12) < 1e-12
        assert model['plans']['gen7_after48']['minimum'] >= 1/12
    combined_hashes = {**data['hashes'], **protected, **hashes}
    assert all(h.sha256(ROOT/p) == sha for p, sha in combined_hashes.items())
    assert {str(p.relative_to(ROOT)) for p in RECORDS.rglob('*') if p.is_file()} == set(protected)
    return dict(seconds=perf_counter()-start, history=audit,
        counts=dict(all=1049, rare=24, private=163, private_rare=13, unknown_nonrare=875,
                    rare_candidates=11, background=5143, known_negative_pairs=48),
        new_negative_pairs=[dict(team=k, record_id=eligible[i]) for k, i in sorted(new_known)],
        official_round6=official['mia'], other18_all_tpr_mean_upper=(20*overall-1.5)/18,
        full_results=all_results, rare_comparison=rare_rows,
        rare_plans={label: {str(k): [eligible[i] for i in pair] for k, pair in plan.items()} for label, plan in plans.items()},
        rare_models=posterior, auxiliary_c5_qi7=aux_metrics,
        validation=dict(gen7_round6_max_error=exact_error, independent_gen_max_error=independent_error,
            rare_utility_scenarios=checks, joint_full_scenarios=240, protected_files=len(protected),
            protected_files_unchanged=True, record_ids_consistent=True),
        inputs_sha256=combined_hashes, record_ids=ids.tolist(), raw_scores=record_scores)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--save', action='store_true')
    opt = args.parse_args()
    if opt.save and (HERE/'比較結果.json').exists():
        raise FileExistsError('既存の比較結果.jsonは上書きしません。--saveを外すと読み取り専用で再検証できます。')
    result = run()
    if opt.save:
        with (HERE/'比較結果.json').open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            f.write('\n')
    compact = {k: v for k, v in result.items() if k not in ['raw_scores', 'inputs_sha256', 'record_ids', 'rare_comparison']}
    print(json.dumps(compact, ensure_ascii=False, allow_nan=False))
