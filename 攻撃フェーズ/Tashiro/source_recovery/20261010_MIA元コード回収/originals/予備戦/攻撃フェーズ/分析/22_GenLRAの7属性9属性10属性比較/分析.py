"""Gen-LRAの7/9/10属性比較。分析JSON以外は生成・更新しない。

実行: python3 -B 分析.py --save   # 比較結果.jsonを新規保存（上書き拒否）
再確認: python3 -B 分析.py        # ファイルを一切書き込まない
必要: NumPy, pandas。Codex同梱Pythonで実行可能。通信なし。
計算時間の予想: 20〜90秒。20対象の9属性と喫煙の分解、希少の同時分布を計算。
7属性・10属性は分析21の検証済み結果を、入力ハッシュを確認して再利用する。
各対象終了時に実測時間と残り計算時間の目安を表示。

主比較: 7属性、7+BMI+FPGの9属性、9+喫煙の10属性。h倍率1、k=10、rho=.98。
9/10は数値7列とhが同じ。7/9は自動hも変わり、純粋な固定帯域比較ではない。
喫煙の分解では近傍のみ・カーネルのみを替えた診断値を計算するが、提出案にはしない。
正解ラベルがない部分のTPRは推定済みと呼ばない。新しい個別会員を確定しない。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
PREVIOUS = HERE.parent/'21_MIA全体と希少のGenLRA比較'
spec = importlib.util.spec_from_file_location('analysis21', PREVIOUS/'分析.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
ROOT, DATA, DIST, h, gen = a.ROOT, a.DATA, a.DIST, a.h, a.gen
NUM9 = gen.FEATURES[1][1]
CAT9 = gen.FEATURES[0][2]
PRIMARY = ['qi7', 'attrs9', 'attrs10']


def smoke_decomposition(bg, c, q):
    """9→10属性の影響を密度カーネルと近傍選択に分解。結果を調整しない。"""
    gx = gen.numeric(bg, NUM9)
    scale = (np.quantile(gx, .75, axis=0)-np.quantile(gx, .25, axis=0))/1.349
    gx, cx, qx = gx/scale, gen.numeric(c, NUM9)/scale, gen.numeric(q, NUM9)/scale
    bw = (len(bg)*(len(NUM9)+2)/4)**(-1/(len(NUM9)+4))
    freq = gen.category_frequencies(bg, CAT9+['smoking'])
    cg, cq = gen.distance2(cx, gx), gen.distance2(cx, qx)
    cg_log = gen.log_kernel(cg, gen.log_category(c, bg, CAT9, freq), bw, len(NUM9))
    cq_log = gen.log_kernel(cq, gen.log_category(c, q, CAT9, freq), bw, len(NUM9))
    smoke_cg = gen.log_category(c, bg, ['smoking'], freq)
    smoke_cq = gen.log_category(c, q, ['smoking'], freq)
    delta9 = gen.influence(cq_log, (gen.logsumexp(cg_log, 1)-math.log(len(bg)))[:, None], len(bg)).T
    delta10 = gen.influence(cq_log+smoke_cq,
        (gen.logsumexp(cg_log+smoke_cg, 1)-math.log(len(bg)))[:, None], len(bg)).T
    d9 = cq.T.copy()
    for cat in CAT9:
        d9 += q[cat].to_numpy()[:, None] != c[cat].to_numpy()[None, :]
    mismatch = q.smoking.to_numpy()[:, None] != c.smoking.to_numpy()[None, :]
    near9 = np.argsort(d9, axis=1, kind='stable')[:, :10]
    near10 = np.argsort(d9+mismatch, axis=1, kind='stable')[:, :10]
    def aggregate(delta, near):
        return np.take_along_axis(delta, near, axis=1).sum(axis=1)
    return dict(attrs9=aggregate(delta9, near9), attrs10=aggregate(delta10, near10),
        smoking_kernel_only=aggregate(delta10, near9), smoking_neighbors_only=aggregate(delta9, near10),
        neighbors_shared=np.array([len(set(x)&set(y)) for x, y in zip(near9, near10)]),
        smoking_match9=1-np.take_along_axis(mismatch, near9, 1).mean(axis=1),
        smoking_match10=1-np.take_along_axis(mismatch, near10, 1).mean(axis=1),
        bandwidth=bw)


def posterior(data, groups, known, raw, plans, old_top):
    """同一の6仮定で全案を評価。患者単位の重複共有禁止も維持する。"""
    specs = [('constraints_balanced', None, 0, 'balanced'),
             ('constraints_uniform', None, 0, 'uniform'),
             ('gen7_rank_weak', 'qi7', 1, 'balanced'),
             ('gen7_rank_medium', 'qi7', 2, 'balanced'),
             ('gen9_rank_weak', 'attrs9', 1, 'balanced'),
             ('gen10_rank_weak', 'attrs10', 1, 'balanced')]
    ranks = {(k, name): a.rank(v) for k, vals in raw.items() for name, v in vals.items()}
    result = []
    for name, metric, beta, prior in specs:
        local = h.make_groups(data, groups, prior)
        terms_by_group = []
        for states in local:
            terms = []
            for mask, edges, weight in states:
                if any(e in known for e in edges):
                    continue
                members = defaultdict(list)
                for k, i in edges:
                    members[k].append(i)
                old_units = sum(a.utility_units(old_top[k], ms) for k, ms in members.items())
                if old_units > 1 or any(a.utility_units(old_top[k], ms) for k, ms in members.items() if k in (1, 2)):
                    continue
                if metric:
                    weight *= math.exp(beta*sum((11-ranks[k, metric][i])/10 for k, i in edges))
                terms.append((mask, old_units, members, weight))
            terms_by_group.append(terms)
        results = {}
        norm = None
        for label, pairs in plans.items():
            whole = {(0, 0, 0): 1.}
            for terms in terms_by_group:
                poly = defaultdict(float)
                for mask, old_u, members, weight in terms:
                    new_u = sum(a.utility_units(pairs[k], ms) for k, ms in members.items())
                    poly[mask, old_u, new_u] += weight
                whole = a.convolve(whole, poly)
            z = sum(v for (mask, old_u, new_u), v in whole.items() if old_u == 1)
            assert z > 0 and (norm is None or abs(z/norm-1) < 1e-10)
            norm = z
            dist = defaultdict(float)
            for (mask, old_u, new_u), value in whole.items():
                if old_u == 1:
                    dist[new_u] += value/z
            assert abs(sum(dist.values())-1) < 1e-9
            results[label] = dict(expected=sum(u*p/12 for u, p in dist.items()),
                zero=dist[0], improve=sum(p for u, p in dist.items() if u > 1),
                minimum=min(u for u, p in dist.items() if p > 0)/12,
                maximum=max(u for u, p in dist.items() if p > 0)/12,
                distribution={str(u/12): p for u, p in sorted(dist.items()) if p > 0})
        result.append(dict(model=name, normalizer=norm, plans=results))
        print(f'希少の共有条件: {name} 照合完了', flush=True)
    assert abs(result[1]['normalizer']-585179373) < .01
    return result


def run():
    start = perf_counter()
    print('予想20〜90秒。7/9/10属性を比較。回答CSV・ZIPは作成しません。', flush=True)
    previous = h.read_json(PREVIOUS/'比較結果.json')
    extra = h.read_json(PREVIOUS/'追加検証結果.json')
    hashes = previous['inputs_sha256'].copy()
    assert all(h.sha256(ROOT/p) == v for p, v in hashes.items()), '既存計算の入力が変わっています'
    for path in [PREVIOUS/'比較結果.json', PREVIOUS/'追加検証結果.json', Path(__file__)]:
        hashes[str(path.relative_to(ROOT))] = h.sha256(path)
    protected = {str(p.relative_to(ROOT)): h.sha256(p) for p in a.RECORDS.rglob('*') if p.is_file()}
    data = h.load_inputs()
    groups, audit = h.audit_history(data)
    b = pd.read_csv(DATA/'PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/B_self.csv', float_precision='round_trip')
    bg = pd.read_csv(DATA/'PWSCup2026_prelim_common/A_bg.csv', float_precision='round_trip')
    ch = pd.read_csv(DIST/'aia_challenge_5.csv', float_precision='round_trip')
    qi = gen.FEATURES[0][1]+CAT9
    private_ids = set(b.merge(ch[qi], on=qi, validate='one_to_one').record_id.astype(int))
    ids = b.record_id.to_numpy(int)
    assert ids.tolist() == previous['record_ids']
    private = np.isin(ids, list(private_ids))
    rare = b.is_rare.to_numpy(bool)
    use = ~private & ~rare
    assert (len(bg), len(b), int(private.sum()), int(rare.sum()), int(use.sum())) == (5143, 1049, 163, 24, 875)
    eligible = data['ids']
    pos = {int(rid): j for j, rid in enumerate(ids)}
    eligible_pos = np.array([pos[rid] for rid in eligible])
    known = data['old'] | data['new'] | {(r['team'], eligible.index(r['record_id'])) for r in previous['new_negative_pairs']}
    assert len(known) == 48
    # 第6回の全体帯と総得点も再確認。1・2を希少の的中先から除く条件。
    observed = h.read_result(a.R6/'result.txt')
    assert observed['mia']['tpr_rare'] == 1/12
    assert observed['mia_per_target']['1'] == observed['mia_per_target']['2'] == 4
    assert .2140-.00005-np.mean(list(observed['aia_per_target'].values())) > observed['mia']['tpr_at_low_fpr']
    plans = {name: {int(k): tuple(eligible.index(rid) for rid in pair) for k, pair in plan.items()}
             for name, plan in previous['rare_plans'].items()}
    plans['gen9_after48'] = {}
    plans['gen7_keep12_refresh_second'] = {int(k): tuple(eligible.index(rid) for rid in pair)
        for k, pair in extra['previous_keep12_plan'].items()}
    raws, rare_raw, full, rare_rows, details = {}, {}, [], [], []
    max_error = independent_error = 0.
    counts = Counter()
    prefectures = sorted(set(b.loc[private, 'prefecture']))
    same_pref = b.prefecture.isin(prefectures).to_numpy() & ~rare
    for step, k in enumerate(data['teams'], 1):
        c = pd.read_csv(DIST/f'C_{k}.csv', float_precision='round_trip')
        computed9 = a.compute(bg, c, b, NUM9, CAT9)
        independent_error = max(independent_error, computed9['independent_error'])
        parts = smoke_decomposition(bg, c, b)
        values = {'qi7': np.array(previous['raw_scores'][str(k)]['qi7_gen']),
                  'attrs9': computed9['gen'], 'attrs10': np.array(previous['raw_scores'][str(k)]['attrs10_gen'])}
        for feature in ['attrs9', 'attrs10']:
            error = float(np.max(np.abs(values[feature]-parts[feature])))
            assert error < 1e-10
            max_error = max(max_error, error)
        independent = {'smoking_kernel_only': parts['smoking_kernel_only'],
                       'smoking_neighbors_only': parts['smoking_neighbors_only']}
        raws[str(k)] = {name: v.tolist() for name, v in {'attrs9': values['attrs9'], **independent}.items()}
        rare_raw[k] = {name: v[eligible_pos] for name, v in values.items()}
        remain = [i for i in range(11) if (k, i) not in known]
        diagnostics = {}
        top_sets = {}
        for name, scores in {**values, **independent}.items():
            assert np.isfinite(scores).all()
            raw_order = a.order(scores, ids, ~rare)
            remaining_order = a.order(scores, ids, use)
            pref_order = a.order(scores, ids, same_pref)
            top_sets[name] = set(remaining_order[:20])
            diagnostics[name] = dict(top20=[int(ids[j]) for j in remaining_order[:20]],
                private_in_raw_top={str(n): int(private[raw_order[:n]].sum()) for n in [10, 20, 40]},
                private_ids_raw_top20=[int(ids[j]) for j in raw_order[:20] if private[j]],
                within_private_pref_top20_negative=int(private[pref_order[:20]].sum()),
                top20_boundary_tie_size=int(sum(scores[~rare] == scores[raw_order[19]])),
                overlap_baseline_top20=len(set(diagnostics.get('baseline', {}).get('top20', []))))
            baseline_order = a.order(np.array(previous['raw_scores'][str(k)]['round6']), ids, use)
            diagnostics[name]['overlap_baseline_top20'] = len(set(remaining_order[:20]) & set(baseline_order[:20]))
            if name in PRIMARY:
                order11 = sorted(range(11), key=lambda i: (-scores[eligible_pos[i]], eligible[i]))
                allowed_order = [i for i in order11 if i in remain]
                assert len(set(scores[eligible_pos[remain]])) == len(remain), '希少候補に同点あり: 効用式を見直す'
                diagnostics[name]['rare_top2'] = [eligible[i] for i in allowed_order[:2]]
                diagnostics[name]['raw_top11_negative'] = (k, order11[0]) in known
                diagnostics[name]['private_rare_above_best_remaining'] = int((private & rare & (scores > scores[eligible_pos[allowed_order[0]]])).sum())
                if name == 'attrs9':
                    plans['gen9_after48'][k] = tuple(allowed_order[:2])
                else:
                    assert plans['gen7_after48' if name == 'qi7' else 'gen10_after48'][k] == tuple(allowed_order[:2])
        full.append(dict(team=k, methods=diagnostics,
            overlap7_9_top20=len(top_sets['qi7'] & top_sets['attrs9']),
            overlap9_10_top20=len(top_sets['attrs9'] & top_sets['attrs10']),
            overlap9_neighbor_only_top20=len(top_sets['attrs9'] & top_sets['smoking_neighbors_only']),
            overlap10_kernel_only_top20=len(top_sets['attrs10'] & top_sets['smoking_kernel_only'])))
        for i, rid in enumerate(eligible):
            rare_rows.append(dict(team=k, record_id=rid, status='x' if (k, i) in known else '?',
                **{name: float(values[name][pos[rid]]) for name in PRIMARY},
                **{name+'_rank11': int(a.rank(rare_raw[k][name])[i]) for name in PRIMARY}))
        details.append(dict(team=k, bandwidth9=parts['bandwidth'], bandwidth10=parts['bandwidth'],
            mean_neighbors_shared=float(parts['neighbors_shared'][use].mean()),
            identical_neighbor_sets=int(sum(parts['neighbors_shared'][use] == 10)),
            mean_smoking_match9=float(parts['smoking_match9'][use].mean()),
            mean_smoking_match10=float(parts['smoking_match10'][use].mean())))
        for pair in set(plan[k] for plan in plans.values()):
            order = list(pair)+[i for i in remain if i not in pair]
            s = np.zeros(24)
            for j, i in enumerate(order):
                s[data['rare_ids'].index(eligible[i])] = 1-j/24
            for r in (1, 2):
                for members in combinations(remain, r):
                    assert abs(h.official_utility(s, members, data['rare_ids'], eligible)-a.utility_units(pair, members)/2) < 1e-14
                    counts['rare_utility_checks'] += 1
        elapsed = perf_counter()-start
        print(f'{step}/20対象完了、経過{elapsed:.1f}秒、残り計算約{elapsed/step*(20-step):.1f}秒', flush=True)
    models = posterior(data, groups, known, rare_raw, plans, plans['round6'])
    previous_models = {r['model']: r for r in extra['rare_models']}
    for model in models:
        if model['model'] in previous_models:
            old = previous_models[model['model']]
            for label, vals in old['plans'].items():
                for field in ['expected', 'zero', 'improve']:
                    assert abs(vals[field]-model['plans'][label][field]) < 1e-12
        assert abs(model['plans']['round6']['expected']-1/12) < 1e-12
        assert model['plans']['gen7_keep12_refresh_second']['minimum'] >= 1/12
    assert all(h.sha256(ROOT/p) == digest for p, digest in {**hashes, **protected}.items())
    assert {str(p.relative_to(ROOT)) for p in a.RECORDS.rglob('*') if p.is_file()} == set(protected)
    return dict(seconds=perf_counter()-start, full_results=full, rare_rows=rare_rows, smoking_decomposition=details,
        plans={name: {str(k): [eligible[i] for i in pair] for k, pair in plan.items()} for name, plan in plans.items()},
        rare_models=models, raw_new_scores=raws, record_ids=ids.tolist(),
        population=dict(nonrare=1025, private_nonrare=150, unknown_nonrare=875, private_prefectures=prefectures,
            nonrare_in_private_prefectures=int(same_pref.sum()), private_smoking_counts=b.loc[private & ~rare, 'smoking'].value_counts().to_dict(),
            unknown_smoking_counts=b.loc[use, 'smoking'].value_counts().to_dict()),
        validation=dict(**counts, decomposition_max_error=max_error, independent9_max_error=independent_error,
            unchanged_protected_files=len(protected), input_hashes_unchanged=True, history=audit),
        input_hashes=hashes, full_tpr_unknown=True,
        unavailable_auxiliary_columns=[col for col in ['BMI', 'FPG', 'smoking'] if col not in ch])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    opt = parser.parse_args()
    output = HERE/'比較結果.json'
    if opt.save and output.exists():
        raise FileExistsError('既存結果は上書きしません。--saveを外して再検証してください。')
    result = run()
    if opt.save:
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
    print(json.dumps({k: result[k] for k in ['seconds', 'population', 'validation', 'full_tpr_unknown', 'unavailable_auxiliary_columns']}, ensure_ascii=False))
