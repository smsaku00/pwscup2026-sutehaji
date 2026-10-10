"""比較結果を独立照合し、全体・希少同時変更の採点条件を検査する。

python3 -B 検証.py --save  # このフォルダの追加検証結果.jsonだけを新規作成
python3 -B 検証.py         # 再照合のみ（ファイルは更新しない）
予想時間5〜30秒。仮ラベルは採点の単体テストだけに使用し、攻撃性能として報告しない。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
import importlib.util
spec = importlib.util.spec_from_file_location('analysis21', HERE/'分析.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def run():
    start = perf_counter()
    result = a.h.read_json(HERE/'比較結果.json')
    assert all(a.h.sha256(a.ROOT/p) == digest for p, digest in result['inputs_sha256'].items())
    protected = {p: a.h.sha256(p) for p in a.RECORDS.rglob('*') if p.is_file()}
    data = a.h.load_inputs()
    groups, _ = a.h.audit_history(data)
    ids = result['record_ids']
    eligible = data['ids']
    pos = {rid: j for j, rid in enumerate(ids)}
    b = pd.read_csv(a.DATA/'PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/B_self.csv', float_precision='round_trip')
    challenge = pd.read_csv(a.DIST/'aia_challenge_5.csv', float_precision='round_trip')
    qi = a.gen.FEATURES[0][1]+a.gen.FEATURES[0][2]
    private_ids = set(b.merge(challenge[qi], on=qi, validate='one_to_one').record_id.astype(int))
    private = np.isin(ids, list(private_ids))
    rare = b.is_rare.to_numpy(bool)
    unknown_nonrare = ~private & ~rare
    rare_positions = [pos[rid] for rid in data['rare_ids']]
    plans = {name: {int(k): tuple(eligible.index(rid) for rid in pair) for k, pair in plan.items()}
             for name, plan in result['rare_plans'].items()}
    known = data['old'] | data['new'] | {(r['team'], eligible.index(r['record_id'])) for r in result['new_negative_pairs']}
    raw = {k: {metric: np.array([result['raw_scores'][str(k)][metric+'_gen'][pos[rid]] for rid in eligible])
               for metric in ['qi7', 'attrs10']} for k in data['teams']}
    # 前の会話の「12チームの1位を保ち、2位は旧3位へ」の案も復元。
    # 2位の一律非会員認定ではない。唯一の的中先だった条件下でのみ旧2位が非会員となる。
    plans['gen7_keep12_refresh_second'] = {}
    for k in data['teams']:
        old_order = sorted(range(11), key=lambda i: (-result['raw_scores'][str(k)]['round6'][pos[eligible[i]]], eligible[i]))
        if (k, old_order[0]) in known:
            pair = plans['gen7_after48'][k]
        else:
            pair = (old_order[0], old_order[2])
        assert not any((k, i) in known for i in pair)
        plans['gen7_keep12_refresh_second'][k] = pair
    post = a.rare_analysis(data, groups, plans['round6'], known, raw, plans)
    for old, new in zip(result['rare_models'], post):
        assert old['model'] == new['model']
        for label in old['plans']:
            for quantity in ['expected', 'zero', 'improve']:
                assert abs(old['plans'][label][quantity]-new['plans'][label][quantity]) < 1e-12
        assert new['plans']['gen7_keep12_refresh_second']['minimum'] >= 1/12

    # 同点を含む別実装のROC補間。標本配置に依存する性能値は出力しない。
    def tpr_reference(scores, truth):
        positive = int(sum(truth))
        if positive == 0 or positive == len(truth):
            return 0.
        counts = defaultdict(lambda: [0, 0])
        for score, y in zip(scores, truth):
            counts[float(score)][int(y)] += 1
        target = .01*(len(truth)-positive)
        fp = tp = 0
        for score, (n, p) in sorted(counts.items(), reverse=True):
            if fp+n > target:
                return (tp+p*(target-fp)/n)/positive
            fp += n
            tp += p
        return 1.

    full_checks = rare_checks = reference_checks = 0
    diagnostics = []
    for k in data['teams']:
        allowed = [i for i in range(11) if (k, i) not in known]
        original = np.array(result['raw_scores'][str(k)]['round6'])
        cap = float(original[rare].max())
        for feature in ['qi7', 'attrs10']:
            raw_values = np.array(result['raw_scores'][str(k)][feature+'_gen'])
            # すべての非希少候補を希少スコアの上に置く検査用の構成。
            current = original.copy()
            if k not in (1, 2):
                ranks = pd.Series(raw_values[unknown_nonrare]).rank(method='min').to_numpy()
                current[unknown_nonrare] = cap+(1-cap)*(.1+.8*ranks/len(ranks))
            assert (current[private] == 0).all()
            assert np.isfinite(current).all() and ((current >= 0) & (current <= 1)).all()
            for label, plan in plans.items():
                combined = current.copy()
                if label != 'round6':
                    pair = plan[k]
                    rest = sorted([i for i in allowed if i not in pair], key=lambda i: (-raw[k]['qi7'][i], eligible[i]))
                    combined[rare] = 0.
                    for j, i in enumerate([*pair, *rest]):
                        combined[pos[eligible[i]]] = cap*(.9-.8*j/len(allowed))
                    assert all(combined[pos[eligible[i]]] == 0 for kk, i in known if kk == k)
                    for n in (1, 2):
                        for ms in combinations(allowed, n):
                            truth_r = np.array([int(eligible[i] in {eligible[x] for x in ms}) if i >= 0 else 0
                                               for i in [eligible.index(rid) if rid in eligible else -1 for rid in data['rare_ids']]])
                            official_tpr = a.h._tpr_at_fpr(combined[rare_positions], truth_r, .01)[0]
                            expected = a.utility_units(pair, ms)/2
                            assert abs(official_tpr-expected) < 1e-14
                            assert abs(official_tpr-tpr_reference(combined[rare_positions], truth_r)) < 1e-14
                            rare_checks += 1
                assert (combined[private] == 0).all()
                assert int((combined[~rare] > combined[rare].max()).sum()) >= 38
                for seed in range(12):
                    rng = np.random.default_rng(7000*k+seed)
                    y = np.zeros(len(ids), int)
                    r = seed % 3
                    positives = list(rng.choice([pos[eligible[i]] for i in allowed], r, replace=False))
                    positives += list(rng.choice(np.flatnonzero(unknown_nonrare), 25+seed % 3-r, replace=False))
                    y[positives] = 1
                    official_new = a.score_mia(combined, y, {'scoring': {'low_fpr': .01}}, rare_mask=rare).breakdown
                    official_control = a.score_mia(current, y, {'scoring': {'low_fpr': .01}}, rare_mask=rare).breakdown
                    assert official_new['tpr_at_low_fpr'] == official_control['tpr_at_low_fpr']
                    assert abs(official_new['tpr_at_low_fpr']-tpr_reference(combined, y)) < 1e-14
                    if r:
                        assert abs(official_new['tpr_rare']-tpr_reference(combined[rare], y[rare])) < 1e-14
                    if k in (1, 2):
                        assert official_new['tpr_at_low_fpr'] == a.h._tpr_at_fpr(original, y, .01)[0]
                    full_checks += 1
                    reference_checks += 1
            for metric in ['gen', 'distance', 'point']:
                values = np.array(result['raw_scores'][str(k)][feature+'_'+metric])
                order = a.order(values, ids, ~rare)
                diagnostics.append(dict(team=k, feature=feature, metric=metric,
                    private_in_raw_top20=int(private[order[:20]].sum()),
                    private_in_raw_top40=int(private[order[:40]].sum())))
    # 同点補間を個別に点検。n人同点に会員r人全員が入る場合の理論式。
    tie_checks = 0
    for n in range(2, 12):
        for r in (1, 2):
            if n <= r:
                continue
            scores = np.r_[np.ones(n), np.zeros(24-n)]
            truth = np.r_[np.ones(r, int), np.zeros(24-r, int)]
            expected = .01*(24-r)/(n-r)
            assert abs(a.h._tpr_at_fpr(scores, truth, .01)[0]-expected) < 1e-14
            assert abs(tpr_reference(scores, truth)-expected) < 1e-14
            tie_checks += 1
    assert all(a.h.sha256(p) == sha for p, sha in protected.items())
    assert {p for p in a.RECORDS.rglob('*') if p.is_file()} == set(protected)
    return dict(seconds=perf_counter()-start, full_and_rare_joint_checks=full_checks,
        rare_label_configurations=rare_checks, independent_roc_checks=reference_checks,
        tie_formula_checks=tie_checks, protected_files_unchanged=True,
        rare_models=post, negative_diagnostics=diagnostics,
        full_metrics_are_unknown=True, auxiliary_does_not_estimate_other_teams=True,
        previous_keep12_plan={str(k): [eligible[i] for i in pair] for k, pair in plans['gen7_keep12_refresh_second'].items()},
        inputs_sha256={'分析.py': a.h.sha256(HERE/'分析.py'), '比較結果.json': a.h.sha256(HERE/'比較結果.json'),
                       '検証.py': a.h.sha256(Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    opt = parser.parse_args()
    output = HERE/'追加検証結果.json'
    if opt.save and output.exists():
        raise FileExistsError('既存結果は上書きしません。--saveなしで再検証できます。')
    result = run()
    if opt.save:
        with output.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['rare_models', 'negative_diagnostics', 'inputs_sha256']}, ensure_ascii=False))
