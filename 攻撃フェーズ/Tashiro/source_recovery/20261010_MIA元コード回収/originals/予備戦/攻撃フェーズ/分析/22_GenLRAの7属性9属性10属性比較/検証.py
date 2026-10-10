"""9属性の独立再計算、9対10の同一割当での比較、全体・希少の組合せ検査。

python3 -B 検証.py --save  # 検証結果.jsonを新規保存。既存ファイルは上書きしない。
python3 -B 検証.py         # 読み取り専用の再照合
予想時間5〜40秒。NumPy・pandasのみ。仮ラベルは実装検査で、性能評価ではない。
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('analysis22', HERE/'分析.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
a, h = m.a, m.h


def independent9(bg, c, q):
    """Gen-LRA関数を呼ばず、RとR+xのカーネル密度を直接比較する。"""
    cols = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']
    cats = ['sex', 'prefecture']
    def num(f):
        x = f[cols].to_numpy(float).copy()
        for name in ['TG', 'ALT']:
            x[:, cols.index(name)] = np.log1p(x[:, cols.index(name)])
        return x
    gx = num(bg)
    scale = (np.percentile(gx, 75, axis=0)-np.percentile(gx, 25, axis=0))/1.349
    gx, cx, qx = gx/scale, num(c)/scale, num(q)/scale
    bw = (len(bg)*9/4)**(-1/11)
    freq = {cat: {key: (n+1)/(len(bg)+bg[cat].nunique()) for key, n in bg[cat].value_counts().items()} for cat in cats}
    cb_log = np.zeros((len(c), len(bg)))
    for j in range(7):
        cb_log -= .5*((cx[:, j, None]-gx[None, :, j])/bw)**2
    cb_log -= 7*math.log(bw)+3.5*math.log(2*math.pi)
    for cat in cats:
        f = np.array([freq[cat][v] for v in c[cat]])
        cb_log += np.log(.98*(c[cat].to_numpy()[:, None] == bg[cat].to_numpy()[None, :])+.02*f[:, None])
    def logmean(logs):
        shift = float(np.max(logs))
        return shift+math.log(float(np.exp(logs-shift).mean()))
    outputs = []
    for j in range(len(q)):
        delta = cx-qx[j]
        d2 = (delta*delta).sum(axis=1)
        logk = -.5*d2/(bw*bw)-7*math.log(bw)-3.5*math.log(2*math.pi)
        for cat in cats:
            same = c[cat].to_numpy() == q.iloc[j][cat]
            d2 += ~same
            f = np.array([freq[cat][v] for v in c[cat]])
            logk += np.log(.98*same+.02*f)
        near = np.argsort(d2, kind='stable')[:10]
        score = sum(logmean(np.r_[cb_log[s], logk[s]])-logmean(cb_log[s]) for s in near)
        outputs.append(score)
    return np.array(outputs)


def paired_comparison(data, groups, known, raw, plans):
    """同じ正解割当で9属性と10属性の得点差を計算。周辺分布の独立仮定は置かない。"""
    specs = [('constraints_balanced', None, 0, 'balanced'), ('constraints_uniform', None, 0, 'uniform'),
             ('gen7_rank_weak', 'qi7', 1, 'balanced'), ('gen7_rank_medium', 'qi7', 2, 'balanced'),
             ('gen9_rank_weak', 'attrs9', 1, 'balanced'), ('gen10_rank_weak', 'attrs10', 1, 'balanced')]
    ranks = {(k, name): a.rank(v) for k, values in raw.items() for name, v in values.items()}
    out = []
    for name, metric, beta, prior in specs:
        whole = {(0, 0, 0): 1.}
        for group in h.make_groups(data, groups, prior):
            poly = defaultdict(float)
            for mask, edges, weight in group:
                if any(e in known for e in edges):
                    continue
                by_team = defaultdict(list)
                for k, i in edges:
                    by_team[k].append(i)
                old = sum(a.utility_units(plans['round6'][k], ms) for k, ms in by_team.items())
                if old > 1 or any(a.utility_units(plans['round6'][k], ms) for k, ms in by_team.items() if k in (1, 2)):
                    continue
                if metric:
                    weight *= math.exp(beta*sum((11-ranks[k, metric][i])/10 for k, i in edges))
                difference = sum(a.utility_units(plans['gen9_after48'][k], ms)-a.utility_units(plans['gen10_after48'][k], ms)
                                 for k, ms in by_team.items())
                poly[mask, old, difference] += weight
            whole = a.convolve(whole, poly)
        norm = sum(w for (mask, old, diff), w in whole.items() if old == 1)
        dist = defaultdict(float)
        for (mask, old, diff), w in whole.items():
            if old == 1:
                dist[diff] += w/norm
        assert abs(sum(dist.values())-1) < 1e-9
        out.append(dict(model=name, mean_score_difference=sum(d*p/12 for d, p in dist.items()),
                        nine_better=sum(p for d, p in dist.items() if d > 0),
                        ten_better=sum(p for d, p in dist.items() if d < 0), tie=dist[0]))
    return out


def run():
    start = perf_counter()
    result = h.read_json(HERE/'比較結果.json')
    old = h.read_json(m.PREVIOUS/'比較結果.json')
    assert all(h.sha256(m.ROOT/p) == digest for p, digest in result['input_hashes'].items())
    protected = {p: h.sha256(p) for p in a.RECORDS.rglob('*') if p.is_file()}
    data = h.load_inputs()
    groups, _ = h.audit_history(data)
    ids, eligible = result['record_ids'], data['ids']
    pos = {rid: i for i, rid in enumerate(ids)}
    plans = {label: {int(k): tuple(eligible.index(rid) for rid in pair) for k, pair in plan.items()}
             for label, plan in result['plans'].items()}
    known = data['old'] | data['new'] | {(r['team'], eligible.index(r['record_id'])) for r in old['new_negative_pairs']}
    raw = {k: {name: np.array([(result['raw_new_scores'][str(k)]['attrs9'] if name == 'attrs9' else old['raw_scores'][str(k)][name+'_gen'])[pos[rid]]
                              for rid in eligible]) for name in m.PRIMARY} for k in data['teams']}
    paired = paired_comparison(data, groups, known, raw, plans)
    model_map = {r['model']: r for r in result['rare_models']}
    for r in paired:
        p = model_map[r['model']]['plans']
        assert abs(r['mean_score_difference']-(p['gen9_after48']['expected']-p['gen10_after48']['expected'])) < 1e-12
    b = pd.read_csv(m.DATA/'PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/B_self.csv', float_precision='round_trip')
    bg = pd.read_csv(m.DATA/'PWSCup2026_prelim_common/A_bg.csv', float_precision='round_trip')
    ch = pd.read_csv(m.DIST/'aia_challenge_5.csv', float_precision='round_trip')
    qi = a.gen.FEATURES[0][1]+a.gen.FEATURES[0][2]
    private_ids = set(b.merge(ch[qi], on=qi, validate='one_to_one').record_id.astype(int))
    private, rare = np.isin(ids, list(private_ids)), b.is_rare.to_numpy(bool)
    use = ~private & ~rare
    error = 0.
    count = joint_count = 0
    for k in data['teams']:
        raw9 = np.array(result['raw_new_scores'][str(k)]['attrs9'])
        candidate = a.order(raw9, ids, use)
        select = sorted(set(np.flatnonzero(rare)) | set(candidate[:10]) | set(np.flatnonzero(private & ~rare)[:5]))
        q = b.iloc[select].reset_index(drop=True)
        c = pd.read_csv(m.DIST/f'C_{k}.csv', float_precision='round_trip')
        independent = independent9(bg, c, q)
        error = max(error, float(np.max(np.abs(independent-raw9[select]))))
        count += len(select)
        assert error < 1e-10
        original = np.array(old['raw_scores'][str(k)]['round6'])
        cap = original[rare].max()
        control = original.copy()
        if k not in (1, 2):
            ranks = pd.Series(raw9[use]).rank(method='min').to_numpy()
            control[use] = cap+(1-cap)*(.1+.8*ranks/len(ranks))
        combined = control.copy()
        combined[rare] = 0.
        allowed = [i for i in range(11) if (k, i) not in known]
        order9 = sorted(allowed, key=lambda i: (-raw[k]['attrs9'][i], eligible[i]))
        for j, i in enumerate(order9):
            combined[pos[eligible[i]]] = cap*(.9-.8*j/len(order9))
        assert (combined[private] == 0).all()
        assert all(combined[pos[eligible[i]]] == 0 for kk, i in known if kk == k)
        assert int(sum(combined[~rare] > combined[rare].max())) >= 38
        assert np.isfinite(combined).all() and ((combined >= 0) & (combined <= 1)).all()
        for seed in range(12):
            rng = np.random.default_rng(k*1000+seed)
            truth = np.zeros(len(ids), int)
            r = seed % 3
            selected = list(rng.choice([pos[eligible[i]] for i in allowed], r, replace=False))
            selected += list(rng.choice(np.flatnonzero(use), 25+seed % 3-r, replace=False))
            truth[selected] = 1
            before = a.score_mia(control, truth, {'scoring': {'low_fpr': .01}}, rare_mask=rare).breakdown
            after = a.score_mia(combined, truth, {'scoring': {'low_fpr': .01}}, rare_mask=rare).breakdown
            assert before['tpr_at_low_fpr'] == after['tpr_at_low_fpr']
            if r:
                assert after['tpr_rare'] == h._tpr_at_fpr(combined[rare], truth[rare], .01)[0]
            if k in (1, 2):
                assert after['tpr_at_low_fpr'] == h._tpr_at_fpr(original, truth, .01)[0]
            joint_count += 1
    assert all(h.sha256(p) == digest for p, digest in protected.items())
    assert {p for p in a.RECORDS.rglob('*') if p.is_file()} == set(protected)
    return dict(seconds=perf_counter()-start, independent9_values=count, max_error=error,
                joint_all_rare_cases=joint_count, paired_nine_vs_ten=paired,
                protected_files_unchanged=True, protected_count=len(protected),
                input_hashes={str(p.relative_to(m.ROOT)): h.sha256(p) for p in [HERE/'分析.py', HERE/'比較結果.json', Path(__file__)]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    opt = parser.parse_args()
    output = HERE/'検証結果.json'
    if opt.save and output.exists():
        raise FileExistsError('既存結果は上書きしません。--saveなしで再確認してください。')
    result = run()
    if opt.save:
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'input_hashes'}, ensure_ascii=False))
