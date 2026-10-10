"""分析.pyを使わず、元CSV・明示的なR+x密度・公式採点で独立照合。

python3 -B 検証.py  # 予想5〜60秒。読み取り専用。検証結果は画面にだけ出す。
"""
import hashlib
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
sys.path.insert(0, str(ROOT/'スターターキット/改訂版_20260824/starter/src'))
from pwscup2026.scoring.attack_score import _tpr_at_fpr


def read(name):
    return json.loads((HERE/name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(frame, cols):
    values = []
    for col in cols:
        a = frame[col].to_numpy(float)
        values.append(np.log1p(a) if col in ('TG', 'ALT') else a)
    return np.column_stack(values)


def logk(query, refs, num, cat, scales, h, frequencies):
    q, r = transform(query, num), transform(refs, num)
    result = np.zeros((len(query), len(refs)))
    for j in range(len(num)):
        z = (q[:, j, None]-r[None, :, j])/(scales[j]*h)
        result += -.5*z*z-math.log(h*math.sqrt(2*math.pi))
    for col in cat:
        eq = query[col].to_numpy()[:, None] == refs[col].to_numpy()[None, :]
        f = np.array([frequencies[col][str(v)] for v in query[col]])
        result += np.log(np.where(eq, .98, 0.)+.02*f[:, None])
    return result


def logmean(matrix):
    top = matrix.max(axis=1)
    return top+np.log(np.exp(matrix-top[:, None]).mean(axis=1))


def matching_exists(active, candidates):
    # Hall式は使わない。r個のteamスロットからIDへの増加路を実際に探す。
    slots = [team for team, r in active for _ in range(r)]
    owner = {}
    def assign(slot, seen):
        for rid in candidates[slots[slot]]:
            if rid in seen:
                continue
            seen.add(rid)
            if rid not in owner or assign(owner[rid], seen):
                owner[rid] = slot
                return True
        return False
    return all(assign(i, set()) for i in range(len(slots)))


def main():
    start = perf_counter()
    original = {p.name: sha(p) for p in HERE.iterdir() if p.is_file()}
    hashes = read('入力ファイルSHA256.json')
    assert all(not p.startswith('アーカイブ/') and sha(ROOT/p) == v for p, v in hashes.items())
    b = pd.read_csv(DATA/'PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/B_self.csv')
    bg = pd.read_csv(DATA/'PWSCup2026_prelim_common/A_bg.csv')
    c = pd.read_csv(DATA/'PWSCup2026_prelim_attack/C_20.csv')
    rare = b.loc[b.is_rare.eq(1)].sort_values('record_id').reset_index(drop=True)
    ids = rare.record_id.astype(int).tolist()
    facts, plans = read('検証結果.json'), read('固定条件と前処理.json')
    records, settings = read('希少24人の全設定順位.json'), read('設定比較.json')
    rows = {(r['setting'], r['record_id']): r for r in records}
    bases = {(r['feature'], r['bandwidth_factor'], r['record_id']): r for r in read('同条件の距離と点密度比.json')}
    contrib = {(r['record_id'], r['neighbor_rank']): r for r in read('主設定の近傍別寄与.json')}
    assert len(records) == len(rows) == 432 and len(settings) == 18
    assert plans['factors'] == [.5, 1., 2.] and plans['ks'] == [1, 10, 20] and plans['primary'] == 'qi7_h1_k10'
    assert len(ids) == 24 and len(bg) == 5143 and len(c) == 1049
    ch5 = pd.read_csv(DATA/'PWSCup2026_prelim_attack/aia_challenge_5.csv')
    qi = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture']
    private = b.merge(ch5[qi], on=qi, validate='one_to_one')
    assert len(private) == 163
    assert sorted(private.loc[private.is_rare.eq(1), 'record_id']) == facts['private_rare_ids']
    eligible, remaining = facts['eligible_ids'], facts['remaining_ids']
    assert sorted(set(ids)-set(facts['private_rare_ids'])) == eligible
    assert sorted(set(eligible)-set(remaining)) == [201, 238]
    maximum_error, values_checked = 0., 0
    for config in plans['features']:
        num, cat, family = config['numeric'], config['categorical'], config['feature']
        gx = transform(bg, num)
        scales = np.subtract(*np.percentile(gx, [75, 25], axis=0))/1.349
        np.testing.assert_allclose(scales, config['scale'], rtol=1e-14)
        base_h = (4/(len(bg)*(len(num)+2)))**(1/(len(num)+4))
        assert abs(base_h-config['silverman_factor']) < 1e-15
        frequencies = config['category_frequencies']
        for col in cat:
            assert abs(sum(frequencies[col].values())-1) < 1e-14
            for key, value in frequencies[col].items():
                count = int(bg[col].astype(str).eq(key).sum())
                assert abs(value-(count+1)/(len(bg)+bg[col].nunique())) < 1e-15
                assert abs(sum(.98*int(k == key)+.02*v for k, v in frequencies[col].items())-1) < 1e-14
        for rid in ids:
            q = rare.loc[rare.record_id.eq(rid)]
            z = (transform(c, num)-transform(q, num))/scales
            d2 = (z*z).sum(axis=1)
            for col in cat:
                d2 += c[col].to_numpy() != q[col].iloc[0]
            nearest = np.argsort(d2, kind='stable')[:20]
            selected = c.iloc[nearest]
            for factor in [.5, 1., 2.]:
                h = base_h*factor
                reference_logkernels = logk(selected, bg, num, cat, scales, h, frequencies)
                added_logkernel = logk(selected, q, num, cat, scales, h, frequencies)
                # R+xを明示的に連結し、各々の平均密度を再計算する。
                log_before = logmean(reference_logkernels)
                log_after = logmean(np.concatenate([reference_logkernels, added_logkernel], axis=1))
                differences = log_after-log_before
                for k in [1, 10, 20]:
                    setting = f'{family}_h{factor:g}_k{k}'
                    expected = rows[setting, rid]['influence']
                    actual = float(differences[:k].sum())
                    err = abs(actual-expected)
                    maximum_error = max(maximum_error, err)
                    assert err < 2e-10, (setting, rid, actual, expected)
                    values_checked += 1
                baseline = bases[family, factor, rid]
                assert abs(-math.sqrt(d2.min()/(len(num)+len(cat)))-baseline['dcr_score']) < 1e-13
                ratio = logmean(logk(q, c, num, cat, scales, h, frequencies))[0]-logmean(logk(q, bg, num, cat, scales, h, frequencies))[0]
                assert abs(ratio-baseline['pointwise_log_density_ratio']) < 2e-10
                if family == 'qi7' and factor == 1.:
                    for j, crow in enumerate(nearest[:10], 1):
                        saved = contrib[rid, j]
                        assert saved['c_row_index'] == int(crow) and saved['c_csv_line'] == int(crow)+2
                        assert abs(saved['log_density_gain']-differences[j-1]) < 2e-11
                        assert abs(saved['log_reference_density']-log_before[j-1]) < 2e-11
    for setting in settings:
        key = setting['setting']
        values = {rid: rows[key, rid]['influence'] for rid in ids}
        assert sorted(eligible, key=lambda rid: (-values[rid], rid)) == setting['order11']
        assert sorted(remaining, key=lambda rid: (-values[rid], rid)) == setting['order9']
        for rid in ids:
            assert rows[key, rid]['rank24'] == 1+sum(v > values[rid] for v in values.values())
            if rid in eligible:
                assert rows[key, rid]['rank11'] == 1+sum(values[r] > values[rid] for r in eligible)
            if rid in remaining:
                assert rows[key, rid]['rank9'] == 1+sum(values[r] > values[rid] for r in remaining)
    # 他19チーム同点の範囲を、公式TPRと実際の二部マッチングで再計算。
    upstream = json.loads((HERE.parent/'14_MIA希少の距離と対応固有性/距離と対応の診断.json').read_text())
    negatives = {(x['target_k'], x['record_id']) for x in upstream if x['metric'] == 'l1_cat1'
                 and x['record_id'] in eligible and x['status'] in ('第1〜3回からの既知非会員', '第5回からの既知非会員')}
    assert len(negatives) == 40
    bounds = read('希少スコアの範囲.json')
    allowed = {int(t): [rid for rid in eligible if (int(t), rid) not in negatives] for t in bounds['remaining_counts']}
    tpr_cache = {}
    for n, r in product([7, 8, 9, 10], [1, 2]):
        score, truth = np.zeros(24), np.zeros(24, int)
        score[:n], truth[:r] = 1, 1
        tpr_cache[n, r] = _tpr_at_fpr(score, truth, .01)[0]
    groups = facts['history_audit']['groups']
    totals, others, halfs = [], [], []
    for a, (b1, b2), (c1, c2), r20 in product(groups['10'], permutations(groups['9'], 2), permutations(groups['chosen'], 2), [1, 2]):
        active = [(a, 1), (b1, 1), (b2, 2), (c1, 1), (c2, 2), (20, r20)]
        assert matching_exists(active, allowed)
        other = sum(tpr_cache[len(allowed[t]), r] for t, r in active if t != 20)/6
        others.append(other)
        totals.append(other+tpr_cache[9, r20]/6)
        if r20 == 2:
            halfs.append(other+1/12)
    assert len(totals) == bounds['feasible_count_arrangements'] == 13440
    for name, vals in [('all20_tied', totals), ('other19_tied', others), ('team20_unique_first_false', others),
                       ('team20_unique_one_of_two', halfs), ('team20_unique_all_true', [v+1/6 for v in others])]:
        assert abs(min(vals)-bounds[name]['min']) < 1e-14 and abs(max(vals)-bounds[name]['max']) < 1e-14
    utility = read('Team20の45状態と採点.json')
    order = utility['unique_order']
    assert order == facts['primary']['order9']
    for state in utility['states']:
        truth = np.array([int(rid in state['members']) for rid in ids])
        score = np.array([(len(order)-order.index(rid))/len(order) if rid in order else 0 for rid in ids])
        assert _tpr_at_fpr(score, truth, .01)[0] == state['unique_order_tpr']
        assert abs(_tpr_at_fpr(np.array([int(rid in order) for rid in ids]), truth, .01)[0]-state['all9_tied_tpr']) < 1e-15
    assert len(utility['states']) == 45
    # 説明中の主要な3表も、保存済み数値・設定に照合する。
    report_rows = [[cell.strip() for cell in line.strip().split('|')[1:-1]]
                   for line in (HERE/'説明.md').read_text().splitlines() if line.startswith('|')]
    report_checks = 0
    for cells in report_rows:
        if len(cells) == 7 and cells[0].isdigit():
            rid = int(cells[0])
            saved = rows['qi7_h1_k10', rid]
            assert abs(float(cells[2].replace('−', '-'))-saved['influence']) <= .000000501
            assert int(cells[3]) == saved['rank11']
            assert cells[4] == ('除外' if saved['rank9'] is None else str(saved['rank9']))
            assert int(cells[5]) == bases['qi7', 1., rid]['dcr_rank11']
            assert int(cells[6]) == bases['qi7', 1., rid]['pointwise_rank11']
            report_checks += 1
        elif len(cells) == 3 and cells[0].isdigit():
            rid = int(cells[0])
            saved = rows['qi7_h1_k10', rid]
            assert abs(float(cells[1].replace('−', '-'))-saved['influence']) <= .000000501
            assert rid in facts['private_rare_ids'] and int(cells[2]) == saved['rank24']
            report_checks += 1
        elif len(cells) == 8 and cells[0] in ('7', '7（主設定）', '10'):
            family = 'qi7' if cells[0].startswith('7') else 'attrs10'
            key = f'{family}_h{float(cells[1]):g}_k{cells[2]}'
            saved = next(s for s in settings if s['setting'] == key)
            assert int(cells[3]) == saved['order11'][0]
            assert cells[4] == '・'.join(map(str, saved['order9'][:2]))
            assert int(cells[5]) == saved['negative201_rank']
            assert int(cells[6]) == saved['negative238_rank']
            assert int(cells[7]) == len(saved['private_above_best_remaining'])
            report_checks += 1
    assert report_checks == 42
    assert private.loc[private.is_rare.eq(1), 'prefecture'].unique().tolist() == ['東京都']
    assert {p.name: sha(p) for p in HERE.iterdir() if p.is_file()} == original
    assert all(sha(ROOT/p) == v for p, v in hashes.items())
    print(json.dumps(dict(independent_influence_values=values_checked, max_absolute_error=maximum_error,
                          explicit_augmented_kde_fits=144, independent_baseline_values=288,
                          primary_neighbor_contributions=240, matched_count_arrangements=len(totals),
                          official_team20_states=45, report_table_rows=report_checks,
                          inputs_and_outputs_unchanged=True,
                          elapsed_seconds=perf_counter()-start), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
