"""360件と派生集計を読み取り専用で照合する。Python標準ライブラリのみ。"""
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
import csv
import math

HERE = Path(__file__).resolve().parent
NAMES = {'Zhang11': 'zhang_11_normal_wald', 'Zhang04': 'zhang_04_copula_bgtime', 'Seo03': 'seo-03-順位混合Wald調整'}


def read(name):
    with (HERE / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def same(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-12), (a, b)


def score(hm, hc, nm, nc):
    return max(0, (hm / nm - hc / nc) / (1 - hc / nc)) if hc < nc else 0


def stats(values):
    return dict(zip(['最小', '中央値', '平均', '最大'], [min(values), median(values), mean(values), max(values)]))


def main():
    rows = read('360件の詳細.csv')
    source = {(r['defense'], r['attack'], int(r['panel'])): r for r in read('../summary/AIA全2085比較.csv') if r['suite'] == 'main10'}
    constants = {int(r['パネル']): r for r in read('定数10年_パネル別.csv')}
    old_constant = {int(r['panel']): r for r in read('../summary/C非依存_全員10年.csv') if r['suite'] == 'main10'}
    assert len(rows) == 360 and set(constants) == set(range(1, 11))
    assert len({(r['加工案'], r['攻撃手法'], r['パネル']) for r in rows}) == 360
    groups, panels = defaultdict(list), defaultdict(list)
    for p, r in constants.items():
        assert int(r['予測年数']) == 10
        same(r['AIA得点'], score(*[int(r[k]) for k in ['標的的中人数', '対照的中人数', '標的人数', '対照人数']]))
        same(r['AIA得点'], old_constant[p]['R_all'])
        same(r['標的的中人数'], old_constant[p]['member_hits']); same(r['対照的中人数'], old_constant[p]['control_hits'])
    for r in rows:
        p = int(r['パネル']); key = (r['加工案'], r['攻撃手法'])
        old = source[NAMES[r['加工案']], r['攻撃手法'], p]
        for new, previous in [('標的的中人数', 'member_hits'), ('標的人数', 'n_member'), ('対照的中人数', 'control_hits'), ('対照人数', 'n_control'), ('AIA得点', 'R_all')]:
            same(r[new], old[previous])
        same(r['AIA得点'], score(*[int(r[k]) for k in ['標的的中人数', '対照的中人数', '標的人数', '対照人数']]))
        same(r['定数10年の得点'], constants[p]['AIA得点'])
        same(r['定数との差'], float(r['AIA得点']) - float(constants[p]['AIA得点']))
        groups[key].append(r); panels[r['加工案'], p].append(r)
    for r in read('手法別統計と定数比較.csv'):
        chosen = groups[r['加工案'], r['攻撃手法']]; assert len(chosen) == 10
        for field, prefix in [('AIA得点', ''), ('定数との差', '定数との差_')]:
            for k, v in stats([float(x[field]) for x in chosen]).items(): same(v, r[prefix + k])
        ds = [float(x['定数との差']) for x in chosen]
        for k, v in [('定数超え', sum(d > 1e-12 for d in ds)), ('定数と同じ', sum(abs(d) <= 1e-12 for d in ds)), ('定数未満', sum(d < -1e-12 for d in ds))]: same(v, r[k])
    for n in NAMES:
        matrix = read(n + '_手法別パネル表.csv'); assert len(matrix) == 13
        for r in matrix:
            vs = [float(r['P' + str(i)]) for i in range(1, 11)]
            expected = [float(constants[i]['AIA得点']) for i in range(1, 11)] if r['区分'] == '定数' else [float(x['AIA得点']) for x in sorted(groups[n, r['攻撃手法']], key=lambda x: int(x['パネル']))]
            for a, b in zip(vs, expected): same(a, b)
            for k, v in stats(vs).items(): same(v, r[k])
    for r in read('パネル別比較.csv'):
        chosen = panels[r['加工案'], int(r['パネル'])]; assert len(chosen) == 12
        for k, v in stats([float(x['AIA得点']) for x in chosen]).items(): same(v, r['12攻撃の' + k])
        mx = float(r['12攻撃の最大']); b = float(constants[int(r['パネル'])]['AIA得点'])
        same(r['定数10年'], b); same(r['最大と定数の差'], mx - b); same(r['定数込み13手法の最大'], max(mx, b))
    for r in read('3案の要約.csv'):
        n = r['加工案']; chosen = [x for x in rows if x['加工案'] == n]
        maxima = [max(float(x['AIA得点']) for x in panels[n, p]) for p in range(1, 11)]
        included = [max(v, float(constants[p]['AIA得点'])) for p, v in enumerate(maxima, 1)]
        for prefix, vs in [('全120件_', [float(x['AIA得点']) for x in chosen]), ('パネル最大10値_', maxima), ('定数込みパネル最大10値_', included)]:
            for k, v in stats(vs).items(): same(v, r[prefix + k])
    print('360件・定数10パネル・3案の全手法表・手法別統計・パネル別統計・要約の一致を確認しました。')


if __name__ == '__main__':
    main()
