"""共有フォルダのCSVを再集計・照合する。加工・攻撃の実行やファイル更新はしない。"""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def same(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-12, rel_tol=1e-12), (a, b)


def main():
    manifest = json.loads((ROOT/'ファイル一覧とハッシュ.json').read_text(encoding='utf-8'))
    for f in manifest['files']:
        assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest() == f['sha256'], f['path']
    current = ROOT/'01_本戦Team28_AIA980対戦/詳細CSV'
    data, utility = rows(current/'全980対戦.csv'), rows(current/'有用性140件.csv')
    pairs, summary = rows(current/'加工攻撃別集約.csv'), rows(current/'加工案別集約.csv')
    assert (len(data), len(utility), len(pairs), len(summary)) == (980, 140, 98, 14)
    names = [r['加工案'] for r in summary]
    attacks = list(dict.fromkeys(r['attack'] for r in data))
    assert len(set(names)) == 14 and len(attacks) == 7
    scores = {(int(r['panel']), r['defense'], r['attack']): float(r['R_all']) for r in data}
    assert len(scores) == 980
    assert set(scores) == {(p, d, a) for p in range(1, 11) for d in names for a in attacks}
    for r in data:
        cm, cc = int(r['member_hits']), int(r['control_hits'])
        value = max(0., (cm-cc)/(163-cc)) if cc < 163 else 0.
        same(value, r['R_all'])
    for r in pairs:
        values = [scores[(p, r['加工案'], r['攻撃案'])] for p in range(1, 11)]
        for key, value in [('中央値', median(values)), ('最小', min(values)), ('最大', max(values))]:
            same(value, r[key])
        for p, value in enumerate(values, 1):
            same(value, r[f'panel_{p:02d}'])
    print('本戦Team28: 980対戦・有用性140件・98組の集約を確認')
    print('加工案\tU中央値\tAIA最大の中央値\tAIA最大の範囲')
    for r in summary:
        d = r['加工案']
        values = [max(scores[(p, d, a)] for a in attacks) for p in range(1, 11)]
        u = [float(v['U']) for v in utility if v['defense'] == d]
        assert len(u) == 10
        for field, value in [('U中央値', median(u)), ('U最小', min(u)), ('U最大', max(u)),
                             ('AIA観測最大の中央値', median(values)), ('AIA観測最大の最小', min(values)),
                             ('AIA観測最大の最大', max(values))]:
            same(value, r[field])
        print(f'{d}\t{median(u):.4f}\t{median(values):.4f}\t{min(values):.4f}–{max(values):.4f}')
    previous = ROOT/'02_参考_予備戦Team05_112対戦/詳細CSV'
    summary = rows(previous/'加工案別集約.csv')
    assert len(summary) == 14
    for kind, expected in [('MIA', 56), ('AIA', 98)]:
        data = rows(previous/f'バトル結果_{kind}.csv')
        assert len(data) == 112
        scored = [r for r in data if r['スコア'] != '']
        assert len(scored) == expected
        for r in summary:
            maximum = max(float(x['スコア']) for x in scored if x['加工案名'] == r['加工案名'])
            same(maximum, r[f'{kind}_観測最大_補正前'])
    print('予備戦Team05: 112対戦、MIA56件・AIA98件の集約を確認')
    print(f"同梱ファイル{len(manifest['files'])}点のハッシュ一致。すべての照合が通りました。")


if __name__ == '__main__':
    main()
