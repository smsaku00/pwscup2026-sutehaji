"""掲載済みCSVを再集計する。標準ライブラリのみ。ファイルの書き込みはしない。"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent


def rows(name):
    with (HERE / '集計' / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), (a, b)


def percentile(values, q):
    v = sorted(values)
    pos = (len(v) - 1) * q
    i = int(pos)
    return v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * (pos - i)


def summarize(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values), p05=percentile(values, .05),
                p95=percentile(values, .95), positive_n=sum(v > 0 for v in values))


def main():
    plan = json.loads((HERE / '計画.json').read_text())
    phi = plan['phi']
    case_rows = rows('12攻撃最大値_全補完ケース.csv')
    attack_rows = rows('全手法_全補完ケース.csv')
    assert len(case_rows) == 800 and len(attack_rows) == 11200
    attacks = {}
    for row in attack_rows:
        key = row['setting'], row['iteration'], row['candidate']
        attacks.setdefault(key, {})[row['attack']] = row
        close(row['raw_AIA'], max(float(row['R_all']), float(row['R_rare'])))
        close(row['phi_adjusted_AIA'], max(0, float(row['raw_AIA']) - phi))
    groups = {}
    for row in case_rows:
        key = row['setting'], row['iteration'], row['candidate']
        group = attacks[key]
        assert set(group) == {f'A{i:02d}' for i in range(1, 13)} | {'B01', 'B02'}
        maximum = max(float(group[f'A{i:02d}']['raw_AIA']) for i in range(1, 13))
        close(row['raw_max12'], maximum)
        close(row['phi_adjusted_max12'], max(0, maximum - phi))
        close(row['AIA_only_protection'], 1 - float(row['phi_adjusted_max12']))
        for attack, label in [('B01', 'constant'), ('B02', 'background')]:
            close(row[label + '_raw'], group[attack]['raw_AIA'])
            close(row[label + '_phi'], group[attack]['phi_adjusted_AIA'])
        groups.setdefault((row['setting'], row['candidate']), []).append(row)
    for row in rows('候補別集計.csv'):
        group = groups[row['setting'], row['candidate']]
        assert len(group) == 100 and len({x['iteration'] for x in group}) == 100
        calculated = summarize([float(x[row['metric']]) for x in group])
        for k, v in calculated.items():
            close(row[k], v)
    for row in rows('手法別集計.csv'):
        values = [float(x[row['metric']]) for x in attack_rows
                  if x['setting'] == row['setting'] and x['candidate'] == row['candidate']
                  and x['attack'] == row['attack']]
        assert len(values) == 100
        for k, v in summarize(values).items():
            close(row[k], v)
    paired = []
    for row in rows('交換前後の対応比較.csv'):
        a = {x['iteration']: float(x['phi_adjusted_max12']) for x in groups[row['setting'], 'Zhang11']}
        b = {x['iteration']: float(x['phi_adjusted_max12']) for x in groups[row['setting'], 'Zhang11-HCv2-1']}
        difference = [b[i] - a[i] for i in a]
        for k, v in [('better', sum(x < -1e-12 for x in difference)),
                     ('tie', sum(abs(x) <= 1e-12 for x in difference)),
                     ('worse', sum(x > 1e-12 for x in difference)),
                     ('difference_after_minus_before', statistics.mean(difference))]:
            close(row[k], v)
        paired.append(row)
    verification = json.loads((HERE / '検証結果.json').read_text())
    digest = hashlib.sha256((HERE / '実験コード/当時の分析.py').read_bytes()).hexdigest()
    assert digest == verification['analysis_sha256'] == plan['analysis_sha256']
    validation = rows('予備戦による補完検証.csv')
    assert len(validation) == 21
    assert sum(x['long_covered'] == 'True' for x in validation) == 16
    assert sum(x['rare_long_covered'] == 'True' for x in validation) == 18
    main_summary = {candidate: summarize([float(x['phi_adjusted_max12'])
                    for x in groups['主条件', candidate]])
                    for candidate in ['Zhang11', 'Zhang11-HCv2-1']}
    print(json.dumps({'published_tables_consistent': True, 'case_count': len(case_rows),
                     'attack_score_count': len(attack_rows), 'main_summary': main_summary,
                     'paired': paired,
                     'meaning': '掲載表の整合性確認。補完モデルの正しさや本番得点の保証ではない。'},
                    ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
