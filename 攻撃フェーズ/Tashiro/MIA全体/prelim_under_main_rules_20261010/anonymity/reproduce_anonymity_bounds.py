"""Read-only verification of official preliminary XLSX and main-rule anonymity bounds.

No attack or person-level data are generated or modified. The historical attack
submissions, truth, rare flags, participants and all official per-attacker scores
are held fixed. Unknown preliminary-world phi_i is never substituted with zero.
Uses only the Python standard library; run with the primary runtime Python.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import zipfile
import xml.etree.ElementTree as ET

import argparse
SOURCE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--official-workbook',type=Path,required=True)
parser.add_argument('--out-dir',type=Path,required=True)
args=parser.parse_args()
OUT=args.out_dir
OUT.mkdir(parents=True,exist_ok=True)
XLSX=args.official_workbook
TEAMS20 = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 24, 26, 29]
SOURCE_URL = 'https://drive.google.com/file/d/1BLiwflR7FJ9tp5lCeSkP-kJga4ByHRnB/view'
RULE_URL = 'https://github.com/pwscup/pwscup2026-kit/blob/3dd71bd3436f2abfdbb64da97e860f22729ea054/rulebook/PWS_Cup2026_ルールブック_main-process-20260912.pdf'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def close(x, y):
    assert math.isclose(float(x), float(y), abs_tol=1e-12, rel_tol=0), (x, y)


def clip(v):
    return min(1.0, max(0.0, v))


def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (OUT / name).open(encoding='utf-8-sig', newline='') as f:
        check = list(csv.DictReader(f))
    assert len(check) == len(rows)


def read_xlsx():
    with zipfile.ZipFile(XLSX) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(t.text or '' for t in si.iter('{' + NS['s'] + '}t'))
                       for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si', NS)]

        def sheet(number):
            rows = []
            for row in ET.fromstring(z.read(f'xl/worksheets/sheet{number}.xml')).findall('.//s:sheetData/s:row', NS):
                record = {'row': int(row.get('r'))}
                for c in row.findall('s:c', NS):
                    assert c.find('s:f', NS) is None, 'Unexpected formula: cached value would require separate verification.'
                    v = c.find('s:v', NS)
                    value = v.text if v is not None else None
                    if c.get('t') == 's':
                        value = strings[int(value)]
                    if c.get('t') == 'inlineStr':
                        value = ''.join(t.text or '' for t in c.iter('{' + NS['s'] + '}t'))
                    record[c.get('r').rstrip('0123456789')] = value
                rows.append(record)
            return rows
        return sheet(1), sheet(3), sheet(4), sheet(7)


def main():
    initial_hash = hashlib.sha256(XLSX.read_bytes()).hexdigest()
    assert initial_hash == '8d6380feafa50a850b28aa50eb4b04cd0ffe132ca15904dc0c84b85fd8ff6508'
    summary, aia, mia, metadata = read_xlsx()
    meta = {r['A']: r.get('B') for r in metadata[1:]}
    assert meta['config.scoring.low_fpr'] == '0.01'
    assert meta['config.scoring.mia_fold_K'] == '12'
    assert meta['config.scoring.mia_null_reps'] == '20000'
    assert meta['config.scoring.mia_null_quantile'] == '0.95'
    assert meta['config.scoring.aia_rare_se_z'] == '1.64'
    assert meta['config.aia.tau_time'] == '0.5'
    aia_by_team = {int(r['A']): r for r in aia[1:]}
    snapshot = json.loads((SOURCE_DIR / 'official_source_snapshot.json').read_text())
    attack_files = [f for f in snapshot['files'] if f[1] == 'attack_score.py']
    assert len(attack_files) == 3
    assert len({f[2]['sha'] for f in attack_files}) == 1
    assert len({f[2]['content'] for f in attack_files}) == 1
    current_cfg = next(f[2]['content'] for f in snapshot['files'] if f[1] == 'scoring_config.yaml')
    for snippet in ['low_fpr: 0.01', 'mia_fold_K: 12', 'aia_rare_se_z: 1.64', 'tau_time: 0.5']:
        assert snippet in current_cfg
    rows = []
    for s in summary[1:]:
        team = int(s['A'])
        out = {
            'team': team,
            'team_name': s['B'],
            'in_previous20_comparison': team in TEAMS20,
            'official_prelim_available': s.get('C') is not None,
            'official_prelim_U': '',
            'official_prelim_beta_raw': '',
            'official_prelim_beta_chance': '',
            'official_prelim_beta': '',
            'official_prelim_MIA': '',
            'official_prelim_AIA': '',
            'official_prelim_Anon': '',
            'official_prelim_total': '',
            'fixed_attackers_count': '',
            'official_AIA_nonmissing_attackers_count': '',
            'main_formula_MIA_fixed_attacks': '',
            'prelim_world_phi': '',
            'prelim_world_phi_status': '未取得。予備戦世界・コホート・標的対照構成での校正が必要',
            'phi_logical_lower': 0.0,
            'phi_logical_upper': 1.0,
            'main_formula_AIA_lower': '',
            'main_formula_AIA_upper': '',
            'main_formula_Anon_lower': '',
            'main_formula_Anon_upper': '',
            'main_formula_Anon_exact': '',
            'Anon_independent_of_phi': '',
            'max_Anon_increase_due_to_phi': '',
            'phi_threshold_for_Anon_upper': '',
            'source_cells': f'総括!A{s["row"]}:N{s["row"]}',
            'official_result_url': SOURCE_URL,
            'main_rule_url': RULE_URL,
            'counterfactual_scope': '予備戦のC・全攻撃提出・真値・希少フラグ・参加者を固定し、本戦の集計式だけを適用。本戦実結果ではない',
            'missing_reason': '',
        }
        if s.get('C') is None:
            assert team == 21
            out['missing_reason'] = '公式総括のU・攻撃成分・匿名性が空欄。0として扱わない'
            rows.append(out)
            continue
        u, raw, chance, beta, m, a, anon, total = [float(s[c]) for c in ['C', 'H', 'I', 'J', 'K', 'L', 'M', 'N']]
        close(u, min(float(s[c]) for c in ['D', 'E', 'F', 'G']))
        close(beta, max(raw - chance + 0.01, 0.01))
        close(m, clip((beta / 0.01 - 1) / 11))
        close(anon, 1 - max(m, a))
        close(total, min(u, anon))
        attackers = [r for r in mia[1:] if int(r['B']) == team]
        assert len(attackers) == 20
        for t in attackers:
            expected = max(float(t['E']), float(t['F'])) if int(t['D']) > 0 else float(t['E'])
            if int(t['D']) == 0:
                assert t.get('F') is None
            close(t['G'], expected)
        close(raw, statistics.mean(float(r['G']) for r in attackers))
        ar = aia_by_team[team]
        av = [float(v) for k, v in ar.items() if k not in ['row', 'A'] and v is not None]
        # The published AIA matrix has one extra blank: defender 23, attacker 4.
        # Preserve this missingness rather than treating it as zero or inventing a score.
        assert len(av) == (19 if team == 23 else 20)
        close(a, max(av))
        lower = 1 - max(m, a)
        upper = 1 - m
        independent = a <= m
        threshold = max(0.0, a - m)
        # Verify endpoints, bounds and monotonicity on a dense diagnostic grid.
        vals = [1 - max(m, clip(a - phi / 1000)) for phi in range(1001)]
        close(vals[0], lower)
        close(vals[-1], upper)
        assert all(lower - 1e-14 <= x <= upper + 1e-14 for x in vals)
        assert all(x <= y + 1e-14 for x, y in zip(vals, vals[1:]))
        close(1 - max(m, clip(a - threshold)), upper)
        out.update({
            'official_prelim_U': u,
            'official_prelim_beta_raw': raw,
            'official_prelim_beta_chance': chance,
            'official_prelim_beta': beta,
            'official_prelim_MIA': m,
            'official_prelim_AIA': a,
            'official_prelim_Anon': anon,
            'official_prelim_total': total,
            'fixed_attackers_count': len(attackers),
            'official_AIA_nonmissing_attackers_count': len(av),
            'main_formula_MIA_fixed_attacks': m,
            'main_formula_AIA_lower': 0.0,
            'main_formula_AIA_upper': a,
            'main_formula_Anon_lower': lower,
            'main_formula_Anon_upper': upper,
            'main_formula_Anon_exact': lower if independent else '',
            'Anon_independent_of_phi': independent,
            'max_Anon_increase_due_to_phi': upper - lower,
            'phi_threshold_for_Anon_upper': threshold,
        })
        rows.append(out)
    assert len(rows) == 22
    valid = [r for r in rows if r['official_prelim_available']]
    assert len(valid) == 21
    selected = [r for r in rows if r['team'] in TEAMS20]
    assert len(selected) == 20
    write_csv('main_rule_anonymity_bounds_all22.csv', rows)
    write_csv('main_rule_anonymity_bounds_previous20.csv', selected)
    with (OUT / 'main_rule_anonymity_bounds_all22.json').open('w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    report = {
        'official_result_sha256': initial_hash,
        'rows': 22, 'valid': 21, 'blank_official_team': 21,
        'exact_Anon_teams': [r['team'] for r in valid if r['Anon_independent_of_phi']],
        'phi_dependent_Anon_teams': [r['team'] for r in valid if not r['Anon_independent_of_phi']],
        'identical_attack_score_git_blob_all_three_phases': attack_files[0][2]['sha'],
        'MIA_null_method': 'Reuse official 20,000-replicate beta_chance; same historical submitted confidences, truth, rare flags and attacker set are held fixed. Null permutations were not rerun.',
        'AIA_phi_method': 'Unknown preliminary-world-specific calibration; no actual main-world phi is substituted.',
        'bounds_meaning': 'Logical bounds over 0 <= phi_i <= 1, not a confidence interval or prediction; endpoint attainability by the actual unknown calibration is not asserted.',
        'AIA_missingness': 'Defender 23 / attacker 4 is blank in the official AIA matrix (19 nonmissing values for defender 23; 20 otherwise). Reuse the official maximum over available entries; do not invent the missing attack result.',
        'checks': ['all stored official summary identities reproduce', 'all 420 per-attacker MIA t_j match max(all, rare) with no-rare fallback', 'all 21 beta_raw equal 20-attacker means', 'all 21 AIA maxima match the nonmissing entries in the published attacker matrix', 'same official attack_score.py in all three tags', 'K/FPR/AIA tolerance/rare deadzone identical', 'all bounds endpoints and monotonicity checked', 'source workbook unchanged'],
        'formulae': {
            'aia': 'A_main_i(phi_i) = clip(A_pre_i - phi_i, 0, 1)',
            'anonymity': 'Anon_main_i(phi_i) = 1 - max(M_pre_i, A_main_i(phi_i))',
            'anonymity_bounds': 'Anon_pre_i <= Anon_main_i <= 1 - M_pre_i',
            'score': 'Score_i = min(U_main_i, Anon_main_i(phi_i))',
            'score_bounds_U_exact': '[min(U_main_i, Anon_pre_i), min(U_main_i, 1-M_pre_i)]',
            'score_bounds_U_interval': 'If U_main_i in [L_i,H_i], Score_i in [min(L_i,Anon_pre_i),min(H_i,1-M_pre_i)]',
        },
    }
    assert hashlib.sha256(XLSX.read_bytes()).hexdigest() == initial_hash
    (OUT / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for r in valid:
        print(f"{r['team']:2d} MIA={r['official_prelim_MIA']:.6f} AIA={r['official_prelim_AIA']:.6f} Anon=[{r['main_formula_Anon_lower']:.6f},{r['main_formula_Anon_upper']:.6f}]")


if __name__ == '__main__':
    main()
