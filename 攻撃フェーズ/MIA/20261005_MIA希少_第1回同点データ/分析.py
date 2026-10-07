"""希少MIAの1回目を作成。--check は保存結果を変更せず再確認する。

実行はプロジェクトルートの .venv/bin/python -B を使用する。
既存の初回回答・200ケースの設計を読み、結果/の新規作成だけを行う。
本戦の所属真値、予備戦の正解、提出APIは使用しない。
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import itertools
import json
import platform
import sys
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent / '20261005_MIA初回_GenLRA比較'
SIM = HERE.parent / '20261005_MIA希少_2回200通り検証'
INPUT = HERE.parent / '20260928_配布データ基本分析/入力'
OUT = HERE / '結果'
QI7 = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture']
TEAMS = list(map(str, [*range(18), 22, 25, 26, 27]))
HIGH, LOW = '0.4', '0.1'
BASE_SHA = 'f183400ca0f7b121cab1f42ebf001b24acf3f5dfb68f69c6e12cf2c6aa36a4a2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jread(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.reader(f))


def jwrite(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def cwrite(path, data):
    with path.open('x', encoding='utf-8', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(data)


def load_inputs():
    paths = [BASE/'結果/F_mia.csv', BASE/'結果/本戦_非会員163人.csv',
             BASE/'結果/出力SHA256.json', BASE/'結果/入力SHA256.json',
             SIM/'結果/200通り結果.json', SIM/'結果/出力SHA256.json',
             SIM/'結果/同点人数.csv', INPUT/'aia_challenge_28.csv', INPUT/'mia_columns.json',
             BASE/'参照コード/validate_20260928.py', BASE/'参照コード/attack_score_20260928.py',
             HERE/'分析.py']
    bpaths = list((ROOT/'本戦').glob('配布*/PWSCup2026_final_team28_*/B_self.csv'))
    assert len(bpaths) == 1
    bpath = bpaths[0]
    paths.append(bpath)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    old_inputs = jread(BASE/'結果/入力SHA256.json')
    for p in [bpath, INPUT/'aia_challenge_28.csv', INPUT/'mia_columns.json',
              BASE/'参照コード/validate_20260928.py', BASE/'参照コード/attack_score_20260928.py']:
        assert sha(p) == old_inputs[str(p.relative_to(ROOT))], str(p)
    old_outputs = jread(BASE/'結果/出力SHA256.json')
    for name in ['F_mia.csv', '本戦_非会員163人.csv']:
        assert sha(BASE/'結果'/name) == old_outputs[name]
    assert sha(BASE/'結果/F_mia.csv') == BASE_SHA
    sim_outputs = jread(SIM/'結果/出力SHA256.json')
    for name in ['200通り結果.json', '同点人数.csv']:
        assert sha(SIM/'結果'/name) == sim_outputs[name]
    design = jread(SIM/'結果/200通り結果.json')['design']
    design_csv = rows(SIM/'結果/同点人数.csv')
    assert design == [[int(r[1]), int(r[3])] for r in design_csv[1:]]
    assert len(design) == 22 and all(11 <= n <= 24 for pair in design for n in pair)
    b = pd.read_csv(bpath, float_precision='round_trip')
    q = pd.read_csv(INPUT/'aia_challenge_28.csv', float_precision='round_trip')
    assert len(b) == 1049 and b.record_id.tolist() == list(range(1049))
    assert set(b.is_rare) == {0, 1}
    assert not b.duplicated(QI7).any() and not q.duplicated(QI7).any()
    matched = b.merge(q[QI7], on=QI7, validate='one_to_one')
    private = set(map(int, matched.record_id))
    rare = set(map(int, b.loc[b.is_rare == 1, 'record_id']))
    old_private = pd.read_csv(BASE/'結果/本戦_非会員163人.csv')
    assert private == set(old_private.record_id)
    assert rare & private == set(old_private.loc[old_private.is_rare == 1, 'record_id'])
    unknown, negative = sorted(rare - private), sorted(rare & private)
    assert (len(q), len(private), len(rare), len(unknown), len(negative)) == (326, 163, 24, 11, 13)
    declared = jread(INPUT/'mia_columns.json')
    assert [t for t in declared['target_columns'] if t != '28'] == TEAMS
    original = rows(BASE/'結果/F_mia.csv')
    assert original[0] == ['record_id', *TEAMS]
    assert [int(r[0]) for r in original[1:]] == list(range(1049))
    plan = {'round': 1, 'own_team': 28, 'baseline_sha256': BASE_SHA,
            'design_source': str((SIM/'結果/200通り結果.json').relative_to(ROOT)),
            'high_value': HIGH, 'low_value': LOW, 'rare_record_ids': sorted(rare),
            'unknown_rare_record_ids': unknown, 'known_nonmember_rare_record_ids': negative,
            'padding_order': 'known_nonmember_rare_record_idsのrecord_id昇順',
            'teams': [{'simulation_label': f'T{i+1:02d}', 'target_k': t,
                       'round1_tied': ns[0], 'round2_tied': ns[1],
                       'round1_high_record_ids': sorted(unknown + negative[:ns[0]-11])}
                      for i, (t, ns) in enumerate(zip(TEAMS, design))]}
    return original, plan, declared, hashes


def official_tools(hashes):
    kit = next((ROOT/'本戦').glob('配布*/pwscup2026-kit-main_revised0912'))
    sys.path.insert(0, str(kit/'starter/src'))
    path = BASE/'参照コード/validate_20260928.py'
    name = 'pwscup2026.kit.validate_rare_round1'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for name, loaded in list(sys.modules.items()):
        path = getattr(loaded, '__file__', None)
        if name.startswith('pwscup2026') and path:
            p = Path(path).resolve()
            if p.suffix == '.py':
                hashes[str(p.relative_to(ROOT))] = sha(p)
    scorepath = BASE/'参照コード/attack_score_20260928.py'
    tree = ast.parse(scorepath.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_tpr_at_fpr')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(scorepath), 'exec'), namespace)
    return module.validate_mia_submission, namespace['_tpr_at_fpr']


def create_rows(original, plan):
    data = [r.copy() for r in original]
    row_index = {int(r[0]): i for i, r in enumerate(data[1:], 1)}
    for j, t in enumerate(plan['teams'], 1):
        high = set(t['round1_high_record_ids'])
        for rid in plan['rare_record_ids']:
            data[row_index[rid]][j] = HIGH if rid in high else LOW
    return data


def audit(data, original, plan, declared, validator, score):
    assert len(data) == 1050 and data[0] == original[0]
    assert all(len(r) == 23 for r in data)
    rare = set(plan['rare_record_ids'])
    changed = 0
    for r, old in zip(data[1:], original[1:]):
        assert r[0] == old[0]
        if int(r[0]) not in rare:
            assert r == old  # 通常1,025人は小数の文字列も維持する。
        changed += sum(a != b for a, b in zip(r[1:], old[1:]))
    frame = pd.DataFrame(data[1:], columns=data[0]).apply(pd.to_numeric)
    assert np.isfinite(frame.to_numpy()).all()
    values = frame[TEAMS].to_numpy()
    assert ((values >= 0) & (values <= 1)).all()
    result = validator(frame, {}, {'mia_columns': declared, 'team_id': 28,
                                  'attacker_record_ids': set(range(1049))})
    assert result.ok, result.errors
    rare_frame = frame.set_index('record_id').loc[plan['rare_record_ids']]
    unknown = set(plan['unknown_rare_record_ids'])
    unknown_positions = [i for i, rid in enumerate(rare_frame.index) if rid in unknown]
    checks, error = 0, 0.0
    for t in plan['teams']:
        conf = rare_frame[t['target_k']].to_numpy(float)
        high_ids = set(rare_frame.index[conf == float(HIGH)])
        assert high_ids == set(t['round1_high_record_ids'])
        assert len(high_ids) == t['round1_tied'] and unknown <= high_ids
        assert np.all(np.isin(conf, [float(HIGH), float(LOW)]))
        # 未確定11人から1人または2人を選ぶ全66通りで、実CSVの採点を照合。
        for m in (1, 2):
            expected = float(Fraction(24-m, 100*(t['round1_tied']-m)))
            for selected in itertools.combinations(unknown_positions, m):
                truth = np.zeros(24, dtype=int)
                truth[list(selected)] = 1
                actual, _ = score(conf, truth, .01)
                error = max(error, abs(actual-expected))
                assert abs(actual-expected) < 1e-12
                checks += 1
    return {'official_mia_format': True, 'rows': 1049, 'target_columns': 22,
            'unchanged_nonrare_rows': 1025, 'unchanged_nonrare_cells': 1025*22,
            'rare_rows': 24, 'rare_cells': 24*22, 'changed_cells': changed,
            'all_unknown_11_tied_high': True, 'team_mapping_matches_saved_design': True,
            'official_score_checks': checks, 'max_abs_score_error': error,
            'actual_main_score': '未採点', 'submitted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not args.check and OUT.exists():
        raise FileExistsError('結果/は既に存在します。上書きせず--checkで確認してください。')
    original, plan, declared, hashes = load_inputs()
    validator, score = official_tools(hashes)
    if args.check:
        for name, digest in jread(OUT/'出力SHA256.json').items():
            assert sha(OUT/name) == digest, name
        assert jread(OUT/'入力SHA256.json') == hashes
        assert jread(OUT/'設定.json') == plan
        report = audit(rows(OUT/'F_mia.csv'), original, plan, declared, validator, score)
        assert report == jread(OUT/'検証結果.json')
        print(json.dumps(report, ensure_ascii=False))
        return
    data = create_rows(original, plan)
    audit(data, original, plan, declared, validator, score)
    assert all(sha(ROOT/p) == digest for p, digest in hashes.items()), '入力変更を検出'
    OUT.mkdir()
    cwrite(OUT/'F_mia.csv', data)
    # 保存・読み戻し後の実ファイルを検証。
    report = audit(rows(OUT/'F_mia.csv'), original, plan, declared, validator, score)
    jwrite(OUT/'設定.json', plan)
    cwrite(OUT/'同点設定.csv', [['設計上の名前', '本戦チーム番号', '1回目同点人数', '1回目追加する非会員',
                              '2回目同点人数_未作成', '2回目追加する非会員_未作成']]
           + [[t['simulation_label'], t['target_k'], t['round1_tied'], t['round1_tied']-11,
               t['round2_tied'], t['round2_tied']-11] for t in plan['teams']])
    rare_data = [data[0]] + [r for r in data[1:] if int(r[0]) in set(plan['rare_record_ids'])]
    cwrite(OUT/'確認用_希少24人.csv', rare_data)
    jwrite(OUT/'検証結果.json', report)
    jwrite(OUT/'入力SHA256.json', hashes)
    jwrite(OUT/'実行記録.json', {'created_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
           'python': platform.python_version(), 'executable': sys.executable,
           'numpy': np.__version__, 'pandas': pd.__version__, 'command': sys.argv,
           'scope': '希少MIAの1回目。AIA・token・ZIP・提出なし',
           'baseline': str((BASE/'結果/F_mia.csv').relative_to(ROOT))})
    assert all(sha(ROOT/p) == digest for p, digest in hashes.items()), '入力変更を検出'
    jwrite(OUT/'出力SHA256.json', {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    print(json.dumps(report, ensure_ascii=False))
    print(f'保存先: {OUT / "F_mia.csv"}')


if __name__ == '__main__':
    main()
