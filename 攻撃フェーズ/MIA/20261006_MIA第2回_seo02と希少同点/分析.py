"""承認済みの第2回F_miaを作成する。公式提出は行わない。

実行: ルート .venv/bin/python -B <このファイル>
検査が全て成功した場合のみ結果/へ移す。既存結果は上書きしない。
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = HERE.parent
FIRST = PARENT / '20261005_MIA希少_第1回同点データ'
PROOF = PARENT / '20261006_MIA希少_本戦102配置検証/結果'
INPUT = PARENT / '20260928_配布データ基本分析'
SEO = ROOT / '本戦/バトル/0923/programs/attack/seo-02-多尺度密度比と差分時間窓/seo-02-多尺度密度比と差分時間窓.py'
SEO_SHA = 'f5ebe3d5fad771d7504c8aafce5d29861764cd090161a26fe3aa3f7c25d55511'
OUT = HERE / '結果'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(rows)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if not __debug__:
        raise RuntimeError('検査を省略する実行オプションは禁止です')
    if OUT.exists():
        raise FileExistsError('結果/は上書きしません。検証.pyで再確認してください。')
    start = perf_counter()
    assert Path(sys.executable).absolute() == ROOT / '.venv/bin/python'
    assert sha(SEO) == SEO_SHA
    assert sha(PROOF/'初期条件.json') == '67285b9d6aeebf1a75ec43f692db1f15869e1a3db6f099340155378a8cae4ccb'
    assert sha(PROOF/'人数102通り.json') == '51be9f8d08ac1c785c01e4719bce39f44d907af828803de4e886a9d812897201'
    initial = read_json(PROOF/'初期条件.json')
    for name, digest in initial['source_sha256'].items():
        assert sha(Path(name)) == digest, name
    first = load_module('round1_builder', FIRST/'分析.py')
    _, first_plan, declared, first_hashes = first.load_inputs()
    for name, digest in read_json(FIRST/'結果/出力SHA256.json').items():
        assert sha(FIRST/'結果'/name) == digest, name
    public = initial['public']
    teams = public['teams']
    assert teams == first.TEAMS
    assert public['rare_ids'] == first_plan['rare_record_ids']
    assert public['unknown_ids'] == first_plan['unknown_rare_record_ids']
    assert public['known_nonmember_ids'] == first_plan['known_nonmember_rare_record_ids']
    revised = [t['round2_tied'] for t in first_plan['teams']]
    assert revised[teams.index('25')] == 14
    revised[teams.index('25')] = 21
    assert revised == public['round2_sizes']
    manifest = read_json(INPUT/'入力記録.json')
    for entry in manifest['files']:
        assert sha(INPUT/entry['file']) == entry['sha256'], entry['file']
    seo = load_module('seo02', SEO)
    bpath, = (ROOT/'本戦').glob('配布*/PWSCup2026_final_team28_*/B.csv')
    b = seo.read_table(bpath, 'B')
    a = seo.read_table(INPUT/'入力/A_bg.csv', 'A')
    assert len(b) == 1049 and len(a) == 5143
    assert b.record_id.tolist() == list(range(1049))
    # 希少フラグと既知非会員は提出値の組立側だけで使い、seo-02へは渡さない。
    bself = pd.read_csv(bpath.with_name('B_self.csv'), float_precision='round_trip')
    assert bself.drop(columns='is_rare').equals(pd.read_csv(bpath, float_precision='round_trip'))
    private = pd.read_csv(first.BASE/'結果/本戦_非会員163人.csv').record_id.tolist()
    rare = set(public['rare_ids'])
    regular_ids = sorted(set(b.record_id) - rare - set(private))
    negative_ids = sorted(set(private) - rare)
    assert (len(regular_ids), len(negative_ids)) == (875, 150)
    hashes = {str(ROOT/p): digest for p, digest in first_hashes.items()}
    paths = [Path(__file__), HERE/'検証.py', HERE/'結果読取.py', SEO, bpath,
             FIRST/'結果/F_mia.csv', FIRST/'結果/設定.json', FIRST/'結果/出力SHA256.json',
             PROOF/'初期条件.json', PROOF/'人数102通り.json', PROOF/'検証結果.json',
             INPUT/'入力記録.json', ROOT/'requirements.txt']
    paths += [INPUT/entry['file'] for entry in manifest['files']]
    paths += [Path(p) for p in initial['source_sha256']]
    hashes.update({str(p): sha(p) for p in paths})
    plan = {
        'round': 2, 'own_team': 28, 'teams': teams,
        'overall_method': 'seo-02-多尺度密度比と差分時間窓 のMIA部分',
        'overall_source_sha256': SEO_SHA,
        'overall_candidate_ids': regular_ids, 'nonrare_known_nonmember_ids': negative_ids,
        'overall_mapping': '0.55 + 0.40 * average_rank / 875。同点は維持。',
        'rare_ids': public['rare_ids'], 'unknown_rare_ids': public['unknown_ids'],
        'known_nonmember_rare_ids': public['known_nonmember_ids'],
        'rare_high': '0.4', 'rare_low': '0.1', 'first_rare_score': public['first_score'],
        'rare_design_change': '元の第2回設計からチーム25だけ14人→21人。他21チームは同じ。',
        'rare_settings': [{'target_k': t, 'tied_count': n,
                           'high_record_ids': sorted(public['unknown_ids'] + public['known_nonmember_ids'][:n-11])}
                          for t, n in zip(teams, revised)],
        'assumptions': ['希少24人、未確定11人、確定非会員13人', '各対象チームの希少会員は0〜2人',
                        '対象22チームへの希少会員数の合計は11以下',
                        '1回目は保存済みCSVと同じ内容で提出された',
                        '保存した公式採点コードと同じ採点と平均方法が使われる'],
        'submission_status': '未提出。F_miaのみ。AIA・token・提出ZIPは含まない。',
    }
    raw = pd.DataFrame({'record_id': b.record_id})
    answer = {rid: ['0'] * len(teams) for rid in b.record_id}
    scaler = seo.Scaler(a)
    for col, t in enumerate(teams):
        c = seo.read_table(INPUT/f'入力/C_{t}.csv', 'C')
        assert len(c) == 1049
        scores = seo.mia_scores(b, a, c, scaler)
        assert scores.shape == (1049,) and np.isfinite(scores).all()
        raw[t] = scores
        ranks = rankdata(scores[regular_ids], method='average')
        # 同じ式を整数係数で計算し、最大値が0.9500000000000001になることを防ぐ。
        mapped = (55 * len(regular_ids) + 40 * ranks) / (100 * len(regular_ids))
        for rid, value in zip(regular_ids, mapped):
            answer[rid][col] = format(float(value), '.17g')
        high = set(plan['rare_settings'][col]['high_record_ids'])
        for rid in public['rare_ids']:
            answer[rid][col] = '0.4' if rid in high else '0.1'
        print(f'全体MIA {col+1}/22 完了: team {t}', flush=True)
    # この時点では完成版の保存先を作らない。別の検査プログラムへ実CSVを渡す。
    with tempfile.TemporaryDirectory(prefix='.検証中_', dir=HERE) as tmp:
        stage = Path(tmp)/'結果'
        stage.mkdir()
        write_csv(stage/'F_mia.csv', [['record_id', *teams]] + [[rid, *answer[rid]] for rid in b.record_id])
        with (stage/'全体手法の出力.csv').open('x', encoding='utf-8', newline='') as f:
            raw.to_csv(f, index=False, float_format='%.17g')
        write_json(stage/'設定.json', plan)
        write_json(stage/'入力SHA256.json', hashes)
        shutil.copyfile(SEO, stage/'参照seo02.py')
        check = subprocess.run([sys.executable, '-B', str(HERE/'検証.py'), '--directory', str(stage), '--write'],
                               cwd=ROOT, check=True, text=True)
        assert check.returncode == 0
        assert all(sha(Path(p)) == digest for p, digest in hashes.items()), '作業中に入力が変化しました'
        write_json(stage/'実行記録.json', {
            'created_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            'python': platform.python_version(), 'executable': sys.executable,
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'elapsed_seconds': perf_counter()-start, 'command': sys.argv,
            'scope': '承認された第2回MIAのみを作成。未公開正解・提出APIは利用していない。',
        })
        write_json(stage/'出力SHA256.json', {p.name: sha(p) for p in sorted(stage.iterdir()) if p.is_file()})
        stage.rename(OUT)
    print(f'全検査完了: {OUT / "F_mia.csv"}', flush=True)


if __name__ == '__main__':
    main()
