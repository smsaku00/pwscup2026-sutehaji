"""保存済みCSVの形式・hash・希少値を確認し、任意で予備戦を公式式で再採点する。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[2]
QI7 = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(score_prelim):
    manifest = json.loads((HERE / 'manifest.json').read_text())
    evidence = json.loads((HERE / '検証結果.json').read_text())
    for rel, expected in evidence['artifact_sha256'].items():
        if sha(HERE / rel) != expected:
            raise ValueError('成果物hash不一致: ' + rel)
    for rel, expected in manifest['dependency_sha256'].items():
        if sha(ROOT / rel) != expected:
            raise ValueError('依存コードhash不一致: ' + rel)
    frames, masks = {}, {}
    for phase, spec in manifest['phases'].items():
        for rel, expected in spec['input_sha256'].items():
            if sha(ROOT / rel) != expected:
                raise ValueError('公開入力hash不一致: ' + rel)
        bpath = next(ROOT / rel for rel in spec['input_sha256'] if rel.endswith('/B_self.csv'))
        own = 5 if phase == 'prelim' else 28
        qpath = next(ROOT / rel for rel in spec['input_sha256'] if rel.endswith(f'/aia_challenge_{own}.csv'))
        b = pd.read_csv(bpath, float_precision='round_trip')
        q = pd.read_csv(qpath, float_precision='round_trip')
        private = pd.MultiIndex.from_frame(b[QI7]).isin(pd.MultiIndex.from_frame(q[QI7]))
        rare = b.is_rare.to_numpy(bool)
        known = private & ~rare
        frame = pd.read_csv(HERE / f'outputs/{phase}/F_mia.csv', float_precision='round_trip')
        if list(frame.columns) != ['record_id', *map(str, spec['targets'])] or not np.array_equal(frame.record_id, b.record_id):
            raise ValueError('CSVの列・ID順不一致')
        values = frame.iloc[:, 1:].to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1) or not np.all(values[known] == 0):
            raise ValueError('CSV値域・既知非会員不一致')
        if (int((~private & ~rare).sum()), int(known.sum()), int(rare.sum())) != (875, 150, 24):
            raise ValueError('公開マスク不一致')
        if phase == 'main':
            baseline = ROOT / '攻撃フェーズ/MIA/20261008_第3回MIA_固定5bitと非会員較正/結果/F_mia.csv'
            if sha(baseline) != spec['baseline_sha256']:
                raise ValueError('本戦基準hash不一致')
            original = pd.read_csv(baseline, float_precision='round_trip')
            if not np.array_equal(values[rare], original.iloc[:, 1:].to_numpy(float)[rare]):
                raise ValueError('本戦希少24行が原本と不一致')
        elif not np.all(values[rare] == 0):
            raise ValueError('予備戦の比較用希少値不一致')
        frames[phase], masks[phase] = frame, ~rare
    if score_prelim:
        # 両フェーズ全CSVのhash・形式検証完了後にだけ予備戦所属正解を読む。
        scorer_path = next(ROOT / p for p in manifest['dependency_sha256'] if p.endswith('/attack_score.py'))
        tree = ast.parse(scorer_path.read_text(), filename=str(scorer_path))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_tpr_at_fpr')
        scope = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(scorer_path), 'exec'), scope)
        truthdir = ROOT / '公式データ/予備戦/結果/正解データ'
        own_truth = pd.read_csv(truthdir / 'pool_ids_5.csv').set_index('record_id').loc[frames['prelim'].record_id]
        baseline = pd.read_csv(HERE / 'outputs/prelim/第3回通常者.csv', float_precision='round_trip')
        expected = pd.read_csv(HERE / '予備戦対象別.csv', float_precision='round_trip').set_index('team')
        rows = []
        mask = masks['prelim']
        for t in manifest['phases']['prelim']['targets']:
            pool = pd.read_csv(truthdir / f'pool_ids_{t}.csv')
            truth = own_truth.pool_id.isin(pool.pool_id).to_numpy(int)
            old, _ = scope['_tpr_at_fpr'](baseline[str(t)].to_numpy()[mask], truth[mask], .01)
            new, _ = scope['_tpr_at_fpr'](frames['prelim'][str(t)].to_numpy()[mask], truth[mask], .01)
            if abs(old - expected.loc[t, '第3回通常者']) > 1e-12 or abs(new - expected.loc[t, '背景LOOゲート_縮約100']) > 1e-12:
                raise ValueError('公式式再採点の不一致: Team' + str(t))
            rows.append({'team': t, 'baseline': old, 'cycle67': new, 'delta': new-old})
        table = pd.DataFrame(rows)
        output = HERE / '.runs'
        output.mkdir(exist_ok=True)
        table.to_csv(output / '予備戦再採点.csv', index=False, float_format='%.17g')
        print(json.dumps({'score_status': 'PASS', 'baseline_mean': table.baseline.mean(),
                          'cycle67_mean': table.cycle67.mean()}, ensure_ascii=False))
    print('PASS: 全公開入力hash・CSV形式・既知150・本戦希少24原本維持')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score-prelim', action='store_true')
    run(parser.parse_args().score_prelim)
