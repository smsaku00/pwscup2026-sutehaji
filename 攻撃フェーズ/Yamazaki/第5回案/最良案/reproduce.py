"""cycle67を公開入力から再計算する。探索用cacheや正解データは使わない。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True


def deny_truth(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, Path)):
        path = args[0].decode(errors='replace') if isinstance(args[0], bytes) else str(args[0])
        if 'pool_ids' in path or '正解' in path or path.lower().endswith(('.xlsx', '.xls')):
            raise PermissionError('予測生成では正解・結果表を読めません: ' + path)


sys.addaudithook(deny_truth)

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[2]
NUM = ('age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG', 'time')
CAT = ('sex', 'smoking', 'onset', 'death')
COMMON = ROOT / '攻撃フェーズ/Tashiro/MIA全体/experiments/20261007'
MAIN_BASE = ROOT / '攻撃フェーズ/MIA/20261008_第3回MIA_固定5bitと非会員較正/結果/F_mia.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def nearest(query, reference, qcat, rcat, loo=False):
    result = np.empty(len(query))
    for start in range(0, len(query), 512):
        stop = min(start + 512, len(query))
        distance = cdist(query[start:stop], reference, 'sqeuclidean')
        for q, r in zip(qcat, rcat):
            distance += q[start:stop, None] != r[None, :]
        if loo:
            distance[np.arange(stop-start), np.arange(start, stop)] = np.inf
        result[start:stop] = distance.min(axis=1)
    return result


def ranks(values, normal):
    result = np.zeros_like(values)
    result[normal] = np.column_stack([
        rankdata(column[normal], method='average') / normal.sum() for column in values.T
    ])
    return result


def final(values, normal):
    result = np.zeros_like(values)
    # 第3回と同じ演算順序を保ち、浮動小数点の最下位bitまで再現する。
    result[normal] = .6 + .3 * np.column_stack([
        rankdata(column[normal], method='min') for column in values.T
    ]) / normal.sum()
    return result


def cross(values, normal):
    rank = ranks(values, normal)
    competitors = np.column_stack([
        np.median(np.delete(rank, j, axis=1), axis=1) for j in range(rank.shape[1])
    ])
    return final(rank - .1 * competitors, normal)


def write(path, ids, teams, values):
    frame = pd.DataFrame(values, columns=list(map(str, teams)))
    frame.insert(0, 'record_id', ids)
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')
    if not np.array_equal(pd.read_csv(path, float_precision='round_trip').to_numpy(), frame.to_numpy()):
        raise ValueError('CSV roundtrip mismatch')


def run(phase, output):
    output = output.resolve()
    if not output.is_relative_to(BASE.resolve()) or output == BASE.resolve() or output.exists():
        raise ValueError('第5回案内の新規出力先を指定してください（上書き不可）')
    manifest = json.loads((HERE / 'manifest.json').read_text())
    pinned = manifest['phases'][phase]
    for rel, expected in {**manifest['dependency_sha256'], **pinned['input_sha256']}.items():
        if sha(ROOT / rel) != expected:
            raise ValueError('入力・依存コードのhash不一致: ' + rel)
    common = load(COMMON / '分析.py', '分析')
    calibration = load(COMMON / '非会員を使う順位較正の検討.py', 'minimal_calibration')
    if phase == 'main':
        b, a, _, private, rare, cs, targets, self_team = common.phase_inputs(ROOT, phase)
    else:
        data = ROOT / '公式データ/予備戦'
        own = next(data.glob('PWSCup2026_prelim_team05_*'))
        targets = data / 'PWSCup2026_prelim_attack'
        self_team = 5
        b = common.read(own / 'B_self.csv')
        a = common.read(data / 'PWSCup2026_prelim_common/A_bg.csv')
        q = common.read(targets / 'aia_challenge_5.csv')
        private = pd.MultiIndex.from_frame(b[list(common.QI7)]).isin(pd.MultiIndex.from_frame(q[list(common.QI7)]))
        rare = b.is_rare.to_numpy(bool)
        cs = {int(p.stem.split('_')[-1]): common.read(p) for p in sorted(targets.glob('C_*.csv'))}
        cs.pop(self_team, None)
    teams = [int(t) for t in common.read_json(targets / 'mia_columns.json')['target_columns'] if int(t) != self_team]
    ids = b.record_id.to_numpy(int)
    normal, known = ~private & ~rare, private & ~rare
    if teams != pinned['targets'] or set(cs) != set(teams) or len(a) != 5143 or not np.array_equal(ids, np.arange(1049)):
        raise ValueError('入力件数・対象・ID順の不一致')
    if (int(normal.sum()), int(known.sum()), int(rare.sum())) != (875, 150, 24):
        raise ValueError('公開マスクの不一致')
    qs = {t: common.read(targets / f'aia_challenge_{t}.csv') for t in sorted(teams + [self_team])}
    scale = a[list(NUM)].std(ddof=1).to_numpy(float)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.)
    an, bn = [f[list(NUM)].to_numpy(float) / scale for f in (a, b)]
    ac, bc = [[f[k].to_numpy() for k in CAT] for f in (a, b)]
    daa = nearest(an, an, ac, ac, loo=True)
    dba = nearest(bn, an, bc, ac)
    nn, cdf100, cdf500, tail_counts = [], [], [], []
    for j, t in enumerate(teams):
        c = cs[t]
        cn, cc = c[list(NUM)].to_numpy(float) / scale, [c[k].to_numpy() for k in CAT]
        sa = np.log(daa + 1e-12) - np.log(nearest(an, cn, ac, cc) + 1e-12)
        sb = np.log(dba + 1e-12) - np.log(nearest(bn, cn, bc, cc) + 1e-12)
        count = len(a) - np.searchsorted(np.sort(sa), sb, side='left')
        tail_counts.append(int(((1 + count[normal]) / (len(a) + 1) <= .001).sum()))
        nn.append(sb)
        controls = pd.concat([qs[u][list(common.QI7)] for u in sorted(qs) if u != t], ignore_index=True)
        if len(controls) != pinned['foreign_Q_count'] or controls.duplicated(list(common.QI7)).any():
            raise ValueError('非会員Qの件数・重複不一致')
        query = pd.concat([b[list(common.QI7)], controls], ignore_index=True)
        scores = common.core(a, query, c, common.NUM7, (10,))[10]
        for shrink, dest in ((100., cdf100), (500., cdf500)):
            dest.append(calibration.calibrate(scores[:len(b)], b, scores[len(b):], controls, ['prefecture', 'sex'], shrink))
        print(f'{phase} {j+1}/{len(teams)} Team{t}', flush=True)
    baseline = cross(np.column_stack(cdf500), normal)
    fallback = cross(np.column_stack(cdf100), normal)
    native = final(ranks(np.column_stack(nn), normal), normal)
    gate = np.asarray(tail_counts) >= 6
    prediction = np.where(gate, native, fallback)
    if phase == 'main':
        original = pd.read_csv(MAIN_BASE, float_precision='round_trip')
        if list(original.columns) != ['record_id', *map(str, teams)] or not np.array_equal(original.record_id, ids):
            raise ValueError('本戦原本の列・ID不一致')
        for values in (baseline, prediction):
            values[rare] = original.iloc[:, 1:].to_numpy(float)[rare]
        if not np.array_equal(baseline, original.iloc[:, 1:].to_numpy(float)):
            raise ValueError('第3回本戦通常者再現の不一致')
    if not np.isfinite(prediction).all() or np.any(prediction < 0) or np.any(prediction > 1) or not np.all(prediction[known] == 0):
        raise ValueError('予測値・既知非会員の不一致')
    for rel, expected in {**manifest['dependency_sha256'], **pinned['input_sha256']}.items():
        if sha(ROOT / rel) != expected:
            raise ValueError('生成中に入力が変更されました: ' + rel)
    output.mkdir(parents=True)
    write(output / 'F_mia.csv', ids, teams, prediction)
    write(output / '第3回通常者.csv', ids, teams, baseline)
    checks = {'F_mia.csv': pinned['prediction_sha256'], '第3回通常者.csv': pinned['baseline_sha256']}
    for name, expected in checks.items():
        if sha(output / name) != expected:
            raise ValueError('凍結済みCSVとのhash不一致: ' + name)
    record = {'phase': phase, 'cycle': 67, 'active_targets': [t for t, active in zip(teams, gate) if active],
              'unknown_tail_counts': tail_counts, 'source_sha256': sha(Path(__file__)),
              'manifest_sha256': sha(HERE / 'manifest.json'), 'input_sha256': pinned['input_sha256'],
              'dependency_sha256': manifest['dependency_sha256'], 'output_sha256': checks,
              'truth_read': False, 'seed': None, 'counts': {'unknown': 875, 'known': 150, 'rare': 24},
              'versions': {'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__}}
    (output / '再現記録.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'phase': phase, 'active_targets': record['active_targets']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prelim', 'main'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.phase, args.output)
