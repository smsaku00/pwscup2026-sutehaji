"""seo / 攻撃案01 / 近傍対比と条件付時間窓 / v1.0 / 2026-09-21

MIA + AIA, standalone Python. No private data, answers, or trained model embedded.
Rule reference: PWS Cup 2026 main-process-20260912, sections 4, 5.2, 6.2, 6.3.
This script does not calculate official scores, phi, protection, or Anon.

日本語の概要:
  MIA: C対A_bgのExtraTrees識別スコアと、背景に対する相対最近傍距離を
       それぞれ順位化し、固定1:1で平均する。確率ではなく順位用のスコア。
  AIA: CだけでQIからtimeへのExtraTreesを学習し、同じ葉に入るCの行から
       条件付き時間分布を推定する。重みが最大となる幅1年の窓の中心を答える。
       離れたtimeを単純平均しない。BのtimeをAIAの答えとして照合しない。
  使用情報: 指定した攻撃者B、公開A_bg、C、混合されたAIA課題だけ。
  未検証: 実競技データでの性能・未知加工への有効性。最強の手法ではない。
  自チーム検証: Cの加工元BをそのままMIA候補にすると全員が正例になり、
       非会員がいないためFPRやAUCを正しく評価できない。評価担当が別途、
       会員・非会員のある模擬攻撃者と、標的・対照を混ぜた課題を用意する。
       その正解ラベルはこのスクリプトへ渡さない。自己攻撃は公式攻撃力に入らない。

Dependencies tested: Python 3.12, numpy 2.4.6, pandas 2.3.3,
                     scipy 1.17.1, scikit-learn 1.9.0.
Install: python -m pip install numpy pandas scipy scikit-learn
Run:
  python "seo-01-近傍対比と条件付時間窓.py" --b B_attacker.csv --a-bg A_bg.csv --targets targets.json
  python "seo-01-近傍対比と条件付時間窓.py" --self-test

targets.json (paths relative to this JSON, NOT the current working directory):
  [{"target_k": 5, "c_path": "C_my_team.csv",
    "aia_challenge_path": "challenge_local.csv"}]
5 is an example, not an algorithmic team-specific rule.
Output beside this script: F_mia_<script stem>.csv and F_aia_<script stem>.csv.
Existing different outputs require --overwrite; input files are never overwritten.
For official target coverage additionally supply:
  --scope official --mia-columns mia_columns.json --self-team-id YOUR_ID
Local output is not a complete official submission. No ZIP, token, upload or network I/O.

Development: designed from the requested public interface and earlier general
diagnostics. No real-data tuning/evaluation performed for this version. Self-tests
are artificial functional tests, not privacy-attack performance evidence.
Reference API: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
The leaf-weight/window estimator here is custom, not a built-in sklearn feature.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

# Heuristic settings fixed before testing. These are NOT competition constants.
SEED = 20260921
N_TREES = 192
MIN_LEAF_MIA = 5
MIN_LEAF_AIA = 5
BG_NEIGHBORS = 5
DISTANCE_WEIGHT = 0.5
BATCH = 128
EPS = 1e-12

# Public schema / rules, main-process-20260912. No team-specific statistics.
HORIZON = 10.0
TAU_TIME = 0.5
CONT = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']
CAT = ['sex', 'smoking', 'prefecture']
QI_NUM = ['age', 'SBP', 'TG', 'HDL', 'ALT']
QI = QI_NUM + ['sex', 'prefecture']
COLS = ['age', 'sex', 'prefecture', 'BMI', 'SBP', 'TG', 'HDL', 'ALT',
        'smoking', 'FPG', 'time', 'onset', 'death']
RANGES = {'age': (40, 74), 'BMI': (10, 55), 'SBP': (40, 250),
          'TG': (1, 3000), 'HDL': (10, 160), 'ALT': (0.1, 700), 'FPG': (40, 400)}


def integer_series(s, label):
    text = s.astype(str)
    if not text.str.fullmatch(r'[0-9]+').all():
        raise ValueError(f'{label}: nonnegative integer identifiers required')
    values = [int(v) for v in text]
    if any(v > np.iinfo(np.int64).max for v in values):
        raise ValueError(f'{label}: identifier exceeds int64')
    result = pd.Series(values, index=s.index, dtype=np.int64)
    if result.duplicated().any():
        raise ValueError(f'{label}: duplicate identifier')
    return result


def validate_attributes(d, label):
    if d.empty or d.isna().any().any():
        raise ValueError(f'{label}: empty data or missing value')
    for col in d.columns:
        if col in ['record_id', 'challenge_row_id']:
            continue
        if col == 'prefecture':
            if not d[col].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f'{label}: empty/non-string prefecture')
            continue
        x = pd.to_numeric(d[col], errors='raise').to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError(f'{label}: nonfinite {col}')
        if col in ['age', 'sex', 'smoking', 'onset', 'death'] and not np.equal(x, np.rint(x)).all():
            raise ValueError(f'{label}: noninteger {col}')
        if col in ['sex', 'smoking', 'onset', 'death'] and not np.isin(x, [0, 1]).all():
            raise ValueError(f'{label}: {col} must be 0 or 1')
        if col in RANGES and ((x < RANGES[col][0]) | (x > RANGES[col][1])).any():
            raise ValueError(f'{label}: {col} outside public range')
        d[col] = x
    if 'time' in d:
        if ((d.time < 0) | (d.time > HORIZON)).any():
            raise ValueError(f'{label}: time outside [0,10]')
        if ((d.onset == 1) & (d.death == 1)).any():
            raise ValueError(f'{label}: onset/death both 1')
    return d


def read_table(path, kind):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        header = next(csv.reader(f), [])
    if len(header) != len(set(header)):
        raise ValueError(f'{path.name}: duplicate column name')
    required = (['challenge_row_id'] + QI if kind == 'challenge'
                else ['record_id'] + COLS if kind == 'B' else COLS)
    allowed = [set(required)]
    if kind == 'A':
        allowed.append(set(COLS + ['record_id']))
    if set(header) not in allowed:
        raise ValueError(f'{path.name}: wrong columns; truth/extra columns are forbidden; expected {required}')
    d = pd.read_csv(path, encoding='utf-8-sig', float_precision='round_trip',
                    dtype={'record_id': str, 'challenge_row_id': str, 'prefecture': str})
    for col in ['record_id', 'challenge_row_id']:
        if col in d:
            d[col] = integer_series(d[col], f'{path.name}/{col}')
    d = validate_attributes(d, path.name)
    if kind == 'A':
        d = d[COLS]
    if kind in ['B', 'challenge']:
        key = 'record_id' if kind == 'B' else 'challenge_row_id'
        return d[required].sort_values(key, kind='stable').reset_index(drop=True)
    return d[COLS].sort_values(COLS, kind='stable').reset_index(drop=True)


def id_value(value, label):
    if isinstance(value, bool) or not re.fullmatch(r'[0-9]+', str(value)):
        raise ValueError(f'{label}: nonnegative integer required')
    result = int(value)
    if result > np.iinfo(np.int64).max:
        raise ValueError(f'{label}: exceeds int64')
    return result


def load_targets(path):
    items = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(items, list) or not items:
        raise ValueError('targets.json must be a nonempty JSON array')
    targets = []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {'target_k', 'c_path', 'aia_challenge_path'}:
            raise ValueError('Each target needs exactly target_k, c_path, aia_challenge_path; no labels/results')
        k = id_value(item['target_k'], 'target_k')
        if k in seen:
            raise ValueError(f'duplicate target_k={k}; compare different C versions in separate runs')
        seen.add(k)
        paths = []
        for key in ['c_path', 'aia_challenge_path']:
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f'target {k}: invalid {key}')
            p = Path(item[key])
            p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
            if not p.is_file():
                raise ValueError(f'target {k}: missing {key}: {p}')
            paths.append(p)
        targets.append((k, *paths))
    return sorted(targets, key=lambda x: x[0])


def check_scope(targets, scope, columns_path, self_id):
    if scope == 'local':
        if columns_path is not None or self_id is not None:
            raise ValueError('Use --scope official with --mia-columns and --self-team-id')
        return
    if columns_path is None or self_id is None:
        raise ValueError('official scope requires --mia-columns and --self-team-id')
    meta = json.loads(columns_path.read_text(encoding='utf-8-sig'))
    if not isinstance(meta, dict) or not isinstance(meta.get('target_columns'), list):
        raise ValueError('mia_columns.json: target_columns list required')
    ids = [id_value(v, 'target_columns') for v in meta['target_columns']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty/duplicate official targets')
    allowed = set(ids)
    own = id_value(self_id, 'self-team-id')
    got = {t[0] for t in targets}
    if got - allowed or (allowed - {own}) - got:
        raise ValueError('Targets must cover all official opponents, with no unknown targets')


class Features:
    """Robust numerical scales from A only, categories from A/C only, never query labels."""
    def __init__(self, a, c):
        self.center = {}
        self.scale = {}
        for col in CONT:
            v = self.transform_column(a, col)
            self.center[col] = float(np.median(v))
            scale = float(np.quantile(v, .75) - np.quantile(v, .25))
            self.scale[col] = scale if scale > EPS else max(float(v.std()), 1.0)
        self.categories = {col: sorted(set(a[col].astype(str)) | set(c[col].astype(str))) for col in CAT}

    @staticmethod
    def transform_column(d, col):
        v = d[col].to_numpy(float)
        return np.log1p(v) if col in ['TG', 'ALT'] else v

    def numerical(self, d, columns):
        return np.column_stack([(self.transform_column(d, col) - self.center[col]) / self.scale[col]
                                for col in columns])

    def matrix(self, d, aia=False):
        pieces = [self.numerical(d, QI_NUM if aia else CONT)]
        cats = ['sex', 'prefecture'] if aia else CAT
        if not aia:
            pieces.append(d[['time', 'onset', 'death']].to_numpy(float) / [HORIZON, 1., 1.])
        for col in cats:
            values = d[col].astype(str).to_numpy()
            pieces.append(np.column_stack([values == cat for cat in self.categories[col]]).astype(float))
        return np.concatenate(pieces, axis=1)


def neighbor_distance(query, reference, feature, kth):
    """Log/IQR mixed L1 distance over 7 continuous + 3 categorical attributes."""
    q = feature.numerical(query, CONT)
    r = feature.numerical(reference, CONT)
    kth = min(kth, len(reference))
    out = np.empty(len(query))
    for start in range(0, len(query), BATCH):
        stop = min(start + BATCH, len(query))
        distance = cdist(q[start:stop], r, metric='cityblock')
        for col in CAT:
            distance += query[col].to_numpy()[start:stop, None] != reference[col].to_numpy()[None, :]
        out[start:stop] = np.partition(distance, kth-1, axis=1)[:, kth-1] / (len(CONT) + len(CAT))
    return out


def ranks(values):
    # Equal inputs keep equal scores; identifiers never break statistical ties.
    return (rankdata(values, method='average') - .5) / len(values)


def mia_scores(b, a, c, feature, trees=N_TREES):
    x = np.vstack([feature.matrix(c), feature.matrix(a)])
    y = np.r_[np.ones(len(c), dtype=int), np.zeros(len(a), dtype=int)]
    model = ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=MIN_LEAF_MIA,
        max_features=1.0, class_weight='balanced', random_state=SEED, n_jobs=1)
    model.fit(x, y)
    p = model.predict_proba(feature.matrix(b))[:, 1]
    dc = neighbor_distance(b, c, feature, 1)
    da = neighbor_distance(b, a, feature, BG_NEIGHBORS)
    contrast = np.log(da + EPS) - np.log(dc + EPS)
    # A proximity contrast, NOT a consistently normalized kNN density estimate.
    score = DISTANCE_WEIGHT * ranks(contrast) + (1-DISTANCE_WEIGHT) * ranks(p)
    return score


def best_time_window(sorted_times, weights):
    """Maximize empirical mass within +/-0.5 year; prefer 10 then earlier center on ties."""
    prefix = np.r_[0., np.cumsum(weights)]
    right = np.searchsorted(sorted_times, sorted_times + 2*TAU_TIME, side='right')
    masses = prefix[right] - prefix[np.arange(len(sorted_times))]
    centers = (sorted_times + sorted_times[right-1]) / 2
    ten = float(weights[np.abs(sorted_times-HORIZON) <= TAU_TIME].sum())
    maximum = float(masses.max())
    if ten >= maximum - EPS:
        return HORIZON
    # Numerical tolerance for summation only, not an expanded hit tolerance.
    idx = np.flatnonzero(masses >= maximum - EPS)[0]
    return float(np.clip(centers[idx], 0., HORIZON))


def aia_times(challenge, a, c, feature, trees=N_TREES):
    # B is deliberately absent from this signature. A is used only by feature scaling.
    x = feature.matrix(c, aia=True)
    q = feature.matrix(challenge, aia=True)
    model = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=MIN_LEAF_AIA,
        max_features=1.0, random_state=SEED+1, n_jobs=1)
    model.fit(x, c.time.to_numpy(float))
    train_leaves, query_leaves = model.apply(x), model.apply(q)
    members = [{leaf: np.flatnonzero(train_leaves[:, t] == leaf)
                for leaf in np.unique(train_leaves[:, t])} for t in range(trees)]
    order = np.argsort(c.time.to_numpy(float), kind='stable')
    times = c.time.to_numpy(float)[order]
    result = np.empty(len(challenge))
    for i in range(len(challenge)):
        weights = np.zeros(len(c))
        for t in range(trees):
            idx = members[t][query_leaves[i, t]]
            weights[idx] += 1. / (trees * len(idx))
        result[i] = best_time_window(times, weights[order])
    return result


def infer(b, a, loaded_targets, trees=N_TREES):
    if len(a) < BG_NEIGHBORS:
        raise ValueError(f'A_bg needs at least {BG_NEIGHBORS} rows')
    mia = pd.DataFrame({'record_id': b.record_id.to_numpy(np.int64)})
    answers = []
    for k, c, challenge in loaded_targets:
        if len(c) < 2:
            raise ValueError(f'C for target {k} needs at least 2 rows')
        feature = Features(a, c)
        score = mia_scores(b, a, c, feature, trees)
        times = aia_times(challenge, a, c, feature, trees)
        if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise ValueError('MIA output outside [0,1]')
        if not np.isfinite(times).all() or ((times < 0) | (times > HORIZON)).any():
            raise ValueError('AIA output outside [0,10]')
        mia[str(k)] = score
        answers.append(pd.DataFrame({'target_k': k, 'challenge_row_id': challenge.challenge_row_id.to_numpy(np.int64),
                                     'time_hat': times}))
    aia = pd.concat(answers, ignore_index=True).sort_values(['target_k', 'challenge_row_id'], kind='stable')
    if aia.duplicated(['target_k', 'challenge_row_id']).any():
        raise ValueError('Duplicate AIA output key')
    return mia, aia.reset_index(drop=True)


def csv_bytes(d):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(d.columns)
    ids = {'record_id', 'target_k', 'challenge_row_id'}
    for row in d.itertuples(index=False, name=None):
        writer.writerow([str(int(v)) if col in ids else format(float(v), '.17g') for col, v in zip(d.columns, row)])
    return stream.getvalue().encode('utf-8')


def save_outputs(mia, aia, input_paths, overwrite=False):
    script = Path(__file__).resolve()
    blobs = [(script.with_name(prefix + script.stem + '.csv'), csv_bytes(d))
             for prefix, d in [('F_mia_', mia), ('F_aia_', aia)]]
    for dest, data in blobs:
        if dest.resolve() in input_paths:
            raise ValueError('Output path collides with an input')
        if dest.exists() and dest.read_bytes() != data and not overwrite:
            raise ValueError(f'{dest.name} already exists with different content. Move it or explicitly use --overwrite.')
    for dest, data in blobs:
        if dest.exists() and dest.read_bytes() == data:
            continue
        with tempfile.NamedTemporaryFile(dir=script.parent, suffix='.tmp', delete=False) as f:
            temp = Path(f.name)
            f.write(data)
        try:
            os.replace(temp, dest)
        finally:
            temp.unlink(missing_ok=True)
    return {p.name: hashlib.sha256(data).hexdigest() for p, data in blobs}


def self_test():
    """Artificial functional tests only. No user's competition files are accessed."""
    import shutil
    import subprocess
    rng = np.random.default_rng(1337)
    def fake(n):
        d = pd.DataFrame({col: rng.uniform(lo+(hi-lo)*.2, lo+(hi-lo)*.7, n)
                          for col, (lo, hi) in RANGES.items()})
        d['age'] = rng.integers(40, 75, n)
        for col in ['sex', 'smoking']:
            d[col] = rng.integers(0, 2, n)
        d['prefecture'] = rng.choice(['架空A', '架空B'], n)
        d['onset'] = rng.integers(0, 2, n)
        d['death'] = 0
        d['time'] = np.where(d.onset == 0, 10., rng.uniform(.1, 9.8, n))
        return d[COLS]
    b = fake(40); b.insert(0, 'record_id', np.arange(100, 140))
    a, c = fake(80), fake(60)
    q = fake(16)[QI]; q.insert(0, 'challenge_row_id', np.arange(1, 17))
    first = infer(b, a, [(5, c, q)], trees=12)
    repeat = infer(b, a, [(5, c, q)], trees=12)
    assert all(csv_bytes(x) == csv_bytes(y) for x, y in zip(first, repeat))
    assert best_time_window(np.array([2., 10.]), np.array([.6, .4])) == 2.
    assert best_time_window(np.array([2., 10.]), np.array([.4, .6])) == 10.
    assert np.array_equal(ranks([1, 1, 1]), [.5, .5, .5])
    # AIA cannot read B answers. Changing B's time must not change AIA output.
    b2 = b.copy(); b2['time'] = 3.
    changed_b = infer(b2, a, [(5, c, q)], trees=12)
    assert csv_bytes(first[1]) == csv_bytes(changed_b[1])
    switched = infer(b, a, [(19, fake(55), q.iloc[:7])], trees=12)
    assert list(switched[0].columns) == ['record_id', '19'] and len(switched[1]) == 7
    with tempfile.TemporaryDirectory(prefix='seo_attack_test_') as temp_dir:
        folder = Path(temp_dir)
        for name, d in [('b.csv', b), ('a.csv', a), ('c.csv', c), ('q.csv', q)]:
            d.to_csv(folder/name, index=False, encoding='utf-8', lineterminator='\n', float_format='%.17g')
        manifest = folder/'targets.json'
        manifest.write_text(json.dumps([{'target_k': 5, 'c_path': 'c.csv', 'aia_challenge_path': 'q.csv'}]), encoding='utf-8')
        targets = load_targets(manifest)
        assert targets[0][1] == (folder/'c.csv').resolve()
        for name, kind in [('b.csv', 'B'), ('a.csv', 'A'), ('c.csv', 'C'), ('q.csv', 'challenge')]:
            read_table(folder/name, kind)
        # End-to-end command-line test with the DEFAULT 192-tree setting.
        # Keep both generated CSVs confined to this disposable directory.
        script = folder / Path(__file__).name
        shutil.copy2(Path(__file__).resolve(), script)
        command = [sys.executable, str(script), '--b', str(folder/'b.csv'),
                   '--a-bg', str(folder/'a.csv'), '--targets', str(manifest)]
        subprocess.run(command, check=True, capture_output=True)
        out_paths = [folder/(prefix+script.stem+'.csv') for prefix in ['F_mia_', 'F_aia_']]
        original_bytes = [p.read_bytes() for p in out_paths]
        subprocess.run(command, check=True, capture_output=True)
        assert original_bytes == [p.read_bytes() for p in out_paths]
        fm, fa = [pd.read_csv(p) for p in out_paths]
        assert list(fm.columns) == ['record_id', '5'] and set(fm.record_id) == set(b.record_id)
        assert list(fa.columns) == ['target_k', 'challenge_row_id', 'time_hat']
        assert set(fa.challenge_row_id) == set(q.challenge_row_id) and set(fa.target_k) == {5}
        assert len(fa) == len(q) and not fa.duplicated(['target_k', 'challenge_row_id']).any()
        meta = folder/'mia_columns.json'
        meta.write_text(json.dumps({'target_columns': ['5', '19']}), encoding='utf-8')
        check_scope(targets, 'official', meta, 19)
        try:
            check_scope(targets, 'official', meta, 7)
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete official coverage was accepted')
        duplicate = [{'target_k': 5, 'c_path': 'c.csv', 'aia_challenge_path': 'q.csv'}] * 2
        manifest.write_text(json.dumps(duplicate), encoding='utf-8')
        try:
            load_targets(manifest)
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate target ID was accepted')
        d = q.copy(); d['time'] = 10
        d.to_csv(folder/'bad.csv', index=False)
        try:
            read_table(folder/'bad.csv', 'challenge')
        except ValueError:
            pass
        else:
            raise AssertionError('Challenge truth leakage was accepted')
    print(json.dumps({'self_test': 'passed', 'real_data_used': False,
                      'default_cli_csv_byte_reproducible': True,
                      'coverage_and_truth_column_rejection': True,
                      'performance_evaluated': False, 'persistent_outputs_created': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--b', type=Path)
    parser.add_argument('--a-bg', type=Path)
    parser.add_argument('--targets', type=Path)
    parser.add_argument('--scope', choices=['local', 'official'], default='local')
    parser.add_argument('--mia-columns', type=Path)
    parser.add_argument('--self-team-id', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        if args.self_test:
            self_test()
            return 0
        if any(x is None for x in [args.b, args.a_bg, args.targets]):
            raise ValueError('--b, --a-bg and --targets are required unless --self-test is used')
        targets = load_targets(args.targets.resolve())
        check_scope(targets, args.scope, args.mia_columns, args.self_team_id)
        b, a = read_table(args.b, 'B'), read_table(args.a_bg, 'A')
        loaded = [(k, read_table(cp, 'C'), read_table(qp, 'challenge')) for k, cp, qp in targets]
        mia, aia = infer(b, a, loaded)
        inputs = {p.resolve() for p in [args.b, args.a_bg, args.targets]}
        inputs.update(p.resolve() for _, cp, qp in targets for p in [cp, qp])
        if args.mia_columns is not None:
            inputs.add(args.mia_columns.resolve())
        hashes = save_outputs(mia, aia, inputs, args.overwrite)
        print(json.dumps({'name': 'seo', 'version': '1.0', 'scope': args.scope,
            'MIA_rows': len(mia), 'targets': [k for k, _, _ in targets], 'AIA_rows': len(aia),
            'MIA_score_kind': 'fixed rank ensemble, not calibrated membership probability',
            'AIA_time_unit': 'years', 'sha256': hashes, 'seconds': round(time.perf_counter()-start, 3),
            'official_scores': 'not calculated', 'uploaded': False}, ensure_ascii=False))
        if args.scope == 'local':
            print('LOCAL ONLY: self-target tests do not contribute to official attack power.', file=sys.stderr)
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
