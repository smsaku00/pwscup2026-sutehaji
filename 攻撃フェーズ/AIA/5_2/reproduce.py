#!/usr/bin/env python3
"""Rebuild round 5, then split public 2--4 person groups into 9.5/11 answers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE / 'base_model'
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location('original_round5', BASE / 'reproduce.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
CONFIG = json.loads((HERE / 'selected.json').read_text())


def split_groups(diagnostics):
    """No truth labels: rank LR10 within full-challenge public attribute groups."""
    d = diagnostics.sort_values(['target_k', 'challenge_row_id']).reset_index(drop=True).copy()
    if d.duplicated(['target_k', 'challenge_row_id']).any():
        raise ValueError('Duplicate challenge keys')
    if not np.isfinite(d['membership_LR10'].to_numpy(float)).all():
        raise ValueError('Nonfinite membership scores')
    d['age_band'] = (d.age // 5).astype(int)
    d['new_time_hat'] = d.time_hat.copy()
    d['small_group'] = False
    assignments, counts = [], []
    for team, tq in d.groupby('target_k', sort=True):
        if int(team) == 28:
            raise ValueError('Self team must not be present')
        group_counts = {2: 0, 3: 0, 4: 0}
        for group_id, (_, group) in enumerate(tq.groupby(['prefecture', 'sex', 'age_band'], sort=True)):
            n = len(group)
            if n not in (2, 3, 4):
                continue
            group_counts[n] += 1
            ordered = group.sort_values(['membership_LR10', 'challenge_row_id'], ascending=[False, True])
            n_answer = math.ceil(n / 2)
            new = np.array([9.5] * n_answer + [11.] * (n - n_answer))
            d.loc[ordered.index, 'new_time_hat'] = new
            d.loc[ordered.index, 'small_group'] = True
            for rank, (ix, row) in enumerate(ordered.iterrows(), 1):
                assignments.append({'target_k': int(team), 'group_id': group_id, 'group_size': n,
                    'challenge_row_id': int(row.challenge_row_id), 'rank_in_group': rank,
                    'old_time_hat': float(row.time_hat), 'time_hat': float(new[rank - 1]),
                    'changed': bool(row.time_hat != new[rank - 1])})
            assert int((d.loc[group.index, 'new_time_hat'] == 9.5).sum()) == n_answer
        counts.append({'target_k': int(team), 'groups_2': group_counts[2],
            'groups_3': group_counts[3], 'groups_4': group_counts[4],
            'affected_rows': int(d.loc[tq.index, 'small_group'].sum()),
            'changed_rows': int((d.loc[tq.index, 'time_hat'] != d.loc[tq.index, 'new_time_hat']).sum()),
            'from_9_5_to_11': int(((d.loc[tq.index, 'time_hat'] == 9.5) & (d.loc[tq.index, 'new_time_hat'] == 11)).sum()),
            'from_11_to_9_5': int(((d.loc[tq.index, 'time_hat'] == 11) & (d.loc[tq.index, 'new_time_hat'] == 9.5)).sum())})
    assert np.array_equal(d.loc[~d.small_group, 'time_hat'], d.loc[~d.small_group, 'new_time_hat'])
    return d, pd.DataFrame(assignments), pd.DataFrame(counts)


def shared_zip(path, aia, mia):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, raw in [('F_mia.csv', mia), ('F_aia.csv', aia)]:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            z.writestr(info, raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--staged-work', type=Path, help='Already staged inputs from original round5 reproduce.py')
    for name in ['prelim-common', 'prelim-attack', 'prelim-truth', 'main-attack', 'main-team']:
        ap.add_argument('--' + name, type=Path, help='ZIP or directory, required unless --staged-work is given')
    ap.add_argument('--work-dir', type=Path, default=Path('round5_2_work'))
    ap.add_argument('--base-submission', type=Path, required=True, help='MIA and local token copy source; never printed')
    ap.add_argument('--out', type=Path, default=Path('round5_2_output'))
    args = ap.parse_args()
    if args.staged_work:
        root = args.staged_work.resolve()
    else:
        missing = [n for n in ['prelim_common', 'prelim_attack', 'prelim_truth', 'main_attack', 'main_team'] if getattr(args, n) is None]
        if missing:
            ap.error('Missing public inputs: ' + ', '.join(missing))
        root = original.stage(args)
    D, base_pred, _, diagnostics = original.generate_selected(root)
    base_raw = original.csv_bytes(D, base_pred['lr10_75_rare_soft'])
    base_sha = hashlib.sha256(base_raw).hexdigest()
    if base_sha != CONFIG['original_F_aia_sha256']:
        raise ValueError('Original round5 does not reproduce exactly; check inputs and pinned dependencies')
    d, assignments, counts = split_groups(diagnostics)
    pred = {int(k): q.new_time_hat.to_numpy(float) for k, q in d.groupby('target_k', sort=True)}
    raw = original.csv_bytes(D, pred)
    mia = original.read_named(args.base_submission, 'F_mia.csv')
    token = original.read_named(args.base_submission, 'token.txt')
    args.out.mkdir(parents=True, exist_ok=True)
    full = args.out / 'zhang_T28_test05_2_small_groups.zip'
    original.make_zip(full, raw, mia, token)
    validation = original.validate(full, root / 'tmp/main_attack')
    with zipfile.ZipFile(full) as z:
        assert z.read('F_mia.csv') == mia and z.read('token.txt') == token
    shared = args.out / 'zhang_T28_test05_2_small_groups_no_token.zip'
    shared_zip(shared, raw, mia)
    with zipfile.ZipFile(shared) as z:
        assert set(z.namelist()) == {'F_mia.csv', 'F_aia.csv'}
        assert z.read('F_aia.csv') == raw and z.read('F_mia.csv') == mia
    (args.out / 'F_aia.csv').write_bytes(raw)
    assignments.to_csv(args.out / 'group_assignments.csv', index=False)
    counts.to_csv(args.out / 'per_target_changes.csv', index=False)
    d.to_csv(args.out / 'row_diagnostics_local_only.csv', index=False)
    import platform, scipy, sklearn
    audit = {'method': CONFIG['method'], 'source_repository_commit': CONFIG['source_repository_commit'],
        'original_F_aia_sha256': base_sha, 'F_aia_sha256': hashlib.sha256(raw).hexdigest(),
        'F_mia_sha256': hashlib.sha256(mia).hexdigest(), 'validation': validation,
        'grouping': 'within each full challenge: prefecture, sex, floor(age/5); size 2..4',
        'ordering': 'membership_LR10 descending, challenge_row_id ascending',
        'three_person_split': '2 answer 9.5, 1 answers 11',
        'group_counts': {str(n): int(counts[f'groups_{n}'].sum()) for n in [2, 3, 4]},
        'affected_rows': int(counts.affected_rows.sum()), 'changed_rows': int(counts.changed_rows.sum()),
        'from_9_5_to_11': int(counts.from_9_5_to_11.sum()), 'from_11_to_9_5': int(counts.from_11_to_9_5.sum()),
        'MIA_copied_byte_for_byte': True, 'token_copied_byte_for_byte_local_only': True,
        'main_B_used_for_prediction': False, 'main_truth_used': False,
        'official_score_obtained': False,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'sklearn': sklearn.__version__}}
    (args.out / 'generation_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
