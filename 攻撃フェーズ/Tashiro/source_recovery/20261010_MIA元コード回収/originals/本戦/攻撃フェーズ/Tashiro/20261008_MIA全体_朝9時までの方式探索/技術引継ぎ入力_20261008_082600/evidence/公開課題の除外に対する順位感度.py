"""公開非会員の所有者を除外する感度診断。所属ラベル・採点は使わない。"""
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import 共通 as u

hier = u.module('frozen_null_sensitivity_hier', u.HERE / '非会員較正の広い集団を保留検証.py')
METHODS = [hier.CONFIGS[0], hier.CONFIGS[1], hier.CONFIGS[5]]


def head(scores):
    return set(np.flatnonzero(rankdata(-scores, method='min') <= 12))


def phase_run(phase, root_out):
    started = time.perf_counter()
    data = u.setup(phase)
    path = u.HERE / '非会員課題の丸ごと保留結果' / phase / '候補と公開課題の生スコア.npz'
    u.base.INPUT_HASHES[str(path)] = u.sha(path)
    cache = np.load(path)
    assert list(cache['teams']) == data['teams']
    owner = cache['query_owner']
    assert len(owner) == len(data['query'])
    raw = {t: cache['raw'][:, j] for j, t in enumerate(data['teams'])}
    scenarios = [('full', ())] + [(f'without_owner_{t}', (t,)) for t in sorted(data['qs'])]
    if phase == 'prelim':
        scenarios.append(('without_owners_1_2', (1, 2)))
    matrices, detail = {}, []
    for label, removed in scenarios:
        changed = dict(data)
        changed['controls'] = {t: ii[~np.isin(owner[ii], removed)] for t, ii in data['controls'].items()}
        for t, ii in changed['controls'].items():
            assert not np.isin(owner[ii], (*removed, t, -1)).any()
        banks = {c['name']: {t: hier.predict(raw[t], changed, t, c) for t in data['teams']} for c in METHODS}
        matrices[label] = u.matrices(banks, data['teams'], cross=(.1,))
        u.log(stage='public_owner_sensitivity', phase=phase, scenario=label)
    old_path = u.HERE / '広い集団への非会員較正結果' / phase / '内部順位だけ.npz'
    u.base.INPUT_HASHES[str(old_path)] = u.sha(old_path)
    old = np.load(old_path)
    error = 0.
    for method, full in matrices['full'].items():
        index = list(old['names']).index(method)
        delta = float(np.max(np.abs(full - old['scores'][index])))
        error = max(error, delta)
        assert delta < 1e-14, (method, delta)
    summaries = []
    for method, full in matrices['full'].items():
        for j, team in enumerate(data['teams']):
            reference = head(full[:, j])
            reference_ranks = rankdata(-full[:, j], method='average')
            sets, ranks, changes = [], [], []
            for label, removed in scenarios[1:]:
                values = matrices[label][method][:, j]
                selected = head(values)
                r = rankdata(-values, method='average')
                sets.append(selected)
                ranks.append(r)
                changes.append(len(selected - reference))
                detail.append(dict(phase=phase, team=team, method=method, scenario=label,
                                   removed_owners=';'.join(map(str, removed)), full_head_rows=len(reference),
                                   perturbed_head_rows=len(selected), added_rows=len(selected - reference),
                                   removed_rows=len(reference - selected),
                                   rank_correlation=float(np.corrcoef(reference_ranks, r)[0, 1]),
                                   mean_abs_rank_change=float(np.mean(np.abs(r-reference_ranks))),
                                   public_owner_deletion_stress=True, membership_accuracy_unknown=True))
            rank_matrix = np.stack(ranks)
            head_frequency = np.array([sum(i in s for s in sets) / len(sets) for i in sorted(reference)])
            common = set.intersection(*sets)
            spread = rank_matrix.max(axis=0) - rank_matrix.min(axis=0)
            summaries.append(dict(phase=phase, team=team, method=method, scenarios=len(sets),
                                  full_head_rows=len(reference), head_intersection_all_stresses=len(common),
                                  full_head_mean_inclusion=float(head_frequency.mean()),
                                  full_head_min_inclusion=float(head_frequency.min()),
                                  mean_added_head_rows=float(np.mean(changes)), max_added_head_rows=int(max(changes)),
                                  mean_rank_range=float(spread.mean()), p95_rank_range=float(np.quantile(spread, .95)),
                                  sensitivity_is_not_correctness_probability=True))
    out = root_out / phase
    out.mkdir()
    pd.DataFrame(detail).to_csv(out / '課題除外ごとの順位差.csv', index=False, float_format='%.17g')
    pd.DataFrame(summaries).to_csv(out / '対象ごとの課題除外感度.csv', index=False, float_format='%.17g')
    # 個人IDを含めない内部配列。外部掲載用は上記の対象別集計のみ。
    labels, scores = [], []
    for scenario, table in matrices.items():
        for method in sorted(table):
            labels.append(f'{scenario}::{method}')
            scores.append(table[method])
    np.savez_compressed(out / '内部順位だけ.npz', names=np.array(labels), teams=np.array(data['teams']), scores=np.stack(scores))
    assert not any('pool_ids' in p or '正解データ' in p or 'result.txt' in p for p in u.base.INPUT_HASHES)
    for p, h in u.base.INPUT_HASHES.items():
        assert u.sha(p) == h, p
    support = [u.HERE / '共通.py', u.HERE / '非会員較正の広い集団を保留検証.py',
               u.HERE / '公開非会員だけで較正を選ぶ.py', u.OLD / '分析.py', u.CAL / '分析.py']
    record = dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                  seconds=time.perf_counter()-started, code_path=str(Path(__file__).resolve()),
                  code_sha256=u.sha(__file__), input_sha256=dict(u.base.INPUT_HASHES),
                  support_code_sha256={str(p): u.sha(p) for p in support},
                  output_sha256={p.name: u.sha(p) for p in out.iterdir()},
                  baseline_full_rank_max_error=error, membership_labels_used=False,
                  python=sys.executable, python_version=platform.python_version(),
                  meaning='Grouped public-control deletion sensitivity, not confidence intervals or MIA accuracy.',
                  head_rule='rank(-score, method=min)<=12; ties may include more than 12.',
                  no_selection='Three frozen calibration methods are reported, with no choice based on stability.',
                  rare_and_private='Fixed normal-875 mask; rare-MIA and AIA are not evaluated.')
    (out / '実行記録.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    u.log(stage='public_owner_sensitivity_complete', phase=phase, baseline_error=error, seconds=record['seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=u.HERE / '公開課題の除外に対する順位感度結果')
    args = parser.parse_args()
    assert not args.output_dir.exists()
    args.output_dir.mkdir()
    plan = dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(), code_sha256=u.sha(__file__),
                methods=METHODS, perturbation='Delete every public Q owner once; additionally delete owners1,2 together in prelim.',
                cross=.1, member_labels_used=False, selection=False,
                interpretation='Source-composition stress, not independent held-out membership validation.')
    (args.output_dir / '実験方針.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    for phase in ('prelim', 'main'):
        phase_run(phase, args.output_dir)


if __name__ == '__main__':
    main()
