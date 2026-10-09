#!/usr/bin/env python3
"""Audit decision changes from local density-ratio features (published prelim only).

Uses only public model predictions from the completed experiment as candidate scores.
The official aia_truth is used exclusively to *score* prediction changes. This
script never trains or reselects a model, optimizes a threshold, or submits an
attack. It reports aggregate counts by cohort rather than per-person truths.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def select_top75(frame: pd.DataFrame, key: str) -> pd.Series:
    selected = pd.Series(False, index=frame.index)
    for _, part in frame.groupby('sex', sort=False):
        ranked = part.sort_values([key, 'row_id'], ascending=[False, True], kind='stable')
        selected.loc[ranked.index[:math.ceil(0.75 * len(ranked))]] = True
    return selected


def audit(root: Path, results: Path) -> pd.DataFrame:
    predictions = pd.read_csv(results / 'augment_public_predictions.csv')
    subset = predictions[predictions.model.isin(['c_bg', 'c_bg_plus_local'])]
    wide = subset.pivot(index=['team', 'row_id'], columns='model', values='score').reset_index()
    public_q = pd.read_csv(results / 'prior_public_scores.csv')[['team', 'row_id', 'sex']]
    wide = wide.merge(public_q, on=['team', 'row_id'], validate='one_to_one')
    teams = sorted(int(x) for x in wide.team.unique())
    assert len(teams) == 21 and len(wide) == 326 * 21

    folder = root / '公式データ' / '予備戦' / '結果' / '正解データ'
    truth = []
    for team in teams:
        frame = pd.read_csv(folder / f'aia_truth_{team}.csv')
        assert len(frame) == 326
        frame['team'] = team
        truth.append(frame)
    truth = pd.concat(truth, ignore_index=True)
    joined = wide.merge(truth, left_on=['team', 'row_id'],
                        right_on=['team', 'challenge_row_id'], validate='one_to_one')
    assert len(joined) == 326 * 21

    baseline = pd.read_csv(results / 'augment_target_metrics.csv')
    baseline = baseline[(baseline.policy == 'base_0') &
                        baseline.model.isin(['c_bg', 'c_bg_plus_local'])]

    rows = []
    for team, part in joined.groupby('team', sort=True):
        b = select_top75(part, 'c_bg').to_numpy(bool)
        a = select_top75(part, 'c_bg_plus_local').to_numpy(bool)
        time = part.time.to_numpy(float)
        hb = np.abs(np.where(b, 9.5, 0) - time) <= .5
        ha = np.abs(np.where(a, 9.5, 0) - time) <= .5
        member = part.label.to_numpy() == 'member'
        control = ~member
        assert member.sum() == control.sum() == 163
        delta = a != b
        row = dict(
            team=int(team), changed=int(delta.sum()),
            changed_member=int((delta & member).sum()),
            changed_control=int((delta & control).sum()),
            member_hits_added=int(((~hb) & ha & member).sum()),
            member_hits_lost=int((hb & (~ha) & member).sum()),
            control_hits_added=int(((~hb) & ha & control).sum()),
            control_hits_lost=int((hb & (~ha) & control).sum()),
            net_member_change=int(ha[member].sum() - hb[member].sum()),
            net_control_change=int(ha[control].sum() - hb[control].sum()),
        )
        # Independent check against the already published official-score table.
        for name, hits in [('c_bg', hb), ('c_bg_plus_local', ha)]:
            reference = baseline[(baseline.team == team) & (baseline.model == name)]
            assert len(reference) == 1
            assert int(reference.iloc[0].hit_m) == int(hits[member].sum())
            assert int(reference.iloc[0].hit_c) == int(hits[control].sum())
        rows.append(row)

    output = pd.DataFrame(rows)
    assert int(output.net_member_change.sum()) == int((output.member_hits_added - output.member_hits_lost).sum())
    assert int(output.net_control_change.sum()) == int((output.control_hits_added - output.control_hits_lost).sum())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Checkout containing 公式データ/予備戦/...')
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    parser.add_argument('--out', type=Path, default=HERE / 'results' / 'switch_diagnostic.csv')
    args = parser.parse_args()
    result = audit(args.root, args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    for name, exclude in [('20target', {5}), ('processed18', {1, 2, 5})]:
        p = result[~result.team.isin(exclude)]
        total = p.drop(columns=['team']).sum()
        print(name, 'cohorts', len(p), 'changed', int(total.changed),
              'member correct delta', int(total.net_member_change),
              'control correct delta', int(total.net_control_change),
              'signed hit delta', int(total.net_member_change - total.net_control_change))
    print('Saved', args.out)


if __name__ == '__main__':
    main()
