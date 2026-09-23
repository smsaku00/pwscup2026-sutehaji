"""Prototype: Zhang 04 copula and background times, seo02-style Wald calibration.

Inputs: final B.csv, A_bg.csv and public schema.json. No B_self or hidden truth.
The only change to Zhang 04 is selection of the independent synthetic outcome
iteration by a combined Cox coefficient / Wald statistic discrepancy.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from shared_methods import covariates, public_outliers, fit, xf

SEED = 28201
METHOD = 'prototype_full_wald'


def outcomes_background_wald(c, b, a, schema, group, rng, steps=80, wald_weight=.3):
    """Draw event times from A_bg; choose an iteration using Wald as well as beta."""
    target = fit(b)
    bx = xf(b)
    sd = bx.std(ddof=0).to_numpy()
    mean = bx.mean().to_numpy()
    if np.any(sd <= 0):
        raise ValueError('Constant Cox predictor')
    xx = (xf(c).to_numpy() - mean) / sd
    target_beta = target.params_.to_numpy() * sd
    target_se = target.standard_errors_.to_numpy() * sd
    target_wald = target_beta / target_se
    theta = target_beta.copy()
    best, best_loss, best_detail = None, np.inf, None

    # Preserve Zhang 04's independent outcome mechanism and random stream.
    fixed_noise = np.log(-np.log(np.clip(rng.random(len(c)), 1e-10, 1)))
    death = np.zeros(len(c), int)
    death_score = xx[:, 0] * .5 + rng.gumbel(size=len(c))
    for g in np.unique(group):
        ids = np.flatnonzero(group == g)
        nd = int(b.iloc[ids].death.sum())
        if nd:
            death[ids[np.argsort(death_score[ids])[-nd:]]] = 1

    pools = {}
    for g in np.unique(group):
        ids = np.flatnonzero(group == g)
        bg = a.loc[public_outliers(a, schema) == bool(g)]
        n_on = int(b.iloc[ids].onset.sum())
        source = bg.loc[bg.onset == 1, 'time'].to_numpy()
        if n_on and len(source) == 0:
            raise ValueError('A_bg has no event time in one stratum')
        pools[g] = np.sort(np.clip(rng.choice(source, n_on, replace=True) +
                                   rng.normal(0, .35, n_on), .003, 9.99))

    for it in range(steps):
        candidate = c.copy()
        candidate['onset'] = 0
        candidate['death'] = death
        candidate['time'] = 10.
        clock = fixed_noise - xx @ theta
        for g in np.unique(group):
            ids = np.flatnonzero(group == g)
            eligible = ids[death[ids] == 0]
            chosen = eligible[np.argsort(clock[eligible])[:len(pools[g])]]
            candidate.loc[chosen, 'onset'] = 1
            candidate.loc[chosen, 'time'] = pools[g]
            dids = ids[death[ids] == 1]
            src = a.loc[public_outliers(a, schema) == bool(g)]
            death_times = src.loc[src.death == 1, 'time'].to_numpy()
            if len(dids) and not len(death_times):
                raise ValueError('A_bg has no death time in one stratum')
            candidate.loc[dids, 'time'] = np.sort(rng.choice(death_times, len(dids), replace=True))
        try:
            fitted = fit(candidate)
        except Exception:
            break
        beta = fitted.params_.to_numpy() * sd
        se = fitted.standard_errors_.to_numpy() * sd
        beta_loss = float(np.mean(((target_beta - beta) / target_se) ** 2))
        wald_loss = float(np.mean((target_wald - beta / se) ** 2))
        # Wald is the seo02 contribution. Both losses use 9 dimensionless terms.
        loss = (1. - wald_weight) * beta_loss + wald_weight * wald_loss
        if np.isfinite(loss) and loss < best_loss:
            best, best_loss = candidate.copy(), loss
            best_detail = {'iteration': it, 'cox_beta_loss': beta_loss,
                           'cox_wald_loss': wald_loss, 'combined_loss': loss}
        theta += .6 * (target_beta - beta)
    if best is None:
        raise ValueError('Cox outcome calibration failed')
    return best, best_detail


def generate(b, a, schema, wald_weight=.3):
    cols = schema['submitted_c_columns']
    if set(b.columns) not in (set(cols), set(cols + ['record_id'])):
        raise ValueError('Use final B.csv; hidden labels are not accepted')
    if set(a.columns) not in (set(cols), set(cols + ['record_id'])):
        raise ValueError('Use final A_bg.csv')
    b = b[cols].reset_index(drop=True)
    a = a[cols].reset_index(drop=True)
    if b.isna().any().any() or a.isna().any().any() or len(b) < 50:
        raise ValueError('Missing values or small cohorts are unsupported')
    rng = np.random.default_rng(SEED)
    c, group = covariates(b, a, schema, rng, 0)
    c, cal = outcomes_background_wald(c, b, a, schema, group, rng,
                                      wald_weight=wald_weight)
    c = c[cols].iloc[rng.permutation(len(c))].reset_index(drop=True)
    for col in ('age', 'sex', 'smoking', 'onset', 'death'):
        c[col] = c[col].astype(int)
    return c, {'method': METHOD, 'seed': SEED, 'wald_weight': wald_weight,
               'outcome_calibration': cal,
               'B_public_outlier_rows': int(group.sum()),
               'C_public_outlier_rows': int(public_outliers(c, schema).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('b', 'a-bg', 'schema', 'out'):
        ap.add_argument('--' + arg, type=Path, required=True)
    ap.add_argument('--wald-weight', type=float, default=.3)
    args = ap.parse_args()
    b = pd.read_csv(args.b, float_precision='round_trip')
    a = pd.read_csv(args.a_bg, float_precision='round_trip')
    schema = json.loads(args.schema.read_text(encoding='utf-8'))
    if not 0 <= args.wald_weight <= .5:
        ap.error('--wald-weight must be between 0 and 0.5')
    c, meta = generate(b, a, schema, wald_weight=args.wald_weight)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = c.to_csv(index=False, float_format='%.17g', lineterminator='\n').encode()
    args.out.write_bytes(data)
    meta['C_sha256'] = hashlib.sha256(data).hexdigest()
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
