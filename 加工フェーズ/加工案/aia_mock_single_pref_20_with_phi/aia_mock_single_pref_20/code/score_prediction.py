"""Score an AIA prediction against one mock panel (overall R and adjusted R)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--panel",required=True,help="e.g. prelim/panel_01")
    ap.add_argument("--prediction",type=Path,required=True,
                    help="CSV: target_k,challenge_row_id,time_hat")
    args = ap.parse_args()
    root = args.dataset
    results = pd.read_csv(root/"phi_results.csv")
    r = results.loc[results.panel.eq(args.panel)]
    if len(r) != 1:
        raise ValueError("Unknown panel")
    expected_target = int(r.iloc[0].mock_target_k)
    truth = pd.read_csv(root/"private"/args.panel/"truth.csv")
    pred = pd.read_csv(args.prediction)
    if pred.columns.to_list() != ["target_k","challenge_row_id","time_hat"]:
        raise ValueError("Expected exactly: target_k,challenge_row_id,time_hat")
    if len(pred)!=326 or not pred.challenge_row_id.is_unique or not pred.target_k.eq(expected_target).all():
        raise ValueError("Incorrect target ID, duplicate rows, or incorrect number of rows")
    if not np.isfinite(pred.time_hat.to_numpy(float)).all():
        raise ValueError("Invalid time_hat")
    joined = truth.merge(pred,on="challenge_row_id",validate="one_to_one")
    if len(joined)!=326:
        raise ValueError("Unmatched challenge_row_id")
    success = (joined.time - joined.time_hat).abs() <= .5
    member = joined.label.eq("member")
    pm,pc = success[member].mean(),success[~member].mean()
    raw = float(np.clip((pm-pc)/max(1-pc,1e-9),0,1))
    phi = float(r.iloc[0].phi_mock)
    print(f"panel={args.panel} p_m={pm:.6f} p_c={pc:.6f}")
    print(f"R_all={raw:.6f} phi_mock={phi:.6f} clip(R_all-phi_mock)={max(0,raw-phi):.6f}")
    print("Mock overall only; actual P and control rare truth unavailable.")


if __name__ == "__main__":
    main()
