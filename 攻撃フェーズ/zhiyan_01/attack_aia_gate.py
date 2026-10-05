#!/usr/bin/env python3
"""PWS Cup 2026 本戦初回提出: Cが十分近い場合だけCのtimeを採用する。

依存: numpy, pandas, scipy。README.mdに実行例と判定条件を記載。
入力の本戦配布ZIPとチーム配布ZIPを変更しない。
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


CONT = ["age", "SBP", "TG", "HDL", "ALT"]
MIA_CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
BLOCK = ["sex", "prefecture"]


def table(archive: ZipFile, name: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(name)))


def nearest(reference: pd.DataFrame, query: pd.DataFrame, scale: np.ndarray):
    """公式AIAと同じブロック・距離・同距離time平均。距離も返す。"""
    if not len(reference):
        raise ValueError("参照表が空です")
    ref_x = reference[CONT].to_numpy(dtype=float) / scale
    q_x = query[CONT].to_numpy(dtype=float) / scale
    ref_t = reference["time"].to_numpy(dtype=float)
    by_both = reference.groupby(BLOCK).indices
    by_sex = reference.groupby("sex").indices
    all_idx = np.arange(len(reference))
    distances = np.empty(len(query), dtype=float)
    predicted = np.empty(len(query), dtype=float)
    for i, (sex, pref) in enumerate(query[BLOCK].itertuples(index=False, name=None)):
        idx = by_both.get((sex, pref))
        if idx is None or not len(idx):
            idx = by_sex.get(sex)
        if idx is None or not len(idx):
            idx = all_idx
        d = np.sqrt(np.mean((ref_x[idx] - q_x[i]) ** 2, axis=1))
        distances[i] = d.min()
        predicted[i] = float(np.mean(np.sort(ref_t[idx[d <= d.min()]])))
    return distances, predicted


def mia_reference(c: pd.DataFrame, bg: pd.DataFrame, b: pd.DataFrame, ref_density) -> np.ndarray:
    """配布キットの参照MIAと同じ7列のKDE密度比。"""
    mean = bg[MIA_CONT].mean().to_numpy(dtype=float)
    std = bg[MIA_CONT].std(ddof=0).to_numpy(dtype=float)
    std = np.where(std > 0, std, 1.0)
    syn = (c[MIA_CONT].to_numpy(dtype=float) - mean) / std
    targets = (b[MIA_CONT].to_numpy(dtype=float) - mean) / std
    try:
        syn_density = gaussian_kde(syn.T)(targets.T)
    except np.linalg.LinAlgError:
        return np.full(len(b), 0.5)
    ratio = syn_density / np.clip(ref_density, 1e-300, None)
    return ratio / (1.0 + ratio)


def make(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.attack_zip) as attack, ZipFile(args.team_zip) as own:
        bg = table(own, "A_bg.csv")
        b = table(own, "B.csv")
        token = own.read("token.txt")
        targets = [int(k) for k in json.loads(attack.read("mia_columns.json"))["target_columns"]]
        rng = np.random.default_rng(args.seed)
        sample_n = len(table(attack, f"C_{targets[0]}.csv"))
        sample_idx = np.sort(rng.choice(len(bg), size=sample_n, replace=False))
        bg_sample = bg.iloc[sample_idx].reset_index(drop=True)
        scale = bg[CONT].std().to_numpy(dtype=float)  # pandas ddof=1; 公式AIAと同じ
        scale = np.where(scale > 0, scale, 1.0)

        mia_mean = bg[MIA_CONT].mean().to_numpy(dtype=float)
        mia_std = bg[MIA_CONT].std(ddof=0).to_numpy(dtype=float)
        mia_std = np.where(mia_std > 0, mia_std, 1.0)
        ref = (bg[MIA_CONT].to_numpy(dtype=float) - mia_mean) / mia_std
        targets_x = (b[MIA_CONT].to_numpy(dtype=float) - mia_mean) / mia_std
        try:
            ref_density = gaussian_kde(ref.T)(targets_x.T)
        except np.linalg.LinAlgError:
            ref_density = None

        mia = pd.DataFrame({"record_id": b["record_id"].to_numpy()})
        aia_frames = []
        summary = []
        for k in targets:
            c = table(attack, f"C_{k}.csv")
            challenge = table(attack, f"aia_challenge_{k}.csv")
            if len(c) != sample_n:
                raise ValueError(f"C_{k}の行数が異なります: {len(c)} != {sample_n}")
            d_c, t_c = nearest(c, challenge, scale)
            d_bg, _ = nearest(bg_sample, challenge, scale)
            choose_c = d_c < args.ratio * d_bg
            time_hat = np.where(choose_c, t_c, 9.5)
            aia_frames.append(pd.DataFrame({
                "target_k": k,
                "challenge_row_id": challenge["challenge_row_id"].to_numpy(),
                "time_hat": time_hat,
            }))
            mia[str(k)] = (np.full(len(b), 0.5) if ref_density is None
                           else mia_reference(c, bg, b, ref_density))
            summary.append((k, int(choose_c.sum()), len(challenge)))

    aia = pd.concat(aia_frames, ignore_index=True)
    if mia.isna().any().any() or aia.isna().any().any():
        raise ValueError("提出値に欠測が含まれます")
    mia.to_csv(out / "F_mia.csv", index=False)
    aia.to_csv(out / "F_aia.csv", index=False)
    (out / "token.txt").write_bytes(token)
    submission = out / "attack_submission.zip"
    with ZipFile(submission, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for name in ["F_mia.csv", "F_aia.csv", "token.txt"]:
            z.write(out / name, arcname=name)
    print(f"提出ZIP: {submission}")
    print(f"対象: {len(targets)}チーム、AIA: {len(aia)}行、MIA: {len(mia)}行")
    print("Cのtimeを採用した行数（チーム: 採用/課題）:")
    print(" ".join(f"{k}:{n}/{total}" for k, n, total in summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack-zip", required=True, type=Path, help="本戦攻撃配布ZIP")
    parser.add_argument("--team-zip", required=True, type=Path, help="自チーム28の配布ZIP")
    parser.add_argument("--out-dir", required=True, type=Path, help="出力先")
    parser.add_argument("--ratio", type=float, default=0.25, help="Cを使う距離比（既定0.25）")
    parser.add_argument("--seed", type=int, default=20261004, help="背景抽出用乱数の種")
    args = parser.parse_args()
    if not 0 < args.ratio <= 1:
        parser.error("--ratio は0超、1以下にしてください")
    make(args)


if __name__ == "__main__":
    main()
