"""seo / 案03 / 順位混合Wald調整 / v1.0 / 2026-09-21.

個票入力は --b と --a-bg のみです。攻撃課題、真値、既存Cは読みません。
本番キット main-process-20260912 の公開スキーマと公開希少検出器を固定しています。
A_bg は入力契約と形式を検査するために読みますが、合成値の計算には使いません。
"""
from __future__ import annotations

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning

# 開発時に固定したアルゴリズム設定（公式の義務値ではない）。
SEED = 9203
DONORS = 3
WALD_ROUNDS = 1
SE_RATIO_BOUNDS = (0.8, 1.2)
LAB_JITTER_IQR = 0.012
EVENT_JITTER_YEARS = 0.015
RISK_BINS = 64
RISK_PENALIZER = 0.05
CALIBRATION_STEPS = 30000
START_TEMPERATURE = 0.012
END_TEMPERATURE = 0.0001
OUTCOME_SEED_OFFSET = 50000
CALIBRATION_SEED_OFFSET = 100003

HORIZON = 10.0
AGE_RANGE = (40, 74)
RANGES = {"BMI": (10., 55.), "SBP": (40., 250.), "TG": (1., 3000.),
          "HDL": (10., 160.), "ALT": (.1, 700.), "FPG": (40., 400.)}
COLS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
        "smoking", "FPG", "time", "onset", "death"]
CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
COX = ["age", "sex", "BMI", "SBP", "log_TG", "HDL", "log_ALT", "FPG", "smoking"]
INTEGERS = ["age", "sex", "smoking", "onset", "death"]

# participant_data/schema.json の公開定数。個人の希少正解ラベルではない。
PUBLIC_OUTLIER_LO = 4.0
PUBLIC_MU = np.array([56.77566349411114, 23.294812108472467, 123.99248206704331,
    98.66064284138636, 47.273300160874896, 18.72786345814881, 91.99791095874004])
PUBLIC_SIGMA = np.array([
    [104.60649311894663, -.8789019242323519, 57.00514018313898, 59.47191157451019, .7750840541190347, -14.643245558651431, 33.39906681671122],
    [-.8789019242323519, 15.689843536286972, 29.030760348444648, 94.2970738843124, -.8920293565829541, 23.064103632916726, 15.311644318894949],
    [57.00514018313898, 29.030760348444648, 365.903631028708, 300.8876636296543, 11.885497463006608, 59.819670509128606, 87.40558091814135],
    [59.47191157451019, 94.2970738843124, 300.8876636296543, 5401.9165088216305, -35.78518366477272, 379.716021717396, 212.23925504519636],
    [.7750840541190347, -.8920293565829541, 11.885497463006608, -35.78518366477272, 69.65428549543113, 19.237832999465834, 15.435874265394421],
    [-14.643245558651431, 23.064103632916726, 59.819670509128606, 379.716021717396, 19.237832999465834, 290.7075680483161, 40.809523864100086],
    [33.39906681671122, 15.311644318894949, 87.40558091814135, 212.23925504519636, 15.435874265394421, 40.809523864100086, 232.01504023996264]])


def validate(df, label, expected=None):
    if list(df.columns) != COLS:
        raise ValueError(f"{label}: 13列の順序が不一致です")
    if expected is not None and len(df) != expected:
        raise ValueError(f"{label}: 行数が不一致です")
    if df.empty or df.isna().any().any():
        raise ValueError(f"{label}: 空表または欠測があります")
    for col in COLS:
        if col == "prefecture":
            if not df[col].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"{label}: prefectureは空でない文字列が必要です")
            continue
        values = df[col].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: {col}に非有限値があります")
        if col in INTEGERS and not np.equal(values, np.rint(values)).all():
            raise ValueError(f"{label}: {col}は整数である必要があります")
        if col in ["sex", "smoking", "onset", "death"] and not np.isin(values, [0, 1]).all():
            raise ValueError(f"{label}: {col}に0/1以外があります")
        lo, hi = RANGES.get(col, AGE_RANGE if col == "age" else (-np.inf, np.inf))
        if ((values < lo) | (values > hi)).any():
            raise ValueError(f"{label}: {col}が公開値域外です")
    if ((df.time <= 0) | (df.time > HORIZON)).any():
        raise ValueError(f"{label}: timeは0より大きく10以下が必要です")
    if ((df.onset == 1) & (df.death == 1)).any():
        raise ValueError(f"{label}: onsetとdeathは同時に1にできません")
    censored = (df.onset == 0) & (df.death == 0)
    if censored.any() and (abs(df.loc[censored, "time"] - HORIZON) > .01).any():
        raise ValueError(f"{label}: 非発症・非死亡のtimeが追跡終了と不整合です")


def read_input(path, label):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError(f"{label}: 重複列名があります")
    df = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip", dtype={"prefecture": str})
    if set(df.columns) not in [set(COLS), set(COLS + ["record_id"])]:
        raise ValueError(f"{label}: 13属性と任意のrecord_id以外は使用できません")
    if "record_id" in df:
        rid = pd.to_numeric(df.record_id, errors="raise").to_numpy(float)
        if not np.isfinite(rid).all() or not np.equal(rid, np.rint(rid)).all() or len(np.unique(rid)) != len(rid):
            raise ValueError(f"{label}: record_idの欠測・非整数・重複")
    df = df[COLS].copy()
    for col in COLS:
        if col != "prefecture":
            df[col] = pd.to_numeric(df[col], errors="raise")
    validate(df, label)
    for col in INTEGERS:
        df[col] = df[col].astype(int)
    return df.reset_index(drop=True)


def cox_frame(df):
    frame = df.copy()
    frame["log_TG"] = np.log(frame.TG.to_numpy(float))
    frame["log_ALT"] = np.log(frame.ALT.to_numpy(float))
    return frame[COX]


def fit_cox(df, penalizer=0.):
    frame = cox_frame(df)
    frame["time"] = df.time.to_numpy(float)
    frame["onset"] = df.onset.to_numpy(int)
    model = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(frame, duration_col="time", event_col="onset")
    if not np.isfinite(model.params_).all() or not (model.standard_errors_ > 0).all():
        raise ValueError("Cox推定が退化しました")
    return model


def public_detector():
    inverse = np.linalg.pinv(PUBLIC_SIGMA)
    def distance(df):
        centered = df[CONT].to_numpy(float) - PUBLIC_MU
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", centered, inverse, centered), 0.))
    return lambda df: distance(df) >= PUBLIC_OUTLIER_LO


def mix_numeric(b, detect, rng):
    n = len(b)
    original = b[CONT].to_numpy(float)
    latent = np.column_stack([
        norm.ppf(pd.Series(original[:, j]).rank(method="average").to_numpy() / (n + 1))
        for j in range(len(CONT))
    ])
    groups = [np.flatnonzero(~detect(b)), np.flatnonzero(detect(b))]
    if any(0 < len(group) < DONORS for group in groups):
        raise ValueError("公開外れ度群の人数が混合数未満です")
    mixed = np.empty_like(latent)
    for group in groups:
        if len(group):
            mixed[group] = sum(latent[rng.permutation(group)] for _ in range(DONORS)) / np.sqrt(DONORS)

    c = b.copy()
    for j, col in enumerate(CONT):
        order = np.argsort(mixed[:, j], kind="stable")
        values = np.empty(n)
        values[order] = np.sort(original[:, j])
        if col != "age":
            iqr = float(np.subtract(*np.quantile(original[:, j], [.75, .25])))
            values += rng.normal(0., LAB_JITTER_IQR * iqr, n)
            values = np.clip(values, *RANGES[col])
        c[col] = values
    c["age"] = np.clip(np.rint(c.age), *AGE_RANGE).astype(int)

    sd = original.std(0)
    if (sd <= 0).any():
        raise ValueError("Bの連続列に定数列があります")
    cost = (((c[CONT].to_numpy(float)[:, None, :] - original[None, :, :]) / sd) ** 2).sum(axis=2)
    rows, donors = linear_sum_assignment(cost)
    cats = ["sex", "smoking", "prefecture"]
    c.loc[rows, cats] = b.iloc[donors][cats].to_numpy()
    c["sex"] = c.sex.astype(int)
    c["smoking"] = c.smoking.astype(int)
    return c


def initial_outcomes(c, b, model, rng):
    c = c.copy()
    rb = np.log(np.clip(model.predict_partial_hazard(cox_frame(b)).to_numpy(float), 1e-12, None))
    rc = np.log(np.clip(model.predict_partial_hazard(cox_frame(c)).to_numpy(float), 1e-12, None))
    tri = np.empty((len(b), 3))
    raw = b[["time", "onset", "death"]].to_numpy(float)
    for source, target in zip(np.array_split(np.argsort(rb), min(RISK_BINS, len(b))),
                              np.array_split(np.argsort(rc), min(RISK_BINS, len(b)))):
        donors = source.copy()
        rng.shuffle(donors)
        tri[target] = raw[donors]
    event = (tri[:, 1] + tri[:, 2]) > 0
    tri[event, 0] = np.clip(tri[event, 0] + rng.normal(0., EVENT_JITTER_YEARS, int(event.sum())),
                            1 / 365.25, HORIZON - .02)
    censored = b.loc[(b.onset == 0) & (b.death == 0), "time"]
    if censored.empty:
        raise ValueError("非発症・非死亡が0人の入力には対応しません")
    tri[~event, 0] = float(censored.median())
    c[["time", "onset", "death"]] = tri
    return c


def calibrate(c, b, model, detect, rng):
    bx = cox_frame(b).to_numpy(float)
    sd = bx.std(0)
    x = (cox_frame(c).to_numpy(float) - bx.mean(0)) / sd
    beta = model.params_.to_numpy() * sd
    weights = np.exp(np.clip(x @ beta, -25., 25.))
    covariance = model.variance_matrix_.to_numpy() * np.outer(sd, sd)
    tri = c[["time", "onset", "death"]].to_numpy(float)
    slots = np.argsort(tri[:, 0], kind="stable")
    fixed = tri[slots].copy()
    permutation = slots.copy()
    event_idx = np.flatnonzero(fixed[:, 1] == 1)
    event_times, starts, counts = np.unique(fixed[event_idx, 0], return_index=True, return_counts=True)
    risk_start = np.searchsorted(fixed[:, 0], event_times, side="left")
    repeated = np.repeat(np.arange(len(counts)), counts)
    fractions = np.concatenate([np.arange(k) / k for k in counts])
    slot_rare = detect(c)[permutation]
    groups = [np.flatnonzero(~slot_rare), np.flatnonzero(slot_rare)]
    movable = np.flatnonzero((fixed[:, 1] + fixed[:, 2]) > 0)

    def loss():
        w = weights[permutation]
        xx = x[permutation]
        wx = w[:, None] * xx
        risk_w = np.cumsum(w[::-1])[::-1][risk_start]
        risk_x = np.cumsum(wx[::-1], axis=0)[::-1][risk_start]
        event_w = np.add.reduceat(w[event_idx], starts)
        event_x = np.add.reduceat(wx[event_idx], starts, axis=0)
        denominator = risk_w[repeated] - fractions * event_w[repeated]
        expected = (risk_x[repeated] - fractions[:, None] * event_x[repeated]) / denominator[:, None]
        gradient = xx[event_idx].sum(0) - expected.sum(0)
        return float(gradient @ covariance @ gradient)

    current = best = loss()
    best_permutation = permutation.copy()
    for step in range(CALIBRATION_STEPS):
        i = int(rng.choice(movable))
        j = int(rng.choice(groups[int(slot_rare[i])]))
        if i == j:
            continue
        permutation[i], permutation[j] = permutation[j], permutation[i]
        new = loss()
        temperature = START_TEMPERATURE * (END_TEMPERATURE / START_TEMPERATURE) ** (step / max(CALIBRATION_STEPS - 1, 1))
        if new < current or rng.random() < np.exp(min(0., (current - new) / temperature)):
            current = new
            if new < best:
                best = new
                best_permutation = permutation.copy()
        else:
            permutation[i], permutation[j] = permutation[j], permutation[i]
    output = c.copy()
    destination = np.empty_like(tri)
    destination[best_permutation] = fixed
    output[["time", "onset", "death"]] = destination
    for col in INTEGERS:
        output[col] = output[col].astype(int)
    return output, best


def generate(b, a_bg, seed=SEED):
    if len(b) < 30 or len(a_bg) < 30 or int(b.onset.sum()) < 10:
        raise ValueError("B/A_bgは各30行以上、Bの発症は10行以上が必要です")
    detect = public_detector()
    target_model = fit_cox(b)
    risk_model = fit_cox(b, penalizer=RISK_PENALIZER)
    c = mix_numeric(b, detect, np.random.default_rng(seed))
    c = initial_outcomes(c, b, risk_model, np.random.default_rng(seed + OUTCOME_SEED_OFFSET))
    c, loss = calibrate(c, b, target_model, detect, np.random.default_rng(seed + CALIBRATION_SEED_OFFSET))
    losses = [loss]
    for round_index in range(WALD_ROUNDS):
        fitted = fit_cox(c)
        adjusted = copy.deepcopy(target_model)
        # 各係数の Wald z=係数/標準誤差 をBに近づける第2段階。
        ratio = np.clip(fitted.standard_errors_ / target_model.standard_errors_, *SE_RATIO_BOUNDS)
        adjusted.params_ = target_model.params_ * ratio
        c, loss = calibrate(c, b, adjusted, detect,
                            np.random.default_rng(seed + 200003 + round_index * 100003))
        losses.append(loss)
    c = c.iloc[np.random.default_rng(seed + 999983).permutation(len(b))].reset_index(drop=True)
    validate(c, "最終C", len(b))
    diagnostic = {"seed": seed, "donors": DONORS, "wald_rounds": WALD_ROUNDS,
                  "cox_gradient_losses": losses,
                  "official_utility": "not computed by generator"}
    print(json.dumps(diagnostic, ensure_ascii=False), flush=True)
    return c


def csv_bytes(df):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(COLS)
    for row in df[COLS].itertuples(index=False, name=None):
        writer.writerow([str(value) if col == "prefecture" else str(int(value)) if col in INTEGERS
                         else format(float(value), ".17g") for col, value in zip(COLS, row)])
    return stream.getvalue().encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--a-bg", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=SEED, help="安定性検証用。通常は既定値を使用")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        if args.seed < 0:
            raise ValueError("seedは0以上が必要です")
        b = read_input(args.b, "B")
        a_bg = read_input(args.a_bg, "A_bg")
        c = generate(b, a_bg, args.seed)
        data = csv_bytes(c)
        destination = Path(__file__).resolve().with_name("C_" + Path(__file__).stem + ".csv")
        if destination in [args.b.resolve(), args.a_bg.resolve()]:
            raise ValueError("入力と出力に同じパスは使用できません")
        if destination.exists() and destination.read_bytes() != data:
            old = destination.read_bytes()
            archive = destination.parent / "versions"
            archive.mkdir(exist_ok=True)
            previous = archive / (destination.stem + "-" + hashlib.sha256(old).hexdigest()[:12] + ".csv")
            if not previous.exists():
                previous.write_bytes(old)
        destination.write_bytes(data)
        loaded = pd.read_csv(destination, float_precision="round_trip", dtype={"prefecture": str})
        validate(loaded, "保存後C", len(b))
        if csv_bytes(loaded) != data:
            raise ValueError("保存後の数値再読込が一致しません")
        print(json.dumps({"output": destination.name, "rows": len(c), "columns": len(COLS),
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "seconds": round(time.perf_counter() - started, 3)}, ensure_ascii=False))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
