#!/usr/bin/env python3
"""Two preliminary AIA methods ported to PWS Cup 2026 main round, Team 28.

The prediction stage reads only public C, challenge QI, and main A_bg.
B_self and hidden answers are not prediction inputs. The optional base ZIP is
used only to copy the existing MIA CSV and team token into a submission ZIP.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.spatial.distance import cdist
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression

NUM = ["age", "SBP", "TG", "HDL", "ALT"]
QI = NUM + ["sex", "prefecture"]
C_COLUMNS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL",
             "ALT", "smoking", "FPG", "time", "onset", "death"]
CENTERS = np.arange(1, 41, dtype=float) / 4
ABSTAIN = 11.0
SEED = 20261009
METHODS = ("pu_rmax", "differential_top50")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_frame(path, kind):
    df = pd.read_csv(path)
    needed = QI + (["challenge_row_id"] if kind == "query" else ["time"])
    require(all(x in df for x in needed), f"{path}: missing required columns")
    numeric = NUM + ["sex"] + ([] if kind == "query" else ["time"])
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
    require(np.isfinite(df[numeric].to_numpy(float)).all(), f"{path}: non-finite values")
    require(df.prefecture.notna().all(), f"{path}: missing prefecture")
    df["prefecture"] = df.prefecture.astype(str)
    require(df.sex.isin([0, 1]).all(), f"{path}: sex must be 0 or 1")
    require((df.TG >= 0).all() and (df.ALT >= 0).all(), f"{path}: negative TG/ALT")
    expected = {"query": 326, "C": 1049, "background": 5143}[kind]
    require(len(df) == expected, f"{path}: expected {expected} rows")
    if kind == "query":
        ids = pd.to_numeric(df.challenge_row_id, errors="raise").to_numpy(float)
        require(np.isfinite(ids).all() and np.equal(ids, np.floor(ids)).all(),
                f"{path}: invalid query IDs")
        df["challenge_row_id"] = ids.astype(int)
        require(df.challenge_row_id.nunique() == 326 and
                set(df.challenge_row_id) == set(range(326)), f"{path}: invalid ID set")
        require(df.prefecture.nunique() == 1, f"{path}: expected one challenge prefecture")
        return df.sort_values("challenge_row_id", kind="stable").reset_index(drop=True)
    require(df.time.between(0, 10).all(), f"{path}: source time outside 0..10")
    # Public full-row sorting makes tie handling independent of file row order.
    columns = [x for x in C_COLUMNS if x in df]
    return df.sort_values(columns, kind="stable").reset_index(drop=True)


def csv_payload(frame):
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def write_csv(path, frame):
    path.write_bytes(csv_payload(frame))


class Metric:
    """Original round-5 numeric metric: log1p TG/ALT, IQR/1.349, mean L1."""
    def __init__(self, bg):
        a = self.raw(bg)
        scale = (np.quantile(a, .75, axis=0) - np.quantile(a, .25, axis=0)) / 1.349
        require((scale > 0).all(), "Background IQR is zero")
        self.scale = scale
        self.center = np.median(a / scale, axis=0)
        self.outlier_sd = max(float(np.std(np.linalg.norm(a / scale - self.center, axis=1))), 1e-6)

    @staticmethod
    def raw(frame):
        a = frame[NUM].to_numpy(float).copy()
        a[:, 2] = np.log1p(a[:, 2])
        a[:, 4] = np.log1p(a[:, 4])
        return a

    def outlier(self, frame):
        return np.linalg.norm(self.raw(frame) / self.scale - self.center, axis=1)

    def distance(self, x, y):
        return cdist(self.raw(x) / self.scale, self.raw(y) / self.scale, "cityblock") / 5


def neighbours(query, table, metric, count):
    d = metric.distance(query, table)
    ix = np.argsort(d, axis=1, kind="stable")[:, :min(count, len(table))]
    ds = np.take_along_axis(d, ix, axis=1)
    weights = np.exp(-ds / np.maximum(ds[:, -1:], 1e-8))
    weights /= weights.sum(axis=1, keepdims=True)
    return table.time.to_numpy()[ix], weights, d.min(axis=1)


def local_tail(query, qdist, ref, rdist, metric, leave_one_out=False):
    qo, ro = metric.outlier(query), metric.outlier(ref)
    result, counts = np.empty(len(query)), np.empty(len(query), int)
    for i, row in enumerate(query.itertuples()):
        delta = ((ref.age.to_numpy() - row.age) / 10) ** 2 + ((ro - qo[i]) / metric.outlier_sd) ** 2
        if leave_one_out:
            delta[i] = np.inf
        take = min(80, len(ref) - int(leave_one_out))
        require(take > 0, "Too few comparison rows")
        ix = np.argsort(delta, kind="stable")[:take]
        result[i] = (1 + int((rdist[ix] <= qdist[i]).sum())) / (take + 1)
        counts[i] = take
    return result, counts


def pu_membership(p, ref_p):
    # Eight boundaries define seven intervals, as in the recovered source code.
    bins = np.array([0, .025, .05, .1, .2, .4, .7, 1.0000001])
    null = (np.histogram(ref_p, bins)[0] + 20 * np.diff(bins)) / (len(ref_p) + 20)
    counts = np.histogram(p, bins)[0]
    which = np.minimum(np.searchsorted(bins, p, side="right") - 1, len(null) - 1)
    q = np.empty(len(p))
    require(len(p) > 1, "PU inference needs at least two queries in each sex")
    for i, j in enumerate(which):
        h = counts.astype(float).copy()
        h[j] -= 1
        ratio = ((h + 20 * null) / (len(p) - 1 + 20)) / null
        smooth = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(
            np.arange(len(ratio)), ratio, sample_weight=null)
        raw = np.clip(1 - .5 / max(float(ratio[j]), 1e-8), .02, .98)
        monotone = np.clip(1 - .5 / max(float(smooth[j]), 1e-8), .02, .98)
        q[i] = .5 * raw + .5 * monotone
    before = q.copy()
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if expit(logit(q) + mid).mean() < .5:
            lo = mid
        else:
            hi = mid
    q = expit(logit(q) + (lo + hi) / 2)
    require(abs(q.mean() - .5) < 1e-12, "PU prior adjustment failed")
    return q, before


def time_options(ct, cw, bt, bw):
    t = np.r_[ct, bt]
    boundaries = np.unique(np.clip(np.r_[0., 10., t - .5, t + .5], 0, 10))
    values = np.r_[ABSTAIN, (boundaries[:-1] + boundaries[1:]) / 2,
                   np.unique(t), boundaries]
    # Strict <= .5 matches the official scorer; no extra boundary tolerance.
    c = (np.abs(values[:, None] - ct[None, :]) <= .5) @ cw
    b = (np.abs(values[:, None] - bt[None, :]) <= .5) @ bw
    c[0] = b[0] = 0.
    return values, np.clip(c, 0, 1), np.clip(b, 0, 1)


def optimize_time(q, options):
    n = len(q) / 2
    require(abs(q.sum() - n) < 1e-9, "Expected membership count is inconsistent")
    ratio = 0.0
    for iteration in range(100):
        ix = [int(np.argmax(qi * c - (1 - ratio) * (1 - qi) * b))
              for qi, (_, c, b) in zip(q, options)]
        m = sum(qi * op[1][j] for qi, op, j in zip(q, options, ix))
        v = sum((1 - qi) * op[2][j] for qi, op, j in zip(q, options, ix))
        require(v <= n + 1e-8, "Estimated control count is invalid")
        updated = float(np.clip((m - v) / max(n - v, 1e-12), 0, 1))
        residual = float(m - v - ratio * (n - v))
        if abs(updated - ratio) < 1e-12:
            return np.array([op[0][j] for op, j in zip(options, ix)]), ix, {
                "estimated_R_not_official_score": updated,
                "estimated_pm": float(m / n), "estimated_pc": float(v / n),
                "iterations": iteration + 1, "fractional_optimality_residual": residual}
        require(updated + 1e-10 >= ratio, "Fractional optimization did not increase")
        ratio = updated
    raise ValueError("Fractional optimization did not converge")


def pu_rmax(k, challenges, c, bg, metric):
    query = challenges[k]
    pref = query.prefecture.iloc[0]
    sources = sorted(j for j, frame in challenges.items()
                     if j != k and frame.prefecture.iloc[0] == pref)
    cross = pd.concat([challenges[j][QI] for j in sources], ignore_index=True) if sources else query.iloc[:0][QI]
    q, raw_q, tails = np.empty(len(query)), np.empty(len(query)), np.empty(len(query))
    options = [None] * len(query)
    references = []
    for sex in sorted(query.sex.unique()):
        idx = np.flatnonzero(query.sex.eq(sex))
        sub = query.iloc[idx].reset_index(drop=True)
        local_c = c[c.sex.eq(sex) & c.prefecture.eq(pref)].reset_index(drop=True)
        local_bg = bg[bg.sex.eq(sex) & bg.prefecture.eq(pref)].reset_index(drop=True)
        require(len(local_c) > 0 and len(local_bg) > 1,
                f"Team {k}, sex {sex}: strict sex/prefecture matching has too few source rows")
        ref = cross[cross.sex.eq(sex)].reset_index(drop=True)
        cross_count = len(ref)
        if cross_count < 100:
            order = np.argsort(np.abs(local_bg.age.to_numpy() - sub.age.median()), kind="stable")
            add = local_bg.iloc[order[:min(100 - cross_count, len(local_bg))]][QI]
            ref = pd.concat([ref[QI], add], ignore_index=True).drop_duplicates(QI).reset_index(drop=True)
        require(len(ref) > 1, f"Team {k}: insufficient comparison population")
        ct, cw, qd = neighbours(sub, local_c, metric, 20)
        bt, bw, _ = neighbours(sub, local_bg, metric, 100)
        rd = metric.distance(ref, local_c).min(axis=1)
        tails[idx], counts = local_tail(sub, qd, ref, rd, metric)
        ref_tail, _ = local_tail(ref, rd, ref, rd, metric, leave_one_out=True)
        q[idx], raw_q[idx] = pu_membership(tails[idx], ref_tail)
        for at, i in enumerate(idx):
            options[i] = time_options(ct[at], cw[at], bt[at], bw[at])
        references.append({"target_k": k, "sex": int(sex), "query_rows": len(sub),
            "same_prefecture_reference_teams": "/".join(map(str, sources)),
            "other_challenge_rows": cross_count, "background_rows_added": len(ref) - cross_count,
            "reference_rows": len(ref), "local_reference_min": int(counts.min()),
            "same_prefecture_sex_C_rows": len(local_c),
            "C_neighbors": min(20, len(local_c)), "background_neighbors": min(100, len(local_bg)),
            "q_min": float(q[idx].min()), "q_max": float(q[idx].max())})
    predictions, ix, summary = optimize_time(q, options)
    diagnostics = pd.DataFrame({"target_k": k, "challenge_row_id": query.challenge_row_id,
        "reference_tail_fraction": tails, "membership_q_estimate_not_calibrated": q,
        "q_before_prior_adjustment": raw_q, "time_hat": predictions,
        "selected": predictions != ABSTAIN,
        "C_window_mass": [op[1][j] for op, j in zip(options, ix)],
        "background_window_mass": [op[2][j] for op, j in zip(options, ix)]})
    return predictions, diagnostics, references, summary


class DifferentialTimeAttack:
    """Documented preliminary team-9 logic, frozen leaf=35 / top=50."""
    def __init__(self, bg, prefectures):
        self.bg = bg
        self.prefectures = sorted(prefectures)
        self.x_bg = self.features(bg)
        self.background_model = self.regressor(SEED)
        self.background_model.fit(self.x_bg, self.targets(bg))

    def features(self, frame):
        a = Metric.raw(frame)
        sex = frame.sex.to_numpy(float)[:, None]
        pref = np.column_stack([frame.prefecture.eq(p).to_numpy(float) for p in self.prefectures])
        return np.column_stack([a, sex, pref])

    @staticmethod
    def targets(frame):
        return (np.abs(frame.time.to_numpy(float)[:, None] - CENTERS[None, :]) <= .5).astype(float)

    @staticmethod
    def regressor(seed):
        return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=35, max_features=.85,
                                   random_state=seed, n_jobs=1)

    def predict(self, k, c, query):
        xc, xq = self.features(c), self.features(query)
        model = self.regressor(SEED + 1)
        model.fit(xc, self.targets(c))
        pc = np.clip(model.predict(xq), 0, 1)
        pb = np.clip(self.background_model.predict(xq), 0, 1)
        delta = pc - pb
        # np.argmax gives the earliest center for equal contrasts, a frozen policy.
        best = delta.argmax(axis=1)
        rows = np.arange(len(query))
        candidate = CENTERS[best]
        best_delta, best_pc, best_pb = delta[rows, best], pc[rows, best], pb[rows, best]
        # The original classifier's detailed settings were not preserved. These
        # explicit main-round choices use the same 500-tree / leaf-35 complexity.
        classifier = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=35,
            max_features=.85, class_weight="balanced", random_state=SEED + 2, n_jobs=1)
        classifier.fit(np.vstack([xc, self.x_bg]), np.r_[np.ones(len(c), int), np.zeros(len(self.bg), int)])
        q_raw = classifier.predict_proba(xq)[:, list(classifier.classes_).index(1)]
        q_soft = .35 + .30 * pd.Series(q_raw).rank(method="average", pct=True).to_numpy()
        score = np.maximum(best_delta, 0) * np.sqrt(best_pc) * (.5 + q_soft)
        # Query rows have already been sorted by ID; stable ties prefer smaller IDs.
        order = np.argsort(-score, kind="stable")
        selected = np.zeros(len(query), bool)
        selected[order[:50]] = True
        predictions = np.where(selected, candidate, ABSTAIN)
        diagnostics = pd.DataFrame({"target_k": k, "challenge_row_id": query.challenge_row_id,
            "candidate_time": candidate, "p_C_window_model_output": best_pc,
            "p_bg_window_model_output": best_pb, "delta": best_delta,
            "q_raw_C_vs_background": q_raw, "q_soft_rank_not_probability": q_soft,
            "selection_score": score, "selected": selected, "time_hat": predictions})
        return predictions, diagnostics, [], {"selected_rows": 50,
            "selected_positive_delta_rows": int((selected & (best_delta > 0)).sum()),
            "selected_mean_delta": float(best_delta[selected].mean())}


def validate_answers(frame, targets):
    require(list(frame.columns) == ["target_k", "challenge_row_id", "time_hat"], "Invalid output columns")
    require(len(frame) == len(targets) * 326 and not frame.duplicated(["target_k", "challenge_row_id"]).any(),
            "Invalid output row count or duplicate keys")
    require(set(frame.target_k) == set(targets), "Invalid output target set")
    require(np.isfinite(frame.time_hat).all() and ((frame.time_hat.between(0, 10)) |
                                                 frame.time_hat.eq(ABSTAIN)).all(), "Invalid output times")
    for k in targets:
        require(set(frame.loc[frame.target_k.eq(k), "challenge_row_id"]) == set(range(326)),
                f"Team {k}: invalid output IDs")


def base_zip_bytes(path, targets):
    with zipfile.ZipFile(path) as z:
        require(set(z.namelist()) == {"F_mia.csv", "F_aia.csv", "token.txt"}, "Base ZIP is not a flat attack package")
        mia_bytes, token = z.read("F_mia.csv"), z.read("token.txt")
    mia = pd.read_csv(io.BytesIO(mia_bytes))
    require("record_id" in mia and len(mia) == 1049 and mia.record_id.nunique() == 1049,
            "Base MIA has invalid record IDs")
    require(set(pd.to_numeric(mia.record_id)) == set(range(1049)), "Base MIA ID set is invalid")
    require({str(k) for k in targets} <= set(mia.columns), "Base MIA misses a main-round target")
    numeric = mia[[str(k) for k in targets]].to_numpy(float)
    require(np.isfinite(numeric).all() and ((numeric >= 0) & (numeric <= 1)).all(), "Invalid base MIA confidence")
    lines = [line.strip() for line in token.decode("utf-8").splitlines() if line.strip()]
    require(len(lines) == 1 and re.fullmatch(r"[A-Za-z0-9]{16,}", lines[0]) is not None,
            "Base team token has invalid format")
    return mia_bytes, token


def build_submission(path, mia_bytes, aia_bytes, token):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in [("F_mia.csv", mia_bytes), ("F_aia.csv", aia_bytes), ("token.txt", token)]:
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 9, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, data)
    with zipfile.ZipFile(path) as z:
        require(set(z.namelist()) == {"F_mia.csv", "F_aia.csv", "token.txt"}, "Invalid submission contents")
        require(z.testzip() is None, "Submission ZIP has a bad CRC")
        require(z.read("F_mia.csv") == mia_bytes and z.read("F_aia.csv") == aia_bytes and
                z.read("token.txt") == token, "Packaged content mismatch")
    require(path.stat().st_size < 20 * 1024 * 1024, "Submission ZIP exceeds 20 MiB")


def run(args):
    out = args.output_dir
    require(not out.exists() or not any(out.iterdir()), "Use a new, empty output directory")
    metadata = json.loads((args.attack_dir / "mia_columns.json").read_text(encoding="utf-8"))
    all_teams = sorted(int(k) for k in metadata["target_columns"])
    require(args.team_id in all_teams, "Own team is missing from main-round metadata")
    targets = [k for k in all_teams if k != args.team_id]
    require(len(targets) == 22, "This port expects the released main-round set of 22 opponents")
    paths = [args.a_bg, args.attack_dir / "mia_columns.json"] + [
        args.attack_dir / f"{stem}_{k}.csv" for k in all_teams for stem in ("C", "aia_challenge")]
    hashes = {str(p.resolve()): digest(p.read_bytes()) for p in paths}
    bg = read_frame(args.a_bg, "background")
    challenges = {k: read_frame(args.attack_dir / f"aia_challenge_{k}.csv", "query") for k in all_teams}
    cs = {k: read_frame(args.attack_dir / f"C_{k}.csv", "C") for k in all_teams}
    require(not pd.concat([x[QI] for x in challenges.values()], ignore_index=True).duplicated(QI).any(),
            "Repeated QI across challenges; known-nonmember construction needs review")
    methods = METHODS if args.method == "both" else (args.method,)
    out.mkdir(parents=True, exist_ok=True)
    metric = Metric(bg) if "pu_rmax" in methods else None
    differential = None
    if "differential_top50" in methods:
        prefectures = set(bg.prefecture)
        for frame in [*cs.values(), *challenges.values()]:
            prefectures.update(frame.prefecture)
        print("Fitting shared background time-window model", flush=True)
        differential = DifferentialTimeAttack(bg, prefectures)
    base = base_zip_bytes(args.base_submission, targets) if args.base_submission else None
    for method in methods:
        folder = out / method
        folder.mkdir()
        frames, diagnostics, references, summaries = [], [], [], []
        for k in targets:
            start = time.monotonic()
            if method == "pu_rmax":
                values, diag, ref, summary = pu_rmax(k, challenges, cs[k], bg, metric)
            else:
                values, diag, ref, summary = differential.predict(k, cs[k], challenges[k])
            frame = pd.DataFrame({"target_k": k, "challenge_row_id": challenges[k].challenge_row_id,
                                  "time_hat": values})
            frames.append(frame); diagnostics.append(diag); references.extend(ref)
            summary.update(target_k=k, rows=326, answered=int((values != ABSTAIN).sum()),
                intentional_misses_11=int((values == ABSTAIN).sum()),
                time_min=float(values.min()), time_max=float(values.max()))
            summaries.append(summary)
            print(f"{method} / team {k}: {summary['answered']} answers, "
                  f"{summary['intentional_misses_11']} at 11 ({time.monotonic()-start:.1f}s)", flush=True)
        answers = pd.concat(frames, ignore_index=True)
        validate_answers(answers, targets)
        payload = csv_payload(answers)
        (folder / "F_aia.csv").write_bytes(payload)
        write_csv(folder / "diagnostics.csv", pd.concat(diagnostics, ignore_index=True))
        write_csv(folder / "summary.csv", pd.DataFrame(summaries))
        if references:
            write_csv(folder / "reference_groups.csv", pd.DataFrame(references))
        result = {"method": method, "team_id": args.team_id, "rows": len(answers),
            "targets": targets, "seed": SEED, "F_aia_sha256": digest(payload),
            "prediction_inputs": "main public C, all challenge QI, main A_bg only",
            "hidden_truth_used": False, "B_self_used_for_prediction": False,
            "official_score": None, "input_sha256": hashes,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
            "parameters": {"intentional_miss_value": 11, "tau_time": .5,
                "selection": "estimated fractional R" if method == "pu_rmax" else "fixed top 50 per opponent",
                "trees": 500 if method == "differential_top50" else None,
                "min_samples_leaf": 35 if method == "differential_top50" else None},
            "summary": summaries}
        if base:
            mia_bytes, token = base
            zip_path = folder / f"zhang_T28_main_{method}_submission.zip"
            build_submission(zip_path, mia_bytes, payload, token)
            result.update(base_submission_name=args.base_submission.name,
                          F_mia_sha256=digest(mia_bytes), submission_zip_sha256=digest(zip_path.read_bytes()),
                          MIA_copied_byte_for_byte=True, token_copied_byte_for_byte=True)
        (folder / "execution_record.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    require(all(digest(Path(p).read_bytes()) == h for p, h in hashes.items()), "Input files changed during execution")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attack-dir", type=Path, required=True, help="Extracted main final_attack directory")
    p.add_argument("--a-bg", type=Path, required=True, help="Main-round A_bg.csv; never use preliminary A_bg")
    p.add_argument("--output-dir", type=Path, required=True, help="New empty directory")
    p.add_argument("--method", choices=["both", *METHODS], default="both")
    p.add_argument("--team-id", type=int, default=28)
    p.add_argument("--base-submission", type=Path, help="Existing valid Team-28 ZIP; copy only F_mia and token")
    run(p.parse_args())


if __name__ == "__main__":
    main()
