"""seo / 攻撃案03 / 正則化生成比と全体割当 / v1.0 / 2026-09-21.

MIA + AIA の単独実行Python。正解、非公開B、過去回答、対象別の例外は内蔵しない。

MIA:
  A_bg の経験分布で連続値を正規得点へ変換し、C と A_bg に正則化
  多変量正規モデルを当てる。カテゴリ頻度比も加え、B 各行の
  log p_C - log p_A_bg を順位化する。出力は確率ではなく順位用スコア。

AIA:
  challenge 全行と C 全行のQI距離を同時に見て、Hungarian法で全体の
  一対一割当を解く。割当先を強く残した局所時間窓の最頻中心を time_hat
  とする。各行を独立に同じC行へ重複対応させる最近傍法とは異なる。

Run:
  python "seo-03-正則化生成比と全体割当.py" --b B.csv --a-bg A_bg.csv --targets targets.json
  python "seo-03-正則化生成比と全体割当.py" --self-test

Dependencies tested: Python 3.12, numpy, pandas. SciPy is not required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np
import pandas as pd

EPS = 1e-12
HORIZON = 10.0
TAU_TIME = 0.5
SHRINKAGE = 0.35
CATEGORY_ALPHA = 0.5
CAT_PENALTY = 1.5
REUSE_PENALTY = 0.25
LOCAL_SUPPORT = 13
ASSIGNED_BOOST = 3.0

CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
CAT = ["sex", "smoking", "prefecture"]
EVENT_CAT = ["onset", "death"]
QI_NUM = ["age", "SBP", "TG", "HDL", "ALT"]
QI_CAT = ["sex", "prefecture"]
QI = QI_NUM + QI_CAT
COLS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
        "smoking", "FPG", "time", "onset", "death"]
RANGES = {"age": (40, 74), "BMI": (10, 55), "SBP": (40, 250),
          "TG": (1, 3000), "HDL": (10, 160), "ALT": (0.1, 700),
          "FPG": (40, 400)}


def integer_series(series, label):
    text = series.astype(str)
    if not text.str.fullmatch(r"[0-9]+").all():
        raise ValueError(f"{label}: nonnegative integer identifiers required")
    values = [int(value) for value in text]
    if any(value > np.iinfo(np.int64).max for value in values):
        raise ValueError(f"{label}: identifier exceeds int64")
    result = pd.Series(values, index=series.index, dtype=np.int64)
    if result.duplicated().any():
        raise ValueError(f"{label}: duplicate identifier")
    return result


def validate_attributes(data, label):
    if data.empty or data.isna().any().any():
        raise ValueError(f"{label}: empty data or missing value")
    for col in data.columns:
        if col in ["record_id", "challenge_row_id"]:
            continue
        if col == "prefecture":
            if not data[col].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise ValueError(f"{label}: empty/non-string prefecture")
            continue
        values = pd.to_numeric(data[col], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{label}: nonfinite {col}")
        if col in ["age", "sex", "smoking", "onset", "death"] and not np.equal(values, np.rint(values)).all():
            raise ValueError(f"{label}: noninteger {col}")
        if col in ["sex", "smoking", "onset", "death"] and not np.isin(values, [0, 1]).all():
            raise ValueError(f"{label}: {col} must be 0 or 1")
        if col in RANGES and ((values < RANGES[col][0]) | (values > RANGES[col][1])).any():
            raise ValueError(f"{label}: {col} outside public range")
        data[col] = values
    if "time" in data:
        if ((data.time < 0) | (data.time > HORIZON)).any():
            raise ValueError(f"{label}: time outside [0,10]")
        if ((data.onset == 1) & (data.death == 1)).any():
            raise ValueError(f"{label}: onset/death both 1")
    return data


def read_table(path, kind):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError(f"{path.name}: duplicate column name")
    required = (["challenge_row_id"] + QI if kind == "challenge"
                else ["record_id"] + COLS if kind == "B" else COLS)
    allowed = [set(required)]
    if kind == "A":
        allowed.append(set(COLS + ["record_id"]))
    if set(header) not in allowed:
        raise ValueError(f"{path.name}: wrong columns or forbidden truth/extra columns")
    data = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip",
                       dtype={"record_id": str, "challenge_row_id": str, "prefecture": str})
    for col in ["record_id", "challenge_row_id"]:
        if col in data:
            data[col] = integer_series(data[col], f"{path.name}/{col}")
    data = validate_attributes(data, path.name)
    if kind == "A":
        data = data[COLS]
    if kind in ["B", "challenge"]:
        key = "record_id" if kind == "B" else "challenge_row_id"
        return data[required].sort_values(key, kind="stable").reset_index(drop=True)
    return data[COLS].sort_values(COLS, kind="stable").reset_index(drop=True)


def id_value(value, label):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"{label}: nonnegative integer required")
    result = int(value)
    if result > np.iinfo(np.int64).max:
        raise ValueError(f"{label}: exceeds int64")
    return result


def load_targets(path):
    items = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(items, list) or not items:
        raise ValueError("targets.json must be a nonempty JSON array")
    targets, seen = [], set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"target_k", "c_path", "aia_challenge_path"}:
            raise ValueError("Each target needs exactly target_k, c_path, aia_challenge_path")
        target = id_value(item["target_k"], "target_k")
        if target in seen:
            raise ValueError(f"duplicate target_k={target}")
        seen.add(target)
        resolved = []
        for key in ["c_path", "aia_challenge_path"]:
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"target {target}: invalid {key}")
            candidate = Path(value)
            candidate = ((path.parent / candidate).resolve() if not candidate.is_absolute()
                         else candidate.resolve())
            if not candidate.is_file():
                raise ValueError(f"target {target}: missing {key}: {candidate}")
            resolved.append(candidate)
        targets.append((target, *resolved))
    return sorted(targets, key=lambda item: item[0])


def check_scope(targets, scope, columns_path, self_id):
    if scope == "local":
        if columns_path is not None or self_id is not None:
            raise ValueError("Use --scope official with --mia-columns and --self-team-id")
        return
    if columns_path is None or self_id is None:
        raise ValueError("official scope requires --mia-columns and --self-team-id")
    meta = json.loads(columns_path.read_text(encoding="utf-8-sig"))
    if not isinstance(meta, dict) or not isinstance(meta.get("target_columns"), list):
        raise ValueError("mia_columns.json: target_columns list required")
    official = [id_value(value, "target_columns") for value in meta["target_columns"]]
    if not official or len(official) != len(set(official)):
        raise ValueError("empty/duplicate official targets")
    own = id_value(self_id, "self-team-id")
    got, allowed = {item[0] for item in targets}, set(official)
    if got - allowed or (allowed - {own}) - got:
        raise ValueError("Targets must cover all official opponents, with no unknown targets")


def percentile(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return (ranks - 0.5) / len(values)


def normal_ppf(probability):
    """Acklam近似による標準正規分布の逆CDF。"""
    p = np.asarray(probability, float)
    a = np.array([-3.969683028665376e+01, 2.209460984245205e+02,
                  -2.759285104469687e+02, 1.383577518672690e+02,
                  -3.066479806614716e+01, 2.506628277459239e+00])
    b = np.array([-5.447609879822406e+01, 1.615858368580409e+02,
                  -1.556989798598866e+02, 6.680131188771972e+01,
                  -1.328068155288572e+01])
    c = np.array([-7.784894002430293e-03, -3.223964580411365e-01,
                  -2.400758277161838e+00, -2.549732539343734e+00,
                   4.374664141464968e+00, 2.938163982698783e+00])
    d = np.array([7.784695709041462e-03, 3.224671290700398e-01,
                  2.445134137142996e+00, 3.754408661907416e+00])
    result = np.empty_like(p)
    low, high = p < 0.02425, p > 0.97575
    middle = ~(low | high)
    if low.any():
        q = np.sqrt(-2.0 * np.log(p[low]))
        result[low] = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                      ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if high.any():
        q = np.sqrt(-2.0 * np.log(1.0 - p[high]))
        result[high] = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                       ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if middle.any():
        q = p[middle] - 0.5
        r = q * q
        result[middle] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
                         (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    return result


class BackgroundTransform:
    """A_bgだけで決める経験CDF変換とAIA用の頑健尺度。"""
    def __init__(self, a):
        self.sorted = {}
        self.center, self.scale = {}, {}
        for col in CONT + ["time"]:
            values = self.base_values(a[col].to_numpy(float), col)
            self.sorted[col] = np.sort(values)
        for col in QI_NUM:
            values = self.base_values(a[col].to_numpy(float), col)
            self.center[col] = float(np.median(values))
            spread = float(np.quantile(values, .75) - np.quantile(values, .25))
            self.scale[col] = spread if spread > EPS else max(float(values.std()), 1.)

    @staticmethod
    def base_values(values, col):
        values = np.asarray(values, float)
        return np.log1p(values) if col in ["TG", "ALT"] else values

    def normal_scores(self, data, columns):
        result = []
        for col in columns:
            values = self.base_values(data[col].to_numpy(float), col)
            reference = self.sorted[col]
            left = np.searchsorted(reference, values, side="left")
            right = np.searchsorted(reference, values, side="right")
            u = (0.5 * (left + right) + 0.5) / (len(reference) + 1.0)
            result.append(normal_ppf(np.clip(u, 1e-4, 1.0 - 1e-4)))
        return np.column_stack(result)

    def qi_numeric(self, data):
        return np.column_stack([(self.base_values(data[col].to_numpy(float), col) - self.center[col]) /
                                self.scale[col] for col in QI_NUM])


class ShrunkGaussian:
    def __init__(self, matrix):
        matrix = np.asarray(matrix, float)
        self.mean = matrix.mean(axis=0)
        empirical = np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))
        diagonal = np.diag(np.maximum(np.diag(empirical), 1e-6))
        self.cov = (1.0 - SHRINKAGE) * empirical + SHRINKAGE * diagonal
        self.cov += np.eye(self.cov.shape[0]) * 1e-6
        sign, self.logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise ValueError("regularized covariance is not positive definite")
        self.precision = np.linalg.inv(self.cov)

    def logpdf(self, matrix):
        delta = np.asarray(matrix, float) - self.mean
        quadratic = np.einsum("ij,jk,ik->i", delta, self.precision, delta)
        return -0.5 * (quadratic + self.logdet + matrix.shape[1] * np.log(2.0 * np.pi))


def categorical_log_ratio(query, c, a, columns):
    result = np.zeros(len(query), float)
    for col in columns:
        q = query[col].astype(str)
        cv = c[col].astype(str)
        av = a[col].astype(str)
        levels = sorted(set(cv) | set(av) | set(q))
        c_counts, a_counts = cv.value_counts(), av.value_counts()
        c_den = len(c) + CATEGORY_ALPHA * len(levels)
        a_den = len(a) + CATEGORY_ALPHA * len(levels)
        for level in levels:
            mask = q == level
            if mask.any():
                pc = (float(c_counts.get(level, 0)) + CATEGORY_ALPHA) / c_den
                pa = (float(a_counts.get(level, 0)) + CATEGORY_ALPHA) / a_den
                result[mask.to_numpy()] += np.log(pc) - np.log(pa)
    return result


def mia_scores(b, a, c, transform):
    # 属性のみとtimeを含む生成比を分離し、一方のモデル誤差への依存を弱める。
    xa = transform.normal_scores(a, CONT)
    xc = transform.normal_scores(c, CONT)
    xb = transform.normal_scores(b, CONT)
    attr_lr = ShrunkGaussian(xc).logpdf(xb) - ShrunkGaussian(xa).logpdf(xb)

    full_columns = CONT + ["time"]
    xaf = transform.normal_scores(a, full_columns)
    xcf = transform.normal_scores(c, full_columns)
    xbf = transform.normal_scores(b, full_columns)
    full_lr = ShrunkGaussian(xcf).logpdf(xbf) - ShrunkGaussian(xaf).logpdf(xbf)
    cat_lr = categorical_log_ratio(b, c, a, CAT + EVENT_CAT)

    combined = 0.45 * percentile(attr_lr) + 0.35 * percentile(full_lr) + 0.20 * percentile(cat_lr)
    return percentile(combined)


def assignment_cost(challenge, c, transform):
    qn, cn = transform.qi_numeric(challenge), transform.qi_numeric(c)
    cost = ((qn[:, None, :] - cn[None, :, :]) ** 2).mean(axis=2)
    for col in QI_CAT:
        mismatch = (challenge[col].astype(str).to_numpy()[:, None] !=
                    c[col].astype(str).to_numpy()[None, :])
        cost += CAT_PENALTY * mismatch
    # 再現可能な極小項。同距離だけを安定に決める。
    cost += np.arange(len(c), dtype=float)[None, :] * 1e-12 / max(len(c), 1)
    return cost


def global_assign(cost):
    rows, columns = cost.shape
    matrix = cost
    if rows > columns:
        layers = int(np.ceil(rows / columns))
        matrix = np.hstack([cost + layer * REUSE_PENALTY for layer in range(layers)])
    # 矩形(n<=m)のHungarian法。外部最適化ライブラリは不要。
    n, m = matrix.shape
    u = np.zeros(n + 1, float)
    v = np.zeros(m + 1, float)
    p = np.zeros(m + 1, int)
    way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            available = np.flatnonzero(~used[1:]) + 1
            current = matrix[i0 - 1, available - 1] - u[i0] - v[available]
            better = current < minv[available]
            changed = available[better]
            minv[changed] = current[better]
            way[changed] = j0
            position = int(np.argmin(minv[available]))
            j1 = int(available[position])
            delta = float(minv[j1])
            occupied = np.flatnonzero(used)
            u[p[occupied]] += delta
            v[occupied] -= delta
            minv[available] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assigned = np.empty(n, dtype=int)
    for j in range(1, m + 1):
        if p[j] != 0:
            assigned[p[j] - 1] = (j - 1) % columns
    return assigned


def best_time_window(times, weights, assigned_time):
    candidates = np.unique(np.clip(np.r_[times - TAU_TIME, times + TAU_TIME,
                                         assigned_time, HORIZON], 0.0, HORIZON))
    best = None
    for center in candidates:
        mass = float(weights[np.abs(times - center) <= TAU_TIME + EPS].sum())
        key = (mass, -abs(center - assigned_time), -abs(center - HORIZON), -center)
        if best is None or key > best[0]:
            best = (key, float(center))
    return best[1]


def aia_times(challenge, c, transform):
    cost = assignment_cost(challenge, c, transform)
    assigned = global_assign(cost)
    c_times = c.time.to_numpy(float)
    answers = np.empty(len(challenge), float)
    support = min(LOCAL_SUPPORT, len(c))
    for row, donor in enumerate(assigned):
        order = np.argsort(cost[row], kind="stable")[:support]
        if donor not in order:
            order[-1] = donor
        local_cost = cost[row, order]
        positive = local_cost[local_cost > local_cost.min() + EPS]
        bandwidth = float(np.median(positive - local_cost.min())) if len(positive) else 1.0
        bandwidth = max(bandwidth, 1e-6)
        weights = np.exp(-(local_cost - local_cost.min()) / bandwidth)
        donor_position = int(np.flatnonzero(order == donor)[0])
        weights[donor_position] *= ASSIGNED_BOOST
        weights /= weights.sum()
        answers[row] = best_time_window(c_times[order], weights, c_times[donor])
    return answers, assigned


def infer(b, a, loaded_targets):
    if len(a) < 16 or len(b) < 2:
        raise ValueError("B/A_bg are too small for the fixed generative model")
    transform = BackgroundTransform(a)
    mia = pd.DataFrame({"record_id": b.record_id.to_numpy(np.int64)})
    answers = []
    for target, c, challenge in loaded_targets:
        if len(c) < 16:
            raise ValueError(f"C for target {target} needs at least 16 rows")
        score = mia_scores(b, a, c, transform)
        times, _ = aia_times(challenge, c, transform)
        if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise ValueError("MIA output outside [0,1]")
        if not np.isfinite(times).all() or ((times < 0) | (times > HORIZON)).any():
            raise ValueError("AIA output outside [0,10]")
        mia[str(target)] = score
        answers.append(pd.DataFrame({"target_k": target,
                                     "challenge_row_id": challenge.challenge_row_id.to_numpy(np.int64),
                                     "time_hat": times}))
    aia = pd.concat(answers, ignore_index=True).sort_values(["target_k", "challenge_row_id"], kind="stable")
    if aia.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("Duplicate AIA output key")
    return mia, aia.reset_index(drop=True)


def csv_bytes(data):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(data.columns)
    ids = {"record_id", "target_k", "challenge_row_id"}
    for row in data.itertuples(index=False, name=None):
        writer.writerow([str(int(value)) if col in ids else format(float(value), ".17g")
                         for col, value in zip(data.columns, row)])
    return stream.getvalue().encode("utf-8")


def save_outputs(mia, aia, input_paths, overwrite=False):
    script = Path(__file__).resolve()
    blobs = [(script.with_name(prefix + script.stem + ".csv"), csv_bytes(data))
             for prefix, data in [("F_mia_", mia), ("F_aia_", aia)]]
    for destination, data in blobs:
        if destination.resolve() in input_paths:
            raise ValueError("Output path collides with an input")
        if destination.exists() and destination.read_bytes() != data and not overwrite:
            raise ValueError(f"{destination.name} exists with different content; use --overwrite")
    for destination, data in blobs:
        if destination.exists() and destination.read_bytes() == data:
            continue
        with tempfile.NamedTemporaryFile(dir=script.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {path.name: hashlib.sha256(data).hexdigest() for path, data in blobs}


def self_test():
    """架空データだけの機能・再現性テスト。大会ファイルは読まない。"""
    import itertools
    import shutil
    import subprocess
    rng = np.random.default_rng(5203)

    def fake(n):
        data = pd.DataFrame({col: rng.uniform(lo + .2 * (hi - lo), lo + .7 * (hi - lo), n)
                             for col, (lo, hi) in RANGES.items()})
        data["age"] = rng.integers(40, 75, n)
        data["sex"] = rng.integers(0, 2, n)
        data["smoking"] = rng.integers(0, 2, n)
        data["prefecture"] = rng.choice(["架空A", "架空B", "架空C"], n)
        data["onset"] = rng.integers(0, 2, n)
        data["death"] = 0
        data["time"] = np.where(data.onset == 0, 10.0, rng.uniform(.1, 9.7, n))
        return data[COLS]

    b = fake(42)
    b.insert(0, "record_id", np.arange(1000, 1042))
    a, c = fake(80), fake(64)
    challenge = fake(18)[QI]
    challenge.insert(0, "challenge_row_id", np.arange(200, 218))
    first = infer(b, a, [(7, c, challenge)])
    second = infer(b, a, [(7, c, challenge)])
    assert all(csv_bytes(x) == csv_bytes(y) for x, y in zip(first, second))
    assert list(first[0].columns) == ["record_id", "7"]
    assert list(first[1].columns) == ["target_k", "challenge_row_id", "time_hat"]
    transform = BackgroundTransform(a)
    _, assigned = aia_times(challenge, c, transform)
    assert len(np.unique(assigned)) == len(challenge)
    small_cost = np.array([[4., 1., 3., 8.], [2., 0., 5., 7.], [3., 2., 2., 6.]])
    small_assignment = global_assign(small_cost)
    obtained = float(small_cost[np.arange(3), small_assignment].sum())
    optimum = min(sum(small_cost[row, col] for row, col in enumerate(cols))
                  for cols in itertools.permutations(range(4), 3))
    assert abs(obtained - optimum) < 1e-12
    b_changed = b.copy()
    b_changed["time"] = 1.0
    assert csv_bytes(first[1]) == csv_bytes(infer(b_changed, a, [(7, c, challenge)])[1])

    with tempfile.TemporaryDirectory(prefix="seo_attack03_test_") as temp_dir:
        folder = Path(temp_dir)
        for name, data in [("b.csv", b), ("a.csv", a), ("c.csv", c), ("q.csv", challenge)]:
            data.to_csv(folder / name, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")
        manifest = folder / "targets.json"
        manifest.write_text(json.dumps([{"target_k": 7, "c_path": "c.csv", "aia_challenge_path": "q.csv"}]), encoding="utf-8")
        script = folder / Path(__file__).name
        shutil.copy2(Path(__file__).resolve(), script)
        command = [sys.executable, str(script), "--b", str(folder / "b.csv"),
                   "--a-bg", str(folder / "a.csv"), "--targets", str(manifest)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"CLI self-test failed: {completed.stderr.strip()} {completed.stdout.strip()}")
        outputs = [folder / (prefix + script.stem + ".csv") for prefix in ["F_mia_", "F_aia_"]]
        blobs = [path.read_bytes() for path in outputs]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"CLI repeat failed: {completed.stderr.strip()} {completed.stdout.strip()}")
        assert blobs == [path.read_bytes() for path in outputs]
        bad = challenge.copy()
        bad["time"] = 10.0
        bad.to_csv(folder / "bad.csv", index=False)
        try:
            read_table(folder / "bad.csv", "challenge")
        except ValueError:
            pass
        else:
            raise AssertionError("challenge truth column was accepted")
    print(json.dumps({"self_test": "passed", "real_data_used": False,
                      "csv_byte_reproducible": True, "global_assignment_unique": True,
                      "AIA_independent_of_B_time": True, "performance_evaluated": False,
                      "persistent_outputs_created": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b", type=Path)
    parser.add_argument("--a-bg", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--scope", choices=["local", "official"], default="local")
    parser.add_argument("--mia-columns", type=Path)
    parser.add_argument("--self-team-id", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        if args.self_test:
            self_test()
            return 0
        if any(value is None for value in [args.b, args.a_bg, args.targets]):
            raise ValueError("--b, --a-bg and --targets are required unless --self-test is used")
        targets = load_targets(args.targets.resolve())
        check_scope(targets, args.scope, args.mia_columns, args.self_team_id)
        b, a = read_table(args.b, "B"), read_table(args.a_bg, "A")
        loaded = [(target, read_table(c_path, "C"), read_table(q_path, "challenge"))
                  for target, c_path, q_path in targets]
        mia, aia = infer(b, a, loaded)
        inputs = {path.resolve() for path in [args.b, args.a_bg, args.targets]}
        inputs.update(path.resolve() for _, c_path, q_path in targets for path in [c_path, q_path])
        if args.mia_columns is not None:
            inputs.add(args.mia_columns.resolve())
        hashes = save_outputs(mia, aia, inputs, args.overwrite)
        print(json.dumps({"name": "seo", "proposal": "03", "version": "1.0",
                          "scope": args.scope, "MIA_rows": len(mia),
                          "targets": [target for target, _, _ in targets], "AIA_rows": len(aia),
                          "MIA_score_kind": "regularized generative rank score, not calibrated probability",
                          "AIA_method": "exact global assignment plus local modal time window",
                          "AIA_time_unit": "years", "sha256": hashes,
                          "seconds": round(time.perf_counter() - started, 3),
                          "official_scores": "not calculated", "uploaded": False}, ensure_ascii=True))
        if args.scope == "local":
            print("LOCAL ONLY: self-target tests do not contribute to official attack power.", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
