"""seo / 攻撃案02 / 多尺度密度比と差分時間窓 / v1.0 / 2026-09-22.

MIA + AIA の単独実行Python。正解、非公開B、過去の回答、学習済みモデルは内蔵しない。

MIA:
  Bの各候補について、CとA_bgへの1,3,7,15近傍距離を比較する。
  属性のみの距離と、time/onset/deathを含む距離を別々に順位化し、
  その中央値を0～1の順位用スコアとする。校正確率ではない。

AIA:
  QIが近いCとA_bgの行にカーネル重みを与え、幅1年（±0.5年）の
  time窓ごとに C質量 - 0.75*A_bg質量 を計算する。差が最大の窓の中心を
  time_hatとする。C固有の局所時間パターンを狙うヒューリスティックである。

公式得点、phi、protection、Anonは計算しない。出力はスクリプトと同じ
フォルダの F_mia_<stem>.csv と F_aia_<stem>.csv。

Run:
  python "seo-02-多尺度密度比と差分時間窓.py" --b B.csv --a-bg A_bg.csv --targets targets.json
  python "seo-02-多尺度密度比と差分時間窓.py" --self-test

Dependencies tested: Python 3.12, numpy 2.4.6, pandas 2.3.3, scipy 1.17.1.
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
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

SEED = 20260922
MIA_NEIGHBORS = (1, 3, 7, 15)
AIA_NEIGHBORS = 48
AIA_MAX_SUPPORT = 128
BACKGROUND_PENALTY = 0.75
BATCH = 128
EPS = 1e-12
HORIZON = 10.0
TAU_TIME = 0.5

CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
CAT = ["sex", "smoking", "prefecture"]
QI_NUM = ["age", "SBP", "TG", "HDL", "ALT"]
QI_CAT = ["sex", "prefecture"]
QI = QI_NUM + QI_CAT
COLS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
        "smoking", "FPG", "time", "onset", "death"]
RANGES = {"age": (40, 74), "BMI": (10, 55), "SBP": (40, 250),
          "TG": (1, 3000), "HDL": (10, 160), "ALT": (0.1, 700), "FPG": (40, 400)}


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
        paths = []
        for key in ["c_path", "aia_challenge_path"]:
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"target {target}: invalid {key}")
            file_path = Path(value)
            file_path = ((path.parent / file_path).resolve() if not file_path.is_absolute()
                         else file_path.resolve())
            if not file_path.is_file():
                raise ValueError(f"target {target}: missing {key}: {file_path}")
            paths.append(file_path)
        targets.append((target, *paths))
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


class Scaler:
    """A_bgのみから決める頑健な数値尺度。"""
    def __init__(self, a):
        self.center, self.scale = {}, {}
        for col in CONT:
            values = self.transform(a[col].to_numpy(float), col)
            self.center[col] = float(np.median(values))
            spread = float(np.quantile(values, .75) - np.quantile(values, .25))
            self.scale[col] = spread if spread > EPS else max(float(values.std()), 1.)

    @staticmethod
    def transform(values, col):
        return np.log1p(values) if col in ["TG", "ALT"] else values

    def numerical(self, data, columns):
        return np.column_stack([(self.transform(data[col].to_numpy(float), col) - self.center[col]) /
                                self.scale[col] for col in columns])


def mixed_distance(query, reference, scaler, numeric, categorical, include_outcomes=False):
    q = scaler.numerical(query, numeric)
    r = scaler.numerical(reference, numeric)
    distance = cdist(q, r, metric="cityblock")
    for col in categorical:
        distance += query[col].astype(str).to_numpy()[:, None] != reference[col].astype(str).to_numpy()[None, :]
    dimensions = len(numeric) + len(categorical)
    if include_outcomes:
        distance += abs(query.time.to_numpy(float)[:, None] - reference.time.to_numpy(float)[None, :]) / HORIZON
        distance += query.onset.to_numpy(float)[:, None] != reference.onset.to_numpy(float)[None, :]
        distance += query.death.to_numpy(float)[:, None] != reference.death.to_numpy(float)[None, :]
        dimensions += 3
    return distance / dimensions


def percentile(values):
    return (rankdata(values, method="average") - .5) / len(values)


def kth_values(distance, neighbors):
    maximum = min(max(neighbors), distance.shape[1])
    selected = np.partition(distance, maximum - 1, axis=1)[:, :maximum]
    selected.sort(axis=1)
    return {k: selected[:, min(k, selected.shape[1]) - 1] for k in neighbors}


def mia_scores(b, a, c, scaler):
    channels = [
        mixed_distance(b, c, scaler, CONT, CAT, False),
        mixed_distance(b, a, scaler, CONT, CAT, False),
        mixed_distance(b, c, scaler, CONT, CAT, True),
        mixed_distance(b, a, scaler, CONT, CAT, True),
    ]
    c_attr, a_attr = kth_values(channels[0], MIA_NEIGHBORS), kth_values(channels[1], MIA_NEIGHBORS)
    c_full, a_full = kth_values(channels[2], MIA_NEIGHBORS), kth_values(channels[3], MIA_NEIGHBORS)
    components = []
    for k in MIA_NEIGHBORS:
        components.append(percentile(np.log(a_attr[k] + EPS) - np.log(c_attr[k] + EPS)))
        components.append(percentile(np.log(a_full[k] + EPS) - np.log(c_full[k] + EPS)))
    combined = np.median(np.column_stack(components), axis=1)
    return percentile(combined)


def local_weights(query, reference, scaler, novelty=None):
    distance = mixed_distance(query, reference, scaler, QI_NUM, QI_CAT, False)
    result = np.zeros_like(distance)
    support = min(AIA_MAX_SUPPORT, len(reference))
    bandwidth_k = min(AIA_NEIGHBORS, len(reference))
    for row in range(len(query)):
        order = np.argsort(distance[row], kind="stable")[:support]
        bandwidth = max(float(distance[row, order[bandwidth_k - 1]]), 1e-6)
        weights = np.exp(-.5 * (distance[row, order] / bandwidth) ** 2)
        if novelty is not None:
            weights *= novelty[order]
        total = float(weights.sum())
        result[row, order] = weights / total if total > EPS else 1. / len(order)
    return result


def c_novelty(c, a, scaler):
    cc = mixed_distance(c, c, scaler, QI_NUM, QI_CAT, False)
    np.fill_diagonal(cc, np.inf)
    ca = mixed_distance(c, a, scaler, QI_NUM, QI_CAT, False)
    local_c = np.min(cc, axis=1)
    local_a = np.min(ca, axis=1)
    contrast = np.log(local_a + EPS) - np.log(local_c + EPS)
    # 0.5～1.5。C固有の局所構造を穏やかに重視する。
    return .5 + percentile(contrast)


def best_contrast_window(c_times, c_weights, a_times, a_weights):
    all_times = np.r_[c_times, a_times]
    candidates = np.unique(np.clip(np.r_[all_times - TAU_TIME, all_times + TAU_TIME, HORIZON], 0., HORIZON))
    best = None
    for center in candidates:
        mass_c = float(c_weights[np.abs(c_times - center) <= TAU_TIME + EPS].sum())
        mass_a = float(a_weights[np.abs(a_times - center) <= TAU_TIME + EPS].sum())
        score = mass_c - BACKGROUND_PENALTY * mass_a
        key = (score, mass_c, -abs(center - HORIZON), -center)
        if best is None or key > best[0]:
            best = (key, float(center))
    return best[1]


def aia_times(challenge, a, c, scaler):
    novelty = c_novelty(c, a, scaler)
    wc = local_weights(challenge, c, scaler, novelty)
    wa = local_weights(challenge, a, scaler)
    c_times, a_times = c.time.to_numpy(float), a.time.to_numpy(float)
    return np.array([best_contrast_window(c_times, wc[i], a_times, wa[i]) for i in range(len(challenge))])


def infer(b, a, loaded_targets):
    if len(a) < max(MIA_NEIGHBORS) or len(b) < 2:
        raise ValueError("B/A_bg are too small for fixed neighborhood settings")
    scaler = Scaler(a)
    mia = pd.DataFrame({"record_id": b.record_id.to_numpy(np.int64)})
    answers = []
    for target, c, challenge in loaded_targets:
        if len(c) < max(MIA_NEIGHBORS):
            raise ValueError(f"C for target {target} needs at least {max(MIA_NEIGHBORS)} rows")
        score = mia_scores(b, a, c, scaler)
        times = aia_times(challenge, a, c, scaler)
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
    """Artificial functional and reproducibility tests. No competition files are read."""
    import shutil
    import subprocess
    rng = np.random.default_rng(4402)

    def fake(n):
        data = pd.DataFrame({col: rng.uniform(lo + .2 * (hi - lo), lo + .7 * (hi - lo), n)
                             for col, (lo, hi) in RANGES.items()})
        data["age"] = rng.integers(40, 75, n)
        data["sex"] = rng.integers(0, 2, n)
        data["smoking"] = rng.integers(0, 2, n)
        data["prefecture"] = rng.choice(["架空A", "架空B", "架空C"], n)
        data["onset"] = rng.integers(0, 2, n)
        data["death"] = 0
        data["time"] = np.where(data.onset == 0, 10., rng.uniform(.1, 9.7, n))
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
    assert len(first[0]) == len(b) and len(first[1]) == len(challenge)
    b_changed = b.copy()
    b_changed["time"] = 1.
    assert csv_bytes(first[1]) == csv_bytes(infer(b_changed, a, [(7, c, challenge)])[1])

    with tempfile.TemporaryDirectory(prefix="seo_attack02_test_") as temp_dir:
        folder = Path(temp_dir)
        for name, data in [("b.csv", b), ("a.csv", a), ("c.csv", c), ("q.csv", challenge)]:
            data.to_csv(folder / name, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")
        manifest = folder / "targets.json"
        manifest.write_text(json.dumps([{"target_k": 7, "c_path": "c.csv", "aia_challenge_path": "q.csv"}]), encoding="utf-8")
        script = folder / Path(__file__).name
        shutil.copy2(Path(__file__).resolve(), script)
        command = [sys.executable, str(script), "--b", str(folder / "b.csv"),
                   "--a-bg", str(folder / "a.csv"), "--targets", str(manifest)]
        subprocess.run(command, check=True, capture_output=True)
        outputs = [folder / (prefix + script.stem + ".csv") for prefix in ["F_mia_", "F_aia_"]]
        blobs = [path.read_bytes() for path in outputs]
        subprocess.run(command, check=True, capture_output=True)
        assert blobs == [path.read_bytes() for path in outputs]
        bad = challenge.copy()
        bad["time"] = 10.
        bad.to_csv(folder / "bad.csv", index=False)
        try:
            read_table(folder / "bad.csv", "challenge")
        except ValueError:
            pass
        else:
            raise AssertionError("challenge truth column was accepted")
    print(json.dumps({"self_test": "passed", "real_data_used": False,
                      "csv_byte_reproducible": True, "AIA_independent_of_B_time": True,
                      "performance_evaluated": False, "persistent_outputs_created": False}))


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
        print(json.dumps({"name": "seo", "proposal": "02", "version": "1.0",
                          "scope": args.scope, "MIA_rows": len(mia),
                          "targets": [target for target, _, _ in targets], "AIA_rows": len(aia),
                          "MIA_score_kind": "multiscale rank score, not calibrated probability",
                          "AIA_time_unit": "years", "sha256": hashes,
                          "seconds": round(time.perf_counter() - started, 3),
                          "official_scores": "not calculated", "uploaded": False}, ensure_ascii=False))
        if args.scope == "local":
            print("LOCAL ONLY: self-target tests do not contribute to official attack power.", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
