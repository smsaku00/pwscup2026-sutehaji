"""PWSCup2026 main attack, cohort 28, round 1 (2026-10-05).

AIA: original S4 selective nearest neighbour, top 75%, remainder time_hat=0.
MIA: official starter-kit DOMIAS KDE reference baseline (7 continuous columns).
Produces F_aia.csv, F_mia.csv and a flat submission ZIP; does not upload.
Never reads B_self, other teams' B, challenge truth or previous predictions.

Python 3.10+, numpy, pandas, scipy. See 第1回提出_説明書_日本語.md.
MIA equations adapted from PWSCup2026 starter (Apache-2.0); see LICENSE_KIT.txt.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import io
import json
import platform
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde

SEED = 20260922
NUM = ["age", "SBP", "TG", "HDL", "ALT"]
CAT = ["sex", "prefecture"]
QI = NUM + CAT
COLS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
        "smoking", "FPG", "time", "onset", "death"]
CONTINUOUS = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
RANGES = {"age": (40, 74), "BMI": (10, 55), "SBP": (40, 250), "TG": (1, 3000),
          "HDL": (10, 160), "ALT": (.1, 700), "FPG": (40, 400)}
KEEP_FRACTION = .75
DESKTOP = Path.home() / "Desktop"
PROCESS_DATA = DESKTOP / "pwscup2026-kit-main-process-20260912" / "participant_data"
DEFAULT_TARGETS = DESKTOP / "PWSCup2026_final_attack"


def natural_id(value, name):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"{name}: nonnegative integer required")
    number = int(value)
    if number > np.iinfo(np.int64).max:
        raise ValueError(f"{name}: int64 overflow")
    return number


def read_s4_table(path, kind):
    """Preserve original S4 string-to-numeric conversion and canonical sort."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    required = ["challenge_row_id"] + QI if kind == "challenge" else COLS
    allowed = [set(required)]
    if kind == "background":
        allowed.append(set(COLS + ["record_id"]))
    if len(header) != len(set(header)) or set(header) not in allowed:
        raise ValueError(f"{path.name}: missing/duplicate/extra columns")
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    if frame.empty:
        raise ValueError(f"{path.name}: empty data")
    for idcol in ("record_id", "challenge_row_id"):
        if idcol in frame:
            ids = [natural_id(x, idcol) for x in frame[idcol]]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{path.name}: duplicate {idcol}")
            frame[idcol] = ids
    for col in required:
        if col == "challenge_row_id":
            continue
        if col == "prefecture":
            if not frame[col].str.strip().str.len().gt(0).all():
                raise ValueError(f"{path.name}: empty prefecture")
            continue
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        values = frame[col].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}: nonfinite {col}")
        if col in ("age", "sex", "smoking", "onset", "death"):
            if not np.equal(values, np.rint(values)).all():
                raise ValueError(f"{path.name}: noninteger {col}")
            frame[col] = values.astype(np.int64)
        if col in ("sex", "smoking", "onset", "death") and not np.isin(values, [0, 1]).all():
            raise ValueError(f"{path.name}: invalid binary {col}")
        if col in RANGES and ((values < RANGES[col][0]) | (values > RANGES[col][1])).any():
            raise ValueError(f"{path.name}: out-of-range {col}")
    if kind != "challenge":
        if not ((frame.time > 0) & (frame.time <= 10)).all():
            raise ValueError(f"{path.name}: time must be in (0,10]")
        if ((frame.onset == 1) & (frame.death == 1)).any():
            raise ValueError(f"{path.name}: onset/death conflict")
    order = ["challenge_row_id"] if kind == "challenge" else COLS
    return frame[required].sort_values(order, kind="stable").reset_index(drop=True)


def read_mia_table(path, own=False):
    """Official round-trip decimal parsing. Own B keeps record_id for row alignment."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    wanted = set(COLS + (["record_id"] if own else []))
    if len(header) != len(set(header)) or set(header) != wanted:
        raise ValueError(f"{Path(path).name}: unexpected columns (truth columns forbidden)")
    frame = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    values = frame[CONTINUOUS].to_numpy(float)
    if frame.empty or not np.isfinite(values).all():
        raise ValueError(f"{Path(path).name}: empty/nonfinite continuous data")
    if own:
        values = pd.to_numeric(frame.record_id, errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all() or (values < 0).any():
            raise ValueError("B: invalid record_id")
        frame.record_id = values.astype(np.int64)
        if frame.record_id.duplicated().any():
            raise ValueError("B: duplicate record_id")
        frame = frame.sort_values("record_id", kind="stable").reset_index(drop=True)
    return frame


class Metric:
    def __init__(self, background):
        raw = self.raw(background)
        self.center = np.median(raw, axis=0)
        self.scale = np.quantile(raw, .75, axis=0) - np.quantile(raw, .25, axis=0)
        self.scale = np.where(self.scale > 1e-8, self.scale, np.maximum(raw.std(0), 1.))

    @staticmethod
    def raw(frame):
        values = frame[NUM].to_numpy(float).copy()
        values[:, 2] = np.log1p(values[:, 2])
        values[:, 4] = np.log1p(values[:, 4])
        return values

    def distance(self, query, reference):
        x = (self.raw(query) - self.center) / self.scale
        y = (self.raw(reference) - self.center) / self.scale
        distance = cdist(x, y, "sqeuclidean") / len(NUM)
        for col in CAT:
            distance += (query[col].astype(str).to_numpy()[:, None] !=
                         reference[col].astype(str).to_numpy()[None, :]).astype(float)
        return distance


def exact_override(query, c, prediction):
    groups = {}
    for i, row in enumerate(c[QI].itertuples(index=False, name=None)):
        groups.setdefault(row, []).append(float(c.time.iloc[i]))
    result = prediction.copy()
    matches = 0
    for i, row in enumerate(query[QI].itertuples(index=False, name=None)):
        times = groups.get(row)
        if times is None:
            continue
        t = np.asarray(times)
        candidates = np.unique(np.r_[t, np.clip(t - .5, 0, 10), np.clip(t + .5, 0, 10)])
        count = (np.abs(t[:, None] - candidates[None, :]) <= .5).sum(0)
        result[i] = candidates[np.argmax(count)]
        matches += 1
    return result, matches


def s4_predict(background, metric, c, query):
    cost = metric.distance(query, c)
    donor = np.argmin(cost, axis=1)
    dc = cost[np.arange(len(query)), donor]
    n = min(len(c), 1200)
    if len(background) <= n:
        reference = background.reset_index(drop=True)
    else:
        ix = np.sort(np.random.default_rng(SEED + 1).choice(len(background), n, replace=False))
        reference = background.iloc[ix].reset_index(drop=True)
    da = metric.distance(query, reference).min(1)
    evidence = np.log((da + 1e-8) / (dc + 1e-8))
    retained = np.argsort(-evidence, kind="stable")[:int(np.ceil(KEEP_FRACTION * len(query)))]
    prediction = np.zeros(len(query))
    prediction[retained] = c.time.to_numpy(float)[donor[retained]]
    prediction, exact = exact_override(query, c, prediction)
    return prediction, {"selected": len(retained), "exact_QI_rows": exact,
                        "zero_predictions": int((prediction == 0).sum())}


def mia_reference(c, background, b):
    """Same equations, scaling, singular-covariance fallback as official overlap_mia."""
    mean = background[CONTINUOUS].mean().to_numpy(dtype=float)
    std = background[CONTINUOUS].std(ddof=0).to_numpy(dtype=float)
    std = np.where(std > 0, std, 1.)
    syn = (c[CONTINUOUS].to_numpy(float) - mean) / std
    ref = (background[CONTINUOUS].to_numpy(float) - mean) / std
    target = (b[CONTINUOUS].to_numpy(float) - mean) / std
    try:
        kde_syn = gaussian_kde(syn.T)
        kde_ref = gaussian_kde(ref.T)
    except np.linalg.LinAlgError:
        return np.full(len(b), .5), True
    p_syn = kde_syn(target.T)
    p_ref = kde_ref(target.T)
    ratio = p_syn / np.clip(p_ref, 1e-300, None)
    result = ratio / (1. + ratio)
    if not np.isfinite(result).all():
        raise ValueError("MIA: nonfinite KDE result")
    return result, False


def csv_blob(header, rows, id_columns):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([str(int(v)) if j in id_columns else format(float(v), ".17g")
                         for j, v in enumerate(row)])
    return stream.getvalue().encode("utf-8")


def validate_outputs(aia, mia, opponents, challenges, b):
    if list(aia) != ["target_k", "challenge_row_id", "time_hat"]:
        raise ValueError("AIA columns")
    if aia.duplicated(["target_k", "challenge_row_id"]).any():
        raise ValueError("AIA duplicate key")
    if set(aia.target_k) != set(opponents):
        raise ValueError("AIA target coverage")
    for k in opponents:
        actual = aia.loc[aia.target_k == k, "challenge_row_id"]
        expected = challenges[k].challenge_row_id
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError(f"AIA target {k}: challenge coverage")
    times = aia.time_hat.to_numpy(float)
    if not np.isfinite(times).all() or ((times < 0) | (times > 10)).any():
        raise ValueError("AIA time range")
    if list(mia) != ["record_id"] + [str(k) for k in opponents]:
        raise ValueError("MIA columns")
    if len(mia) != len(b) or mia.record_id.duplicated().any() or set(mia.record_id) != set(b.record_id):
        raise ValueError("MIA record_id coverage")
    values = mia.iloc[:, 1:].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("MIA confidence range")


def save(path, blob, inputs, overwrite):
    if path.resolve() in inputs:
        raise ValueError("Output collides with input")
    if path.exists() and path.read_bytes() != blob and not overwrite:
        raise FileExistsError(f"Different output exists: {path}; use --overwrite")
    if not path.exists() or path.read_bytes() != blob:
        path.write_bytes(blob)


def zip_blob(blobs):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in ("F_aia.csv", "F_mia.csv", "token.txt"):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 5, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, blobs[name], compresslevel=9)
    return stream.getvalue()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-dir", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--a-bg", type=Path, default=PROCESS_DATA / "A_bg.csv")
    parser.add_argument("--b", type=Path, default=PROCESS_DATA / "B.csv")
    parser.add_argument("--token", type=Path, default=PROCESS_DATA / "token.txt")
    parser.add_argument("--self-team-id", type=int, default=28)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.self_team_id < 0:
        raise ValueError("Invalid self-team-id")
    meta_path = args.targets_dir / "mia_columns.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    ids = [natural_id(value, "target_columns") for value in meta["target_columns"]]
    if len(ids) != len(set(ids)) or args.self_team_id not in ids:
        raise ValueError("Invalid official target metadata/self cohort")
    opponents = sorted(set(ids) - {args.self_team_id})
    if not opponents:
        raise ValueError("No opponents")
    inputs = {p.resolve() for p in (meta_path, args.a_bg, args.b, args.token)}
    c_paths = {k: args.targets_dir / f"C_{k}.csv" for k in opponents}
    q_paths = {k: args.targets_dir / f"aia_challenge_{k}.csv" for k in opponents}
    inputs.update(p.resolve() for p in (*c_paths.values(), *q_paths.values()))
    token = args.token.read_bytes()
    try:
        lines = [line.strip() for line in token.decode("utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError:
        raise ValueError("token.txt must be UTF-8") from None
    if len(lines) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", lines[0]):
        raise ValueError("Invalid token format; use actual distributed token.txt")
    # Reject known bundled sample tokens without displaying any token content.
    samples = [PROCESS_DATA / "sample_submissions" / "token.txt",
               DESKTOP / "pwscup2026-kit-main-attack-20260928" / "participant_data" / "token.txt"]
    for sample in samples:
        if sample.is_file() and token.strip() == sample.read_bytes().strip():
            raise ValueError("Practice/sample token is forbidden")
    background = read_s4_table(args.a_bg, "background")
    # MIA parsing follows official read_scoring_csv independently of native S4 parsing.
    bg_mia = read_mia_table(args.a_bg, own=True)
    b = read_mia_table(args.b, own=True)
    metric = Metric(background)
    challenges = {k: read_s4_table(q_paths[k], "challenge") for k in opponents}
    c_s4 = {k: read_s4_table(c_paths[k], "C") for k in opponents}
    c_mia = {k: read_mia_table(c_paths[k]) for k in opponents}
    aia_parts, details = [], []
    mia = pd.DataFrame({"record_id": b.record_id})
    for k in opponents:
        prediction, diagnostic = s4_predict(background, metric, c_s4[k], challenges[k])
        aia_parts.append(pd.DataFrame({"target_k": k,
                                      "challenge_row_id": challenges[k].challenge_row_id,
                                      "time_hat": prediction}))
        scores, singular = mia_reference(c_mia[k], bg_mia, b)
        mia[str(k)] = scores
        details.append({"target_k": k, **diagnostic,
                        "mia_singular_covariance_fallback": singular,
                        "mia_unique_scores": int(np.unique(scores).size)})
        print(f"cohort {k}: AIA/MIA calculated", flush=True)
    aia = pd.concat(aia_parts, ignore_index=True).sort_values(
        ["target_k", "challenge_row_id"], kind="stable").reset_index(drop=True)
    blobs = {"F_aia.csv": csv_blob(list(aia), aia.itertuples(index=False, name=None), {0, 1}),
             "F_mia.csv": csv_blob(list(mia), mia.itertuples(index=False, name=None), {0}),
             "token.txt": token}
    # Validate saved decimal representations, rather than just in-memory predictions.
    reread_aia = pd.read_csv(io.BytesIO(blobs["F_aia.csv"]), float_precision="round_trip")
    reread_mia = pd.read_csv(io.BytesIO(blobs["F_mia.csv"]), float_precision="round_trip")
    validate_outputs(reread_aia, reread_mia, opponents, challenges, b)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("F_aia.csv", "F_mia.csv"):
        save(args.output_dir / name, blobs[name], inputs, args.overwrite)
    zipname = f"PWSCup2026_T{args.self_team_id}_round1.zip"
    archive_blob = zip_blob(blobs)
    with zipfile.ZipFile(io.BytesIO(archive_blob)) as archive:
        if archive.testzip() is not None or sorted(archive.namelist()) != sorted(blobs):
            raise ValueError("ZIP integrity/entry error")
        if any(archive.read(name) != blob for name, blob in blobs.items()):
            raise ValueError("ZIP content differs")
    save(args.output_dir / zipname, archive_blob, inputs, args.overwrite)
    record = {"round": 1, "self_cohort_excluded": args.self_team_id,
              "opponents": opponents, "AIA_method": "S4 native top75%; remainder=0",
              "MIA_method": "official DOMIAS KDE reference, continuous7",
              "AIA_rows": len(aia), "MIA_rows": len(mia), "diagnostics": details,
              "csv_sha256": {name: hashlib.sha256(blobs[name]).hexdigest()
                              for name in ("F_aia.csv", "F_mia.csv")},
              "zip_name": zipname, "zip_sha256": hashlib.sha256(archive_blob).hexdigest(),
              "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(inputs) if p != args.token.resolve()},
              "token_source": str(args.token.resolve()), "token_registration_verified": False,
              "internal_format_check": "passed", "official_score": None, "uploaded": False,
              "dependencies": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "scipy": scipy.__version__}}
    blob = (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    save(args.output_dir / "generation_record.json", blob, inputs, args.overwrite)
    print(json.dumps({"zip": str(args.output_dir / zipname), "AIA_rows": len(aia),
                      "MIA_rows": len(mia), "targets": len(opponents),
                      "format": "passed", "uploaded": False}, ensure_ascii=True))


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
