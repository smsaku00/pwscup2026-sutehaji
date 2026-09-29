"""Gen-LRA membership inference for the PWS Cup 2026 F_mia.csv format.

Paper: Ward, Wang and Cheng, arXiv:2508.21146v2, Algorithm 1.
The KDE update is evaluated by its exact fixed-bandwidth identity, avoiding
one KDE refit for every candidate. No membership labels or tokens are used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import rankdata


CONTINUOUS = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
COVARIATES = (
    "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG"
)
OUTCOMES = ("time", "onset", "death")
FEATURE_SETS = {
    "continuous": CONTINUOUS,
    "covariates": COVARIATES,
    "all": COVARIATES + OUTCOMES,
}


def _feature_matrices(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    synthetic: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordinal encode categories, then center and scale with reference data.

    A single shared encoding is used for R, B and C. The original record_id
    is deliberately excluded. This is an empirical encoding choice for the
    mixed-type PWS Cup data, not an invariance claim about a finite KDE.
    """
    frames = (reference, candidates, synthetic)
    for name, frame in zip(("A_bg", "B", "C"), frames):
        absent = set(columns) - set(frame.columns)
        if absent:
            raise ValueError(f"{name} is missing columns: {sorted(absent)}")
        if frame.loc[:, columns].isna().any().any():
            raise ValueError(f"{name} has missing feature values")

    arrays: list[list[np.ndarray]] = [[], [], []]
    for column in columns:
        if any(not pd.api.types.is_numeric_dtype(f[column]) for f in frames):
            # The category universe is shared; sorting makes encoding stable
            # even when a target C omits a category present in A_bg.
            categories = sorted(set().union(*(f[column].astype(str).unique() for f in frames)))
            mapping = {category: i for i, category in enumerate(categories)}
            values = [f[column].astype(str).map(mapping).to_numpy(dtype=np.float64) for f in frames]
        else:
            values = [f[column].to_numpy(dtype=np.float64) for f in frames]
        for dest, value in zip(arrays, values):
            if not np.isfinite(value).all():
                raise ValueError(f"{column} has nonfinite values")
            dest.append(value)

    ref, cand, syn = (np.column_stack(parts) for parts in arrays)
    mean, scale = ref.mean(axis=0), ref.std(axis=0)
    scale[scale == 0] = 1.0
    return (ref - mean) / scale, (cand - mean) / scale, (syn - mean) / scale


def silverman_bandwidth(n: int, dimension: int) -> float:
    """Multivariate Silverman rule after standardizing on the reference set."""
    if n < 2 or dimension < 1:
        raise ValueError("KDE requires at least two reference rows and one feature")
    return (4.0 / (dimension + 2) / n) ** (1.0 / (dimension + 4))


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Clipping removes tiny negative roundoff from the matrix product.
    return np.maximum(
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T,
        0.0,
    )


def log_reference_kernel_sums(
    reference: np.ndarray, synthetic: np.ndarray, bandwidth: float, batch_size: int = 128
) -> np.ndarray:
    """log(sum_r exp(-||s-r||^2 / (2 h^2))) for each synthetic row."""
    if bandwidth <= 0 or batch_size < 1:
        raise ValueError("bandwidth and batch_size must be positive")
    chunks = []
    for start in range(0, len(synthetic), batch_size):
        sqdist = _squared_distances(synthetic[start : start + batch_size], reference)
        chunks.append(logsumexp(-sqdist / (2.0 * bandwidth**2), axis=1))
    return np.concatenate(chunks)


def gen_lra_from_matrices(
    reference: np.ndarray,
    candidates: np.ndarray,
    synthetic: np.ndarray,
    *,
    k: int = 10,
    bandwidth: float | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Return the paper's local log likelihood-ratio score for each candidate.

    With the same bandwidth before and after adding x, the exact identity is
      p_(R+x)(s)/p_R(s) = m/(m+1) * (1 + K_h(s-x)/sum_r K_h(s-r)).
    The Gaussian normalization cancels, so log kernels suffice. Neighbors are
    selected by Euclidean distance in the same encoded space used by the KDE.
    A higher result indicates more evidence of membership.
    """
    reference = np.asarray(reference, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    synthetic = np.asarray(synthetic, dtype=np.float64)
    if any(a.ndim != 2 for a in (reference, candidates, synthetic)):
        raise ValueError("R, B and C must be 2-D matrices")
    if not (reference.shape[1] == candidates.shape[1] == synthetic.shape[1]):
        raise ValueError("R, B and C need the same number of features")
    if min(len(reference), len(candidates), len(synthetic)) < 1:
        raise ValueError("R, B and C must all contain rows")
    if not all(np.isfinite(a).all() for a in (reference, candidates, synthetic)):
        raise ValueError("R, B and C must contain finite values")
    if k < 1 or k > len(synthetic):
        raise ValueError("k must be between 1 and the number of C rows")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    m, dimension = reference.shape
    h = silverman_bandwidth(m, dimension) if bandwidth is None else float(bandwidth)
    log_base = log_reference_kernel_sums(reference, synthetic, h, batch_size)
    baseline = np.log(m / (m + 1.0))
    output = np.empty(len(candidates), dtype=np.float64)

    for start in range(0, len(candidates), batch_size):
        sqdist = _squared_distances(candidates[start : start + batch_size], synthetic)
        nearest = np.argpartition(sqdist, k - 1, axis=1)[:, :k]
        local_dist = np.take_along_axis(sqdist, nearest, axis=1)
        log_kernel = -local_dist / (2.0 * h**2)
        log_denom = log_base[nearest]
        # logaddexp computes log(1 + K/sum K_R) without overflow or underflow.
        terms = baseline + np.logaddexp(log_denom, log_kernel) - log_denom
        output[start : start + len(sqdist)] = terms.sum(axis=1)
    return output


def score_target(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    features: str = "all",
    k: int = 10,
    bandwidth_scale: float = 1.0,
    batch_size: int = 128,
) -> np.ndarray:
    if features not in FEATURE_SETS:
        raise ValueError(f"unknown feature set: {features}")
    if bandwidth_scale <= 0:
        raise ValueError("bandwidth_scale must be positive")
    ref, cand, syn = _feature_matrices(reference, candidates, synthetic, FEATURE_SETS[features])
    h = silverman_bandwidth(len(ref), ref.shape[1]) * bandwidth_scale
    return gen_lra_from_matrices(ref, cand, syn, k=k, bandwidth=h, batch_size=batch_size)


def _distribution_files(dist: Path) -> tuple[Path, Path, Path]:
    b = dist / "B.csv"
    if not b.exists():
        b = dist / "B_practice.csv"
    target_dir = dist / "targets" if (dist / "targets" / "mia_columns.json").exists() else dist
    paths = (b, dist / "A_bg.csv", target_dir / "mia_columns.json")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input is missing: {path}")
    return paths


def run(dist: Path, out: Path, features: str, k: int, bandwidth_scale: float, batch_size: int) -> None:
    b_path, ref_path, columns_path = _distribution_files(dist)
    target_dir = columns_path.parent
    target_ids = json.loads(columns_path.read_text(encoding="utf-8"))["target_columns"]
    target_ids = [str(i) for i in target_ids]
    if not target_ids or len(target_ids) != len(set(target_ids)):
        raise ValueError("mia_columns.json must contain distinct target columns")
    missing = [i for i in target_ids if not (target_dir / f"C_{i}.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing C files for target columns: {missing}")

    candidates, reference = pd.read_csv(b_path), pd.read_csv(ref_path)
    if "record_id" not in candidates or candidates["record_id"].isna().any():
        raise ValueError("B must contain a nonmissing record_id column")
    if candidates["record_id"].duplicated().any():
        raise ValueError("B contains duplicate record_id values")
    result = pd.DataFrame({"record_id": candidates["record_id"]})
    for target in target_ids:
        synthetic = pd.read_csv(target_dir / f"C_{target}.csv")
        if len(synthetic) != len(candidates):
            raise ValueError(f"C_{target}.csv row count differs from B")
        raw = score_target(
            reference, candidates, synthetic,
            features=features, k=k, bandwidth_scale=bandwidth_scale, batch_size=batch_size,
        )
        if not np.isfinite(raw).all():
            raise ArithmeticError(f"nonfinite Gen-LRA scores for target {target}")
        # PWS Cup judges rankings at low FPR. A monotone empirical percentile
        # maps raw log scores to [0,1], retaining ties and the attack order.
        result[target] = (rankdata(raw, method="average") - 0.5) / len(raw)
        print(f"Scored target {target}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Wrote {out} ({len(result)} rows, {len(result.columns)} columns)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gen-LRA MIA for PWS Cup 2026")
    parser.add_argument("--dist", type=Path, required=True, help="folder with B, A_bg, C and mia_columns.json")
    parser.add_argument("--out", type=Path, default=Path("F_mia.csv"))
    parser.add_argument("--features", choices=FEATURE_SETS, default="all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run(args.dist, args.out, args.features, args.k, args.bandwidth_scale, args.batch_size)


if __name__ == "__main__":
    main()
