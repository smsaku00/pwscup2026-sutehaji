"""Simulated-annealing outcome assignment for a PoolMatch covariate pool.

──────────────────────────────────────────────────────────────────────────
【提出物との対応】 このスクリプトが作るのは → prelim_team05/methods/C_sa.csv
    ＝ T05_SA.zip に入る C.csv そのもの (sha256 af18ca3e…)
    ログ: prelim_team05/methods/sa.json
          (method = "stratified_outcome_simulated_annealing")

  実際に使ったコマンド:
      python anneal_defense.py --dist data_prelim_team05 \
        --build-seed 5312 --frac-true-like 0.90 \
        --seed 7100 --steps 5000 --chains 1 \
        --temp-start 0.0015 --temp-end 0.00001 \
        --top-k 12 --newton-iter 5 --rare-move-probability 0.0 --calibrate \
        --out prelim_team05/methods/C_sa.csv \
        --log prelim_team05/methods/sa.json

  出発点は Pool と同じ PoolMatch 共変量プール（--build-seed 5312 で defense.build_C
  を呼び直す）。転帰の多重集合は不変なので、C_sa.csv の time 多重集合は
  C_raw_seed5312.csv と完全一致する（系統の確認に使える）。
  提出物と生成コードの一覧 → prelim_team05/PROVENANCE.md
──────────────────────────────────────────────────────────────────────────

The covariates are kept fixed, so the A_bg provenance and near-twin filtering of
the base candidate are preserved.  A move exchanges the complete
``(time, onset, death)`` triplet of an onset row and a non-onset row *inside the
same detected-rare stratum*.  Consequently the following are exact invariants:

* the multiset of follow-up outcomes (and therefore the KM curve),
* the overall and detected-rare onset counts,
* every covariate value and every row's detected-rare membership.

Unlike ``fastcox.refine_assignment`` this optimizer is not greedy: a proposal
with a lower Cox-IO objective is accepted with probability ``exp(delta / T)``.
The temperature follows a geometric schedule, which permits escape from local
optima early and becomes nearly greedy near the end.

Examples
--------
Anneal an existing safe PoolMatch candidate::

    python anneal_defense.py --dist data_prelim_team05 \
        --base prelim_team05/C_raw_seed5312.csv --seed 6100 \
        --steps 8000 --chains 1 --temp-start 0.0015 --temp-end 0.00001 \
        --out prelim_team05/C_anneal.csv --log prelim_team05/anneal.json

Build the PoolMatch covariate pool first, then anneal it::

    python anneal_defense.py --dist data_prelim_team05 --seed 6100 \
        --build-seed 5312 --frac-true-like 0.90 --det-offset 4 \
        --n-knots 128 --polish-steps 4000 --out C_anneal.csv --log anneal.json

This module never reads or writes ``token.txt``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import defense as D  # noqa: E402
import fastcox as F  # noqa: E402
from calibration import calibrate_cox_columns  # noqa: E402
from harness import Bench  # noqa: E402


REQUIRED_DIST_FILES = (
    "A_bg.csv",
    "B_self.csv",
    "schema.json",
    "scoring_config.yaml",
    "utility_ref.json",
)


@dataclass(frozen=True)
class AnnealConfig:
    """Parameters for independent simulated-annealing chains."""

    steps: int = 8000
    chains: int = 1
    temp_start: float = 0.0015
    temp_end: float = 0.00001
    top_k: int = 12
    newton_iter: int = 5
    rare_move_probability: float = 0.0

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.chains < 1:
            raise ValueError("chains must be at least 1")
        if not (np.isfinite(self.temp_start) and self.temp_start > 0.0):
            raise ValueError("temp_start must be finite and positive")
        if not (np.isfinite(self.temp_end) and self.temp_end > 0.0):
            raise ValueError("temp_end must be finite and positive")
        if self.temp_end > self.temp_start:
            raise ValueError("temp_end must not exceed temp_start")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.newton_iter < 2:
            raise ValueError("newton_iter must be at least 2")
        if not 0.0 <= self.rare_move_probability <= 1.0:
            raise ValueError("rare_move_probability must lie in [0, 1]")


@dataclass
class _Snapshot:
    fast_io: float
    chain: int
    step: int
    time: np.ndarray
    onset: np.ndarray
    death: np.ndarray


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_manifest(dist: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_DIST_FILES if not (dist / name).is_file()]
    if missing:
        raise FileNotFoundError("missing distributed files: " + ", ".join(missing))
    return {
        "dist": str(dist),
        "sha256": {name: _sha256(dist / name) for name in REQUIRED_DIST_FILES},
    }


def _temperature(config: AnnealConfig, step: int) -> float:
    if config.steps == 1:
        return config.temp_end
    fraction = step / (config.steps - 1)
    return float(config.temp_start * (config.temp_end / config.temp_start) ** fraction)


def _push_snapshot(snapshots: list[_Snapshot], snapshot: _Snapshot, top_k: int) -> None:
    """Keep a small diverse set for the slower official scorer."""
    snapshots.append(snapshot)
    snapshots.sort(key=lambda item: item.fast_io, reverse=True)
    del snapshots[top_k:]


def _propose_pair(
    onset: np.ndarray,
    rare: np.ndarray,
    rng: np.random.Generator,
    rare_probability: float,
) -> tuple[int, int] | None:
    """Choose event/non-event rows in one rare stratum."""
    valid: list[tuple[bool, np.ndarray, np.ndarray]] = []
    for stratum in (False, True):
        ev = np.flatnonzero((onset == 1) & (rare == stratum))
        non = np.flatnonzero((onset == 0) & (rare == stratum))
        if len(ev) and len(non):
            valid.append((stratum, ev, non))
    if not valid:
        return None

    by_label = {label: (ev, non) for label, ev, non in valid}
    if True in by_label and False in by_label:
        label = bool(rng.random() < rare_probability)
    else:
        label = valid[0][0]
    ev, non = by_label[label]
    return int(ev[rng.integers(len(ev))]), int(non[rng.integers(len(non))])


def _replace_outcomes(
    base: pd.DataFrame,
    time_values: np.ndarray,
    onset_values: np.ndarray,
    death_values: np.ndarray,
) -> pd.DataFrame:
    out = base.copy(deep=True)
    out["time"] = time_values
    out["onset"] = onset_values.astype(int)
    out["death"] = death_values.astype(int)
    return out


def _outcome_multiset(df: pd.DataFrame) -> list[tuple[float, int, int]]:
    values = zip(
        df["time"].to_numpy(dtype=float),
        df["onset"].to_numpy(dtype=int),
        df["death"].to_numpy(dtype=int),
    )
    return sorted((float(t), int(o), int(d)) for t, o, d in values)


def _clinical_exact_counts(C: pd.DataFrame, B: pd.DataFrame) -> dict[str, int]:
    """Count C values that exactly occur in B for the six clinical columns."""
    result: dict[str, int] = {}
    for column in D.CLIN:
        b_values = set(B[column].to_numpy(dtype=float).tolist())
        result[column] = int(sum(float(value) in b_values for value in C[column]))
    return result


def _remove_clinical_exact_matches(
    C: pd.DataFrame,
    B: pd.DataFrame,
    *,
    rel_tol: float = 1e-9,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Move accidental B-value matches by a negligible deterministic amount.

    PoolMatch jitter normally prevents equality, but a clipped or repaired row
    can rarely land on one B value.  The perturbation is about 1e-7 relative,
    far below any scored distributional resolution, and is increased only if it
    would still lie within ``rel_tol`` of another B value.
    """
    out = C.copy(deep=True)
    changed: dict[str, int] = {}
    max_abs_change: dict[str, float] = {}
    for column in D.CLIN:
        reference = B[column].to_numpy(dtype=float)
        values = out[column].to_numpy(dtype=float).copy()
        count = 0
        largest = 0.0
        lo, hi = D.RANGES[column]
        for row, value in enumerate(values):
            tolerance = rel_tol * np.maximum(np.abs(reference), 1e-9)
            if not bool((np.abs(reference - value) <= tolerance).any()):
                continue
            base_step = max(abs(value), 1.0) * 1e-7
            replacement: float | None = None
            for multiplier in range(1, 101):
                for direction in (1.0, -1.0):
                    candidate = float(value + direction * multiplier * base_step)
                    if not lo <= candidate <= hi:
                        continue
                    if not bool((np.abs(reference - candidate) <= tolerance).any()):
                        replacement = candidate
                        break
                if replacement is not None:
                    break
            if replacement is None:
                raise ValueError(f"could not sanitize an accidental {column} match")
            values[row] = replacement
            count += 1
            largest = max(largest, abs(replacement - value))
        out[column] = values
        changed[column] = count
        max_abs_change[column] = float(largest)
    return out, {
        "relative_tolerance": float(rel_tol),
        "changed": changed,
        "changed_total": int(sum(changed.values())),
        "max_absolute_change": max_abs_change,
    }


def _full_row_matches(C: pd.DataFrame, B: pd.DataFrame) -> int:
    columns = [column for column in D.SUBMIT_COLS if column in C and column in B]
    left = C[columns].drop_duplicates()
    right = B[columns].drop_duplicates()
    return int(len(left.merge(right, on=columns, how="inner")))


def _score_summary(score: dict[str, Any]) -> dict[str, Any]:
    spec = score["detail"]["U_spec"]
    rare = score["detail"]["U_rare"]
    return {
        "U": float(score["U"]),
        "facets": {name: float(value) for name, value in score["facets"].items()},
        "io_part": float(spec.get("io_part", 0.0)),
        "tstr_cal": float(spec.get("tstr_cal", 0.0)),
        "rare_detected_C": int(rare.get("rare_detected_C", 0)),
    }


def anneal_outcomes(
    C: pd.DataFrame,
    bench: Bench,
    *,
    seed: int,
    config: AnnealConfig = AnnealConfig(),
    sanitize_exact: bool = True,
    apply_calibration: bool = False,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Optimize outcome assignment with genuine simulated-annealing acceptance.

    The official scorer is applied to the best fast-IO snapshots and the input;
    the result is rolled back unless official ``U`` strictly improves.  This
    protects against approximation error in the fast Breslow Cox fit and against
    a possible ``U_valid`` change.
    """
    config.validate()
    missing = sorted(set(D.SUBMIT_COLS) - set(C.columns))
    if missing:
        raise ValueError(f"base candidate is missing columns: {missing}")
    if len(C) != len(bench.B):
        raise ValueError("base candidate and B must contain the same number of rows")

    base = C[D.SUBMIT_COLS].copy(deep=True).reset_index(drop=True)
    base_official = bench.score(base)
    rare = bench._rare_mask(base).astype(bool)
    X = D._design(base)
    cph_b = D.fit_cox_raw(bench.B, "onset")
    beta_b = cph_b.params_.to_numpy(dtype=float)
    se_b = cph_b.standard_errors_.to_numpy(dtype=float)

    base_time = base["time"].to_numpy(dtype=float)
    base_onset = base["onset"].to_numpy(dtype=int)
    base_death = base["death"].to_numpy(dtype=int)
    beta0, se0, ok0 = F.cox_fit(X, base_time, base_onset)
    if not ok0:
        raise ValueError("fast Cox fit failed for the base candidate")
    base_fast_io = float(F.io_score(beta_b, se_b, beta0, se0))

    snapshots: list[_Snapshot] = [
        _Snapshot(base_fast_io, -1, 0, base_time.copy(), base_onset.copy(), base_death.copy())
    ]
    chains_report: list[dict[str, Any]] = []
    started = time.monotonic()

    for chain in range(config.chains):
        rng = np.random.default_rng(seed + 1000003 * chain)
        t = base_time.copy()
        o = base_onset.copy()
        d = base_death.copy()
        beta, se = beta0.copy(), se0.copy()
        current = base_fast_io
        best_chain = current
        accepted_up = 0
        accepted_down = 0
        rejected = 0
        failed = 0

        for step in range(config.steps):
            pair = _propose_pair(o, rare, rng, config.rare_move_probability)
            if pair is None:
                break
            i, j = pair
            t2, o2, d2 = t.copy(), o.copy(), d.copy()
            t2[i], t2[j] = t[j], t[i]
            o2[i], o2[j] = o[j], o[i]
            d2[i], d2[j] = d[j], d[i]
            beta2, se2, ok2 = F.cox_fit(
                X, t2, o2, n_iter=config.newton_iter, tol=1e-8, beta0=beta
            )
            if not ok2:
                failed += 1
                continue
            proposed = float(F.io_score(beta_b, se_b, beta2, se2))
            delta = proposed - current
            temperature = _temperature(config, step)
            accepted = delta >= 0.0 or rng.random() < math.exp(max(delta / temperature, -745.0))
            if not accepted:
                rejected += 1
                continue
            t, o, d = t2, o2, d2
            beta, se = beta2, se2
            current = proposed
            if delta >= 0.0:
                accepted_up += 1
            else:
                accepted_down += 1
            if proposed > best_chain + 1e-12:
                best_chain = proposed
                _push_snapshot(
                    snapshots,
                    _Snapshot(proposed, chain, step + 1, t.copy(), o.copy(), d.copy()),
                    config.top_k,
                )

        chains_report.append({
            "chain": chain,
            "seed": int(seed + 1000003 * chain),
            "start_fast_io": base_fast_io,
            "end_fast_io": current,
            "best_fast_io": best_chain,
            "accepted_up_or_equal": accepted_up,
            "accepted_downhill": accepted_down,
            "rejected": rejected,
            "cox_failed": failed,
        })
        if verbose:
            print(
                f"chain {chain + 1}/{config.chains}: fast_io={best_chain:.6f}, "
                f"up={accepted_up}, downhill={accepted_down}, rejected={rejected}",
                flush=True,
            )

    # Fast IO is an approximation.  Resolve selection with the official scorer,
    # including the unchanged input as an explicit rollback candidate.
    official_candidates: list[tuple[float, float, pd.DataFrame, dict[str, Any]]] = []
    seen: set[bytes] = set()
    for snapshot in snapshots:
        signature = snapshot.onset.tobytes() + snapshot.time.tobytes() + snapshot.death.tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        candidate = _replace_outcomes(base, snapshot.time, snapshot.onset, snapshot.death)
        score = bench.score(candidate)
        official_candidates.append((
            float(score["U"]),
            snapshot.fast_io,
            candidate,
            {
                "chain": snapshot.chain,
                "step": snapshot.step,
                "fast_io": snapshot.fast_io,
                "official": _score_summary(score),
            },
        ))

    official_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen_u, _, chosen, chosen_record = official_candidates[0]
    accepted = bool(chosen_u > float(base_official["U"]) + 1e-12)
    if not accepted:
        chosen = base
        chosen_record = {
            "chain": -1,
            "step": 0,
            "fast_io": base_fast_io,
            "official": _score_summary(base_official),
        }

    annealed = chosen.copy(deep=True)
    annealed_score = bench.score(annealed)
    sanitation_report: dict[str, Any] | None = None
    finalized = annealed
    if sanitize_exact:
        finalized, sanitation_report = _remove_clinical_exact_matches(annealed, bench.B)
    calibration_report: dict[str, Any] | None = None
    if apply_calibration:
        chosen, calibration_report = calibrate_cox_columns(finalized, bench)
    else:
        chosen = finalized
    final_score = bench.score(chosen)

    anneal_covariates_unchanged = all(
        np.array_equal(annealed[column].to_numpy(), base[column].to_numpy())
        for column in D.SUBMIT_COLS
        if column not in {"time", "onset", "death"}
    )
    final_covariates_unchanged = all(
        np.array_equal(chosen[column].to_numpy(), base[column].to_numpy())
        for column in D.SUBMIT_COLS
        if column not in {"time", "onset", "death"}
    )
    report = {
        "method": "stratified_outcome_simulated_annealing",
        "seed": int(seed),
        "config": asdict(config),
        "accepted": accepted,
        "elapsed_seconds": float(time.monotonic() - started),
        "pre": _score_summary(base_official),
        "anneal_post": _score_summary(annealed_score),
        "post": _score_summary(final_score),
        "improvement": float(final_score["U"] - base_official["U"]),
        "selected": {key: value for key, value in chosen_record.items() if key != "official"},
        "chains": chains_report,
        "official_snapshot_count": len(official_candidates),
        "calibration": calibration_report,
        "exact_match_sanitation": sanitation_report,
        "annealing_invariants": {
            "covariates_unchanged": bool(anneal_covariates_unchanged),
            "outcome_multiset_unchanged": _outcome_multiset(annealed) == _outcome_multiset(base),
            "overall_onset_count_unchanged": int(annealed.onset.sum()) == int(base.onset.sum()),
            "rare_onset_count_unchanged": int(annealed.loc[rare, "onset"].sum())
            == int(base.loc[rare, "onset"].sum()),
            "detected_rare_mask_unchanged": bool(np.array_equal(bench._rare_mask(annealed), rare)),
        },
        "final_audit": {
            "covariates_unchanged_from_base": bool(final_covariates_unchanged),
            "outcome_multiset_unchanged": _outcome_multiset(chosen) == _outcome_multiset(base),
            "overall_onset_count_unchanged": int(chosen.onset.sum()) == int(base.onset.sum()),
            "detected_rare_count": int(bench._rare_mask(chosen).sum()),
        },
        "exact_matches": {
            "clinical_values": _clinical_exact_counts(chosen, bench.B),
            "full_rows": _full_row_matches(chosen, bench.B),
        },
    }
    return chosen[D.SUBMIT_COLS].reset_index(drop=True), report


def _build_base(args: argparse.Namespace, bench: Bench) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.base is not None:
        base_path = args.base.expanduser().resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"base candidate does not exist: {base_path}")
        return pd.read_csv(base_path), {
            "mode": "existing_candidate",
            "path": str(base_path),
            "sha256": _sha256(base_path),
        }

    n_det_b = int(bench._rare_mask(bench.B).sum())
    params = {
        "seed": int(args.build_seed),
        "frac_true_like": float(args.frac_true_like),
        "det_target": int(n_det_b + args.det_offset),
        "n_knots": int(args.n_knots),
        "polish_steps": int(args.polish_steps),
        "min_dist_q": float(args.min_dist_q),
    }
    base = D.build_C(
        bench.B,
        bench.A_bg,
        bench.is_rare,
        bench.ref.mu,
        bench.ref.sigma,
        float(bench.ref.band["lo"]),
        **params,
    )
    return base, {"mode": "built_poolmatch_base", "parameters": params}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=None,
                        help="safe base C.csv; omit to build one from A_bg")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6100)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--temp-start", type=float, default=0.0015)
    parser.add_argument("--temp-end", type=float, default=0.00001)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--newton-iter", type=int, default=5)
    parser.add_argument("--rare-move-probability", type=float, default=0.0)
    parser.add_argument("--calibrate", action="store_true",
                        help="apply candidate-specific safe Cox affine/power calibration")
    parser.add_argument("--no-sanitize-exact", action="store_true",
                        help="do not remove accidental clinical-value matches")
    parser.add_argument("--build-seed", type=int, default=5312)
    parser.add_argument("--frac-true-like", type=float, default=0.90)
    parser.add_argument("--det-offset", type=int, default=4)
    parser.add_argument("--n-knots", type=int, default=128)
    parser.add_argument("--polish-steps", type=int, default=4000)
    parser.add_argument("--min-dist-q", type=float, default=0.15)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = AnnealConfig(
        steps=args.steps,
        chains=args.chains,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        top_k=args.top_k,
        newton_iter=args.newton_iter,
        rare_move_probability=args.rare_move_probability,
    )
    config.validate()
    if not 0.0 <= args.frac_true_like <= 1.0:
        raise ValueError("--frac-true-like must lie in [0, 1]")
    if args.det_offset < 0 or args.n_knots < 2 or args.polish_steps < 0:
        raise ValueError("invalid PoolMatch base parameters")
    if not 0.0 <= args.min_dist_q < 1.0:
        raise ValueError("--min-dist-q must lie in [0, 1)")

    dist = args.dist.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    log_path = args.log.expanduser().resolve()
    if out_path == log_path:
        raise ValueError("--out and --log must be different paths")
    if not args.force:
        existing = [str(path) for path in (out_path, log_path) if path.exists()]
        if existing:
            raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = _dataset_manifest(dist)
    manifest["driver_sha256"] = _sha256(Path(__file__).resolve())
    bench = Bench(dist)
    base, base_info = _build_base(args, bench)
    candidate, report = anneal_outcomes(
        base,
        bench,
        seed=args.seed,
        config=config,
        sanitize_exact=not args.no_sanitize_exact,
        apply_calibration=args.calibrate,
        verbose=args.verbose,
    )
    candidate.to_csv(out_path, index=False)
    report["dataset"] = manifest
    report["base"] = base_info
    report["candidate_sha256"] = _sha256(out_path)
    with log_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)

    post = report["post"]
    facets = post["facets"]
    print(
        f"U={post['U']:.6f} gen={facets['U_gen']:.6f} "
        f"spec={facets['U_spec']:.6f} rare={facets['U_rare']:.6f} "
        f"valid={facets['U_valid']:.6f} elapsed={report['elapsed_seconds']:.1f}s"
    )
    print(f"candidate: {out_path}")
    print(f"report:    {log_path}")


if __name__ == "__main__":
    main()
