"""Outcome-aware same-cohort MIA red team for Pool, SA, and RareCopula.

The earlier attacks used only baseline covariates.  This module explicitly tests
whether ``time/onset/death`` or their joint relationship with covariates leaks
which rows were used to build a candidate.

Primary evaluation is a B_in/B_out rebuild holdout.  Every candidate is rebuilt
from B_in only; B_out is used only as same-cohort non-members.  Results are
diagnostic AUC and TPR@1%FPR with paired truth/true-rare permutation nulls.  No
official a_MIA or Anon value is computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

import attacks as AT
import defense as D
from anneal_defense import AnnealConfig, anneal_outcomes
from calibration import calibrate_cox_columns
from harness import Bench
from mia_redteam import (
    FINAL_FILES,
    _aggregate_with_null,
    _metric_pair,
    _pool_base,
    _sha256,
    _subbench,
    _tie_safe_score,
)
from third_defense import build_rare_copula


ATTACK_NAMES = (
    "event_time_exact",
    "event_time_near",
    "joint_nn_gap",
    "joint_knn",
    "joint_shadow_c2st",
    "survival_likelihood",
    "cox_residual",
    "outcome_ensemble",
)


def _event_code(frame: pd.DataFrame) -> np.ndarray:
    return frame["onset"].to_numpy(dtype=int) + 2 * frame["death"].to_numpy(dtype=int)


def event_time_exact(C: pd.DataFrame, A: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """Rarity-weighted exact ``(onset, death, time)`` linkage.

    Administrative censoring at 10 years is common and receives no confidence;
    unique event-time matches receive the largest confidence.
    """
    del A
    code_c, code_x = _event_code(C), _event_code(X)
    tc = C["time"].to_numpy(dtype=float)
    tx = X["time"].to_numpy(dtype=float)
    score = np.zeros(len(X), dtype=float)
    for group in (0, 1, 2):
        ctimes = tc[code_c == group]
        positions = np.flatnonzero(code_x == group)
        if len(ctimes) == 0 or len(positions) == 0:
            continue
        for row in positions:
            # Do not reward the schema-wide administrative value.
            if group == 0 and np.isclose(tx[row], D.HORIZON, atol=1e-2):
                continue
            hit = np.isclose(ctimes, tx[row], rtol=1e-9, atol=1e-10)
            if bool(hit.any()):
                frequency = int(hit.sum())
                score[row] = float(-np.log((frequency + 0.5) / (len(ctimes) + 1.0)))
    return score


def event_time_near(C: pd.DataFrame, A: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """Nearest event-time C/A_bg density ratio within the same event triplet."""
    code_c, code_a, code_x = _event_code(C), _event_code(A), _event_code(X)
    tc = C["time"].to_numpy(dtype=float)
    ta = A["time"].to_numpy(dtype=float)
    tx = X["time"].to_numpy(dtype=float)
    out = np.zeros(len(X), dtype=float)
    epsilon = 1.0 / 36525.0  # one hundredth of a day
    for group in (0, 1, 2):
        ctimes = tc[code_c == group]
        atimes = ta[code_a == group]
        positions = np.flatnonzero(code_x == group)
        if len(ctimes) == 0 or len(atimes) == 0 or len(positions) == 0:
            continue
        dc = NearestNeighbors(n_neighbors=1).fit(ctimes[:, None]).kneighbors(tx[positions, None])[0][:, 0]
        da = NearestNeighbors(n_neighbors=min(2, len(atimes))).fit(atimes[:, None]).kneighbors(
            tx[positions, None]
        )[0][:, -1]
        out[positions] = (
            np.log(np.clip(da, epsilon, None))
            - np.log(np.clip(dc, epsilon, None))
            + np.log(len(atimes) / len(ctimes))
        )
        if group == 0:
            admin = np.isclose(tx[positions], D.HORIZON, atol=1e-2)
            out[positions[admin]] = 0.0
    return out


def _joint_features(frame: pd.DataFrame, A: pd.DataFrame) -> np.ndarray:
    av = A[AT.CONT].to_numpy(dtype=float)
    z = (frame[AT.CONT].to_numpy(dtype=float) - av.mean(axis=0)) / np.where(
        av.std(axis=0) > 0.0, av.std(axis=0), 1.0
    )
    time_a = A["time"].to_numpy(dtype=float)
    time_z = (
        (frame["time"].to_numpy(dtype=float) - time_a.mean())
        / max(float(time_a.std()), 1e-9)
    )[:, None]
    onset = frame["onset"].to_numpy(dtype=float)[:, None]
    death = frame["death"].to_numpy(dtype=float)[:, None]
    cats = frame[["sex", "smoking"]].to_numpy(dtype=float)
    return np.column_stack([
        z,
        time_z,
        2.5 * onset,
        2.5 * death,
        time_z * onset,
        time_z * death,
        cats,
    ])


def joint_nn_gap(C: pd.DataFrame, A: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """Nearest-neighbour linkage in covariate + complete outcome space."""
    cz, az, xz = _joint_features(C, A), _joint_features(A, A), _joint_features(X, A)
    dc = NearestNeighbors(n_neighbors=1).fit(cz).kneighbors(xz)[0][:, 0]
    da = NearestNeighbors(n_neighbors=2).fit(az).kneighbors(xz)[0][:, -1]
    return np.log(np.clip(da, 1e-12, None)) - np.log(np.clip(dc, 1e-12, None))


def joint_knn(C: pd.DataFrame, A: pd.DataFrame, X: pd.DataFrame, k: int = 15) -> np.ndarray:
    """k-NN C/A_bg density ratio in joint covariate-outcome space."""
    cz, az, xz = _joint_features(C, A), _joint_features(A, A), _joint_features(X, A)
    k_c, k_a = min(k, len(C)), min(k, len(A))
    dc = NearestNeighbors(n_neighbors=k_c).fit(cz).kneighbors(xz)[0][:, -1]
    da = NearestNeighbors(n_neighbors=k_a).fit(az).kneighbors(xz)[0][:, -1]
    dimension = cz.shape[1]
    return dimension * (
        np.log(np.clip(da, 1e-12, None)) - np.log(np.clip(dc, 1e-12, None))
    ) + np.log(len(A) / len(C))


def joint_shadow_c2st(
    C: pd.DataFrame,
    A: pd.DataFrame,
    X: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    """Flexible C-vs-A source classifier including complete outcomes."""
    xc, xa, xt = _joint_features(C, A), _joint_features(A, A), _joint_features(X, A)
    train = np.vstack([xc, xa])
    labels = np.concatenate([np.ones(len(xc), dtype=int), np.zeros(len(xa), dtype=int)])
    weights = np.concatenate([
        np.full(len(xc), 0.5 / len(xc)),
        np.full(len(xa), 0.5 / len(xa)),
    ])
    trees = ExtraTreesClassifier(
        n_estimators=240,
        min_samples_leaf=4,
        max_features=0.8,
        random_state=seed,
        n_jobs=1,
    )
    trees.fit(train, labels, sample_weight=weights)
    boosted = HistGradientBoostingClassifier(
        learning_rate=0.055,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=18,
        l2_regularization=1.0,
        random_state=seed + 1,
    )
    boosted.fit(train, labels, sample_weight=weights)
    return 0.5 * trees.predict_proba(xt)[:, 1] + 0.5 * boosted.predict_proba(xt)[:, 1]


@dataclass
class _CauseModel:
    beta: np.ndarray
    center: np.ndarray
    rates: np.ndarray
    edges: np.ndarray

    def components(self, frame: pd.DataFrame, event_col: str) -> tuple[np.ndarray, np.ndarray]:
        X = D._design(frame)
        eta = np.clip((X - self.center) @ self.beta, -20.0, 20.0)
        weight = np.exp(eta)
        times = frame["time"].to_numpy(dtype=float)
        exposure = _exposure(times, self.edges)
        cumulative = weight * (exposure @ self.rates)
        event = frame[event_col].to_numpy(dtype=int)
        event_bin = np.clip(np.searchsorted(self.edges, times, side="right") - 1, 0, len(self.rates) - 1)
        log_likelihood = -cumulative + event * (
            np.log(np.clip(self.rates[event_bin], 1e-12, None)) + eta
        )
        martingale = event - cumulative
        residual_loss = martingale**2 / np.clip(cumulative + 0.1, 0.1, None)
        return log_likelihood, residual_loss


def _exposure(times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    starts, ends = edges[:-1], edges[1:]
    return np.clip(
        np.minimum(times[:, None], ends[None, :]) - starts[None, :],
        0.0,
        ends - starts,
    )


def _fit_cause(frame: pd.DataFrame, event_col: str, edges: np.ndarray) -> _CauseModel:
    X = D._design(frame)
    center = X.mean(axis=0)
    try:
        beta = D.fit_cox_raw(frame, event_col).params_.to_numpy(dtype=float)
    except Exception:
        beta = np.zeros(X.shape[1], dtype=float)
    eta = np.clip((X - center) @ beta, -20.0, 20.0)
    risk_weight = np.exp(eta)
    exposure = _exposure(frame["time"].to_numpy(dtype=float), edges)
    denominator = (exposure * risk_weight[:, None]).sum(axis=0)
    event = frame[event_col].to_numpy(dtype=int)
    times = frame["time"].to_numpy(dtype=float)
    event_bin = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(event_bin[event == 1], minlength=len(edges) - 1).astype(float)
    global_rate = (event.sum() + 0.5) / max(float(times.sum()), 1e-9)
    prior_exposure = 2.0
    rates = (counts + global_rate * prior_exposure) / np.clip(
        denominator + prior_exposure, 1e-12, None
    )
    return _CauseModel(beta=beta, center=center, rates=rates, edges=edges)


def survival_scores(
    C: pd.DataFrame,
    A: pd.DataFrame,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """C/A cause-specific piecewise-Cox likelihood and residual advantages."""
    edges = np.linspace(0.0, D.HORIZON, 21)
    models_c = {cause: _fit_cause(C, cause, edges) for cause in ("onset", "death")}
    models_a = {cause: _fit_cause(A, cause, edges) for cause in ("onset", "death")}
    ll_c = np.zeros(len(X)); ll_a = np.zeros(len(X))
    loss_c = np.zeros(len(X)); loss_a = np.zeros(len(X))
    for cause in ("onset", "death"):
        ll, loss = models_c[cause].components(X, cause)
        ll_c += ll; loss_c += loss
        ll, loss = models_a[cause].components(X, cause)
        ll_a += ll; loss_a += loss
    return ll_c - ll_a, loss_a - loss_c


def _rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def outcome_attack_scores(
    C: pd.DataFrame,
    A: pd.DataFrame,
    X: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    likelihood, residual = survival_scores(C, A, X)
    parts = {
        "event_time_exact": event_time_exact(C, A, X),
        "event_time_near": event_time_near(C, A, X),
        "joint_nn_gap": joint_nn_gap(C, A, X),
        "joint_knn": joint_knn(C, A, X),
        "joint_shadow_c2st": joint_shadow_c2st(C, A, X, seed=seed),
        "survival_likelihood": likelihood,
        "cox_residual": residual,
    }
    ensemble_parts = [
        _rank01(parts[name])
        for name in (
            "event_time_near",
            "joint_nn_gap",
            "joint_knn",
            "joint_shadow_c2st",
            "survival_likelihood",
            "cox_residual",
        )
    ]
    parts["outcome_ensemble"] = np.mean(ensemble_parts, axis=0)
    return parts


class _SpecScoreBench:
    """Use U_spec as the selection objective on a smaller holdout rebuild."""

    def __init__(self, base: Bench):
        self.base = base

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def score(self, candidate: pd.DataFrame) -> dict[str, Any]:
        result = self.base.score(candidate)
        result["U"] = float(result["facets"]["U_spec"])
        return result


def _build_outcome_methods(
    bench: Bench,
    B_in: pd.DataFrame,
    rare_in: np.ndarray,
    split: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    started = time.monotonic()
    sub = _subbench(bench, B_in, rare_in)
    base, base_report = _pool_base(sub)
    pool, pool_cal = calibrate_cox_columns(base, sub)

    # The literal min-U selector rolls SA back on small B_in because unstable
    # U_rare binds. The full candidate was selected through U_spec, so force the
    # same declared Cox objective here to actually test outcome optimization.
    spec_bench = _SpecScoreBench(sub)
    config = AnnealConfig(
        steps=5000,
        chains=1,
        temp_start=0.0015,
        temp_end=0.00001,
        top_k=12,
        newton_iter=5,
        rare_move_probability=0.0,
    )
    sa, sa_report = anneal_outcomes(
        base,
        spec_bench,
        seed=7100,
        config=config,
        sanitize_exact=True,
        apply_calibration=True,
        verbose=False,
    )
    copula = build_rare_copula(
        sub,
        seed=6314,
        rare_target=int(sub._rare_mask(sub.B).sum()) + 4,
        rare_shrink=0.45,
        near_q=0.05,
        moment_strength=0.5,
        refine_steps=300,
    )
    copula, copula_cal = calibrate_cox_columns(copula, sub)
    methods = {"Pool": pool, "SA": sa, "Copula": copula}
    report = {
        "split": int(split),
        "n_train": int(len(B_in)),
        "true_rare_train": int(np.sum(rare_in)),
        "pool_base": base_report,
        "methods": {
            "Pool": {
                "actual_U": float(sub.score(pool)["U"]),
                "calibration_lambda": float(pool_cal["selected_lambda"]),
            },
            "SA": {
                "actual_U": float(sub.score(sa)["U"]),
                "selection_U_spec": float(sa_report["post"]["U"]),
                "selected_step": int(sa_report["selected"]["step"]),
                "downhill_accepted": int(sum(x["accepted_downhill"] for x in sa_report["chains"])),
                "calibration_lambda": float(sa_report["calibration"]["selected_lambda"]),
                "forced_spec_selection": True,
            },
            "Copula": {
                "actual_U": float(sub.score(copula)["U"]),
                "calibration_lambda": float(copula_cal["selected_lambda"]),
            },
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }
    return methods, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    bench = Bench(args.dist)
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {attack: [] for attack in ATTACK_NAMES} for method in FINAL_FILES
    }
    sanity: dict[str, dict[str, list[dict[str, Any]]]] = {
        label: {attack: [] for attack in ATTACK_NAMES} for label in ("identity", "null")
    }
    builds: list[dict[str, Any]] = []

    for split in range(args.splits):
        rng = np.random.default_rng(args.seed + split)
        index = rng.permutation(len(bench.B))
        out_index, in_index = index[: args.n_hold], index[args.n_hold :]
        B_in = bench.B.iloc[in_index].reset_index(drop=True)
        rare_in = bench.is_rare[in_index]
        methods, build_report = _build_outcome_methods(bench, B_in, rare_in, split)
        build_report["n_holdout"] = int(len(out_index))
        build_report["true_rare_holdout"] = int(bench.is_rare[out_index].sum())
        builds.append(build_report)
        truth = np.zeros(len(bench.B), dtype=int)
        truth[in_index] = 1
        rare = bench.is_rare.copy()

        for method_index, (method, C) in enumerate(methods.items()):
            scores = outcome_attack_scores(
                C, bench.A_bg, bench.B,
                seed=args.seed + split * 10000 + method_index * 100,
            )
            for attack_index, attack in enumerate(ATTACK_NAMES):
                score = _tie_safe_score(
                    scores[attack],
                    args.seed + split * 10000 + method_index * 100 + attack_index,
                )
                records[method][attack].append({
                    "score": score,
                    "truth": truth,
                    "rare": rare,
                    "metrics": _metric_pair(score, truth, rare, args.fpr),
                })

        identity = B_in[D.SUBMIT_COLS].reset_index(drop=True)
        null_index = rng.choice(len(bench.A_bg), size=len(B_in), replace=False)
        null_candidate = bench.A_bg.iloc[null_index][D.SUBMIT_COLS].reset_index(drop=True)
        for label_index, (label, C) in enumerate((('identity', identity), ('null', null_candidate))):
            scores = outcome_attack_scores(
                C, bench.A_bg, bench.B,
                seed=args.seed + 900000 + split * 100 + label_index,
            )
            for attack_index, attack in enumerate(ATTACK_NAMES):
                score = _tie_safe_score(
                    scores[attack],
                    args.seed + 910000 + split * 100 + label_index * 10 + attack_index,
                )
                sanity[label][attack].append({
                    "score": score,
                    "truth": truth,
                    "rare": rare,
                    "metrics": _metric_pair(score, truth, rare, args.fpr),
                })
        print(
            f"outcome split {split + 1}/{args.splits} complete "
            f"({build_report['elapsed_seconds']:.1f}s build)",
            flush=True,
        )

    permutation_rng = np.random.default_rng(args.seed + 515151)
    permutations = [
        [permutation_rng.permutation(len(bench.B)) for _ in range(args.permutations)]
        for _ in range(args.splits)
    ]
    comparison = {
        method: {
            attack: _aggregate_with_null(split_records, permutations, args.fpr)
            for attack, split_records in by_attack.items()
        }
        for method, by_attack in records.items()
    }
    sanity_result = {
        label: {
            attack: _aggregate_with_null(split_records, permutations, args.fpr)
            for attack, split_records in by_attack.items()
        }
        for label, by_attack in sanity.items()
    }
    candidate_hashes = {
        method: _sha256(args.candidate_root / relative) for method, relative in FINAL_FILES.items()
    }
    return {
        "kind": "non_official_outcome_aware_same_cohort_mia_redteam",
        "official_a_mia_or_anon_computed": False,
        "parameters": {
            "split_seed_base": int(args.seed),
            "splits": int(args.splits),
            "n_train": int(len(bench.B) - args.n_hold),
            "n_holdout": int(args.n_hold),
            "low_fpr": float(args.fpr),
            "null_permutations": int(args.permutations),
            "null": "truth and true-rare labels permuted as a pair; 95th percentile of split mean",
            "attacks": list(ATTACK_NAMES),
            "sa_holdout_selection": (
                "U_spec-forced to prevent small-split U_rare rollback; 5000 steps, seed7100, "
                "T=.0015->1e-5, non-rare outcome swaps, candidate-specific calibration"
            ),
        },
        "attack_definitions": {
            "event_time_exact": "rarity-weighted exact event-triplet/time linkage; admin-10 excluded",
            "event_time_near": "same-event nearest-time C/A_bg ratio",
            "joint_nn_gap": "covariates + time + onset + death nearest-neighbour gap",
            "joint_knn": "joint-space k=15 density ratio",
            "joint_shadow_c2st": "ExtraTrees + HistGradientBoosting C-vs-A_bg posterior on joint features",
            "survival_likelihood": "C-vs-A_bg piecewise cause-specific Cox individual log-likelihood ratio",
            "cox_residual": "A_bg-vs-C cause-specific martingale residual loss advantage",
            "outcome_ensemble": "mean rank of the six continuous outcome-aware attacks",
        },
        "dataset": {
            "dist": str(args.dist),
            "rows_B": int(len(bench.B)),
            "true_rare_B": int(bench.is_rare.sum()),
            "rows_A_bg": int(len(bench.A_bg)),
            "final_candidate_sha256": candidate_hashes,
            "driver_sha256": _sha256(Path(__file__).resolve()),
        },
        "builds": builds,
        "comparison": comparison,
        "sanity": sanity_result,
        "limitations": [
            "This is a memorization red-team diagnostic, not official MIA power or anonymity.",
            "Three rebuild splits and 6-8 true-rare holdout rows make rare low-FPR estimates coarse.",
            "SA selection is intentionally forced through U_spec on B_in so outcome swaps are actually exercised.",
            "Per-attack 95% thresholds are not family-wise adjusted across attacks or methods.",
            "The holdout has many more members and fewer non-members than an official attacker cohort.",
        ],
        "elapsed_seconds": float(time.monotonic() - started),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--n-hold", type=int, default=300)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.dist = args.dist.expanduser().resolve()
    args.candidate_root = args.candidate_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if args.splits < 3:
        raise ValueError("at least three splits are required")
    if not 1 <= args.n_hold < 1049:
        raise ValueError("n_hold must lie in [1,1048]")
    if args.permutations < 100:
        raise ValueError("at least 100 permutations are required")
    if not 0.0 < args.fpr < 1.0:
        raise ValueError("fpr must lie in (0,1)")
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = run(args)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"outcome MIA report: {args.out} ({report['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
