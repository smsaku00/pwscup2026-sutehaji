"""Same-cohort rebuild red-team evaluation for the three preliminary defenses.

This is deliberately *not* an official anonymity scorer.  Each split uses only
``B_in`` to rebuild a candidate and treats held-out rows from the same cohort as
non-members.  Thus a score cannot win merely by recognizing the own-cohort vs
``A_bg`` distribution shift.

Two additional attacks are included:

``shadow_c2st``
    A balanced ExtraTrees + histogram-gradient-boosting classifier estimates
    ``P(source=C | x)`` from C versus A_bg.  This is a flexible classifier
    density-ratio attack and uses no membership labels.

``multiview_tail``
    A linkage attack that evaluates local C/A_bg k-NN density ratios in several
    clinically meaningful feature views, calibrates each view on A_bg, and
    takes the most C-like view.  It can detect partial/rare-tail copying hidden
    by a seven-dimensional full-space distance.

The official metric uses TPR at FPR=1%, including a true-rare worst-case.  This
script additionally reports AUC for diagnostic power and permutation-calibrates
both metrics.  It never converts the results to a_MIA or Anon.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

import attacks as AT
import defense as D
import run_defense as RD
from anneal_defense import AnnealConfig, anneal_outcomes
from calibration import calibrate_cox_columns
from harness import Bench
from third_defense import build_rare_copula


Attack = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], np.ndarray]
FINAL_FILES = {
    "Pool": "C_candidate.csv",
    "SA": "methods/C_sa.csv",
    "Copula": "methods/C_copula.csv",
}
CLINICAL_VIEWS = (
    tuple(AT.CONT),
    ("age", "BMI", "FPG"),
    ("age", "TG", "HDL", "ALT"),
    ("BMI", "SBP", "FPG"),
    ("TG", "HDL", "ALT", "FPG"),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _shadow_features(frame: pd.DataFrame, A: pd.DataFrame) -> np.ndarray:
    """Continuous phenotypes plus allowed low-cardinality covariates."""
    raw = frame[AT.CONT].to_numpy(dtype=float)
    mean = A[AT.CONT].to_numpy(dtype=float).mean(axis=0)
    sd = A[AT.CONT].to_numpy(dtype=float).std(axis=0)
    z = (raw - mean) / np.where(sd > 0.0, sd, 1.0)
    log_tg = np.log(np.clip(frame["TG"].to_numpy(dtype=float), 1e-9, None))[:, None]
    log_alt = np.log(np.clip(frame["ALT"].to_numpy(dtype=float), 1e-9, None))[:, None]
    cats = frame[["sex", "smoking"]].to_numpy(dtype=float)
    return np.column_stack([z, log_tg, log_alt, cats])


def shadow_c2st(
    C: pd.DataFrame,
    A: pd.DataFrame,
    X: pd.DataFrame,
    *,
    seed: int = 20260812,
) -> np.ndarray:
    """Flexible classifier density-ratio MIA, trained only on public C/A_bg.

    Balanced source weights make the posterior monotone in p_C(x)/p_A(x).
    Averaging a smooth boosted model and a local tree ensemble covers both
    distributional overfit and record-local memorization.
    """
    xc = _shadow_features(C, A)
    xa = _shadow_features(A, A)
    xt = _shadow_features(X, A)
    train = np.vstack([xc, xa])
    labels = np.concatenate([np.ones(len(xc), dtype=int), np.zeros(len(xa), dtype=int)])
    weights = np.concatenate([
        np.full(len(xc), 0.5 / len(xc), dtype=float),
        np.full(len(xa), 0.5 / len(xa), dtype=float),
    ])

    trees = ExtraTreesClassifier(
        n_estimators=240,
        min_samples_leaf=4,
        max_features=0.8,
        class_weight=None,
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


def _scaled(frame: pd.DataFrame, A: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    av = A[list(columns)].to_numpy(dtype=float)
    sd = av.std(axis=0)
    return (frame[list(columns)].to_numpy(dtype=float) - av.mean(axis=0)) / np.where(
        sd > 0.0, sd, 1.0
    )


def multiview_tail_linkage(
    C: pd.DataFrame,
    A: pd.DataFrame,
    X: pd.DataFrame,
    *,
    k: int = 3,
) -> np.ndarray:
    """Maximum A_bg-calibrated local-density linkage over phenotype views."""
    if k < 1:
        raise ValueError("k must be positive")
    scores: list[np.ndarray] = []
    for columns in CLINICAL_VIEWS:
        cz, az, xz = _scaled(C, A, columns), _scaled(A, A, columns), _scaled(X, A, columns)
        nn_c = NearestNeighbors(n_neighbors=min(k, len(C))).fit(cz)
        nn_a = NearestNeighbors(n_neighbors=min(k + 1, len(A))).fit(az)
        dc_x = nn_c.kneighbors(xz)[0][:, -1]
        da_x = nn_a.kneighbors(xz)[0][:, min(k - 1, nn_a.n_neighbors - 1)]
        raw_x = len(columns) * (
            np.log(np.clip(da_x, 1e-12, None)) - np.log(np.clip(dc_x, 1e-12, None))
        ) + np.log(len(A) / len(C))

        # Calibrate view scales without membership labels. A queries require
        # k+1 neighbours because their first A neighbour is themselves.
        dc_a = nn_c.kneighbors(az)[0][:, -1]
        da_a = nn_a.kneighbors(az)[0][:, -1]
        raw_a = len(columns) * (
            np.log(np.clip(da_a, 1e-12, None)) - np.log(np.clip(dc_a, 1e-12, None))
        ) + np.log(len(A) / len(C))
        baseline = np.sort(raw_a)
        percentile = np.searchsorted(baseline, raw_x, side="right") / len(baseline)
        scores.append(percentile)
    return np.max(np.vstack(scores), axis=0)


def _subbench(bench: Bench, B_in: pd.DataFrame, rare_in: np.ndarray) -> Bench:
    sub = copy.copy(bench)
    sub.B = B_in.reset_index(drop=True).copy()
    sub.is_rare = np.asarray(rare_in, dtype=bool).copy()
    sub.ref = copy.copy(bench.ref)
    sub.ref.b_is_rare = sub.is_rare
    return sub


def _pool_base(sub: Bench) -> tuple[pd.DataFrame, dict[str, Any]]:
    detected = int(sub._rare_mask(sub.B).sum())
    C = D.build_C(
        sub.B,
        sub.A_bg,
        sub.is_rare,
        sub.ref.mu,
        sub.ref.sigma,
        float(sub.ref.band["lo"]),
        seed=5312,
        n_knots=128,
        frac_true_like=0.90,
        det_target=detected + 4,
        jitter=0.25,
        polish_steps=4000,
        min_dist_q=0.15,
        rare_model="logit",
        verbose=False,
    )
    score = sub.score(C)
    refined = False
    if score["facets"]["U_spec"] == min(score["facets"].values()) and score["facets"]["U_spec"] > 0.85:
        candidate = RD.refine(C, sub.B, sub, 150, 5312)
        candidate_score = sub.score(candidate)
        if candidate_score["U"] > score["U"]:
            C, score, refined = candidate, candidate_score, True
    return C, {
        "detected_rare_train": detected,
        "refined": refined,
        "raw_U": float(score["U"]),
    }


def _build_methods(
    bench: Bench,
    B_in: pd.DataFrame,
    rare_in: np.ndarray,
    split: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    started = time.monotonic()
    sub = _subbench(bench, B_in, rare_in)
    base, base_report = _pool_base(sub)

    pool, pool_cal = calibrate_cox_columns(base, sub)
    sa_config = AnnealConfig(
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
        sub,
        seed=7100,
        config=sa_config,
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

    candidates = {"Pool": pool, "SA": sa, "Copula": copula}
    report = {
        "split": int(split),
        "n_train": int(len(B_in)),
        "true_rare_train": int(np.sum(rare_in)),
        "pool_base": base_report,
        "methods": {
            "Pool": {
                "U": float(sub.score(pool)["U"]),
                "calibration_lambda": float(pool_cal["selected_lambda"]),
            },
            "SA": {
                "U": float(sa_report["post"]["U"]),
                "anneal_U": float(sa_report["anneal_post"]["U"]),
                "selected_step": int(sa_report["selected"]["step"]),
                "calibration_lambda": float(sa_report["calibration"]["selected_lambda"]),
            },
            "Copula": {
                "U": float(sub.score(copula)["U"]),
                "calibration_lambda": float(copula_cal["selected_lambda"]),
            },
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }
    return candidates, report


def _tie_safe_score(score: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(score, dtype=float).copy()
    if not np.isfinite(values).all():
        raise ValueError("attack returned non-finite scores")
    rng = np.random.default_rng(seed)
    return values + rng.uniform(-0.5, 0.5, len(values)) * (np.ptp(values) + 1.0) * 1e-12


def _auc(score: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=int)
    n1, n0 = int(truth.sum()), int((1 - truth).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float((ranks[truth == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _tpr_at(score: np.ndarray, truth: np.ndarray, fpr: float) -> float:
    order = np.argsort(-score, kind="mergesort")
    y = np.asarray(truth, dtype=int)[order]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    tp = np.concatenate([[0.0], np.cumsum(y) / n1])
    fp = np.concatenate([[0.0], np.cumsum(1 - y) / n0])
    return float(np.interp(fpr, fp, tp))


def _metric_pair(
    score: np.ndarray,
    truth: np.ndarray,
    rare: np.ndarray,
    fpr: float,
) -> dict[str, float]:
    rare = np.asarray(rare, dtype=bool)
    return {
        "auc": _auc(score, truth),
        "tpr": _tpr_at(score, truth, fpr),
        "auc_rare": _auc(score[rare], truth[rare]),
        "tpr_rare": _tpr_at(score[rare], truth[rare], fpr),
    }


def _evaluate_attack(
    attack: Attack,
    C: pd.DataFrame,
    bench: Bench,
    truth: np.ndarray,
    rare: np.ndarray,
    seed: int,
    fpr: float,
) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(attack(C, bench.A_bg, bench.B), dtype=float)
    score = _tie_safe_score(raw, seed)
    return score, _metric_pair(score, truth, rare, fpr)


def _aggregate_with_null(
    split_records: list[dict[str, Any]],
    permutations: list[list[np.ndarray]],
    fpr: float,
) -> dict[str, Any]:
    actual_keys = ("auc", "tpr", "auc_rare", "tpr_rare")
    actual = {
        key: float(np.nanmean([record["metrics"][key] for record in split_records]))
        for key in actual_keys
    }
    n_perm = len(permutations[0])
    null = {key: np.empty(n_perm, dtype=float) for key in actual_keys}
    for p in range(n_perm):
        values = {key: [] for key in actual_keys}
        for split, record in enumerate(split_records):
            perm = permutations[split][p]
            truth_p = record["truth"][perm]
            rare_p = record["rare"][perm]
            metrics = _metric_pair(record["score"], truth_p, rare_p, fpr)
            for key in actual_keys:
                values[key].append(metrics[key])
        for key in actual_keys:
            null[key][p] = float(np.nanmean(values[key]))
    null95 = {key: float(np.nanpercentile(values, 95)) for key, values in null.items()}
    return {
        "actual": actual,
        "null95": null95,
        "excess": {
            "auc": max(0.0, actual["auc"] - null95["auc"]),
            "tpr": max(0.0, actual["tpr"] - null95["tpr"]),
            "auc_rare": max(0.0, actual["auc_rare"] - null95["auc_rare"]),
            "tpr_rare": max(0.0, actual["tpr_rare"] - null95["tpr_rare"]),
        },
        "per_split": [record["metrics"] for record in split_records],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    bench = Bench(args.dist)
    attacks: dict[str, Attack] = {
        "exact": AT.m2_exact_match,
        "nn_gap": AT.m3_nn_gap,
        "domias": AT.m1_domias,
        "knn": AT.m4_knn_density,
        "pref": AT.m5_pref_prior,
        "ensemble": AT.mia_ensemble,
        "shadow_c2st": lambda C, A, X: shadow_c2st(C, A, X, seed=args.seed + 7001),
        "multiview_tail": multiview_tail_linkage,
    }
    new_attacks = {name: attacks[name] for name in ("shadow_c2st", "multiview_tail")}
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        method: {name: [] for name in attacks} for method in FINAL_FILES
    }
    sanity: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        label: {"shadow_c2st": {}, "multiview_tail": {}}
        for label in ("identity", "null")
    }
    build_reports: list[dict[str, Any]] = []
    split_truth: list[np.ndarray] = []
    split_rare: list[np.ndarray] = []

    for split in range(args.splits):
        rng = np.random.default_rng(args.seed + split)
        index = rng.permutation(len(bench.B))
        out_index, in_index = index[: args.n_hold], index[args.n_hold :]
        B_in = bench.B.iloc[in_index].reset_index(drop=True)
        rare_in = bench.is_rare[in_index]
        candidates, build_report = _build_methods(bench, B_in, rare_in, split)
        build_report["n_holdout"] = int(len(out_index))
        build_report["true_rare_holdout"] = int(bench.is_rare[out_index].sum())
        build_reports.append(build_report)

        truth = np.zeros(len(bench.B), dtype=int)
        truth[in_index] = 1
        rare = bench.is_rare.copy()
        split_truth.append(truth)
        split_rare.append(rare)
        for method_index, (method, candidate) in enumerate(candidates.items()):
            for attack_index, (name, attack) in enumerate(attacks.items()):
                score, metrics = _evaluate_attack(
                    attack,
                    candidate,
                    bench,
                    truth,
                    rare,
                    args.seed + split * 10000 + method_index * 100 + attack_index,
                    args.fpr,
                )
                records[method][name].append({
                    "score": score,
                    "truth": truth,
                    "rare": rare,
                    "metrics": metrics,
                })

        # Endpoint validation for the new attacks: identity must detect and an
        # A_bg-only null candidate must not distinguish random B_in/B_out.
        identity = B_in[D.SUBMIT_COLS].copy().reset_index(drop=True)
        null_index = rng.choice(len(bench.A_bg), size=len(B_in), replace=False)
        null_candidate = bench.A_bg.iloc[null_index][D.SUBMIT_COLS].reset_index(drop=True)
        for label_index, (label, candidate) in enumerate(
            (("identity", identity), ("null", null_candidate))
        ):
            for attack_index, (name, attack) in enumerate(new_attacks.items()):
                score, metrics = _evaluate_attack(
                    attack,
                    candidate,
                    bench,
                    truth,
                    rare,
                    args.seed + 900000 + split * 100 + label_index * 10 + attack_index,
                    args.fpr,
                )
                sanity[label][name][split] = {
                    "score": score,
                    "truth": truth,
                    "rare": rare,
                    "metrics": metrics,
                }
        print(
            f"split {split + 1}/{args.splits} built and attacked "
            f"({build_report['elapsed_seconds']:.1f}s build)",
            flush=True,
        )

    permutation_rng = np.random.default_rng(args.seed + 424242)
    permutations: list[list[np.ndarray]] = [
        [permutation_rng.permutation(len(bench.B)) for _ in range(args.permutations)]
        for _ in range(args.splits)
    ]
    comparison = {
        method: {
            name: _aggregate_with_null(split_records, permutations, args.fpr)
            for name, split_records in method_records.items()
        }
        for method, method_records in records.items()
    }
    sanity_result: dict[str, dict[str, Any]] = {}
    for label, by_attack in sanity.items():
        sanity_result[label] = {}
        for name, by_split in by_attack.items():
            ordered = [by_split[split] for split in range(args.splits)]
            sanity_result[label][name] = _aggregate_with_null(ordered, permutations, args.fpr)

    candidate_root = args.candidate_root
    final_hashes = {
        method: _sha256(candidate_root / relative) for method, relative in FINAL_FILES.items()
    }
    return {
        "kind": "non_official_same_cohort_rebuild_mia_redteam",
        "official_a_mia_or_anon_computed": False,
        "official_rule_alignment": {
            "metric": "ROC-linear-interpolated TPR at FPR=0.01",
            "rare": "true-rare endpoint reported separately; official scorer takes max(overall, rare)",
            "differences": (
                "membership positives are all B_in rows rather than hidden 25-27 overlaps; "
                "results are averaged over rebuild splits; permutation count is diagnostic"
            ),
        },
        "parameters": {
            "split_seed_base": int(args.seed),
            "splits": int(args.splits),
            "n_train": int(len(bench.B) - args.n_hold),
            "n_holdout": int(args.n_hold),
            "low_fpr": float(args.fpr),
            "null_permutations": int(args.permutations),
            "tie_handling": "fixed independent 1e-12 jitter before ROC/AUC",
            "attacks": list(attacks),
            "new_attacks": {
                "shadow_c2st": (
                    "balanced ExtraTrees(240, leaf=4) + HistGradientBoosting(180, leaf=18) "
                    "C-vs-A_bg posterior"
                ),
                "multiview_tail": (
                    "k=3 local C/A_bg density ratio; five phenotype views; "
                    "A_bg percentile calibration; max over views"
                ),
            },
            "rebuild_pipeline": {
                "Pool": "seed5312 ftl=.90 det+4 knots128 polish4000 near_q=.15 refine150 + calibration",
                "SA": "same Pool base + seed7100 5000-step T=.0015->1e-5 outcome SA + calibration",
                "Copula": "seed6314 rare_shrink=.45 near_q=.05 moment=.5 refine300 + calibration",
            },
        },
        "dataset": {
            "dist": str(args.dist),
            "rows_B": int(len(bench.B)),
            "true_rare_B": int(bench.is_rare.sum()),
            "rows_A_bg": int(len(bench.A_bg)),
            "final_candidate_sha256": final_hashes,
            "driver_sha256": _sha256(Path(__file__).resolve()),
        },
        "builds": build_reports,
        "comparison": comparison,
        "new_attack_sanity": sanity_result,
        "limitations": [
            "This experiment estimates memorization signal, not official attack power or anonymity.",
            "Only three rebuild splits are used by default; the true-rare endpoint has few held-out rows.",
            "B_in contains many more members than an official attacker cohort, improving diagnostic power but changing variance.",
            "Candidate-specific utility rollback can select a different calibration lambda on a smaller B_in.",
            "Per-attack 95% null thresholds are not family-wise adjusted across attacks or methods.",
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
        raise ValueError("n_hold must lie in [1, 1048]")
    if args.permutations < 100:
        raise ValueError("at least 100 null permutations are required")
    if not 0.0 < args.fpr < 1.0:
        raise ValueError("fpr must lie in (0,1)")
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = run(args)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"red-team report: {args.out} ({report['elapsed_seconds']:.1f}s)")


if __name__ == "__main__":
    main()
