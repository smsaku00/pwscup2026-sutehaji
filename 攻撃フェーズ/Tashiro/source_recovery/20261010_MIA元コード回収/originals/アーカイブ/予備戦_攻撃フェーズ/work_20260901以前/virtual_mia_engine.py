"""MIA engine for target-centric virtual attack/defence games.

This module deliberately stops at an official-*like* sparse MIA diagnostic.  It
does not calculate the competition anonymity score.  A virtual attacker case
contains exactly the data that one real attacker column would contain:

* 1,049 candidate people;
* 25--27 members of the target team;
* 24 true-rare candidates, including 0--2 member positives.

The important aggregation order is implemented here rather than left to each
experiment driver::

    max(overall TPR@1% FPR, true-rare TPR@1% FPR) per attacker
    -> mean over attackers
    -> paired-label permutation null of that mean
    -> subtract its 95th percentile (while restoring the 1% chance floor)

The same deterministic permutation plan is reused when Pool/SA/Copula are
evaluated with the same attacker IDs.  Candidate differences are consequently
paired; the rare label always travels with the membership label during a null
permutation.

Public attacks are adapters around ``attacks.py``, ``mia_redteam.py`` and
``outcome_mia_redteam.py``.  The observable recovered-source part of
``whitebox_audit.py`` is also exposed.  Oracle leave-one-out attacks requiring
hidden generator traces are intentionally not mixed into the attacker average;
they can be supplied as extra diagnostic components by the caller.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors

import attacks as AT
import mia_redteam as MR
import outcome_mia_redteam as OM
import whitebox_audit as WB


EXPECTED_ROWS = 1049
EXPECTED_MEMBERS = (25, 27)
EXPECTED_TRUE_RARE = 24
EXPECTED_RARE_MEMBERS = (0, 2)
LOW_FPR = 0.01


@dataclass(frozen=True)
class VirtualAttackerCase:
    """One official-shaped attacker column against one target candidate."""

    attacker_id: str
    candidates: pd.DataFrame
    membership: np.ndarray
    true_rare: np.ndarray
    record_ids: np.ndarray | None = None

    def checked(self, *, strict_official_shape: bool = True) -> "VirtualAttackerCase":
        y = np.asarray(self.membership, dtype=int)
        rare = np.asarray(self.true_rare, dtype=bool)
        n = len(self.candidates)
        if y.shape != (n,) or rare.shape != (n,):
            raise ValueError(f"{self.attacker_id}: labels must both have length {n}")
        if not np.isin(y, (0, 1)).all():
            raise ValueError(f"{self.attacker_id}: membership must be binary")
        if self.record_ids is not None and np.asarray(self.record_ids).shape != (n,):
            raise ValueError(f"{self.attacker_id}: record_ids must have length {n}")
        if strict_official_shape:
            members = int(y.sum())
            rare_members = int(y[rare].sum())
            if n != EXPECTED_ROWS:
                raise ValueError(f"{self.attacker_id}: expected 1049 rows, got {n}")
            if not EXPECTED_MEMBERS[0] <= members <= EXPECTED_MEMBERS[1]:
                raise ValueError(
                    f"{self.attacker_id}: expected 25--27 positives, got {members}"
                )
            if int(rare.sum()) != EXPECTED_TRUE_RARE:
                raise ValueError(
                    f"{self.attacker_id}: expected 24 true-rare rows, got {int(rare.sum())}"
                )
            if not EXPECTED_RARE_MEMBERS[0] <= rare_members <= EXPECTED_RARE_MEMBERS[1]:
                raise ValueError(
                    f"{self.attacker_id}: expected 0--2 rare positives, got {rare_members}"
                )
        return self


@dataclass(frozen=True)
class AttackProfile:
    """A truth-blind score rule used by one virtual attacker team."""

    name: str
    weights: Mapping[str, float]
    fallback: str | None = None
    exact_gate: bool = False
    note: str = ""


@dataclass(frozen=True)
class VirtualMIAConfig:
    low_fpr: float = LOW_FPR
    permutations: int = 5000
    null_quantile: float = 0.95
    seed: int = 20260812
    strict_official_shape: bool = True
    permutation_batch: int = 256

    def checked(self) -> "VirtualMIAConfig":
        if not 0.0 < self.low_fpr < 1.0:
            raise ValueError("low_fpr must lie in (0,1)")
        if self.permutations < 100:
            raise ValueError("at least 100 permutations are required")
        if not 0.5 < self.null_quantile < 1.0:
            raise ValueError("null_quantile must lie in (0.5,1)")
        if self.permutation_batch < 1:
            raise ValueError("permutation_batch must be positive")
        return self


DEFAULT_PROFILES: dict[str, AttackProfile] = {
    "classic_generalist": AttackProfile(
        "classic_generalist",
        {"cov_exact": 3.0, "cov_nn_gap": 1.0, "cov_knn": 1.0,
         "domias": 0.7, "pref_prior": 0.5},
        exact_gate=True,
        note="existing rank ensemble; observable exact-match gate",
    ),
    "covariate_quick": AttackProfile(
        "covariate_quick",
        {"cov_nn_gap": 1.0, "cov_knn": 1.0, "domias": 0.8,
         "multiview_tail": 1.0},
        fallback="classic_generalist",
        note="classifier-free covariate screen",
    ),
    "covariate_density": AttackProfile(
        "covariate_density",
        {"cov_nn_gap": 1.0, "cov_knn": 1.0, "domias": 0.8,
         "shadow_c2st": 1.2, "multiview_tail": 1.0},
        fallback="covariate_quick",
        note="smooth/local C-vs-A density and multiview linkage",
    ),
    "outcome_quick": AttackProfile(
        "outcome_quick",
        {"event_time_near": 1.2, "joint_nn_gap": 1.0, "joint_knn": 1.0,
         "survival_likelihood": 1.0, "cox_residual": 0.8},
        fallback="classic_generalist",
        note="classifier-free outcome/survival screen",
    ),
    "outcome_generalist": AttackProfile(
        "outcome_generalist",
        {"event_time_near": 1.2, "joint_nn_gap": 1.0, "joint_knn": 1.0,
         "joint_shadow_c2st": 1.0, "survival_likelihood": 1.0,
         "cox_residual": 0.8},
        fallback="outcome_quick",
        note="time/event/joint/survival family; no hidden membership tuning",
    ),
    "survival_specialist": AttackProfile(
        "survival_specialist",
        {"event_time_exact": 0.8, "event_time_near": 1.2,
         "survival_likelihood": 1.4, "cox_residual": 1.0},
        fallback="outcome_generalist",
        note="cause-specific event-time and Cox model leakage",
    ),
    "whitebox_recovered": AttackProfile(
        "whitebox_recovered",
        {"wb_recovered_gap": 1.4, "wb_selected_negative": 0.8,
         "cov_nn_gap": 0.5},
        fallback="classic_generalist",
        note=("observable public-source reconstruction for Pool-like generators; "
              "not the oracle LOO bound"),
    ),
}

POPULATION_PROFILES = (
    "classic_generalist",
    "covariate_density",
    "outcome_generalist",
    "survival_specialist",
    "whitebox_recovered",
)


QUICK_COMPONENTS = frozenset({
    "cov_exact", "cov_nn_gap", "cov_knn", "pref_prior", "domias", "multiview_tail",
    "event_time_exact", "event_time_near", "joint_nn_gap", "joint_knn",
    "survival_likelihood", "cox_residual",
})
FULL_COMPONENTS = QUICK_COMPONENTS | {
    "domias", "shadow_c2st", "joint_shadow_c2st",
    "wb_recovered_gap", "wb_selected_negative",
}


def _stable_seed(base: int, *parts: object) -> int:
    raw = "\x1f".join([str(base), *(str(x) for x in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little")


def _sklearn_seed(seed: int) -> int:
    """Keep a deterministic seed in sklearn's uint32 range, with room for +1."""
    return int(seed % (2**32 - 2))


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("attack component contains non-finite scores")
    return (rankdata(values, method="average") - 0.5) / len(values)


def _domias_chunked(
    C: pd.DataFrame,
    A: pd.DataFrame,
    X: pd.DataFrame,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    """The existing DOMIAS score with one KDE fit and bounded query memory."""
    mean, std = AT._scaler(A, AT.CONT)
    try:
        kde_c = gaussian_kde(AT._z(C, AT.CONT, mean, std).T)
        kde_a = gaussian_kde(AT._z(A, AT.CONT, mean, std).T)
    except np.linalg.LinAlgError:
        return np.full(len(X), 0.5, dtype=float)
    query = AT._z(X, AT.CONT, mean, std).T
    result = np.empty(len(X), dtype=float)
    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        ratio = kde_c(query[:, start:stop]) / np.clip(
            kde_a(query[:, start:stop]), 1e-300, None
        )
        result[start:stop] = ratio / (1.0 + ratio)
    return result


def _whitebox_recovered_components(
    C: pd.DataFrame,
    A: pd.DataFrame,
    X: pd.DataFrame,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Observable subset of the existing white-box source-selection attack.

    Recovered A_bg provenance is public information.  The score asks whether a
    query resembles public rows the Pool-like generator systematically did not
    select.  It does *not* use the captured true source indices or target B.
    """
    _indices, _distances, selected = WB._link_source_rows(A, C, k=100)
    selected = np.unique(np.asarray(selected, dtype=int))
    center = A[WB.CONT].mean().to_numpy(dtype=float)
    scale = A[WB.CONT].std().to_numpy(dtype=float)
    scale = np.where(scale > 0.0, scale, 1.0)
    xa = WB._scaled(A, center, scale)
    xx = WB._scaled(X, center, scale)
    d_selected = WB._knn_mean(xa[selected], xx, 1)
    d_all = WB._knn_mean(xa, xx, 1)
    gap = d_selected - d_all

    mask = np.zeros(len(A), dtype=int)
    mask[selected] = 1
    if len(np.unique(mask)) < 2:
        selected_negative = np.zeros(len(X), dtype=float)
    else:
        model = ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=12,
            max_features=0.8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
        model.fit(xa, mask)
        selected_negative = -model.predict_proba(xx)[:, 1]
    return {"wb_recovered_gap": gap, "wb_selected_negative": selected_negative}


def score_public_components(
    C: pd.DataFrame,
    A_bg: pd.DataFrame,
    cases: Sequence[VirtualAttackerCase],
    *,
    tier: str = "full",
    seed: int = 20260812,
    components: Iterable[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit each public attack once to C/A_bg and score all attackers in one batch.

    Returned scores remain raw.  Rank conversion is deliberately performed
    separately inside each attacker cohort when a profile is composed.
    """
    checked = [case.checked(strict_official_shape=False) for case in cases]
    if not checked:
        raise ValueError("at least one attacker case is required")
    if len({case.attacker_id for case in checked}) != len(checked):
        raise ValueError("attacker_id values must be unique")
    if tier not in {"quick", "full"}:
        raise ValueError("tier must be 'quick' or 'full'")
    requested = set(QUICK_COMPONENTS if tier == "quick" else FULL_COMPONENTS)
    if components is not None:
        requested &= set(components)
    X = pd.concat([case.candidates for case in checked], ignore_index=True)
    raw: dict[str, np.ndarray] = {}

    simple = {
        "cov_exact": AT.m2_exact_match,
        "cov_nn_gap": AT.m3_nn_gap,
        "cov_knn": AT.m4_knn_density,
        "pref_prior": AT.m5_pref_prior,
        "multiview_tail": MR.multiview_tail_linkage,
        "event_time_exact": OM.event_time_exact,
        "event_time_near": OM.event_time_near,
        "joint_nn_gap": OM.joint_nn_gap,
        "joint_knn": OM.joint_knn,
    }
    for name, attack in simple.items():
        if name in requested:
            raw[name] = np.asarray(attack(C, A_bg, X), dtype=float)
    if "domias" in requested:
        raw["domias"] = _domias_chunked(C, A_bg, X)
    if "shadow_c2st" in requested:
        raw["shadow_c2st"] = MR.shadow_c2st(
            C, A_bg, X, seed=_sklearn_seed(seed + 101)
        )
    if "joint_shadow_c2st" in requested:
        raw["joint_shadow_c2st"] = OM.joint_shadow_c2st(
            C, A_bg, X, seed=_sklearn_seed(seed + 202)
        )
    if {"survival_likelihood", "cox_residual"} & requested:
        likelihood, residual = OM.survival_scores(C, A_bg, X)
        if "survival_likelihood" in requested:
            raw["survival_likelihood"] = likelihood
        if "cox_residual" in requested:
            raw["cox_residual"] = residual
    if {"wb_recovered_gap", "wb_selected_negative"} & requested:
        whitebox = _whitebox_recovered_components(
            C, A_bg, X, seed=_sklearn_seed(seed + 303)
        )
        raw.update({name: value for name, value in whitebox.items() if name in requested})

    for name, values in raw.items():
        if values.shape != (len(X),) or not np.isfinite(values).all():
            raise ValueError(f"invalid output from component {name}: {values.shape}")
    result: dict[str, dict[str, np.ndarray]] = {}
    offset = 0
    for case in checked:
        stop = offset + len(case.candidates)
        result[case.attacker_id] = {
            name: values[offset:stop].copy() for name, values in raw.items()
        }
        offset = stop
    return result


def compose_profile_score(
    components: Mapping[str, np.ndarray],
    profile: AttackProfile | str,
    *,
    profiles: Mapping[str, AttackProfile] = DEFAULT_PROFILES,
) -> np.ndarray:
    """Combine components using only within-attacker ranks.

    Missing components trigger the declared fallback.  This lets a cheap first
    pass omit classifiers without silently changing a profile's weights.
    """
    if isinstance(profile, str):
        if profile not in profiles:
            raise KeyError(f"unknown attack profile: {profile}")
        profile = profiles[profile]
    missing = set(profile.weights) - set(components)
    if missing:
        if profile.fallback is None:
            raise KeyError(f"{profile.name}: missing components {sorted(missing)}")
        if profile.fallback == profile.name:
            raise ValueError(f"{profile.name}: recursive fallback")
        return compose_profile_score(components, profile.fallback, profiles=profiles)

    # Preserve the observable shortcut already used by attacks.mia_ensemble.
    if profile.exact_gate:
        exact = np.asarray(components["cov_exact"], dtype=float)
        if exact.max(initial=0.0) > 0.5 and float(np.mean(exact > 0.5)) < 0.5:
            return 0.98 * _rank01(exact) + 0.02 * _rank01(components["cov_nn_gap"])
    total = float(sum(profile.weights.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{profile.name}: weights must have a positive finite sum")
    return sum(
        float(weight) * _rank01(components[name])
        for name, weight in profile.weights.items()
    ) / total


def balanced_profile_assignment(
    attacker_ids: Sequence[str],
    *,
    profile_names: Sequence[str] = POPULATION_PROFILES,
    seed: int = 20260812,
) -> dict[str, str]:
    """Truth-blind, approximately balanced virtual-team profile assignment."""
    if not profile_names:
        raise ValueError("profile_names must not be empty")
    ids = list(attacker_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    return {attacker: profile_names[i % len(profile_names)] for i, attacker in enumerate(ids)}


def _tie_safe(values: np.ndarray, *, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    if not np.isfinite(values).all():
        raise ValueError("attack scores must be finite")
    rng = np.random.default_rng(seed)
    return values + rng.uniform(-0.5, 0.5, len(values)) * (np.ptp(values) + 1.0) * 1e-12


def _auc(values: np.ndarray, truth: np.ndarray) -> float:
    y = np.asarray(truth, dtype=int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(values, method="average")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _tpr_at(values: np.ndarray, truth: np.ndarray, fpr: float) -> float:
    order = np.argsort(-np.asarray(values, dtype=float), kind="mergesort")
    y = np.asarray(truth, dtype=int)[order]
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return 0.0
    tp = np.concatenate([[0.0], np.cumsum(y, dtype=float) / n1])
    fp = np.concatenate([[0.0], np.cumsum(1 - y, dtype=float) / n0])
    return float(np.interp(fpr, fp, tp))


def _one_attacker_metrics(
    score: np.ndarray,
    membership: np.ndarray,
    rare: np.ndarray,
    fpr: float,
) -> dict[str, float | None]:
    y = np.asarray(membership, dtype=int)
    r = np.asarray(rare, dtype=bool)
    all_tpr = _tpr_at(score, y, fpr)
    rare_tpr = _tpr_at(score[r], y[r], fpr) if r.any() and y[r].sum() else 0.0
    rare_auc = _auc(score[r], y[r])
    return {
        "tpr_all": all_tpr,
        "tpr_true_rare": rare_tpr,
        "tpr_worst": max(all_tpr, rare_tpr),
        "auc_all": _auc(score, y),
        "auc_true_rare": rare_auc if np.isfinite(rare_auc) else None,
    }


def _null_worst_tpr(
    score: np.ndarray,
    membership: np.ndarray,
    rare: np.ndarray,
    *,
    fpr: float,
    permutations: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Paired-label null for one attacker, batched without repeated ROC sorts."""
    rng = np.random.default_rng(seed)
    y = np.asarray(membership, dtype=int)
    r = np.asarray(rare, dtype=bool)
    out = np.empty(permutations, dtype=float)
    score_order = np.argsort(-score, kind="mergesort")

    def ordered_batch_tpr(ordered_y: np.ndarray) -> np.ndarray:
        """Match np.interp ROC semantics at a fixed FPR for many label rows."""
        n_pos = ordered_y.sum(axis=1)
        n_neg = ordered_y.shape[1] - n_pos
        # np.interp uses the last vertical ROC point at a repeated FPR.  This
        # equals counting positives seen before the next false positive beyond
        # floor(fpr*n_neg).
        allowed = np.floor(fpr * n_neg + 1e-12).astype(int)
        false_seen = np.cumsum(1 - ordered_y, axis=1)
        caught = np.sum(ordered_y * (false_seen <= allowed[:, None]), axis=1)
        return np.divide(caught, n_pos, out=np.zeros_like(caught, dtype=float), where=n_pos > 0)

    # Permuting one shared index is what keeps (membership, true_rare) paired.
    for start in range(0, permutations, batch_size):
        stop = min(start + batch_size, permutations)
        indices = np.stack([rng.permutation(len(y)) for _ in range(stop - start)])
        yp = y[indices][:, score_order]
        rp = r[indices][:, score_order]
        all_tpr = ordered_batch_tpr(yp)
        # Every permutation preserves the rare count, so boolean extraction is
        # rectangular after reshape and remains in descending-score order.
        rare_y = yp[rp].reshape(stop - start, int(r.sum()))
        rare_tpr = ordered_batch_tpr(rare_y)
        out[start:stop] = np.maximum(all_tpr, rare_tpr)
    return out


def evaluate_attacker_average(
    cases: Sequence[VirtualAttackerCase],
    scores: Mapping[str, np.ndarray],
    *,
    config: VirtualMIAConfig = VirtualMIAConfig(),
    score_rule_id: str = "attack_profile",
) -> dict:
    """Official-shaped attacker mean with paired permutation chance correction.

    The returned ``chance_corrected_beta`` is kept on the TPR scale.  No
    a_MIA/Anon transformation is performed.
    """
    config = config.checked()
    checked = [case.checked(strict_official_shape=config.strict_official_shape) for case in cases]
    if set(scores) != {case.attacker_id for case in checked}:
        raise ValueError("scores keys must exactly equal attacker IDs")
    actual: dict[str, dict[str, float]] = {}
    null_rows: list[np.ndarray] = []
    for case in checked:
        attack_score = _tie_safe(
            np.asarray(scores[case.attacker_id], dtype=float),
            seed=_stable_seed(config.seed, "tie", score_rule_id, case.attacker_id),
        )
        if attack_score.shape != (len(case.candidates),):
            raise ValueError(f"{case.attacker_id}: invalid score length")
        actual[case.attacker_id] = _one_attacker_metrics(
            attack_score, case.membership, case.true_rare, config.low_fpr
        )
        # The seed intentionally omits candidate/method identity.  Evaluating
        # Pool, SA and Copula with the same IDs therefore reuses label pairings.
        null_rows.append(_null_worst_tpr(
            attack_score,
            case.membership,
            case.true_rare,
            fpr=config.low_fpr,
            permutations=config.permutations,
            seed=_stable_seed(config.seed, "paired-null", case.attacker_id),
            batch_size=config.permutation_batch,
        ))
    beta_raw = float(np.mean([row["tpr_worst"] for row in actual.values()]))
    # Correct order: average attackers within each permutation, then take q95.
    null_mean = np.mean(np.vstack(null_rows), axis=0)
    beta_chance = float(np.quantile(null_mean, config.null_quantile))
    excess = float(max(0.0, beta_raw - beta_chance))
    chance_corrected = float(max(config.low_fpr, excess + config.low_fpr))
    rare_aucs = [
        row["auc_true_rare"] for row in actual.values()
        if row["auc_true_rare"] is not None
    ]
    return {
        "score_rule_id": score_rule_id,
        "n_attackers": len(checked),
        "low_fpr": float(config.low_fpr),
        "beta_raw_attacker_mean": beta_raw,
        "beta_chance_q95_of_attacker_mean": beta_chance,
        "tpr_excess_over_q95": excess,
        "chance_corrected_beta": chance_corrected,
        "mean_tpr_all": float(np.mean([row["tpr_all"] for row in actual.values()])),
        "mean_tpr_true_rare": float(
            np.mean([row["tpr_true_rare"] for row in actual.values()])
        ),
        "mean_auc_all": float(np.nanmean([row["auc_all"] for row in actual.values()])),
        "mean_auc_true_rare": (
            float(np.mean(rare_aucs)) if rare_aucs else None
        ),
        "null": {
            "permutations": int(config.permutations),
            "pairing": "membership and true_rare moved by the same permutation",
            "mean": float(np.mean(null_mean)),
            "sd": float(np.std(null_mean)),
            "q50": float(np.quantile(null_mean, 0.50)),
            "q95": beta_chance,
            "q99": float(np.quantile(null_mean, 0.99)),
        },
        "per_attacker": actual,
        "official_anonymity_conversion_computed": False,
    }


def evaluate_profile_population(
    cases: Sequence[VirtualAttackerCase],
    component_scores: Mapping[str, Mapping[str, np.ndarray]],
    assignment: Mapping[str, str],
    *,
    profiles: Mapping[str, AttackProfile] = DEFAULT_PROFILES,
    config: VirtualMIAConfig = VirtualMIAConfig(),
) -> dict:
    """Give every virtual team one pre-assigned strategy, then average teams."""
    ids = {case.attacker_id for case in cases}
    if set(component_scores) != ids or set(assignment) != ids:
        raise ValueError("cases, component_scores and assignment must share attacker IDs")
    scores = {
        attacker_id: compose_profile_score(
            component_scores[attacker_id], assignment[attacker_id], profiles=profiles
        )
        for attacker_id in ids
    }
    rule_hash = hashlib.sha256(
        "|".join(f"{key}:{assignment[key]}" for key in sorted(assignment)).encode()
    ).hexdigest()[:12]
    report = evaluate_attacker_average(
        cases,
        scores,
        config=config,
        score_rule_id=f"profile_population:{rule_hash}",
    )
    report["profile_assignment"] = dict(sorted(assignment.items()))
    report["profile_counts"] = {
        name: int(sum(value == name for value in assignment.values()))
        for name in sorted(set(assignment.values()))
    }
    return report


def merge_extra_components(
    base: Mapping[str, Mapping[str, np.ndarray]],
    extra: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Attach external diagnostics, e.g. an explicitly labelled oracle score."""
    if set(base) != set(extra):
        raise ValueError("base and extra must share attacker IDs")
    result: dict[str, dict[str, np.ndarray]] = {}
    for attacker in base:
        overlap = set(base[attacker]) & set(extra[attacker])
        if overlap:
            raise ValueError(f"duplicate components for {attacker}: {sorted(overlap)}")
        result[attacker] = {
            **{k: np.asarray(v, dtype=float) for k, v in base[attacker].items()},
            **{k: np.asarray(v, dtype=float) for k, v in extra[attacker].items()},
        }
    return result


def sanity_contract() -> dict[str, str]:
    """Machine-readable checklist expected of a complete game driver."""
    return {
        "shape": "each attacker: n=1049, positives=25..27, true_rare=24, rare positives=0..2",
        "same_cohort_null": "non-members come from the same cohort generator, never bare A_bg",
        "paired_methods": "Pool/SA/Copula reuse target B, attacker pools, truth, rare and seeds",
        "identity_positive": "identity/copy candidate must trigger exact and NN attacks",
        "independent_null": "independent same-distribution candidate must have zero q95 excess",
        "constant_score": "constant score must have zero q95 excess",
        "row_shuffle": "joint row/label shuffle must not change aggregate metrics",
        "monotone": "strictly monotone score transforms must not change ROC metrics",
        "outcome_ablation": "time/event shuffle should remove outcome-family advantage",
        "whitebox_scope": "observable recovered-source and oracle LOO are reported separately",
        "truth_blind_selection": "profiles/weights are frozen or selected on disjoint shadow games",
        "finite": "all component arrays are finite and record_id/token are never model features",
    }


__all__ = [
    "AttackProfile",
    "DEFAULT_PROFILES",
    "FULL_COMPONENTS",
    "POPULATION_PROFILES",
    "QUICK_COMPONENTS",
    "VirtualAttackerCase",
    "VirtualMIAConfig",
    "balanced_profile_assignment",
    "compose_profile_score",
    "evaluate_attacker_average",
    "evaluate_profile_population",
    "merge_extra_components",
    "sanity_contract",
    "score_public_components",
]
