"""White-box provenance and leave-one-out audit for the three Team05 candidates.

This is a sensitivity experiment, not an implementation of the official anonymity
score.  It asks two deliberately stronger questions than the public black-box
attacks:

1. Can each PoolMatch/SA output row be linked back to the public A_bg row used as
   its material?
2. After the public source set has been recovered, does that set reveal which
   same-cohort B rows were used to fit/select it?  The latter is measured on
   repeated B_in/B_out splits, including a targeted leave-one-out likelihood
   ratio for the 24 true-rare records.

The targeted likelihood ratio is an *oracle upper bound*: it gives the attacker
all training rows except the target and one same-class replacement.  It must not
be read as an attack available from C alone.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

import audit as AU
import defense as D
from harness import Bench
import third_defense as TD


CONT = list(D.CONT)
CLIN = list(D.CLIN)


def _fake_outcomes(C: pd.DataFrame, B: pd.DataFrame, rng: np.random.Generator,
                   *args: Any, **kwargs: Any) -> pd.DataFrame:
    out = C.copy()
    out["time"] = 10.0
    out["onset"] = 0
    out["death"] = 0
    return out


def _build_capture(bench: Bench, B: pd.DataFrame, rare: np.ndarray, seed: int,
                   polish_steps: int = 4000) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Run PoolMatch while capturing the final public-source indices."""
    cap: dict[str, Any] = {}
    old_repair, old_outcome = D.repair_rare, D.make_outcomes

    def capture_repair(C: pd.DataFrame, Apool: pd.DataFrame, sel: np.ndarray,
                       *args: Any, **kwargs: Any):
        out, final_sel = old_repair(C, Apool, sel, *args, **kwargs)
        cap["Apool"] = Apool.copy()
        cap["sel"] = np.asarray(final_sel, dtype=int).copy()
        return out, final_sel

    try:
        D.repair_rare = capture_repair
        D.make_outcomes = _fake_outcomes
        det_target = int((D.maha(B, bench.ref.mu, bench.ref.sigma)
                          >= float(bench.ref.band["lo"])).sum()) + 4
        C = D.build_C(
            B, bench.A_bg, rare, bench.ref.mu, bench.ref.sigma,
            float(bench.ref.band["lo"]), seed=seed, n_knots=128,
            frac_true_like=0.9, det_target=det_target, jitter=0.25,
            polish_steps=polish_steps, min_dist_q=0.15, rare_model="logit",
        )
    finally:
        D.repair_rare, D.make_outcomes = old_repair, old_outcome
    return C, cap["Apool"], cap["sel"]


def _map_A_to_C(A: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """Attacker-side monotone reconstruction of PoolMatch's transformed pool."""
    T = A[CONT + ["sex", "smoking"]].copy()
    for col in CLIN:
        T[col] = D.quantile_align(
            A[col].to_numpy(float), C[col].to_numpy(float), 64,
            np.random.default_rng(0), 0.0,
        )
    T["age"] = np.clip(np.rint(D.quantile_align(
        A["age"].to_numpy(float), C["age"].to_numpy(float), 40,
        np.random.default_rng(1), 0.0,
    )), *D.AGE_RANGE).astype(int)
    return T


def _link_source_rows(A: pd.DataFrame, C: pd.DataFrame, k: int = 100
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return top-k A candidates per C row and a size-|C| source-set estimate."""
    T = _map_A_to_C(A, C)
    sd = C[CLIN].std().to_numpy(float)
    sd = np.where(sd > 0, sd, 1.0)
    xa = np.column_stack([
        T[CLIN].to_numpy(float) / sd,
        T["sex"].to_numpy(float) * 20.0,
        T["smoking"].to_numpy(float) * 20.0,
    ])
    xc = np.column_stack([
        C[CLIN].to_numpy(float) / sd,
        C["sex"].to_numpy(float) * 20.0,
        C["smoking"].to_numpy(float) * 20.0,
    ])
    dist, ind = NearestNeighbors(n_neighbors=min(k, len(A))).fit(xa).kneighbors(xc)
    # A source-set estimate: lexicographically sort by its best row-wise rank,
    # then distance, and retain exactly |C| public rows.
    min_rank = np.full(len(A), k + 1, dtype=int)
    min_dist = np.full(len(A), np.inf, dtype=float)
    for i in range(len(C)):
        for rank, (j, d) in enumerate(zip(ind[i], dist[i])):
            if rank < min_rank[j] or (rank == min_rank[j] and d < min_dist[j]):
                min_rank[j], min_dist[j] = rank, d
    order = np.lexsort((min_dist, min_rank))
    recovered = order[:len(C)]
    return ind, dist, recovered


def _source_metrics(ind: np.ndarray, recovered: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=int)
    row = {}
    for k in (1, 5, 20, 50):
        kk = min(k, ind.shape[1])
        row[f"top{k}"] = float(np.mean([truth[i] in ind[i, :kk] for i in range(len(truth))]))
    ts, rs = set(truth.tolist()), set(recovered.tolist())
    hit = len(ts & rs)
    return {
        "row_link": row,
        "set_n": int(len(rs)),
        "set_precision": float(hit / max(len(rs), 1)),
        "set_recall": float(hit / max(len(ts), 1)),
    }


def _scaled(frame: pd.DataFrame, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = (frame[CONT].to_numpy(float) - center) / scale
    return np.column_stack([
        x,
        frame["sex"].to_numpy(float) * 2.0,
        frame["smoking"].to_numpy(float) * 2.0,
    ])


def _knn_mean(ref: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    kk = min(k, len(ref))
    return NearestNeighbors(n_neighbors=kk).fit(ref).kneighbors(query)[0].mean(axis=1)


def _artifact_features(A: pd.DataFrame, Apool: pd.DataFrame, selected: np.ndarray,
                       Bcand: pd.DataFrame, det_pool: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Candidate scores derivable from a hypothesized public source set."""
    selected = np.unique(np.asarray(selected, dtype=int))
    mask = np.zeros(len(A), dtype=bool)
    mask[selected] = True
    cen = A[CONT].mean().to_numpy(float)
    scale = A[CONT].std().to_numpy(float)
    scale = np.where(scale > 0, scale, 1.0)
    xa, xb = _scaled(A, cen, scale), _scaled(Bcand, cen, scale)
    xsel = xa[mask]
    raw1 = _knn_mean(xsel, xb, 1) - _knn_mean(xa, xb, 1)
    raw5 = _knn_mean(xsel, xb, 5) - _knn_mean(xa, xb, 5)

    # Near-twin filtering predicts *large* selected-source distance for members.
    # A smooth selected-vs-unselected classifier tests the other selection rules.
    clf = ExtraTreesClassifier(
        n_estimators=100, min_samples_leaf=12, max_features=0.8,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    clf.fit(xa, mask.astype(int))
    psel = clf.predict_proba(xb)[:, 1]

    cp = Apool[CONT].mean().to_numpy(float)
    sp = Apool[CONT].std().to_numpy(float)
    sp = np.where(sp > 0, sp, 1.0)
    xp, xbp = _scaled(Apool, cp, sp), _scaled(Bcand, cp, sp)
    pool1 = _knn_mean(xp[mask], xbp, 1) - _knn_mean(xp, xbp, 1)

    dr = np.asarray(det_pool, dtype=bool)
    sel_rare = mask & dr
    if sel_rare.sum() >= 2 and dr.sum() >= 2:
        rare1 = _knn_mean(xp[sel_rare], xbp, 1) - _knn_mean(xp[dr], xbp, 1)
        rare5 = _knn_mean(xp[sel_rare], xbp, 5) - _knn_mean(xp[dr], xbp, 5)
    else:
        rare1 = np.zeros(len(Bcand))
        rare5 = np.zeros(len(Bcand))
    return {
        "raw_nn_exclusion": raw1,
        "raw_knn5_exclusion": raw5,
        "pool_nn_exclusion": pool1,
        "selected_classifier": psel,
        "selected_classifier_negative": -psel,
        "rare_nn_exclusion": rare1,
        "rare_knn5_exclusion": rare5,
    }


def _make_apool(A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    Apool = A[CONT + ["sex", "smoking"]].copy()
    for col in CLIN:
        Apool[col] = D.quantile_align(
            A[col].to_numpy(float), B[col].to_numpy(float), 128,
            np.random.default_rng(999), 0.0,
        )
    Apool["age"] = np.clip(np.rint(D.quantile_align(
        A["age"].to_numpy(float), B["age"].to_numpy(float), 40,
        np.random.default_rng(998), 0.0,
    )), *D.AGE_RANGE).astype(int)
    return Apool


def _selection_loglik(A: pd.DataFrame, B: pd.DataFrame, rare: np.ndarray,
                      selected: np.ndarray, bench: Bench) -> float:
    """Approximate detected-source likelihood under one membership hypothesis."""
    Apool = _make_apool(A, B)
    det = D.maha(Apool, bench.ref.mu, bench.ref.sigma) >= float(bench.ref.band["lo"])
    p = D.rare_type_score(B, rare, Apool, model="logit")
    dmin = D.min_dist_to_B(Apool, B)
    keep = dmin >= np.quantile(dmin, 0.15)
    idx = np.flatnonzero(det & keep)
    if len(idx) == 0:
        return -np.inf
    threshold = max(float(np.quantile(p[idx], 0.5)), 0.15)
    true_like = p[idx] >= threshold
    n_det = int(det.sum()) + 4
    n_tl = min(int(round(n_det * 0.9)), int(true_like.sum()))
    n_ml = min(n_det - n_tl, int((~true_like).sum()))
    q = np.where(
        true_like,
        n_tl / max(int(true_like.sum()), 1),
        n_ml / max(int((~true_like).sum()), 1),
    )
    q = np.clip(q, 1e-8, 1.0 - 1e-8)
    sm = np.zeros(len(A), dtype=bool)
    sm[np.asarray(selected, dtype=int)] = True
    y = sm[idx].astype(float)
    return float(np.sum(y * np.log(q) + (1.0 - y) * np.log1p(-q)))


def _targeted_rare_scores(A: pd.DataFrame, B: pd.DataFrame, true_rare: np.ndarray,
                          in_idx: np.ndarray, selected: np.ndarray,
                          bench: Bench) -> tuple[np.ndarray, np.ndarray]:
    """Oracle LOO likelihood ratio for all 24 true-rare targets in one split."""
    rare_idx = np.flatnonzero(true_rare)
    in_set = set(np.asarray(in_idx, dtype=int).tolist())
    scores, labels = [], []
    for x in rare_idx:
        member = x in in_set
        actual = list(np.asarray(in_idx, dtype=int))
        if member:
            alternatives = [j for j in rare_idx if j not in in_set and j != x]
            if not alternatives:
                continue
            z = min(alternatives, key=lambda j: abs(float(B.iloc[j].age) - float(B.iloc[x].age)))
            h1 = actual
            h0 = [z if j == x else j for j in actual]
        else:
            alternatives = [j for j in rare_idx if j in in_set and j != x]
            if not alternatives:
                continue
            z = min(alternatives, key=lambda j: abs(float(B.iloc[j].age) - float(B.iloc[x].age)))
            h0 = actual
            h1 = [x if j == z else j for j in actual]
        b0, b1 = B.iloc[h0].reset_index(drop=True), B.iloc[h1].reset_index(drop=True)
        r0, r1 = true_rare[h0], true_rare[h1]
        ll0 = _selection_loglik(A, b0, r0, selected, bench)
        ll1 = _selection_loglik(A, b1, r1, selected, bench)
        scores.append(ll1 - ll0)
        labels.append(int(member))
    return np.asarray(scores, float), np.asarray(labels, int)


def _metrics(score: np.ndarray, y: np.ndarray, groups: np.ndarray,
             rare: np.ndarray, rng: np.random.Generator, n_perm: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, mask in (("all", np.ones(len(y), bool)), ("true_rare", rare)):
        yy, ss, gg = y[mask], score[mask], groups[mask]
        if len(np.unique(yy)) < 2:
            continue
        tpr = AU._tpr_at(ss, yy, 0.01)
        auc = roc_auc_score(yy, ss)
        null = []
        for _ in range(n_perm):
            yp = yy.copy()
            for g in np.unique(gg):
                m = gg == g
                yp[m] = rng.permutation(yp[m])
            null.append(AU._tpr_at(ss, yp, 0.01))
        out[label] = {
            "n": int(len(yy)), "positive": int(yy.sum()),
            "auc": float(auc), "tpr_at_1pct": float(tpr),
            "null95_tpr": float(np.quantile(null, 0.95)),
            "tpr_excess": float(max(0.0, tpr - np.quantile(null, 0.95))),
        }
    return out


def _model_loo_sensitivity(bench: Bench, Apool: pd.DataFrame) -> dict[str, Any]:
    """Record-level parameter leverage, separate from membership exploitability."""
    B, rare = bench.B, bench.is_rare
    p0 = D.rare_type_score(B, rare, Apool, model="logit")
    det = D.maha(Apool, bench.ref.mu, bench.ref.sigma) >= float(bench.ref.band["lo"])
    thr0 = max(float(np.quantile(p0[det], 0.5)), 0.15)
    base_group = p0 >= thr0
    rng = np.random.default_rng(20260812)
    maj_sample = rng.choice(np.flatnonzero(~rare), size=int(rare.sum()), replace=False)

    def pool_changes(indices: np.ndarray) -> list[dict[str, float]]:
        vals = []
        for j in indices:
            keep = np.arange(len(B)) != int(j)
            p = D.rare_type_score(B.iloc[keep].reset_index(drop=True), rare[keep], Apool, model="logit")
            thr = max(float(np.quantile(p[det], 0.5)), 0.15)
            vals.append({
                "mean_abs_p": float(np.mean(np.abs(p - p0))),
                "max_abs_p": float(np.max(np.abs(p - p0))),
                "det_group_flip_rate": float(np.mean((p[det] >= thr) != base_group[det])),
            })
        return vals

    z0, _ = TD._latent_reference(B)
    normal0, rare0 = TD._fit_components(z0, rare, 0.45)

    def copula_changes(indices: np.ndarray) -> list[dict[str, float]]:
        vals = []
        for j in indices:
            keep = np.arange(len(B)) != int(j)
            z, _ = TD._latent_reference(B.iloc[keep].reset_index(drop=True))
            _normal, rr = TD._fit_components(z, rare[keep], 0.45)
            vals.append({
                "rare_mu_l2": float(np.linalg.norm(rr[0] - rare0[0])),
                "rare_cov_relative_fro": float(
                    np.linalg.norm(rr[1] - rare0[1]) / max(np.linalg.norm(rare0[1]), 1e-12)
                ),
            })
        return vals

    def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        return {
            key: {"median": float(np.median([r[key] for r in rows])),
                  "max": float(np.max([r[key] for r in rows]))}
            for key in rows[0]
        }

    return {
        "PoolMatch_rare_logit": {
            "remove_true_rare": summarize(pool_changes(np.flatnonzero(rare))),
            "remove_24_majority": summarize(pool_changes(maj_sample)),
        },
        "RareCopula_component": {
            "remove_true_rare": summarize(copula_changes(np.flatnonzero(rare))),
            "remove_24_majority": summarize(copula_changes(maj_sample)),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    bench = Bench(args.dist)
    A, B = bench.A_bg, bench.B

    # Exact full-data source truth, recovered by instrumenting the published code.
    _raw, Apool_full, true_sel = _build_capture(bench, B, bench.is_rare, 5312, 4000)
    candidates = {
        "Pool": pd.read_csv(args.candidates / "C_candidate.csv"),
        "SA": pd.read_csv(args.candidates / "methods" / "C_sa.csv"),
        "Copula": pd.read_csv(args.candidates / "methods" / "C_copula.csv"),
    }
    provenance: dict[str, Any] = {}
    recovered_full: dict[str, np.ndarray] = {}
    for name, C in candidates.items():
        ind, _dist, rec = _link_source_rows(A, C, k=100)
        recovered_full[name] = rec
        # Copula has no source truth. Comparing with the unrelated Pool source set
        # is explicitly a negative-control pseudo-link rate.
        provenance[name] = _source_metrics(ind, rec, true_sel)
        provenance[name]["truth_interpretation"] = (
            "actual PoolMatch source" if name in {"Pool", "SA"}
            else "negative control against unrelated Pool source; Copula has no source rows"
        )

    rows: dict[str, list[np.ndarray]] = {}
    labels, rare_flags, split_ids = [], [], []
    targeted_oracle, targeted_recovered, targeted_y, targeted_split = [], [], [], []
    split_source_recall = []
    for s in range(args.splits):
        rng = np.random.default_rng(args.seed + s)
        order = rng.permutation(len(B))
        out_idx, in_idx = order[:args.holdout], order[args.holdout:]
        Bin = B.iloc[in_idx].reset_index(drop=True)
        rin = bench.is_rare[in_idx]
        C, Apool, sel = _build_capture(bench, Bin, rin, args.seed + s, args.polish_steps)
        ind, _d, rec = _link_source_rows(A, C, k=100)
        split_source_recall.append(_source_metrics(ind, rec, sel)["set_recall"])
        det_pool = D.maha(Apool, bench.ref.mu, bench.ref.sigma) >= float(bench.ref.band["lo"])
        for prefix, source in (("oracle", sel), ("recovered", rec)):
            feat = _artifact_features(A, Apool, source, B, det_pool, args.seed + s)
            for name, val in feat.items():
                rows.setdefault(f"{prefix}:{name}", []).append(np.asarray(val, float))
        y = np.zeros(len(B), dtype=int)
        y[in_idx] = 1
        labels.append(y)
        rare_flags.append(bench.is_rare.copy())
        split_ids.append(np.full(len(B), s, dtype=int))

        so, yo = _targeted_rare_scores(A, B, bench.is_rare, in_idx, sel, bench)
        sr, yr = _targeted_rare_scores(A, B, bench.is_rare, in_idx, rec, bench)
        targeted_oracle.append(so)
        targeted_recovered.append(sr)
        targeted_y.append(yo)
        targeted_split.append(np.full(len(yo), s, dtype=int))
        print(f"split {s + 1}/{args.splits}: recovered source recall={split_source_recall[-1]:.3f}", flush=True)

    y = np.concatenate(labels)
    rare = np.concatenate(rare_flags)
    groups = np.concatenate(split_ids)
    rng_metric = np.random.default_rng(args.seed + 99999)
    shadow = {
        name: _metrics(np.concatenate(vals), y, groups, rare, rng_metric, args.permutations)
        for name, vals in rows.items()
    }
    ty = np.concatenate(targeted_y)
    tg = np.concatenate(targeted_split)
    tr = np.ones(len(ty), dtype=bool)
    shadow["oracle:targeted_rare_loo_loglik"] = _metrics(
        np.concatenate(targeted_oracle), ty, tg, tr, rng_metric, args.permutations,
    )
    shadow["recovered:targeted_rare_loo_loglik"] = _metrics(
        np.concatenate(targeted_recovered), ty, tg, tr, rng_metric, args.permutations,
    )

    return {
        "scope": "non-official white-box sensitivity audit; no official anonymity conversion",
        "split_design": {
            "n_total": int(len(B)), "holdout": int(args.holdout),
            "train": int(len(B) - args.holdout), "splits": int(args.splits),
            "true_rare_total": int(bench.is_rare.sum()),
            "pool_polish_steps": int(args.polish_steps),
            "source_recovery_recall_mean": float(np.mean(split_source_recall)),
            "source_recovery_recall_each": [float(x) for x in split_source_recall],
        },
        "full_candidate_provenance": provenance,
        "shadow_B_in_vs_B_out": shadow,
        "full_data_leave_one_out_parameter_sensitivity": _model_loo_sensitivity(bench, Apool_full),
        "limitations": [
            "A_bg provenance is public-row disclosure, not by itself private B membership disclosure.",
            "The targeted rare likelihood ratio assumes every other training row and a same-class replacement are known; it is an oracle upper bound.",
            "Only 24 true-rare people exist, so TPR@1%FPR is discrete and correlated across repeated splits.",
            "Copula has no A_bg source identity; its comparison is model-parameter LOO sensitivity, not source linkage.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, default=Path("work/data_prelim_team05"))
    ap.add_argument("--candidates", type=Path, default=Path("work/prelim_team05"))
    ap.add_argument("--out", type=Path, default=Path("work/prelim_team05/whitebox_audit.json"))
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--holdout", type=int, default=300)
    ap.add_argument("--seed", type=int, default=8800)
    ap.add_argument("--polish-steps", type=int, default=4000)
    ap.add_argument("--permutations", type=int, default=200)
    args = ap.parse_args()
    if args.out.exists():
        ap.error(f"refusing to overwrite {args.out}")
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
