#!/usr/bin/env python3
"""Cross-C consensus diagnostics for the preliminary-round MIA task.

The attacker knows its own B and sees every submitted C.  A row of B can be a
member of at most one target cohort, so closeness to C_k can be contrasted with
closeness to all other C files.  This script evaluates that idea without using
any hidden membership labels.

It also tests a literal C-to-C bridge, B -> C_self -> C_k.  The bridge is kept
separate because common A_bg provenance can create C-to-C neighbours that are
not evidence of private membership.

Outputs are diagnostics and candidate rankings, not an attack submission.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
CAT = ["sex", "smoking", "prefecture"]
SELF = 5
EPS = 1e-12


def robust_scale(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    med = np.median(x, axis=axis, keepdims=True)
    mad = 1.4826 * np.median(np.abs(x - med), axis=axis, keepdims=True)
    mad = np.where(mad > 1e-9, mad, 1.0)
    return med, mad


def z_values(df: pd.DataFrame, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (df[CONT].to_numpy(float) - mean) / sd


def target_ids(dist: Path) -> List[int]:
    spec = json.loads((dist / "mia_columns.json").read_text())
    ids = [int(x) for x in spec["target_columns"]]
    return [k for k in ids if k != SELF and (dist / f"C_{k}.csv").exists()]


def distance_matrix(
    x: pd.DataFrame,
    cs: Mapping[int, pd.DataFrame],
    targets: Sequence[int],
    mean: np.ndarray,
    sd: np.ndarray,
    kth: int = 1,
) -> np.ndarray:
    zx = z_values(x, mean, sd)
    out = np.empty((len(x), len(targets)), dtype=float)
    for j, k in enumerate(targets):
        nn = NearestNeighbors(n_neighbors=kth).fit(z_values(cs[k], mean, sd))
        out[:, j] = nn.kneighbors(zx, return_distance=True)[0][:, kth - 1]
    return out


def clean_public_pool(
    a: pd.DataFrame,
    cs: Mapping[int, pd.DataFrame],
    targets: Sequence[int],
    mean: np.ndarray,
    sd: np.ndarray,
    threshold: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Remove A_bg points appearing very close to any released C.

    Several submissions directly reuse A_bg rows.  Such a row is a true
    non-member but would look like an exact member under a distance attack, so
    it is unsuitable as a reference distribution for calibrating the attack.
    We report how many rows this removes rather than silently hiding it.
    """
    za = z_values(a, mean, sd)
    dmin = np.full(len(a), np.inf)
    for k in targets:
        nn = NearestNeighbors(n_neighbors=1).fit(z_values(cs[k], mean, sd))
        dmin = np.minimum(dmin, nn.kneighbors(za)[0][:, 0])
    keep = dmin > threshold
    return a.loc[keep].reset_index(drop=True), dmin


def matched_controls(
    b: pd.DataFrame,
    pool: pd.DataFrame,
    mean: np.ndarray,
    sd: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Greedy, without-replacement public controls matched to each B row.

    Exact sex/smoking matches are required.  Exact prefecture is preferred;
    otherwise the closest same-sex/smoking row is used.  Continuous matching
    uses the public A_bg scale.  This is only a diagnostic null, never truth.
    """
    rng = np.random.default_rng(seed)
    zb = z_values(b, mean, sd)
    za = z_values(pool, mean, sd)
    used = np.zeros(len(pool), dtype=bool)
    chosen = np.full(len(b), -1, dtype=int)

    # Harder rows first: small exact strata get first access to their matches.
    keys_b = list(zip(b["sex"], b["smoking"], b["prefecture"]))
    counts = pd.Series(keys_b).value_counts().to_dict()
    order = np.arange(len(b))
    jitter = rng.random(len(b)) * 1e-6
    order = order[np.argsort(np.array([counts[x] for x in keys_b]) + jitter)]

    # Query many neighbours once; expand to all rows only on the rare fallback.
    n_query = min(512, len(pool))
    nn = NearestNeighbors(n_neighbors=n_query).fit(za)
    neigh = nn.kneighbors(zb, return_distance=False)
    sex_a = pool["sex"].to_numpy()
    smk_a = pool["smoking"].to_numpy()
    pref_a = pool["prefecture"].astype(str).to_numpy()

    for i in order:
        cand = neigh[i]
        same_ss = (sex_a[cand] == b.iloc[i]["sex"]) & (smk_a[cand] == b.iloc[i]["smoking"])
        same_pref = pref_a[cand] == str(b.iloc[i]["prefecture"])
        tiers = [cand[same_ss & same_pref], cand[same_ss], cand]
        pick = -1
        for tier in tiers:
            free = tier[~used[tier]]
            if len(free):
                pick = int(free[0])
                break
        if pick < 0:
            free = np.flatnonzero(~used)
            if not len(free):
                raise RuntimeError("control pool exhausted")
            d2 = ((za[free] - zb[i]) ** 2).sum(axis=1)
            pick = int(free[np.argmin(d2)])
        chosen[i] = pick
        used[pick] = True
    return pool.iloc[chosen].reset_index(drop=True)


def empirical_percentile(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Column-wise empirical P(ref <= x), with no model fit."""
    out = np.empty_like(x, dtype=float)
    for j in range(x.shape[1]):
        s = np.sort(ref[:, j])
        out[:, j] = np.searchsorted(s, x[:, j], side="right") / (len(s) + 1.0)
    return np.clip(out, 1.0 / (len(ref) + 1.0), len(ref) / (len(ref) + 1.0))


def consensus_scores(d_x: np.ndarray, d_ref: np.ndarray) -> Dict[str, np.ndarray]:
    """Return raw and cross-target scores; larger means more member-like."""
    lx = -np.log(np.clip(d_x, EPS, None))
    lr = -np.log(np.clip(d_ref, EPS, None))

    med, scale = robust_scale(lr, axis=0)
    direct_z = (lx - med) / scale

    # Public-null tail strength for each C.  This calibrates teams with very
    # different smoothing/noise levels before comparing columns.
    pct = empirical_percentile(lx, lr)
    tail = -np.log10(np.clip(1.0 - pct, 1.0 / (len(lr) + 1.0), 1.0))

    # Leave-one-column-out specificity.  A naturally common-looking person is
    # close to many C files; a membership signal should peak for one target.
    row_med = np.median(tail, axis=1, keepdims=True)
    row_mad = 1.4826 * np.median(np.abs(tail - row_med), axis=1, keepdims=True)
    row_mad = np.where(row_mad > 0.05, row_mad, 0.05)
    specific = (tail - row_med) / row_mad

    # A gentler alternative: subtract the mean percentile of all other teams.
    p_sum = pct.sum(axis=1, keepdims=True)
    rank_resid = pct - (p_sum - pct) / max(pct.shape[1] - 1, 1)
    return {
        "direct_z": direct_z,
        "public_tail": tail,
        "cross_specific": specific,
        "rank_residual": rank_resid,
    }


def bridge_distances(
    x: pd.DataFrame,
    c_self: pd.DataFrame,
    cs: Mapping[int, pd.DataFrame],
    targets: Sequence[int],
    mean: np.ndarray,
    sd: np.ndarray,
    n_self_neighbours: int = 5,
) -> np.ndarray:
    """Approximate min_u d(x,C5_u)+d(C5_u,Ck) using five C5 neighbours."""
    zx = z_values(x, mean, sd)
    z5 = z_values(c_self, mean, sd)
    n5 = min(n_self_neighbours, len(c_self))
    db5, ib5 = NearestNeighbors(n_neighbors=n5).fit(z5).kneighbors(zx)
    out = np.empty((len(x), len(targets)), dtype=float)
    for j, k in enumerate(targets):
        zk = z_values(cs[k], mean, sd)
        d5k = NearestNeighbors(n_neighbors=1).fit(zk).kneighbors(z5)[0][:, 0]
        out[:, j] = np.min(db5 + d5k[ib5], axis=1)
    return out


def top_metrics(score: np.ndarray, positives: Sequence[int], topn: int = 26) -> dict:
    positives = np.array(sorted(set(int(x) for x in positives)), dtype=int)
    order = np.argsort(-score)
    ranks = np.empty(len(score), dtype=int)
    ranks[order] = np.arange(1, len(score) + 1)
    out = {
        "known_positive_count": int(len(positives)),
        f"recall_at_{topn}": float(np.mean(ranks[positives] <= topn)) if len(positives) else math.nan,
        f"hits_at_{topn}": int(np.sum(ranks[positives] <= topn)),
        "median_known_rank": float(np.median(ranks[positives])) if len(positives) else math.nan,
        "worst_known_rank": int(np.max(ranks[positives])) if len(positives) else None,
    }
    for n in (50, 100):
        out[f"recall_at_{n}"] = float(np.mean(ranks[positives] <= n)) if len(positives) else math.nan
        out[f"hits_at_{n}"] = int(np.sum(ranks[positives] <= n))
    return out


def one_percent_union_diagnostic(score_b: np.ndarray, score_r: np.ndarray) -> dict:
    sb = np.max(score_b, axis=1)
    sr = np.max(score_r, axis=1)
    y = np.r_[np.ones(len(sb), dtype=int), np.zeros(len(sr), dtype=int)]
    s = np.r_[sb, sr]
    threshold = float(np.quantile(sr, 0.99))
    return {
        "auc_B_vs_matched_public": float(roc_auc_score(y, s)),
        "control_p99": threshold,
        "B_above_control_p99": int(np.sum(sb > threshold)),
        "control_above_control_p99": int(np.sum(sr > threshold)),
        "B_rate_above_control_p99": float(np.mean(sb > threshold)),
    }


def capacity_assignment(score: np.ndarray, targets: Sequence[int], capacity: int) -> Dict[int, List[int]]:
    """Assign exactly capacity unique B rows to every target."""
    expanded = np.repeat(np.asarray(targets), capacity)
    cols = np.repeat(np.arange(len(targets)), capacity)
    # rectangular assignment: every target slot receives a distinct B row
    rows, slots = linear_sum_assignment(-score[:, cols])
    out = {int(k): [] for k in targets}
    for i, s in zip(rows, slots):
        out[int(expanded[s])].append(int(i))
    for k in out:
        out[k].sort()
    return out


def pairwise_c_summary(
    cs: Mapping[int, pd.DataFrame],
    ids: Sequence[int],
    public_a: pd.DataFrame,
    mean: np.ndarray,
    sd: np.ndarray,
) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    zc = {k: z_values(cs[k], mean, sd) for k in ids}
    keys = {
        k: set(map(tuple, np.round(cs[k][CONT].to_numpy(float), 9)))
        for k in ids
    }
    public_keys = set(map(tuple, np.round(public_a[CONT].to_numpy(float), 9)))
    for ia, a in enumerate(ids):
        for b in ids[ia + 1 :]:
            dab = NearestNeighbors(n_neighbors=1).fit(zc[b]).kneighbors(zc[a])[0][:, 0]
            dba = NearestNeighbors(n_neighbors=1).fit(zc[a]).kneighbors(zc[b])[0][:, 0]
            d = np.r_[dab, dba]
            exact = keys[a] & keys[b]
            rows.append(
                {
                    "a": int(a),
                    "b": int(b),
                    "median_symmetric_nn": float(np.median(d)),
                    "q01_symmetric_nn": float(np.quantile(d, 0.01)),
                    "n_lt_1e-6": int(np.sum(d < 1e-6)),
                    "n_lt_0.01": int(np.sum(d < 0.01)),
                    "n_lt_0.05": int(np.sum(d < 0.05)),
                    "exact_unique_cont7": int(len(exact)),
                    "exact_unique_from_A_bg": int(len(exact & public_keys)),
                    "exact_unique_not_A_bg": int(len(exact - public_keys)),
                }
            )
    frame = pd.DataFrame(rows)
    with_self = frame[(frame["a"] == SELF) | (frame["b"] == SELF)]
    without_self = frame[(frame["a"] != SELF) & (frame["b"] != SELF)]
    summary = {
        "n_pairs": int(len(frame)),
        "self_pairs": {
            "median_pair_median": float(with_self["median_symmetric_nn"].median()),
            "median_n_lt_0.05": float(with_self["n_lt_0.05"].median()),
            "max_n_lt_0.05": int(with_self["n_lt_0.05"].max()),
        },
        "other_pairs": {
            "median_pair_median": float(without_self["median_symmetric_nn"].median()),
            "median_n_lt_0.05": float(without_self["n_lt_0.05"].median()),
            "max_n_lt_0.05": int(without_self["n_lt_0.05"].max()),
        },
        "non_public_exact_pairs": frame.loc[
            frame["exact_unique_not_A_bg"] > 0,
            ["a", "b", "exact_unique_cont7", "exact_unique_from_A_bg", "exact_unique_not_A_bg"],
        ].to_dict(orient="records"),
        "public_exact_pairs": frame.loc[
            frame["exact_unique_from_A_bg"] > 0,
            ["a", "b", "exact_unique_cont7", "exact_unique_from_A_bg", "exact_unique_not_A_bg"],
        ].to_dict(orient="records"),
    }
    return rows, summary


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--clean-threshold", type=float, default=0.05)
    parser.add_argument("--capacity", type=int, default=26)
    args = parser.parse_args()

    root = args.root.resolve()
    dist = root / "予備戦/配布データ" / "PWSCup2026_prelim_attack"
    private = root / "work" / "data_prelim_team05"
    outdir = Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)

    bs = pd.read_csv(private / "B_self.csv")
    b = bs.drop(columns=[c for c in ["is_rare"] if c in bs.columns]).reset_index(drop=True)
    a = pd.read_csv(private / "A_bg.csv").reset_index(drop=True)
    targets = target_ids(dist)
    all_ids = sorted(set(targets + [SELF]))
    cs = {k: pd.read_csv(dist / f"C_{k}.csv") for k in all_ids}

    mean = a[CONT].mean().to_numpy(float)
    sd = a[CONT].std(ddof=0).to_numpy(float)
    sd = np.where(sd > 0, sd, 1.0)

    clean, public_min = clean_public_pool(a, cs, targets, mean, sd, args.clean_threshold)
    controls = matched_controls(b, clean, mean, sd, args.seed)

    d1_b = distance_matrix(b, cs, targets, mean, sd, kth=1)
    d1_r = distance_matrix(controls, cs, targets, mean, sd, kth=1)
    d5_b = distance_matrix(b, cs, targets, mean, sd, kth=5)
    d5_r = distance_matrix(controls, cs, targets, mean, sd, kth=5)
    sc1_b = consensus_scores(d1_b, d1_r)
    sc1_r = consensus_scores(d1_r, d1_r)
    sc5_b = consensus_scores(d5_b, d5_r)
    sc5_r = consensus_scores(d5_r, d5_r)

    bridge_b = bridge_distances(b, cs[SELF], cs, targets, mean, sd)
    bridge_r = bridge_distances(controls, cs[SELF], cs, targets, mean, sd)
    scb_b = consensus_scores(bridge_b, bridge_r)
    scb_r = consensus_scores(bridge_r, bridge_r)

    methods_b = {
        "nn1_raw_closeness": -d1_b,
        "nn1_public_null_z": sc1_b["direct_z"],
        "nn1_cross_specific": sc1_b["cross_specific"],
        "nn1_rank_residual": sc1_b["rank_residual"],
        "nn5_cross_specific": sc5_b["cross_specific"],
        "c5_bridge_cross_specific": scb_b["cross_specific"],
    }
    methods_r = {
        "nn1_raw_closeness": -d1_r,
        "nn1_public_null_z": sc1_r["direct_z"],
        "nn1_cross_specific": sc1_r["cross_specific"],
        "nn1_rank_residual": sc1_r["rank_residual"],
        "nn5_cross_specific": sc5_r["cross_specific"],
        "c5_bridge_cross_specific": scb_r["cross_specific"],
    }

    confirmed_path = root / "work" / "submit_audit_out" / "confirmed.json"
    confirmed = json.loads(confirmed_path.read_text()) if confirmed_path.exists() else {}
    fingerprint_path = root / "work" / "submit_audit_out" / "fingerprint.json"
    fingerprints = json.loads(fingerprint_path.read_text()) if fingerprint_path.exists() else {}
    evaluations: Dict[str, dict] = {}
    for name, score in methods_b.items():
        per_known = {}
        for k_text, positives in confirmed.items():
            k = int(k_text)
            if k in targets:
                per_known[str(k)] = top_metrics(score[:, targets.index(k)], positives)
        per_fingerprint = {}
        for k_text, entry in fingerprints.items():
            k = int(k_text)
            positives = entry.get("rows", [])
            if k in targets and positives:
                per_fingerprint[str(k)] = top_metrics(score[:, targets.index(k)], positives)
        evaluations[name] = {
            "known_positive_proxy": per_known,
            "independent_time_or_cell_fingerprint_proxy": per_fingerprint,
            "union_null_diagnostic": one_percent_union_diagnostic(score, methods_r[name]),
        }

    # A useful structure-only check: non-Tokyo rows are known to be shared with
    # some cohort, whereas only a minority of Tokyo rows are shared.  It cannot
    # validate the assigned target, but it tests whether max specificity carries
    # any shared-membership signal at all.
    tokyo = (b["prefecture"].astype(str) == "東京都").to_numpy()
    for name, score in methods_b.items():
        mx = score.max(axis=1)
        _, p = mannwhitneyu(mx[~tokyo], mx[tokyo], alternative="greater")
        evaluations[name]["union_structure_check"] = {
            "median_non_tokyo": float(np.median(mx[~tokyo])),
            "median_tokyo": float(np.median(mx[tokyo])),
            "mannwhitney_non_tokyo_greater_p": float(p),
        }

    # The balanced matching is intentionally based on the gentler rank residual.
    assignment = capacity_assignment(methods_b["nn1_rank_residual"], targets, args.capacity)
    assignment_rows = []
    for k, indices in assignment.items():
        j = targets.index(k)
        for i in indices:
            assignment_rows.append(
                {
                    "target_k": k,
                    "row_index": i,
                    "record_id": int(b.iloc[i]["record_id"]),
                    "score": float(methods_b["nn1_rank_residual"][i, j]),
                    "distance": float(d1_b[i, j]),
                    "is_rare": int(bs.iloc[i]["is_rare"]),
                    "is_tokyo": int(tokyo[i]),
                }
            )
    pd.DataFrame(assignment_rows).sort_values(["target_k", "score"], ascending=[True, False]).to_csv(
        outdir / "capacity26_candidates.csv", index=False
    )

    # Long-form top-40 list for auditability; no record-level attributes are
    # exported, only the private record_id already required by F_mia.
    rank_rows = []
    for name, score in methods_b.items():
        for j, k in enumerate(targets):
            order = np.argsort(-score[:, j])[:40]
            for rank, i in enumerate(order, start=1):
                rank_rows.append(
                    {
                        "method": name,
                        "target_k": k,
                        "rank": rank,
                        "record_id": int(b.iloc[i]["record_id"]),
                        "row_index": int(i),
                        "score": float(score[i, j]),
                        "distance": float(d1_b[i, j]),
                        "is_rare": int(bs.iloc[i]["is_rare"]),
                        "is_tokyo": int(tokyo[i]),
                    }
                )
    pd.DataFrame(rank_rows).to_csv(outdir / "top40_by_method.csv", index=False)

    pair_rows, pair_summary = pairwise_c_summary(cs, all_ids, a, mean, sd)
    pd.DataFrame(pair_rows).to_csv(outdir / "pairwise_c_distances.csv", index=False)

    report = {
        "design": {
            "targets": targets,
            "self": SELF,
            "continuous_columns": CONT,
            "distance": "Euclidean after A_bg mean/SD scaling",
            "cross_specificity": "public-null tail strength minus robust within-row cross-team baseline",
            "bridge": "min over 5 nearest C_self rows of d(B,C_self)+d(C_self,C_target)",
            "capacity": args.capacity,
            "warning": "No hidden per-target membership truth is available; known-positive checks are partial/circular proxies.",
        },
        "data": {
            "n_B": len(b),
            "n_A_bg": len(a),
            "n_targets": len(targets),
            "public_rows_removed_as_C_near": int(np.sum(public_min <= args.clean_threshold)),
            "clean_public_controls_available": len(clean),
            "matched_controls": len(controls),
        },
        "evaluations": evaluations,
        "pairwise_C": pair_summary,
        "assignment": {
            "method": "nn1_rank_residual",
            "rows_assigned": int(sum(len(v) for v in assignment.values())),
            "unique_rows_assigned": int(len(set(i for v in assignment.values() for i in v))),
            "per_target": {str(k): len(v) for k, v in assignment.items()},
        },
    }
    (outdir / "cross_c_consensus_results.json").write_text(
        json.dumps(jsonable(report), ensure_ascii=False, indent=2) + "\n"
    )

    print(json.dumps(jsonable(report), ensure_ascii=False, indent=2))
    print(f"\nOutputs: {outdir}")


if __name__ == "__main__":
    main()
