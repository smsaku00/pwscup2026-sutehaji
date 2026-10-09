"""P02: AIA-derived home-prior, two out-of-scope classes, preliminary MIA.

EXPLORATORY ONLY. Own B_self and public inputs produce all scores first.
Published pool_ids are opened solely afterwards by the audit/evaluation stage.
Never publish candidate-level scores or any private data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import rankdata

BASE_REL = Path("攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py")
BASE_SHA256 = "28b6a7c6106d89c9bcca6ca138c0aaae23c0b7c03e8dc299c34efd98c9bd4cb3"
EXPERIMENT = {
    "name": "P02 AIA home prior and two out-of-scope categories",
    "phase": "preliminary",
    "self_team": 5,
    "baseline": "P01 with original calibrator 500 and median 0.1",
    "k_targets": 20,
    "home_features": ["prefecture"],
    "private_total": 313,
    "known_private": 163,
    "private_unknown_capacity": 150,
    "all_shared_total": 736,
    "pair_bounds": [25, 27],
    "joint_temperature": 0.05,
    "private_home_prior_gammas": [0.0, 0.2, 0.5],
    "joint_mixing": [0.10, 0.25],
    "direct_nonhome_offsets": [0.05, 0.10],
    "gate_top_fraction": 0.10,
    "backgrounds": ["full", "sub80_seed20261009", "sub80_seed20261010", "sub80_seed20261011"],
    "fpr": [0.005, 0.01, 0.02],
    "low_fpr_primary": 0.01,
    "keep_rare": "known private=0, other rare=0.4, normal rank in (0.6,0.9]",
    "truth_boundary": "read official pool_ids only after prediction hashes were saved",
    "independence": "same explored preliminary round, not an independent validation",
    "main_submission": "not generated or changed",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_base(root):
    p = root / BASE_REL
    if sha(p) != BASE_SHA256:
        raise RuntimeError("Pinned baseline source changed: " + str(p))
    spec = importlib.util.spec_from_file_location("p02_pinned_baseline", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def box_bounds(n, k):
    """Unknown records: exactly 150 self-private and 736 - sum(target shared) elsewhere."""
    absent = 29 - k
    if absent < 0:
        raise ValueError("More than 29 possible other cohorts")
    t_low = max(25 * k, 736 - 27 * absent)
    t_high = min(27 * k, 736 - 25 * absent)
    if t_low > t_high:
        raise ValueError("Infeasible pair capacities")
    lo = np.r_[np.repeat(25.0, k), 150.0, 736.0 - t_high]
    hi = np.r_[np.repeat(27.0, k), 150.0, 736.0 - t_low]
    if not (lo.sum() <= n <= hi.sum()):
        raise ValueError("Infeasible overall mass")
    if k == 20:
        assert int(lo[-1]) == 225 and int(hi[-1]) == 236
    return lo, hi


def ordinary_baseline(base, raw, cal, normal):
    r = base.ranks(cal[normal])
    med = base.rivals(r, np.median)
    p01 = r - 0.1 * med
    return r, p01


def make_predictions(base, raw, cal, b, qs, teams, normal, private, rare):
    """No truth files are read here. The unknown rare rows enter capacity only."""
    home_set = set(qs[5].prefecture.astype(str).unique())
    if len(home_set) != 1:
        raise ValueError("Expected exactly one AIA home prefecture")
    home = next(iter(home_set))
    unknown = ~private
    assert int(unknown.sum()) == 886 and int(normal.sum()) == 875
    is_home = b.prefecture.astype(str).eq(home).to_numpy()
    is_home_unknown = is_home[unknown].astype(float)
    is_nonhome_normal = (~is_home[normal]).astype(float)
    normal_within_unknown = normal[unknown]

    r, p01 = ordinary_baseline(base, raw, cal, normal)
    latent = base.extend_ranks(cal, normal, unknown)
    # A P01 row can be evaluated also for 11 rare unknowns without using hidden truth.
    p01_latent = latent - 0.1 * base.rivals(latent, np.median)
    if not np.allclose(p01_latent[normal_within_unknown], p01, atol=0, rtol=0):
        raise RuntimeError("P01 cost mismatch for normal candidates")

    methods = {"P01": p01}
    for weight in [0.05, 0.10]:
        name = f"direct_nonhome_{weight:.2f}"
        methods[name] = p01 + weight * is_nonhome_normal[:, None]

    # Common control: old one-dummy model, but with P01 costs and weak mixture.
    n, k = p01_latent.shape
    lo1, hi1 = base.capacity_bounds(n, k)
    cost1 = np.column_stack([p01_latent, np.full(n, 0.5)])
    p1, diag1 = base.soft_assignment(cost1, lo1, hi1, 0.05)
    prior_rank1 = base.ranks(p1[normal_within_unknown, :k])
    methods["one_dummy_p01_mix010"] = 0.90 * p01 + 0.10 * prior_rank1
    diag = [{"method": "one_dummy_p01_mix010", **diag1}]
    
    lo2, hi2 = box_bounds(n, k)
    probs = {}
    for gamma in [0.0, 0.2, 0.5]:
        # Keep unobserved shared group independent of the home prefecture score.
        cost2 = np.column_stack([
            p01_latent,
            0.5 + gamma * is_home_unknown,
            np.full(n, 0.5),
        ])
        p2, diag2 = base.soft_assignment(cost2, lo2, hi2, 0.05)
        probs[gamma] = p2
        pr = base.ranks(p2[normal_within_unknown, :k])
        methods[f"two_dummy_g{gamma:.1f}_mix010"] = 0.90 * p01 + 0.10 * pr
        if gamma in [0.2, 0.5]:
            methods[f"two_dummy_g{gamma:.1f}_mix025"] = 0.75 * p01 + 0.25 * pr
        if gamma == 0.5:
            # Apply the perturbation only to the strongest 10% baseline candidates
            # within each target, avoiding low-rank noise and preserving tie policy.
            r_p01 = base.ranks(p01)
            eligible = r_p01 >= (1 - EXPERIMENT["gate_top_fraction"])
            methods["two_dummy_g0.5_gate_top10_mix010"] = np.where(
                eligible, 0.90 * p01 + 0.10 * pr, p01
            )
        diag.append({
            "method": f"two_dummy_g{gamma:.1f}",
            "private_home_mass": float(p2[is_home_unknown > 0.5, k].sum()),
            "private_nonhome_mass": float(p2[is_home_unknown < 0.5, k].sum()),
            **diag2,
        })

    assert len(methods) == 10, list(methods)
    assert all(np.isfinite(x).all() for x in methods.values())
    return methods, diag, {
        "home_prefecture": home,
        "unknown_home_count": int(is_home_unknown.sum()),
        "unknown_nonhome_count": int(n - is_home_unknown.sum()),
        "unknown_regular_count": int(normal.sum()),
        "split_lower": lo2.tolist(),
        "split_upper": hi2.tolist(),
        "one_dummy_lower": lo1.tolist(),
        "one_dummy_upper": hi1.tolist(),
    }


def score_all(base, values, private, rare, normal):
    scores = base.confidence(values, normal, private, rare)
    assert scores.shape == (1049, values.shape[1])
    assert np.all(scores[private] == 0.0)
    assert np.all(scores[rare & ~private] == 0.4)
    return scores


def evaluate(base, predictions, y, teams, private, rare, normal):
    rows = []
    for view, methods in predictions:
        for method, vals in methods.items():
            out = score_all(base, vals, private, rare, normal)
            for j, target in enumerate(teams):
                m = base.metrics(out[:, j], y[:, j])
                rows.append({
                    "background": view, "target": target, "method": method,
                    "members": int(y[:, j].sum()), **m
                })
    table = pd.DataFrame(rows)
    summaries = []
    evaluation_sets = {
        "all20": [], "without1_2": [1, 2], "without1_2_12": [1, 2, 12]
    }
    for background, panel in table.groupby("background", sort=False):
        ref = panel[panel.method == "P01"].set_index("target")["tpr_0.01"]
        for eval_set, ignored in evaluation_sets.items():
            for method, g in panel[~panel.target.isin(ignored)].groupby("method", sort=False):
                diff = g.set_index("target")["tpr_0.01"] - ref.loc[g.target]
                summaries.append({
                    "background": background, "evaluation": eval_set, "method": method,
                    "targets": len(g), "mean_tpr": float(g["tpr_0.01"].mean()),
                    "mean_delta_p01": float(diff.mean()),
                    "wins": int((diff > 1e-12).sum()),
                    "ties": int((diff.abs() <= 1e-12).sum()),
                    "losses": int((diff < -1e-12).sum()),
                    "worst_delta": float(diff.min()),
                    "best_delta": float(diff.max()),
                    "band2plus": int((g.band >= 2).sum())
                })
    return table, pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output_dir
    if out.exists():
        raise FileExistsError(f"Output destination already exists: {out}")
    out.mkdir(parents=True)
    start = time.perf_counter()
    base = load_base(root)
    c, calib, basedir, paths, b, a, qs, cs, teams, private, rare, normal = base.load_inputs(root)
    assert len(teams) == 20
    dump(out / "protocol.json", EXPERIMENT)
    inputs = {str(p.relative_to(root)): sha(p) for p in paths}
    inputs[str(BASE_REL)] = sha(root / BASE_REL)
    inputs["run_p02.py"] = sha(__file__)
    dump(out / "input_sha256.json", inputs)

    pred = []
    diag = []
    hashes = []
    start_utc = datetime.now(timezone.utc).isoformat()
    views = EXPERIMENT["backgrounds"][:1] if args.full_only else EXPERIMENT["backgrounds"]
    for view in views:
        if view == "full":
            aa = a
        else:
            seed = int(view.split("seed")[1])
            random = np.random.default_rng(seed)
            aa = a.iloc[np.sort(random.choice(len(a), int(0.8 * len(a)), replace=False))]
        raw, cal = base.raw_scores(c, calib, b, aa, qs, cs, teams)
        methods, d, meta = make_predictions(
            base, raw, cal, b, qs, teams, normal, private, rare
        )
        pred.append((view, methods))
        for row in d:
            diag.append({"background": view, **row})
        for method, values in methods.items():
            hashes.append({
                "background": view, "method": method, "shape": list(values.shape),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(values, dtype="<f8").tobytes()
                ).hexdigest()
            })
        dump(out / "public_group_constraints.json", meta)
        print(f"Predictions frozen in memory: {view}", flush=True)
    dump(out / "prediction_hashes.json", hashes)
    dump(out / "optimizer_diagnostics.json", diag)
    freeze_utc = datetime.now(timezone.utc).isoformat()
    # SECURITY / EXPERIMENT SEPARATION: the only point where truth is accessed.
    y, audit, truthpaths = base.audit_truth(basedir, b, private, rare, teams)
    dump(out / "constraint_audit.json", audit)
    dump(out / "truth_input_sha256.json", {
        str(p.relative_to(root)): sha(p) for p in truthpaths
    })
    table, summary = evaluate(base, pred, y, teams, private, rare, normal)
    table.to_csv(out / "by_target.csv", index=False, float_format="%.17g")
    summary.to_csv(out / "summary.csv", index=False, float_format="%.17g")
    full = summary[
        (summary.background == "full") & (summary.evaluation == "all20")
    ].set_index("method")
    ref = float(full.loc["P01", "mean_tpr"])
    if abs(ref - 0.08079772079772081) >= 1e-12:
        raise RuntimeError("P01 reference regression failed: " + repr(ref))
    dump(out / "run_record.json", {
        "started_utc": start_utc,
        "frozen_utc": freeze_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.perf_counter() - start,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "input_hashes_before_truth": True,
        "prediction_conditions": len(hashes),
        "evaluation_rows": len(table),
        "P01_regression_full_all20": ref,
        "no_main_data": True, "no_competition_submission": True
    })
    print(summary[(summary.background == "full") &
                  (summary.evaluation == "all20")].to_string(index=False))
    print("OUTPUT:", str(out), flush=True)


if __name__ == "__main__":
    main()
