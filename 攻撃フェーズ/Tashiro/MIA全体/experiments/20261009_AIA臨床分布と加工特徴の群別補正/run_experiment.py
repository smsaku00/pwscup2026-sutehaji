"""PWS Cup 2026 preliminary MIA: AIA clinical-distribution deconfounding.

An explorative, entirely label-free method family fixed before evaluation.
The official preliminary pool_ids are not opened until all prediction hashes and
observable team grouping have been written. No main-round submissions are made.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PREVIOUS_SHA = "28b6a7c6106d89c9bcca6ca138c0aaae23c0b7c03e8dc299c34efd98c9bd4cb3"
PREVIOUS_PATH = "攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py"
NUMERIC = ("age", "SBP", "TG", "HDL", "ALT")
BACKGROUNDS = ("full", "sub80_seed20261009", "sub80_seed20261010", "sub80_seed20261011")
MAIN_TARGETS = tuple(t for t in (1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20, 22, 23, 24, 26, 29) if t not in (1, 2))
MAIN_BASELINE = 0.025495409939854385
SECONDARY_BASELINE_20 = 0.08079772079772081
SECONDARY_EVAL18 = 0.029775245330800885
PARAMS = {
    "study": "AIA clinical Q distribution vs target C; unlabeled deconfounding",
    "primary": "18-target inference excluding teams 1 and 2, mean target TPR@FPR=0.01",
    "secondary": "20-target inference, report all20 and evaluation-only without teams 1 and 2",
    "fpr_metrics": [0.005, 0.01, 0.02],
    "source_baseline_sha": PREVIOUS_SHA,
    "public_features": list(NUMERIC),
    "ratio_private_removed": 313 / 1049,
    "remove_count": 313,
    "q_to_c_reference": "Q target to 1049 C rows vs Q target to fixed 1049 subsampled A_bg rows",
    "q_to_c_seed": 20261009,
    "model_groups": ["aia_imprint_high", "quantized_c", "c_home_above_median"],
    "retained_score": "P01, known private163=0, other rare11=0.4, normal875 only modified",
    "no_truth_used_for_prediction_or_group_selection": True,
    "truth_read_only_after_hashes": True,
    "evaluation_warning": "Labels are published for this very round and repeatedly explored; group observations are descriptive, not independently validated",
    "full_round_submission": "unchanged",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_baseline(root):
    p = root / PREVIOUS_PATH
    if digest(p) != PREVIOUS_SHA:
        raise RuntimeError("Baseline source hash differs from pinned reference")
    spec = importlib.util.spec_from_file_location("pinned_joint_base", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def get_transformed(common, df, scale):
    return common.numbers(df, NUMERIC) / scale


def proxy_keep_indices(C, Q, home, k_remove=313):
    """Rank likely local/private-looking C rows by AIA closeness; keep remainder.

    Q contains 163 private and 163 controls; closeness is a proxy, NEVER truth.
    The home-prefecture mismatch penalty is fixed before reading labels.
    """
    qdist = cdist(C, Q, "sqeuclidean").min(axis=1)
    mismatches = (np.asarray(home["c_pref"]) != home["q_pref"])
    priority = qdist + 0.25 * mismatches
    n = len(priority)
    if not 0 < k_remove < n:
        raise ValueError("Invalid private-proxy removal fraction")
    order = np.argsort(priority, kind="stable")
    mask = np.ones(n, dtype=bool)
    mask[order[:k_remove]] = False
    return mask, qdist, float(np.median(qdist))


def build_public_features(common, b, a, qs, cs, teams):
    """Uses only public C, public AIA Q, public A_bg and self B_self."""
    scale = common.scales(a, NUMERIC)
    bx = get_transformed(common, b, scale)
    rng = np.random.default_rng(PARAMS["q_to_c_seed"])
    bg_sample_ix = np.sort(rng.choice(len(a), 1049, replace=False))
    bgref = get_transformed(common, a.iloc[bg_sample_ix], scale)

    distC, distQ, distResid, characteristics = {}, {}, {}, []
    for team in teams:
        q, c = qs[team], cs[team]
        qhome = sorted(q.prefecture.astype(str).unique())
        if len(qhome) != 1:
            raise RuntimeError(f"Q_{team} does not have a unique home prefecture")
        Q, C = get_transformed(common, q, scale), get_transformed(common, c, scale)
        assert len(q) == 326 and len(c) == 1049
        qc = cdist(Q, C, "sqeuclidean").min(axis=1)
        qa = cdist(Q, bgref, "sqeuclidean").min(axis=1)
        # Small pseudocount is only to keep logs finite for exact coincidences.
        imprint = float(np.log((np.median(qa) + 1e-6) /
                              (np.median(qc) + 1e-6)))
        is_local = c.prefecture.astype(str).eq(qhome[0]).to_numpy()
        mask, _, med_C_to_Q = proxy_keep_indices(
            C, Q, {"c_pref": c.prefecture.astype(str).to_numpy(),
                   "q_pref": qhome[0]}, PARAMS["remove_count"])
        distances = cdist(bx, C, "sqeuclidean")
        distC[team] = distances.min(axis=1)
        distQ[team] = cdist(bx, Q, "sqeuclidean").min(axis=1)
        distResid[team] = distances[:, mask].min(axis=1)

        unique_fracs = [
            float(c[col].nunique(dropna=False) / len(c)) for col in NUMERIC
        ]
        characteristics.append({
            "team": int(team),
            "aia_home_prefecture": qhome[0],
            "C_home_fraction": float(is_local.mean()),
            "C_min_numeric_unique_fraction": float(min(unique_fracs)),
            "C_median_numeric_unique_fraction": float(np.median(unique_fracs)),
            "AIA_imprint_log_BG_to_C_median_Q_NN": imprint,
            "Q_to_C_nn_median": float(np.median(qc)),
            "Q_to_bg_nn_median": float(np.median(qa)),
            "C_to_Q_nn_median": med_C_to_Q,
            "C_rows_removed_proxy": int((~mask).sum()),
            "C_rows_kept_proxy": int(mask.sum()),
        })
        print(f"public AIA×C feature completed target {team}", flush=True)
    descriptors = pd.DataFrame(characteristics).sort_values("team").reset_index(drop=True)
    ref = descriptors[descriptors.team.isin(MAIN_TARGETS)]
    assert len(ref) == 18
    cut_imprint = float(ref["AIA_imprint_log_BG_to_C_median_Q_NN"].median())
    cut_home = float(ref["C_home_fraction"].median())
    descriptors["aia_imprint_high"] = (
        descriptors["AIA_imprint_log_BG_to_C_median_Q_NN"] >= cut_imprint
    ).astype(int)
    descriptors["quantized_c"] = (
        descriptors["C_median_numeric_unique_fraction"] < .90
    ).astype(int)
    descriptors["c_home_above_median"] = (
        descriptors["C_home_fraction"] >= cut_home
    ).astype(int)
    thresholds = {
        "threshold_source": "18 public C/AIA groups excluding Team1/2; no MIA labels",
        "imprint_median": cut_imprint,
        "c_home_median": cut_home,
        "quantized_c_threshold": 0.90,
        "background_sample_seed": PARAMS["q_to_c_seed"],
        "public_reference_A_bg_rows": int(len(bgref)),
        "unobserved_AIA_labels": "not accessed or inferred",
    }
    matrices = {
        name: np.column_stack([values[t] for t in teams])
        for name, values in [("c", distC), ("q", distQ), ("resid", distResid)]
    }
    return matrices, descriptors, thresholds


def score_variants(baseline, cal, normal, indices, selected_teams,
                   distances, groups):
    r = baseline.ranks(cal[normal][:, indices])
    p01 = r - 0.10 * baseline.rivals(r, np.median)
    zc = baseline.ranks(-distances["c"][normal][:, indices])
    zq = baseline.ranks(-distances["q"][normal][:, indices])
    zr = baseline.ranks(-distances["resid"][normal][:, indices])
    contrast = zc - zq
    residual_change = zr - zc
    flags = groups.set_index("team")
    high = np.array([int(flags.loc[t, "aia_imprint_high"]) for t in selected_teams],
                    dtype=float)[None, :]
    quantized = np.array([int(flags.loc[t, "quantized_c"]) for t in selected_teams],
                         dtype=float)[None, :]
    methods = {
        "P01": p01,
        "Cnn_add010": p01 + 0.10 * zc,
        "Qpenalty_005": p01 - 0.05 * zq,
        "Qpenalty_010": p01 - 0.10 * zq,
        "AIA_contrast_010": p01 + 0.10 * contrast,
        "AIA_contrast_020": p01 + 0.20 * contrast,
        "AIA_residual_add010": p01 + 0.10 * zr,
        "AIA_residual_delta010": p01 + 0.10 * residual_change,
        "AIA_residual_delta020": p01 + 0.20 * residual_change,
        "AIA_contrast_high_imprint_010": p01 + 0.10 * contrast * high,
        "AIA_residual_high_imprint_010": p01 + 0.10 * residual_change * high,
        "AIA_residual_quantized_010": p01 + 0.10 * residual_change * quantized,
    }
    assert len(methods) == 12
    expected = (int(normal.sum()), len(selected_teams))
    for name, z in methods.items():
        if z.shape != expected or not np.isfinite(z).all():
            raise RuntimeError(f"Invalid score {name}: {z.shape} expected {expected}")
    np.testing.assert_array_equal(methods["P01"], p01)
    return methods


def mean_agg(df, baseline, cols=("tpr_0.01",)):
    deltas = df.set_index("team")["tpr_0.01"] - baseline.loc[df.team].to_numpy()
    ret = {
        "targets": len(df),
        "mean_tpr": float(df["tpr_0.01"].mean()),
        "mean_delta_P01": float(deltas.mean()),
        "wins": int((deltas > 1e-12).sum()),
        "ties": int((np.abs(deltas) <= 1e-12).sum()),
        "losses": int((deltas < -1e-12).sum()),
        "worst_delta": float(deltas.min()),
        "best_delta": float(deltas.max()),
        "band2plus": int((df["band"] >= 2).sum()),
    }
    for col in cols:
        ret["mean_" + col] = float(df[col].mean())
    return ret


def evaluate(predictions, y, allteams, base, normal, private, rare, groups):
    rows = []
    for background, infer, ids, methods in predictions:
        for method, values in methods.items():
            conf = base.confidence(values, normal, private, rare)
            assert np.all(conf[private] == 0)
            assert np.all(conf[rare & ~private] == 0.4)
            for j, target in enumerate(ids):
                truth = y[:, allteams.index(target)]
                rows.append({
                    "background": background, "inference": infer,
                    "team": int(target), "method": method,
                    "members": int(truth.sum()),
                    **base.metrics(conf[:, j], truth)
                })
    table = pd.DataFrame(rows)
    summary, group_summary = [], []
    pub = groups.set_index("team")
    for (view, infer), panel in table.groupby(["background", "inference"], sort=False):
        reference = panel[panel.method == "P01"].set_index("team")["tpr_0.01"]
        evsets = {
            "without1_2": [t for t in sorted(panel.team.unique()) if t not in (1, 2)],
        }
        if infer == "all20":
            evsets["all20"] = sorted(panel.team.unique())
        for evname, included in evsets.items():
            for method, g in panel[panel.team.isin(included)].groupby("method", sort=False):
                summary.append({
                    "background": view, "inference": infer, "evaluation": evname,
                    "method": method, **mean_agg(g, reference,
                        cols=("tpr_0.005", "tpr_0.01", "tpr_0.02", "auc"))
                })
                if evname != "without1_2":
                    continue
                for axis in ("aia_imprint_high", "quantized_c", "c_home_above_median"):
                    for value in sorted(set(pub.loc[included, axis].astype(int))):
                        members = [t for t in included if int(pub.loc[t, axis]) == value]
                        gg = g[g.team.isin(members)]
                        group_summary.append({
                            "background": view, "inference": infer, "method": method,
                            "group_axis": axis, "group_value": int(value),
                            **mean_agg(gg, reference,
                                cols=("tpr_0.005", "tpr_0.01", "tpr_0.02")),
                        })
    return table, pd.DataFrame(summary), pd.DataFrame(group_summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--full-only", action="store_true")
    args = ap.parse_args()
    root, out = args.root.resolve(), args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Refuse overwrite: {out}")
    out.mkdir(parents=True)
    started = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    write_json(out / "protocol.json", PARAMS)
    base = load_baseline(root)
    common, calmod, basedir, paths, b, a, qs, cs, teams, private, rare, normal = base.load_inputs(root)
    assert len(teams) == 20 and set(MAIN_TARGETS).issubset(teams)
    paths = list(paths) + [root / PREVIOUS_PATH, Path(__file__).resolve()]
    write_json(out / "prediction_input_hashes.json",
               {str(p.relative_to(root)): digest(p) for p in paths})
    dist, groups, gates = build_public_features(common, b, a, qs, cs, teams)
    groups.to_csv(out / "public_target_groups.csv", index=False, float_format="%.17g")
    write_json(out / "group_thresholds.json", gates)

    prediction_list, hashes = [], []
    backgrounds = BACKGROUNDS[:1] if args.full_only else BACKGROUNDS
    for view in backgrounds:
        if view == "full":
            aa = a
        else:
            seed = int(view.split("seed")[1])
            rng = np.random.default_rng(seed)
            aa = a.iloc[np.sort(rng.choice(len(a), int(0.8 * len(a)), replace=False))]
        raw, cal = base.raw_scores(common, calmod, b, aa, qs, cs, teams)
        infsets = [("without1_2", MAIN_TARGETS)]
        if view == "full":
            infsets += [("all20", teams)]
        for infname, ids in infsets:
            ix = [teams.index(t) for t in ids]
            methods = score_variants(base, cal, normal, ix, ids, dist, groups)
            for method, scores in methods.items():
                hashes.append({
                    "background": view, "inference": infname,
                    "method": method, "shape": list(scores.shape),
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(scores, dtype="<f8").tobytes()
                    ).hexdigest()
                })
            prediction_list.append((view, infname, ids, methods))
    write_json(out / "prediction_hashes.json", hashes)
    frozen = datetime.now(timezone.utc).isoformat()
    print("FROZEN ALL PREDICTIONS; only now reading preliminary labels", flush=True)
    y, audit, truth_paths = base.audit_truth(basedir, b, private, rare, teams)
    write_json(out / "truth_audit.json", audit)
    write_json(out / "truth_sha256.json",
               {str(p.relative_to(root)): digest(p) for p in truth_paths})
    by_target, summary, group_summary = evaluate(
        prediction_list, y, teams, base, normal, private, rare, groups
    )
    by_target.to_csv(out / "by_target.csv", index=False, float_format="%.17g")
    summary.to_csv(out / "summary.csv", index=False, float_format="%.17g")
    group_summary.to_csv(out / "group_summary.csv", index=False, float_format="%.17g")
    baselines = [
        ("full", "without1_2", "without1_2", MAIN_BASELINE),
        ("full", "all20", "all20", SECONDARY_BASELINE_20),
        ("full", "all20", "without1_2", SECONDARY_EVAL18)
    ]
    for view, inference, evaluation, expected in baselines:
        hit = summary[(summary.background == view) &
                      (summary.inference == inference) &
                      (summary.evaluation == evaluation) &
                      (summary.method == "P01")]
        assert len(hit) == 1
        measured = float(hit.iloc[0].mean_tpr)
        if abs(measured - expected) > 1e-12:
            raise RuntimeError(f"Baseline {inference}/{evaluation}={measured}, wanted={expected}")
    write_json(out / "run_record.json", {
        "started_utc": now, "frozen_utc": frozen,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.perf_counter() - started,
        "prediction_conditions": len(hashes), "evaluation_rows": len(by_target),
        "primary_target_count": 18,
        "primary_inference_excludes_1_2": True,
        "truth_read_after_prediction_freeze": True,
        "python": platform.python_version(),
        "no_main_data": True, "no_competition_submission": True,
    })
    print(summary[(summary.background == "full") &
                  (summary.inference == "without1_2")].to_string(index=False))
    print("OUTPUT:", out)


if __name__ == "__main__":
    main()
