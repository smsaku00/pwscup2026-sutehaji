#!/usr/bin/env python3
"""Fixed population-CDF counterexample; no competition data or scores are read."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from zoneinfo import ZoneInfo

from scipy.integrate import quad
from scipy.optimize import brentq


SCRIPT = Path(__file__).resolve()
OUT = SCRIPT.parent / "理想CDF縮約の条件付き尾率結果"
PLAN = OUT / "事前固定計画.json"
PIS = (0.001, 0.01, 0.1)
LAMBDAS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 2000, 5000, "global")
N = 1000
ALPHA = 0.01
OUTPUT_NAMES = (
    "固定グリッドの群別尾率.csv",
    "独立積分と分位点の検証.csv",
    "数式と解釈の制限.json",
    "条件付き尾率の曲線.svg",
    "条件付き尾率の曲線.png",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def write_json(path: Path, obj: object) -> None:
    with path.open("x", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")


def fixed_plan() -> dict:
    return {
        "kind": "preregistered_fixed_mathematical_counterexample",
        "created_jst": now(),
        "source_sha256": sha(SCRIPT),
        "N": N,
        "alpha": ALPHA,
        "pi_grid": list(PIS),
        "lambda_grid": list(LAMBDAS),
        "group0": "Uniform[0,1], mass 1-pi, fixed count n0=(1-pi)*N",
        "group1": "Uniform[2,3], mass pi, fixed count n1=pi*N",
        "known_population_CDFs": True,
        "empirical_CDF_or_sample_uncertainty": False,
        "weights": "w_g=n_g/(n_g+lambda); global means both weights exactly zero",
        "transform": "U_g=w_g*F_g(X_g)+(1-w_g)*F(X_g)",
        "F": "(1-pi)*F0+pi*F1",
        "a0": "1-pi*(1-w0)",
        "a1": "pi+(1-pi)*w1",
        "distribution_of_U0": "Uniform[0,a0]",
        "distribution_of_U1": "Uniform[1-a1,1]",
        "tail0": "max(0,1-(1-alpha)/a0)",
        "tail1": "min(1,alpha/a1)",
        "overall_tail": "(1-pi)*tail0+pi*tail1",
        "verification": [
            "independent exact Fraction arithmetic at every fixed case",
            "numerical integration of strict upper-tail indicator on original X support, with roots found by brentq",
            "numerical inverse of the explicit two-population CDF blend at probabilities .1/.5/.9",
            "lambda0 conditional and overall tails alpha; global overall tail alpha",
        ],
        "data_files_read": [],
        "membership_labels_or_official_scores": False,
        "calibration_or_model_or_lambda_selection": False,
        "finite_sample_conformal_p_value_implementation": False,
        "real_data_or_MIA_performance_claim": False,
        "expected_grid_rows": len(PIS) * len(LAMBDAS),
        "expected_group_verification_rows": len(PIS) * len(LAMBDAS) * 2,
        "expected_outputs": list(OUTPUT_NAMES) + ["実行記録.json"],
        "figure_visual_review": "root will inspect static PNG/SVG; script checks only mathematical and file integrity",
    }


def coefficients(pi: float, lam: int | str) -> tuple:
    n1 = int(round(pi * N))
    n0 = N - n1
    assert math.isclose(n1, pi * N, abs_tol=0)
    if lam == "global":
        w0 = w1 = 0.0
    else:
        w0, w1 = n0 / (n0 + lam), n1 / (n1 + lam)
    a0, a1 = 1 - pi * (1 - w0), pi + (1 - pi) * w1
    return n0, n1, w0, w1, a0, a1


def closed_case(pi: float, lam: int | str) -> dict:
    n0, n1, w0, w1, a0, a1 = coefficients(pi, lam)
    tail0 = max(0.0, 1 - (1 - ALPHA) / a0)
    tail1 = min(1.0, ALPHA / a1)
    total = (1 - pi) * tail0 + pi * tail1
    return {
        "pi": pi, "N": N, "n0": n0, "n1": n1,
        "lambda_label": str(lam), "lambda_numeric": "" if lam == "global" else lam,
        "alpha": ALPHA, "w0": w0, "w1": w1, "a0": a0, "a1": a1,
        "group0_tail_probability": tail0,
        "group1_tail_probability": tail1,
        "overall_tail_probability": total,
        "group0_tail_over_alpha": tail0 / ALPHA,
        "group1_tail_over_alpha": tail1 / ALPHA,
        "overall_tail_over_alpha": total / ALPHA,
        "overall_tail_minus_alpha": total - ALPHA,
    }


def original_support_blend(x: float, pi: float, weight: float, group: int) -> float:
    """Explicit original-score CDFs, independently of a0/a1 formulas."""
    f0 = max(0.0, min(1.0, x))
    f1 = max(0.0, min(1.0, x - 2.0))
    fg = (f0, f1)[group]
    f = (1.0 - pi) * f0 + pi * f1
    return weight * fg + (1.0 - weight) * f


def exact_tails(pi: float, lam: int | str) -> tuple[Fraction, Fraction, Fraction]:
    p = Fraction(str(pi))
    alpha = Fraction(str(ALPHA))
    n0, n1 = (1 - p) * N, p * N
    if lam == "global":
        w0 = w1 = Fraction(0)
    else:
        w0, w1 = n0 / (n0 + lam), n1 / (n1 + lam)
    # Use original affine blend slopes/intercepts rather than production coefficients.
    slope0, intercept0 = w0 + (1 - w0) * (1 - p), Fraction(0)
    slope1, intercept1 = w1 + (1 - w1) * p, (1 - w1) * (1 - p)
    tails = []
    for slope, intercept in ((slope0, intercept0), (slope1, intercept1)):
        threshold_in_v = (1 - alpha - intercept) / slope
        tails.append(max(Fraction(0), min(Fraction(1), 1 - threshold_in_v)))
    return tails[0], tails[1], (1 - p) * tails[0] + p * tails[1]


def verify_case(row: dict) -> list[dict]:
    pi, lam = row["pi"], row["lambda_label"]
    if lam != "global":
        lam = int(lam)
    exact0, exact1, exact_total = exact_tails(pi, lam)
    assert abs(float(exact_total) - row["overall_tail_probability"]) < 2e-14
    result = []
    for group, exact in enumerate((exact0, exact1)):
        lo, hi = (0.0, 1.0) if group == 0 else (2.0, 3.0)
        weight = row[f"w{group}"]
        blend = lambda x: original_support_blend(x, pi, weight, group)
        threshold = 1 - ALPHA
        low_u, high_u = blend(lo), blend(hi)
        root = None
        if threshold <= low_u:
            integral = quad(lambda x: 1.0, lo, hi, epsabs=1e-12, epsrel=1e-12)[0]
        elif threshold >= high_u:
            integral = quad(lambda x: 0.0, lo, hi, epsabs=1e-12, epsrel=1e-12)[0]
        else:
            root = brentq(lambda x: blend(x) - threshold, lo, hi, xtol=5e-15, rtol=1e-14)
            integral = quad(lambda x: float(blend(x) > threshold), lo, hi,
                            points=[root], epsabs=1e-12, epsrel=1e-12)[0]
        quantile_error = 0.0
        for v in (0.1, 0.5, 0.9):
            u_quantile = low_u + (high_u - low_u) * v
            x_back = brentq(lambda x: blend(x) - u_quantile, lo, hi, xtol=5e-15, rtol=1e-14)
            quantile_error = max(quantile_error, abs(x_back - (lo + v)))
        closed = row[f"group{group}_tail_probability"]
        exact_error, integral_error = abs(float(exact) - closed), abs(integral - closed)
        assert exact_error < 2e-14 and integral_error < 2e-12 and quantile_error < 2e-12
        result.append({
            "pi": pi, "lambda_label": str(lam), "group": group,
            "closed_tail_probability": closed,
            "exact_tail_numerator": exact.numerator, "exact_tail_denominator": exact.denominator,
            "integral_tail_probability": integral,
            "exact_abs_error": exact_error, "integral_abs_error": integral_error,
            "inverse_quantile_max_abs_error": quantile_error,
            "original_score_threshold_root": "" if root is None else root,
        })
    if lam == 0:
        assert all(abs(row[k] - ALPHA) < 2e-14 for k in (
            "group0_tail_probability", "group1_tail_probability", "overall_tail_probability"))
    if lam == "global":
        assert abs(row["overall_tail_probability"] - ALPHA) < 2e-14
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure(rows: list[dict]) -> None:
    with tempfile.TemporaryDirectory(prefix="population-cdf-mpl-", dir="/private/tmp") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter

        fields = ("group0_tail_probability", "group1_tail_probability", "overall_tail_probability")
        titles = ("Group 0: lower-score population", "Group 1: upper-score population", "Population mixture")
        colors = ("#2563eb", "#ea580c", "#16a34a")
        fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.4), gridspec_kw={"width_ratios": [7, 1.1]}, sharey="row")
        for i, (field, title) in enumerate(zip(fields, titles)):
            ax, limit_ax = axes[i]
            for pi, color in zip(PIS, colors):
                finite = [r for r in rows if r["pi"] == pi and r["lambda_label"] != "global"]
                endpoint = next(r for r in rows if r["pi"] == pi and r["lambda_label"] == "global")
                ax.plot([r["lambda_numeric"] for r in finite], [r[field] for r in finite],
                        marker="o", markersize=4, linewidth=1.8, color=color, label=f"pi = {pi:g}")
                limit_ax.plot(0, endpoint[field], marker="D", markersize=6, color=color)
            for a in (ax, limit_ax):
                a.axhline(ALPHA, color="#475569", linestyle="--", linewidth=1, zorder=0)
                a.yaxis.set_major_formatter(PercentFormatter(1.0))
                a.grid(axis="y", color="#e2e8f0", linewidth=0.7)
                a.spines[["top", "right"]].set_visible(False)
            ax.set_xscale("symlog", linthresh=25)
            ax.set_xticks((0, 25, 100, 500, 1000, 5000), labels=("0", "25", "100", "500", "1000", "5000"))
            ax.set_title(title, loc="left", fontsize=11)
            ax.set_ylabel("P(U > 0.99)")
            limit_ax.set_xticks((0,), labels=("global",))
            limit_ax.set_xlim(-0.5, 0.5)
            limit_ax.tick_params(axis="y", left=False, labelleft=False)
            if i == 1:
                ax.set_yscale("log")
                ax.set_ylim(0.008, 1.35)
                ax.set_yticks((0.01, 0.1, 1.0))
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            else:
                ax.set_ylim(bottom=0)
        axes[0, 0].legend(loc="lower left", ncols=3, fontsize=9, frameon=False)
        axes[2, 0].set_xlabel("lambda (fixed grid; symlog axis)")
        fig.suptitle("Population-CDF shrinkage: conditional upper tails", x=0.08, ha="left", fontsize=15)
        fig.text(0.08, 0.939, "N = 1000, alpha = 0.01.  Group 0 = Unif[0,1]; group 1 = Unif[2,3].", fontsize=10)
        fig.text(0.08, 0.027,
                 "w_g = n_g / (n_g + lambda). Global markers set both weights to zero. Dashed line = 1%.\n"
                 "Known population CDFs with fixed count-derived weights; no finite-sample p values or real-data performance claim.",
                 fontsize=9, color="#334155")
        fig.subplots_adjust(top=0.89, bottom=0.10, left=0.08, right=0.98, hspace=0.35, wspace=0.10)
        fig.savefig(OUT / "条件付き尾率の曲線.svg", metadata={"Date": None})
        fig.savefig(OUT / "条件付き尾率の曲線.png", dpi=180)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-plan", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.write_plan:
        write_json(PLAN, fixed_plan())
        print(json.dumps({"plan_written": str(PLAN), "plan_sha256": sha(PLAN), "source_sha256": sha(SCRIPT)}))
        return
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["source_sha256"] == sha(SCRIPT), "source changed after preregistration"
    assert plan["pi_grid"] == list(PIS) and plan["lambda_grid"] == list(LAMBDAS)
    assert plan["N"] == N and plan["alpha"] == ALPHA
    assert all(not (OUT / name).exists() for name in (*OUTPUT_NAMES, "実行記録.json")), "do not overwrite outputs"
    started = now()
    rows = [closed_case(pi, lam) for pi in PIS for lam in LAMBDAS]
    checks = [check for row in rows for check in verify_case(row)]
    assert len(rows) == plan["expected_grid_rows"] and len(checks) == plan["expected_group_verification_rows"]
    write_csv(OUT / OUTPUT_NAMES[0], rows)
    write_csv(OUT / OUTPUT_NAMES[1], checks)
    equations = {
        "kind": "mathematical_explanation_material_not_final_experiment_prose",
        "assumptions": {
            "disjoint_ordered_nonmember_score_supports": {"group0": [0, 1], "group1": [2, 3]},
            "group1_mass": list(PIS), "N": N, "alpha": ALPHA,
            "n0": "(1-pi)*N", "n1": "pi*N", "counts_fixed_not_random": True,
            "exact_population_CDFs_used": True,
        },
        "derivation": [
            "Let V=F_g(X_g)~Uniform[0,1]. On group0 support F(X0)=(1-pi)*V; on group1 support F(X1)=1-pi+pi*V.",
            "U0=[w0+(1-w0)*(1-pi)]*V=a0*V, a0=1-pi*(1-w0).",
            "U1=(1-w1)*(1-pi)+[w1+(1-w1)*pi]*V=1-a1+a1*V, a1=pi+(1-pi)*w1.",
            "P(U0>1-alpha)=max(0,1-(1-alpha)/a0). P(U1>1-alpha)=min(1,alpha/a1).",
            "The unconditional tail is (1-pi)*tail0+pi*tail1. Group counts fix shrinkage weights; probabilities refer to population draws.",
            "For lambda0, both local weights are1 and every tail equals alpha. For global, both weights are0: tail0=max(0,(alpha-pi)/(1-pi)), tail1=min(1,alpha/pi), overall=alpha.",
            "For any lambda>0 and 0<pi<1, a1<1 and tail1>alpha. Therefore fixed shrinkage does not guarantee a conditional alpha upper tail, even with perfectly known local and global CDFs.",
            "Because w0 and w1 depend on unequal counts, their mixture is not a single common transformed global CDF. The unconditional tail at finite lambda can also differ from alpha.",
        ],
        "limits": [
            "Constructed population counterexample only. It is not an empirical-CDF, finite-sample conformal-p-value, exchangeability, or estimated-CDF uncertainty calculation.",
            "The score groups are deliberately disjoint and ordered. Competition score distributions need not have these supports or proportions.",
            "No lower bound or estimate of actual false positives, MIA accuracy, attack suitability, or real-data tail behavior follows from this table.",
            "No calibration plan, lambda, group mass, model, or seed is selected from these results. All grid values were fixed before calculation.",
            "The counts N and n_g are hypothetical deterministic inputs to the weights, not observed records. No personal records, memberships, or official scores are read or written.",
            "100 and500 refer here only to this one-level two-group blend, not to a numerical reconstruction of the actual multi-level/mixed-source/fold-averaged pipeline.",
            "Continuous distributions make strict > and >= tails identical up to probability-zero endpoints. This does not establish the same property for discrete empirical ranks.",
            "Overall global PIT uniformity does not imply conditional group uniformity. Conversely lambda0 ideal local PIT uniformity does not supply a finite-sample guarantee for estimated local CDFs.",
        ],
        "verification": {
            "group_cases": len(checks),
            "max_exact_fraction_abs_error": max(r["exact_abs_error"] for r in checks),
            "max_independent_integral_abs_error": max(r["integral_abs_error"] for r in checks),
            "max_inverse_quantile_abs_error": max(r["inverse_quantile_max_abs_error"] for r in checks),
            "lambda0_and_global_endpoint_checks": "PASS",
            "visual_review": "pending root inspection; no visual-pass claim by script",
        },
        "source_sha256": sha(SCRIPT), "plan_sha256": sha(PLAN),
    }
    write_json(OUT / OUTPUT_NAMES[2], equations)
    figure(rows)
    # Saved text artifacts are independently reread for counts, labels, and probability ranges.
    for filename, count in ((OUTPUT_NAMES[0], len(rows)), (OUTPUT_NAMES[1], len(checks))):
        with (OUT / filename).open(newline="", encoding="utf-8") as fh:
            saved = list(csv.DictReader(fh))
        assert len(saved) == count
        assert {(r["pi"], r["lambda_label"]) for r in saved} == {
            (str(pi), str(lam)) for pi in PIS for lam in LAMBDAS}
    for row in rows:
        assert all(0 <= row[k] <= 1 for k in (
            "group0_tail_probability", "group1_tail_probability", "overall_tail_probability"))
    record = {
        "status": "completed", "started_jst": started, "completed_jst": now(),
        "command": [sys.executable, "-B", str(SCRIPT)],
        "python": sys.version, "platform": platform.platform(),
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
        "source_sha256": sha(SCRIPT), "plan_sha256": sha(PLAN),
        "data_input_paths": [], "competition_data_or_labels_or_scores_accessed": False,
        "output_sha256": {name: sha(OUT / name) for name in OUTPUT_NAMES},
        "tail_rows": len(rows), "group_verification_rows": len(checks),
        "verification": equations["verification"], "existing_experiments_modified": False,
        "external_action": False, "model_or_parameter_selection": False,
    }
    write_json(OUT / "実行記録.json", record)
    print(json.dumps({"status": "completed", "tail_rows": len(rows), "checks": len(checks),
                      "verification": equations["verification"], "source_sha256": sha(SCRIPT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
