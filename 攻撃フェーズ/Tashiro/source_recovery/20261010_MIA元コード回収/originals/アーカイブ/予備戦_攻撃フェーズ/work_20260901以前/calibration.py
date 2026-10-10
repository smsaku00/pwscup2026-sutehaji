"""Cox CI を候補ごとに合わせる、最終段の affine/power calibration。

──────────────────────────────────────────────────────────────────────────
【提出物との対応】 このスクリプトが作るのは → prelim_team05/C_candidate.csv
    ＝ T05_Pool.zip に入る C.csv そのもの (sha256 1e1a384e…)

  入力は run_defense.py の出力 C_raw_seed5312.csv (sha256 df775af7…)。
  入出力の sha256 は prelim_team05/calibration_report.json に記録される。

  実際に使ったコマンド:
      python calibration.py --dist data_prelim_team05 \
        --candidate prelim_team05/C_raw_seed5312.csv \
        --out prelim_team05/C_candidate.csv \
        --report prelim_team05/calibration_report.json

  ※ 変換の強さ lambda は CLI 引数ではなく、このモジュールが探索して選ぶ
     （Pool では selected_lambda = 0.25 が採用された）。
  ※ anneal_defense.py と third_defense.py からは、CLI ではなく関数
     calibrate_cox_columns として --calibrate 指定時に呼ばれる
     （SA は lambda=1.0、Copula は lambda=0.5 が選ばれた）。
  提出物と生成コードの一覧 → prelim_team05/PROVENANCE.md
──────────────────────────────────────────────────────────────────────────

このモジュールは固定の変換係数を持たない。B と、転帰割り当てまで終わった候補 C を
毎回 Cox fit し直し、公式 U_spec の CI overlap を直接改善できる正の尺度だけを求める。
尺度変更で傷みうる U_gen/U_rare/U_valid は公式採点器で再評価し、総合 U が改善した
候補だけを返す。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import defense as D


_COLUMN_SPECS = (
    ("BMI", "BMI", False),
    ("SBP", "SBP", False),
    ("log_TG", "TG", True),
    ("HDL", "HDL", False),
    ("log_ALT", "ALT", True),
    ("FPG", "FPG", False),
)
_OUTCOME_COLUMNS = ("time", "onset", "death")
_A_GRID = np.linspace(0.85, 1.15, 3001)
_LAMBDA_GRID = (1.0, 0.75, 0.5, 0.25)
_B_GRID_SIZE = 1201
_B_SEARCH_SD = 0.15
_U_IMPROVEMENT_EPS = 1e-6


@dataclass(frozen=True)
class _FitTerm:
    beta: float
    se: float
    lo: float
    hi: float
    significant: bool


def _as_float(value: Any) -> float:
    return float(value)


def _fit_terms(df: pd.DataFrame) -> tuple[Any, dict[str, _FitTerm]]:
    cph = D.fit_cox_raw(df, "onset")
    terms: dict[str, _FitTerm] = {}
    for name in D.COX_COLS:
        lo, hi = (float(v) for v in cph.confidence_intervals_.loc[name])
        terms[name] = _FitTerm(
            beta=float(cph.params_[name]),
            se=float(cph.standard_errors_[name]),
            lo=lo,
            hi=hi,
            significant=bool((lo > 0.0) or (hi < 0.0)),
        )
    return cph, terms


def _terms_report(terms: dict[str, _FitTerm]) -> dict[str, dict[str, float | bool]]:
    return {
        name: {
            "beta": term.beta,
            "se": term.se,
            "ci_low": term.lo,
            "ci_high": term.hi,
            "significant": term.significant,
            "z": term.beta / term.se,
        }
        for name, term in terms.items()
    }


def _io_overlap(lo_b: float, hi_b: float, lo_c: float, hi_c: float,
                beta_b: float, beta_c: float, se_b: float) -> float:
    """公式 scorer の1係数 IO と同じ計算。"""
    width_b, width_c = hi_b - lo_b, hi_c - lo_c
    if width_b > 0.0 and width_c > 0.0:
        overlap = min(hi_b, hi_c) - max(lo_b, lo_c)
        io = 0.5 * (overlap / width_b + overlap / width_c)
    else:
        io = 1.0 - min(abs(beta_b - beta_c) / (4.0 * max(se_b, 1e-12)), 1.0)
    return float(np.clip(io, 0.0, 1.0))


def _coefficient_contribution(b: _FitTerm, c: _FitTerm) -> tuple[float, bool, str | None]:
    if b.significant != c.significant:
        return 0.0, False, "significance_mismatch"
    if b.significant and c.significant and np.sign(b.beta) != np.sign(c.beta):
        return 0.0, False, "sign_mismatch"
    return _io_overlap(b.lo, b.hi, c.lo, c.hi, b.beta, c.beta, b.se), True, None


def _scaled_io(b: _FitTerm, c: _FitTerm, a: float) -> float:
    return _io_overlap(
        b.lo, b.hi, c.lo / a, c.hi / a, b.beta, c.beta / a, b.se
    )


def _domain(values: np.ndarray, is_log: bool) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("affine calibration対象列に非有限値があります")
    if is_log:
        if bool((x <= 0.0).any()):
            raise ValueError("log calibration対象列に0以下の値があります")
        return np.log(x)
    return x


def _domain_range(column: str, is_log: bool) -> tuple[float, float]:
    lo, hi = (float(v) for v in D.RANGES[column])
    return (float(np.log(lo)), float(np.log(hi))) if is_log else (lo, hi)


def _feasible_shift_interval(z_c: np.ndarray, a: float, column: str,
                             is_log: bool) -> tuple[float, float] | None:
    lo, hi = _domain_range(column, is_log)
    b_lo = lo - a * float(np.min(z_c))
    b_hi = hi - a * float(np.max(z_c))
    if b_lo > b_hi:
        return None
    return b_lo, b_hi


def _count_reference_matches(values: np.ndarray, reference: np.ndarray,
                             rel_tol: float) -> int:
    """reference の何値が values に相対許容内で存在するかを数える。"""
    cv = np.sort(np.asarray(values, dtype=float))
    rv = np.asarray(reference, dtype=float)
    pos = np.searchsorted(cv, rv)
    best = np.full(len(rv), np.inf, dtype=float)
    for p in (pos - 1, pos):
        ok = (p >= 0) & (p < len(cv))
        if not bool(ok.any()):
            continue
        clipped = np.clip(p, 0, max(len(cv) - 1, 0))
        d = np.full(len(rv), np.inf, dtype=float)
        d[ok] = np.abs(cv[clipped[ok]] - rv[ok])
        best = np.minimum(best, d)
    tol = rel_tol * np.maximum(np.abs(rv), 1e-9)
    return int(np.sum(best <= tol))


def _to_output(z: np.ndarray, is_log: bool) -> np.ndarray:
    return np.exp(z) if is_log else z


def _choose_shift(z_c: np.ndarray, z_b: np.ndarray, raw_b: np.ndarray, a: float,
                  column: str, is_log: bool, exact_rel_tol: float) -> dict[str, Any] | None:
    feasible = _feasible_shift_interval(z_c, a, column, is_log)
    if feasible is None:
        return None
    b_lo, b_hi = feasible
    base = a * z_c
    b0 = float(np.median(z_b) - np.median(base))
    sd_b = float(np.std(z_b, ddof=0))
    radius = _B_SEARCH_SD * sd_b
    search_lo = max(b_lo, b0 - radius)
    search_hi = min(b_hi, b0 + radius)
    if search_lo > search_hi:
        return None
    if search_lo == search_hi:
        b_grid = np.asarray([search_lo], dtype=float)
    else:
        b_grid = np.linspace(search_lo, search_hi, _B_GRID_SIZE)

    # IO とは独立に、周辺分布を最も壊さない shift を選ぶ。log は単調変換なので
    # log-domain KS と raw-domain KS は等しい。
    ks_values = np.asarray([
        float(ks_2samp(base + shift, z_b).statistic) for shift in b_grid
    ])
    order = np.lexsort((np.abs(b_grid - b0), ks_values))
    for idx in order:
        shift = float(b_grid[int(idx)])
        output = _to_output(base + shift, is_log)
        matches = _count_reference_matches(output, raw_b, exact_rel_tol)
        if matches == 0:
            return {
                "b": shift,
                "ks": float(ks_values[int(idx)]),
                "exact_matches": 0,
                "b0": b0,
                "search_interval": [float(search_lo), float(search_hi)],
            }
    return None


def _choose_scale(b_term: _FitTerm, c_term: _FitTerm, z_c: np.ndarray,
                  column: str, is_log: bool) -> dict[str, Any]:
    io_before, repairable, issue = _coefficient_contribution(b_term, c_term)
    if not repairable:
        return {
            "a_star": 1.0,
            "io_before": io_before,
            "io_at_a_star": io_before,
            "repairable": False,
            "issue": issue,
            "boundary_optimum": False,
        }

    feasible = np.asarray([
        _feasible_shift_interval(z_c, float(a), column, is_log) is not None
        for a in _A_GRID
    ], dtype=bool)
    io = np.full(len(_A_GRID), -np.inf, dtype=float)
    io[feasible] = np.asarray([
        _scaled_io(b_term, c_term, float(a)) for a in _A_GRID[feasible]
    ])
    if not bool(np.isfinite(io).any()):
        return {
            "a_star": 1.0,
            "io_before": io_before,
            "io_at_a_star": io_before,
            "repairable": False,
            "issue": "no_feasible_scale",
            "boundary_optimum": False,
        }
    best_io = float(np.max(io))
    tied = np.flatnonzero(np.isclose(io, best_io, rtol=0.0, atol=1e-12))
    best_idx = int(tied[np.argmin(np.abs(_A_GRID[tied] - 1.0))])
    a_star = float(_A_GRID[best_idx])
    return {
        "a_star": a_star,
        "io_before": io_before,
        "io_at_a_star": best_io,
        "repairable": True,
        "issue": None,
        "boundary_optimum": bool(best_idx in (0, len(_A_GRID) - 1)),
    }


def _score_report(score: dict[str, Any]) -> dict[str, Any]:
    spec = score.get("detail", {}).get("U_spec", {})
    rare = score.get("detail", {}).get("U_rare", {})
    return {
        "U": float(score["U"]),
        "facets": {name: float(value) for name, value in score["facets"].items()},
        "io_part": (float(spec["io_part"]) if spec.get("io_part") is not None else None),
        "tstr_cal": (float(spec["tstr_cal"]) if spec.get("tstr_cal") is not None else None),
        "cox_converged": bool(spec.get("cox_converged", False)),
        "rare_detected_C": (int(rare["rare_detected_C"])
                            if rare.get("rare_detected_C") is not None else None),
        "rare_gate": (int(rare["rare_gate"]) if rare.get("rare_gate") is not None else None),
    }


def _validate_candidate(candidate: pd.DataFrame, original: pd.DataFrame, bench: Any,
                        exact_rel_tol: float) -> dict[str, Any]:
    numeric_columns = (
        "age", "sex", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG",
        "time", "onset", "death",
    )
    all_numeric_finite = all(
        bool(np.isfinite(candidate[col].to_numpy(dtype=float)).all())
        for col in numeric_columns
    )
    finite = all_numeric_finite
    in_range = True
    ranges: dict[str, dict[str, float | bool]] = {}
    exact: dict[str, int] = {}
    for _, column, _ in _COLUMN_SPECS:
        values = candidate[column].to_numpy(dtype=float)
        col_finite = bool(np.isfinite(values).all())
        lo, hi = (float(v) for v in D.RANGES[column])
        col_range = bool(col_finite and (values >= lo).all() and (values <= hi).all())
        finite &= col_finite
        in_range &= col_range
        ranges[column] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "finite": col_finite,
            "in_range": col_range,
        }
        exact[column] = _count_reference_matches(
            values, bench.B[column].to_numpy(dtype=float), exact_rel_tol
        )

    outcomes_unchanged = all(
        np.array_equal(candidate[col].to_numpy(), original[col].to_numpy())
        for col in _OUTCOME_COLUMNS
    )
    age_valid = bool(candidate["age"].between(*D.AGE_RANGE).all())
    binary_valid = all(
        bool(candidate[col].isin([0, 1]).all())
        for col in ("sex", "smoking", "onset", "death")
    )
    outcome_combinations_valid = bool(
        ((candidate["onset"] + candidate["death"]) <= 1).all()
    )
    time_values = candidate["time"].to_numpy(dtype=float)
    time_valid = bool(
        np.isfinite(time_values).all()
        and (time_values >= 0.0).all()
        and (time_values <= D.HORIZON).all()
    )
    exact_total = int(sum(exact.values()))
    return {
        "finite": bool(finite),
        "in_range": bool(in_range),
        "outcomes_unchanged": bool(outcomes_unchanged),
        "age_valid": age_valid,
        "binary_valid": binary_valid,
        "outcome_combinations_valid": outcome_combinations_valid,
        "time_valid": time_valid,
        "ranges": ranges,
        "exact_matches": exact,
        "exact_matches_total": exact_total,
        "valid": bool(
            finite and in_range and outcomes_unchanged and age_valid and binary_valid
            and outcome_combinations_valid and time_valid and exact_total == 0
        ),
    }


def _apply_lambda(C: pd.DataFrame, bench: Any, domains_c: dict[str, np.ndarray],
                  domains_b: dict[str, np.ndarray], scale_plan: dict[str, dict[str, Any]],
                  lam: float, exact_rel_tol: float) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    out = C.copy(deep=True)
    factors: dict[str, dict[str, Any]] = {}
    for term_name, column, is_log in _COLUMN_SPECS:
        plan = scale_plan[term_name]
        if not bool(plan["repairable"]):
            factors[column] = {
                "term": term_name,
                "mode": "log_power" if is_log else "affine",
                "a": 1.0,
                "b": 0.0,
                "ks": None,
                "exact_matches": _count_reference_matches(
                    out[column].to_numpy(dtype=float),
                    bench.B[column].to_numpy(dtype=float), exact_rel_tol,
                ),
                "applied": False,
                "issue": plan["issue"],
            }
            continue

        a = float(1.0 + lam * (float(plan["a_star"]) - 1.0))
        shift = _choose_shift(
            domains_c[column], domains_b[column],
            bench.B[column].to_numpy(dtype=float), a, column, is_log, exact_rel_tol,
        )
        if shift is None:
            return None, {
                "lambda": float(lam),
                "valid": False,
                "reason": f"exact-free feasible shift not found: {column}",
                "factors": factors,
            }
        transformed_domain = a * domains_c[column] + float(shift["b"])
        transformed = _to_output(transformed_domain, is_log)
        out[column] = transformed
        factors[column] = {
            "term": term_name,
            "mode": "log_power" if is_log else "affine",
            "a": a,
            "b": float(shift["b"]),
            "ks": float(shift["ks"]),
            "exact_matches": int(shift["exact_matches"]),
            "applied": True,
            "issue": None,
            "predicted_io": _scaled_io(
                plan["b_term"], plan["c_term"], a
            ),
        }
    return out, {"lambda": float(lam), "valid": True, "factors": factors}


def calibrate_cox_columns(
    C: pd.DataFrame,
    bench: Any,
    *,
    exact_rel_tol: float = 1e-9,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """候補 C の連続 Cox 列を調整し、公式 U が改善した場合だけ採用する。

    Parameters
    ----------
    C:
        ``time/onset/death`` の生成と任意の outcome refine が完了した候補。
    bench:
        ``harness.Bench`` 互換オブジェクト。``B`` と ``score`` を使う。
    exact_rel_tol:
        B の臨床値との完全一致を拒否する相対許容。

    Returns
    -------
    (C_best, report):
        改善を公式採点で確認できなければ、内容を変えない C のコピーを返す。
    """
    if exact_rel_tol < 0.0 or not np.isfinite(exact_rel_tol):
        raise ValueError("exact_rel_tol は有限な0以上で指定してください")
    required = {column for _, column, _ in _COLUMN_SPECS} | set(_OUTCOME_COLUMNS)
    missing = sorted(required - set(C.columns))
    if missing:
        raise ValueError(f"C に必要な列がありません: {missing}")

    original = C.copy(deep=True)
    base_score_raw = bench.score(original)
    base_score = _score_report(base_score_raw)
    _, b_terms = _fit_terms(bench.B)
    try:
        _, c_terms = _fit_terms(original)
    except Exception as exc:  # scorerと同様、Cox不成立なら安全にrollbackする
        report = {
            "accepted": False,
            "reason": f"candidate Cox fit failed: {type(exc).__name__}: {exc}",
            "exact_rel_tol": float(exact_rel_tol),
            "pre": base_score,
            "post": base_score,
            "coefficients": {"B": _terms_report(b_terms), "pre": None, "post": None},
            "scale_plan": {},
            "attempts": [],
        }
        return original, report

    domains_c: dict[str, np.ndarray] = {}
    domains_b: dict[str, np.ndarray] = {}
    scale_plan: dict[str, dict[str, Any]] = {}
    for term_name, column, is_log in _COLUMN_SPECS:
        z_c = _domain(original[column].to_numpy(dtype=float), is_log)
        z_b = _domain(bench.B[column].to_numpy(dtype=float), is_log)
        domains_c[column], domains_b[column] = z_c, z_b
        plan = _choose_scale(b_terms[term_name], c_terms[term_name], z_c, column, is_log)
        plan["b_term"] = b_terms[term_name]
        plan["c_term"] = c_terms[term_name]
        scale_plan[term_name] = plan

    base_validation = _validate_candidate(original, original, bench, exact_rel_tol)
    attempts: list[dict[str, Any]] = []
    eligible: list[tuple[float, float, pd.DataFrame, dict[str, Any], dict[str, Any]]] = []
    for lam in _LAMBDA_GRID:
        candidate, attempt = _apply_lambda(
            original, bench, domains_c, domains_b, scale_plan, lam, exact_rel_tol
        )
        if candidate is None:
            attempts.append(attempt)
            continue
        validation = _validate_candidate(candidate, original, bench, exact_rel_tol)
        attempt["validation"] = validation
        if not validation["valid"]:
            attempt["valid"] = False
            attempt["reason"] = "candidate validation failed"
            attempts.append(attempt)
            continue
        score_raw = bench.score(candidate)
        score = _score_report(score_raw)
        attempt["score"] = score
        rare_gate_ok = (
            score["rare_gate"] is None
            or (score["rare_detected_C"] is not None
                and score["rare_detected_C"] >= score["rare_gate"])
        )
        attempt["rare_gate_ok"] = bool(rare_gate_ok)
        attempt["valid"] = bool(score["cox_converged"] and rare_gate_ok)
        if not attempt["valid"]:
            attempt["reason"] = "official score convergence/rare gate failed"
            attempts.append(attempt)
            continue
        attempts.append(attempt)
        eligible.append((float(score["U"]), -float(lam), candidate, attempt, score_raw))

    # U同点なら変形の小さいlambdaを選ぶ。baseは明示的なrollback候補。
    selected_candidate = original
    selected_attempt: dict[str, Any] | None = None
    selected_score_raw = base_score_raw
    if eligible:
        best_u = max(item[0] for item in eligible)
        tied = [item for item in eligible if abs(item[0] - best_u) <= 1e-12]
        _, _, selected_candidate, selected_attempt, selected_score_raw = max(tied, key=lambda x: x[1])

    accepted = bool(
        selected_attempt is not None
        and float(selected_score_raw["U"]) > float(base_score_raw["U"]) + _U_IMPROVEMENT_EPS
    )
    if not accepted:
        selected_candidate = original
        selected_attempt = None
        selected_score_raw = base_score_raw

    post_score = _score_report(selected_score_raw)
    try:
        _, post_terms = _fit_terms(selected_candidate)
    except Exception:
        post_terms = c_terms

    # dataclassを含む内部項目はreportへ直接出さない。
    public_plan = {
        term: {key: value for key, value in plan.items() if key not in {"b_term", "c_term"}}
        for term, plan in scale_plan.items()
    }
    report = {
        "accepted": accepted,
        "reason": ("official U improved" if accepted else "rollback: official U did not improve"),
        "exact_rel_tol": float(exact_rel_tol),
        "pre": base_score,
        "post": post_score,
        "improvement": float(post_score["U"] - base_score["U"]),
        "selected_lambda": (float(selected_attempt["lambda"]) if selected_attempt else 0.0),
        "selected_factors": (selected_attempt["factors"] if selected_attempt else {
            column: {
                "term": term_name,
                "mode": "log_power" if is_log else "affine",
                "a": 1.0,
                "b": 0.0,
                "applied": False,
            }
            for term_name, column, is_log in _COLUMN_SPECS
        }),
        "coefficients": {
            "B": _terms_report(b_terms),
            "pre": _terms_report(c_terms),
            "post": _terms_report(post_terms),
        },
        "scale_plan": public_plan,
        "base_validation": base_validation,
        "post_validation": _validate_candidate(
            selected_candidate, original, bench, exact_rel_tol
        ),
        "attempts": attempts,
    }
    return selected_candidate.copy(deep=True), report


__all__ = ["calibrate_cox_columns"]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="候補Cの最終Cox補正（公式U改善時だけ採用）")
    ap.add_argument("--dist", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dist = args.dist.expanduser().resolve()
    source = args.candidate.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if out_path == report_path:
        ap.error("--out と --report には別のパスを指定してください")
    if source in {out_path, report_path}:
        ap.error("入力候補を直接上書きしません。別の出力パスを指定してください")
    if not source.is_file():
        ap.error(f"候補Cがありません: {source}")
    if not args.force:
        existing = [str(p) for p in (out_path, report_path) if p.exists()]
        if existing:
            ap.error("既存ファイルを上書きしません: " + ", ".join(existing))

    # import時の既定distへ依存させず、CLIで指定された世界を必ず明示する。
    from harness import Bench

    bench = Bench(dist)
    original_bytes = source.read_bytes()
    candidate = pd.read_csv(source)
    calibrated, report = calibrate_cox_columns(candidate, bench)
    csv_bytes = calibrated.to_csv(index=False).encode("utf-8")
    report["dataset"] = {
        "dist": str(dist),
        "candidate_source": str(source),
        "candidate_source_sha256": _sha256_bytes(original_bytes),
        "candidate_output_sha256": _sha256_bytes(csv_bytes),
    }
    report_text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(csv_bytes)
    report_path.write_text(report_text, encoding="utf-8")
    print(f"accepted={report['accepted']} U={report['post']['U']:.6f} -> {out_path}")


if __name__ == "__main__":
    main()
