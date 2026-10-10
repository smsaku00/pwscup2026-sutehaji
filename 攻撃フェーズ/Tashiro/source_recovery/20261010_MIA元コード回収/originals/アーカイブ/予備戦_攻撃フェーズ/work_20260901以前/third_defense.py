"""第三の防御候補: RareCopula（希少混合・正則化 Gaussian copula 合成）。

──────────────────────────────────────────────────────────────────────────
【提出物との対応】 このスクリプトが作るのは → prelim_team05/methods/C_copula.csv
    ＝ T05_Copula.zip に入る C.csv そのもの (sha256 6a5a29e7…)
    ログ: prelim_team05/methods/copula.json
          (method = "RareCopula"、driver_sha256 は本ファイル自身の sha256)

  実際に使ったコマンド:
      python third_defense.py --dist data_prelim_team05 --seed 6314 \
        --rare-offset 4 --rare-shrink 0.45 --near-q 0.05 \
        --moment-strength 0.5 --refine-steps 300 --calibrate \
        --out prelim_team05/methods/C_copula.csv \
        --log prelim_team05/methods/copula.json

  Pool / SA と違い A_bg の行を選ばないので、time 多重集合も一致しない
  ＝ 独立な第三系統であることが出力から確認できる。
  提出物と生成コードの一覧 → prelim_team05/PROVENANCE.md
──────────────────────────────────────────────────────────────────────────

PoolMatch のように A_bg の行を選ばず、B の行も再標本化しない。B_self から得る
順位相関と、真の希少型/多数派それぞれの低次集計量（潜在正規空間の平均・共分散）
だけを使い、全共変量を連続な確率モデルから新しく発生させる。

* 通常成分と希少成分を別々に正則化して、公開 Mahalanobis 検出数を再現する。
* B に近すぎる合成点は棄却し、連続値の完全一致も最終段で除く。
* 転帰は個票をコピーせず、B に fit した cause-specific Cox から再生成する。
* 転帰の多重集合を保つ交換探索で Cox の CI overlap だけを改善する。

これは DP を保証する実装ではない。個票を直接出さないモデルベース合成の比較候補である。

例:
    python third_defense.py --dist data_prelim_team05 --seed 6100 \
        --trials 8 --out prelim_team05/C_rarecopula.csv \
        --log prelim_team05/rarecopula.json --calibrate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

from harness import Bench
import defense as D
import fastcox as F


MODEL_COLS = list(D.CONT)
CLINICAL = list(D.CLIN)
REQUIRED_DIST_FILES = (
    "A_bg.csv", "B_self.csv", "schema.json", "scoring_config.yaml", "utility_ref.json",
)
_EPS = 1e-6


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _nearest_correlation(cov: np.ndarray) -> np.ndarray:
    """数値誤差を除き、正定値な共分散行列へ丸める。"""
    cov = np.asarray(cov, dtype=float)
    cov = np.nan_to_num(0.5 * (cov + cov.T), nan=0.0)
    val, vec = np.linalg.eigh(cov)
    return vec @ np.diag(np.clip(val, 1e-5, None)) @ vec.T


def _latent_reference(B: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """B の各周辺順位を標準正規へ移し、逆CDF用のソート値も返す。"""
    n = len(B)
    z = np.empty((n, len(MODEL_COLS)), dtype=float)
    sorted_values: dict[str, np.ndarray] = {}
    for j, col in enumerate(MODEL_COLS):
        x = B[col].to_numpy(float)
        ranks = pd.Series(x).rank(method="average").to_numpy(float)
        z[:, j] = norm.ppf(np.clip(ranks / (n + 1.0), _EPS, 1.0 - _EPS))
        sorted_values[col] = np.sort(x)
    return z, sorted_values


def _fit_components(z: np.ndarray, rare: np.ndarray, rare_shrink: float
                    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """潜在正規空間で通常/希少の平均と正則化共分散を推定する。"""
    rare = np.asarray(rare, dtype=bool)
    normal = ~rare
    mu_all = z.mean(axis=0)
    cov_all = LedoitWolf().fit(z).covariance_

    mu_normal = z[normal].mean(axis=0)
    cov_normal = LedoitWolf().fit(z[normal]).covariance_
    # 希少24人の標本共分散をそのまま使うと個人近傍を作りやすい。Ledoit-Wolf に加え、
    # 全体共分散へ明示的に縮約する。平均も少し全体側へ戻し、極端な外挿を避ける。
    mu_rare_raw = z[rare].mean(axis=0)
    cov_rare_raw = LedoitWolf().fit(z[rare]).covariance_
    s = float(np.clip(rare_shrink, 0.0, 1.0))
    mu_rare = (1.0 - 0.12 * s) * mu_rare_raw + (0.12 * s) * mu_all
    cov_rare = (1.0 - s) * cov_rare_raw + s * cov_all
    return ((mu_normal, _nearest_correlation(cov_normal)),
            (mu_rare, _nearest_correlation(cov_rare)))


def _inverse_marginals(z: np.ndarray, sorted_values: dict[str, np.ndarray]) -> pd.DataFrame:
    """潜在正規サンプルを B の平滑な経験逆CDFへ戻す。"""
    u = np.clip(norm.cdf(z), _EPS, 1.0 - _EPS)
    out = pd.DataFrame(index=np.arange(len(z)))
    for j, col in enumerate(MODEL_COLS):
        out[col] = np.quantile(sorted_values[col], u[:, j], method="linear")
    return out


def _maha(frame: pd.DataFrame, bench: Bench) -> np.ndarray:
    return D.maha(frame, bench.ref.mu, bench.ref.sigma)


def _distance_threshold(B: pd.DataFrame, A: pd.DataFrame, quantile: float) -> float:
    """A_bg→B距離分布から、偶然の近傍双子を棄却する閾値を決める。"""
    sd = B[MODEL_COLS].to_numpy(float).std(axis=0)
    sd = np.where(sd > 0.0, sd, 1.0)
    nn = NearestNeighbors(n_neighbors=1).fit(B[MODEL_COLS].to_numpy(float) / sd)
    dist = nn.kneighbors(A[MODEL_COLS].to_numpy(float) / sd)[0][:, 0]
    return float(np.quantile(dist, quantile))


def _distance_to_B(frame: pd.DataFrame, B: pd.DataFrame) -> np.ndarray:
    sd = B[MODEL_COLS].to_numpy(float).std(axis=0)
    sd = np.where(sd > 0.0, sd, 1.0)
    nn = NearestNeighbors(n_neighbors=1).fit(B[MODEL_COLS].to_numpy(float) / sd)
    return nn.kneighbors(frame[MODEL_COLS].to_numpy(float) / sd)[0][:, 0]


def _draw_conditioned(
    n: int,
    component: tuple[np.ndarray, np.ndarray],
    sorted_values: dict[str, np.ndarray],
    bench: Bench,
    want_rare: bool,
    min_distance: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """連続モデルから引き、公開外れ値条件と近傍安全条件で棄却サンプリングする。"""
    mu, cov = component
    accepted: list[pd.DataFrame] = []
    total = 0
    for _ in range(18):
        need = n - total
        if need <= 0:
            break
        batch = max(800, 18 * need if want_rare else 3 * need)
        z = rng.multivariate_normal(mu, cov, size=batch)
        draw = _inverse_marginals(z, sorted_values)
        det = _maha(draw, bench) >= float(bench.ref.band["lo"])
        far = _distance_to_B(draw, bench.B) >= min_distance
        take = (det == want_rare) & far
        if bool(take.any()):
            part = draw.loc[take].iloc[:need].copy()
            accepted.append(part)
            total += len(part)
    if total < n:
        raise RuntimeError(
            f"conditional copula sampling exhausted: wanted {n}, obtained {total}, rare={want_rare}"
        )
    return pd.concat(accepted, ignore_index=True).iloc[:n].reset_index(drop=True)


def _fixed_cell_labels(B: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """sex×smoking の4セル数を集計量として保存し、合成行へ独立に割り当てる。"""
    labels: list[tuple[int, int]] = []
    for sex in (0, 1):
        for smoking in (0, 1):
            count = int(((B["sex"] == sex) & (B["smoking"] == smoking)).sum())
            labels.extend([(sex, smoking)] * count)
    rng.shuffle(labels)
    arr = np.asarray(labels, dtype=int)
    return arr[:, 0], arr[:, 1]


def _exact_matches(values: np.ndarray, reference: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    """各合成値が B のどれかと数値一致するかを高速に返す。"""
    x = np.asarray(values, dtype=float)
    ref = np.sort(np.asarray(reference, dtype=float))
    pos = np.searchsorted(ref, x)
    hit = np.zeros(len(x), dtype=bool)
    for p in (pos - 1, pos):
        ok = (p >= 0) & (p < len(ref))
        q = np.clip(p, 0, len(ref) - 1)
        hit[ok] |= np.isclose(x[ok], ref[q[ok]], rtol=rtol, atol=1e-12)
    return hit


def _remove_exact_matches(C: pd.DataFrame, B: pd.DataFrame,
                          rng: np.random.Generator) -> pd.DataFrame:
    """完全一致照合の手掛かりになる6臨床値を、ごく小さい連続ジッタで除く。"""
    out = C.copy()
    for col in CLINICAL:
        lo, hi = D.RANGES[col]
        scale = max(float(B[col].std()), 1.0)
        for _ in range(8):
            hit = _exact_matches(out[col].to_numpy(float), B[col].to_numpy(float))
            if not bool(hit.any()):
                break
            delta = rng.choice([-1.0, 1.0], size=int(hit.sum())) * scale * 1e-7
            vals = out.loc[hit, col].to_numpy(float) + delta
            out.loc[hit, col] = np.clip(vals, lo, hi)
        if bool(_exact_matches(out[col].to_numpy(float), B[col].to_numpy(float)).any()):
            raise RuntimeError(f"could not remove exact matches for {col}")
    return out


def _matrix_sqrt(cov: np.ndarray, inverse: bool = False) -> np.ndarray:
    val, vec = np.linalg.eigh(_nearest_correlation(cov))
    power = 1.0 / np.sqrt(np.clip(val, 1e-7, None)) if inverse else np.sqrt(np.clip(val, 0.0, None))
    return vec @ np.diag(power) @ vec.T


def _moment_match_design(C: pd.DataFrame, B: pd.DataFrame, strength: float,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Cox設計の条件付き二次モーメントへ、連続6変数を部分的に合わせる。

    これは行選択や値の交換ではなく、合成標本全体への whitening/coloring 変換である。
    最後に単調な分位写像をかけ、周辺分布を直しつつ得られた順位相関を残す。
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return C
    out = C.copy()
    yc = np.column_stack([
        out["BMI"], out["SBP"], np.log(np.clip(out["TG"], 1e-6, None)),
        out["HDL"], np.log(np.clip(out["ALT"], 1e-6, None)), out["FPG"],
    ]).astype(float)
    yb = np.column_stack([
        B["BMI"], B["SBP"], np.log(np.clip(B["TG"], 1e-6, None)),
        B["HDL"], np.log(np.clip(B["ALT"], 1e-6, None)), B["FPG"],
    ]).astype(float)
    wc = out[["age", "sex", "smoking"]].to_numpy(float)
    wb = B[["age", "sex", "smoking"]].to_numpy(float)
    wc0, wb0 = wc - wc.mean(axis=0), wb - wb.mean(axis=0)
    yc0, yb0 = yc - yc.mean(axis=0), yb - yb.mean(axis=0)
    ac = np.linalg.pinv(wc0.T @ wc0) @ wc0.T @ yc0
    ab = np.linalg.pinv(wb0.T @ wb0) @ wb0.T @ yb0
    rc, rb = yc0 - wc0 @ ac, yb0 - wb0 @ ab
    cs, ct = np.cov(rc, rowvar=False), np.cov(rb, rowvar=False)
    recolored = rc @ _matrix_sqrt(cs, inverse=True) @ _matrix_sqrt(ct, inverse=False)
    y_target = yb.mean(axis=0) + wc0 @ ab + recolored
    y = (1.0 - s) * yc + s * y_target
    out["BMI"], out["SBP"], out["TG"] = y[:, 0], y[:, 1], np.exp(y[:, 2])
    out["HDL"], out["ALT"], out["FPG"] = y[:, 3], np.exp(y[:, 4]), y[:, 5]
    # 二次モーメント変換で動いた周辺だけを、粗いノット集計へ戻す。
    for col in CLINICAL:
        out[col] = D.quantile_align(
            out[col].to_numpy(float), B[col].to_numpy(float), 64, rng, jitter=0.18,
        )
        lo, hi = D.RANGES[col]
        out[col] = np.clip(out[col].to_numpy(float), lo, hi)
    return out


def _refine_outcomes(C: pd.DataFrame, bench: Bench, seed: int,
                     n_steps: int) -> pd.DataFrame:
    if n_steps <= 0:
        return C
    cb = D.fit_cox_raw(bench.B, "onset")
    swap_ok = ~bench._rare_mask(C)
    t, o, d, _ = F.refine_assignment(
        D._design(C), C["time"].to_numpy(float), C["onset"].to_numpy(int),
        C["death"].to_numpy(int), cb.params_.to_numpy(float),
        cb.standard_errors_.to_numpy(float), np.random.default_rng(seed), swap_ok,
        n_steps=n_steps, n_cand=12,
    )
    out = C.copy()
    out["time"], out["onset"], out["death"] = t, o, d
    return out


def build_rare_copula(
    bench: Bench,
    *,
    seed: int,
    rare_target: int | None = None,
    rare_shrink: float = 0.45,
    near_q: float = 0.05,
    moment_strength: float = 0.0,
    refine_steps: int = 300,
) -> pd.DataFrame:
    """希少混合 Gaussian copula から提出Cを1つ作る。"""
    rng = np.random.default_rng(seed)
    B = bench.B
    n = len(B)
    target = int(bench._rare_mask(B).sum()) if rare_target is None else int(rare_target)
    if not 1 <= target < n:
        raise ValueError("rare_target must lie in [1, n-1]")

    z, inverse = _latent_reference(B)
    normal_component, rare_component = _fit_components(z, bench.is_rare, rare_shrink)
    min_distance = _distance_threshold(B, bench.A_bg, near_q)
    rare_cov = _draw_conditioned(
        target, rare_component, inverse, bench, True, min_distance, rng,
    )
    normal_cov = _draw_conditioned(
        n - target, normal_component, inverse, bench, False, min_distance, rng,
    )
    C = pd.concat([rare_cov, normal_cov], ignore_index=True)
    C = C.iloc[rng.permutation(n)].reset_index(drop=True)

    # age は低 cardinality の公開ヒストグラムを保存。個票との対応はランダムである。
    order = np.argsort(C["age"].to_numpy(float), kind="mergesort")
    age = np.empty(n, dtype=int)
    age[order] = np.sort(B["age"].to_numpy(int))
    C["age"] = np.clip(age, *D.AGE_RANGE).astype(int)
    C["sex"], C["smoking"] = _fixed_cell_labels(B, rng)
    C["prefecture"] = rng.permutation(B["prefecture"].to_numpy(object))
    C = _moment_match_design(C, B, moment_strength, rng)
    for col, (lo, hi) in D.RANGES.items():
        C[col] = np.clip(C[col].to_numpy(float), lo, hi)
    C = _remove_exact_matches(C, B, rng)

    C = D.make_outcomes(C, B, rng, n_iter=80, damp=0.6)
    C = _refine_outcomes(C, bench, seed + 100_000, refine_steps)
    C = _remove_exact_matches(C, B, rng)
    return C[D.SUBMIT_COLS].reset_index(drop=True)


def _candidate_diagnostics(C: pd.DataFrame, bench: Bench) -> dict[str, Any]:
    score = bench.score(C)
    exact = {
        col: int(_exact_matches(C[col].to_numpy(float), bench.B[col].to_numpy(float)).sum())
        for col in CLINICAL
    }
    dist = _distance_to_B(C, bench.B)
    return {
        "U": float(score["U"]),
        "facets": {k: float(v) for k, v in score["facets"].items()},
        "rare_detected": int(bench._rare_mask(C).sum()),
        "exact_matches": exact,
        "nearest_B": {
            "min": float(np.min(dist)),
            "q01": float(np.quantile(dist, 0.01)),
            "median": float(np.median(dist)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="RareCopula model-based defense candidate")
    ap.add_argument("--dist", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=6100)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--rare-offset", type=int, default=4)
    ap.add_argument("--rare-shrink", type=float, default=0.45)
    ap.add_argument("--near-q", type=float, default=0.05)
    ap.add_argument("--moment-strength", type=float, default=0.5)
    ap.add_argument("--refine-steps", type=int, default=300)
    ap.add_argument("--calibrate", action="store_true",
                    help="探索後の最良候補に安全なCox affine/power補正を試す")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.trials < 1:
        ap.error("--trials must be >= 1")
    if args.rare_offset < -20 or args.rare_offset > 50:
        ap.error("--rare-offset must be between -20 and 50")
    if not 0.0 <= args.rare_shrink <= 1.0:
        ap.error("--rare-shrink must be in [0,1]")
    if not 0.0 <= args.near_q <= 0.5:
        ap.error("--near-q must be in [0,0.5]")
    if not 0.0 <= args.moment_strength <= 1.0:
        ap.error("--moment-strength must be in [0,1]")
    if args.refine_steps < 0:
        ap.error("--refine-steps must be >= 0")

    dist = args.dist.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    log_path = (args.log.expanduser().resolve() if args.log is not None
                else out_path.with_suffix(".json"))
    if out_path == log_path:
        ap.error("--out and --log must differ")
    missing = [name for name in REQUIRED_DIST_FILES if not (dist / name).is_file()]
    if missing:
        ap.error("distribution is missing: " + ", ".join(missing))
    if not args.force:
        existing = [str(p) for p in (out_path, log_path) if p.exists()]
        if existing:
            ap.error("refusing to overwrite existing files: " + ", ".join(existing))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    bench = Bench(dist)
    target = int(bench._rare_mask(bench.B).sum()) + int(args.rare_offset)
    records: list[dict[str, Any]] = []
    best: tuple[float, pd.DataFrame | None, dict[str, Any] | None] = (-1.0, None, None)
    for k in range(args.trials):
        seed = int(args.seed + k)
        C = build_rare_copula(
            bench, seed=seed, rare_target=target, rare_shrink=float(args.rare_shrink),
            near_q=float(args.near_q), moment_strength=float(args.moment_strength),
            refine_steps=int(args.refine_steps),
        )
        rec = {"seed": seed, "rare_target": target,
               "rare_shrink": float(args.rare_shrink), "near_q": float(args.near_q),
               "moment_strength": float(args.moment_strength),
               "refine_steps": int(args.refine_steps), **_candidate_diagnostics(C, bench)}
        records.append(rec)
        if rec["U"] > best[0]:
            best = (float(rec["U"]), C.copy(), rec)
        f = rec["facets"]
        print(f"[{k + 1}/{args.trials}] seed={seed} U={rec['U']:.6f} "
              f"gen={f['U_gen']:.4f} spec={f['U_spec']:.4f} "
              f"rare={f['U_rare']:.4f} valid={f['U_valid']:.4f}", flush=True)

    assert best[1] is not None and best[2] is not None
    calibration_report = None
    if args.calibrate:
        from calibration import calibrate_cox_columns
        calibrated, calibration_report = calibrate_cox_columns(best[1], bench)
        calibrated_diag = _candidate_diagnostics(calibrated, bench)
        if calibrated_diag["U"] > best[0]:
            best = (float(calibrated_diag["U"]), calibrated,
                    {**best[2], **calibrated_diag, "calibrated": True,
                     "calibration_lambda": float(calibration_report["selected_lambda"])})
        else:
            best[2]["calibrated"] = False
    best[1].to_csv(out_path, index=False)
    manifest = {
        "method": "RareCopula",
        "principle": "regularized two-component Gaussian copula; no B/A_bg row resampling",
        "distribution_sha256": {name: _sha256(dist / name) for name in REQUIRED_DIST_FILES},
        "driver_sha256": _sha256(Path(__file__).resolve()),
        "candidate_sha256": _sha256(out_path),
        "best": best[2],
        "trials": records,
        "calibration": calibration_report,
    }
    log_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"best U={best[0]:.6f} -> {out_path}")


if __name__ == "__main__":
    main()
