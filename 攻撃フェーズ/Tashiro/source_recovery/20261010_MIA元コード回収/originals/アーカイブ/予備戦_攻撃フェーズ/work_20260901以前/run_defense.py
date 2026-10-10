"""加工提出 C.csv を作る本番ドライバ。

──────────────────────────────────────────────────────────────────────────
【提出物との対応】 このスクリプトが作るのは → prelim_team05/C_raw_seed5312.csv
    （Pool 候補の“素”の出力。単体では提出しない）

  Pool 候補の生成チェーン:
      run_defense.py  →  C_raw_seed5312.csv   (sha256 df775af7…)
      calibration.py  →  C_candidate.csv      (sha256 1e1a384e…)
      package_defense.py → prelim_team05/T05_Pool.zip

  実際に使ったコマンド:
      python run_defense.py --dist data_prelim_team05 --trials 1 --seed0 5312 \
        --frac-true-like 0.9 --det-offset 4 --n-knots 128 --polish-steps 4000 \
        --refine-steps 150 \
        --out prelim_team05/C_raw_seed5312.csv \
        --log prelim_team05/search_seed5312.json
      （--det-offset 4 は「B の外れ値検出数 67 + 4 = 71」を希少目標にする指定）

  中核の実装は defense.py（build_C ＝ PoolMatch）と fastcox.py（転帰割当の改善）。
  提出物と生成コードの一覧 → prelim_team05/PROVENANCE.md
──────────────────────────────────────────────────────────────────────────

ローカル自己採点＝サーバ採点（ルールブック §5.4）で、加工フェーズの提出回数は無制限。
したがって「作って → 自己採点して → 一番良いものを出す」がそのまま正攻法になる。
乱数シードと少数のパラメータを振って best-of-N を取り、U_spec が律速のときだけ
転帰割り当ての山登り（多重集合不変スワップ）で追い込む。

    python run_defense.py --dist data_prelim_team05 --trials 40 \
        --out prelim_candidates/C.csv --log prelim_candidates/search_log.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from harness import Bench  # noqa: E402
import defense as D  # noqa: E402
import fastcox as F  # noqa: E402


REQUIRED_DIST_FILES = (
    "A_bg.csv",
    "B_self.csv",
    "schema.json",
    "scoring_config.yaml",
    "utility_ref.json",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_manifest(dist: Path) -> dict:
    missing = [name for name in REQUIRED_DIST_FILES if not (dist / name).is_file()]
    if missing:
        raise FileNotFoundError(f"配布データが不足しています: {', '.join(missing)}")
    return {
        "dist": str(dist),
        "sha256": {name: _sha256(dist / name) for name in REQUIRED_DIST_FILES},
    }


def refine(C: pd.DataFrame, B: pd.DataFrame, bench: Bench, n_steps: int, seed: int) -> pd.DataFrame:
    """U_spec（Cox の CI 重なり）を転帰の入れ替えだけで追い込む。"""
    cb = D.fit_cox_raw(B, "onset")
    X = D._design(C)
    swap_ok = ~bench._rare_mask(C)          # 希少層の発症率は動かさない
    t, o, d, io = F.refine_assignment(
        X, C["time"].to_numpy(float), C["onset"].to_numpy(int), C["death"].to_numpy(int),
        cb.params_.to_numpy(float), cb.standard_errors_.to_numpy(float),
        np.random.default_rng(seed), swap_ok, n_steps=n_steps, n_cand=10)
    out = C.copy()
    out["time"], out["onset"], out["death"] = t, o, d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, required=True,
                    help="B_self.csv/A_bg.csv等をまとめた配布データディレクトリ（必須）")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--out", default="C.csv")
    ap.add_argument("--log", default="search_log.json")
    ap.add_argument("--refine-steps", type=int, default=150)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--frac-true-like", type=float, default=None,
                    help="固定値で再現実行する場合に指定（未指定なら探索）")
    ap.add_argument("--det-offset", type=int, default=None,
                    help="Bの希少検出数に足す固定値（未指定なら探索）")
    ap.add_argument("--n-knots", type=int, default=None,
                    help="固定値で再現実行する場合に指定（未指定なら探索）")
    ap.add_argument("--polish-steps", type=int, default=4000)
    ap.add_argument("--force", action="store_true",
                    help="既存の出力・ログを上書きする（通常は候補ごとに別名を推奨）")
    args = ap.parse_args()

    if args.trials < 1:
        ap.error("--trials は1以上で指定してください")
    if args.refine_steps < 0:
        ap.error("--refine-steps は0以上で指定してください")
    if args.frac_true_like is not None and not 0.0 <= args.frac_true_like <= 1.0:
        ap.error("--frac-true-like は0以上1以下で指定してください")
    if args.det_offset is not None and args.det_offset < 0:
        ap.error("--det-offset は0以上で指定してください")
    if args.n_knots is not None and args.n_knots < 2:
        ap.error("--n-knots は2以上で指定してください")
    if args.polish_steps < 0:
        ap.error("--polish-steps は0以上で指定してください")

    dist = args.dist.expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    if out_path == log_path:
        ap.error("--out と --log には別のパスを指定してください")
    if not args.force:
        existing = [str(p) for p in (out_path, log_path) if p.exists()]
        if existing:
            ap.error("既存ファイルを上書きしません。別名を指定するか --force を使用してください: "
                     + ", ".join(existing))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = _dataset_manifest(dist)
    manifest["driver_sha256"] = _sha256(Path(__file__).resolve())
    manifest["arguments"] = {
        "trials": int(args.trials),
        "seed0": int(args.seed0),
        "refine_steps": int(args.refine_steps),
        "frac_true_like": args.frac_true_like,
        "det_offset": args.det_offset,
        "n_knots": args.n_knots,
        "polish_steps": int(args.polish_steps),
    }
    print(f"配布データ: {dist}")
    print(f"出力候補: {out_path}")
    bench = Bench(dist)
    B, A = bench.B, bench.A_bg
    rng = np.random.default_rng(12345)
    best = (-1.0, None, None)
    log = []
    t0 = time.time()
    for k in range(args.trials):
        seed = args.seed0 + k
        # det_target は B_i の検出数からの相対オフセットで振る（本番の B_i でもそのまま動く）
        n_det_B = int(bench._rare_mask(B).sum())
        p = dict(
            frac_true_like=(float(args.frac_true_like) if args.frac_true_like is not None
                            else float(rng.choice([0.7, 0.8, 0.9]))),
            det_target=n_det_B + (int(args.det_offset) if args.det_offset is not None
                                  else int(rng.choice([0, 2, 4, 8]))),
            n_knots=(int(args.n_knots) if args.n_knots is not None
                     else int(rng.choice([96, 128, 160]))),
            polish_steps=int(args.polish_steps),
        )
        C = D.build_C(B, A, bench.is_rare, bench.ref.mu, bench.ref.sigma,
                      float(bench.ref.band["lo"]), seed=seed, **p)
        out = bench.score(C)
        f = out["facets"]
        refined = False
        if args.refine_steps > 0 and f["U_spec"] == min(f.values()) and f["U_spec"] > 0.85:
            C2 = refine(C, B, bench, args.refine_steps, seed)
            o2 = bench.score(C2)
            if o2["U"] > out["U"]:
                C, out, f = C2, o2, o2["facets"]
                refined = True
        rec = dict(seed=seed, U=round(out["U"], 5),
                   **{k2: round(v, 5) for k2, v in f.items()},
                   refined=refined, **p)
        log.append(rec)
        if out["U"] > best[0]:
            best = (out["U"], C.copy(), rec)
            C.to_csv(out_path, index=False)
        print(f"[{k+1}/{args.trials}] U={out['U']:.4f} "
              f"gen={f['U_gen']:.4f} spec={f['U_spec']:.4f} rare={f['U_rare']:.4f} "
              f"valid={f['U_valid']:.4f}  best={best[0]:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    manifest["candidate_sha256"] = _sha256(out_path)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump({"dataset": manifest, "best": best[2], "log": log},
                  f, ensure_ascii=False, indent=1)
    print(f"\nbest U = {best[0]:.4f} -> {out_path}\n{best[2]}")


if __name__ == "__main__":
    main()
