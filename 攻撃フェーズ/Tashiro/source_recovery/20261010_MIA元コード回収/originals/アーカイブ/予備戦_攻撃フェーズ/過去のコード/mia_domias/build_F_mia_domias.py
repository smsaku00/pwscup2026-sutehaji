# -*- coding: utf-8 -*-
"""F_mia.csv を「公式参照実装と同じ Gaussian KDE の DOMIAS 密度比」だけで組み立てる。

    score(x) = p_syn(x) / p_ref(x)      p_syn = C_k 上の KDE, p_ref = A_bg 上の KDE
    conf(x)  = score / (1 + score)      = expit(log p_syn − log p_ref)  ∈ (0,1)

ただし **aia_challenge_<自分>.csv と QI で一致する自コホートの行は確信度 0 に固定する**。
challenge の会員側 163 行は非共有ブロック P_self の住人なので、どの標的 k の O_ik にも
入っていないことが構造から確定している（ルールブック §4.2）。採点は TPR@low-FPR なので、
確定した非メンバーを最下位に落とすぶんだけ FPR 予算が本当に不確実な行に回る。

使い方:
    python build_F_mia_domias.py --out out_r1
    python build_F_mia_domias.py --out out_r1 --bw 0.5     # 帯域を狭める（記録レベル寄り）
    python build_F_mia_domias.py --out out_r1 --no-zero    # 0 固定を切って素の DOMIAS を見る

出力:
    <out>/F_mia.csv      record_id + 標的列（自列は除外）、値域 [0,1]
    <out>/domias.json    列ごとの診断（Δ=log比 の分位点、0 固定で救った FPR 予算 など）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import gaussian_kde

ROOT = Path("/Users/taisei1105/pws2026-local")
DIST_DEFAULT = ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack"
ABG_DEFAULT = ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"
BSELF_DEFAULT = ROOT / "予備戦/配布データ/PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/B_self.csv"

# 公式 schema.CONTINUOUS。DOMIAS はここだけを見る（カテゴリは B で確定しているので情報が無い）
CONTINUOUS = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
# aia_challenge が持っている列。P_self の同定はこの 7 列の完全一致で行う
QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
LOW_FPR = 0.01  # scoring_config.yaml の scoring.low_fpr。診断表示にだけ使う


# ---------------------------------------------------------------- 密度
def zscaler(abg: pd.DataFrame, cols: list[str]):
    """A_bg の平均・標準偏差（ddof=0）で標準化する関数を返す。公式参照実装と同一の空間。"""
    mu = abg[cols].mean().to_numpy(float)
    sd = abg[cols].std(ddof=0).to_numpy(float)
    sd = np.where(sd > 0, sd, 1.0)
    return lambda df: (df[cols].to_numpy(float) - mu) / sd


def fit_kde(X: np.ndarray, bw: float) -> gaussian_kde | None:
    """Scott 則 × bw で KDE を張る。共分散が特異なら None（判別不能）。

    bw=1.0 が公式参照実装（scipy 既定の Scott）。C_k が低次元部分空間に潰れている
    （重い加法ノイズ・過激なマイクロ集約）と LinAlgError が出るので None で示す。
    """
    n, d = X.shape
    scott = n ** (-1.0 / (d + 4))
    try:
        return gaussian_kde(X.T, bw_method=scott * bw)
    except np.linalg.LinAlgError:
        return None


def logpdf(kde: gaussian_kde, Y: np.ndarray, chunk: int = 2000) -> np.ndarray:
    """logpdf をチャンクに割って評価する（5,143×1,049 の距離行列を一度に持たないため）。"""
    out = np.empty(len(Y))
    for s in range(0, len(Y), chunk):
        out[s:s + chunk] = kde.logpdf(Y[s:s + chunk].T)
    return out


# ---------------------------------------------------------------- 非共有ブロックの同定
def identify_nonshared(B: pd.DataFrame, challenge_self: pd.DataFrame) -> np.ndarray:
    """aia_challenge_<自分> と QI 完全一致する B の行 = 非共有ブロック P_self。

    challenge 326 行の内訳は「実在の会員 163 + (sex,age) を揃えた合成おとり 163」。
    おとりは実在しないので B とは一致せず、一致した行がそのまま P_self になる。
    浮動小数は round(9) で丸めてから突き合わせる（CSV の往復で最下位ビットが動くため）。
    """
    def key(df: pd.DataFrame) -> list[tuple]:
        num = np.round(df[["age", "SBP", "TG", "HDL", "ALT"]].to_numpy(float), 9)
        return [tuple(r) + (int(s), str(p)) for r, s, p in zip(num, df["sex"], df["prefecture"])]

    kb = {k: i for i, k in enumerate(key(B))}
    mask = np.zeros(len(B), bool)
    for k in key(challenge_self):
        if k in kb:
            mask[kb[k]] = True
    return mask


# ---------------------------------------------------------------- 本体
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    ap.add_argument("--self", dest="self_k", type=int, default=5, help="自分のコホート番号")
    ap.add_argument("--dist", default=str(DIST_DEFAULT), help="C_k / aia_challenge_k / mia_columns.json の場所")
    ap.add_argument("--abg", default=str(ABG_DEFAULT))
    ap.add_argument("--bself", default=str(BSELF_DEFAULT))
    ap.add_argument("--bw", type=float, default=1.0, help="Scott 帯域の倍率（1.0 = 公式）")
    ap.add_argument("--no-zero", action="store_true", help="P_self の 0 固定をしない（素の DOMIAS）")
    ap.add_argument("--prior-weight", type=float, default=1.0,
                    help="県の事前オッズを log で足すときの重み（0 で無効。1.0 = 額面どおり）")
    ap.add_argument("--n-nonshared", type=int, default=313,
                    help="非共有ブロック P_i の人数（ルールブック §4.1）。事前オッズの導出にだけ使う")
    a = ap.parse_args()

    dist = Path(a.dist)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    A = pd.read_csv(a.abg)
    Bs = pd.read_csv(a.bself)
    rid = Bs["record_id"].to_numpy(int)
    targets = sorted(
        int(k) for k in json.loads((dist / "mia_columns.json").read_text())["target_columns"]
        if int(k) != a.self_k
    )
    print(f"候補 {len(Bs)} 行 / 標的 {len(targets)} 列 / 参照 A_bg {len(A)} 行 / 帯域 ×{a.bw:g}")

    # ---- 非共有ブロック P_self ----
    ch = pd.read_csv(dist / f"aia_challenge_{a.self_k}.csv")
    p_self = identify_nonshared(Bs, ch)
    print(f"P_self（challenge と QI 完全一致）= {p_self.sum()} 行"
          f" / challenge {len(ch)} 行中の会員想定 {len(ch)//2} 行")
    if p_self.sum() != len(ch) // 2:
        print("  ★ 想定 163 と違う。QI の丸めか自コホート番号を疑うこと")
    if p_self.any():
        pref = Bs["prefecture"][p_self].value_counts()
        print(f"  県構成: {dict(pref)}（全員が地元県なら想定どおり）")

    # ---- 県の事前オッズ ----
    # 非共有ブロックは全員が地元県なので、地元県のうち共有されているのは
    # (地元県の人数 − 313) 人だけ。challenge で特定済みの 163 人を除いた残りが、
    # 「まだ非共有かもしれない」塊になる。非地元県は共有率 100%。
    log_prior = np.zeros(len(Bs))
    home = np.zeros(len(Bs), bool)
    rate = float("nan")
    if a.prior_weight != 0.0 and p_self.any():
        home_pref = Bs["prefecture"][p_self].mode().iloc[0]      # P_self の県 = 地元県
        home = (Bs["prefecture"] == home_pref).to_numpy()
        n_shared_home = int(home.sum()) - a.n_nonshared
        n_rest = int((home & ~p_self).sum())
        if n_shared_home > 0 and n_rest > 0:
            rate = n_shared_home / n_rest
            log_prior = np.where(home & ~p_self, a.prior_weight * np.log(rate), 0.0)
            print(f"県の事前オッズ: {home_pref} の残り {n_rest} 行の共有率 {rate:.3f}"
                  f"（非{home_pref}は 100%）→ log で {a.prior_weight*np.log(rate):+.3f} を加算")
        else:
            print(f"  ★ 事前オッズを算出できない（地元県 {int(home.sum())} 行 / P_i {a.n_nonshared}）ので無効化")

    # ---- 密度 ----
    z = zscaler(A, CONTINUOUS)
    ref = z(A)
    tgt = z(Bs)
    kde_ref = fit_kde(ref, a.bw)
    if kde_ref is None:
        raise SystemExit("A_bg の KDE が張れない（特異共分散）")
    l_ref = logpdf(kde_ref, tgt)

    # 希少軸の診断用（採点は max(tpr_all, tpr_rare) なので、希少 24 人の中の 1 位が誰かは効く）
    rare = Bs["is_rare"].to_numpy(bool) if "is_rare" in Bs.columns else np.zeros(len(Bs), bool)

    F = pd.DataFrame({"record_id": rid})
    diag: dict[str, dict] = {}
    for k in targets:
        C = pd.read_csv(dist / f"C_{k}.csv")
        kde_syn = fit_kde(z(C), a.bw)
        if kde_syn is None:
            conf = np.full(len(Bs), 0.5)          # 公式と同じ退避: 判別不能は chance
            delta = np.zeros(len(Bs))
            singular = True
        else:
            delta = logpdf(kde_syn, tgt) - l_ref + log_prior   # log(p_syn/p_ref) + log 事前オッズ
            conf = expit(delta)                   # = ratio/(1+ratio) を数値安定に
            singular = False

        # 0 固定の前に「どれだけ FPR 予算を食っていたか」を測る（この列だけの話ではなく全列共通）
        budget = max(1, int(round(LOW_FPR * (len(Bs) - 1))))   # 陽性 ≈ 25 なので陰性 ≈ 1024
        topb = np.argsort(-conf, kind="mergesort")[:budget]
        wasted = int(p_self[topb].sum())

        if not a.no_zero:
            conf = np.where(p_self, 0.0, conf)

        F[str(k)] = conf
        # 希少帯の 1 位（＝この列の実質的な「指名」）。値は変えずに記録だけ取る
        nominee, margin = None, None
        if rare.any():
            rr = conf[rare]
            srt = np.sort(rr)
            nominee = int(rid[rare][int(rr.argmax())])
            margin = float(srt[-1] - srt[-2]) if len(srt) > 1 else float("nan")
        q = np.quantile(delta[~p_self], [0.0, 0.5, 0.99, 1.0])
        diag[str(k)] = {
            "singular_kde": singular,
            "delta_min": float(q[0]), "delta_med": float(q[1]),
            "delta_p99": float(q[2]), "delta_max": float(q[3]),
            "conf_max": float(conf.max()), "conf_min": float(conf.min()),
            "n_unique": int(np.unique(conf).size),
            "top_budget": budget, "p_self_in_top_budget": wasted,
            "rare_nominee": nominee, "rare_margin": margin,
        }
        print(f"  C_{k:<2}: Δ 中央 {q[1]:+.3f} / 99% {q[2]:+.3f} / 最大 {q[3]:+.3f}"
              f" | 上位 {budget} 件のうち P_self だった {wasted} 件を排除"
              + ("  ★KDE 特異" if singular else ""))

    # ---- 書き出しと形式の自己点検 ----
    F.to_csv(out / "F_mia.csv", index=False)
    v = F[[str(k) for k in targets]].to_numpy()
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), "値域違反"
    n_all_zero = int((v == 0).all(1).sum())
    saved = sum(d["p_self_in_top_budget"] for d in diag.values())
    diag["_summary"] = {
        "self_k": a.self_k, "targets": targets, "bw": a.bw, "zeroed": not a.no_zero,
        "prior_weight": a.prior_weight, "home_share_rate": rate,
        "n_rows": len(F), "n_p_self": int(p_self.sum()), "n_all_zero_rows": n_all_zero,
        "p_self_in_top_budget_total": saved,
        "record_id_p_self": rid[p_self].tolist(),
    }
    (out / "domias.json").write_text(json.dumps(diag, ensure_ascii=False, indent=1))
    print(f"\n{out/'F_mia.csv'}  ({len(F)} 行 × {len(targets)} 列, 値域 {v.min():.3g}〜{v.max():.3g})")
    print(f"全列 0 の行 = {n_all_zero}（= P_self）")
    print(f"0 固定で {len(targets)} 列合計 {saved} 件の確定偽陽性を FPR 予算から追い出した"
          f"（予算は 1 列あたり {max(1, int(round(LOW_FPR*(len(Bs)-1))))} 件）")
    if rare.any():
        print(f"希少候補 {int(rare.sum())} 人のうち {int((rare & p_self).sum())} 人が P_self → "
              f"希少軸の実質候補 {int((rare & ~p_self).sum())} 人")
        print("列ごとの希少 1 位（指名）: "
              + ", ".join(f"{k}→{diag[str(k)]['rare_nominee']}" for k in targets))


if __name__ == "__main__":
    main()
