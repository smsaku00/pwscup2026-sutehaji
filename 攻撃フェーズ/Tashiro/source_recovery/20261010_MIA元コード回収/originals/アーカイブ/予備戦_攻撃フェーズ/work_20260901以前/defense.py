"""加工（防御）パイプライン: PoolMatch。

考え方
------
C の素材を **B ではなく A_bg（背景データ）から取る**。A_bg は
  * 母集団 A から取った実レコードで、連続共変量の分布は B とほぼ同じ（列別 KS 0.02〜0.03）、
  * どのチームのコホートにも入っていない ＝ MIA の候補にも AIA の標的/対照にもならない、
  * 全チームに同じものが配られている公開データ、
なので「A_bg の行をそのまま出す」ことは原理的にメンバーシップも属性も漏らさない。
したがって C = f(A_bg) の形にしておけば匿名性はほぼ上限、あとは
**B の標本統計に合わせ込む作業（＝有用性）だけ**の問題になる。

合わせ込むのは採点器が見る量そのもの:
  U_gen   : 列別分位（KS）・pMSE・KM 曲線
  U_spec  : Cox 9 係数の点推定と CI 幅（有意ステータスまで含めて）
  U_rare  : 外れ値検出数・希少層のプロファイル/発症率/同時分布
  U_valid : 打ち切り整合・関係の非捏造

漏らさないための規律（重要）
  * B の**個票の値をそのまま出さない**。列の分位合わせは K 個のノット（＝集計量）だけを使い、
    ノット間は線形補間 ＋ 微小ジッタで「B に存在しない値」にする。
    （B の値をそのまま並べ替えて出すスワップ系は、後述の完全一致照合攻撃で即死する）
  * time/onset/death は条件付きに**作り直す**（実在個人の追跡年数を写さない ＝ AIA 対策）。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
CLIN = ["BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
COX_COLS = ["age", "sex", "BMI", "SBP", "log_TG", "HDL", "log_ALT", "FPG", "smoking"]
RANGES = {"BMI": (10.0, 55.0), "SBP": (40.0, 250.0), "TG": (1.0, 3000.0),
          "HDL": (10.0, 160.0), "ALT": (0.1, 700.0), "FPG": (40.0, 400.0)}
AGE_RANGE = (40, 74)
HORIZON = 10.0
SUBMIT_COLS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
               "smoking", "FPG", "time", "onset", "death"]


# --------------------------------------------------------------------------- #
# 素材プールの準備
# --------------------------------------------------------------------------- #
def maha(df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = df[CONT].to_numpy(float) - mu
    inv = np.linalg.pinv(sigma)
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", x, inv, x), 0, None))


def rare_type_score(B: pd.DataFrame, is_rare: np.ndarray, A: pd.DataFrame,
                    model: str = "logit") -> np.ndarray:
    """B_self の真値 is_rare で学習した「非典型糖尿病らしさ」を A_bg に付与する。

    ★ model="rf" は**使ってはいけない**（実測で最近傍距離攻撃に破られる）。
    ランダムフォレストは 24 人しかいない陽性例の近傍を記憶するので、これで A_bg を
    選ぶと「B の希少患者のそっくりさん」を選び出すことになり、C の希少層が
    B の希少個人の周りに張り付く。ホールドアウト監査で nn_gap の β_raw が
    0.08（安全）→ 0.65（chance 0.38 を大きく超過・a_MIA=1.0）に跳ねる。
    既定の "logit"（強い L2・線形）は個人を指させないので、型の平均的な特徴だけを写す。
    """
    Xb = B[CONT].to_numpy(float)
    mu, sd = Xb.mean(0), np.where(Xb.std(0) > 0, Xb.std(0), 1.0)
    if model == "rf":
        clf = RandomForestClassifier(500, random_state=0, n_jobs=1, class_weight="balanced")
        clf.fit(Xb, is_rare.astype(int))
        return clf.predict_proba(A[CONT].to_numpy(float))[:, 1]
    clf = LogisticRegression(penalty="l2", C=0.05, max_iter=2000, class_weight="balanced")
    clf.fit((Xb - mu) / sd, is_rare.astype(int))
    return clf.predict_proba((A[CONT].to_numpy(float) - mu) / sd)[:, 1]


def min_dist_to_B(A: pd.DataFrame, B: pd.DataFrame) -> np.ndarray:
    """A_bg 各行から B の最近傍までの距離（A_bg の SD で標準化した 7 次元空間）。"""
    sd = B[CONT].to_numpy(float).std(0)
    sd = np.where(sd > 0, sd, 1.0)
    xb = B[CONT].to_numpy(float) / sd
    xa = A[CONT].to_numpy(float) / sd
    return NearestNeighbors(n_neighbors=1).fit(xb).kneighbors(xa)[0][:, 0]


# --------------------------------------------------------------------------- #
# 行の選択
# --------------------------------------------------------------------------- #
def select_rows(A: pd.DataFrame, B: pd.DataFrame, det_A: np.ndarray, p_rare_A: np.ndarray,
                n_det_target: int, frac_true_like: float, rng: np.random.Generator,
                keep: np.ndarray | None = None) -> np.ndarray:
    """A_bg から n=|B| 行を選ぶ。

    * 外れ値（Mahalanobis 検出）を n_det_target 行、うち frac_true_like を「真の希少型らしい」
      行から取る（残りは多数派の外れ値＝B の検出集合と同じ組成にする）。
    * sex×smoking の 4 セル人数を B に一致させる（TVD=0）。
    """
    n = len(B)
    ok = np.ones(len(A), bool) if keep is None else keep
    idx_det = np.flatnonzero(det_A & ok)
    idx_non = np.flatnonzero((~det_A) & ok)

    # 希少型らしさで検出集合を2分割
    thr = np.quantile(p_rare_A[idx_det], 0.5)
    tl = idx_det[p_rare_A[idx_det] >= max(thr, 0.15)]
    ml = idx_det[p_rare_A[idx_det] < max(thr, 0.15)]
    n_tl = int(round(n_det_target * frac_true_like))
    n_tl = min(n_tl, len(tl))
    n_ml = min(n_det_target - n_tl, len(ml))
    pick_det = np.concatenate([rng.choice(tl, n_tl, replace=False),
                               rng.choice(ml, n_ml, replace=False)])

    # 残りは sex×smoking のセル人数を B に合わせて非外れ値から
    need = {}
    for s in (0, 1):
        for sm in (0, 1):
            need[(s, sm)] = int(((B.sex == s) & (B.smoking == sm)).sum())
    for i in pick_det:
        k = (int(A.sex.iloc[i]), int(A.smoking.iloc[i]))
        need[k] = need.get(k, 0) - 1
    picks = [pick_det]
    for (s, sm), cnt in need.items():
        if cnt <= 0:
            continue
        pool = idx_non[(A.sex.to_numpy()[idx_non] == s) & (A.smoking.to_numpy()[idx_non] == sm)]
        picks.append(rng.choice(pool, min(cnt, len(pool)), replace=False))
    out = np.concatenate(picks)
    if len(out) < n:  # 端数は非外れ値から補充
        rest = np.setdiff1d(idx_non, out)
        out = np.concatenate([out, rng.choice(rest, n - len(out), replace=False)])
    return out[:n]


# --------------------------------------------------------------------------- #
# 周辺分布の合わせ込み（ノット分位マッピング）
# --------------------------------------------------------------------------- #
def quantile_align(x: np.ndarray, b: np.ndarray, n_knots: int, rng: np.random.Generator,
                   jitter: float = 0.25) -> np.ndarray:
    """x（A_bg 由来）を B の周辺分布へ単調写像する。

    K 個のノット分位だけを B から借りる（＝集計量のみ）。ノット間は線形補間なので、
    出力値は B の個票値と一致しない。さらに局所間隔 × jitter の微小ノイズを足して
    「完全一致照合」攻撃の手がかりを消す。KS はノット間の粒度 1/K 程度に収まる。
    """
    ps = np.linspace(0.0, 1.0, n_knots + 1)
    xa = np.quantile(x, ps, method="linear")
    ba = np.quantile(b, ps, method="linear")
    xa = np.maximum.accumulate(xa)
    ba = np.maximum.accumulate(ba)
    dup = np.diff(xa) <= 0
    if dup.any():  # 退化ノットを除去
        keep = np.concatenate([[True], ~dup])
        xa, ba = xa[keep], ba[keep]
    y = np.interp(x, xa, ba)
    if jitter > 0:
        step = np.interp(x, xa, np.gradient(ba, xa)) * (np.diff(xa).mean())
        y = y + rng.uniform(-0.5, 0.5, len(y)) * jitter * step
    return y


# --------------------------------------------------------------------------- #
# 共分散の磨き込み（Cox の CI 幅＝SE を B に揃えるため）
# --------------------------------------------------------------------------- #
def polish_covariance(sel: np.ndarray, Xpool: np.ndarray, target_cov: np.ndarray,
                      groups: np.ndarray, rng: np.random.Generator,
                      n_steps: int = 4000, n_cand: int = 60,
                      hz: np.ndarray | None = None, target_wcov: np.ndarray | None = None,
                      w_weighted: float = 3.0, freeze_rare: bool = True,
                      banned: np.ndarray | None = None) -> np.ndarray:
    """選んだ行集合を貪欲スワップして設計行列の共分散を B に合わせる。

    Cox の SE（＝CI 幅）は情報行列 ≈ イベント数 × Cov(X) でほぼ決まるので、
    Cov(X) を B に揃えると IO（信頼区間の重なり）が上がる。B は県で偏重サンプリング
    されているため母集団 A_bg とは相関構造が実際にずれており（実測 max 0.09）、
    ここを直さないと SE 比が 0.84〜1.17 になって IO が 0.93 前後で頭打ちになる。

    交換は `groups`（(sex, smoking, 外れ値クラス) 等）が同じ行同士に限定するので、
    周辺分布・希少組成・sex×smoking セル人数は保存される。
    """
    n = len(sel)
    sd = np.sqrt(np.diag(target_cov))
    W = 1.0 / np.outer(sd, sd)
    in_sel = np.zeros(len(Xpool), bool)
    in_sel[sel] = True
    if banned is not None:
        in_sel |= banned
    S = Xpool[sel].T @ Xpool[sel]
    m = Xpool[sel].sum(axis=0)

    use_w = hz is not None and target_wcov is not None
    if use_w:
        sdw = np.sqrt(np.diag(target_wcov))
        Ww = 1.0 / np.outer(sdw, sdw)
        hz = hz / hz.mean()
        Sw = (hz[sel, None] * Xpool[sel]).T @ Xpool[sel]
        mw = (hz[sel, None] * Xpool[sel]).sum(axis=0)
        w0 = hz[sel].sum()
    else:
        Sw = mw = None
        w0 = 0.0

    def loss(S, m, Sw=None, mw=None, w0=0.0):
        cov = S / n - np.outer(m, m) / n**2
        d = (cov - target_cov) * W
        val = float((d * d).sum())
        if use_w and Sw is not None:
            wcov = Sw / w0 - np.outer(mw, mw) / w0**2
            dw = (wcov - target_wcov) * Ww
            val += w_weighted * float((dw * dw).sum())
        return val

    cur = loss(S, m, Sw, mw, w0)
    by_group: dict[int, np.ndarray] = {}
    for g in np.unique(groups):
        by_group[int(g)] = np.flatnonzero(groups == g)
    sel = sel.copy()
    movable = np.flatnonzero(np.isin(groups[sel] % 3, [0])) if freeze_rare else np.arange(n)
    if len(movable) == 0:
        return sel
    for _ in range(n_steps):
        pos = int(movable[rng.integers(0, len(movable))])
        o = sel[pos]
        g = int(groups[o])
        pool = by_group[g]
        cand = pool[rng.integers(0, len(pool), n_cand)]
        cand = cand[~in_sel[cand]]
        if len(cand) == 0:
            continue
        xo = Xpool[o]
        So = S - np.outer(xo, xo)
        mo = m - xo
        if use_w:
            Swo = Sw - hz[o] * np.outer(xo, xo)
            mwo = mw - hz[o] * xo
            w0o = w0 - hz[o]
        best_j, best_l = None, cur
        for j in cand:
            xj = Xpool[j]
            if use_w:
                l = loss(So + np.outer(xj, xj), mo + xj,
                         Swo + hz[j] * np.outer(xj, xj), mwo + hz[j] * xj, w0o + hz[j])
            else:
                l = loss(So + np.outer(xj, xj), mo + xj)
            if l < best_l:
                best_j, best_l = j, l
        if best_j is None:
            continue
        xj = Xpool[best_j]
        S = So + np.outer(xj, xj)
        m = mo + xj
        if use_w:
            Sw = Swo + hz[best_j] * np.outer(xj, xj)
            mw = mwo + hz[best_j] * xj
            w0 = w0o + hz[best_j]
        cur = best_l
        in_sel[o] = False
        in_sel[best_j] = True
        sel[pos] = best_j
    return sel


def repair_rare(C: pd.DataFrame, Apool: pd.DataFrame, sel: np.ndarray, mu, sigma,
                band_lo: float, det_target: int, p_rare_A: np.ndarray,
                rng: np.random.Generator, banned: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """最終仕上げ後の外れ値検出数を目標へ戻す。

    U_rare は「C で外れ値と判定される行」だけを見るので、この人数と組成が採点の要。
    足りなければプールの高 p_rare 側から補充し、多すぎれば p_rare の低い検出行を
    非検出行と入れ替える。周辺分布はここでは動かさない（数行の差し替えなので KS 影響は 1/1049 級）。
    """
    m_pool = maha(Apool, mu, sigma)
    det_pool = m_pool >= band_lo
    in_sel = np.zeros(len(Apool), bool)
    in_sel[sel] = True
    if banned is not None:
        in_sel |= banned          # 禁止行は「使用済み」扱いで選ばれないようにする
    for _ in range(60):
        m_c = maha(C, mu, sigma)
        det_c = m_c >= band_lo
        cnt = int(det_c.sum())
        if abs(cnt - det_target) <= 2:
            break
        if cnt < det_target:                       # 足りない: 高 p_rare の外れ値を入れる
            cand = np.flatnonzero(det_pool & ~in_sel)
            if len(cand) == 0:
                break
            j = cand[np.argmax(p_rare_A[cand] + rng.normal(0, 0.02, len(cand)))]
            pos = int(np.flatnonzero(~det_c)[np.argmin(m_c[~det_c])])
        else:                                      # 多すぎ: p_rare の低い検出行を外す
            cand = np.flatnonzero(~det_pool & ~in_sel)
            if len(cand) == 0:
                break
            j = cand[rng.integers(0, len(cand))]
            di = np.flatnonzero(det_c)
            pos = int(di[np.argmin(m_c[di])])
        old = sel[pos]
        sel = sel.copy()
        sel[pos] = j
        in_sel[old] = False
        in_sel[j] = True
        C = C.copy()
        C.iloc[pos, [C.columns.get_loc(c) for c in CONT + ["sex", "smoking"]]] = \
            Apool.iloc[j][CONT + ["sex", "smoking"]].to_numpy()
    return C.reset_index(drop=True), sel


# --------------------------------------------------------------------------- #
# 転帰の生成（cause-specific Cox・B の推定値へ逆算合わせ）
# --------------------------------------------------------------------------- #
def _design(df: pd.DataFrame) -> np.ndarray:
    d = pd.DataFrame({
        "age": df.age.astype(float), "sex": df.sex.astype(float), "BMI": df.BMI.astype(float),
        "SBP": df.SBP.astype(float), "log_TG": np.log(np.clip(df.TG.astype(float), 1e-6, None)),
        "HDL": df.HDL.astype(float), "log_ALT": np.log(np.clip(df.ALT.astype(float), 1e-6, None)),
        "FPG": df.FPG.astype(float), "smoking": df.smoking.astype(float)})
    return d[COX_COLS].to_numpy(float)


def fit_cox_raw(df: pd.DataFrame, event_col: str = "onset", penalizer: float = 0.0):
    d = pd.DataFrame(_design(df), columns=COX_COLS)
    d["time"] = df["time"].to_numpy(float)
    d["_e"] = df[event_col].to_numpy(int)
    cph = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d, duration_col="time", event_col="_e")
    return cph


def baseline_hazard(df: pd.DataFrame, beta: np.ndarray, event_col: str,
                    center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Breslow の基準累積ハザード（β を固定して計算）。center を渡すと中心化設計で計算する。"""
    t = df["time"].to_numpy(float)
    e = df[event_col].to_numpy(int)
    X = _design(df)
    if center is not None:
        X = X - center
    ph = np.exp(X @ beta)
    order = np.argsort(t)
    t, e, ph = t[order], e[order], ph[order]
    rev = np.cumsum(ph[::-1])[::-1]          # at-risk 合計
    inc = np.where(e == 1, 1.0 / np.clip(rev, 1e-12, None), 0.0)
    H = np.cumsum(inc)
    m = e == 1
    return t[m], H[m]


def sample_times(X: np.ndarray, beta: np.ndarray, t_grid: np.ndarray, H_grid: np.ndarray,
                 u: np.ndarray, scale: float) -> np.ndarray:
    ph = np.exp(X @ beta) * scale
    target = -np.log(np.clip(u, 1e-12, 1.0)) / np.clip(ph, 1e-12, None)
    t = np.interp(target, H_grid, t_grid, left=t_grid[0], right=np.inf)
    return np.where(target > H_grid[-1], np.inf, t)


def _assemble(Ccov, to, td, censor_t):
    n = len(Ccov)
    onset = ((to <= HORIZON) & (to <= td)).astype(int)
    death = ((td <= HORIZON) & (td < to)).astype(int)
    time = np.full(n, censor_t)
    time[onset == 1] = np.clip(to[onset == 1], 1.0 / 365.25, HORIZON)
    time[death == 1] = np.clip(td[death == 1], 1.0 / 365.25, HORIZON)
    out = Ccov.copy()
    out["time"], out["onset"], out["death"] = time, onset, death
    return out


def _tune_scale(X, beta, t_g, H_g, u, target_n, other_t, lo=-25.0, hi=25.0) -> float:
    """イベント数が target_n になるよう log スケールを二分探索（Cox フィット不要で速い）。"""
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        t = sample_times(X, beta, t_g, H_g, u, np.exp(mid))
        cnt = int(((t <= HORIZON) & (t <= other_t)).sum())
        if cnt < target_n:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def make_outcomes(Ccov: pd.DataFrame, B: pd.DataFrame, rng: np.random.Generator,
                  n_iter: int = 60, damp: float = 0.6, verbose: bool = False) -> pd.DataFrame:
    """C の共変量に対して (time, onset, death) を作る。

    B にフィットした競合リスク Cox から条件付きにサンプルし、**C を採点用 Cox で
    再推定した係数が B の推定値に一致する**まで β_model を不動点反復で補正する。
    共通乱数（u を固定）で反復するので β_model → β̂_C の写像は決定的・ほぼ恒等勾配になり、
    β_model* = β_target − ε に収束する。イベント数は毎回二分探索で B に固定し、
    CI 幅（＝SE）も B に揃える。個票の追跡年数は一切写さない。
    """
    cph_b = fit_cox_raw(B, "onset")
    beta_target = cph_b.params_.to_numpy(float)
    se_target = cph_b.standard_errors_.to_numpy(float)
    n_onset_b = int(B.onset.sum())
    n_death_b = int(B.death.sum())

    center = _design(B).mean(axis=0)
    t_o, H_o = baseline_hazard(B, beta_target, "onset", center)
    beta_d = np.zeros(len(COX_COLS))                      # 死亡は共変量非依存（§3: 発症と独立）
    t_d, H_d = baseline_hazard(B, beta_d, "death", center)

    X_raw = _design(Ccov)
    X = X_raw - center                                    # 中心化＝スケールと β を分離
    n = len(Ccov)
    u_o, u_d = rng.uniform(size=n), rng.uniform(size=n)
    censor_t = float(B.loc[(B.onset == 0) & (B.death == 0), "time"].median())

    s_d = _tune_scale(X * 0.0, beta_d, t_d, H_d, u_d, n_death_b, np.full(n, np.inf))
    td = sample_times(X * 0.0, beta_d, t_d, H_d, u_d, s_d)

    beta_m = beta_target.copy()
    best = (-1.0, None)
    for it in range(n_iter):
        s_o = _tune_scale(X, beta_m, t_o, H_o, u_o, n_onset_b, td)
        to = sample_times(X, beta_m, t_o, H_o, u_o, s_o)
        out = _assemble(Ccov, to, td, censor_t)
        try:
            cph_c = fit_cox_raw(out, "onset")
        except Exception:
            beta_m = 0.5 * (beta_m + beta_target)
            continue
        bc = cph_c.params_.to_numpy(float)
        sc = cph_c.standard_errors_.to_numpy(float)
        io = _io_estimate(beta_target, se_target, bc, sc)
        if io > best[0]:
            best = (io, out.copy())
        if verbose and (it % 5 == 0 or io > 0.9):
            print(f"   it{it:02d} io~{io:.4f} maxerr={np.abs((bc-beta_target)/se_target).max():.2f}SE "
                  f"seratio={np.abs(sc/se_target-1).max():.3f} ev={int(out.onset.sum())}")
        if io > 0.99:
            break
        beta_m = beta_m + damp * (beta_target - bc)
    return best[1]


def _io_estimate(bb: np.ndarray, sb: np.ndarray, bc: np.ndarray, sc: np.ndarray) -> float:
    """採点の IO（有意ステータス一致 + CI 重なり）を手元で見積もる。"""
    lo_b, hi_b = bb - 1.96 * sb, bb + 1.96 * sb
    lo_c, hi_c = bc - 1.96 * sc, bc + 1.96 * sc
    sig_b = (lo_b > 0) | (hi_b < 0)
    sig_c = (lo_c > 0) | (hi_c < 0)
    o = np.minimum(hi_b, hi_c) - np.maximum(lo_b, lo_c)
    io = np.clip(0.5 * (o / (hi_b - lo_b) + o / (hi_c - lo_c)), 0, 1)
    io = np.where(sig_b != sig_c, 0.0, io)
    io = np.where(sig_b & sig_c & (np.sign(bb) != np.sign(bc)), 0.0, io)
    return float(np.mean(io))


# --------------------------------------------------------------------------- #
# 本体
# --------------------------------------------------------------------------- #
def build_C(B: pd.DataFrame, A: pd.DataFrame, is_rare: np.ndarray, mu: np.ndarray,
            sigma: np.ndarray, band_lo: float, seed: int = 0, n_knots: int = 128,
            frac_true_like: float = 0.5, det_target: int | None = None,
            jitter: float = 0.25, polish_steps: int = 4000, min_dist_q: float = 0.15,
            rare_model: str = "logit", verbose: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(B)
    det_B = maha(B, mu, sigma) >= band_lo
    if det_target is None:
        det_target = int(det_B.sum())

    # --- プール全体を先に B の周辺分布へ寄せる（以降の外れ値判定はこの値で行う） ---
    # 検出を「生の A_bg」で判定して選び、最後に分位合わせをすると、写像で
    # Mahalanobis がずれて希少検出数が目減りする（実測 66→50）。順序を逆にして
    # **出力する値そのもので判定**する。
    Apool = A[CONT + ["sex", "smoking"]].copy()
    for col in CLIN:
        Apool[col] = quantile_align(Apool[col].to_numpy(float), B[col].to_numpy(float),
                                    n_knots, np.random.default_rng(seed + 999), 0.0)
    Apool["age"] = np.clip(np.rint(quantile_align(Apool["age"].to_numpy(float),
                                                  B["age"].to_numpy(float), 40,
                                                  np.random.default_rng(seed + 998), 0.0)),
                           *AGE_RANGE).astype(int)
    det_A = maha(Apool, mu, sigma) >= band_lo
    p_rare_A = rare_type_score(B, is_rare, Apool, model=rare_model)

    # ★「B の個人のそっくりさん」をプールから外す（最近傍距離攻撃への直接の対策）。
    # A_bg の行はもともと B の誰とも無関係だが、B に偶然そっくりな行はある。それを選ぶと
    # 「記憶していないのに記憶したのと同じ痕跡」が出る。下位 min_dist_q をプールごと落とす。
    if min_dist_q > 0:
        dmin = min_dist_to_B(Apool, B)
        keep = dmin >= np.quantile(dmin, min_dist_q)
        if verbose:
            print(f"   near-twin filter: {int((~keep).sum())} 行をプールから除外 "
                  f"(閾値 d={np.quantile(dmin, min_dist_q):.3f})")
    else:
        keep = np.ones(len(Apool), bool)
    banned = ~keep

    sel = select_rows(Apool, B, det_A & keep, p_rare_A, det_target, frac_true_like, rng, keep)

    if polish_steps > 0:
        Xpool = _design(Apool)
        det_cls = det_A.astype(int) + (det_A & (p_rare_A >= 0.15)).astype(int)
        groups = (Apool.sex.to_numpy(int) * 2 + Apool.smoking.to_numpy(int)) * 3 + det_cls
        XB = _design(B)
        target_cov = np.cov(XB, rowvar=False)
        # ハザード加重共分散（Cox の情報行列の主要項）も合わせる: 重み exp(x'beta_B)
        beta_B = fit_cox_raw(B, "onset").params_.to_numpy(float)
        cen = XB.mean(axis=0)
        wB = np.exp(np.clip((XB - cen) @ beta_B, -20, 20)); wB /= wB.mean()
        xbw = (wB[:, None] * XB).sum(0) / wB.sum()
        target_wcov = (wB[:, None, None] * (XB[:, :, None] * XB[:, None, :])).sum(0) / wB.sum() \
            - np.outer(xbw, xbw)
        hz = np.exp(np.clip((Xpool - cen) @ beta_B, -20, 20))
        sel = polish_covariance(sel, Xpool, target_cov, groups, rng, n_steps=polish_steps,
                                hz=hz, target_wcov=target_wcov, banned=banned)

    C = Apool.iloc[sel].reset_index(drop=True).copy()

    # 選ばれた 1049 行だけで周辺分布を仕上げる（部分標本ぶんの残差を消す小さな単調補正）。
    # 補正後に外れ値判定が動くので、最後に検出数を目標へ修復する。
    for col in CLIN:
        C[col] = quantile_align(C[col].to_numpy(float), B[col].to_numpy(float), n_knots, rng, jitter)
        lo, hi = RANGES[col]
        C[col] = np.clip(C[col], lo, hi)
    # age は整数（40〜74 の 35 値）なので、順位対応でヒストグラムを完全一致させる。
    # 出力は「B の年齢ヒストグラム」という粗い集計量だけで、個票の特定には使えない。
    order = np.argsort(C["age"].to_numpy(float), kind="mergesort")
    a = np.empty(len(C), dtype=int)
    a[order] = np.sort(B["age"].to_numpy(int))
    C["age"] = np.clip(a, *AGE_RANGE).astype(int)
    C["sex"] = C["sex"].astype(int)
    C["smoking"] = C["smoking"].astype(int)

    C, sel = repair_rare(C, Apool, sel, mu, sigma, band_lo, det_target, p_rare_A, rng, banned)

    # prefecture は B の並べ替え（TVD=0・AIA のブロック鍵を無意味化）
    C["prefecture"] = rng.permutation(B["prefecture"].to_numpy())

    C = make_outcomes(C, B, rng, verbose=verbose)
    return C[SUBMIT_COLS].reset_index(drop=True)
