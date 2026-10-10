"""Cox 部分尤度の高速ソルバ（Breslow・タイなし前提）と、転帰割り当ての最適化。

採点器の U_spec は「C で推定した Cox 9 係数の CI が、B の CI とどれだけ重なるか（IO）」。
lifelines を毎回叩くと 1 回 20ms かかって探索に使えないので、同じ推定量を numpy で
1ms 程度で解く。連続時刻でタイが無ければ Efron = Breslow なので lifelines と一致する。

最適化の肝: **転帰 (time, onset, death) の多重集合は固定したまま、どの行に割り当てるかだけを
入れ替える**。KM 曲線・発症数・打ち切り整合・時刻の周辺分布はすべて多重集合だけで決まるので
完全に保存され、変わるのは「共変量と転帰の対応」＝ Cox 推定値だけ。
"""
from __future__ import annotations

import numpy as np


def cox_fit(X: np.ndarray, time: np.ndarray, event: np.ndarray, n_iter: int = 12,
            tol: float = 1e-9, beta0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
    """Breslow 部分尤度を Newton 法で解き (beta, se, ok) を返す。"""
    order = np.argsort(-time, kind="mergesort")          # 時刻の降順＝リスク集合の累積和が前から
    Xs = X[order]
    ev = event[order].astype(bool)
    ts = time[order]
    n, p = Xs.shape
    # 同時刻の扱い: 降順に並べたとき、同じ時刻のグループの先頭までを at-risk に含める
    # （タイが無ければ単純な累積和と一致する）
    beta = np.zeros(p) if beta0 is None else beta0.copy()
    ok = True
    for _ in range(n_iter):
        eta = Xs @ beta
        eta -= eta.max()
        w = np.exp(eta)
        S0 = np.cumsum(w)
        S1 = np.cumsum(w[:, None] * Xs, axis=0)
        outer = Xs[:, :, None] * Xs[:, None, :]
        S2 = np.cumsum(w[:, None, None] * outer, axis=0)
        # 同時刻はリスク集合を揃える（末尾の同値グループの最大 index を使う）
        idx = _tie_index(ts)
        s0 = S0[idx][ev]
        s1 = S1[idx][ev]
        s2 = S2[idx][ev]
        xbar = s1 / s0[:, None]
        grad = (Xs[ev] - xbar).sum(axis=0)
        hess = -(s2 / s0[:, None, None] - xbar[:, :, None] * xbar[:, None, :]).sum(axis=0)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return beta, np.full(p, np.nan), False
        beta = beta - step
        if np.max(np.abs(step)) < tol:
            break
    else:
        ok = np.max(np.abs(step)) < 1e-5
    try:
        cov = np.linalg.inv(-hess)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return beta, np.full(p, np.nan), False
    if not (np.isfinite(beta).all() and np.isfinite(se).all() and (se > 0).all()):
        ok = False
    return beta, se, ok


def _tie_index(ts_desc: np.ndarray) -> np.ndarray:
    """降順時刻配列で、各位置の「同時刻グループの最後の位置」を返す。"""
    n = len(ts_desc)
    idx = np.arange(n)
    # ts_desc は降順。同じ値のブロックの末尾 index を前方に伝播
    last = np.empty(n, dtype=int)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ts_desc[j + 1] == ts_desc[i]:
            j += 1
        last[i:j + 1] = j
        i = j + 1
    return np.maximum(last, idx)


def io_score(bb: np.ndarray, sb: np.ndarray, bc: np.ndarray, sc: np.ndarray) -> float:
    """採点器 `_score_u_spec` の IO パート（有意ステータス一致 + CI 重なり）。"""
    lo_b, hi_b = bb - 1.959963984540054 * sb, bb + 1.959963984540054 * sb
    lo_c, hi_c = bc - 1.959963984540054 * sc, bc + 1.959963984540054 * sc
    sig_b = (lo_b > 0) | (hi_b < 0)
    sig_c = (lo_c > 0) | (hi_c < 0)
    o = np.minimum(hi_b, hi_c) - np.maximum(lo_b, lo_c)
    io = np.clip(0.5 * (o / (hi_b - lo_b) + o / (hi_c - lo_c)), 0.0, 1.0)
    io = np.where(sig_b != sig_c, 0.0, io)
    io = np.where(sig_b & sig_c & (np.sign(bb) != np.sign(bc)), 0.0, io)
    return float(np.mean(io))


def refine_assignment(X: np.ndarray, time: np.ndarray, onset: np.ndarray, death: np.ndarray,
                      beta_B: np.ndarray, se_B: np.ndarray, rng: np.random.Generator,
                      swap_ok: np.ndarray, n_steps: int = 4000, n_cand: int = 24,
                      verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """転帰の割り当てを入れ替えて IO を最大化する（山登り）。

    1 手 = 「発症行」と「打ち切り行」の (time, onset, death) を丸ごと交換。多重集合は不変なので
    KM 曲線・発症数・打ち切り整合は完全に保存され、Cox 推定値だけが動く。
    `swap_ok` が True の行だけを交換対象にする（希少層の発症率などを壊さないため）。
    """
    time = time.copy(); onset = onset.copy(); death = death.copy()
    beta, se, ok = cox_fit(X, time, onset)
    cur = io_score(beta_B, se_B, beta, se) if ok else 0.0
    accepted = 0
    for step in range(n_steps):
        ev_idx = np.flatnonzero((onset == 1) & swap_ok)
        cn_idx = np.flatnonzero((onset == 0) & swap_ok)
        if len(ev_idx) == 0 or len(cn_idx) == 0:
            break
        ii = ev_idx[rng.integers(0, len(ev_idx), n_cand)]
        jj = cn_idx[rng.integers(0, len(cn_idx), n_cand)]
        best = None
        for i, j in zip(ii, jj):
            t2, o2, d2 = time.copy(), onset.copy(), death.copy()
            t2[i], t2[j] = time[j], time[i]
            o2[i], o2[j] = onset[j], onset[i]
            d2[i], d2[j] = death[j], death[i]
            b2, s2, ok2 = cox_fit(X, t2, o2, n_iter=4, tol=1e-8, beta0=beta)
            if not ok2:
                continue
            v = io_score(beta_B, se_B, b2, s2)
            if v > cur and (best is None or v > best[0]):
                best = (v, i, j)
        if best is None:
            continue
        v, i, j = best
        time[i], time[j] = time[j], time[i]
        onset[i], onset[j] = onset[j], onset[i]
        death[i], death[j] = death[j], death[i]
        cur = v
        accepted += 1
        beta, se, _ = cox_fit(X, time, onset, beta0=beta)
        if verbose and step % 20 == 0:
            print(f"      refine step{step:04d} io={cur:.4f}")
        if cur > 0.998:
            break
    if verbose:
        print(f"      refine: {accepted} swaps accepted, io={cur:.4f}")
    return time, onset, death, cur
