"""攻撃ライブラリ（MIA / AIA）。攻撃フェーズの提出物づくりと、自分の C の自己監査の両方に使う。

MIA（メンバーシップ推論）
  m1 domias        : p_C/p_A の密度比（参照実装・ルールブック §4.1）
  m2 exact_match   : 連続列の**値の完全/近傍一致**。スワップ・マイクロアグリゲーション・
                     ランクスワップ・k-匿名化など「B の値をそのまま並べ替えて出す」加工を一撃で破る。
                     C の各列の値集合に候補の値がそのまま入っていれば、その人は B に居た。
  m3 nn_gap        : 正規化最近傍距離 d(x, C) を d(x, A_bg) で較正した値。摂動系（ノイズ付加・
                     マイクロアグリゲーション）に効く。DOMIAS より局所的で、記憶した1点を拾う。
  m4 lof_ratio     : k 近傍密度比（k-NN 密度の C/A_bg 比）。KDE より裾（希少層）に強い。
  m5 pref_prior    : 県の周辺分布。B_i は県で偏重サンプリングされているので、C_i の県分布から
                     「相手の地元」が読め、自コホートの候補にその県の人が居れば member 事前確率が上がる。
                     加工では潰せない構造的チャネル（TVD が採点されるため）。
  ensemble         : 上記の順位平均。

AIA（属性推論）
  a1 nn_time       : QI 最近傍の time をそのまま答える（参照実装・§4.2）
  a2 gated         : 「記憶の証拠」があるときだけ NN の time を答え、無ければ打ち切り値 10 を答える。
                     R=(p_m−p_c)/(1−p_c) は対照で当たる分を引くので、自信の無い行で 10 を答えるのは
                     両群同じだけ当たり相殺され、証拠のある行だけが素点として残る。
  a3 gated_suppress: さらに「対照らしい行」にわざと外れ値を答えて p_c を下げる（R の分母を突く）。
                     メトリクスの穴を突く攻撃なので既定では無効。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors

CONT = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
CLIN = ["BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
AIA_QI = ["age", "SBP", "TG", "HDL", "ALT"]
HORIZON = 10.0


def _z(df: pd.DataFrame, cols, mean, std) -> np.ndarray:
    return (df[cols].to_numpy(float) - mean) / std


def _scaler(A: pd.DataFrame, cols):
    mean = A[cols].mean().to_numpy(float)
    std = A[cols].std(ddof=0).to_numpy(float)
    return mean, np.where(std > 0, std, 1.0)


# --------------------------------------------------------------------------- #
# MIA
# --------------------------------------------------------------------------- #
def m1_domias(C: pd.DataFrame, A: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    mean, std = _scaler(A, CONT)
    try:
        ks = gaussian_kde(_z(C, CONT, mean, std).T)
        kr = gaussian_kde(_z(A, CONT, mean, std).T)
    except np.linalg.LinAlgError:
        return np.full(len(X), 0.5)
    t = _z(X, CONT, mean, std).T
    r = ks(t) / np.clip(kr(t), 1e-300, None)
    return r / (1.0 + r)


def m2_exact_match(C: pd.DataFrame, A: pd.DataFrame, X: pd.DataFrame, rel_tol: float = 1e-9) -> np.ndarray:
    """候補の各連続値が C の同じ列に（ほぼ）そのまま存在するか。列ごとの一致数を返す。

    値を並べ替えただけの加工（スワップ/シャッフル/マイクロアグリゲーション/k-匿名化の代表値）は
    ここで完全に割れる。相対許容 rel_tol を少し緩めると「微小ジッタで誤魔化した」加工も拾える。
    """
    score = np.zeros(len(X))
    for col in CLIN:
        cv = np.sort(C[col].to_numpy(float))
        xv = X[col].to_numpy(float)
        pos = np.searchsorted(cv, xv)
        best = np.full(len(xv), np.inf)
        for p in (pos - 1, pos):
            ok = (p >= 0) & (p < len(cv))
            d = np.full(len(xv), np.inf)
            d[ok] = np.abs(cv[np.clip(p, 0, len(cv) - 1)][ok] - xv[ok])
            best = np.minimum(best, d)
        score += (best <= rel_tol * np.maximum(np.abs(xv), 1e-9)).astype(float)
    return score / len(CLIN)


def m3_nn_gap(C: pd.DataFrame, A: pd.DataFrame, X: pd.DataFrame, k: int = 1) -> np.ndarray:
    """log d_k(x, A_bg) − log d_k(x, C)。C の中に「自分そっくりの行」がある人ほど大きい。"""
    mean, std = _scaler(A, CONT)
    xc = _z(X, CONT, mean, std)
    dc = NearestNeighbors(n_neighbors=k).fit(_z(C, CONT, mean, std)).kneighbors(xc)[0][:, -1]
    da = NearestNeighbors(n_neighbors=k + 1).fit(_z(A, CONT, mean, std)).kneighbors(xc)[0][:, -1]
    return np.log(np.clip(da, 1e-12, None)) - np.log(np.clip(dc, 1e-12, None))


def m4_knn_density(C: pd.DataFrame, A: pd.DataFrame, X: pd.DataFrame, k: int = 15) -> np.ndarray:
    """k-NN 密度比 log(rho_C / rho_A)。KDE より裾で解像度が高い（希少層 worst-case 用）。"""
    mean, std = _scaler(A, CONT)
    xc = _z(X, CONT, mean, std)
    dc = NearestNeighbors(n_neighbors=k).fit(_z(C, CONT, mean, std)).kneighbors(xc)[0][:, -1]
    da = NearestNeighbors(n_neighbors=k).fit(_z(A, CONT, mean, std)).kneighbors(xc)[0][:, -1]
    d = len(CONT)
    return d * (np.log(np.clip(da, 1e-12, None)) - np.log(np.clip(dc, 1e-12, None))) \
        + np.log(len(A) / len(C))


def m5_pref_prior(C: pd.DataFrame, A: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """県の周辺分布から作る事前確率 log(p_C(pref)/p_A(pref))。

    B_i は割り当てられた県に偏重しており、その偏りは C_i の県分布に必ず残る（TVD が採点対象）。
    自分のコホートの候補のうち相手の地元県の人は、共有ブロックに入っている確率が高い。
    加工側では消せない構造的チャネル。
    """
    pc = C["prefecture"].value_counts(normalize=True)
    pa = A["prefecture"].value_counts(normalize=True)
    lo = 1.0 / (len(A) + 1)
    return np.log(X["prefecture"].map(pc).fillna(lo).to_numpy(float)) - \
        np.log(X["prefecture"].map(pa).fillna(lo).to_numpy(float))


def _rank01(v: np.ndarray) -> np.ndarray:
    r = pd.Series(v).rank(method="average").to_numpy()
    return (r - 0.5) / len(r)


def mia_ensemble(C: pd.DataFrame, A: pd.DataFrame, X: pd.DataFrame,
                 weights: dict[str, float] | None = None) -> np.ndarray:
    """複数攻撃の順位平均。個別に効く相手が違うので、総当たりでは平均が安定して強い。"""
    w = weights or {"exact": 3.0, "nn": 1.0, "knn": 1.0, "domias": 0.7, "pref": 0.5}
    parts = {
        "exact": m2_exact_match(C, A, X),
        "nn": m3_nn_gap(C, A, X),
        "knn": m4_knn_density(C, A, X),
        "domias": m1_domias(C, A, X),
        "pref": m5_pref_prior(C, A, X),
    }
    # 完全一致が実際に効いている相手（＝値を並べ替えただけ）なら、それだけで決める
    if parts["exact"].max() > 0.5 and (parts["exact"] > 0.5).mean() < 0.5:
        return 0.98 * _rank01(parts["exact"]) + 0.02 * _rank01(parts["nn"])
    tot = sum(w.values())
    return sum(w[k] * _rank01(v) for k, v in parts.items()) / tot


# --------------------------------------------------------------------------- #
# AIA
# --------------------------------------------------------------------------- #
def _aia_nn_raw(C: pd.DataFrame, Q: pd.DataFrame, A: pd.DataFrame, block: bool = True):
    """QI 最近傍の (距離, time) を返す。block=True で (sex, prefecture) ブロック内検索。"""
    C = C.reset_index(drop=True)
    Q = Q.reset_index(drop=True)
    mean, std = _scaler(A, AIA_QI)
    cz = _z(C, AIA_QI, mean, std)
    qz = _z(Q, AIA_QI, mean, std)
    ct = C["time"].to_numpy(float)
    n = len(Q)
    dist = np.full(n, np.inf)
    that = np.full(n, HORIZON)
    if block:
        groups: dict[tuple, np.ndarray] = {}
        for key, idx in C.groupby(["sex", "prefecture"]).groups.items():
            groups[tuple(key) if isinstance(key, tuple) else (key,)] = np.asarray(idx)
        sexg = {s: np.asarray(i) for s, i in C.groupby("sex").groups.items()}
    allidx = np.arange(len(C))
    for i in range(n):
        if block:
            idx = groups.get((Q["sex"].iloc[i], Q["prefecture"].iloc[i]))
            if idx is None or len(idx) == 0:
                idx = sexg.get(Q["sex"].iloc[i], allidx)
        else:
            idx = allidx
        d = np.sqrt(np.mean((cz[idx] - qz[i]) ** 2, axis=1))
        j = int(np.argmin(d))
        dist[i] = d[j]
        that[i] = ct[idx[j]]
    return dist, that


def a1_nn_time(C, Q, A) -> np.ndarray:
    return _aia_nn_raw(C, Q, A)[1]


def a2_gated(C, Q, A, q_gate: float = 0.12, censor_value: float = HORIZON) -> np.ndarray:
    """記憶の証拠（極端に近い最近傍）があるときだけ NN の time を答え、他は打ち切り値。

    ほとんどの人は 10 年で発症しない＝ 10 と答えれば標的でも対照でも当たるので、R には効かない。
    差が出るのは「その人固有の追跡年数を C から読めた」行だけ。よって自信の無い行を
    打ち切り値に寄せるほど、ノイズで p_c を上げずに済み R が伸びる。
    """
    dist, that = _aia_nn_raw(C, Q, A)
    thr = np.quantile(dist, q_gate)
    out = np.full(len(Q), censor_value)
    take = (dist <= thr) & (that < HORIZON - 0.5)
    out[take] = that[take]
    return out


def a3_gated_suppress(C, Q, A, q_gate: float = 0.12, q_far: float = 0.5,
                      wrong_value: float = 4.5) -> np.ndarray:
    """a2 に加え、「遠い＝対照らしい」行にわざと外す値を答えて p_c を下げる。

    R = (p_m − p_c)/(1 − p_c) は分母にも p_c が入るので、対照の正解率を下げると R が伸びる。
    メンバーシップ判定が当てずっぽうなら両群で同じだけ外れて R は不変（clip で下振れも無い）。
    メトリクスの穴を突く攻撃であり、使うかどうかは各チームの判断。
    """
    dist, that = _aia_nn_raw(C, Q, A)
    out = a2_gated(C, Q, A, q_gate)
    far = dist >= np.quantile(dist, q_far)
    out[far] = wrong_value
    return out
