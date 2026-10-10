"""提出前の固定 C.csv に「MIA / AIA 耐性がある」と客観的に言うための監査器。

ルールブック §6.2 の得点式は手元では再現できない（S_ij も B_j も未開示、β_chance は
全攻撃者の提出物に依存する）。そこで本ツールは「得点の予想」ではなく **反証可能な主張**
を作ることを目的にする。主張の型は次の 3 つ:

  (1) 必要条件: C に B の個票がそのまま（あるいは列単位で）残っていない。
      → 完全一致・許容付き一致の件数がゼロ。ゼロでなければ即アウトなので、
        まずここを通す。

  (2) 交換可能性: 「B の行」と「B に居ない同型の行（対照）」が、C から見て
      区別できない。対照は公式 AIA の対照定義（県・性別・年齢帯・発症をそろえた
      A_bg の別人）をそのまま使う。区別できなければ、どんな攻撃も原理的に
      偶然以上には当てられない。指標は
        - DCR share  = P(d(member, C) < d(matched control, C))  … 0.5 が無漏洩
        - 攻撃 AUC   = 各 MIA 攻撃スコアの member/control 判別 AUC … 0.5 が無漏洩
        - TPR@1%FPR  = 公式と同じ動作点（1,049 候補・26 メンバー）を再標本化で読む
      いずれも置換検定 / 二項検定で「偶然の範囲」を同時に出す。

  (3) 検出力: 同じ監査を **陽性対照**（B そのもの、B を k 行だけ混ぜた C）に当てて、
      「この監査は k 行の混入なら検出できる」という下限を実測する。
      (2) が陰性であることは、この下限とセットで初めて意味を持つ。

MIA と AIA の両方について (1)(2)(3) を出し、最後に公式スケール（a_MIA, R）へ写す。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import attacks as AT  # noqa: E402
from audit import _tpr_at  # noqa: E402
from harness import Bench  # noqa: E402

CONT = AT.CONT                      # age BMI SBP TG HDL ALT FPG
AIA_QI = AT.AIA_QI                  # age SBP TG HDL ALT
COLS13 = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT",
          "smoking", "FPG", "time", "onset", "death"]
CONT8 = CONT + ["time"]             # 転帰まで含めた結合ベクトル
LOW_FPR = 0.01
K_MIA = 12.0
TAU = 0.5


# --------------------------------------------------------------------------- #
# 対照集合（公式 AIA の対照定義: 県・性別・年齢帯・発症をそろえた A_bg の別人）
# --------------------------------------------------------------------------- #
def matched_controls(B: pd.DataFrame, A: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """B の各行に 1 対 1 で対照を引く。段階的にキーを緩めて必ず 1 人当てる。"""
    rng = np.random.default_rng(seed)
    a = A.reset_index(drop=True)
    ab = (a["age"] // 5).astype(int).to_numpy()
    keys = [
        list(zip(a["sex"], a["prefecture"], ab, a["onset"])),
        list(zip(a["sex"], ab, a["onset"])),
        list(zip(a["sex"], a["onset"])),
        list(zip(a["onset"])),
    ]
    pools = []
    for k in keys:
        d: dict = {}
        for i, key in enumerate(k):
            d.setdefault(key, []).append(i)
        for v in d.values():
            rng.shuffle(v)
        pools.append(d)
    bb = (B["age"] // 5).astype(int).to_numpy()
    tkeys = [
        list(zip(B["sex"], B["prefecture"], bb, B["onset"])),
        list(zip(B["sex"], bb, B["onset"])),
        list(zip(B["sex"], B["onset"])),
        list(zip(B["onset"])),
    ]
    used: set[int] = set()
    picked, level = [], []
    for r in range(len(B)):
        got = None
        for lv in range(4):
            pool = pools[lv].get(tkeys[lv][r])
            while pool:
                j = pool[-1]
                if j in used:
                    pool.pop()
                    continue
                got = j
                break
            if got is not None:
                break
        if got is None:                       # ここまで来ることは無いが保険
            cand = [i for i in range(len(a)) if i not in used]
            got, lv = int(rng.choice(cand)), 4
        used.add(got)
        picked.append(got)
        level.append(lv)
    out = a.iloc[picked].reset_index(drop=True)
    out.attrs["match_level"] = np.asarray(level)
    return out


# --------------------------------------------------------------------------- #
# (1) 一致検査
# --------------------------------------------------------------------------- #
def _cell_match_frac(C: pd.DataFrame, X: pd.DataFrame, cols, tol: float) -> np.ndarray:
    """X の各行について「その値が C の同じ列に tol 以内で存在する」列の割合。"""
    score = np.zeros(len(X))
    for col in cols:
        cv = np.sort(C[col].to_numpy(float))
        xv = X[col].to_numpy(float)
        pos = np.searchsorted(cv, xv)
        best = np.full(len(xv), np.inf)
        for p in (pos - 1, pos):
            ok = (p >= 0) & (p < len(cv))
            d = np.full(len(xv), np.inf)
            idx = np.clip(p, 0, len(cv) - 1)
            d[ok] = np.abs(cv[idx][ok] - xv[ok])
            best = np.minimum(best, d)
        score += (best <= tol * np.maximum(np.abs(xv), 1e-12)).astype(float)
    return score / len(cols)


def _row_match_count(C: pd.DataFrame, X: pd.DataFrame, cols, tol: float) -> int:
    """X の行のうち「全列が tol 以内で一致する C の行が存在する」件数。"""
    cv = C[cols].to_numpy(float)
    xv = X[cols].to_numpy(float)
    scale = np.maximum(np.abs(xv), 1e-12)
    n = 0
    for i in range(len(xv)):
        rel = np.abs(cv - xv[i]) / scale[i]
        if (rel.max(axis=1) <= tol).any():
            n += 1
    return n


def copy_checks(C: pd.DataFrame, B: pd.DataFrame, R: pd.DataFrame, rare: np.ndarray) -> dict:
    """完全一致・許容付き一致。R は対照（同じ統計を出して基準線にする）。"""
    out = {}
    key13 = lambda df: set(map(tuple, df[COLS13].round(12).to_numpy(object)))
    out["exact_row_13col"] = len(key13(B) & key13(C))
    out["exact_row_13col_control"] = len(key13(R) & key13(C))
    for name, cols in (("cont7", CONT), ("qi5", AIA_QI), ("cont8_with_time", CONT8)):
        for tol, lab in ((0.0, "exact"), (1e-9, "tol1e-9"), (1e-6, "tol1e-6"), (1e-3, "tol1e-3")):
            out[f"row_{name}_{lab}"] = _row_match_count(C, B, cols, tol)
            out[f"row_{name}_{lab}_control"] = _row_match_count(C, R, cols, tol)
    # 列単位（スワップ／マイクロ集約／k-匿名化の検出）
    for tol, lab in ((0.0, "exact"), (1e-9, "tol1e-9"), (1e-6, "tol1e-6")):
        f_b = _cell_match_frac(C, B, CONT, tol)
        f_r = _cell_match_frac(C, R, CONT, tol)
        out[f"cell_cont7_{lab}_mean_B"] = float(f_b.mean())
        out[f"cell_cont7_{lab}_mean_control"] = float(f_r.mean())
        out[f"cell_cont7_{lab}_rows_all7_B"] = int((f_b >= 1.0).sum())
    out["rare_exact_cont7"] = _row_match_count(C, B[rare], CONT, 0.0)
    out["rare_row_cont7_tol1e-6"] = _row_match_count(C, B[rare], CONT, 1e-6)
    return out


# --------------------------------------------------------------------------- #
# (2) 距離: DCR / NNDR
# --------------------------------------------------------------------------- #
def _z(df, cols, mean, std):
    return (df[cols].to_numpy(float) - mean) / std


def dcr_stats(C: pd.DataFrame, B: pd.DataFrame, R: pd.DataFrame, A: pd.DataFrame,
              rare_B: np.ndarray, cols=CONT) -> dict:
    from sklearn.neighbors import NearestNeighbors
    mean = A[cols].mean().to_numpy(float)
    std = A[cols].std(ddof=0).to_numpy(float)
    std = np.where(std > 0, std, 1.0)
    nn = NearestNeighbors(n_neighbors=2).fit(_z(C, cols, mean, std))
    db, ib = nn.kneighbors(_z(B, cols, mean, std))
    dr, _ = nn.kneighbors(_z(R, cols, mean, std))
    d1b, d1r = db[:, 0], dr[:, 0]
    n = len(d1b)
    win = (d1b < d1r).mean()
    se = np.sqrt(0.25 / n)
    out = {
        "dcr_share": float(win),
        "dcr_share_se": float(se),
        "dcr_share_z": float((win - 0.5) / se),
        "dcr_median_B": float(np.median(d1b)),
        "dcr_median_control": float(np.median(d1r)),
        "dcr_p01_B": float(np.quantile(d1b, 0.01)),
        "dcr_p01_control": float(np.quantile(d1r, 0.01)),
        "nndr_median_B": float(np.median(db[:, 0] / np.clip(db[:, 1], 1e-12, None))),
        "nndr_median_control": float(np.median(dr[:, 0] / np.clip(dr[:, 1], 1e-12, None))),
    }
    # 左裾（TPR@1%FPR が見ている領域）: 対照の 1% 分位を閾値にしたときの member 側の割合
    thr = np.quantile(d1r, 0.01)
    out["left_tail_ratio"] = float((d1b <= thr).mean() / 0.01)
    if rare_B.any():
        wr = (d1b[rare_B] < d1r[rare_B]).mean()
        out["dcr_share_rare"] = float(wr)
        out["dcr_share_rare_n"] = int(rare_B.sum())
        out["dcr_share_rare_z"] = float((wr - 0.5) / np.sqrt(0.25 / rare_B.sum()))
        out["dcr_min_rare_B"] = float(d1b[rare_B].min())
        out["dcr_min_rare_control"] = float(d1r[rare_B].min())
    # C 側から見た向き（C の行はどこから来たか）
    nb = NearestNeighbors(n_neighbors=1).fit(_z(B, cols, mean, std))
    na = NearestNeighbors(n_neighbors=1).fit(_z(A, cols, mean, std))
    cz = _z(C, cols, mean, std)
    dcb = nb.kneighbors(cz)[0][:, 0]
    dca = na.kneighbors(cz)[0][:, 0]
    out["C_to_B_median"] = float(np.median(dcb))
    out["C_to_Abg_median"] = float(np.median(dca))
    out["C_closer_to_B_share"] = float((dcb < dca * (len(A) / len(B)) ** (1 / len(cols))).mean())
    return out


# --------------------------------------------------------------------------- #
# (2) 判別: MIA 攻撃スコア → AUC と公式動作点
# --------------------------------------------------------------------------- #
def m6_outcome_nn(C, A, X):
    """転帰 time まで含めた 8 次元の最近傍ギャップ。個票の (共変量, 追跡年数) 対の記憶を見る。"""
    from sklearn.neighbors import NearestNeighbors
    mean = A[CONT8].mean().to_numpy(float)
    std = A[CONT8].std(ddof=0).to_numpy(float)
    std = np.where(std > 0, std, 1.0)
    xz = _z(X, CONT8, mean, std)
    dc = NearestNeighbors(n_neighbors=1).fit(_z(C, CONT8, mean, std)).kneighbors(xz)[0][:, 0]
    da = NearestNeighbors(n_neighbors=2).fit(_z(A, CONT8, mean, std)).kneighbors(xz)[0][:, 1]
    return np.log(np.clip(da, 1e-12, None)) - np.log(np.clip(dc, 1e-12, None))


MIA_ATTACKS = {
    "exact": AT.m2_exact_match,
    "nn_gap": AT.m3_nn_gap,
    "knn": AT.m4_knn_density,
    "domias": AT.m1_domias,
    "pref": AT.m5_pref_prior,
    "outcome_nn": m6_outcome_nn,
    "ensemble": AT.mia_ensemble,
}


def _auc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def mia_report(C: pd.DataFrame, B: pd.DataFrame, R: pd.DataFrame, A: pd.DataFrame,
               rare_B: np.ndarray, rare_R: np.ndarray, n_rep: int = 60,
               n_members: int = 26, n_cand: int = 1049, n_perm: int = 400,
               seed: int = 0) -> dict:
    """member=B / nonmember=対照 で各攻撃を採点し、AUC と公式動作点の両方を出す。

    公式の候補構成（1,049 人中メンバー 26 人、うち希少 1 人程度）は再標本化で再現する。
    スコアは 1 回だけ計算し、標本を取り替えるので速い。
    """
    rng = np.random.default_rng(seed)
    X = pd.concat([B, R], ignore_index=True)
    y = np.concatenate([np.ones(len(B), int), np.zeros(len(R), int)])
    rare = np.concatenate([rare_B, rare_R])
    res = {}
    for name, fn in MIA_ATTACKS.items():
        sc = np.asarray(fn(C, A, X), dtype=float)
        auc = _auc(sc, y)
        auc_rare = _auc(sc[rare], y[rare]) if 0 < y[rare].sum() < rare.sum() else float("nan")
        nb = np.sqrt((len(B) + len(R)) / (12.0 * len(B) * len(R) / (len(B) + len(R))))
        # AUC の帰無 SE（Hanley-McNeil 近似の簡易版）: 0.5 + 1.96*sqrt(1/(12 n_pos n_neg /N))
        se_auc = np.sqrt((len(B) + len(R) + 1) / (12.0 * len(B) * len(R)))
        # 公式動作点
        m_idx = np.flatnonzero(y == 1)
        n_idx = np.flatnonzero(y == 0)
        m_rare = m_idx[rare[m_idx]]
        m_maj = m_idx[~rare[m_idx]]
        ts, packs = [], []
        for _ in range(n_rep):
            take_r = min(1, len(m_rare))
            mm = np.concatenate([rng.choice(m_rare, take_r, replace=False),
                                 rng.choice(m_maj, n_members - take_r, replace=False)])
            nn_ = rng.choice(n_idx, n_cand - n_members, replace=False)
            idx = np.concatenate([mm, nn_])
            sh = rng.permutation(len(idx))
            idx = idx[sh]
            s, t, ra = sc[idx], y[idx], rare[idx]
            t_all = _tpr_at(s, t, LOW_FPR)
            t_rare = _tpr_at(s[ra], t[ra], LOW_FPR) if ra.any() and 0 < t[ra].sum() < ra.sum() else 0.0
            ts.append(max(t_all, t_rare))
            packs.append((s, t, ra))
        beta_raw = float(np.mean(ts))
        betas = np.empty(n_perm)
        for p in range(n_perm):
            vals = []
            for s, t, ra in packs:
                pm = rng.permutation(len(s))
                tt, rr = t[pm], ra[pm]
                a1 = _tpr_at(s, tt, LOW_FPR)
                a2 = _tpr_at(s[rr], tt[rr], LOW_FPR) if rr.any() and 0 < tt[rr].sum() < rr.sum() else 0.0
                vals.append(max(a1, a2))
            betas[p] = np.mean(vals)
        beta_chance = float(np.percentile(betas, 95))
        beta = max(beta_raw - beta_chance + LOW_FPR, LOW_FPR)
        a_mia = float(np.clip((beta / LOW_FPR - 1.0) / (K_MIA - 1.0), 0.0, 1.0))
        # 全行版の動作点 f（陽性 1,049 人・陰性 1,049 人）。SE≈0.003 で解像度が桁違いに高い。
        # 攻撃者ごとのメンバーは 26 人だが、その 26 人は B の一様な部分集合なので
        # 期待 TPR は f に一致する。a_mia_from_f は f をそのまま公式式に入れた換算値。
        yb = np.concatenate([np.ones(len(B), int), np.zeros(len(R), int)])
        # ★ 両側で読む。C が B から「遠ざかりすぎ」ても、攻撃者はスコアを反転させれば
        #   同じだけメンバーを当てられる。片側だけ見ると過保護を安全と誤認する。
        f_pos, f_neg = _tpr_at(sc, yb, LOW_FPR), _tpr_at(-sc, yb, LOW_FPR)
        f_all = max(f_pos, f_neg)
        if 0 < yb[rare].sum() < rare.sum():
            f_rare = max(_tpr_at(sc[rare], yb[rare], LOW_FPR),
                         _tpr_at(-sc[rare], yb[rare], LOW_FPR))
        else:
            f_rare = float("nan")
        se_f = float(np.sqrt(LOW_FPR * (1 - LOW_FPR) / len(B)))
        res[name] = {
            "auc": auc, "auc_rare": auc_rare, "auc_null_se": float(se_auc),
            "auc_z": float((auc - 0.5) / se_auc),
            "f_tpr_full": float(f_all), "f_tpr_pos": float(f_pos), "f_tpr_neg": float(f_neg),
            "f_tpr_full_rare": float(f_rare), "f_se": se_f,
            "f_z": float((f_all - LOW_FPR) / se_f),
            "a_mia_from_f": float(np.clip((max(f_all, 0.0) / LOW_FPR - 1.0) / (K_MIA - 1.0), 0, 1)),
            "beta_raw": beta_raw, "beta_chance": beta_chance, "beta": beta, "a_mia": a_mia,
        }
    res["_worst_a_mia"] = max(v["a_mia"] for v in res.values() if isinstance(v, dict))
    res["_worst_auc_z"] = max(v["auc_z"] for v in res.values() if isinstance(v, dict))
    res["_worst_f"] = max(v["f_tpr_full"] for v in res.values() if isinstance(v, dict))
    res["_worst_a_mia_from_f"] = max(v["a_mia_from_f"] for v in res.values() if isinstance(v, dict))
    return res


# --------------------------------------------------------------------------- #
# (2) AIA
# --------------------------------------------------------------------------- #
def _aia_gbm(C, Q, A):
    """C だけで学習した勾配ブースティングで time を予測（NN より強い一般化攻撃）。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    feat = AIA_QI + ["sex"]
    Xtr = C[feat].to_numpy(float)
    Xte = Q[feat].to_numpy(float)
    m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Xtr, C["time"].to_numpy(float))
    return m.predict(Xte)


AIA_ATTACKS = {
    "nn_time": AT.a1_nn_time,
    "gated": AT.a2_gated,
    "gbm": _aia_gbm,
    "const10": lambda C, Q, A: np.full(len(Q), 10.0),
}


def aia_report(C: pd.DataFrame, B: pd.DataFrame, R: pd.DataFrame, A: pd.DataFrame,
               rare_B: np.ndarray, rare_R: np.ndarray, n_seed: int = 200,
               n_target: int = 163, seed: int = 0) -> dict:
    """標的=B の全行 / 対照=1 対 1 対照 で成否を 1 回だけ計算し、
    公式サイズ（163 対）は対を再標本化して分布を出す。"""
    rng = np.random.default_rng(seed)
    both = pd.concat([B, R], ignore_index=True)
    nb = len(B)
    res = {}
    for name, fn in AIA_ATTACKS.items():
        hat = np.asarray(fn(C, both, A), dtype=float)
        ok = np.abs(hat - both["time"].to_numpy(float)) <= TAU
        sm, sc_ = ok[:nb].astype(float), ok[nb:].astype(float)
        pm, pc = float(sm.mean()), float(sc_.mean())
        r_full = float(np.clip((pm - pc) / max(1 - pc, 1e-9), 0, 1))
        # 対でのブートストラップ CI
        boot = []
        for _ in range(400):
            j = rng.integers(0, nb, nb)
            a, b = sm[j].mean(), sc_[j].mean()
            boot.append(np.clip((a - b) / max(1 - b, 1e-9), 0, 1))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # 公式サイズ（163 対、発症 16 人・希少 13 人の内訳をそろえる）
        onset = B["onset"].to_numpy().astype(bool)
        strata = {
            "rare_onset": np.flatnonzero(rare_B & onset),
            "rare_noonset": np.flatnonzero(rare_B & ~onset),
            "maj_onset": np.flatnonzero(~rare_B & onset),
            "maj_noonset": np.flatnonzero(~rare_B & ~onset),
        }
        want = {"rare_onset": 4, "rare_noonset": 9, "maj_onset": 12, "maj_noonset": 138}
        rs, rs_rare = [], []
        for _ in range(n_seed):
            idx = []
            for k, w in want.items():
                pool = strata[k]
                if len(pool) == 0:
                    continue
                idx.append(rng.choice(pool, min(w, len(pool)), replace=len(pool) < w))
            idx = np.concatenate(idx)
            a, b = sm[idx].mean(), sc_[idx].mean()
            rs.append(np.clip((a - b) / max(1 - b, 1e-9), 0, 1))
            ridx = idx[rare_B[idx]]
            if len(ridx) > 2:
                ar, br = sm[ridx].mean(), sc_[ridx].mean()
                se = np.sqrt(ar * (1 - ar) / len(ridx) + br * (1 - br) / len(ridx))
                rs_rare.append(np.clip((ar - br - 1.64 * se) / max(1 - br, 1e-9), 0, 1))
        # 情報のある部分集合（追跡年数が打ち切り 10 年でない人）。ここは「10」と答えても当たらない
        # ので、当たれば C から個人の追跡年数を読めたことの直接の証拠になる。検出力が高い。
        inf_m = B["time"].to_numpy(float) < 9.5
        inf_c = R["time"].to_numpy(float) < 9.5
        pmi = float(sm[inf_m].mean()) if inf_m.any() else float("nan")
        pci = float(sc_[inf_c].mean()) if inf_c.any() else float("nan")
        sei = float(np.sqrt(pmi * (1 - pmi) / max(inf_m.sum(), 1)
                            + pci * (1 - pci) / max(inf_c.sum(), 1)))
        res[name] = {
            "p_member_informative": pmi, "p_control_informative": pci,
            "n_informative": [int(inf_m.sum()), int(inf_c.sum())],
            "delta_informative": pmi - pci, "delta_informative_se": sei,
            "delta_informative_z": float((pmi - pci) / sei) if sei > 0 else 0.0,
            "p_member_full": pm, "p_control_full": pc, "R_full": r_full,
            "R_full_ci": [float(lo), float(hi)],
            "R_official_mean": float(np.mean(rs)), "R_official_p95": float(np.percentile(rs, 95)),
            "R_official_max": float(np.max(rs)),
            "R_rare_mean": float(np.mean(rs_rare)) if rs_rare else float("nan"),
            "R_rare_p95": float(np.percentile(rs_rare, 95)) if rs_rare else float("nan"),
        }
    res["_worst_R_full"] = max(v["R_full"] for k, v in res.items()
                               if isinstance(v, dict) and k != "const10")
    res["_worst_R_official_p95"] = max(v["R_official_p95"] for k, v in res.items()
                                       if isinstance(v, dict) and k != "const10")
    return res


# --------------------------------------------------------------------------- #
# (3) 検出力: 陽性対照と検出下限
# --------------------------------------------------------------------------- #
def make_anchor(kind: str, B: pd.DataFrame, pool: pd.DataFrame, k: int = 0,
                seed: int = 7, comp: pd.DataFrame | None = None) -> pd.DataFrame:
    """陽性/陰性の対照 C を作る。pool は対照 R と重ならない A_bg の残り。

    identity    : C = B（最悪ケース。監査が飽和することを確認する）
    null        : C = pool から 1,049 行（B と個人の重なりゼロ・構成もそろえない）
    null_matched: C = pool から B の (県,性別,年齢帯,発症) 構成をそろえて 1,049 行。
                  「有用性は満点だが記憶はゼロ」の理想の匿名化データ。**判定の基準線はこれ**。
    inject_k    : null_matched の k 行を B の実在行で置き換える（検出下限を測る）。
    """
    rng = np.random.default_rng(seed)
    if kind == "identity":
        return B[COLS13].reset_index(drop=True).copy()
    if kind == "null":
        return pool.iloc[rng.choice(len(pool), len(B), replace=False)][COLS13].reset_index(drop=True).copy()
    base = (comp if comp is not None else matched_controls(B, pool, seed=seed))[COLS13]
    base = base.reset_index(drop=True).copy()
    if kind == "null_matched":
        return base
    if kind == "inject":
        rows = B.iloc[rng.choice(len(B), k, replace=False)][COLS13].reset_index(drop=True)
        pos = rng.choice(len(base), k, replace=False)
        base.iloc[pos] = rows.to_numpy()
        return base
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
def audit_one(C: pd.DataFrame, ctx: dict, quick: bool = False) -> dict:
    B, R, A = ctx["B"], ctx["R"], ctx["A"]
    rb, rr = ctx["rare_B"], ctx["rare_R"]
    out = {}
    out["format"] = {
        "n_rows": int(len(C)),
        "n_rows_ok": bool(len(C) == len(B)),
        "cols_ok": bool(list(C.columns) == COLS13),
        "n_rare_detected": int(ctx["bench"]._rare_mask(C).sum()),
    }
    out["copy"] = copy_checks(C, B, R, ctx["rare_B_true"])
    out["dcr"] = dcr_stats(C, B, R, A, rb)
    out["mia"] = mia_report(C, B, R, A, rb, rr,
                            n_rep=20 if quick else 60, n_perm=200 if quick else 400)
    out["aia"] = aia_report(C, B, R, A, ctx["rare_B_true"], rr, n_seed=80 if quick else 200)
    out["summary"] = {
        "exact_copies_cont7": out["copy"]["row_cont7_exact"],
        "cell_match_mean_B": out["copy"]["cell_cont7_exact_mean_B"],
        "dcr_share": out["dcr"]["dcr_share"],
        "dcr_share_z": out["dcr"]["dcr_share_z"],
        "worst_mia_auc_z": out["mia"]["_worst_auc_z"],
        "worst_f_tpr_full": out["mia"]["_worst_f"],
        "worst_a_mia_from_f": out["mia"]["_worst_a_mia_from_f"],
        "worst_a_mia": out["mia"]["_worst_a_mia"],
        "worst_R_full": out["aia"]["_worst_R_full"],
        "worst_R_official_p95": out["aia"]["_worst_R_official_p95"],
    }
    a_mia = max(out["mia"]["_worst_a_mia"], out["mia"]["_worst_a_mia_from_f"])
    a_aia = out["aia"]["_worst_R_official_p95"]
    out["summary"]["anon_lower_bound"] = float(1.0 - max(a_mia, a_aia))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="*", help="C.csv のパス")
    ap.add_argument("--dist", default=str(HERE / "data_prelim_team05"))
    ap.add_argument("--out", default=str(HERE / "submit_audit_out" / "report.json"))
    ap.add_argument("--anchors", action="store_true", help="陽性/陰性対照と検出下限も測る")
    ap.add_argument("--anchor-set", default="all", choices=["all", "null_only"],
                    help="null_only: 帰無アンカーだけ（複数 seed で帰無分布を作るとき用）")
    ap.add_argument("--inject", default="1,3,10,30,100")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bench = Bench(Path(args.dist))
    B, A = bench.B[COLS13].reset_index(drop=True), bench.A_bg
    R = matched_controls(B, A, seed=args.seed)
    lv = R.attrs["match_level"]
    # ★ 対照 R は A_bg の中にあるので、攻撃の参照にも A_bg をそのまま使うと
    #   「参照集合に自分が居る」ぶんだけ対照のスコアが下がり、B が有利に見える（偽の漏洩）。
    #   実測では漏洩ゼロの C でも domias AUC=0.69 / nn_gap AUC=0.68 が出た。
    #   本番の非メンバーは他コホートの人＝A_bg には居ないので、参照から R を抜いて再現する。
    used = set(map(tuple, R[COLS13].to_numpy(object)))
    keep = ~np.array([tuple(t) in used for t in A[COLS13].to_numpy(object)])
    A_ref = A[keep].reset_index(drop=True)          # 攻撃の参照（B も R も含まない）
    pool = A_ref                                     # アンカー素材（評価点ではないので参照と重なって良い）
    ctx = {"B": B, "R": R[COLS13].reset_index(drop=True), "A": A_ref, "A_full": A, "bench": bench,
           "rare_B": bench._rare_mask(B), "rare_R": bench._rare_mask(R),
           "rare_B_true": bench.is_rare, "pool": pool}
    print(f"[対照] 完全一致キー(県+性別+年齢帯+発症)={int((lv==0).sum())}/{len(B)}  "
          f"緩和1={int((lv==1).sum())} 緩和2={int((lv==2).sum())} 緩和3={int((lv==3).sum())}")
    print(f"[希少] 検出器 B={int(ctx['rare_B'].sum())} 対照={int(ctx['rare_R'].sum())} "
          f"/ 真ラベル B={int(ctx['rare_B_true'].sum())}   [攻撃の参照 A_ref] {len(A_ref)}行")

    report = {}
    targets: list[tuple[str, pd.DataFrame]] = []
    if args.anchors:
        comp = matched_controls(B, pool, seed=args.seed + 101)
        targets.append(("ANCHOR_null_matched", make_anchor("null_matched", B, pool, comp=comp)))
        if args.anchor_set == "all":
            targets.append(("ANCHOR_null", make_anchor("null", B, pool)))
            for k in [int(x) for x in args.inject.split(",") if x]:
                targets.append((f"ANCHOR_inject{k}", make_anchor("inject", B, pool, k, comp=comp)))
            targets.append(("ANCHOR_identity", make_anchor("identity", B, pool)))
    for p in args.candidates:
        C = pd.read_csv(p)
        if "record_id" in C.columns:
            C = C.drop(columns=["record_id"])
        targets.append((p, C))

    for label, C in targets:
        print(f"\n===== {label} =====", flush=True)
        r = audit_one(C, ctx, quick=args.quick)
        report[label] = r
        s = r["summary"]
        print(f"  形式: rows={r['format']['n_rows']} 希少検出={r['format']['n_rare_detected']}")
        print(f"  一致: 完全一致(7連続列)={s['exact_copies_cont7']}行 / "
              f"対照={r['copy']['row_cont7_exact_control']}行  "
              f"セル一致率={s['cell_match_mean_B']:.4f} (対照 {r['copy']['cell_cont7_exact_mean_control']:.4f})")
        print(f"  距離: DCR share={s['dcr_share']:.4f} (z={s['dcr_share_z']:+.2f})  "
              f"左裾比={r['dcr']['left_tail_ratio']:.2f}  希少 share={r['dcr'].get('dcr_share_rare', float('nan')):.3f}")
        for k, v in r["mia"].items():
            if isinstance(v, dict):
                print(f"  MIA {k:11s} AUC={v['auc']:.4f}(z={v['auc_z']:+5.1f}) "
                      f"f=TPR@1%={v['f_tpr_full']:.4f}[{v['f_tpr_pos']:.3f}/{v['f_tpr_neg']:.3f}]"
                      f"(z={v['f_z']:+5.1f}, 希少{v['f_tpr_full_rare']:.3f}) "
                      f"a_mia|f={v['a_mia_from_f']:.3f} | beta={v['beta_raw']:.4f}/{v['beta_chance']:.4f} "
                      f"a_mia={v['a_mia']:.3f}")
        for k, v in r["aia"].items():
            if isinstance(v, dict):
                print(f"  AIA {k:8s} pm={v['p_member_full']:.4f} pc={v['p_control_full']:.4f} "
                      f"R_full={v['R_full']:.4f} CI={v['R_full_ci'][0]:.3f}-{v['R_full_ci'][1]:.3f} "
                      f"R163_p95={v['R_official_p95']:.4f} | 情報部分 "
                      f"{v['p_member_informative']:.3f}vs{v['p_control_informative']:.3f} "
                      f"(z={v['delta_informative_z']:+.2f})")
        print(f"  >>> worst a_MIA={s['worst_a_mia']:.4f}  worst R(163,p95)={s['worst_R_official_p95']:.4f}"
              f"  => Anon 下限 ≈ {s['anon_lower_bound']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\n[保存] {args.out}")


if __name__ == "__main__":
    main()
