"""自己監査: 手元では採点できない匿名性 Anon を、ルールブック §6.2 の式で近似する。

サーバ側との違いは「攻撃者が誰か」だけなので、
  * MIA: 攻撃者 j の候補 1,049 人（うちメンバー 26 人）を A_bg と B から合成して J 人ぶん作り、
         t_j = max(TPR@1%FPR 全体, 希少層) → β_raw = mean_j t_j
         → (truth, rare) を対にしたまま入れ替える置換検定で β_chance（95 パーセンタイル）
         → β = max(β_raw − β_chance + 0.01, 0.01) → a_MIA = clip((β/0.01−1)/11, 0, 1)
  * AIA: 標的 163 人（B から）＋ 対照 163 人（A_bg から県・性別・年齢帯・発症をそろえて）で
         R = clip((p_m − p_c)/(1 − p_c), 0, 1)、希少層は 1.64·SE のデッドゾーン付きで max。
を再現する。Anon ≈ 1 − max(a_MIA, R_AIA)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import attacks as AT

K_MIA = 12.0
LOW_FPR = 0.01


_TIE_RNG = np.random.default_rng(20260811)


def _tpr_at(conf: np.ndarray, truth: np.ndarray, fpr: float = LOW_FPR) -> float:
    """TPR@FPR。同点はランダムに並べ替える。

    ★同点をランダムに割らないと、確信度が定数の攻撃（例: 一致ゼロの完全一致照合）で
    argsort が入力順を返し、陽性を先に並べた候補表では TPR=1.0 という嘘が出る。
    本番採点でも候補は record_id 順＝メンバーシップと無関係なので、ランダム化が正しい近似。
    """
    conf = conf + _TIE_RNG.uniform(0, 1e-12, len(conf)) * (np.ptp(conf) + 1e-12)
    order = np.argsort(-conf, kind="mergesort")
    y = truth[order]
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return 0.0
    tpr = np.concatenate([[0.0], np.cumsum(y) / npos])
    fp = np.concatenate([[0.0], np.cumsum(1 - y) / nneg])
    return float(np.interp(fpr, fp, tpr))


def mia_audit(C: pd.DataFrame, bench, attack=AT.m1_domias, n_attackers: int = 20,
              n_members: int = 26, n_perm: int = 2000, seed: int = 0,
              members: pd.DataFrame | None = None, member_rare: np.ndarray | None = None,
              negatives: pd.DataFrame | None = None, neg_rare: np.ndarray | None = None) -> dict:
    """メンバー/非メンバー候補を合成して β と a_MIA を出す。

    既定の非メンバーは A_bg だが、**A_bg は県構成が全チーム平均なのでコホートとの分布差が
    あり、記憶していなくても密度比が効いてしまう**（＝過大評価）。本番の候補は攻撃者の
    コホート B_j の中の人なので、記憶だけを測りたいときは `negatives` に
    「同じ加工に使わなかった同一コホートの行（ホールドアウト）」を渡すこと。
    """
    rng = np.random.default_rng(seed)
    B, A = bench.B, bench.A_bg
    M = B if members is None else members
    M_rare = bench.is_rare if member_rare is None else member_rare
    N = A if negatives is None else negatives
    N_rare = bench._rare_mask(N) if neg_rare is None else neg_rare
    rare_idx = np.flatnonzero(M_rare)
    maj_idx = np.flatnonzero(~M_rare)
    n_neg = min(len(B) - n_members, len(N))
    ts, packs = [], []
    for _ in range(n_attackers):
        # メンバー 26 人（希少の内数はコホート比率どおり 1 人程度）
        m = np.concatenate([rng.choice(rare_idx, min(1, len(rare_idx)), replace=False),
                            rng.choice(maj_idx, n_members - min(1, len(rare_idx)), replace=False)])
        neg = rng.choice(len(N), n_neg, replace=False)
        cand = pd.concat([M.iloc[m], N.iloc[neg]], ignore_index=True)
        truth = np.concatenate([np.ones(len(m), int), np.zeros(len(neg), int)])
        rare = np.concatenate([M_rare[m], N_rare[neg]])
        # ★行順で漏れないようシャッフル（確信度が定数のとき argsort が入力順を返すため）
        sh = rng.permutation(len(cand))
        cand, truth, rare = cand.iloc[sh].reset_index(drop=True), truth[sh], rare[sh]
        conf = np.asarray(attack(C, A, cand), dtype=float)
        t_all = _tpr_at(conf, truth)
        t_rare = _tpr_at(conf[rare], truth[rare]) if rare.any() and truth[rare].sum() > 0 else 0.0
        ts.append(max(t_all, t_rare))
        packs.append((conf, truth, rare))
    beta_raw = float(np.mean(ts))

    # 置換検定（truth と rare を対にしたまま入れ替える＝人数構成は保存）
    betas = np.empty(n_perm)
    for p in range(n_perm):
        vals = []
        for conf, truth, rare in packs:
            perm = rng.permutation(len(conf))
            tr, ra = truth[perm], rare[perm]
            t_all = _tpr_at(conf, tr)
            t_rare = _tpr_at(conf[ra], tr[ra]) if ra.any() and tr[ra].sum() > 0 else 0.0
            vals.append(max(t_all, t_rare))
        betas[p] = np.mean(vals)
    beta_chance = float(np.percentile(betas, 95))
    beta = max(beta_raw - beta_chance + LOW_FPR, LOW_FPR)
    a_mia = float(np.clip((beta / LOW_FPR - 1.0) / (K_MIA - 1.0), 0.0, 1.0))
    return {"beta_raw": beta_raw, "beta_chance": beta_chance, "beta": beta, "a_mia": a_mia}


def make_challenge(bench, n_target: int = 163, seed: int = 0):
    """標的（B から）と対照（A_bg から県・性別・年齢帯・発症をそろえて）を作る。"""
    rng = np.random.default_rng(seed)
    B, A = bench.B, bench.A_bg
    tgt = B.iloc[rng.choice(len(B), n_target, replace=False)].reset_index(drop=True)
    ctl = bench._match_controls(tgt, rng)
    return tgt, ctl


def aia_audit(C: pd.DataFrame, bench, attack=AT.a1_nn_time, seed: int = 0,
              n_target: int = 163, tau: float = 0.5) -> dict:
    tgt, ctl = make_challenge(bench, n_target, seed)
    A = bench.A_bg
    both = pd.concat([tgt, ctl], ignore_index=True)
    hat = np.asarray(attack(C, both, A), dtype=float)
    succ = np.abs(hat - both["time"].to_numpy(float)) <= tau
    nm = len(tgt)
    pm, pc = float(succ[:nm].mean()), float(succ[nm:].mean())
    r_all = float(np.clip((pm - pc) / max(1 - pc, 1e-9), 0, 1))
    rm = bench._rare_mask(tgt)
    rc = bench._rare_mask(ctl)
    r_rare = float("nan")
    if rm.any() and rc.any():
        pmr, pcr = float(succ[:nm][rm].mean()), float(succ[nm:][rc].mean())
        se = np.sqrt(pmr * (1 - pmr) / rm.sum() + pcr * (1 - pcr) / rc.sum())
        r_rare = float(np.clip((pmr - pcr - 1.64 * se) / max(1 - pcr, 1e-9), 0, 1))
    r = max(r_all, 0.0 if np.isnan(r_rare) else r_rare)
    return {"R": r, "R_all": r_all, "R_rare": r_rare, "p_member": pm, "p_control": pc}


def leak_auc(build_fn, bench, n_hold: int = 300, n_splits: int = 4, seed: int = 0,
             attacks: dict | None = None) -> dict:
    """記憶の**高感度**検出: 加工に使った行 vs 使わなかった行を攻撃スコアで見分けられるか（AUC）。

    公式の a_MIA は「候補 1,049 人中メンバー 26 人の TPR@1%FPR」なので、1 回の測定が
    ほぼ 0/1 の粗い量になり、監査の乱数だけで β_raw が 0.03〜0.65 も振れる（実測）。
    設定の比較にはまったく足りないので、同じ攻撃スコアを **全行（749 vs 300）の AUC** で読む。
    AUC=0.5 が「漏れていない」。希少層だけの AUC も別に出す（worst-case はここに出るため）。

    AUC は公式得点ではないが、漏洩の有無の判定力が桁違いに高い。運用は
    「まず AUC で漏れていないことを確かめ、最後に公式式で a_MIA を出す」の二段構え。
    """
    atk = attacks or {"domias": AT.m1_domias, "nn_gap": AT.m3_nn_gap, "knn": AT.m4_knn_density,
                      "exact": AT.m2_exact_match}
    B, A = bench.B, bench.A_bg
    acc: dict[str, list[float]] = {k: [] for k in atk}
    acc_rare: dict[str, list[float]] = {k: [] for k in atk}
    for s in range(n_splits):
        rng = np.random.default_rng(seed + s)
        idx = rng.permutation(len(B))
        out_i, in_i = idx[:n_hold], idx[n_hold:]
        B_in = B.iloc[in_i].reset_index(drop=True)
        C = build_fn(B_in, bench.is_rare[in_i], s)
        y = np.zeros(len(B), int)
        y[in_i] = 1
        rare = bench.is_rare
        for name, fn in atk.items():
            sc = np.asarray(fn(C, A, B), dtype=float)
            acc[name].append(_auc(sc, y))
            m = rare
            if m.sum() > 3 and 0 < y[m].sum() < m.sum():
                acc_rare[name].append(_auc(sc[m], y[m]))
    return {"auc": {k: float(np.mean(v)) for k, v in acc.items()},
            "auc_rare": {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc_rare.items()},
            "auc_sd": {k: float(np.std(v)) for k, v in acc.items()}}


def leak_tpr(build_fn, bench, n_hold: int = 300, n_splits: int = 6, seed: int = 0,
             attacks: dict | None = None, fpr: float = 0.01) -> dict:
    """公式と同じ動作点（TPR@1%FPR）を、**検出力を上げて**測る。

    公式 a_MIA は陽性 26 人しかないので 1 回の測定がほぼ 0/1 になり設定比較に使えない。
    ここでは陽性＝加工に使った行から 300 人、陰性＝使わなかった 300 人にして同じ動作点を読む
    （陽性が 11 倍なので SE は 1/3.5、分割平均でさらに 1/√n_splits）。
    帰無水準は同じ設定でラベルを入れ替えた置換の 95 パーセンタイル。
    """
    atk = attacks or {"domias": AT.m1_domias, "nn_gap": AT.m3_nn_gap, "knn": AT.m4_knn_density,
                      "exact": AT.m2_exact_match}
    B, A = bench.B, bench.A_bg
    res: dict[str, dict[str, list[float]]] = {
        k: {"all": [], "rare": [], "null": [], "null_rare": []} for k in atk}
    rng0 = np.random.default_rng(seed)
    for s in range(n_splits):
        rng = np.random.default_rng(seed + s)
        idx = rng.permutation(len(B))
        out_i, in_i = idx[:n_hold], idx[n_hold:]
        B_in = B.iloc[in_i].reset_index(drop=True)
        C = build_fn(B_in, bench.is_rare[in_i], s)
        pos_i = rng.choice(in_i, n_hold, replace=False)
        cand_i = np.concatenate([pos_i, out_i])
        truth = np.concatenate([np.ones(len(pos_i), int), np.zeros(len(out_i), int)])
        rare = bench.is_rare[cand_i]
        cand = B.iloc[cand_i].reset_index(drop=True)
        for name, fn in atk.items():
            sc = np.asarray(fn(C, A, cand), dtype=float)
            res[name]["all"].append(_tpr_at(sc, truth, fpr))
            if rare.sum() > 3 and 0 < truth[rare].sum() < rare.sum():
                res[name]["rare"].append(_tpr_at(sc[rare], truth[rare], fpr))
            res[name]["null"].append(np.percentile(
                [_tpr_at(sc, truth[rng0.permutation(len(truth))], fpr) for _ in range(60)], 95))
            if rare.sum() > 3 and 0 < truth[rare].sum() < rare.sum():
                tr, sr = truth[rare], sc[rare]
                res[name]["null_rare"].append(np.percentile(
                    [_tpr_at(sr, tr[rng0.permutation(len(tr))], fpr) for _ in range(60)], 95))
    return {k: {"tpr": float(np.mean(v["all"])),
                "tpr_rare": float(np.mean(v["rare"])) if v["rare"] else float("nan"),
                "null95": float(np.mean(v["null"])),
                "null95_rare": float(np.mean(v["null_rare"])) if v["null_rare"] else float("nan"),
                "excess": max(0.0, float(np.mean(v["all"])) - float(np.mean(v["null"]))),
                "excess_rare": (max(0.0, float(np.mean(v["rare"])) - float(np.mean(v["null_rare"])))
                                if v["rare"] and v["null_rare"] else 0.0)}
            for k, v in res.items()}


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def holdout_audit(build_fn, bench, n_hold: int = 300, seed: int = 0, quick: bool = True) -> dict:
    """記憶だけを測る MIA 監査（シャドウ/ホールドアウト法）。

    B を「加工に使う B_in」と「使わない B_out」に分け、B_in だけから C' を作る。
    候補のメンバーは B_in から、非メンバーは B_out から取る ⸺ 両者は同じコホートの同じ分布
    なので、差が出るとしたら**その行が加工に使われたから**（＝記憶）しかない。
    `build_fn(B_in) -> C'` を渡す。
    """
    rng = np.random.default_rng(seed)
    B = bench.B
    idx = rng.permutation(len(B))
    out_idx, in_idx = idx[:n_hold], idx[n_hold:]
    B_in = B.iloc[in_idx].reset_index(drop=True)
    B_out = B.iloc[out_idx].reset_index(drop=True)
    rare_in = bench.is_rare[in_idx]
    rare_out = bench.is_rare[out_idx]
    Cp = build_fn(B_in, rare_in)
    res = {}
    mia_set = {"domias": AT.m1_domias, "exact": AT.m2_exact_match, "nn_gap": AT.m3_nn_gap,
               "knn": AT.m4_knn_density}
    if not quick:
        mia_set["ens"] = AT.mia_ensemble
    for name, fn in mia_set.items():
        res[name] = mia_audit(Cp, bench, fn, n_attackers=8 if quick else 16,
                              n_perm=400 if quick else 1200, n_members=26,
                              members=B_in, member_rare=rare_in,
                              negatives=B_out, neg_rare=rare_out, seed=seed + 1)
    a = max(v["a_mia"] for v in res.values())
    return {"per_attack": res, "a_mia_holdout": a}


def full_audit(C: pd.DataFrame, bench, label: str = "", quick: bool = False) -> dict:
    """MIA 5 種 × AIA 3 種を当てて、最悪ケースの Anon を見積もる。"""
    mia_set = {"domias": AT.m1_domias, "exact": AT.m2_exact_match, "nn_gap": AT.m3_nn_gap,
               "knn": AT.m4_knn_density, "pref": AT.m5_pref_prior, "ens": AT.mia_ensemble}
    if quick:
        mia_set = {"domias": AT.m1_domias, "exact": AT.m2_exact_match, "nn_gap": AT.m3_nn_gap}
    res_m = {}
    for name, fn in mia_set.items():
        res_m[name] = mia_audit(C, bench, fn, n_attackers=8 if quick else 16,
                                n_perm=400 if quick else 1500)
    aia_set = {"nn": AT.a1_nn_time, "gated": AT.a2_gated}
    if not quick:
        aia_set["suppress"] = AT.a3_gated_suppress
    res_a = {name: aia_audit(C, bench, fn) for name, fn in aia_set.items()}
    a_mia = max(v["a_mia"] for v in res_m.values())
    a_aia = max(v["R"] for v in res_a.values())
    anon = 1.0 - max(a_mia, a_aia)
    if label:
        print(f"--- {label} ---")
        for k, v in res_m.items():
            print(f"   MIA {k:7s} beta_raw={v['beta_raw']:.4f} chance={v['beta_chance']:.4f} "
                  f"-> a_mia={v['a_mia']:.4f}")
        for k, v in res_a.items():
            print(f"   AIA {k:9s} R={v['R']:.4f} (all={v['R_all']:.4f} rare={v['R_rare']:.4f} "
                  f"pm={v['p_member']:.3f} pc={v['p_control']:.3f})")
        print(f"   => a_MIA={a_mia:.4f}  a_AIA={a_aia:.4f}  Anon≈{anon:.4f}")
    return {"mia": res_m, "aia": res_a, "a_mia": a_mia, "a_aia": a_aia, "anon": anon}
