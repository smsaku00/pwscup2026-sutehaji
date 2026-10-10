# -*- coding: utf-8 -*-
"""攻撃提出 F_mia.csv / F_aia.csv を組み立てる。

使い方: python build_attack.py --round 1 --out prelim_attack/r1
  --round s で希少軸の指名をずらす（ラテン方格: 標的 i に候補 (i+s) mod 10 を1位指名）
"""
import argparse, json, sys, shutil
sys.path.insert(0, "/Users/taisei1105/pws2026-local/work")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from submit_audit import COLS13, CONT, _z
from harness import Bench

DIST = Path("/Users/taisei1105/pws2026-local/予備戦/配布データ/PWSCup2026_prelim_attack")
TOKEN = Path("/Users/taisei1105/pws2026-local/work/data_prelim_team05/token.txt")
QI = ["age", "SBP", "TG", "HDL", "ALT"]
SELF = 5
CENS = 9.499          # 定数回答。窓 [8.999, 9.999] が打ち切り点質量と非打ち切りを両取りする
TOKYO = "東京都"

def load():
    b = Bench(Path("data_prelim_team05"))
    A = b.A_bg.reset_index(drop=True)
    Bd = b.B.reset_index(drop=True)                    # record_id 付き
    targets = sorted(int(p.stem.split("_")[1]) for p in DIST.glob("C_*.csv"))
    targets = [k for k in targets if k != SELF]        # ★自チーム列は必ず外す
    return b, A, Bd, targets

def nn_dist(C, X, cols, mean, sd, want_idx=False):
    nn = NearestNeighbors(n_neighbors=1).fit(_z(C, cols, mean, sd))
    d, j = nn.kneighbors(_z(X, cols, mean, sd))
    return (d[:, 0], j[:, 0]) if want_idx else d[:, 0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", default="prelim_attack/r1")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    b, A, Bd, targets = load()
    Bc = Bd[COLS13]
    rare = b.is_rare
    rid = Bd["record_id"].to_numpy(int)
    tokyo = (Bd["prefecture"] == TOKYO).to_numpy()
    mean = A[CONT].mean().to_numpy(float)
    sd = A[CONT].std(ddof=0).to_numpy(float); sd = np.where(sd > 0, sd, 1.0)
    qm = A[QI].mean().to_numpy(float)
    qs = A[QI].std(ddof=0).to_numpy(float); qs = np.where(qs > 0, qs, 1.0)

    # time 指紋（自Bにあり A_bg に無い非打ち切り値で、B 内で1人しか持たないもの）
    at = set(np.round(A["time"].to_numpy(float), 9))
    vc = pd.Series(np.round(Bd["time"].to_numpy(float), 9)).value_counts()
    fp = {v for v in set(np.round(Bd["time"].to_numpy(float), 9)) - at if vc.get(v, 0) == 1}

    # 全ターゲットへの最近傍距離行列（残差スコア用）
    Dmat = np.zeros((len(Bc), len(targets)))
    conf_sets, aia_rows = {}, []
    for ti, k in enumerate(targets):
        C = pd.read_csv(DIST / f"C_{k}.csv")
        d = nn_dist(C, Bc, CONT, mean, sd)
        Dmat[:, ti] = d
        ts = set(np.flatnonzero(np.isin(np.round(Bd["time"].to_numpy(float), 9),
                                        list(fp & set(np.round(C["time"].to_numpy(float), 9))))).tolist())
        tight = set(np.flatnonzero(d < 0.05).tolist())
        loose = set(np.flatnonzero(d < 0.30).tolist())
        ds = tight if (k == 1 or len(loose | ts) < 8) else loose
        conf_sets[k] = ds | ts

        # ---- AIA
        Q = pd.read_csv(DIST / f"aia_challenge_{k}.csv")
        th = np.full(len(Q), CENS)
        if k in (1, 2):
            dq = np.full(len(Q), np.inf); jq = np.full(len(Q), -1)
            if k == 1:   # 標準化ユークリッド（0.02〜0.10 で 120 行の塊）
                for key, ii in C.groupby(["sex", "prefecture"]).groups.items():
                    m = ((Q["sex"] == key[0]) & (Q["prefecture"] == key[1])).to_numpy()
                    ii = np.asarray(ii)
                    if m.sum() == 0 or len(ii) == 0: continue
                    dd, jj = NearestNeighbors(n_neighbors=1).fit(
                        (C.loc[ii, QI].to_numpy(float) - qm) / qs).kneighbors(
                        (Q.loc[m, QI].to_numpy(float) - qm) / qs)
                    dq[m] = dd[:, 0]; jq[m] = ii[jj[:, 0]]
                flag = dq < 0.10
            else:        # 対数比RMS（乗法摂動に強い。純度は未検証＝#1で確かめる）
                cv = np.log(np.clip(C[QI].to_numpy(float), 1e-9, None))
                qv = np.log(np.clip(Q[QI].to_numpy(float), 1e-9, None))
                for key, ii in C.groupby(["sex", "prefecture"]).groups.items():
                    m = ((Q["sex"] == key[0]) & (Q["prefecture"] == key[1])).to_numpy()
                    ii = np.asarray(ii)
                    if m.sum() == 0 or len(ii) == 0: continue
                    for r in np.flatnonzero(m):
                        e = np.sqrt(np.mean((cv[ii] - qv[r]) ** 2, axis=1))
                        j = int(np.argmin(e)); dq[r] = e[j]; jq[r] = ii[j]
                flag = dq < 0.0404
            th[flag] = C["time"].to_numpy(float)[jq[flag].astype(int)]
        aia_rows.append(pd.DataFrame({"target_k": k,
                                      "challenge_row_id": Q["challenge_row_id"].to_numpy(int),
                                      "time_hat": th}))

    # 残差スコア: 行効果（その行がどれだけ外れ値か）を列間ダブルセンタリングで落とす
    L = -np.log(np.clip(Dmat, 1e-12, None))
    resid = L - L.mean(axis=1, keepdims=True) - L.mean(axis=0, keepdims=True) + L.mean()

    # 希少の指名候補（非東京都の希少のみ。東京都の希少は非共有ブロック側で確定）
    rare_nt = np.flatnonzero(rare & ~tokyo)
    rare_all = np.flatnonzero(rare)
    print(f"希少 {len(rare_all)}人 / うち非東京都 {len(rare_nt)}人: {rid[rare_nt].tolist()}")

    # ---- F_mia
    F = pd.DataFrame({"record_id": rid})
    for ti, k in enumerate(targets):
        c = np.zeros(len(Bc))
        # 帯4: 東京都 0.00-0.10 / 帯3: 非東京都 0.10-0.50（残差順）
        r = resid[:, ti]
        rk = (pd.Series(r).rank(pct=True).to_numpy())
        c[tokyo] = 0.02 + 0.08 * rk[tokyo]
        c[~tokyo] = 0.12 + 0.38 * rk[~tokyo]
        # 帯2: 非東京都の上位（1%FPR予算ぶん＝約10行）を 0.70-0.80 に押し上げる
        cand = np.flatnonzero(~tokyo & ~rare)
        top = cand[np.argsort(-r[cand])[:12]]
        c[top] = 0.70 + 0.10 * np.linspace(1, 0, len(top))
        # 帯2.5: 希少24人は予算より下の狭い帯 0.52-0.58。非東京都>東京都、指名1人が帯の先頭
        pick = rare_nt[(ti + a.round) % len(rare_nt)]
        order = [pick] + [x for x in rare_nt if x != pick] + [x for x in rare_all if tokyo[x]]
        for pos, ix in enumerate(order):
            c[ix] = 0.58 - 0.06 * pos / max(len(order) - 1, 1)
        # 帯1: 確定メンバーは 1.00（真陽性なのでFPR予算を食わない）
        for ix in conf_sets[k]:
            c[ix] = 1.0
        F[str(k)] = np.clip(c, 0.0, 1.0)

    F.to_csv(out / "F_mia.csv", index=False)
    pd.concat(aia_rows, ignore_index=True).to_csv(out / "F_aia.csv", index=False)
    shutil.copy(TOKEN, out / "token.txt")
    n_conf = {k: len(v) for k, v in conf_sets.items() if v}
    print(f"F_mia: {F.shape[0]}行 × {F.shape[1]}列（record_id + {len(targets)}標的）")
    print(f"F_aia: {sum(len(x) for x in aia_rows)}行")
    print(f"確定メンバーを置いた標的: {n_conf}")
    print(f"希少の指名（ラウンド{a.round}）: " +
          ", ".join(f"{k}→{rid[rare_nt[(i + a.round) % len(rare_nt)]]}" for i, k in enumerate(targets[:6])) + " ...")
    print(f"出力: {out}")

if __name__ == "__main__":
    main()
