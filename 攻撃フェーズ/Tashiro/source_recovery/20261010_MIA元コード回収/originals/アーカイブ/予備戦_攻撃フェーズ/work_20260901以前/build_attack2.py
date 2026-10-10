# -*- coding: utf-8 -*-
"""攻撃提出 F_mia.csv / F_aia.csv を組み立てる（2026-09-01 版）。

旧版からの変更:
  - 撤回した time 指紋を全廃（日グリッドなので個人指紋にならない）
  - 確定メンバーを「離散キー一致＋検査値6列の距離」で A_bg 対照つきに測り直す
  - aia_challenge_自分 から非共有ブロック P_i の 163 人を特定し、全列で 0（非メンバー確定）に
  - 事前確率を 3 段に（非地元県 100% > 残りの地元県 37.2% > 特定した P_i 0%）
  - AIA を定数一本から「challenge 行を C_k に照合して time を読む」に変更
  - 希少の指名を状態ファイルで持ち回り、帯 4 が出た組を据え置く逐次探索に

使い方:
    python build_attack2.py --round 1 --out prelim_attack/r1
    # 採点結果を見て、帯 4 になった標的を記録してから次ラウンド
    python build_attack2.py --round 2 --out prelim_attack/r2 --hit 7=201 --hit 13=738
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/Users/taisei1105/pws2026-local")
DIST = ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack"
DATA = ROOT / "share_team05/data"
TOKEN = DATA / "token.txt"
STATE = ROOT / "work/attack_state.json"
SELF, HOME, CENS = 5, "東京都", 9.4990
LAB = ["BMI", "SBP", "TG", "HDL", "ALT", "FPG"]      # 確定メンバー照合に使う（Cの全列）
QI5 = ["age", "SBP", "TG", "HDL", "ALT"]             # challenge に入っている連続列
QI4 = ["SBP", "TG", "HDL", "ALT"]
KEY = ["sex", "prefecture", "smoking", "onset", "death"]


def zmat(df, cols, mu, sd):
    return (df[cols].to_numpy(float) - mu) / sd


def key_index(df, keys):
    idx = {}
    for j, (t, a) in enumerate(zip(map(tuple, df[keys].to_numpy()), df["age"].to_numpy())):
        idx.setdefault(t, []).append((a, j))
    return idx


def nn_by_key(qz, qkeys, qage, cz, cidx, tol=1):
    """離散キーが完全一致し年齢が ±tol の候補だけを見て、最小距離とその行番号を返す。"""
    d = np.full(len(qz), np.inf)
    j = np.full(len(qz), -1)
    for i, (t, a) in enumerate(zip(qkeys, qage)):
        js = [x for aa, x in cidx.get(t, []) if abs(aa - a) <= tol]
        if not js:
            continue
        dd = np.sqrt(((cz[js] - qz[i]) ** 2).sum(1))
        m = int(dd.argmin())
        d[i], j[i] = dd[m], js[m]
    return d, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--out", default="prelim_attack/r1")
    ap.add_argument("--hit", action="append", default=[],
                    help="前ラウンドで帯4になった組 例 --hit 7=201")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    Bs = pd.read_csv(DATA / "B_self.csv")
    B = Bs.drop(columns=["is_rare"])
    A = pd.read_csv(DATA / "A_bg.csv")
    rid = Bs["record_id"].to_numpy(int)
    rare = Bs["is_rare"].to_numpy(bool)
    home = (Bs["prefecture"] == HOME).to_numpy()
    targets = sorted(int(k) for k in json.loads((DIST / "mia_columns.json").read_text())["target_columns"]
                     if int(k) != SELF)

    # ---- 非共有ブロック P_i を aia_challenge_自分 から特定（§4.2 の 163 人） ----
    ch_self = pd.read_csv(DIST / f"aia_challenge_{SELF}.csv")
    km = {tuple(t): i for i, t in enumerate(np.round(Bs[QI5].to_numpy(float), 9))}
    p_self = np.zeros(len(Bs), bool)
    for t in np.round(ch_self[QI5].to_numpy(float), 9):
        if tuple(t) in km:
            p_self[km[tuple(t)]] = True
    assert p_self.sum() == 163, f"P_i の特定に失敗: {p_self.sum()} 行"
    assert home[p_self].all(), "特定した P_i に地元県以外が混ざっている"

    mu6, sd6 = A[LAB].mean().to_numpy(), A[LAB].std(ddof=0).to_numpy()
    mu4, sd4 = A[QI4].mean().to_numpy(), A[QI4].std(ddof=0).to_numpy()
    bz6 = zmat(B, LAB, mu6, sd6)
    bkeys = list(map(tuple, B[KEY].to_numpy())); bage = B["age"].to_numpy()
    az6 = zmat(A, LAB, mu6, sd6)
    akeys = list(map(tuple, A[KEY].to_numpy())); aage = A["age"].to_numpy()
    aidx = key_index(A, KEY)

    # 対照: 自分の各行と同じ離散キー・年齢±1 の A_bg 行（A_bg∩B=∅ なので非メンバー確定）
    ctrl = []
    for s in range(6):
        rng = np.random.default_rng(s); pick = []
        for t, ag in zip(bkeys, bage):
            js = [x for aa, x in aidx.get(t, []) if abs(aa - ag) <= 1]
            pick.append(rng.choice(js) if js else -1)
        ctrl.append(np.array(pick))

    conf_sets, aia_rows, dmat, report = {}, [], np.zeros((len(B), len(targets))), {}
    for ti, k in enumerate(targets):
        C = pd.read_csv(DIST / f"C_{k}.csv")
        cz6 = zmat(C, LAB, mu6, sd6); cidx = key_index(C, KEY)
        d, _ = nn_by_key(bz6, bkeys, bage, cz6, cidx)
        dmat[:, ti] = np.where(np.isfinite(d), d, 9.9)

        # しきい値は「対照が 1 行未満」を保つ最大のものを選ぶ
        dc = np.array([nn_by_key(az6[p[p >= 0]], [akeys[j] for j in p[p >= 0]],
                                 aage[p[p >= 0]], cz6, cidx)[0] for p in ctrl])
        th = 0.0
        for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
            if (dc < t).sum(1).mean() < 1.0:
                th = t
        members = np.flatnonzero((d < th) & ~p_self) if th else np.array([], int)
        if len(members) < 3:            # 1〜2 行の一致は偽陽性と見分けがつかない
            members = np.array([], int)
        conf_sets[k] = set(members.tolist())
        report[k] = {"threshold": th, "n": len(members), "rid": rid[members].tolist()}

        # ---- AIA: challenge 行を C_k に照合して time を読む ----
        Q = pd.read_csv(DIST / f"aia_challenge_{k}.csv")
        qz4 = zmat(Q, QI4, mu4, sd4); cz4 = zmat(C, QI4, mu4, sd4)
        qk = list(zip(Q["sex"], Q["prefecture"])); qa = Q["age"].to_numpy()
        ci2 = key_index(C, ["sex", "prefecture"])
        dq, jq = nn_by_key(qz4, qk, qa, cz4, ci2)
        # 帰無: challenge と同じ (sex, prefecture, age) 分布で A_bg から引いた行
        ai2 = key_index(A, ["sex", "prefecture"])
        dns = []
        for sd_ in range(8):
            rng = np.random.default_rng(sd_); pn = []
            for (s, p), ag in zip(qk, qa):
                js = [x for aa, x in ai2.get((s, p), []) if abs(aa - ag) <= 1]
                if js: pn.append(rng.choice(js))
            dns.append(nn_by_key(zmat(A.iloc[pn], QI4, mu4, sd4),
                                 [(A.sex[j], A.prefecture[j]) for j in pn], aage[pn], cz4, ci2)[0])
        bestm, bestt = 0.0, None
        # しきい値 0.40 は帰無モデルが壊れる領域（合成チームでも z=3〜5 が出る）ので使わない
        for t in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
            nv = np.array([(dn < t).sum() * len(Q) / len(dn) for dn in dns])
            m = ((dq < t).sum() - nv.mean()) / 163.0
            if m > bestm: bestm, bestt = m, t
        th_hat = np.full(len(Q), CENS)
        if bestm >= 0.35 and bestt is not None:   # 1・2 は 0.71〜0.76、次点は 0.13 なので余裕がある
            f = (dq < bestt) & (jq >= 0)
            th_hat[f] = C["time"].to_numpy(float)[jq[f]]
        else:
            bestt = None
        report[k]["aia_m"] = round(float(min(bestm, 1.0)), 3)
        report[k]["aia_th"] = bestt
        aia_rows.append(pd.DataFrame({"target_k": k,
                                      "challenge_row_id": Q["challenge_row_id"].to_numpy(int),
                                      "time_hat": th_hat}))

    # ---- 希少の指名（状態ファイルで持ち回る逐次探索） ----
    st = json.loads(STATE.read_text()) if STATE.exists() else {"found": {}, "ruled": {}}
    for h in a.hit:
        t, p = h.split("="); st["found"][t] = int(p)
    cand = [int(r) for r in rid[rare & ~p_self & ~home]]        # 非地元県の希少＝共有率100%
    tail = [int(r) for r in rid[rare & ~p_self & home]]         # 残りの地元県の希少（共有率37%）
    pool = cand + tail
    located = set(st["found"].values())
    nominee = {}
    for i, k in enumerate(targets):
        ks = str(k)
        if ks in st["found"]:
            nominee[k] = st["found"][ks]; continue
        av = [p for p in pool if p not in st["ruled"].get(ks, []) and p not in located] \
             or [p for p in pool if p not in st["ruled"].get(ks, [])] or pool
        nominee[k] = av[(i + a.round) % len(av)]
    for k in targets:
        ks = str(k)
        if ks not in st["found"]:
            st["ruled"].setdefault(ks, [])
            if nominee[k] not in st["ruled"][ks]: st["ruled"][ks].append(nominee[k])
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))

    # ---- F_mia ----
    L = -np.log(np.clip(dmat, 1e-12, None))
    resid = L - L.mean(1, keepdims=True) - L.mean(0, keepdims=True) + L.mean()
    rare_idx = np.flatnonzero(rare)
    F = pd.DataFrame({"record_id": rid})
    for ti, k in enumerate(targets):
        c = np.zeros(len(B))
        r = resid[:, ti]; rk = pd.Series(r).rank(pct=True).to_numpy()
        c[home] = 0.02 + 0.08 * rk[home]                       # 帯4: 残りの地元県（共有 37%）
        c[~home] = 0.12 + 0.38 * rk[~home]                     # 帯3: 非地元県（共有 100%）
        # 帯2.5: 希少 24 人を 0.52-0.58 に固定配置。指名 1 人が先頭、確定 P_i の 13 人は最後尾。
        nom = int(np.flatnonzero(rid == nominee[k])[0])
        order = [nom] + [x for x in rare_idx if x != nom and not p_self[x]] \
                      + [x for x in rare_idx if x != nom and p_self[x]]
        for pos, ix in enumerate(order):
            c[ix] = 0.58 - 0.06 * pos / max(len(order) - 1, 1)
        # 帯2: FPR 予算ぶん（約10行）の勝負手。非地元県・非希少・非 P_i の残差上位 12。
        pool_b2 = np.flatnonzero(~home & ~rare & ~p_self)
        top = pool_b2[np.argsort(-r[pool_b2])[:12]]
        c[top] = 0.70 + 0.10 * np.linspace(1, 0, len(top))
        c[p_self] = 0.0                                        # 非メンバー確定
        for ix in conf_sets[k]:
            c[ix] = 1.0                                        # 帯1: 確定メンバー
        F[str(k)] = np.clip(c, 0.0, 1.0)

    F.to_csv(out / "F_mia.csv", index=False)
    pd.concat(aia_rows, ignore_index=True).to_csv(out / "F_aia.csv", index=False)
    shutil.copy(TOKEN, out / "token.txt")
    (out / "report.json").write_text(json.dumps(
        {"round": a.round, "nominee": {str(k): v for k, v in nominee.items()}, "per_target": report},
        ensure_ascii=False, indent=1))
    print(f"F_mia {F.shape[0]}行×{F.shape[1]}列 / F_aia {sum(len(x) for x in aia_rows)}行 → {out}")
    print(f"確定メンバー: " + ", ".join(f"{k}:{v['n']}" for k, v in report.items() if v["n"]))
    print(f"AIA 上書き:  " + ", ".join(f"{k}:m={v['aia_m']}" for k, v in report.items() if v["aia_th"]))
    print(f"P_i 確定の非メンバー {int(p_self.sum())} 行を全列 0 に")
    print(f"希少の指名（ラウンド{a.round}）: " + ", ".join(f"{k}→{nominee[k]}" for k in targets[:8]) + " …")


if __name__ == "__main__":
    main()
