"""密度比 AIA: challenge_5 の秘匿属性 time を、密度関数を差し替えて復元する。

会員 = challenge_5 の 163 メンバー（B_self から真の time が既知）
対照 = A_bg から sex 一致・連続QI最近傍・検閲傾向スコア一致で 163 人
       （非メンバー確定・真の time 既知。定数攻撃の R が 0 になるよう素性を揃えた）
予測 = 結合空間 (QI5, time) で
   λ=0  argmax_t  log p_C(x,t)                      条件付き密度 AIA
   λ=1  argmax_t [log p_C(x,t) − log p_ref(x,t)]     密度比 AIA
   gate λ=1 の確信度上位 q だけ答え、残りは定数 9.4990（照合できた行だけ答える運用形）
採点 = 公式と同じ R = clip((p_m − p_c)/(1 − p_c), 0, 1), 成功 = |t̂ − t| ≤ 0.5
"""
from __future__ import annotations
import json, warnings, time, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common, densities
from densities import TwoDensityRatio
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from pwscup2026.scoring.attack_score import R_from_rate, R_from_rate_deadzone

SCHEMA = common.BASE + 'PWSCup2026_prelim_common/schema.json'
TAU, CONST, CENS = 0.5, 9.4990, 9.4986


def pick_controls(abg, mem_qi, z):
    """sex 一致・QI 最近傍・検閲傾向スコア一致で非メンバー対照を選ぶ（重複なし）。"""
    Xa = np.column_stack([z(abg), abg.sex.values])
    lr = LogisticRegression(max_iter=2000).fit(Xa, (abg.time >= CENS).astype(int))
    pa = lr.predict_proba(Xa)[:, 1]
    pm = lr.predict_proba(np.column_stack([z(mem_qi), mem_qi.sex.values]))[:, 1]
    out, taken = [], set()
    for s in (0, 1):
        mi = np.flatnonzero(mem_qi.sex.values == s); ci = np.flatnonzero(abg.sex.values == s)
        tree = cKDTree(np.column_stack([z(abg)[ci], 3.0 * pa[ci]]))
        Tq = np.column_stack([z(mem_qi)[mi], 3.0 * pm[mi]])
        for r in range(len(mi)):
            for j in np.atleast_1d(tree.query(Tq[r], k=40)[1]):
                g = ci[j]
                if g not in taken:
                    taken.add(g); out.append(g); break
    return abg.iloc[sorted(out)].reset_index(drop=True)


def joint_scores(mk, C6, REF6, Y, key, cache):
    """(query×time候補) 並びの評価点 Y に対し log p_C と log p_ref を返す。"""
    r = mk()
    if isinstance(r, TwoDensityRatio):
        if key not in cache:
            cache[key] = r.maker().fit(REF6).logpdf(Y)
        return r.maker().fit(C6).logpdf(Y), cache[key], True
    return r.fit(C6, REF6).log_ratio(Y), None, False


def nn1_time(C, Q, z):
    """公式参照実装と同じ 1-NN AIA（sex ブロック内 per-feature-RMS 最近傍の time）。"""
    out = np.empty(len(Q))
    for s in (0, 1):
        mq = np.flatnonzero(Q.sex.values == s); mc = np.flatnonzero(C.sex.values == s)
        if len(mc) == 0:
            mc = np.arange(len(C))
        d, j = cKDTree(z(C)[mc]).query(z(Q)[mq], k=1)
        out[mq] = C.time.values[mc[np.atleast_1d(j)]]
    return out


if __name__ == '__main__':
    d = common.load(); abg, B, ch = d['abg'], d['B'], d['ch5']
    z5 = common.zscaler(abg, common.CONT5)
    mem_qi = ch.iloc[d['mem_ch']].reset_index(drop=True)
    mem_B = B.iloc[d['mem_B']].reset_index(drop=True)
    ctl = pick_controls(abg, mem_qi, z5)
    t_m, t_c = mem_B.time.to_numpy(float), ctl.time.to_numpy(float)
    rare_m = mem_B.is_rare.to_numpy(bool)
    rare_c = common.rare_M(ctl, SCHEMA) >= 4.0
    Q = pd.concat([mem_qi[common.CONT5 + ['sex']], ctl[common.CONT5 + ['sex']]], ignore_index=True)
    print(f'会員 163（希少 {rare_m.sum()}） / 対照 {len(ctl)}（希少検出器 {rare_c.sum()}）')

    def rep(name, th, extra=''):
        sm = (np.abs(th[:163] - t_m) <= TAU).astype(float)
        sc = (np.abs(th[163:] - t_c) <= TAU).astype(float)
        R = R_from_rate(sm, sc)
        Rr = R_from_rate_deadzone(sm[rare_m], sc[rare_c], 1.64)
        return dict(R_all=R, R_rare=Rr, score_aia=float(np.nanmax([R, Rr])),
                    p_m=sm.mean(), p_c=sc.mean(),
                    pm_rare=sm[rare_m].mean(), pc_rare=sc[rare_c].mean())

    base = rep('const', np.full(326, CONST))
    print(f'[基準] 定数{CONST}      R={base["R_all"]:.4f} p_m={base["p_m"]:.4f} p_c={base["p_c"]:.4f}')

    tmu, tsd = abg.time.mean(), abg.time.std(ddof=0)
    zt = lambda v: (np.asarray(v, float) - tmu) / tsd
    REF6 = np.column_stack([z5(abg), zt(abg.time.values)])

    T = [('C_5 (提出=CopulaSA_Utility)', d['C5'])]
    for lab, p in [('Zhou Balanced', 'Zhouさん/T05_CopulaSA_Hybrid_Balanced_NewToken'),
                   ('Zhou Protected', 'Zhouさん/T05_CopulaSA_Hybrid_Protected_NewToken'),
                   ('Seo SA Hardened', 'Seoさん/PWS_Cup2026_T05_SA_MIA_AIA_Hardened'),
                   ('Seo SA Refined', 'Seoさん/PWS_Cup2026_T05_SA_PrivacyRefined'),
                   ('Tashiro Copula', '加工データ候補0812_Tashiro/T05_Copula_Tashiro'),
                   ('Tashiro Pool', '加工データ候補0812_Tashiro/T05_Pool_Tashiro'),
                   ('Tashiro SA', '加工データ候補0812_Tashiro/T05_SA_Tashiro')]:
        T.append((f'自陣 {lab}', pd.read_csv(common.SUB + p + '/C.csv')))
    for k in [int(k) for k in json.load(open(common.ATK + 'mia_columns.json'))['target_columns']]:
        if k != 5:
            T.append((f'[帰無] C_{k}', pd.read_csv(common.ATK + f'C_{k}.csv')))
    sd = B[common.CONT7].std(ddof=0).to_numpy(float)
    for sig in (0.1, 0.3):
        r = np.random.default_rng(common.SEED + 400); X = B.copy()
        X[common.CONT7] = B[common.CONT7].to_numpy(float) + r.normal(0, sig * sd, (len(B), 7))
        T.append((f'[漏洩] B+{sig:g}σ雑音', X))

    # time 候補は全対象で共通の固定グリッド（参照密度の評価を使い回すため）
    cand = np.unique(np.round(np.r_[np.quantile(abg.time.values, np.linspace(0, 1, 170)),
                                    np.linspace(0.0, 10.0, 50), 9.998631, CONST], 9))
    nt, nq = len(cand), len(Q)
    Y = np.column_stack([np.repeat(z5(Q), nt, axis=0), np.tile(zt(cand), nq)])
    print(f'time 候補グリッド {nt} 点（{cand.min():.3f}〜{cand.max():.3f}）')

    zoo = densities.ratio_zoo()
    cache, rows = {}, [dict(target='—', est='[基準] 定数9.4990', mode='const', **base)]
    for label, C in T:
        C6 = np.column_stack([z5(C), zt(C.time.values)])
        r0 = rep('nn1', nn1_time(C, Q, z5))
        rows.append(dict(target=label, est='[基準] 公式1-NN', mode='nn1', **r0))
        print(f'{label:26s} [基準] 公式1-NN        R={r0["R_all"]:.4f} p_m={r0["p_m"]:.3f} p_c={r0["p_c"]:.3f}', flush=True)
        for est_name, mk in zoo:
            t0 = time.time()
            lc, lr, has_cond = joint_scores(mk, C6, REF6, Y, est_name, cache)
            outs = {}
            if has_cond:
                outs['λ=0 条件付き'] = lc.reshape(nq, nt)
                G = lr.reshape(nq, nt)
                S = np.where(G >= G.max(1, keepdims=True) - 10.0, (lc - lr).reshape(nq, nt), -np.inf)
                outs['λ=1 密度比'] = S
            else:
                outs['λ=1 密度比'] = lc.reshape(nq, nt)
            for mode, S in outs.items():
                th = cand[np.argmax(S, axis=1)]
                m = rep(mode, th)
                rows.append(dict(target=label, est=est_name, mode=mode, **m))
                if mode.startswith('λ=1'):                       # 確信度上位だけ答える運用形
                    conf = S.max(1)
                    for q in (0.10, 0.30):
                        g = th.copy(); g[conf < np.quantile(conf, 1 - q)] = CONST
                        rows.append(dict(target=label, est=est_name, mode=f'λ=1 gate{int(q*100)}%',
                                         **rep(mode, g)))
            print(f'{label:26s} {est_name:20s} ' +
                  ' '.join(f'{k}:R={v["R_all"]:.3f}' for k, v in
                           ((r['mode'], r) for r in rows[-(1 + 2 * len(outs)):] if r['target'] == label
                            and r['est'] == est_name)) + f' ({time.time()-t0:.1f}s)', flush=True)
    pd.DataFrame(rows).to_csv('aia_results.csv', index=False)
    print('\nwrote aia_results.csv', len(rows), 'rows')
