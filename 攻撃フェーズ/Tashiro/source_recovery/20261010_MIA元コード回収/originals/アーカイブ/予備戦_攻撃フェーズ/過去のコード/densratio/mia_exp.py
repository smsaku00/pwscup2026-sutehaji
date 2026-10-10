"""密度比 MIA: C_5 に対して密度関数を差し替えて実験する。

Panel A : 陽性 = B_5 の 1049 行 / 陰性 = A_bg から (sex,県) を合わせて引いた非メンバー
          参照 = A_bg の別半分（陰性と素）。特徴 = 連続7列（公式参照実装と同じ空間）。
Panel B : 陽性 = aia_challenge_5 のメンバー163 / 陰性 = 同 challenge の対照163
          （事務局が同じ地元県から用意した対照 ＝ 交絡なしの正しい帰無）。特徴 = 連続5列。
Panel A-naive : 参照に A_bg 全体を渡す誤った監査（既知の偽陽性を再現して比較する）。
"""
from __future__ import annotations
import sys, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common, densities
from densities import TwoDensityRatio
from scipy.stats import mannwhitneyu
from pwscup2026.scoring.attack_score import _tpr_at_fpr

SCHEMA = common.BASE + 'PWSCup2026_prelim_common/schema.json'


def build_targets(abg, B, C5, exclude_ids):
    pool = abg[~abg.record_id.isin(exclude_ids)]
    T = [('C_5 (提出=CopulaSA_Utility)', C5)]
    for s in range(3):
        r = np.random.default_rng(common.SEED + 100 + s)
        T.append((f'[帰無] A_bg実在行 #{s}',
                  pool.iloc[r.choice(len(pool), 1049, replace=False)].reset_index(drop=True)))
    sd = B[common.CONT7].std(ddof=0).to_numpy(float)
    for sig in (0.1, 0.2, 0.5, 1.0):
        r = np.random.default_rng(common.SEED + 200)
        X = B.copy()
        X[common.CONT7] = B[common.CONT7].to_numpy(float) + r.normal(0, sig * sd, (len(B), 7))
        T.append((f'[漏洩] B+{sig:g}σ雑音', X))
    for lab, p in [('Zhou Balanced',   'Zhouさん/T05_CopulaSA_Hybrid_Balanced_NewToken'),
                   ('Zhou Protected',  'Zhouさん/T05_CopulaSA_Hybrid_Protected_NewToken'),
                   ('Seo SA Hardened', 'Seoさん/PWS_Cup2026_T05_SA_MIA_AIA_Hardened'),
                   ('Seo SA Refined',  'Seoさん/PWS_Cup2026_T05_SA_PrivacyRefined'),
                   ('Tashiro Copula',  '加工データ候補0812_Tashiro/T05_Copula_Tashiro'),
                   ('Tashiro Pool',    '加工データ候補0812_Tashiro/T05_Pool_Tashiro'),
                   ('Tashiro SA',      '加工データ候補0812_Tashiro/T05_SA_Tashiro')]:
        T.append((f'(参考) {lab}', pd.read_csv(common.SUB + p + '/C.csv')))
    return T


def evaluate(scores, y, rare=None):
    tpr1, auc = _tpr_at_fpr(scores, y, 0.01)
    tpr10, _ = _tpr_at_fpr(scores, y, 0.10)
    p = mannwhitneyu(scores[y == 1], scores[y == 0], alternative='greater').pvalue
    out = dict(auc=auc, tpr1=tpr1, tpr10=tpr10, p=p, tpr1_rare=np.nan, score_mia=tpr1)
    if rare is not None and rare.any() and y[rare].sum() > 0:
        out['tpr1_rare'] = _tpr_at_fpr(scores[rare], y[rare], 0.01)[0]
        out['score_mia'] = max(tpr1, out['tpr1_rare'])
    return out


def run_panel(panel, cols, abg, ref_df, pos_df, neg_df, T, zoo, rare=None):
    z = common.zscaler(abg, cols)
    REF, EVAL = z(ref_df), np.vstack([z(pos_df), z(neg_df)])
    y = np.r_[np.ones(len(pos_df), int), np.zeros(len(neg_df), int)]
    logref = {}                                   # 参照密度は対象によらないので使い回す
    rows = []
    for label, C in T:
        SYN = z(C)
        for est_name, mk in zoo:
            t0 = time.time()
            try:
                r = mk()
                if isinstance(r, TwoDensityRatio):
                    if est_name not in logref:
                        logref[est_name] = r.maker().fit(REF).logpdf(EVAL)
                    s = np.clip(r.maker().fit(SYN).logpdf(EVAL) - logref[est_name], -200, 200)
                else:
                    s = r.fit(SYN, REF).log_ratio(EVAL)
                m = evaluate(s, y, rare)
            except Exception as e:
                print('  ERR', est_name, repr(e))
                m = dict(auc=np.nan, tpr1=np.nan, tpr10=np.nan, p=np.nan,
                         tpr1_rare=np.nan, score_mia=np.nan)
            m.update(panel=panel, target=label, est=est_name, sec=round(time.time() - t0, 1))
            rows.append(m)
            print(f'  {panel:8s}|{label:30s}|{est_name:20s} AUC={m["auc"]:.4f} '
                  f'TPR@1%={m["tpr1"]:.4f} rare={m["tpr1_rare"]:.3f} p={m["p"]:.1e} ({m["sec"]}s)')
    return rows


if __name__ == '__main__':
    d = common.load(); abg, B = d['abg'], d['B']
    rng = np.random.default_rng(common.SEED)
    A_ref, A_pool = common.split_abg(abg, rng)
    neg, _ = common.matched_negatives(A_pool, B, 1049, rng)
    T = build_targets(abg, B, d['C5'], set(neg.record_id))
    zoo = densities.ratio_zoo()
    rows = []

    M = np.r_[common.rare_M(B, SCHEMA), common.rare_M(neg, SCHEMA)]
    print(f'=== Panel A: 陽性 B_5(1049) / 陰性 A_bg整合({len(neg)}) / 参照 A_bg別半分({len(A_ref)}) / 連続7列'
          f' / 希少(M>=4) {int((M>=4).sum())}行 ===')
    rows += run_panel('A', common.CONT7, abg, A_ref, B, neg, T, zoo, rare=(M >= 4.0))

    print(f'=== Panel B: 陽性 challenge_5メンバー163 / 陰性 challenge_5対照163 / 参照 A_bg全体 / 連続5列 ===')
    ch = d['ch5']
    rows += run_panel('B', common.CONT5, abg, abg,
                      ch.iloc[d['mem_ch']].reset_index(drop=True),
                      ch.iloc[d['ctl_ch']].reset_index(drop=True), T, zoo)

    print(f'=== Panel A-naive: 参照に A_bg 全体（対照を含む）を渡した誤った監査 ===')
    rows += run_panel('A-naive', common.CONT7, abg, abg, B, neg,
                      [t for t in T if t[0].startswith(('C_5', '[帰無] A_bg実在行 #0'))], zoo)

    pd.DataFrame(rows).to_csv('mia_results.csv', index=False)
    print('\nwrote mia_results.csv', len(rows), 'rows')
