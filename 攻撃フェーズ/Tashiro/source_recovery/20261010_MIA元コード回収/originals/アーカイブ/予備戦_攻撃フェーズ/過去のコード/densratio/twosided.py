"""符号を校正できる攻撃者を想定した両側評価。

score と −score の両方で TPR@1%FPR / AUC を取り、大きい方を採る。帰無帯も同じ両側統計で作る
（両側にすると帰無側も底上げされるので、その分を含めて比較しないと過大評価になる）。
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common, densities
from densities import TwoDensityRatio
from pwscup2026.scoring.attack_score import _tpr_at_fpr

d = common.load(); abg, ch, B = d['abg'], d['ch5'], d['B']
KS = [int(k) for k in json.load(open(common.ATK + 'mia_columns.json'))['target_columns']]
z5 = common.zscaler(abg, common.CONT5)
REF, EVAL = z5(abg), np.vstack([z5(ch.iloc[d['mem_ch']]), z5(ch.iloc[d['ctl_ch']])])
y = np.r_[np.ones(163, int), np.zeros(163, int)]
T = [('C_5 提出', d['C5'])]
for lab, p in [('Zhou Balanced', 'Zhouさん/T05_CopulaSA_Hybrid_Balanced_NewToken'),
               ('Zhou Protected', 'Zhouさん/T05_CopulaSA_Hybrid_Protected_NewToken'),
               ('Seo SA Hardened', 'Seoさん/PWS_Cup2026_T05_SA_MIA_AIA_Hardened'),
               ('Seo SA Refined', 'Seoさん/PWS_Cup2026_T05_SA_PrivacyRefined'),
               ('Tashiro Copula', '加工データ候補0812_Tashiro/T05_Copula_Tashiro'),
               ('Tashiro Pool', '加工データ候補0812_Tashiro/T05_Pool_Tashiro'),
               ('Tashiro SA', '加工データ候補0812_Tashiro/T05_SA_Tashiro')]:
    T.append((lab, pd.read_csv(common.SUB + p + '/C.csv')))
NUL = [(f'C_{k}', pd.read_csv(common.ATK + f'C_{k}.csv')) for k in KS if k != 5]
zoo = densities.ratio_zoo()
cache, A, P = {}, {}, {}
for lab, C in T + NUL:
    S = z5(C); ra, rp = {}, {}
    for est, mk in zoo:
        r = mk()
        if isinstance(r, TwoDensityRatio):
            if est not in cache:
                cache[est] = r.maker().fit(REF).logpdf(EVAL)
            s = r.maker().fit(S).logpdf(EVAL) - cache[est]
        else:
            s = r.fit(S, REF).log_ratio(EVAL)
        t1, a1 = _tpr_at_fpr(s, y, 0.01); t2, a2 = _tpr_at_fpr(-s, y, 0.01)
        ra[est] = max(a1, a2); rp[est] = max(t1, t2)
    A[lab], P[lab] = ra, rp
nlA = pd.DataFrame({l: A[l] for l, _ in NUL}).T
nlP = pd.DataFrame({l: P[l] for l, _ in NUL}).T
for nm, cur, nl in [('AUC(両側)', A, nlA), ('TPR@1%FPR(両側)', P, nlP)]:
    print(f'\n■ Panel B {nm} — 帰無=他チーム C_k 19本')
    print(f'{"密度関数":<20}{"帰無μ":>7}{"帰無σ":>7}{"帰無max":>8}  ' + ''.join(f'{l[:10]:>12}' for l, _ in T))
    for est, _ in zoo:
        mu, sg = nl[est].mean(), nl[est].std(ddof=1)
        print(f'{est:<20}{mu:>7.3f}{sg:>7.3f}{nl[est].max():>8.3f}  '
              + ''.join(f'{cur[l][est]:>7.3f}({(cur[l][est]-mu)/sg:+.1f})' for l, _ in T))
