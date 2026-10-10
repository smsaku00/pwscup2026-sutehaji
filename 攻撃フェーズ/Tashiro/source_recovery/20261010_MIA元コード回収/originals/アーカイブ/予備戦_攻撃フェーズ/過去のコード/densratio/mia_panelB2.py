"""Panel B 拡張: 帰無アンカーを「他チームの C_k」にする。

C_k (k≠5) は別コホート B_k から作られており、B_5 のメンバーシップとは無関係＝真の帰無。
19 本の実物 C で経験帯を作り、そこから自陣候補がどれだけ外れるかを見る。
"""
from __future__ import annotations
import json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common, densities
from densities import TwoDensityRatio
from scipy.stats import mannwhitneyu
from pwscup2026.scoring.attack_score import _tpr_at_fpr

d = common.load(); abg = d['abg']; B = d['B']
KS = [int(k) for k in json.load(open(common.ATK + 'mia_columns.json'))['target_columns']]
print('配布 C:', KS)

T = [('C_5 (提出=CopulaSA_Utility)', d['C5'])]
for lab, p in [('Zhou Balanced', 'Zhouさん/T05_CopulaSA_Hybrid_Balanced_NewToken'),
               ('Zhou Protected', 'Zhouさん/T05_CopulaSA_Hybrid_Protected_NewToken'),
               ('Seo SA Hardened', 'Seoさん/PWS_Cup2026_T05_SA_MIA_AIA_Hardened'),
               ('Seo SA Refined', 'Seoさん/PWS_Cup2026_T05_SA_PrivacyRefined'),
               ('Tashiro Copula', '加工データ候補0812_Tashiro/T05_Copula_Tashiro'),
               ('Tashiro Pool', '加工データ候補0812_Tashiro/T05_Pool_Tashiro'),
               ('Tashiro SA', '加工データ候補0812_Tashiro/T05_SA_Tashiro')]:
    T.append((f'自陣 {lab}', pd.read_csv(common.SUB + p + '/C.csv')))
for k in KS:
    if k != 5:
        T.append((f'[帰無] C_{k} (他チーム)', pd.read_csv(common.ATK + f'C_{k}.csv')))
sd = B[common.CONT7].std(ddof=0).to_numpy(float)
for sig in (0.1, 0.3):
    r = np.random.default_rng(common.SEED + 300)
    X = B.copy(); X[common.CONT7] = B[common.CONT7].to_numpy(float) + r.normal(0, sig * sd, (len(B), 7))
    T.append((f'[漏洩] B+{sig:g}σ雑音', X))

ch = d['ch5']
z = common.zscaler(abg, common.CONT5)
REF = z(abg)
EVAL = np.vstack([z(ch.iloc[d['mem_ch']]), z(ch.iloc[d['ctl_ch']])])
y = np.r_[np.ones(163, int), np.zeros(163, int)]
zoo = densities.ratio_zoo()
logref, rows = {}, []
for label, C in T:
    S = z(C)
    line = [f'{label:28s}']
    for est_name, mk in zoo:
        r = mk()
        if isinstance(r, TwoDensityRatio):
            if est_name not in logref:
                logref[est_name] = r.maker().fit(REF).logpdf(EVAL)
            s = np.clip(r.maker().fit(S).logpdf(EVAL) - logref[est_name], -200, 200)
        else:
            s = r.fit(S, REF).log_ratio(EVAL)
        tpr1, auc = _tpr_at_fpr(s, y, 0.01)
        rows.append(dict(target=label, est=est_name, auc=auc, tpr1=tpr1,
                         p=mannwhitneyu(s[y == 1], s[y == 0], alternative='greater').pvalue))
        line.append(f'{auc:.3f}')
    print(' '.join(line), flush=True)
pd.DataFrame(rows).to_csv('mia_panelB2.csv', index=False)
print('\n列順:', [n for n, _ in zoo])
