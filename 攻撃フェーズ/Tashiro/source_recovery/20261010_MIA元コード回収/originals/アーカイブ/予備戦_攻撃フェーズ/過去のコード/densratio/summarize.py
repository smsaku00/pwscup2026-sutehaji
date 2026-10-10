"""最終まとめ表。帰無帯＝他チーム C_k 19本、z = (値 − 帰無μ)/帰無σ。"""
import numpy as np, pandas as pd
pd.set_option('display.width', 300)
OURS = ['C_5 (提出=CopulaSA_Utility)', '自陣 Zhou Balanced', '自陣 Zhou Protected', '自陣 Tashiro Copula',
        '自陣 Seo SA Hardened', '自陣 Seo SA Refined', '自陣 Tashiro Pool', '自陣 Tashiro SA']
SHORT = ['C_5提出', 'Zhou-Bal', 'Zhou-Prot', 'Tash-Cop', 'Seo-SA-H', 'Seo-SA-R', 'Tash-Pool', 'Tash-SA']


def band(df, ours, metric, leaks=()):
    nl = df[df.target.str.startswith('[帰無] C_')]
    rows = []
    for est in df.est.unique():
        n = nl[nl.est == est][metric]
        if len(n) < 5:
            continue
        mu, sg = n.mean(), n.std(ddof=1)
        r = {'密度関数': est, '帰無μ': mu, '帰無σ': sg, '帰無max': n.max()}
        for lab, sh in zip(ours, SHORT):
            v = df[(df.est == est) & (df.target == lab)][metric]
            r[sh] = f'{v.iloc[0]:.3f}({(v.iloc[0]-mu)/sg:+.1f})' if len(v) else '—'
        for lk in leaks:
            v = df[(df.est == est) & (df.target == lk)][metric]
            r[lk.replace('[漏洩] ', '')] = f'{v.iloc[0]:.3f}' if len(v) else '—'
        rows.append(r)
    out = pd.DataFrame(rows).set_index('密度関数')
    out[['帰無μ', '帰無σ', '帰無max']] = out[['帰無μ', '帰無σ', '帰無max']].round(3)
    return out


# ---- Panel A（陽性 B_5 全1049 / 陰性 A_bg整合700 / 連続7列）
a = pd.concat([pd.read_csv('mia_results.csv').query("panel=='A'"),
               pd.read_csv('mia_panelA2.csv')], ignore_index=True)
a['target'] = a.target.str.replace('(参考) ', '自陣 ', regex=False)
print('■ MIA Panel A — AUC（陽性 B_5 1049 / 陰性 A_bg整合 700 / 参照 A_bg別半分 / 連続7列）')
print(band(a, OURS, 'auc', ['[漏洩] B+0.2σ雑音', '[漏洩] B+0.5σ雑音']).to_string())
print('\n■ MIA Panel A — TPR@1%FPR（＝公式の採点通貨）')
print(band(a, OURS, 'tpr1', ['[漏洩] B+0.2σ雑音', '[漏洩] B+0.5σ雑音']).to_string())

# ---- Panel B（陽性 challenge会員163 / 陰性 challenge対照163 / 連続5列）
b = pd.read_csv('mia_panelB2.csv')
print('\n■ MIA Panel B — AUC（陽性=challenge_5 会員163 / 陰性=同 対照163 ＝事務局が用意した交絡なしの帰無）')
print(band(b, OURS, 'auc', ['[漏洩] B+0.1σ雑音', '[漏洩] B+0.3σ雑音']).to_string())
print('\n■ MIA Panel B — TPR@1%FPR')
print(band(b, OURS, 'tpr1', ['[漏洩] B+0.1σ雑音', '[漏洩] B+0.3σ雑音']).to_string())
