"""AIA を対照群に依存しない統計量で見る。

p_m = 会員163人の time を τ=0.5 以内で当てた割合。C にしか依存しないので、
代理対照の作り方による揺れを完全に排除できる。帰無帯は他チーム C_k 19本。
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common, densities
from densities import TwoDensityRatio

d = common.load(); abg, B, ch = d['abg'], d['B'], d['ch5']
KS = [int(k) for k in json.load(open(common.ATK + 'mia_columns.json'))['target_columns']]
z5 = common.zscaler(abg, common.CONT5)
mem_qi = ch.iloc[d['mem_ch']].reset_index(drop=True)
t_m = B.iloc[d['mem_B']].time.to_numpy(float)
rare_m = B.iloc[d['mem_B']].is_rare.to_numpy(bool)
TAU, CONST = 0.5, 9.4990
tmu, tsd = abg.time.mean(), abg.time.std(ddof=0); zt = lambda v: (np.asarray(v, float) - tmu) / tsd
cand = np.unique(np.round(np.r_[np.quantile(abg.time.values, np.linspace(0, 1, 170)),
                                np.linspace(0, 10, 50), 9.998631, CONST], 9))
Y = np.column_stack([np.repeat(z5(mem_qi), len(cand), axis=0), np.tile(zt(cand), len(mem_qi))])
REF6 = np.column_stack([z5(abg), zt(abg.time.values)])

T = [('C_5 提出', d['C5'])]
for lab, p in [('Zhou Balanced', 'Zhouさん/T05_CopulaSA_Hybrid_Balanced_NewToken'),
               ('Zhou Protected', 'Zhouさん/T05_CopulaSA_Hybrid_Protected_NewToken'),
               ('Seo SA Hardened', 'Seoさん/PWS_Cup2026_T05_SA_MIA_AIA_Hardened'),
               ('Seo SA Refined', 'Seoさん/PWS_Cup2026_T05_SA_PrivacyRefined'),
               ('Tashiro Copula', '加工データ候補0812_Tashiro/T05_Copula_Tashiro'),
               ('Tashiro Pool', '加工データ候補0812_Tashiro/T05_Pool_Tashiro'),
               ('Tashiro SA', '加工データ候補0812_Tashiro/T05_SA_Tashiro')]:
    T.append((lab, pd.read_csv(common.SUB + p + '/C.csv')))
NULLS = [(f'C_{k}', pd.read_csv(common.ATK + f'C_{k}.csv')) for k in KS if k != 5]
sd = B[common.CONT7].std(ddof=0).to_numpy(float)
LEAK = []
for sig in (0.1, 0.3):
    r = np.random.default_rng(common.SEED + 400); X = B.copy()
    X[common.CONT7] = B[common.CONT7].to_numpy(float) + r.normal(0, sig * sd, (len(B), 7))
    LEAK.append((f'B+{sig:g}σ雑音', X))

print(f'定数{CONST} を置いたときの p_m = {np.mean(np.abs(t_m-CONST)<=TAU):.4f}')
zoo = [e for e in densities.ratio_zoo() if e[0].split()[0] in ('D1', 'D1b', 'D1c', 'D2', 'D3', 'D4', 'D5b', 'D6', 'D7')]
cache, rows = {}, []
for lab, C in T + NULLS + LEAK:
    C6 = np.column_stack([z5(C), zt(C.time.values)])
    rec = {'target': lab}
    for est, mk in zoo:
        r = mk()
        if isinstance(r, TwoDensityRatio):
            if est not in cache:
                cache[est] = r.maker().fit(REF6).logpdf(Y)
            lref = cache[est]
            S = np.where(lref.reshape(163, -1) >= lref.reshape(163, -1).max(1, keepdims=True) - 10.0,
                         (r.maker().fit(C6).logpdf(Y) - lref).reshape(163, -1), -np.inf)
        else:
            S = r.fit(C6, REF6).log_ratio(Y).reshape(163, -1)
        ok = np.abs(cand[np.argmax(S, 1)] - t_m) <= TAU
        rec[est] = ok.mean(); rec[est + '/希少'] = ok[rare_m].mean()
    rows.append(rec)
    print(f'{lab:18s} ' + ' '.join(f'{k.split()[0]}:{v:.3f}' for k, v in rec.items()
                                   if k != 'target' and '希少' not in k), flush=True)
df = pd.DataFrame(rows).set_index('target'); df.to_csv('aia_pm.csv')
nl = df.loc[[l for l, _ in NULLS]]
print('\n=== p_m の帰無帯（他チーム C_k 19本）と自陣候補 ===')
print(f'{"手法":<20}{"帰無μ":>8}{"帰無σ":>8}{"帰無max":>8}  ' + ''.join(f'{l[:11]:>12}' for l, _ in T))
for est, _ in zoo:
    m, s = nl[est].mean(), nl[est].std(ddof=1)
    line = f'{est:<20}{m:>8.3f}{s:>8.3f}{nl[est].max():>8.3f}  '
    line += ''.join(f'{df.loc[l,est]:>7.3f}({(df.loc[l,est]-m)/s:+.1f})' for l, _ in T)
    print(line)
print('\n漏洩アンカー:'); print(df.loc[[l for l, _ in LEAK], [e for e, _ in zoo]].round(3).to_string())
