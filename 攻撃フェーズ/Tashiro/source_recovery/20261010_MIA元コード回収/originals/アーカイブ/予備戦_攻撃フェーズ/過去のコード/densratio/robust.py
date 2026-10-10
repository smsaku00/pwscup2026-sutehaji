"""見出しの2手法（D5b Clf-DRE(HGB) の MIA と D1c KDE(h=0.25) の 密度比AIA）を
乱数種・陰性再抽出に対して振り、帰無帯からの外れが安定かを確かめる。"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common, densities
from densities import KDEModel, ClassifierRatio
from pwscup2026.scoring.attack_score import _tpr_at_fpr, R_from_rate
from sklearn.linear_model import LogisticRegression
from scipy.spatial import cKDTree

d = common.load(); abg, B, ch = d['abg'], d['B'], d['ch5']
KS = [int(k) for k in json.load(open(common.ATK + 'mia_columns.json'))['target_columns']]
z5 = common.zscaler(abg, common.CONT5)
REF5 = z5(abg)
EVAL = np.vstack([z5(ch.iloc[d['mem_ch']]), z5(ch.iloc[d['ctl_ch']])])
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
NULLS = [(f'C_{k}', pd.read_csv(common.ATK + f'C_{k}.csv')) for k in KS if k != 5]

print('=== MIA / D5b Clf-DRE(HGB) の AUC を種5本で（平均±SD）===')
res = {}
for lab, C in T + NULLS:
    S = z5(C); a = []
    for sd_ in range(5):
        s = ClassifierRatio('hgb', seed=sd_).fit(S, REF5).log_ratio(EVAL)
        a.append(_tpr_at_fpr(s, y, 0.01)[1])
    res[lab] = (np.mean(a), np.std(a))
nv = np.array([res[l][0] for l, _ in NULLS])
print(f'帰無19本: μ={nv.mean():.4f} σ={nv.std(ddof=1):.4f} 範囲 {nv.min():.3f}–{nv.max():.3f}')
for lab, _ in T:
    m, s = res[lab]
    print(f'  {lab:18s} AUC={m:.4f} ±{s:.4f}   z={(m-nv.mean())/nv.std(ddof=1):+.2f}')

print('\n=== AIA / D1c KDE(h=0.25) λ=1 の R を対照3組で ===')
CENS = 9.4986; TAU = 0.5; CONST = 9.4990
mem_qi = ch.iloc[d['mem_ch']].reset_index(drop=True); mem_B = B.iloc[d['mem_B']].reset_index(drop=True)
Xa = np.column_stack([z5(abg), abg.sex.values])
lr = LogisticRegression(max_iter=2000).fit(Xa, (abg.time >= CENS).astype(int))
pa = lr.predict_proba(Xa)[:, 1]; pm = lr.predict_proba(np.column_stack([z5(mem_qi), mem_qi.sex.values]))[:, 1]
tmu, tsd = abg.time.mean(), abg.time.std(ddof=0); zt = lambda v: (np.asarray(v, float) - tmu) / tsd
cand = np.unique(np.round(np.r_[np.quantile(abg.time.values, np.linspace(0, 1, 170)),
                                np.linspace(0, 10, 50), 9.998631, CONST], 9))
allres = {}
for rep in range(3):
    rng = np.random.default_rng(1000 + rep)
    out, taken = [], set()
    for s in (0, 1):
        mi = np.flatnonzero(mem_qi.sex.values == s); ci = np.flatnonzero(abg.sex.values == s)
        tree = cKDTree(np.column_stack([z5(abg)[ci], 3.0 * pa[ci]]))
        Tq = np.column_stack([z5(mem_qi)[mi], 3.0 * pm[mi]])
        for r in range(len(mi)):
            jj = tree.query(Tq[r], k=40)[1]; rng.shuffle(jj)     # 同程度の近傍から無作為に取る
            for j in np.atleast_1d(jj):
                if ci[j] not in taken: taken.add(ci[j]); out.append(ci[j]); break
    ctl = abg.iloc[sorted(out)].reset_index(drop=True)
    Q = pd.concat([mem_qi[common.CONT5 + ['sex']], ctl[common.CONT5 + ['sex']]], ignore_index=True)
    Y = np.column_stack([np.repeat(z5(Q), len(cand), axis=0), np.tile(zt(cand), len(Q))])
    lref = KDEModel(0.25).fit(np.column_stack([z5(abg), zt(abg.time.values)])).logpdf(Y)
    G = lref.reshape(len(Q), len(cand))
    t_m, t_c = mem_B.time.values, ctl.time.values
    pc0 = np.mean(np.abs(t_c - CONST) <= TAU)
    for lab, C in T + NULLS:
        lc = KDEModel(0.25).fit(np.column_stack([z5(C), zt(C.time.values)])).logpdf(Y)
        S = np.where(G >= G.max(1, keepdims=True) - 10.0, (lc - lref).reshape(len(Q), len(cand)), -np.inf)
        th = cand[np.argmax(S, 1)]
        R = R_from_rate((np.abs(th[:163] - t_m) <= TAU).astype(float),
                        (np.abs(th[163:] - t_c) <= TAU).astype(float))
        allres.setdefault(lab, []).append(R)
    print(f'  対照組#{rep}: 定数攻撃の p_c={pc0:.4f}')
nv = np.array([np.mean(allres[l]) for l, _ in NULLS])
print(f'帰無19本: μ={nv.mean():.4f} σ={nv.std(ddof=1):.4f} 範囲 {nv.min():.3f}–{nv.max():.3f}')
for lab, _ in T:
    v = np.array(allres[lab])
    print(f'  {lab:18s} R={v.mean():.4f} ±{v.std(ddof=1):.4f}   z={(v.mean()-nv.mean())/nv.std(ddof=1):+.2f}')
