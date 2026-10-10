"""AIA の結論が代理対照群の作り方にどれだけ依存するかを測る。

公式の対照（どのコホートにも A_bg にも居ない別人）の time は入手できないので、
A_bg から作った代理対照を 4 通り用意し、R の帰無帯と自陣候補の位置を比べる。
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common
from densities import KDEModel
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from pwscup2026.scoring.attack_score import R_from_rate

d = common.load(); abg, B, ch = d['abg'], d['B'], d['ch5']
KS = [int(k) for k in json.load(open(common.ATK + 'mia_columns.json'))['target_columns']]
z5 = common.zscaler(abg, common.CONT5)
mem_qi = ch.iloc[d['mem_ch']].reset_index(drop=True); mem_B = B.iloc[d['mem_B']].reset_index(drop=True)
t_m = mem_B.time.to_numpy(float); TAU, CONST, CENS = 0.5, 9.4990, 9.4986
Xa = np.column_stack([z5(abg), abg.sex.values])
pa = LogisticRegression(max_iter=2000).fit(Xa, (abg.time >= CENS).astype(int)).predict_proba(Xa)[:, 1]
lr2 = LogisticRegression(max_iter=2000).fit(Xa, (abg.time >= CENS).astype(int))
pm = lr2.predict_proba(np.column_stack([z5(mem_qi), mem_qi.sex.values]))[:, 1]


def controls(mode, seed=0):
    """mode: qi=純粋QI最近傍 / ps=QI+傾向スコア / loose=40近傍から無作為 / tokyo=東京限定"""
    rng = np.random.default_rng(seed)
    src = abg[abg.prefecture == '東京都'].reset_index(drop=True) if mode == 'tokyo' else abg
    Zs = z5(src); Zm = z5(mem_qi)
    ps_s = pa[abg.prefecture.values == '東京都'] if mode == 'tokyo' else pa
    F = np.column_stack([Zs, 3.0 * ps_s]) if mode in ('ps', 'loose') else Zs
    Tq = np.column_stack([Zm, 3.0 * pm]) if mode in ('ps', 'loose') else Zm
    out, taken = [], set()
    for s in (0, 1):
        mi = np.flatnonzero(mem_qi.sex.values == s); ci = np.flatnonzero(src.sex.values == s)
        tree = cKDTree(F[ci]); k = min(40, len(ci))
        for r in mi:
            jj = np.atleast_1d(tree.query(Tq[r], k=k)[1])
            if mode == 'loose':
                rng.shuffle(jj)
            for j in jj:
                if ci[j] not in taken: taken.add(ci[j]); out.append(ci[j]); break
    return src.iloc[sorted(out)].reset_index(drop=True)


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
tmu, tsd = abg.time.mean(), abg.time.std(ddof=0); zt = lambda v: (np.asarray(v, float) - tmu) / tsd
cand = np.unique(np.round(np.r_[np.quantile(abg.time.values, np.linspace(0, 1, 170)),
                                np.linspace(0, 10, 50), 9.998631, CONST], 9))
REF6 = np.column_stack([z5(abg), zt(abg.time.values)])
kref = KDEModel(0.25).fit(REF6)

print(f'{"対照の作り方":<26}{"定数のp_c":>10}{"定数のR":>9}{"帰無μ":>8}{"帰無σ":>8}  '
      + ''.join(f'{l[:11]:>13}' for l, _ in T))
for mode, name in [('qi', '純粋QI最近傍(全国)'), ('ps', 'QI+検閲傾向スコア'),
                   ('tokyo', '東京限定QI最近傍'), ('loose', '40近傍から無作為')]:
    ctl = controls(mode, seed=7)
    t_c = ctl.time.to_numpy(float)
    Q = pd.concat([mem_qi[common.CONT5 + ['sex']], ctl[common.CONT5 + ['sex']]], ignore_index=True)
    Y = np.column_stack([np.repeat(z5(Q), len(cand), axis=0), np.tile(zt(cand), len(Q))])
    lref = kref.logpdf(Y); G = lref.reshape(len(Q), -1)
    pc0 = np.mean(np.abs(t_c - CONST) <= TAU); pm0 = np.mean(np.abs(t_m - CONST) <= TAU)
    vals = {}
    for lab, C in T + NULLS:
        lc = KDEModel(0.25).fit(np.column_stack([z5(C), zt(C.time.values)])).logpdf(Y)
        S = np.where(G >= G.max(1, keepdims=True) - 10.0, (lc - lref).reshape(len(Q), -1), -np.inf)
        th = cand[np.argmax(S, 1)]
        vals[lab] = R_from_rate((np.abs(th[:163] - t_m) <= TAU).astype(float),
                                (np.abs(th[163:] - t_c) <= TAU).astype(float))
    nv = np.array([vals[l] for l, _ in NULLS]); mu, sg = nv.mean(), nv.std(ddof=1)
    print(f'{name:<26}{pc0:>10.4f}{(pm0-pc0)/(1-pc0):>9.3f}{mu:>8.3f}{sg:>8.3f}  '
          + ''.join(f'{vals[l]:>7.3f}({(vals[l]-mu)/sg:+.1f})' for l, _ in T))
