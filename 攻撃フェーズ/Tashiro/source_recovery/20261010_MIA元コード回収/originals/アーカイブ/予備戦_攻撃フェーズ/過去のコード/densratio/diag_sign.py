"""Panel B で Copula 系の AUC が 0.5 を下回る向きの正体を調べる。

自作の素朴なガウシアンコピュラ合成を (a) B_5 から (b) A_bg から作り、同じ Panel B にかける。
B_5 から作った合成も下振れするなら「コピュラ合成一般の性質」、しないなら「C_5 固有」。
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common
from densities import KDEModel, ClassifierRatio
from scipy.stats import norm, rankdata
from pwscup2026.scoring.attack_score import _tpr_at_fpr

d = common.load(); abg, B, ch = d['abg'], d['B'], d['ch5']
COLS = common.CONT7


def copula_synth(src, n, seed):
    rng = np.random.default_rng(seed)
    X = src[COLS].to_numpy(float)
    U = (rankdata(X, axis=0) - 0.5) / len(X)
    Z = norm.ppf(U)
    R = np.corrcoef(Z.T)
    L = np.linalg.cholesky(R + 1e-8 * np.eye(len(R)))
    Zs = rng.standard_normal((n, len(R))) @ L.T
    Us = norm.cdf(Zs)
    out = np.column_stack([np.quantile(X[:, j], Us[:, j]) for j in range(X.shape[1])])
    df = pd.DataFrame(out, columns=COLS)
    for c in ['sex', 'prefecture', 'smoking', 'onset', 'death', 'time']:
        df[c] = src[c].sample(n, replace=True, random_state=seed).to_numpy()
    return df


z5 = common.zscaler(abg, common.CONT5)
REF = z5(abg)
EVAL = np.vstack([z5(ch.iloc[d['mem_ch']]), z5(ch.iloc[d['ctl_ch']])])
y = np.r_[np.ones(163, int), np.zeros(163, int)]
KS = [int(k) for k in json.load(open(common.ATK + 'mia_columns.json'))['target_columns']]

T = [('C_5 提出', d['C5']),
     ('自作コピュラ← B_5', copula_synth(B, 1049, 1)),
     ('自作コピュラ← B_5 (種2)', copula_synth(B, 1049, 2)),
     ('自作コピュラ← A_bg', copula_synth(abg, 1049, 3)),
     ('B_5 そのまま', B),
     ('A_bg 抽出1049', abg.sample(1049, random_state=4).reset_index(drop=True)),
     ('自陣 Tashiro Pool', pd.read_csv(common.SUB + '加工データ候補0812_Tashiro/T05_Pool_Tashiro/C.csv'))]
NUL = [(f'C_{k}', pd.read_csv(common.ATK + f'C_{k}.csv')) for k in KS if k != 5]

res = {}
for lab, C in T + NUL:
    S = z5(C); row = {}
    for nm, mk in [('KDE(Scott)', lambda: KDEModel(1.0)), ('KDE(h=0.25)', lambda: KDEModel(0.25))]:
        row[nm] = _tpr_at_fpr(mk().fit(S).logpdf(EVAL) - mk().fit(REF).logpdf(EVAL), y, 0.01)[1]
    row['Clf-DRE(HGB)'] = _tpr_at_fpr(ClassifierRatio('hgb').fit(S, REF).log_ratio(EVAL), y, 0.01)[1]
    res[lab] = row
nl = pd.DataFrame({l: res[l] for l, _ in NUL}).T
print(f'{"C の作り方":<24}' + ''.join(f'{c:>16}' for c in nl.columns))
print(f'{"[帰無] 他チーム19本 μ±σ":<24}' + ''.join(f'{nl[c].mean():>9.3f}±{nl[c].std(ddof=1):.3f}' for c in nl.columns))
for lab, _ in T:
    print(f'{lab:<24}' + ''.join(f'{res[lab][c]:>10.3f}({(res[lab][c]-nl[c].mean())/nl[c].std(ddof=1):+.1f})'
                                 for c in nl.columns))
