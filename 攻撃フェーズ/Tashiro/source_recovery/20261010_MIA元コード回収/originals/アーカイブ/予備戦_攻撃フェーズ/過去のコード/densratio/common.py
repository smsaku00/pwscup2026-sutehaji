"""データ読み込みと評価集合の構成。"""
from __future__ import annotations
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/Users/taisei1105/pws2026-local/スターターキット/改訂版_20260824/starter/src')

BASE = '/Users/taisei1105/pws2026-local/予備戦/配布データ/'
ATK  = BASE + 'PWSCup2026_prelim_attack/'
TEAM = BASE + 'PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/'
SUB  = '/Users/taisei1105/pws2026-local/提出データ/'

CONT7 = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']       # schema.CONTINUOUS
CONT5 = ['age', 'SBP', 'TG', 'HDL', 'ALT']                     # aia_challenge が持つ連続列
SEED  = 20260901


def load():
    d = {}
    d['abg'] = pd.read_csv(BASE + 'PWSCup2026_prelim_common/A_bg.csv')
    d['B']   = pd.read_csv(TEAM + 'B_self.csv')
    d['C5']  = pd.read_csv(ATK + 'C_5.csv')
    d['ch5'] = pd.read_csv(ATK + 'aia_challenge_5.csv')
    d['mem_ch'] = np.load('p5_member_challenge_rows.npy')
    d['mem_B']  = np.load('p5_member_B_rows.npy')
    d['ctl_ch'] = np.load('p5_control_challenge_rows.npy')
    return d


def zscaler(abg, cols):
    mu = abg[cols].mean().to_numpy(float)
    sd = abg[cols].std(ddof=0).to_numpy(float)
    sd = np.where(sd > 0, sd, 1.0)
    return lambda df: (df[cols].to_numpy(float) - mu) / sd


def split_abg(abg, rng):
    """A_bg を県で層化して 参照用 / 対照プール に二分する（参照と対照を素にする）。"""
    ref, pool = [], []
    for _, idx in abg.groupby('prefecture').groups.items():
        idx = np.asarray(idx); rng.shuffle(idx)
        h = len(idx) // 2
        ref.extend(idx[:h]); pool.extend(idx[h:])
    return abg.loc[sorted(ref)].reset_index(drop=True), abg.loc[sorted(pool)].reset_index(drop=True)


def matched_negatives(pool, B, n_target, rng):
    """B の (sex, prefecture) 同時分布に合わせて pool から非メンバー対照を引く。"""
    w = B.groupby(['sex', 'prefecture']).size() / len(B)
    take, cap = {}, []
    for key, frac in w.items():
        avail = pool.index[(pool.sex == key[0]) & (pool.prefecture == key[1])].to_numpy()
        cap.append(len(avail) / frac if frac > 0 else np.inf)
        take[key] = avail
    n = int(min(n_target, min(cap)))
    out = []
    for key, frac in w.items():
        k = int(round(frac * n))
        av = take[key]
        if k > 0 and len(av) > 0:
            out.extend(rng.choice(av, min(k, len(av)), replace=False))
    return pool.loc[sorted(set(out))].reset_index(drop=True), n


def rare_M(df, schema_json):
    """schema.json の rare_detector（マハラノビス）。真の is_rare ではないので近似指標。"""
    import json
    rd = json.load(open(schema_json))['rare_detector']
    cols, mu, S = rd['columns'], np.array(rd['mu']), np.array(rd['sigma'])
    X = df[cols].to_numpy(float) - mu
    Si = np.linalg.inv(S)
    return np.sqrt(np.einsum('ij,jk,ik->i', X, Si, X))
