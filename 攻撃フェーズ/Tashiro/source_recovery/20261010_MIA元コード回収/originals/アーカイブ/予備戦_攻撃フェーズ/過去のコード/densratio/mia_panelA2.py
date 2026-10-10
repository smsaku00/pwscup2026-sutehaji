"""Panel A に他チーム C_k 19本の帰無帯を足す（陽性 B_5 1049 / 陰性 A_bg整合）。"""
import json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common, densities, mia_exp
d = common.load(); abg, B = d['abg'], d['B']
rng = np.random.default_rng(common.SEED)
A_ref, A_pool = common.split_abg(abg, rng)
neg, _ = common.matched_negatives(A_pool, B, 1049, rng)
KS = [int(k) for k in json.load(open(common.ATK + 'mia_columns.json'))['target_columns']]
T = [(f'[帰無] C_{k} (他チーム)', pd.read_csv(common.ATK + f'C_{k}.csv')) for k in KS if k != 5]
M = np.r_[common.rare_M(B, mia_exp.SCHEMA), common.rare_M(neg, mia_exp.SCHEMA)]
rows = mia_exp.run_panel('A', common.CONT7, abg, A_ref, B, neg, T, densities.ratio_zoo(), rare=(M >= 4.0))
pd.DataFrame(rows).to_csv('mia_panelA2.csv', index=False)
print('wrote', len(rows))
