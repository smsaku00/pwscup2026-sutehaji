import numpy as np, pandas as pd, json, os
BASE='/Users/taisei1105/pws2026-local/予備戦/配布データ/'
ATK=BASE+'PWSCup2026_prelim_attack/'
TEAM=BASE+'PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/'
QI=['age','SBP','TG','HDL','ALT','sex','prefecture']

B   = pd.read_csv(TEAM+'B_self.csv')
ch5 = pd.read_csv(ATK+'aia_challenge_5.csv')
abg = pd.read_csv(BASE+'PWSCup2026_prelim_common/A_bg.csv')
C5  = pd.read_csv(ATK+'C_5.csv')
print('B',B.shape,'ch5',ch5.shape,'abg',abg.shape,'C5',C5.shape)
print('challenge prefecture:',ch5.prefecture.unique())

# 完全一致キー（float をそのまま文字列化）
def key(df):
    return (df.age.astype(int).astype(str)+'|'+df.sex.astype(int).astype(str)+'|'+df.prefecture+'|'
            +df.SBP.map(repr)+'|'+df.TG.map(repr)+'|'+df.HDL.map(repr)+'|'+df.ALT.map(repr))
kB = key(B); kC = key(ch5)
m = pd.Series(kB.index.values, index=kB.values)
hit = kC.map(m)
print('challenge_5 → B_5 完全一致:', hit.notna().sum(), '/', len(ch5))
mem_rows   = hit.dropna().astype(int)          # index=challenge_row, value=B行番号
print('うち onset=1:', int(B.onset.iloc[mem_rows.values].sum()),
      ' is_rare=1:', int(B.is_rare.iloc[mem_rows.values].sum()),
      ' 希少かつonset:', int(((B.onset.iloc[mem_rows.values].values==1)&(B.is_rare.iloc[mem_rows.values].values==1)).sum()))
print('B_5 全体 is_rare:', int(B.is_rare.sum()))
# A_bg に居ないことの確認（対照163行）
kA = set(key(abg).values)
ctrl = ch5.index.difference(mem_rows.index)
print('対照163行のうち A_bg 完全一致:', sum(1 for i in ctrl if kC[i] in kA))

np.save('p5_member_challenge_rows.npy', mem_rows.index.values)
np.save('p5_member_B_rows.npy',        mem_rows.values)
np.save('p5_control_challenge_rows.npy', ctrl.values)
print('saved.')
