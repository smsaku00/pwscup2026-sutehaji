"""AIA challenge を「校正済み MIA 検証台」として使い、スコア関数を比較する。
   対角 (i=k): 163 member + 163 non-member / 非対角 (i!=k): 326 全部 non-member（真値 0）
   → 非対角 20 本を null 反復として使い、対角の超過分だけを TPR と読む（A_bg 基準の偏りを回避）"""
import numpy as np, pandas as pd, json
from scipy.spatial import cKDTree
BASE='/Users/taisei1105/pws2026-local/予備戦/配布データ/'; ATK=BASE+'PWSCup2026_prelim_attack/'
CONT=['age','SBP','TG','HDL','ALT']
KS=[int(k) for k in json.load(open(ATK+'mia_columns.json'))['target_columns']]
abg=pd.read_csv(BASE+'PWSCup2026_prelim_common/A_bg.csv'); SD=abg[CONT].std().values
CH={k:pd.read_csv(ATK+f'aia_challenge_{k}.csv') for k in KS}
LOC={k:CH[k]['prefecture'].iloc[0] for k in KS}
CS={k:pd.read_csv(ATK+f'C_{k}.csv') for k in KS}

def trees(D, pref=None):
    t={}
    for s in (0,1):
        m=(D['sex'].values==s)
        if pref is not None: m &= (D['prefecture'].values==pref)
        if m.sum()>=3: t[s]=cKDTree(D.loc[m,CONT].values/SD)
    return t
def kd(X,t,k=5):
    out=np.full((len(X),k),np.nan); Xv=X[CONT].values/SD
    for s,tr in t.items():
        m=X['sex'].values==s
        if m.sum()==0: continue
        d,_=tr.query(Xv[m],k=min(k,tr.n)); d=np.atleast_2d(d)
        out[m,:d.shape[1]]=d
    return out
TA={}                                  # A_bg 側の木（県別）
for p in set(LOC.values()): TA[p]=trees(abg[abg['prefecture']==p])
TAall=trees(abg)

def feats(X,Ck,pref):
    dc=kd(X,trees(Ck)); dcp=kd(X,trees(Ck,pref)); da=kd(X,TAall)
    eps=1e-9
    return {'d1':dc[:,0], 'd1_pref':dcp[:,0], 'r12':dc[:,0]/(dc[:,1]+eps),
            'dr1':np.log(da[:,0]+eps)-np.log(dc[:,0]+eps),
            'dr5':np.log(da[:,4]+eps)-np.log(dc[:,4]+eps)}
SIGN={'d1':-1,'d1_pref':-1,'r12':-1,'dr1':+1,'dr5':+1}   # 大きいほどメンバーらしい向きに

rows=[]
for k in KS:
    if k==5: continue
    S={i:feats(CH[i],CS[k],LOC[i]) for i in KS}
    for f,sg in SIGN.items():
        null=np.concatenate([sg*S[i][f] for i in KS if i!=k])
        null=null[np.isfinite(null)]; null.sort()
        on=sg*S[k][f]; on=on[np.isfinite(on)]
        r={'k':k,'feat':f}
        for a in (0.01,0.05):
            thr=np.quantile(null,1-a)
            hit=(on>thr).sum(); exp=len(on)*a
            r[f'tpr@{a}']=round((hit-exp)/163,3)
            r[f'z@{a}']=round((hit-exp)/np.sqrt(exp+1e-9),2)
        rows.append(r)
df=pd.DataFrame(rows)
pv=df.pivot(index='k',columns='feat',values='z@0.01')
pv5=df.pivot(index='k',columns='feat',values='z@0.05')
pd.set_option('display.width',250)
print('=== z @ FPR 1% ===');  print(pv.round(1).to_string())
print('\n=== z @ FPR 5% ===');print(pv5.round(1).to_string())
df.to_csv('/Users/taisei1105/pws2026-local/work/aia2mia/bench.csv',index=False)
print('\n=== 対角 TPR@1% (z>=3 のみ) ===')
print(df[df['z@0.01']>=3][['k','feat','tpr@0.01','z@0.01','tpr@0.05','z@0.05']].to_string(index=False))
