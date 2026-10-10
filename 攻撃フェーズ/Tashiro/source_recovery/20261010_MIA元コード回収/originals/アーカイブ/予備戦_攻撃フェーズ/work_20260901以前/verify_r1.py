# -*- coding: utf-8 -*-
"""ラウンド1の中身を検算する。(a) AIA 上書きを 20 seed の帰無で再判定、(b) 確定メンバーの構造検査。"""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0,'.'); from build_attack2 import zmat,key_index,nn_by_key,DIST,DATA,QI4,LAB,KEY,HOME
from scipy.stats import binomtest
A=pd.read_csv(DATA/'A_bg.csv'); Bs=pd.read_csv(DATA/'B_self.csv'); B=Bs.drop(columns=['is_rare'])
mu4,sd4=A[QI4].mean().to_numpy(),A[QI4].std(ddof=0).to_numpy()
ai2=key_index(A,['sex','prefecture']); aage=A['age'].to_numpy()
print("【AIA】20 seed の帰無で照合率を再判定")
print("コホ   しきい値  challenge一致   帰無 平均±SD      m")
for k in [1,2,10,23,13,22,12]:
    C=pd.read_csv(DIST/f'C_{k}.csv'); Q=pd.read_csv(DIST/f'aia_challenge_{k}.csv')
    cz=zmat(C,QI4,mu4,sd4); ci=key_index(C,['sex','prefecture'])
    qk=list(zip(Q.sex,Q.prefecture)); qa=Q.age.to_numpy()
    dq,_=nn_by_key(zmat(Q,QI4,mu4,sd4),qk,qa,cz,ci)
    nulls=[]
    for s in range(20):
        rng=np.random.default_rng(s); pn=[]
        for (sx,p),ag in zip(qk,qa):
            js=[x for aa,x in ai2.get((sx,p),[]) if abs(aa-ag)<=1]
            if js: pn.append(rng.choice(js))
        dn,_=nn_by_key(zmat(A.iloc[pn],QI4,mu4,sd4),[(A.sex[j],A.prefecture[j]) for j in pn],aage[pn],cz,ci)
        nulls.append(dn)
    best=None
    for t in (0.02,0.05,0.10,0.15,0.20,0.30,0.40):
        h=int((dq<t).sum()); nv=np.array([(dn<t).sum()*len(Q)/len(dn) for dn in nulls])
        m=(h-nv.mean())/163
        if best is None or m>best[0]: best=(m,t,h,nv.mean(),nv.std())
    m,t,h,nm,ns=best
    z=(h-nm)/max(ns,1e-9)
    print(f"{k:3d}{t:9.2f}{h:14d}   {nm:6.1f}±{ns:5.1f}   {min(m,1):.3f}   z={z:6.1f}  {'採用' if m>=0.2 and z>5 else '不採用'}")
print("\n【MIA】確定メンバーの構造検査（共有ブロックは地元県 12.1%・非共有は 38.3%）")
F=pd.read_csv('prelim_attack2/r1/F_mia.csv')
home=(Bs.prefecture==HOME).to_numpy(); rid=Bs.record_id.to_numpy()
sets={}
for k in ['1','2','10']:
    ix=np.flatnonzero(F[k].to_numpy()==1.0); sets[k]=set(rid[ix])
    if len(ix)==0: continue
    nh=int(home[ix].sum())
    p=binomtest(nh,len(ix),0.383,alternative='less').pvalue
    print(f"  コホート {k}: {len(ix)}人  地元県 {nh}/{len(ix)} = {100*nh/len(ix):.0f}%  二項検定 p={p:.4f}  rid={sorted(rid[ix])}")
print(f"  列間排他 1∩2 = {sorted(sets['1']&sets['2'])}   1∩10 = {sorted(sets['1']&sets['10'])}  2∩10 = {sorted(sets['2']&sets['10'])}")
