import sys, json
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0,"/Users/taisei1105/pws2026-local/work")
import attacks as AT
from audit import _tpr_at
D="/Users/taisei1105/pws2026-local/予備戦/配布データ/PWSCup2026_prelim_attack/"
W="/Users/taisei1105/pws2026-local/work/data_prelim_team05/"
CONT=["age","BMI","SBP","TG","HDL","ALT","FPG"]
CONT8=CONT+["time"]
Bs=pd.read_csv(W+"B_self.csv"); isr=Bs["is_rare"].to_numpy(bool); B=Bs.drop(columns=["is_rare"]).reset_index(drop=True)
A=pd.read_csv(W+"A_bg.csv")
mu=A[CONT8].mean().to_numpy(float); sd=A[CONT8].std(ddof=0).to_numpy(float)
def zz(df,cols):
    i=[CONT8.index(c) for c in cols]
    return (df[cols].to_numpy(float)-mu[i])/sd[i]

def matched_controls(T, pool, rng, n=None):
    """1:1 controls from pool matching (sex, prefecture, age//5, onset)."""
    p=pool.reset_index(drop=True)
    pb=(p["age"]//5).astype(int).to_numpy()
    tb=(T["age"]//5).astype(int).to_numpy()
    lvls=[list(zip(p.sex,p.prefecture,pb,p.onset)), list(zip(p.sex,pb,p.onset)), list(zip(p.sex,p.onset)), list(zip(p.onset))]
    tl  =[list(zip(T.sex,T.prefecture,tb,T.onset)), list(zip(T.sex,tb,T.onset)), list(zip(T.sex,T.onset)), list(zip(T.onset))]
    dicts=[]
    for L in lvls:
        d={}
        for i,kk in enumerate(L): d.setdefault(kk,[]).append(i)
        for v in d.values(): rng.shuffle(v)
        dicts.append(d)
    used=set(); out=[]
    for r in range(len(T)):
        got=None
        for lv in range(4):
            lst=dicts[lv].get(tl[lv][r])
            if lst:
                while lst:
                    j=lst.pop()
                    if j not in used: got=j; break
            if got is not None: break
        if got is None: got=int(rng.integers(len(p)))
        used.add(got); out.append(got)
    return p.iloc[out].reset_index(drop=True)

def scores(C, X, Apool):
    out={}
    zc=zz(C,CONT); za=zz(Apool,CONT); zx=zz(X,CONT)
    nnC=NearestNeighbors(n_neighbors=15).fit(zc); nnA=NearestNeighbors(n_neighbors=15).fit(za)
    dC=nnC.kneighbors(zx)[0]; dA=nnA.kneighbors(zx)[0]
    out["nn1_neg"]=-dC[:,0]
    out["nn_gap"]=np.log(dA[:,0]+1e-12)-np.log(dC[:,0]+1e-12)
    out["knn15"]=7*(np.log(dA[:,14]+1e-12)-np.log(dC[:,14]+1e-12))
    zc8=zz(C,CONT8); zx8=zz(X,CONT8)
    out["nn1_8_neg"]=-NearestNeighbors(n_neighbors=1).fit(zc8).kneighbors(zx8)[0][:,0]
    # block NN on (sex,prefecture)
    d=np.full(len(X),np.inf)
    g={k:np.asarray(v) for k,v in C.groupby(["sex","prefecture"]).groups.items()}
    gs={k:np.asarray(v) for k,v in C.groupby("sex").groups.items()}
    for i in range(len(X)):
        idx=g.get((X.sex.iloc[i],X.prefecture.iloc[i]))
        if idx is None or len(idx)==0: idx=gs.get(X.sex.iloc[i], np.arange(len(C)))
        d[i]=np.min(np.sqrt(np.mean((zc[idx]-zx[i])**2,axis=1)))
    out["nnblock_neg"]=-d
    out["domias"]=AT.m1_domias(C,Apool,X)
    return out

def auc(s,y):
    r=pd.Series(s).rank().to_numpy()
    n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

rng=np.random.default_rng(0)
res={}
for k in [24,26,29]:
    C=pd.read_csv(D+f"C_{k}.csv")
    # decontaminate A_bg: drop rows very close to C
    zc=zz(C,CONT); za=zz(A,CONT)
    dac=NearestNeighbors(n_neighbors=1).fit(zc).kneighbors(za)[0][:,0]
    keep=dac>0.05
    Apool=A[keep].reset_index(drop=True)
    print("="*60,"TEAM",k,"  A_bg clean pool:",len(Apool),"/",len(A), " dropped:",(~keep).sum())
    nonTok=(B.prefecture!="東京都").to_numpy()
    Bm=B[nonTok].reset_index(drop=True)   # plausible members only
    isrm=isr[nonTok]
    CTL=matched_controls(Bm,Apool,rng)
    X=pd.concat([Bm,CTL],ignore_index=True); y=np.r_[np.ones(len(Bm),int),np.zeros(len(CTL),int)]
    S=scores(C,X,Apool)
    row={}
    for nm,v in S.items():
        a=auc(v,y)
        row[nm]=round(float(a),4)
    print("  AUC (B non-Tokyo vs matched A_bg controls):",row)
    # DCR share
    n=len(Bm); dcr=float((-S["nn1_neg"][:n] < -S["nn1_neg"][n:]).mean())
    print(f"  DCR share (member closer than its control) = {dcr:.4f}  (0.5=no leak, n={n})")
    from scipy.stats import binomtest
    print("   binom p =", binomtest(int(dcr*n),n,0.5).pvalue)
    res[k]={"auc":row,"dcr":dcr}
json.dump(res,open("mia_auc.json","w"),indent=1)
