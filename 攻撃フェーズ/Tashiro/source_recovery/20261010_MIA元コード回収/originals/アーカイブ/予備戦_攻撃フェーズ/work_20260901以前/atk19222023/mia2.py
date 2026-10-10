import sys, json
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
sys.path.insert(0,"/Users/taisei1105/pws2026-local/work")
from submit_audit import matched_controls
D = Path("/Users/taisei1105/pws2026-local/予備戦/配布データ/PWSCup2026_prelim_attack")
W = Path("/Users/taisei1105/pws2026-local/work/data_prelim_team05")
CONT = ["age","BMI","SBP","TG","HDL","ALT","FPG"]
Bs = pd.read_csv(W/"B_self.csv"); A = pd.read_csv(W/"A_bg.csv")
is_rare = Bs["is_rare"].to_numpy().astype(bool)
B = Bs.drop(columns=["is_rare"])
sd = A[CONT].std(ddof=0).to_numpy(float); mu = A[CONT].mean().to_numpy(float)
def Z(df): return (df[CONT].to_numpy(float)-mu)/sd
def auc(pos, neg):
    x=np.concatenate([pos,neg]); y=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))])
    r=pd.Series(x).rank().to_numpy()
    n1=y.sum(); n0=len(y)-n1
    return float((r[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
zA=Z(A); zB=Z(B)
kdeA = gaussian_kde(zA.T)
res={}
for k in [19,20,22,23]:
    C = pd.read_csv(D/f"C_{k}.csv"); zc=Z(C)
    o={}
    nn=NearestNeighbors(n_neighbors=10).fit(zc)
    nnA=NearestNeighbors(n_neighbors=10).fit(zA)
    kdeC=gaussian_kde(zc.T)
    # quantization grid test
    grids=[np.unique(C[c].to_numpy(float)) for c in CONT]
    def binvec(df):
        out=[]
        for i,c in enumerate(CONT):
            g=grids[i]; v=df[c].to_numpy(float)
            idx=np.searchsorted(g,v); idx=np.clip(idx,1,len(g)-1)
            left=g[idx-1]; right=g[idx]
            out.append(np.where(np.abs(v-left)<=np.abs(v-right), idx-1, idx))
        return np.stack(out,1)
    cb=set(map(tuple, binvec(C)))
    bb=binvec(B)
    o["binmatch_B"]=int(sum(1 for t in map(tuple,bb) if t in cb))
    ctls=[matched_controls(B,A,seed=s) for s in range(3)]
    o["binmatch_ctl"]=[int(sum(1 for t in map(tuple,binvec(c)) if t in cb)) for c in ctls]
    # scores
    def scores(X):
        zx=Z(X)
        dC,_=nn.kneighbors(zx,n_neighbors=1); dA,_=nnA.kneighbors(zx,n_neighbors=1)
        s={}
        s["nn"]=-dC[:,0]
        s["nngap"]=-(dC[:,0]/np.clip(dA[:,0],1e-6,None))
        dC5,_=nn.kneighbors(zx,n_neighbors=5); dA5,_=nnA.kneighbors(zx,n_neighbors=5)
        s["knn5"]=-(dC5[:,4]/np.clip(dA5[:,4],1e-6,None))
        s["domias"]=np.log(np.clip(kdeC(zx.T),1e-300,None))-np.log(np.clip(kdeA(zx.T),1e-300,None))
        return s
    sB=scores(B)
    aucs={}
    for nm in sB:
        vals=[]
        for c in ctls:
            sc=scores(c)
            vals.append(auc(sB[nm],sc[nm]))
        aucs[nm]=[round(v,4) for v in vals]
    o["auc"]=aucs
    # outcome-augmented distance (8 cols incl time), only sensible if time meaningful
    CONT8=CONT+["time"]
    sd8=A[CONT8].std(ddof=0).to_numpy(float); mu8=A[CONT8].mean().to_numpy(float)
    Z8=lambda df:(df[CONT8].to_numpy(float)-mu8)/sd8
    nn8=NearestNeighbors(n_neighbors=1).fit(Z8(C))
    dB8,_=nn8.kneighbors(Z8(B)); vals=[]
    for c in ctls:
        dc8,_=nn8.kneighbors(Z8(c)); vals.append(auc(-dB8[:,0],-dc8[:,0]))
    o["auc_cont8"]=[round(v,4) for v in vals]
    # rare rows: best score rank among the 24
    dC,_=nn.kneighbors(zB,n_neighbors=1)
    dom=sB["domias"]
    ri=np.flatnonzero(is_rare)
    o["rare_domias_top"]=[[int(i), round(float(dom[i]),3), round(float(dC[i,0]),3)] for i in ri[np.argsort(-dom[ri])][:5]]
    o["rare_domias_spread"]=[round(float(x),3) for x in np.sort(dom[ri])[::-1]]
    res[k]=o
print(json.dumps(res,ensure_ascii=False,indent=1))
