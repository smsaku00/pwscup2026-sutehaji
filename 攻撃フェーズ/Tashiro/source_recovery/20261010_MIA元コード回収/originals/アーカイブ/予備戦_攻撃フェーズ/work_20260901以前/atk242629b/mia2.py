import sys, json
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import binomtest
sys.path.insert(0,"/Users/taisei1105/pws2026-local/work")
import attacks as AT
exec(open("mia.py").read().split("rng=np.random")[0].split("import sys, json")[1].replace('sys.path.insert(0,"/Users/taisei1105/pws2026-local/work")','').replace("import attacks as AT","").replace("from audit import _tpr_at",""))
rng=np.random.default_rng(1)
res={}
for k in [24,26,29]:
    C=pd.read_csv(D+f"C_{k}.csv")
    zc=zz(C,CONT); za=zz(A,CONT)
    dac=NearestNeighbors(n_neighbors=1).fit(zc).kneighbors(za)[0][:,0]
    clean=A[dac>0.05].reset_index(drop=True)
    perm=rng.permutation(len(clean))
    half=len(perm)//2
    A_ctl=clean.iloc[perm[:half]].reset_index(drop=True)   # control source
    A_ref=clean.iloc[perm[half:]].reset_index(drop=True)   # reference density (disjoint)
    nonTok=(B.prefecture!="東京都").to_numpy()
    Bm=B[nonTok].reset_index(drop=True); isrm=isr[nonTok]
    CTL=matched_controls(Bm,A_ctl,rng)
    X=pd.concat([Bm,CTL],ignore_index=True); y=np.r_[np.ones(len(Bm),int),np.zeros(len(CTL),int)]
    S=scores(C,X,A_ref)
    print("="*60,"TEAM",k," |A_ref|",len(A_ref),"|A_ctl|",len(A_ctl))
    row={nm:round(float(auc(v,y)),4) for nm,v in S.items()}
    print("  AUC:",row)
    n=len(Bm); dcr=int(((-S["nn1_neg"][:n]) < (-S["nn1_neg"][n:])).sum())
    print(f"  DCR share={dcr/n:.4f} p={binomtest(dcr,n,0.5).pvalue:.3f}")
    # tails: how many members beat the 1st/5th percentile of control distance
    dm=-S["nn1_neg"][:n]; dcv=-S["nn1_neg"][n:]
    for q in [0.001,0.005,0.01,0.05]:
        t=np.quantile(dcv,q); print(f"    d<ctl q{q}: members {(dm<t).sum():4d} vs expected {q*n:.1f}")
    res[k]={"auc":row,"dcr":dcr/n}
json.dump(res,open("mia_auc2.json","w"),indent=1)
