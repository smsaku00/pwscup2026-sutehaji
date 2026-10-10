import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
W = Path("/Users/taisei1105/pws2026-local/work"); sys.path.insert(0,str(W))
import attacks as AT
from submit_audit import matched_controls
from audit import _tpr_at
DIST = Path("/Users/taisei1105/pws2026-local/予備戦/配布データ/PWSCup2026_prelim_attack")
COLS13=["age","sex","prefecture","BMI","SBP","TG","HDL","ALT","smoking","FPG","time","onset","death"]
CONT=AT.CONT
Bs=pd.read_csv(W/"data_prelim_team05"/"B_self.csv"); IS_RARE=Bs["is_rare"].to_numpy().astype(bool)
B=Bs[COLS13].reset_index(drop=True); A=pd.read_csv(W/"data_prelim_team05"/"A_bg.csv")[COLS13]
mean=A[CONT].mean().to_numpy(float); std=A[CONT].std(ddof=0).to_numpy(float)
def z(df,cols=CONT): return (df[cols].to_numpy(float)-mean if cols is CONT else df[cols].to_numpy(float))
def zc(df): return (df[CONT].to_numpy(float)-mean)/std
rng=np.random.default_rng(0)

def tpr_at_op(score_m, score_all_non, n_mem=26, reps=400, fpr=0.01):
    """official op point: 1049 candidates with n_mem members. draw members from B-side scores,
       non-members = remaining B rows. Here we simulate: members = random n_mem of the B rows
       replaced by 'member-like' distribution? Not possible without labels."""
    return None

res={}
for k in [24,26,29]:
    C=pd.read_csv(DIST/f"C_{k}.csv")
    cz=zc(C); nnc=NearestNeighbors(n_neighbors=1).fit(cz)
    dA,_=nnc.kneighbors(zc(A)); clean=dA[:,0]>0.05
    Aclean=A[clean].reset_index(drop=True)
    print("="*74); print(f"TEAM {k}  clean-A pool {len(Aclean)}")
    aucs={}
    for seed in range(5):
        ctl=matched_controls(B,Aclean,seed=seed)
        X=pd.concat([B,ctl],ignore_index=True); y=np.r_[np.ones(len(B)),np.zeros(len(ctl))]
        sc={}
        sc["nn_gap"]=AT.m3_nn_gap(C,Aclean,X)
        sc["knn15"]=AT.m4_knn_density(C,Aclean,X,k=15)
        sc["knn3"]=AT.m4_knn_density(C,Aclean,X,k=3)
        sc["domias"]=AT.m1_domias(C,Aclean,X)
        # outcome-aware distance (cont7 + time)
        C8=C.copy(); X8=X.copy()
        m8=np.r_[mean,[A["time"].mean()]]; s8=np.r_[std,[A["time"].std(ddof=0)]]
        cols8=CONT+["time"]
        cz8=(C8[cols8].to_numpy(float)-m8)/s8; xz8=(X8[cols8].to_numpy(float)-m8)/s8
        d8=NearestNeighbors(n_neighbors=1).fit(cz8).kneighbors(xz8)[0][:,0]
        sc["nn8"]=-d8
        sc["nn_raw"]=-NearestNeighbors(n_neighbors=1).fit(cz).kneighbors(zc(X))[0][:,0]
        for nm,v in sc.items():
            aucs.setdefault(nm,[]).append(roc_auc_score(y,v))
    for nm,v in aucs.items():
        v=np.array(v); print(f"  AUC {nm:8s} = {v.mean():.4f} +- {v.std():.4f}   (0.5=no leak)")
    # rare-only AUC
    ctl=matched_controls(B,Aclean,seed=0)
    Br=B[IS_RARE].reset_index(drop=True); Cr=ctl[IS_RARE].reset_index(drop=True)
    Xr=pd.concat([Br,Cr],ignore_index=True); yr=np.r_[np.ones(len(Br)),np.zeros(len(Cr))]
    s=-NearestNeighbors(n_neighbors=1).fit(cz).kneighbors(zc(Xr))[0][:,0]
    print(f"  rare24 AUC(nn_raw) = {roc_auc_score(yr,s):.4f}  (n=24 vs 24)")
    s2=AT.m4_knn_density(C,Aclean,Xr,k=3); print(f"  rare24 AUC(knn3)   = {roc_auc_score(yr,s2):.4f}")
