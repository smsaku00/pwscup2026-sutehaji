import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/taisei1105/pws2026-local/work")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from harness import Bench
from submit_audit import COLS13, CONT, _z, matched_controls
import attacks as AT
DIST=Path("/Users/taisei1105/pws2026-local/予備戦/配布データ/PWSCup2026_prelim_attack")
b=Bench(Path("/Users/taisei1105/pws2026-local/work/data_prelim_team05"))
B=b.B[COLS13].reset_index(drop=True); A=b.A_bg.reset_index(drop=True)
rare=b.is_rare
print("rare total=%d ; rare&Tokyo=%d ; rare&nonTokyo=%d"%(rare.sum(),
   ((B.prefecture=="東京都").to_numpy()&rare).sum(), ((B.prefecture!="東京都").to_numpy()&rare).sum()))
mean=A[CONT].mean().to_numpy(float); sd=A[CONT].std(ddof=0).to_numpy(float); sd=np.where(sd>0,sd,1.0)
OUT=CONT+["time"]
mo=A[OUT].mean().to_numpy(float); so=A[OUT].std(ddof=0).to_numpy(float); so=np.where(so>0,so,1.0)
def auc(x,y):
    r=pd.Series(np.r_[x,y]).rank().to_numpy(); n1=len(x)
    return (r[:n1].sum()-n1*(n1+1)/2)/(n1*len(y))
def kset(df): return set(map(tuple,np.round(df[CONT].to_numpy(float),9)))
SEEDS=[0,1,2,3,4]
res={}
for k in [19,20,22,23]:
    C=pd.read_csv(DIST/f"C_{k}.csv"); kc=kset(C)
    clean=A[[tuple(t) not in kc for t in np.round(A[CONT].to_numpy(float),9)]].reset_index(drop=True)
    print(f"\n########## C_{k}  (A_bg rows removed as contaminated: {len(A)-len(clean)}) ##########")
    nnC=NearestNeighbors(n_neighbors=1).fit(_z(C,CONT,mean,sd))
    nnCo=NearestNeighbors(n_neighbors=1).fit(_z(C,OUT,mo,so))
    def feats(X):
        f={}
        f["d1"]=nnC.kneighbors(_z(X,CONT,mean,sd))[0][:,0]
        f["d1o"]=nnCo.kneighbors(_z(X,OUT,mo,so))[0][:,0]
        f["nngap"]=AT.m3_nn_gap(C,A,X)
        f["knn"]=AT.m4_knn_density(C,A,X)
        f["domias"]=AT.m1_domias(C,A,X)
        # within (sex,prefecture) block NN on 5 QI
        QI=["age","SBP","TG","HDL","ALT"]
        mq=A[QI].mean().to_numpy(float); sq=A[QI].std(ddof=0).to_numpy(float); sq=np.where(sq>0,sq,1.0)
        dd=np.full(len(X),np.inf)
        for key,idx in C.groupby(["sex","prefecture"]).groups.items():
            m=((X["sex"]==key[0])&(X["prefecture"]==key[1])).to_numpy()
            if m.sum()==0 or len(idx)==0: continue
            dd[m]=NearestNeighbors(n_neighbors=1).fit((C.loc[idx,QI].to_numpy(float)-mq)/sq
                 ).kneighbors((X.loc[m,QI].to_numpy(float)-mq)/sq)[0][:,0]
        f["dblk"]=np.where(np.isfinite(dd),dd,np.nanmax(dd[np.isfinite(dd)]))
        return f
    fb=feats(B)
    rows={}
    ctlf=[]
    for s in SEEDS:
        R=matched_controls(B,clean,seed=s)[COLS13].reset_index(drop=True)
        ctlf.append(feats(R))
    for name in fb:
        sign=-1 if name in ("d1","d1o","dblk") else 1
        a=[auc(sign*fb[name], sign*cf[name]) for cf in ctlf]
        # top-26 enrichment: pool B + one control set, count B in top26
        enr=[]
        for cf in ctlf:
            v=np.r_[sign*fb[name], sign*cf[name]]
            top=np.argsort(-v)[:26]
            enr.append(int((top<len(B)).sum()))
        print("  %-8s AUC=%.4f+-%.4f   top26 B-count=%s (chance 13)"%(name,np.mean(a),np.std(a),enr))
        rows[name]=dict(auc=float(np.mean(a)),auc_sd=float(np.std(a)),top26=enr)
    # tail: how far are the closest B rows vs closest control rows
    print("  closest 8 B  d1 :",np.round(np.sort(fb["d1"])[:8],4))
    print("  closest 8 ctl d1:",np.round(np.sort(ctlf[0]["d1"])[:8],4))
    print("  closest 8 B  dblk:",np.round(np.sort(fb["dblk"])[:8],4))
    print("  closest 8 ctl dblk:",np.round(np.sort(ctlf[0]["dblk"])[:8],4))
    res[k]=rows
json.dump(res,open("/Users/taisei1105/pws2026-local/work/atk19/mia.json","w"),indent=1)
