"""Cost-only full-size fit probe. Repeated views are NOT experiment outputs."""
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import runtime as r
from generate_features import CACHE,VIEWCACHE,CONDITIONS
from run_family import dump
from pathlib import Path

def main():
    u=r.initialize();u.HERE=r.SRC;data=r.data_for(u,'prelim')
    f=r.import_path('benchmark_ablation',r.SRC/'特徴を外して学習を確認.py')
    raw=[np.load((CACHE/'prelim' if name=='full' else VIEWCACHE/'prelim'/name)/'代表特徴.npz')['scores'] for name,_,_ in CONDITIONS[:3]]
    x=np.stack([f.panel_features(raw[k%3],[j for j in range(12) if j!=1]) for k in range(47)])
    y=f.load_truth(data);valid=np.array([j for j,t in enumerate(data['teams']) if t not in (1,2)])
    fold=np.empty(875,int);fold[np.random.default_rng(20261008).permutation(875)]=np.arange(875)%5;trp=np.flatnonzero(fold!=0)
    tf=np.array([sorted(data['teams']).index(t)%4 for t in data['teams']]);rows=[]
    for label,trt in [('fourfold',valid[tf[valid]!=0]),('LOTO',valid[1:])]:
        start=time.perf_counter();train=x[:,trp,:,:][:,:,trt,:].reshape(-1,22);truth=np.tile(y[np.ix_(trp,trt)].ravel(),47);weight=np.repeat(np.full(47,1/47),len(trp)*len(trt))
        scaler=StandardScaler().fit(train,sample_weight=weight);fit=LogisticRegression(C=.1,class_weight='balanced',max_iter=1500);fit.fit(scaler.transform(train),truth,sample_weight=weight)
        rows.append(dict(split=label,seconds=time.perf_counter()-start,training_rows=len(train),features=22,iterations=int(fit.n_iter_[0]),max_iter=1500,views=47))
        print('MULTIVIEW_COST_PROBE',rows[-1],flush=True)
    dump(Path('/workspace/shared/mia_recovered_calibration/multiview_cost_probe.json'),{'rows':rows,'timing_only':True,'limitations':'47-view matrix size is real; three actually generated BG views repeated to fill47. This is a cost probe, not an original model result or accuracy comparison. Final fit timings may differ with full47-view diversity.'})

if __name__=='__main__':main()
