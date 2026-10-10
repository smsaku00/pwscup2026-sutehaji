"""Round-5 LR10 / RF top26 / soft rarity-limited group correction."""
from pathlib import Path
import math
import numpy as np,pandas as pd
from scipy.special import expit,logit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import feature_engine as old
CLIN=old.CLIN
MEMBER=['member_near4','member_knn4','member_near5','member_partial2','rank_member_near4','rank_member_near5','rank_member_partial2']+[f'c_{c.lower()}' for c in CLIN]
BGC=['bg_near4','bg_knn4','bg_near5','rank_bg_near4','rank_bg_knn4']+[f'bg_{c.lower()}' for c in CLIN]
RATIO=['contrast_near4','contrast_knn4','contrast_near5','col4','col2','col1','col4_min','rank_contrast_near4','rank_col4','rank_col2','rank_col1']+[f'ratio_{c.lower()}' for c in CLIN]
COLS=MEMBER+BGC+RATIO

def clinical(q):
    x=q[['age','SBP','TG','HDL','ALT','sex']].copy()
    for c in ['TG','ALT']:x[c]=np.log1p(x[c])
    return x.to_numpy()

def read_phase(root,phase):
    d=root/('tmp/aia/05' if phase=='prelim' else 'tmp/main_attack')
    bg=pd.read_csv(root/('tmp/aia/02/A_bg.csv' if phase=='prelim' else 'tmp/main_team28/A_bg.csv'))
    data={int(f.stem.rsplit('_',1)[1]):(pd.read_csv(f).sort_values('challenge_row_id').reset_index(drop=True),pd.read_csv(f.parent/f'C_{int(f.stem.rsplit("_",1)[1])}.csv'),pd.DataFrame()) for f in d.glob('aia_challenge_*.csv')}
    return bg,dict(sorted(data.items()))

def rare_mask(q,s):
    m=np.zeros(len(q),bool);m[np.lexsort((q.challenge_row_id.to_numpy(),-np.asarray(s)))[:26]]=True;return m

def grouping(q,mask):
    return [np.asarray([i for i in g if mask[i]],int) for g in old.pairs(q,5).values() if any(mask[i] for i in g)]

def binary(q,s,fraction=.75,off=11):
    mask=np.zeros(len(q),bool)
    for sex in sorted(q.sex.unique()):
        ix=np.flatnonzero(q.sex.to_numpy()==sex)
        order=np.lexsort((q.challenge_row_id.to_numpy()[ix],-np.asarray(s)[ix]))
        mask[ix[order[:math.ceil(fraction*len(ix))]]]=True
    return np.where(mask,9.5,off).astype(float)

def model_prob(trainX,y,testX,kind):
    if kind not in ['lr0.04','lr10']:raise ValueError('Round5 uses only the frozen two LR models')
    model=make_pipeline(StandardScaler(),LogisticRegression(C=float(kind[2:]),max_iter=1500))
    model.fit(trainX,y)
    return model.predict_proba(testX)[:,1]

def rare_modify(q,base,mask,p):
    # Balance each estimated rare even-size group softly, never force labels.
    result=base.copy()
    for g in grouping(q,mask):
        if len(g)<2 or len(g)%2:continue
        z=logit(np.clip(p[g],1e-6,1-1e-6));lo,hi=-30.,30.
        for _ in range(60):
            mid=(lo+hi)/2
            if expit(z+mid).mean()<.5:lo=mid
            else:hi=mid
        drop=g[expit(z+.5*(lo+hi)/2)<.5]
        result[drop]=11
    assert np.array_equal(result[~mask],base[~mask])
    return result
