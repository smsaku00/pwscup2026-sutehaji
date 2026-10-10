"""Public feature extraction, retained byte-for-byte from the validated engine.

Only 35 named columns are passed to LR; other computed columns are diagnostics.
No private main-round B records or main-round truth are read here.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata
CLIN=['SBP','TG','HDL','ALT']
TEAMS=[1,2,3,4,5,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]


def vtrans(df,cols=CLIN):
    x=df[list(cols)].to_numpy(dtype=float).copy()
    for n in ('TG','ALT'):
        if n in cols: x[:,list(cols).index(n)] = np.log1p(np.maximum(0,x[:,list(cols).index(n)]))
    return x

def near_1d(s,v):
    s=np.asarray(s,float)
    if not len(s): return np.full(len(v),1e7)
    s=np.sort(s)
    k=np.searchsorted(s,v,side='left')
    out=np.full(len(v),np.inf)
    ok=k<len(s); out[ok]=np.abs(v[ok]-s[k[ok]])
    ok=k>0;out[ok]=np.minimum(out[ok],np.abs(v[ok]-s[k[ok]-1]))
    return out

def k_neighbors(src,dest,k=10):
    if len(src)==0: return np.full((len(dest),k),1e4)
    dis,_=cKDTree(src).query(dest,k=min(k,len(src)),workers=1)
    dis=np.asarray(dis,float)
    if dis.ndim==1:dis=dis[:,None]
    if dis.shape[1]<k:dis=np.pad(dis,((0,0),(0,k-dis.shape[1])),constant_values=1e4)
    return dis

def c_partial2(q5,c5,sexQ,sexC):
    out=np.full(len(q5),1e5)
    for s in (0,1):
        iq=np.where(sexQ==s)[0];ic=np.where(sexC==s)[0]
        if not len(iq) or not len(ic):continue
        # For each donor, keep the two smallest of age and clinical squared deviations.
        p=((q5[iq,None,:]-c5[None,ic,:])**2)
        d=np.partition(p,1,axis=2)[:,:,:2].mean(axis=2)
        out[iq]=np.sqrt(d.min(axis=1))
    return out

def feature_table(bg,D):
    b=vtrans(bg)
    iqr=np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0),1e-5)
    age_scale=max(float(np.quantile(bg.age,.75)-np.quantile(bg.age,.25)),1e-5)
    otherpool={}
    # Same public C across different target datasets; stable subsampling.
    for k,(q,c,t) in D.items():otherpool[k]=c.iloc[::6,:].copy()
    sets={};labels={};
    for k,(q,c,t) in D.items():
        x=vtrans(q);cx=vtrans(c)
        c5=np.column_stack((c.age.to_numpy(float)/age_scale,cx/iqr))
        q5=np.column_stack((q.age.to_numpy(float)/age_scale,x/iqr))
        bx=np.column_stack((bg.age.to_numpy(float)/age_scale,b/iqr))
        same=q.sex.to_numpy(int); cs=c.sex.to_numpy(int); bs=bg.sex.to_numpy(int)
        x4=x/iqr;cx4=cx/iqr;b4=b/iqr
        dC=np.full((len(q),10),1e4);dB=dC.copy();dO=dC.copy()
        dC5=np.full(len(q),1e4);dB5=dC5.copy()
        pool=pd.concat([s for j,s in otherpool.items() if j!=k],ignore_index=True)
        ox=vtrans(pool)/iqr; os=pool.sex.to_numpy(int)
        for s in [0,1]:
            ii=np.where(same==s)[0]
            if not len(ii):continue
            dC[ii]=k_neighbors(cx4[cs==s],x4[ii],10)
            dB[ii]=k_neighbors(b4[bs==s],x4[ii],10)
            dO[ii]=k_neighbors(ox[os==s],x4[ii],10)
            dC5[ii]=k_neighbors(c5[cs==s],q5[ii],1)[:,0]
            dB5[ii]=k_neighbors(bx[bs==s],q5[ii],1)[:,0]
        colC=np.zeros((len(q),4));colB=colC.copy()
        for s in (0,1):
            ii=np.where(same==s)[0]
            if not len(ii):continue
            for j in range(4):
                colC[ii,j]=near_1d(cx[cs==s,j]/iqr[j],x[ii,j]/iqr[j])
                colB[ii,j]=near_1d(b[bs==s,j]/iqr[j],x[ii,j]/iqr[j])
        colEvidence=np.log((colB+0.001)/(colC+0.001))
        # Distinct positive and negative evidence: 1-NN C and public background / other Cs.
        part2=c_partial2(q5,c5,same,cs)
        log=lambda x:np.log1p(np.maximum(x,0))
        f=pd.DataFrame({
            'member_near4':-log(dC[:,0]),'member_knn4':-log(dC[:,4]),
            'member_near5':-log(dC5),'member_partial2':-log(part2),
            'bg_near4':-log(dB[:,0]),'bg_knn4':-log(dB[:,4]),
            'bg_near5':-log(dB5),'otherc_near4':-log(dO[:,0]),
            'contrast_near4':log(dB[:,0])-log(dC[:,0]),
            'contrast_knn4':log(dB[:,4])-log(dC[:,4]),
            'contrast_near5':log(dB5)-log(dC5),
            'contrast_otherc':log(dO[:,0])-log(dC[:,0]),
            'col4':np.mean(colEvidence,axis=1),
            'col2':np.sort(colEvidence,axis=1)[:,-2:].mean(axis=1),
            'col1':colEvidence.max(axis=1),
            'col4_min':colEvidence.min(axis=1),
            'age':q.age.to_numpy(float), 'sex':same.astype(float),
            'raw_sbp':x[:,0]/iqr[0], 'raw_tg':x[:,1]/iqr[1],
            'raw_hdl':x[:,2]/iqr[2], 'raw_alt':x[:,3]/iqr[3],
        })
        for j,n in enumerate(CLIN):
            f[f'c_{n.lower()}']=-log(colC[:,j]);f[f'bg_{n.lower()}']=-log(colB[:,j]);f[f'ratio_{n.lower()}']=colEvidence[:,j]
        base_cols=['member_near4','member_knn4','member_near5','member_partial2',
                   'bg_near4','bg_knn4','contrast_near4','contrast_knn4','contrast_near5',
                   'contrast_otherc','col4','col2','col1']
        for n in base_cols:
            f['rank_'+n]=rankdata(f[n].to_numpy(),method='average')/len(f)
        f['team']=k;f['row_id']=q.challenge_row_id.to_numpy(int)
        assert np.isfinite(f.to_numpy(float)).all()
        sets[k]=f;labels[k]=t
    return sets,labels

def pairs(q,w):
    d={}
    for i,r in q.iterrows():
        k=(str(r.prefecture),int(r.sex),int(r.age//w))
        d.setdefault(k,[]).append(i)
    return d
