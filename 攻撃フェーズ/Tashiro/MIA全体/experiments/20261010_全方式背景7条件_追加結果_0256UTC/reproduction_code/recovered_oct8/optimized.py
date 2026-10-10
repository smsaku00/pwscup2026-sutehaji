"""Exact row-blocking and target-independent denominator reuse; no grid pruning."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import 距離と分布の探索 as geo

ORIGINAL_GEOMETRY=geo.geometry
DEN={}

def clear():DEN.clear()

def mixed_chunked(x,r,query,ref,cats,a,h,block=256):
    # Reductions still span all reference columns in their original order.
    # Only independent query rows are batched.
    out=np.empty(len(query))
    for start in range(0,len(query),block):
        stop=min(start+block,len(query))
        out[start:stop]=geo.mixed_logs(cdist(x[start:stop],r,'sqeuclidean'),query.iloc[start:stop],ref,cats,a,h)
    return out

def geometry(a,query,c,nums,cats,kind,weight,ref_near):
    ar,x,cr=geo.transform(a,[a,query,c],nums,kind)
    dnum=cdist(x,cr,'sqeuclidean');dc=dnum.copy()
    for col in cats:dc+=weight*(query[col].to_numpy()[:,None]!=c[col].to_numpy()[None,:])
    if kind=='sex':dc[query.sex.to_numpy()[:,None]!=c.sex.to_numpy()[None,:]]=1e12
    nc=geo.nearest(dc,30);result={}
    for k in geo.KS:
        cm=nc[:,:k].mean(axis=1)
        result[f'dcr_k{k}']=-np.log(cm+1e-12)
        result[f'nnratio_k{k}']=np.log(ref_near[:,:k].mean(axis=1)+1e-12)-np.log(cm+1e-12)
        result[f'nnratio_mass_k{k}']=np.log(ref_near[:,:min(150,5*k)].mean(axis=1)+1e-12)-np.log(cm+1e-12)
    if weight==1 and kind in ('robust','rank'):
        h0=(4/(len(nums)+2)/len(a))**(1/(len(nums)+4))
        for multiplier in (.5,1.,2.):
            h=h0*multiplier;key=(id(a),id(query),tuple(nums),tuple(cats),kind,h)
            if key not in DEN:DEN[key]=mixed_chunked(x,ar,query,a,cats,a,h)
            lc=geo.mixed_logs(dnum,query,c,cats,a,h);la=DEN[key]
            result[f'kde_ratio_h{multiplier:g}']=lc-la
            result[f'kde_witness_h{multiplier:g}']=np.exp(lc)-np.exp(la)
    return result

def anchor_scales(ar,ks,dimension,block=128):
    result={k:np.empty(len(ar)) for k in ks};maxk=max(ks)
    for start in range(0,len(ar),block):
        stop=min(start+block,len(ar));own=cdist(ar[start:stop],ar,'sqeuclidean')
        own[np.arange(stop-start),np.arange(start,stop)]=np.inf
        near=geo.nearest(own,maxk)
        for k in ks:result[k][start:stop]=np.maximum(near[:,k-1]/dimension,1e-8)
    return result

def adaptive(x,ar,query,anchors,a,cats,scales,multiplier,dimension,block=256):
    scale=scales*multiplier**2;out=np.empty(len(query))
    for start in range(0,len(query),block):
        stop=min(start+block,len(query));q=query.iloc[start:stop]
        lp=-.5*cdist(x[start:stop],ar,'sqeuclidean')/scale[None,:]-.5*dimension*np.log(scale)[None,:]
        for col in cats:
            levels=sorted(set(a[col])|set(anchors[col])|set(query[col]));p=(a[col].value_counts().reindex(levels,fill_value=0)+1)/(len(a)+len(levels))
            lp+=np.log(.98*(q[col].to_numpy()[:,None]==anchors[col].to_numpy()[None,:])+.02*q[col].map(p).to_numpy()[:,None])
        out[start:stop]=logsumexp(lp,axis=1)-np.log(len(anchors))
    return out
