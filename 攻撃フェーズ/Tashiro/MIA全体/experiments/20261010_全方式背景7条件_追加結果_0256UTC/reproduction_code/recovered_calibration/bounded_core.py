"""Exact C-row blocking of original Gen-LRA intermediates, not a new score."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

def install(base):
    original=base.core
    def core(reference,candidates,synthetic,columns,ks=(1,3,10),whiten=False):
        if whiten:return original(reference,candidates,synthetic,columns,ks,whiten)
        scale=base.scales(reference,columns)
        r,b,c=[base.numbers(f,columns)/scale for f in (reference,candidates,synthetic)]
        h=(4./(len(columns)+2)/len(r))**(1./(len(columns)+4))
        distance=np.empty((len(b),len(c)));delta=np.empty_like(distance)
        freq={}
        for col in ('sex','prefecture'):
            categories=sorted(set(reference[col])|set(candidates[col])|set(synthetic[col]))
            freq[col]=(reference[col].value_counts().reindex(categories,fill_value=0)+1)/(len(reference)+len(categories))
        for start in range(0,len(c),128):
            stop=min(start+128,len(c));frame=synthetic.iloc[start:stop]
            cr=cdist(c[start:stop],r,metric='sqeuclidean');cb=cdist(c[start:stop],b,metric='sqeuclidean')
            d=cb.T.copy();log_cr=-.5*cr/h**2;log_cb=-.5*cb/h**2
            for col in ('sex','prefecture'):
                cf=frame[col].map(freq[col]).to_numpy()[:,None]
                eq_cr=frame[col].to_numpy()[:,None]==reference[col].to_numpy()[None,:]
                eq_cb=frame[col].to_numpy()[:,None]==candidates[col].to_numpy()[None,:]
                log_cr+=np.log(base.RHO*eq_cr+(1-base.RHO)*cf)
                log_cb+=np.log(base.RHO*eq_cb+(1-base.RHO)*cf)
                d+=~eq_cb.T
            baseline=logsumexp(log_cr,axis=1)
            delta[:,start:stop]=(np.logaddexp(0.,log_cb-baseline[:,None])-np.log1p(1/len(r))).T
            distance[:,start:stop]=d
        near=np.argsort(distance,axis=1,kind='stable')[:,:min(max(ks),len(c))]
        ordered=np.take_along_axis(delta,near,axis=1)
        output={k:ordered[:,:min(k,len(c))].sum(axis=1) for k in ks}
        assert all(np.isfinite(v).all() for v in output.values())
        return output
    base.core=core
    return original,core
