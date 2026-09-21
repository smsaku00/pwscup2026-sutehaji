"""Training-targeted AIA decoy experiment. Not a privacy guarantee."""
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist

def decoy_swaps(c,b,a,seed=28707,proposals=6000):
 c=c.copy().reset_index(drop=True);b=b.reset_index(drop=True)
 cols=['age','SBP','TG','HDL','ALT'];sd=a[cols].std().to_numpy()
 dist=cdist(b[cols]/sd,c[cols]/sd,'sqeuclidean')
 labels=c.prefecture.to_numpy().copy();sex=c.sex.to_numpy();times=c.time.to_numpy();truth=b.time.to_numpy()
 query={key:np.asarray(ix) for key,ix in b.groupby(['sex','prefecture']).indices.items()}
 def loss(key):
  ids=query.get(key,np.array([],int))
  if not len(ids):return 0
  ci=np.flatnonzero((sex==key[0])&(labels==key[1]))
  if not len(ci):ci=np.flatnonzero(sex==key[0])
  near=ci[np.argmin(dist[np.ix_(ids,ci)],axis=1)]
  return int((abs(times[near]-truth[ids])<=.5).sum())
 losses={key:loss(key) for key in query};start=sum(losses.values());rng=np.random.default_rng(seed);accepted=0
 # Preserve sex × prefecture × onset × death counts; only QI combinations change.
 pools=list(c.groupby(['sex','onset','death']).indices.values())
 pools=[np.asarray(x) for x in pools if len(x)>1]
 for _ in range(proposals):
  pool=pools[rng.integers(len(pools))];i,j=rng.choice(pool,2,replace=False)
  if labels[i]==labels[j]:continue
  keys=[(sex[i],labels[i]),(sex[j],labels[j])];before=sum(losses.get(k,0) for k in keys)
  labels[i],labels[j]=labels[j],labels[i];after=[loss(k) for k in keys]
  if sum(after)<before:
   losses.update(dict(zip(keys,after)));accepted+=1
  else:labels[i],labels[j]=labels[j],labels[i]
 c['prefecture']=labels
 return c,{'proposals':proposals,'accepted':accepted,'training_hits_before':start,'training_hits_after':sum(losses.values()),'training_n':len(b)}

