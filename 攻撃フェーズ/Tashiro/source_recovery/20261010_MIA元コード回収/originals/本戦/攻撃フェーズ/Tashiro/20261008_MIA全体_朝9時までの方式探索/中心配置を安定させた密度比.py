"""ランダム中心への依存を除き、公開C/Aの交差検証で密度比を選ぶ。"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import 共通 as u
from 距離と分布の探索 import transform

WIDTHS=(.5,1.,2.)
REGS=(.01,.1,1.)

def embed(a,c,query,nums,cats):
 ar,cr,x=transform(a,[a,c,query],nums,'robust')
 frames=[a,c,query];arrays=[ar,cr,x]
 for col in cats:
  levels=np.array(sorted(set(a[col])|set(c[col])|set(query[col])))
  for i,frame in enumerate(frames):arrays[i]=np.column_stack([arrays[i],(frame[col].to_numpy()[:,None]==levels[None,:])/np.sqrt(2)])
 return arrays

def centers(cr,style,seed):
 if style=='all':return cr
 if style=='kmeans100':return KMeans(n_clusters=100,n_init=3,max_iter=200,random_state=seed).fit(cr).cluster_centers_
 # 最大最小距離でCの広い領域を覆う。初点は平均に最も近いC。
 distance=cdist(cr,cr,'sqeuclidean');first=int(np.argmin(((cr-cr.mean(axis=0))**2).sum(axis=1)))
 ids=[first];best=distance[:,first].copy()
 for _ in range(99):
  chosen=int(np.argmax(best));ids.append(chosen);best=np.minimum(best,distance[:,chosen])
 return cr[ids]

def fit_scores(ar,cr,query,z,check=None):
 da=cdist(ar,z,'sqeuclidean');dc=cdist(cr,z,'sqeuclidean');dx=cdist(query,z,'sqeuclidean')
 result={};risks={}
 for width in WIDTHS:
  ka=np.exp(-.5*da/width**2);kc=np.exp(-.5*dc/width**2);kx=np.exp(-.5*dx/width**2)
  matrix=.9*(ka.T@ka/len(ar))+.1*(kc.T@kc/len(cr));mean=kc.mean(axis=0)
  eigen,vectors=np.linalg.eigh(matrix);assert eigen.min()>-1e-9
  projected=vectors.T@mean
  for reg in REGS:
   strength=reg*len(z)/100
   theta=vectors@(projected/(np.maximum(eigen,0)+strength));beta=np.maximum(0,theta)
   assert np.linalg.norm((matrix+strength*np.eye(len(z)))@theta-mean)<1e-6
   value=kx@beta;assert np.isfinite(value).all();result[(width,reg)]=value
   if check:
    na,nc=check;va=value[:na];vc=value[na:na+nc]
    risks[(width,reg)]=float(.5*(.9*np.mean(va**2)+.1*np.mean(vc**2))-np.mean(vc))
 return result,risks

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=u.HERE/'密度比_中心配置結果')
 args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();banks={};specs=[];selection=[]
 for p in [Path(__file__),u.HERE/'距離と分布の探索.py']:u.base.INPUT_HASHES[str(p.resolve())]=u.sha(p)
 for attributes,nums,cats in [('q7',u.NUM7,u.CAT7),('q13',u.NUM13,u.CAT13)]:
  query=data['query'] if attributes=='q7' else data['b'];calibration=attributes=='q7'
  for index,(t,c) in enumerate(data['cs'].items(),1):
   ar,cr,x=embed(data['a'],c,query,nums,cats)
   rng=np.random.default_rng(20261018+t)
   ia=rng.permutation(len(ar));ic=rng.permutation(len(cr));cut_a=int(.8*len(ar));cut_c=int(.8*len(cr))
   train_a,test_a=ar[ia[:cut_a]],ar[ia[cut_a:]];train_c,test_c=cr[ic[:cut_c]],cr[ic[cut_c:]]
   scores={}
   for style in ('farthest100','kmeans100','all'):
    cvquery=np.vstack([test_a,test_c]);cvcenters=centers(train_c,style,20261018+t)
    _,risks=fit_scores(train_a,train_c,cvquery,cvcenters,(len(test_a),len(test_c)))
    choice=min(risks,key=lambda k:(risks[k],k))
    fitted,_=fit_scores(ar,cr,x,centers(cr,style,20261018+t))
    for (width,reg),value in fitted.items():scores[f'{attributes}_{style}_h{width:g}_reg{reg:g}']=value
    scores[f'{attributes}_{style}_publicCV']=fitted[choice]
    selection.append(dict(team=t,attributes=attributes,style=style,selected_width=choice[0],selected_reg=choice[1],cv_risk=risks[choice],validation_A=len(test_a),validation_C=len(test_c),membership_labels_used=False))
   for name,value in u.variants(scores,data,t,calibration).items():banks.setdefault(name,{})[t]=value
   if index==1:
    for name in scores:specs.append(dict(raw_method=name,family='stable_basis_direct_ratio',attributes=attributes,relative=.1,regularizer_scaling='lambda=reg*number_of_centers/100',uses_public_null=calibration,
                                      source_selection='publicCV minimizes held-out row-source relative density-ratio quadratic risk; no membership labels'))
   u.log(stage='stable_basis',attributes=attributes,completed_targets=index,targets=len(data['cs']),seconds=time.perf_counter()-started)
 predictions=u.matrices(banks,data['teams'])
 u.log(stage='predictions_fixed',raw_specs=len(specs),models=len(predictions),seconds=time.perf_counter()-started)
 u.save(args.output_dir,data,predictions,specs,started,{'adaptive_followup':True,'row_source_holdout':.2,'membership_labels_in_model_selection':False})
 __import__('pandas').DataFrame(selection).to_csv(args.output_dir/'公開データでの設定選択.csv',index=False,float_format='%.17g')

if __name__=='__main__':main()
