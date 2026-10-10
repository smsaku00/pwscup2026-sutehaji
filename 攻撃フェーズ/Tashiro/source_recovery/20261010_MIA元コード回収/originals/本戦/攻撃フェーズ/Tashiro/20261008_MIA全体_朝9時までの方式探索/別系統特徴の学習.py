"""別系統の公開特徴で順位を学習。対象1・2のラベルは一切学習しない。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import 共通 as u

SIGNALS=[
 ('距離と分布結果','q7_cat4_nnratio_k3__raw__cross0'),
 ('距離と分布結果','q7_robust_kde_ratio_h1__county500__cross0'),
 ('距離と分布結果','q7_gaussian_shrink0__county500__cross0'),
 ('距離と分布結果','q13_gaussian_shrink0.5__raw__cross0'),
 ('距離と分布結果','q13_hist10_s100_marginal__raw__cross0'),
 ('距離と分布結果','q13_robust_dcr_k1__raw__cross0'),
 ('距離と分布結果','q13_robust_nnratio_k10__raw__cross0'),
 ('直接密度比結果','q13_robust_ulsif_centers100_h1_a0.1_lambda1__raw__cross0'),
 ('密度比_属性乱数結果','all13_raw_mean__raw__cross0'),
 ('公開データ分類器結果','q7_robust_extra_leaf1__county500__cross0'),
 ('公開データ分類器結果','q7_rank_lr_C0.2__raw__cross0'),
 ('公開データ分類器結果','q9_robust_ocsvmC_gamma0.02__raw__cross0')]
GEN=[('距離と分布結果','gen7_reference__raw__cross0'),('距離と分布結果','gen7_reference__county500__cross0.1')]

def cache_features(data):
 caches={}
 for folder,_ in SIGNALS+GEN:
  if folder in caches:continue
  p=u.HERE/folder/'内部順位キャッシュ.npz';u.base.INPUT_HASHES[str(p.resolve())]=u.sha(p)
  z=np.load(p);assert list(z['teams'])==data['teams']
  assert np.array_equal(z['record_ids'],data['b'].loc[data['normal'],'record_id'].to_numpy())
  caches[folder]=(list(z['names']),z['scores'])
 signals=[]
 for folder,name in SIGNALS+GEN:
  names,bank=caches[folder];signals.append(bank[names.index(name)])
 # 1特徴の順位、他対象の中央値を同じ入力として渡す。IDとチーム番号は渡さない。
 raw=np.stack(signals,axis=2)
 lower=.5/data['normal'].sum()
 logits=logit(np.clip(raw,lower,1-lower))
 med=np.stack([np.column_stack([np.median(np.delete(raw[:,:,k],j,axis=1),axis=1) for j in range(len(data['teams']))]) for k in range(raw.shape[2])],axis=2)
 med=logit(np.clip(med,lower,1-lower))
 x=np.concatenate([logits,med],axis=2)
 names=[name for _,name in SIGNALS+GEN]+['other_median:'+name for _,name in SIGNALS+GEN]
 n=len(SIGNALS);total=raw.shape[2]
 panels={'nogen':list(range(n))+list(range(total,total+n)),
         'combined':list(range(x.shape[2])),
         'gen_only':[n,n+1,total+n,total+n+1]}
 return x,raw,names,panels

def models(seed):
 result=[]
 for strength in (.01,.1,1.):
  result.append((f'lr_C{strength:g}',make_pipeline(StandardScaler(),LogisticRegression(C=strength,class_weight='balanced',max_iter=1500)),False))
 for leaves in (7,15):
  result.append((f'hgb_leaves{leaves}',HistGradientBoostingClassifier(max_leaf_nodes=leaves,max_iter=150,min_samples_leaf=100,l2_regularization=10,learning_rate=.05,class_weight='balanced',early_stopping=False,random_state=seed),False))
 result.append(('extra_leaf50',ExtraTreesClassifier(n_estimators=150,min_samples_leaf=50,max_depth=8,max_features=.8,class_weight='balanced',n_jobs=2,random_state=seed),False))
 result.append(('tail_lr_C0.1',make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1500)),True))
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--seed',type=int,default=20261008)
 parser.add_argument('--output-dir',type=Path,default=u.HERE/'別系統特徴の学習結果')
 args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();x,raw,names,panels=cache_features(data)
 u.base.INPUT_HASHES[str(Path(__file__).resolve())]=u.sha(__file__)
 # 公開特徴を確定してから、モデル学習のための公開正解を読む。
 truthdir=u.base.child(u.base.child(u.ROOT/'予備戦','配布データ')/'結果','正解データ')
 own=u.base.read(truthdir/'pool_ids_5.csv').set_index('record_id').loc[data['b'].record_id]
 truth=np.column_stack([own.pool_id.isin(u.base.read(truthdir/f'pool_ids_{t}.csv').pool_id).to_numpy(int)[data['normal']] for t in data['teams']])
 rng=np.random.default_rng(args.seed);fold=np.empty(875,dtype=int)
 fold[rng.permutation(875)]=np.arange(875)%5
 # 対象の分割にも正解を使わない。1・2はすべての学習から外す。
 tfold=np.array([sorted(data['teams']).index(t)%4 for t in data['teams']])
 allowed=np.array([t not in (1,2) for t in data['teams']])
 predictions={'fixed_county500':raw[:,:,-1]};specs=[];splits=[];importances=[]
 predictions['fixed_nogen_mean']=np.mean(x[:,:,:len(SIGNALS)],axis=2)
 for panel,cols in panels.items():
  for name,_,tail in models(args.seed):
   label=f'{panel}_{name}';output=np.full(truth.shape,np.nan)
   for pf in range(5):
    for tf in range(4):
     trp=np.flatnonzero(fold!=pf);tep=np.flatnonzero(fold==pf)
     trt=np.flatnonzero((tfold!=tf)&allowed);tet=np.flatnonzero(tfold==tf)
     assert not np.intersect1d(trp,tep).size and not np.intersect1d(trt,tet).size
     assert all(data['teams'][i] not in (1,2) for i in trt)
     xx=x[np.ix_(trp,trt,cols)].reshape(-1,len(cols));yy=truth[np.ix_(trp,trt)].ravel()
     model=dict((n,m) for n,m,_ in models(args.seed+pf*4+tf))[name]
     weights=np.ones(len(yy))
     if tail:
      tail_feature=raw[np.ix_(trp,trt,np.arange(len(SIGNALS)))].max(axis=2).ravel()
      weights[(yy==0)&(tail_feature>=.95)]=5.
     if name.startswith(('lr','tail_lr')):model.fit(xx,yy,logisticregression__sample_weight=weights)
     else:model.fit(xx,yy,sample_weight=weights)
     test=x[np.ix_(np.arange(875),tet,cols)].reshape(-1,len(cols))
     score=model.predict_proba(test)[:,1].reshape(875,len(tet))
     ranked=np.column_stack([rankdata(s,method='average')/875 for s in score.T])
     assert np.isnan(output[np.ix_(tep,tet)]).all();output[np.ix_(tep,tet)]=ranked[tep]
     splits.append(dict(method=label,person_fold=pf,target_fold=tf,train_people=len(trp),test_people=len(tep),train_targets=';'.join(str(data['teams'][i]) for i in trt),test_targets=';'.join(str(data['teams'][i]) for i in tet),train_positive=int(yy.sum())))
     if name.startswith(('lr','tail_lr')):
      for col,coef in zip(cols,model.named_steps['logisticregression'].coef_[0]):importances.append(dict(method=label,person_fold=pf,target_fold=tf,feature=names[col],value=float(coef)))
   assert np.isfinite(output).all();predictions[label]=output
   specs.append(dict(method=label,family='membership_label_learning',panel=panel,input_features=';'.join(names[i] for i in cols),seed=args.seed,
                     source_selection='Exploratory representative signals chosen after first screen; held-out-score interpretation must account for prior use of this dataset.'))
   u.log(stage='new_learning_complete',method=label,seconds=time.perf_counter()-started)
 u.save(args.output_dir,data,predictions,specs,started,{'label_usage':'学習と採点。対象1・2のラベルは全学習分割から除外。人物5×対象4分割。',
                                                   'adaptive_features':True,'person_folds':5,'target_folds':4,'seed':args.seed})
 pd.DataFrame(splits).to_csv(args.output_dir/'交差検証分割.csv',index=False)
 pd.DataFrame(importances).to_csv(args.output_dir/'標準化後係数.csv',index=False,float_format='%.17g')
 rpath=args.output_dir/'実行記録.json';r=json.loads(rpath.read_text());r['label_usage']=r['extra']['label_usage']
 r['output_sha256']={p.name:u.sha(p) for p in args.output_dir.iterdir() if p.name!='実行記録.json'}
 rpath.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
