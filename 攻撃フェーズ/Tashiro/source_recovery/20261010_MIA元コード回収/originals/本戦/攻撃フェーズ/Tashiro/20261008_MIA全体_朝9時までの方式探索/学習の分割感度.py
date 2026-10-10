"""Gen-LRAを含まない学習の、人物・対象分割と他対象補正の感度。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import 共通 as u
from 別系統特徴の学習 import cache_features,SIGNALS

SEEDS=(20261008,20261009,20261010,20261011,20261012)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=u.HERE/'学習の分割感度結果')
 args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();x,raw,names,panels=cache_features(data)
 for p in [Path(__file__),u.HERE/'別系統特徴の学習.py']:u.base.INPUT_HASHES[str(p.resolve())]=u.sha(p)
 allowed=np.array([t not in (1,2) for t in data['teams']]);valid=np.flatnonzero(allowed)
 lower=.5/875;n=len(SIGNALS)
 x20=x[:,:,panels['nogen']]
 median18=np.stack([np.column_stack([np.median(raw[:,valid[valid!=j],k],axis=1) for j in range(len(data['teams']))]) for k in range(n)],axis=2)
 x18=np.concatenate([x[:,:,:n],logit(np.clip(median18,lower,1-lower))],axis=2)
 truthdir=u.base.child(u.base.child(u.ROOT/'予備戦','配布データ')/'結果','正解データ')
 own=u.base.read(truthdir/'pool_ids_5.csv').set_index('record_id').loc[data['b'].record_id]
 truth=np.column_stack([own.pool_id.isin(u.base.read(truthdir/f'pool_ids_{t}.csv').pool_id).to_numpy(int)[data['normal']] for t in data['teams']])
 tfold=np.array([sorted(data['teams']).index(t)%4 for t in data['teams']]);predictions={};specs=[];splits=[]
 for median_scope,array in [('median20',x20),('median18',x18)]:
  for target_split in ('fourfold','leave_one_target_out'):
   target_groups=[np.flatnonzero(tfold==tf) for tf in range(4)] if target_split=='fourfold' else [np.array([j]) for j in valid]+[np.flatnonzero(~allowed)]
   for strength in (.01,.1,1.):
    outputs=[]
    for seed in SEEDS:
     fold=np.empty(875,dtype=int);fold[np.random.default_rng(seed).permutation(875)]=np.arange(875)%5
     output=np.full(truth.shape,np.nan)
     label=f'{median_scope}_{target_split}_C{strength:g}_seed{seed}'
     for pf in range(5):
      for group_index,tet in enumerate(target_groups):
       tep=np.flatnonzero(fold==pf);trp=np.flatnonzero(fold!=pf);trt=np.setdiff1d(valid,tet)
       assert not np.intersect1d(trp,tep).size and not np.intersect1d(trt,tet).size
       assert all(data['teams'][j] not in (1,2) for j in trt)
       xx=array[np.ix_(trp,trt,np.arange(array.shape[2]))].reshape(-1,array.shape[2]);yy=truth[np.ix_(trp,trt)].ravel()
       fit=make_pipeline(StandardScaler(),LogisticRegression(C=strength,class_weight='balanced',max_iter=1500))
       fit.fit(xx,yy)
       test=array[np.ix_(np.arange(875),tet,np.arange(array.shape[2]))].reshape(-1,array.shape[2])
       decision=fit.decision_function(test).reshape(875,len(tet))
       ranked=np.column_stack([rankdata(col,method='average')/875 for col in decision.T])
       assert np.isnan(output[np.ix_(tep,tet)]).all();output[np.ix_(tep,tet)]=ranked[tep]
       if strength==.1:splits.append(dict(method=label,person_fold=pf,target_group=group_index,train_targets=';'.join(str(data['teams'][j]) for j in trt),test_targets=';'.join(str(data['teams'][j]) for j in tet),train_people=len(trp),test_people=len(tep),train_positive=int(yy.sum())))
     assert np.isfinite(output).all();predictions[label]=output;outputs.append(output)
     specs.append(dict(method=label,family='nogen_membership_learning',median_scope=median_scope,target_split=target_split,C=strength,seed=seed,features=len(SIGNALS)*2))
     u.log(stage='split_sensitivity',method=label,seconds=time.perf_counter()-started)
    draws=np.stack(outputs)
    prefix=f'{median_scope}_{target_split}_C{strength:g}'
    predictions[prefix+'_five_seed_mean']=draws.mean(axis=0)
    predictions[prefix+'_five_seed_median']=np.median(draws,axis=0)
    specs.extend([dict(method=prefix+'_five_seed_'+r,family='nogen_membership_learning_consensus',seeds=';'.join(map(str,SEEDS))) for r in ('mean','median')])
 u.save(args.output_dir,data,predictions,specs,started,{'label_usage':'学習と採点。全分割で対象1・2のラベルとテスト人物・テスト対象のラベルを学習から除外。','adaptive_followup':True,'seeds':SEEDS,'features':'Gen-LRAを含まない12系統×自身順位・他対象中央値。'})
 pd.DataFrame(splits).to_csv(args.output_dir/'交差検証分割.csv',index=False)
 rpath=args.output_dir/'実行記録.json';r=json.loads(rpath.read_text());r['label_usage']=r['extra']['label_usage']
 r['output_sha256']={p.name:u.sha(p) for p in args.output_dir.iterdir() if p.name!='実行記録.json'}
 rpath.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
