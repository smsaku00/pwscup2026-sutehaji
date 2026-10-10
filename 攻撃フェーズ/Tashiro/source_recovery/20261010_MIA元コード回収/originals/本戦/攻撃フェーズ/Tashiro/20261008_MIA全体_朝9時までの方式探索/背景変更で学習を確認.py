"""固定した別系統学習を複数の公開背景で追試し、条件ごとに記録する。"""
from __future__ import annotations
import argparse,json,os,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import 共通 as u
from 代表特徴を計算 import calculate,learning_features

SEEDS=(20261008,20261009,20261010,20261011,20261012)
DEADLINE=datetime.fromisoformat('2026-10-08T08:40:00+09:00')

def predict(array,fit,fold,tet,pf,output):
 decision=fit.decision_function(array[:,tet,:].reshape(-1,array.shape[2])).reshape(875,len(tet))
 ranked=np.column_stack([rankdata(col,method='average')/875 for col in decision.T])
 people=np.flatnonzero(fold==pf);assert np.isnan(output[np.ix_(people,tet)]).all()
 output[np.ix_(people,tet)]=ranked[people]

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=u.HERE/'背景変更の学習結果')
 args=parser.parse_args();assert not args.output_dir.exists();args.output_dir.mkdir()
 started=time.perf_counter();data=u.setup();teams=data['teams']
 u.base.INPUT_HASHES[str(Path(__file__).resolve())]=u.sha(__file__)
 # 標準の6条件を先に確認し、その後で新しい乱数・比率も追加する。
 conditions=[('full',1.,None)]+[(f'fraction{f:g}_seed{s}',f,s) for f in (.9,.8,.5) for s in (20261007,20261008)]
 conditions += [(f'fraction{f:g}_seed{s}',f,s) for f in (.95,.9,.8,.7,.5) for s in range(20261009,20261017)]
 assert len({c[0] for c in conditions})==len(conditions)
 policy=dict(pid=os.getpid(),started_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),conditions=conditions,deadline_jst=DEADLINE.isoformat(),
             code_sha256=u.sha(__file__),training_labels_exclude=[1,2],person_folds=5,target_folds=4,seeds=SEEDS,regularization=(.01,.1,1.),
             interpretation='同じ母集団の背景変更。独立したコホートではない。初回学習結果を見た後の追試。')
 (args.output_dir/'実験方針.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
 fullpath=u.HERE/'代表特徴の再現結果/代表特徴.npz'
 assert fullpath.exists(),'先に代表特徴の再現確認を完了する。'
 u.base.INPUT_HASHES[str(fullpath.resolve())]=u.sha(fullpath)
 z=np.load(fullpath);assert list(z['teams'])==teams;fullraw=z['scores'];fullx=learning_features(fullraw)
 truthdir=u.base.child(u.base.child(u.ROOT/'予備戦','配布データ')/'結果','正解データ')
 own=u.base.read(truthdir/'pool_ids_5.csv').set_index('record_id').loc[data['b'].record_id]
 truth=np.column_stack([own.pool_id.isin(u.base.read(truthdir/f'pool_ids_{t}.csv').pool_id).to_numpy(int)[data['normal']] for t in teams])
 allowed=np.array([t not in (1,2) for t in teams]);tfold=np.array([sorted(teams).index(t)%4 for t in teams])
 folds={};fullfits={}
 for seed in SEEDS:
  fold=np.empty(875,dtype=int);fold[np.random.default_rng(seed).permutation(875)]=np.arange(875)%5;folds[seed]=fold
  for strength in (.01,.1,1.):
   for pf in range(5):
    for tf in range(4):
     trp=np.flatnonzero(fold!=pf);trt=np.flatnonzero((tfold!=tf)&allowed)
     assert all(teams[j] not in (1,2) for j in trt)
     xx=fullx[np.ix_(trp,trt,np.arange(fullx.shape[2]))].reshape(-1,fullx.shape[2]);yy=truth[np.ix_(trp,trt)].ravel()
     fit=make_pipeline(StandardScaler(),LogisticRegression(C=strength,class_weight='balanced',max_iter=1500));fit.fit(xx,yy)
     fullfits[(seed,strength,pf,tf)]=fit
 completed=[]
 for label,fraction,bgseed in conditions:
  if datetime.now(ZoneInfo('Asia/Tokyo'))>=DEADLINE:break
  cstarted=time.perf_counter()
  aa=data['a'] if label=='full' else u.cal.stratified_subset(data['a'],fraction,bgseed)
  raw=fullraw if label=='full' else calculate(data,aa,label)
  array=learning_features(raw);predictions={'gen7_county500':raw[:,:,-1]}
  for train_context in ('full','same_background'):
   for strength in (.01,.1,1.):
    seed_outputs=[]
    for seed in SEEDS:
     fold=folds[seed];output=np.full(truth.shape,np.nan)
     for pf in range(5):
      for tf in range(4):
       trp=np.flatnonzero(fold!=pf);trt=np.flatnonzero((tfold!=tf)&allowed);tet=np.flatnonzero(tfold==tf)
       assert not np.intersect1d(trt,tet).size and all(teams[j] not in (1,2) for j in trt)
       if train_context=='full':fit=fullfits[(seed,strength,pf,tf)]
       else:
        xx=array[np.ix_(trp,trt,np.arange(array.shape[2]))].reshape(-1,array.shape[2]);yy=truth[np.ix_(trp,trt)].ravel()
        fit=make_pipeline(StandardScaler(),LogisticRegression(C=strength,class_weight='balanced',max_iter=1500));fit.fit(xx,yy)
       predict(array,fit,fold,tet,pf,output)
     assert np.isfinite(output).all();seed_outputs.append(output)
     predictions[f'{train_context}_C{strength:g}_seed{seed}']=output
    predictions[f'{train_context}_C{strength:g}_five_seed_mean']=np.mean(seed_outputs,axis=0)
  table,err,_=u.evaluate(data,predictions)
  ref_condition=label if label in {c[0] for c in conditions[:7]} else 'full'
  summary=u.summaries(table,ref_condition)
  # 標準7条件以外も、同じ条件で再計算した基準との差を保存する。
  summary['comparison_reference']='same_background_recomputed_gen7_county500'
  for point in u.FPRS:
   reference=table[(table.method=='gen7_county500')&(table.fpr==point)].set_index('team').mia_all
   for name,g in table[table.fpr==point].groupby('method'):
    score=g.set_index('team').mia_all;d=(score-reference).drop([1,2]);positive=d[d>1e-12].sort_values(ascending=False)
    mask=(summary.method==name)&(summary.fpr==point)
    summary.loc[mask,'delta_vs500']=d.mean();summary.loc[mask,'wins18']=(d>1e-12).sum();summary.loc[mask,'losses18']=(d< -1e-12).sum()
    summary.loc[mask,'gain_after_removing_best2']=d.drop(positive.index[:2]).sum()
    summary.loc[mask,'worst_delta']=d.min()
    summary.loc[mask,'largest_gain_share']=positive.iloc[0]/positive.sum() if len(positive) else 0.
  out=args.output_dir/label;out.mkdir()
  table.to_csv(out/'対象別TPR.csv',index=False,float_format='%.17g');summary.to_csv(out/'集約比較.csv',index=False,float_format='%.17g')
  np.savez_compressed(out/'特徴と順位.npz',features=raw,names=np.array(sorted(predictions)),scores=np.stack([predictions[n] for n in sorted(predictions)]),teams=np.array(teams))
  for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h,p
  record=dict(condition=label,fraction=fraction,background_seed=bgseed,background_n=len(aa),seconds=time.perf_counter()-cstarted,official_tpr_max_error=err,
              input_sha256=u.base.INPUT_HASHES,code_sha256=u.sha(__file__),output_sha256={p.name:u.sha(p) for p in out.iterdir()},membership_labels='Used for fitted meta-models only, excluding targets1/2 and held-out people/targets in every fold.')
  (out/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
  completed.append(label)
  status=dict(pid=os.getpid(),completed=completed,total_conditions=len(conditions),elapsed_seconds=time.perf_counter()-started,last_completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),state='running')
  (args.output_dir/'進捗.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
  u.log(stage='background_condition_complete',condition=label,completed=len(completed),total=len(conditions),seconds=record['seconds'])
 status['state']='completed' if len(completed)==len(conditions) else 'stopped_before_deadline';status['ended_jst']=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
 (args.output_dir/'進捗.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
 u.log(stage=status['state'],completed=len(completed),seconds=time.perf_counter()-started)

if __name__=='__main__':main()
