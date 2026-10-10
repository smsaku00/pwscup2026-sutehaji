"""昨日の学習を、正解の学習からも対象1・2を除いて確認する。"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import 共通 as u

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=u.HERE/'昨日の学習_18対象結果')
 args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();b=data['b'];a=data['a'];normal=data['normal'];teams=data['teams']
 legacy=u.module('overnight_old_learning',u.OLD/'学習による順位補正の検討.py')
 for p in [Path(__file__),Path(legacy.__file__)]:u.base.INPUT_HASHES[str(p.resolve())]=u.sha(p)
 contexts={};baseline={}
 for name,aa in [('full',a),('half0',a.iloc[::2]),('half1',a.iloc[1::2])]:
  computed={t:u.base.geometry_scores(aa,b,c) for t,c in data['cs'].items()}
  contexts[name],baseline[name]=legacy.features(computed,teams,normal)
  u.log(stage='learning_features',context=name,seconds=time.perf_counter()-started)
 # ラベルを利用するモデル。ここで初めて公開正解を学習に読む。
 truthdir=u.base.child(u.base.child(u.ROOT/'予備戦','配布データ')/'結果','正解データ')
 own=u.base.read(truthdir/'pool_ids_5.csv').set_index('record_id').loc[b.record_id]
 truth=np.column_stack([own.pool_id.isin(u.base.read(truthdir/f'pool_ids_{t}.csv').pool_id).to_numpy(int)[normal] for t in teams])
 assert (truth.sum(axis=0)==25).all()
 rng=np.random.default_rng(20261007);fold=np.empty(normal.sum(),dtype=int)
 fold[rng.permutation(normal.sum())]=np.arange(normal.sum())%5
 tfold=np.array([sorted(teams).index(t)%4 for t in teams])
 include18=np.array([t not in (1,2) for t in teams])
 matrices={'fixed_gen7':baseline['full']}
 med=np.column_stack([np.median(np.delete(baseline['full'],j,axis=1),axis=1) for j in range(len(teams))])
 matrices['fixed_gen7_cross01']=baseline['full']-.1*med
 rows=[];coefficients=[];specs=[]
 for scope in ('all20_reproduction','exclude12_train18'):
  allowed=np.ones(len(teams),dtype=bool) if scope=='all20_reproduction' else include18
  for training in ('full','all_contexts'):
   training_names=['full'] if training=='full' else list(contexts)
   for model,columns in [('rank1',[0]),('nuisance2',[0,1]),('signals5',[0,1,2,3,4])]:
    output=np.full(truth.shape,np.nan)
    name=f'{scope}_{training}_{model}'
    for pf in range(5):
     for tf in range(4):
      trp=np.flatnonzero(fold!=pf);tep=np.flatnonzero(fold==pf)
      trt=np.flatnonzero((tfold!=tf)&allowed);tet=np.flatnonzero(tfold==tf)
      assert not np.intersect1d(trp,tep).size and not np.intersect1d(trt,tet).size
      if scope=='exclude12_train18':assert all(teams[i] not in (1,2) for i in trt)
      xx=np.concatenate([contexts[n][np.ix_(trp,trt,columns)].reshape(-1,len(columns)) for n in training_names])
      yy=np.tile(truth[np.ix_(trp,trt)].ravel(),len(training_names))
      fit=make_pipeline(StandardScaler(),LogisticRegression(C=.5,class_weight='balanced',max_iter=1000))
      fit.fit(xx,yy,logisticregression__sample_weight=np.full(len(yy),1/len(training_names)))
      dx=contexts['full'][np.ix_(np.arange(normal.sum()),tet,columns)].reshape(-1,len(columns))
      decision=fit.decision_function(dx).reshape(normal.sum(),len(tet))
      ranked=np.column_stack([rankdata(col,method='average')/normal.sum() for col in decision.T])
      assert np.isnan(output[np.ix_(tep,tet)]).all();output[np.ix_(tep,tet)]=ranked[tep]
      rows.append(dict(method=name,person_fold=pf,target_fold=tf,train_people=len(trp),test_people=len(tep),
                       train_targets=';'.join(str(teams[i]) for i in trt),test_targets=';'.join(str(teams[i]) for i in tet),
                       train_pairs=len(yy),train_positive=int(yy.sum())))
      coefficients.append(dict(method=name,person_fold=pf,target_fold=tf,**{legacy.FEATURES[k]:float(v) for k,v in zip(columns,fit.named_steps['logisticregression'].coef_[0])}))
    assert np.isfinite(output).all()
    if model=='rank1':
     deviation=float(np.max(np.abs(output-baseline['full'])))
     signs=[r[legacy.FEATURES[0]] for r in coefficients if r['method']==name]
     u.log(stage='rank1_diagnostic',method=name,maximum_deviation=deviation,negative_coefficient_folds=int(np.sum(np.array(signs)<0)))
     if scope=='all20_reproduction':assert deviation<1e-12
    matrices[name]=output
    specs.append(dict(method=name,training_scope=scope,training_contexts=training,input_columns=';'.join(legacy.FEATURES[i] for i in columns)))
    u.log(stage='learning_complete',method=name,seconds=time.perf_counter()-started)
 table,_=u.save(args.output_dir,data,matrices,specs,started,{'label_usage':'学習と採点。新案は1・2のラベルを全学習分割から除外。旧案再現は全20を使用し別名で区別。',
                                                       'person_folds':5,'target_folds':4,'seed':20261007,'features':legacy.FEATURES})
 pd.DataFrame(rows).to_csv(args.output_dir/'交差検証分割.csv',index=False)
 pd.DataFrame(coefficients).to_csv(args.output_dir/'標準化後係数.csv',index=False,float_format='%.17g')
 old=u.base.read(u.OLD/'学習結果_順位校正版/学習_対象別比較.csv')
 errors=[]
 for training in ('full','all_contexts'):
  for model in ('rank1','nuisance2','signals5'):
   oldname=f'full__learn_{training}_{model}';newname=f'all20_reproduction_{training}_{model}'
   expected=old[old.method==oldname].set_index('team').mia_all
   actual=table[(table.method==newname)&(table.fpr==.01)].set_index('team').mia_all
   error=float(np.max(np.abs(expected-actual)));assert error<1e-12;errors.append(dict(method=newname,maximum_error=error))
 pd.DataFrame(errors).to_csv(args.output_dir/'昨日の学習再現.csv',index=False)
 # この実験は公開正解を学習にも使う。汎用保存器の採点専用の既定文を上書きする。
 import json
 record_path=args.output_dir/'実行記録.json'
 record=json.loads(record_path.read_text())
 record['label_usage']='学習と採点。exclude12_train18は対象1・2を全学習分割から除外。all20_reproductionは旧学習の再現用。'
 record['input_sha256']=u.base.INPUT_HASHES
 record['old_learning_reproduction_max_error']=max(x['maximum_error'] for x in errors)
 record['output_sha256']={p.name:u.sha(p) for p in args.output_dir.iterdir() if p.name!='実行記録.json'}
 record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 u.log(stage='old_learning_reproduced',maximum_error=max(x['maximum_error'] for x in errors))

if __name__=='__main__':main()
