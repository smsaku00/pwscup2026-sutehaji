"""公開特徴の複数ビューを学習。患者・対象の保留は全ビューで共通。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import 共通 as u
from 特徴を外して学習を確認 import panel_features,load_truth,SEEDS
from 背景キャッシュで特徴除去を確認 import aggregate_against

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'複数ビューの学習結果');args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();source=u.HERE/'背景変更の学習結果';status=json.loads((source/'進捗.json').read_text());assert status['state']=='completed' and len(status['completed'])==47
 labels=status['completed'];views=[]
 for label in labels:
  p=source/label/'特徴と順位.npz';u.base.INPUT_HASHES[str(p)]=u.sha(p);z=np.load(p);assert list(z['teams'])==data['teams'];views.append(z['features'])
 raw=np.stack(views);full=raw[labels.index('full')]
 p=u.HERE/'密度比_属性乱数結果/内部順位キャッシュ.npz';u.base.INPUT_HASHES[str(p)]=u.sha(p);centers=np.load(p);assert list(centers['teams'])==data['teams'];names=list(centers['names'])
 center_views=[]
 for seed in range(20261008,20261018):
  r=full.copy();r[:,:,7]=centers['scores'][names.index(f'all13_seed{seed}__raw__cross0')];center_views.append(r)
 center_views=np.stack(center_views)
 mp=u.HERE/'本戦の順位診断結果/内部特徴と順位.npz';u.base.INPUT_HASHES[str(mp)]=u.sha(mp);mz=np.load(mp);main_raw=mz['features']
 # 全公開ビューを固定してから、18対象の会員正解を学習側で読む。
 y=load_truth(data);allowed=np.array([t not in (1,2) for t in data['teams']]);tfold=np.array([sorted(data['teams']).index(t)%4 for t in data['teams']])
 predictions=[{'gen7_county500':r[:,:,-1]} for r in raw];main_predictions={'gen7_county500':main_raw[:,:,-1]};specs=[];splits=[];coefficients=[]
 panels={'all12':list(range(12)),'without_kde':[j for j in range(12) if j!=1]}
 args.output_dir.mkdir();policy=dict(pid=__import__('os').getpid(),training_groups=('full_only','background47','center10','balanced_background_and_centers'),panels=panels,C=.1,person_seeds=SEEDS,target_folds=4,
   interpretation='同じ患者・対象の公開特徴の複数ビュー。ラベル数を増やす独立データとは扱わない。ビュー数で標本重みを正規化する。人物・対象の保留は全ビューで共通。')
 (args.output_dir/'実験方針.json').write_text(json.dumps(policy,ensure_ascii=False,indent=2)+'\n')
 for panel,indices in panels.items():
  xx=np.stack([panel_features(r,indices) for r in raw]);cx=np.stack([panel_features(r,indices) for r in center_views]);mx=panel_features(main_raw,indices)
  settings=[('full_only',xx[[labels.index('full')]],np.ones(1)),('background47',xx,np.full(47,1/47)),('center10',cx,np.full(10,1/10)),('balanced_background_and_centers',np.concatenate([xx,cx]),np.r_[np.full(47,.5/47),np.full(10,.5/10)])]
  for context,train_views,view_weight in settings:
   all_draws=[];main_draws=[]
   for seed in SEEDS:
    pfold=np.empty(875,int);pfold[np.random.default_rng(seed).permutation(875)]=np.arange(875)%5
    outputs=np.full((len(labels),875,len(data['teams'])),np.nan);main_fits=[]
    for pf in range(5):
     trp=np.flatnonzero(pfold!=pf);tep=np.flatnonzero(pfold==pf)
     for tf in range(4):
      trt=np.flatnonzero((tfold!=tf)&allowed);tet=np.flatnonzero(tfold==tf)
      assert not np.intersect1d(trp,tep).size and not np.intersect1d(trt,tet).size and all(data['teams'][j] not in (1,2) for j in trt)
      ncols=xx.shape[3];train=train_views[:,trp,:,:][:,:,trt,:].reshape(-1,ncols);truth=np.tile(y[np.ix_(trp,trt)].ravel(),len(train_views));weight=np.repeat(view_weight,len(trp)*len(trt))
      scaler=StandardScaler();scaler.fit(train,sample_weight=weight)
      fit=LogisticRegression(C=.1,class_weight='balanced',max_iter=1500);fit.fit(scaler.transform(train),truth,sample_weight=weight)
      for v in range(len(labels)):
       decision=fit.decision_function(scaler.transform(xx[v,:,tet,:].transpose(1,0,2).reshape(-1,ncols))).reshape(875,len(tet))
       rr=np.column_stack([rankdata(col,method='average')/875 for col in decision.T]);assert np.isnan(outputs[v][np.ix_(tep,tet)]).all();outputs[v][np.ix_(tep,tet)]=rr[tep]
      # 本戦は未知の人物・対象なので、各保留分割のモデルを全行に適用して平均する。
      decision=fit.decision_function(scaler.transform(mx.reshape(-1,ncols))).reshape(mx.shape[:2]);main_fits.append(np.column_stack([rankdata(col,method='average')/875 for col in decision.T]))
      splits.append(dict(panel=panel,context=context,seed=seed,person_fold=pf,target_fold=tf,training_views=len(train_views),effective_training_cells=float(weight.sum()),train_targets=';'.join(str(data['teams'][j]) for j in trt),test_targets=';'.join(str(data['teams'][j]) for j in tet),train_people=len(trp),test_people=len(tep)))
      coefficients.extend(dict(panel=panel,context=context,seed=seed,person_fold=pf,target_fold=tf,feature=k,coefficient=float(value)) for k,value in enumerate(fit.coef_[0]))
    assert np.isfinite(outputs).all();all_draws.append(outputs);main_draws.append(np.mean(main_fits,axis=0));name=f'{panel}_{context}_seed{seed}'
    for v in range(len(labels)):predictions[v][name]=outputs[v]
    main_predictions[name]=main_draws[-1];specs.append(dict(method=name,panel=panel,training_context=context,C=.1,seed=seed,view_weight_sum=float(view_weight.sum())))
    u.log(stage='multiview_seed_complete',panel=panel,context=context,seed=seed,seconds=time.perf_counter()-started)
   consensus=np.mean(all_draws,axis=0);name=f'{panel}_{context}_five_seed_mean'
   for v in range(len(labels)):predictions[v][name]=consensus[v]
   main_predictions[name]=np.mean(main_draws,axis=0);specs.append(dict(method=name,panel=panel,training_context=context,C=.1,person_seeds=';'.join(map(str,SEEDS))))
   (args.output_dir/'進捗.json').write_text(json.dumps(dict(last_completed=panel+'_'+context,pid=__import__('os').getpid(),seconds=time.perf_counter()-started,state='running'),ensure_ascii=False,indent=2)+'\n')
 records=[]
 for v,label in enumerate(labels):
  table,error,_=u.evaluate(data,predictions[v]);agg=aggregate_against(table);out=args.output_dir/label;out.mkdir();table.to_csv(out/'対象別TPR.csv',index=False,float_format='%.17g');agg.to_csv(out/'集約比較.csv',index=False,float_format='%.17g')
  names=sorted(predictions[v]);np.savez_compressed(out/'内部順位キャッシュ.npz',names=np.array(names),scores=np.stack([predictions[v][n] for n in names]),teams=np.array(data['teams']),record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
  (out/'実行記録.json').write_text(json.dumps(dict(condition=label,official_tpr_max_error=error,output_sha256={p.name:u.sha(p) for p in out.iterdir()}),ensure_ascii=False,indent=2)+'\n');records.append(agg.assign(condition=label))
 pd.concat(records,ignore_index=True).to_csv(args.output_dir/'全条件の比較.csv',index=False,float_format='%.17g');pd.DataFrame(splits).to_csv(args.output_dir/'交差検証分割.csv',index=False);pd.DataFrame(coefficients).to_csv(args.output_dir/'学習係数.csv',index=False,float_format='%.17g');pd.DataFrame(specs).to_csv(args.output_dir/'方式一覧.csv',index=False)
 names=sorted(main_predictions);np.savez_compressed(args.output_dir/'本戦の内部順位.npz',names=np.array(names),scores=np.stack([main_predictions[n] for n in names]),teams=mz['teams'],record_ids=mz['record_ids'])
 for f in ['複数背景と乱数で学習.py','特徴を外して学習を確認.py','背景キャッシュで特徴除去を確認.py']:u.base.INPUT_HASHES[str(u.HERE/f)]=u.sha(u.HERE/f)
 for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h,p
 (args.output_dir/'実行記録.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,input_sha256=u.base.INPUT_HASHES,code_sha256=u.sha(__file__),training_labels='18対象のみ。保留する人物・対象・1/2のラベルは全ビューから除外。',main_truth_used=False,adaptive_followup=True),ensure_ascii=False,indent=2)+'\n')
 (args.output_dir/'進捗.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,state='completed'),ensure_ascii=False,indent=2)+'\n')
 u.log(stage='multiview_completed',models=len(main_predictions),backgrounds=len(labels),seconds=time.perf_counter()-started)

if __name__=='__main__':main()
