"""複数ビュー学習の1対象ずつの保留。全ビューで人物・対象を同時に保留する。"""
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
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'複数ビューの対象保留結果');args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();source=u.HERE/'背景変更の学習結果';status=json.loads((source/'進捗.json').read_text());assert status['state']=='completed' and len(status['completed'])==47
 labels=status['completed'];views=[]
 for label in labels:
  p=source/label/'特徴と順位.npz';u.base.INPUT_HASHES[str(p)]=u.sha(p);z=np.load(p);assert list(z['teams'])==data['teams'];views.append(z['features'])
 raw=np.stack(views);full=raw[labels.index('full')]
 p=u.HERE/'密度比_属性乱数結果/内部順位キャッシュ.npz';u.base.INPUT_HASHES[str(p)]=u.sha(p);zc=np.load(p);assert list(zc['teams'])==data['teams'];cn=list(zc['names']);centers=[]
 for seed in range(20261008,20261018):
  r=full.copy();r[:,:,7]=zc['scores'][cn.index(f'all13_seed{seed}__raw__cross0')];centers.append(r)
 indices=[j for j in range(12) if j!=1];x=np.stack([panel_features(r,indices) for r in raw]);cx=np.stack([panel_features(r,indices) for r in centers]);y=load_truth(data)
 valid=np.array([j for j,t in enumerate(data['teams']) if t not in (1,2)]);groups=[np.array([j]) for j in valid]+[np.array([j for j,t in enumerate(data['teams']) if t in (1,2)])]
 settings=[('full_only',x[[labels.index('full')]],np.ones(1)),('background47',x,np.full(47,1/47)),('center10',cx,np.full(10,1/10)),('balanced_background_and_centers',np.concatenate([x,cx]),np.r_[np.full(47,.5/47),np.full(10,.5/10)])]
 predictions=[{'gen7_county500':r[:,:,-1]} for r in raw];splits=[];coefficients=[];convergence=[];specs=[]
 args.output_dir.mkdir();(args.output_dir/'実験方針.json').write_text(json.dumps(dict(pid=__import__('os').getpid(),C=.1,person_seeds=SEEDS,target_split='leave_one_target_out',panels='without_kde',contexts=[s[0] for s in settings],weighting='ビュー数で重みを正規化。背景と中心の混合では各家族1/2。',label_usage='テスト人物・テスト対象・1/2の正解は全ビューの学習から除く。'),ensure_ascii=False,indent=2)+'\n')
 for context,train_views,view_weights in settings:
  draws=[]
  for seed in SEEDS:
   pfold=np.empty(875,int);pfold[np.random.default_rng(seed).permutation(875)]=np.arange(875)%5;outputs=np.full((47,875,len(data['teams'])),np.nan)
   for pf in range(5):
    trp=np.flatnonzero(pfold!=pf);tep=np.flatnonzero(pfold==pf)
    for gi,tet in enumerate(groups):
     trt=np.setdiff1d(valid,tet);assert not np.intersect1d(trp,tep).size and not np.intersect1d(trt,tet).size and all(data['teams'][j] not in (1,2) for j in trt)
     nc=x.shape[3];train=train_views[:,trp,:,:][:,:,trt,:].reshape(-1,nc);truth=np.tile(y[np.ix_(trp,trt)].ravel(),len(train_views));weight=np.repeat(view_weights,len(trp)*len(trt))
     scaler=StandardScaler().fit(train,sample_weight=weight);fit=LogisticRegression(C=.1,class_weight='balanced',max_iter=1500);fit.fit(scaler.transform(train),truth,sample_weight=weight)
     for v in range(47):
      test=x[v,:,tet,:].transpose(1,0,2).reshape(-1,nc);decision=fit.decision_function(scaler.transform(test)).reshape(875,len(tet));rank=np.column_stack([rankdata(col,method='average')/875 for col in decision.T]);assert np.isnan(outputs[v][np.ix_(tep,tet)]).all();outputs[v][np.ix_(tep,tet)]=rank[tep]
     splits.append(dict(context=context,seed=seed,person_fold=pf,target_group=gi,views=len(train_views),effective_cells=float(weight.sum()),train_targets=';'.join(str(data['teams'][j]) for j in trt),test_targets=';'.join(str(data['teams'][j]) for j in tet),train_people=len(trp),test_people=len(tep)))
     convergence.append(dict(context=context,seed=seed,person_fold=pf,target_group=gi,iterations=int(fit.n_iter_[0]),max_iter=1500))
     coefficients.extend(dict(context=context,seed=seed,person_fold=pf,target_group=gi,feature=k,coefficient=float(v)) for k,v in enumerate(fit.coef_[0]))
   assert np.isfinite(outputs).all();draws.append(outputs);name=f'without_kde_LOO_{context}_seed{seed}'
   for v in range(47):predictions[v][name]=outputs[v]
   specs.append(dict(method=name,C=.1,training_context=context,seed=seed,target_split='leave_one_target_out'))
   u.log(stage='multiview_loo_seed_complete',context=context,seed=seed,seconds=time.perf_counter()-started)
   (args.output_dir/'進捗.json').write_text(json.dumps(dict(pid=__import__('os').getpid(),state='running',last_completed=name,seconds=time.perf_counter()-started),ensure_ascii=False,indent=2)+'\n')
  name=f'without_kde_LOO_{context}_five_seed_mean';mean=np.mean(draws,axis=0)
  for v in range(47):predictions[v][name]=mean[v]
  specs.append(dict(method=name,C=.1,training_context=context,seeds=';'.join(map(str,SEEDS)),target_split='leave_one_target_out'))
 records=[]
 for v,label in enumerate(labels):
  table,error,_=u.evaluate(data,predictions[v]);agg=aggregate_against(table);out=args.output_dir/label;out.mkdir();table.to_csv(out/'対象別TPR.csv',index=False,float_format='%.17g');agg.to_csv(out/'集約比較.csv',index=False,float_format='%.17g')
  names=sorted(predictions[v]);np.savez_compressed(out/'内部順位キャッシュ.npz',names=np.array(names),scores=np.stack([predictions[v][n] for n in names]),teams=np.array(data['teams']),record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
  (out/'実行記録.json').write_text(json.dumps(dict(condition=label,official_tpr_max_error=error,output_sha256={p.name:u.sha(p) for p in out.iterdir()}),ensure_ascii=False,indent=2)+'\n');records.append(agg.assign(condition=label))
 pd.concat(records,ignore_index=True).to_csv(args.output_dir/'全条件の比較.csv',index=False,float_format='%.17g');pd.DataFrame(splits).to_csv(args.output_dir/'交差検証分割.csv',index=False);pd.DataFrame(coefficients).to_csv(args.output_dir/'学習係数.csv',index=False,float_format='%.17g');pd.DataFrame(convergence).to_csv(args.output_dir/'収束の確認.csv',index=False);pd.DataFrame(specs).to_csv(args.output_dir/'方式一覧.csv',index=False)
 for f in ['複数ビューの対象保留確認.py','特徴を外して学習を確認.py','背景キャッシュで特徴除去を確認.py']:u.base.INPUT_HASHES[str(u.HERE/f)]=u.sha(u.HERE/f)
 for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h,p
 (args.output_dir/'実行記録.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,input_sha256=u.base.INPUT_HASHES,code_sha256=u.sha(__file__),main_truth_used=False,adaptive_followup=True,label_usage='公開予備戦18対象。テスト人物・対象・1/2の正解を全ビューから除外。'),ensure_ascii=False,indent=2)+'\n')
 (args.output_dir/'進捗.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,state='completed'),ensure_ascii=False,indent=2)+'\n');u.log(stage='multiview_loo_completed',models=len(predictions[0]),seconds=time.perf_counter()-started)

if __name__=='__main__':main()
