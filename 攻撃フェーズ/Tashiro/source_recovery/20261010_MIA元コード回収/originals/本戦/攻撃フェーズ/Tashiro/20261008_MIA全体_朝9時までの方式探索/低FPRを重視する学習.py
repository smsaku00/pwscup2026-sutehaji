"""上位非会員と会員の順位損失を直接学習。対象・人物を双方保留する。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import 共通 as u
from 特徴を外して学習を確認 import panel_features,load_truth,SEEDS

def objective(weights,groups,tail_fraction,tail_weight,regularization):
 w=weights[:-1];bias=weights[-1];loss=.5*regularization*(w@w);gradient=np.r_[regularization*w,0.]
 for positive,negative in groups:
  sp=positive@w;sn=negative@w;k=max(1,int(np.ceil(len(sn)*tail_fraction)))
  chosen=np.argpartition(sn,len(sn)-k)[-k:];xn=negative[chosen];sr=sn[chosen]
  difference=sr[None,:]-sp[:,None];pair=np.logaddexp(0.,difference).mean()
  sigmoid=expit(difference);pair_gradient=(xn.T@sigmoid.sum(axis=0)-positive.T@sigmoid.sum(axis=1))/sigmoid.size
  lp=np.logaddexp(0.,-sp-bias).mean();ln=np.logaddexp(0.,sn+bias).mean();classification=.5*(lp+ln)
  gp=-expit(-sp-bias)/len(sp);gn=expit(sn+bias)/len(sn)
  classification_gradient=.5*(positive.T@gp+negative.T@gn)
  scale=1/len(groups);loss+=scale*(tail_weight*pair+(1-tail_weight)*classification)
  gradient[:-1]+=scale*(tail_weight*pair_gradient+(1-tail_weight)*classification_gradient)
  gradient[-1]+=scale*(1-tail_weight)*.5*(gp.sum()+gn.sum())
 return loss,gradient

def gradient_check():
 rng=np.random.default_rng(20261008);groups=[(rng.normal(size=(8,5)),rng.normal(size=(30,5))) for _ in range(3)];w=rng.normal(size=6)
 checks=[]
 for tail in (.1,.5,1.):
  for mix in (0.,.5,1.):
   _,g=objective(w,groups,tail,mix,.01);numerical=[]
   for i in range(len(w)):
    delta=np.zeros(len(w));delta[i]=1e-6
    numerical.append((objective(w+delta,groups,tail,mix,.01)[0]-objective(w-delta,groups,tail,mix,.01)[0])/2e-6)
   error=float(np.max(np.abs(g-numerical)));assert error<1e-7,(tail,mix,error);checks.append(dict(tail=tail,mix=mix,max_error=error))
 return checks

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gradient-check-only',action='store_true');parser.add_argument('--output-dir',type=Path,default=u.HERE/'低FPR学習結果')
 args=parser.parse_args();checks=gradient_check()
 if args.gradient_check_only:print(json.dumps(checks));return
 assert not args.output_dir.exists();started=time.perf_counter();data=u.setup();p=u.HERE/'代表特徴の再現結果/代表特徴.npz';u.base.INPUT_HASHES[str(p)]=u.sha(p)
 z=np.load(p);assert list(z['teams'])==data['teams'];raw=z['scores'];arrays={'all12':panel_features(raw,range(12)),'without_kde':panel_features(raw,[j for j in range(12) if j!=1])}
 y=load_truth(data);allowed=np.array([t not in (1,2) for t in data['teams']]);tfold=np.arange(len(data['teams']))%4
 predictions={'gen7_county500':raw[:,:,-1]};specs=[];convergence=[];splits=[]
 prepared={}
 for panel,array in arrays.items():
  for seed in SEEDS:
   pfold=np.empty(875,int);pfold[np.random.default_rng(seed).permutation(875)]=np.arange(875)%5
   for pf in range(5):
    for tf in range(4):
     trp=np.flatnonzero(pfold!=pf);tep=np.flatnonzero(pfold==pf);trt=np.flatnonzero((tfold!=tf)&allowed);tet=np.flatnonzero(tfold==tf)
     assert not np.intersect1d(trp,tep).size and not np.intersect1d(trt,tet).size and all(data['teams'][j] not in (1,2) for j in trt)
     xx=array[np.ix_(trp,trt,np.arange(array.shape[2]))].reshape(-1,array.shape[2]);yy=y[np.ix_(trp,trt)].ravel()
     scale=StandardScaler().fit(xx);xx=scale.transform(xx).reshape(len(trp),len(trt),array.shape[2]);groups=[]
     for k,j in enumerate(trt):
      mask=y[trp,j].astype(bool);groups.append((xx[mask,k,:],xx[~mask,k,:]))
     initial=LogisticRegression(C=.1,class_weight='balanced',max_iter=1500).fit(xx.reshape(-1,array.shape[2]),yy)
     w=np.r_[initial.coef_[0],initial.intercept_[0]]
     test=scale.transform(array[:,tet,:].reshape(-1,array.shape[2])).reshape(875,len(tet),array.shape[2])
     prepared[(panel,seed,pf,tf)]=(groups,w,test,tep,tet)
     splits.append(dict(panel=panel,seed=seed,person_fold=pf,target_fold=tf,train_targets=';'.join(str(data['teams'][j]) for j in trt),test_targets=';'.join(str(data['teams'][j]) for j in tet),train_people=len(trp),test_people=len(tep)))
 args.output_dir.mkdir();(args.output_dir/'実験方針.json').write_text(json.dumps(dict(pid=__import__('os').getpid(),started_jst=__import__('datetime').datetime.now().astimezone().isoformat(),panels=list(arrays),person_seeds=SEEDS,target_folds=4,tail_fractions=(.01,.05,1.),tail_weights=(0.,.25,.75,1.),regularization=(.001,.01,.1),label_usage='学習18対象。各学習で人物と対象を保留、1/2ラベル除外。'),ensure_ascii=False,indent=2)+'\n')
 # tail_weight=0ではtail_fractionに依存しないので重複実行しない。
 plans=[(fraction,mix,reg) for mix in (0.,.25,.75,1.) for fraction in ((1.,) if mix==0 else (.01,.05,1.)) for reg in (.001,.01,.1)]
 for panel,array in arrays.items():
  for fraction,mix,reg in plans:
   draws=[];label=f'{panel}_tail{fraction:g}_weight{mix:g}_reg{reg:g}'
   for seed in SEEDS:
    output=np.full(y.shape,np.nan)
    for pf in range(5):
     for tf in range(4):
      groups,w0,test,tep,tet=prepared[(panel,seed,pf,tf)]
      opt=minimize(objective,w0,args=(groups,fraction,mix,reg),jac=True,method='L-BFGS-B',options={'maxiter':200,'maxls':50,'ftol':1e-10,'gtol':1e-6})
      assert np.isfinite(opt.x).all()
      score=test@opt.x[:-1]+opt.x[-1]
      if np.linalg.norm(opt.x[:-1])<1e-8:score.fill(0.)
      rr=np.column_stack([rankdata(col,method='average')/875 for col in score.T]);assert np.isnan(output[np.ix_(tep,tet)]).all();output[np.ix_(tep,tet)]=rr[tep]
      convergence.append(dict(method=label,seed=seed,person_fold=pf,target_fold=tf,success=bool(opt.success),status=int(opt.status),iterations=int(opt.nit),objective=float(opt.fun),coefficient_norm=float(np.linalg.norm(opt.x[:-1])),gradient_norm=float(np.linalg.norm(opt.jac)),message=str(opt.message)))
    assert np.isfinite(output).all();name=label+f'_seed{seed}';predictions[name]=output;draws.append(output)
    specs.append(dict(method=name,panel=panel,tail_fraction=fraction,tail_weight=mix,regularization=reg,seed=seed))
   name=label+'_five_seed_mean';predictions[name]=np.mean(draws,axis=0);specs.append(dict(method=name,panel=panel,tail_fraction=fraction,tail_weight=mix,regularization=reg,seeds=';'.join(map(str,SEEDS))))
   u.log(stage='low_fpr_learning_complete',method=label,seconds=time.perf_counter()-started)
   (args.output_dir/'進捗.json').write_text(json.dumps(dict(last_completed=label,completed_settings=len(specs),seconds=time.perf_counter()-started),ensure_ascii=False,indent=2)+'\n')
 # saveは新規出力を要求するので、記録置場をそのまま残して採点結果は別の子に置く。
 for f in ['低FPRを重視する学習.py','特徴を外して学習を確認.py']:u.base.INPUT_HASHES[str(u.HERE/f)]=u.sha(u.HERE/f)
 out=args.output_dir/'採点';u.save(out,data,predictions,specs,started,{'label_usage':'公開会員正解で18対象を学習。テスト人物・対象・1/2の正解を除外。','objective':'対象内の上位非会員対会員のlogistic順位損失とbalanced分類損失の凸結合。正則化。独自実装で原論文Structural SVMの再現ではない。','gradient_checks':checks,'adaptive_followup':True,'reference':'https://proceedings.mlr.press/v28/narasimhan13.html'})
 pd.DataFrame(convergence).to_csv(out/'最適化の収束.csv',index=False,float_format='%.17g');pd.DataFrame(splits).to_csv(out/'交差検証分割.csv',index=False)
 rpath=out/'実行記録.json';r=json.loads(rpath.read_text());r['label_usage']=r['extra']['label_usage'];r['output_sha256']={p.name:u.sha(p) for p in out.iterdir() if p.name!='実行記録.json'};rpath.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 u.log(stage='low_fpr_experiment_finished',models=len(predictions),seconds=time.perf_counter()-started)

if __name__=='__main__':main()
