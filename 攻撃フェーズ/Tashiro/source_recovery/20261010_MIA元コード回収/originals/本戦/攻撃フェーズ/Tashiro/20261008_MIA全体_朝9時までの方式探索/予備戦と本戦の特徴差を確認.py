"""公開特徴から予備戦・本戦が区別されるかを診断。会員ラベルは不使用。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import 共通 as u
from 特徴を外して学習を確認 import panel_features
from 別系統特徴の学習 import SIGNALS

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'予備戦本戦の特徴差結果');args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();ps=[u.HERE/'代表特徴の再現結果/代表特徴.npz',u.HERE/'本戦の順位診断結果/内部特徴と順位.npz']
 for p in ps:u.base.INPUT_HASHES[str(p)]=u.sha(p)
 zs=[np.load(p) for p in ps];raw=[zs[0]['scores'],zs[1]['features']]
 teams=[zs[0]['teams'],zs[1]['teams']];valid=np.array([t not in (1,2) for t in teams[0]])
 rows=[];correlations=[];splitrows=[];cached={}
 plans=[('all12',list(range(12)),'paired'),('all12_own',list(range(12)),'own'),('qi7',[0,1,2,9,10],'paired'),('without_ulsif',[j for j in range(12) if j not in (7,8)],'paired')]
 for label,indices,mode in plans:
  arrays=[panel_features(r,indices,mode) for r in raw]
  arrays[0]=arrays[0][:,valid,:];tt=[teams[0][valid],teams[1]]
  folds=[]
  for stage in range(2):
   pfold=np.empty(875,int);pfold[np.random.default_rng(20261008+stage).permutation(875)]=np.arange(875)%5
   tfold=np.arange(len(tt[stage]))%4;folds.append((pfold,tfold))
  for kind in ('lr','extra','hgb'):
   predictions=[np.full(a.shape[:2],np.nan) for a in arrays]
   for pf in range(5):
    for tf in range(4):
     xx=[];yy=[];weights=[]
     for stage,array in enumerate(arrays):
      pp=np.flatnonzero(folds[stage][0]!=pf);jj=np.flatnonzero(folds[stage][1]!=tf)
      xx.append(array[np.ix_(pp,jj,np.arange(array.shape[2]))].reshape(-1,array.shape[2]));yy.append(np.full(len(xx[-1]),stage));weights.append(np.full(len(xx[-1]),1/len(xx[-1])))
      splitrows.append(dict(panel=label,model=kind,person_fold=pf,target_fold=tf,stage=stage,training_targets=';'.join(str(tt[stage][j]) for j in jj),training_people=len(pp)))
     if kind=='lr':fit=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500,class_weight='balanced'))
     elif kind=='extra':fit=ExtraTreesClassifier(n_estimators=200,max_depth=8,min_samples_leaf=50,n_jobs=2,class_weight='balanced',random_state=20261008+pf*4+tf)
     else:fit=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=10,class_weight='balanced',early_stopping=False,random_state=20261008+pf*4+tf)
     fit.fit(np.vstack(xx),np.concatenate(yy))
     for stage,array in enumerate(arrays):
      pp=np.flatnonzero(folds[stage][0]==pf);jj=np.flatnonzero(folds[stage][1]==tf)
      pred=fit.predict_proba(array[np.ix_(pp,jj,np.arange(array.shape[2]))].reshape(-1,array.shape[2]))[:,1]
      assert np.isnan(predictions[stage][np.ix_(pp,jj)]).all();predictions[stage][np.ix_(pp,jj)]=pred.reshape(len(pp),len(jj))
   assert all(np.isfinite(p).all() for p in predictions)
   yt=np.r_[np.zeros(predictions[0].size),np.ones(predictions[1].size)];pv=np.r_[predictions[0].ravel(),predictions[1].ravel()]
   rows.append(dict(panel=label,model=kind,source_auc=float(roc_auc_score(yt,pv)),prelim_cells=predictions[0].size,main_cells=predictions[1].size,membership_labels_used=False))
   cached[label+'_'+kind+'_prelim']=predictions[0];cached[label+'_'+kind+'_main']=predictions[1]
   u.log(stage='domain_diagnostic_complete',panel=label,model=kind,auc=rows[-1]['source_auc'],seconds=time.perf_counter()-started)
  if label=='all12':
   corr=[np.corrcoef(a.reshape(-1,a.shape[2]),rowvar=False) for a in arrays]
   for i in range(len(SIGNALS)):
    for j in range(i+1,len(SIGNALS)):
     correlations.append(dict(feature1=SIGNALS[i][1],feature2=SIGNALS[j][1],prelim=float(corr[0][i,j]),main=float(corr[1][i,j]),absolute_change=float(abs(corr[1][i,j]-corr[0][i,j]))))
 out=args.output_dir;out.mkdir();pd.DataFrame(rows).to_csv(out/'公開特徴のフェーズ識別.csv',index=False,float_format='%.17g')
 pd.DataFrame(correlations).sort_values('absolute_change',ascending=False).to_csv(out/'特徴間相関の変化.csv',index=False,float_format='%.17g');pd.DataFrame(splitrows).to_csv(out/'交差検証分割.csv',index=False)
 np.savez_compressed(out/'内部フェーズ識別値.npz',**cached)
 for p in [Path(__file__),u.HERE/'特徴を外して学習を確認.py']:u.base.INPUT_HASHES[str(p)]=u.sha(p)
 for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h,p
 (out/'実行記録.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,code_sha256=u.sha(__file__),input_sha256=u.base.INPUT_HASHES,membership_labels_used=False,person_folds=5,target_folds=4,
  meaning='公開特徴からどちらのフェーズか区別する診断。会員判別の成績・転移失敗の証明ではない。',output_sha256={p.name:u.sha(p) for p in out.iterdir()}),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
