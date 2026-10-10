"""公開Cと背景Aの行を分類する。会員正解でモデルを学習しない。"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier,IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
import 共通 as u
from 距離と分布の探索 import transform

def encode(a,c,query,nums,cats,kind='robust'):
 ar,cr,x=transform(a,[a,c,query],nums,kind)
 arrays=[np.clip(z,-20,20) for z in [ar,cr,x]]
 for col in cats:
  levels=sorted(set(a[col])|set(c[col])|set(query[col]))
  for i,frame in enumerate([a,c,query]):
   onehot=frame[col].to_numpy()[:,None]==np.array(levels)[None,:]
   arrays[i]=np.column_stack([arrays[i],onehot.astype(float)])
 return arrays

def models(seed):
 result=[]
 for strength in (.02,.2,2.,20.):
  result.append((f'lr_C{strength:g}',LogisticRegression(C=strength,class_weight='balanced',max_iter=1500,solver='lbfgs'),False))
 for leaf in (1,5,20,50):
  result.append((f'extra_leaf{leaf}',ExtraTreesClassifier(n_estimators=200,min_samples_leaf=leaf,class_weight='balanced',n_jobs=2,random_state=seed),False))
 for leaf in (5,20):
  result.append((f'rf_leaf{leaf}',RandomForestClassifier(n_estimators=200,min_samples_leaf=leaf,class_weight='balanced',n_jobs=2,random_state=seed),False))
 for leaf in (15,63):
  for penalty in (1.,20.):
   result.append((f'hgb_leaf{leaf}_l2{penalty:g}',HistGradientBoostingClassifier(max_leaf_nodes=leaf,l2_regularization=penalty,max_iter=150,min_samples_leaf=20,learning_rate=.07,early_stopping=False,class_weight='balanced',random_state=seed),False))
 result.append(('isolationC',IsolationForest(n_estimators=200,max_samples=256,n_jobs=2,random_state=seed),True))
 for gamma in (.02,.2):result.append((f'ocsvmC_gamma{gamma:g}',OneClassSVM(gamma=gamma,nu=.1),True))
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--seed',type=int,default=20261008)
 parser.add_argument('--output-dir',type=Path,default=u.HERE/'公開データ分類器結果')
 args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();banks={};specs=[]
 for p in [Path(__file__),u.HERE/'距離と分布の探索.py']:u.base.INPUT_HASHES[str(p.resolve())]=u.sha(p)
 configurations=[('q7_robust',u.NUM7,u.CAT7,'robust'),('q7_rank',u.NUM7,u.CAT7,'rank'),
                 ('q9_robust',u.NUM9,u.CAT7,'robust'),('q13_robust',u.NUM13,u.CAT13,'robust')]
 for label,nums,cats,kind in configurations:
  query=data['query'] if label.startswith('q7_') else data['b']
  calibration=label.startswith('q7_')
  for index,(t,c) in enumerate(data['cs'].items(),1):
   ar,cr,x=encode(data['a'],c,query,nums,cats,kind)
   train=np.row_stack([ar,cr]);y=np.r_[np.zeros(len(ar),dtype=int),np.ones(len(cr),dtype=int)]
   scores={}
   for name,model,novelty in models(args.seed):
    if novelty:
     model.fit(cr);score=model.score_samples(x)
    else:
     model.fit(train,y)
     score=logit(np.clip(model.predict_proba(x)[:,1],1e-8,1-1e-8))
    assert np.isfinite(score).all()
    scores[label+'_'+name]=score
    if index==1:specs.append(dict(raw_method=label+'_'+name,family='source_classifier' if not novelty else 'C_novelty',
                                attributes=len(nums)+len(cats),transform=kind,uses_public_null=calibration,seed=args.seed,
                                membership_labels_used=False,model_parameters=str(model.get_params())))
   for name,value in u.variants(scores,data,t,calibration).items():banks.setdefault(name,{})[t]=value
   u.log(stage='source_classifiers',configuration=label,completed_targets=index,targets=len(data['cs']),seconds=time.perf_counter()-started)
 models_out=u.matrices(banks,data['teams'])
 u.log(stage='predictions_fixed',raw_specs=len(specs),models=len(models_out),seconds=time.perf_counter()-started)
 u.save(args.output_dir,data,models_out,specs,started,{'seed':args.seed,'training':'Row-source labels C=1,A=0; no membership truth. C-only novelty models fit C alone.'})

if __name__=='__main__':main()
