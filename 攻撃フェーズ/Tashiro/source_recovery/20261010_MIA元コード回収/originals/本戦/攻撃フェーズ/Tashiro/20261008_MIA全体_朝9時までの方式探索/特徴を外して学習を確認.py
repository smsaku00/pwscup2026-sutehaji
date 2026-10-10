"""別系統学習の属性・密度比・補正依存を検査。会員学習は18対象のみ。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import 共通 as u
from 代表特徴を計算 import learning_features
from 別系統特徴の学習 import SIGNALS

SEEDS=(20261008,20261009,20261010,20261011,20261012)

def panel_features(raw,indices,mode='paired'):
 x=learning_features(raw);n=len(SIGNALS);indices=list(indices)
 if mode=='own':return x[:,:,indices]
 if mode=='difference':return x[:,:,indices]-x[:,:,[n+i for i in indices]]
 return x[:,:,indices+[n+i for i in indices]]

def load_truth(data):
 path=u.base.child(u.base.child(u.ROOT/'予備戦','配布データ')/'結果','正解データ')
 own=u.base.read(path/'pool_ids_5.csv').set_index('record_id').loc[data['b'].record_id]
 y=np.zeros((875,len(data['teams'])),dtype=int)
 for j,t in enumerate(data['teams']):
  if t not in (1,2):y[:,j]=own.pool_id.isin(u.base.read(path/f'pool_ids_{t}.csv').pool_id).to_numpy(int)[data['normal']]
 return y

def crossfit(x,y,teams,seed,strength=.1,split='fourfold',train_weights=None,coefs=False):
 allowed=np.array([t not in (1,2) for t in teams]);valid=np.flatnonzero(allowed)
 tfold=np.array([sorted(teams).index(t)%4 for t in teams])
 groups=[np.flatnonzero(tfold==k) for k in range(4)] if split=='fourfold' else [np.array([j]) for j in valid]+[np.flatnonzero(~allowed)]
 pfold=np.empty(875,dtype=int);pfold[np.random.default_rng(seed).permutation(875)]=np.arange(875)%5
 output=np.full(y.shape,np.nan);details=[];coefficients=[]
 for pf in range(5):
  trainp=np.flatnonzero(pfold!=pf);testp=np.flatnonzero(pfold==pf)
  for gi,testt in enumerate(groups):
   traint=np.setdiff1d(valid,testt)
   assert not np.intersect1d(trainp,testp).size and not np.intersect1d(traint,testt).size
   assert all(teams[j] not in (1,2) for j in traint)
   xx=x[np.ix_(trainp,traint,np.arange(x.shape[2]))].reshape(-1,x.shape[2]);yy=y[np.ix_(trainp,traint)].ravel()
   fit=make_pipeline(StandardScaler(),LogisticRegression(C=strength,class_weight='balanced',max_iter=1500))
   weights=None if train_weights is None else train_weights[np.ix_(trainp,traint)].ravel()
   fit.fit(xx,yy,logisticregression__sample_weight=weights)
   decision=fit.decision_function(x[:,testt,:].reshape(-1,x.shape[2])).reshape(875,len(testt))
   rr=np.column_stack([rankdata(col,method='average')/875 for col in decision.T])
   assert np.isnan(output[np.ix_(testp,testt)]).all();output[np.ix_(testp,testt)]=rr[testp]
   details.append(dict(seed=seed,person_fold=pf,target_group=gi,train_targets=';'.join(str(teams[j]) for j in traint),test_targets=';'.join(str(teams[j]) for j in testt),train_people=len(trainp),test_people=len(testp),training_positive=int(yy.sum())))
   if coefs:coefficients.extend(dict(seed=seed,person_fold=pf,target_group=gi,feature=k,coefficient=float(v)) for k,v in enumerate(fit.named_steps['logisticregression'].coef_[0]))
 assert np.isfinite(output).all()
 return output,details,coefficients

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'特徴除去の学習結果');args=parser.parse_args()
 assert not args.output_dir.exists();started=time.perf_counter();data=u.setup()
 p=u.HERE/'代表特徴の再現結果/代表特徴.npz';u.base.INPUT_HASHES[str(p)]=u.sha(p);z=np.load(p)
 assert list(z['teams'])==data['teams'];raw=z['scores']
 plans=[('all12',list(range(12)),'paired'),('without_single_ulsif',[i for i in range(12) if i!=7],'paired'),
        ('without_ulsif',[i for i in range(12) if i not in (7,8)],'paired'),('qi7_only',[0,1,2,9,10],'paired'),
        ('qi9_only',[0,1,2,9,10,11],'paired'),('without_gaussian',[i for i in range(12) if i not in (2,3)],'paired'),
        ('without_histogram',[i for i in range(12) if i!=4],'paired'),('only_gaussian_histogram',[2,3,4],'paired'),
        ('all_own_only',list(range(12)),'own'),('all_own_minus_median',list(range(12)),'difference')]
 plans += [(f'without_feature{i}',[j for j in range(12) if j!=i],'paired') for i in range(12) if i not in (4,7)]
 arrays={(label,mode):panel_features(raw,indices,mode) for label,indices,mode in plans}
 # パネルと乱数を決めてから18対象の会員正解を学習側で読む。
 y=load_truth(data);predictions={'gen7_county500':raw[:,:,-1]};specs=[];splits=[];coefficients=[]
 for label,indices,mode in plans:
  x=arrays[(label,mode)]
  target_splits=['fourfold']+(['leave_one_target_out'] if label in ('all12','without_single_ulsif','without_ulsif','qi7_only','qi9_only','all_own_minus_median') else [])
  for target_split in target_splits:
   draws=[]
   for seed in SEEDS:
    name=f'{label}_{target_split}_seed{seed}'
    output,detail,coef=crossfit(x,y,data['teams'],seed,split=target_split,coefs=True)
    predictions[name]=output;draws.append(output)
    splits.extend(dict(method=name,**row) for row in detail);coefficients.extend(dict(method=name,**row) for row in coef)
    specs.append(dict(method=name,features=';'.join(SIGNALS[i][1] for i in indices),mode=mode,C=.1,target_split=target_split,seed=seed))
   name=f'{label}_{target_split}_five_seed_mean';predictions[name]=np.mean(draws,axis=0)
   specs.append(dict(method=name,features=';'.join(SIGNALS[i][1] for i in indices),mode=mode,C=.1,target_split=target_split,seeds=';'.join(map(str,SEEDS))))
   u.log(stage='ablation_complete',panel=label,split=target_split,seconds=time.perf_counter()-started)
 u.base.INPUT_HASHES[str(Path(__file__))]=u.sha(__file__)
 u.save(args.output_dir,data,predictions,specs,started,{'label_usage':'学習は予備戦18対象のみ。テスト人物・テスト対象・1/2の正解を全分割で除外。採点は全20対象。','adaptive_followup':True,'seeds':SEEDS,'C':.1})
 pd.DataFrame(splits).to_csv(args.output_dir/'交差検証分割.csv',index=False)
 pd.DataFrame(coefficients).to_csv(args.output_dir/'学習係数.csv',index=False,float_format='%.17g')
 rpath=args.output_dir/'実行記録.json';r=json.loads(rpath.read_text());r['label_usage']=r['extra']['label_usage'];r['output_sha256']={p.name:u.sha(p) for p in args.output_dir.iterdir() if p.name!='実行記録.json'}
 rpath.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
