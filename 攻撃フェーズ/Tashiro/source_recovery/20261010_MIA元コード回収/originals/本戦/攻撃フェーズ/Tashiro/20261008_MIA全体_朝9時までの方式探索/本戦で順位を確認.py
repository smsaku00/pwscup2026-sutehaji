"""予備戦18対象で学習し、本戦の順位変化だけを確認する。所属正解は使わない。"""
from __future__ import annotations
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import 共通 as u
from 代表特徴を計算 import calculate,learning_features

def main():
 out=u.HERE/'本戦の順位診断結果';assert not out.exists();started=time.perf_counter()
 prelim=u.setup('prelim');p=u.HERE/'代表特徴の再現結果/代表特徴.npz'
 u.base.INPUT_HASHES[str(p.resolve())]=u.sha(p);z=np.load(p);raw=z['scores'];x=learning_features(raw)
 allowed=np.array([t not in (1,2) for t in prelim['teams']]);valid=np.flatnonzero(allowed)
 truthdir=u.base.child(u.base.child(u.ROOT/'予備戦','配布データ')/'結果','正解データ')
 own=u.base.read(truthdir/'pool_ids_5.csv').set_index('record_id').loc[prelim['b'].record_id]
 # 1・2の所属正解は読まない。18対象だけを学習する。
 truth=np.column_stack([own.pool_id.isin(u.base.read(truthdir/f'pool_ids_{t}.csv').pool_id).to_numpy(int)[prelim['normal']] for t in np.array(prelim['teams'])[allowed]])
 assert truth.shape==(875,18) and (truth.sum(axis=0)==25).all()
 main_data=u.setup('main');main_raw=calculate(main_data,main_data['a'],'main');mx=learning_features(main_raw)
 predictions={'gen7_county500':main_raw[:,:,-1]};coefficients=[];disagreements=[]
 for strength in (.01,.1,1.):
  fits=[]
  for leave in [-1]+list(range(18)):
   keep=np.ones(18,dtype=bool)
   if leave>=0:keep[leave]=False
   train=x[:,valid[keep],:].reshape(-1,x.shape[2]);yy=truth[:,keep].ravel()
   fit=make_pipeline(StandardScaler(),LogisticRegression(C=strength,class_weight='balanced',max_iter=1500));fit.fit(train,yy)
   decision=fit.decision_function(mx.reshape(-1,mx.shape[2])).reshape(875,len(main_data['teams']))
   score=np.column_stack([rankdata(col,method='average')/875 for col in decision.T])
   label=f'C{strength:g}_full18' if leave<0 else f'C{strength:g}_leave_prelim{np.array(prelim["teams"])[allowed][leave]}'
   if leave<0:predictions[label]=score
   else:fits.append(score)
   coefficients.append(dict(method=label,left_out_prelim_target=-1 if leave<0 else int(np.array(prelim['teams'])[allowed][leave]),
                            training_targets=18 if leave<0 else 17,**{f'coef_{i}':float(v) for i,v in enumerate(fit.named_steps['logisticregression'].coef_[0])}))
  predictions[f'C{strength:g}_LOO18_mean']=np.mean(fits,axis=0)
  predictions[f'C{strength:g}_LOO18_median']=np.median(fits,axis=0)
  for j,t in enumerate(main_data['teams']):
   heads=[set(np.argsort(-s[:,j],kind='stable')[:12]) for s in fits]
   count=np.bincount(np.concatenate([list(h) for h in heads]),minlength=875)
   disagreements.append(dict(C=strength,team=t,candidates_in_any_LOO_top12=int((count>0).sum()),candidates_in_all_LOO_top12=int((count==18).sum()),
                             candidates_in_at_least_half_LOO_top12=int((count>=9).sum()),members_known=False))
 # 基準の通常875人を第3回回答の実値で照合する。
 saved=u.base.read(u.HERE.parent/'20261008_第3回MIA_固定5bitと非会員較正/結果/F_mia.csv').set_index('record_id').loc[main_data['b'].record_id]
 reference=predictions['gen7_county500'];err=0.
 for j,t in enumerate(main_data['teams']):
  conf=.6+.3*rankdata(reference[:,j],method='min')/875
  err=max(err,float(np.max(np.abs(conf-saved.loc[main_data['normal'],str(t)].to_numpy()))))
 assert err<1e-12
 rows=[]
 for name,mat in predictions.items():
  for j,t in enumerate(main_data['teams']):
   for count in (12,20):
    aa=set(np.argsort(-reference[:,j],kind='stable')[:count]);bb=set(np.argsort(-mat[:,j],kind='stable')[:count])
    rows.append(dict(method=name,team=t,top_count=count,new_candidates=len(bb-aa),removed_candidates=len(aa-bb),
                     rank_correlation=float(np.corrcoef(rankdata(reference[:,j]),rankdata(mat[:,j]))[0,1]),members_known=False))
 out.mkdir();pd.DataFrame(rows).to_csv(out/'本戦の順位変化.csv',index=False,float_format='%.17g')
 pd.DataFrame(coefficients).to_csv(out/'学習係数.csv',index=False,float_format='%.17g');pd.DataFrame(disagreements).to_csv(out/'学習対象を外した順位安定性.csv',index=False)
 names=sorted(predictions);np.savez_compressed(out/'内部特徴と順位.npz',features=main_raw,names=np.array(names),scores=np.stack([predictions[n] for n in names]),teams=np.array(main_data['teams']),record_ids=main_data['b'].loc[main_data['normal'],'record_id'].to_numpy())
 for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h,p
 record=dict(seconds=time.perf_counter()-started,code_sha256=u.sha(__file__),input_sha256=u.base.INPUT_HASHES,
             main_round3_normal_value_max_error=err,main_truth_used=False,training_truth='Preliminary targets except1/2 only',
             models=len(predictions),output_sha256={p.name:u.sha(p) for p in out.iterdir()},
             caveat='順位の入替え・安定性は正解率の証明ではない。本戦チーム9の評価帯2を維持する保証もない。')
 (out/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 u.log(stage='main_diagnostics_completed',seconds=record['seconds'],round3_error=err,models=len(predictions))

if __name__=='__main__':main()
