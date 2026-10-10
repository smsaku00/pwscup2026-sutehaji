"""背景変更・本戦へ移す代表特徴の計算を保存済み配列と照合する。"""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import 共通 as u
from 別系統特徴の学習 import cache_features,SIGNALS,GEN
from 代表特徴を計算 import calculate

def main():
 out=u.HERE/'代表特徴の再現結果';assert not out.exists()
 started=time.perf_counter();data=u.setup();_,expected,_,_=cache_features(data)
 u.base.INPUT_HASHES[str(Path(__file__).resolve())]=u.sha(__file__)
 actual=calculate(data,data['a'],'full_reproduction');rows=[];models={}
 for k,(_,name) in enumerate(SIGNALS+GEN):
  delta=np.abs(actual[:,:,k]-expected[:,:,k])
  rows.append(dict(feature=name,maximum_rank_error=float(delta.max()),changed_values=int((delta>1e-12).sum()),values=int(delta.size)))
  models['cached::'+name]=expected[:,:,k];models['recomputed::'+name]=actual[:,:,k]
 assert np.max(np.abs(actual[:,:,-1]-expected[:,:,-1]))<1e-12
 # 配列確定後に公開所属正解で、順位差がTPRに影響するかを測る。
 table,err,_=u.evaluate(data,models);score_errors=[]
 for _,name in SIGNALS+GEN:
  aa=table[table.method=='cached::'+name].set_index(['team','fpr']).mia_all
  bb=table[table.method=='recomputed::'+name].set_index(['team','fpr']).mia_all
  score_errors.append(dict(feature=name,maximum_tpr_error=float(np.max(np.abs(aa-bb)))))
 out.mkdir()
 pd.DataFrame(rows).to_csv(out/'順位配列の照合.csv',index=False,float_format='%.17g')
 pd.DataFrame(score_errors).to_csv(out/'採点値の照合.csv',index=False,float_format='%.17g')
 np.savez_compressed(out/'代表特徴.npz',scores=actual,names=np.array([name for _,name in SIGNALS+GEN]),teams=np.array(data['teams']),record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
 for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h
 record=dict(input_sha256=u.base.INPUT_HASHES,code_sha256=u.sha(__file__),seconds=time.perf_counter()-started,
             official_tpr_max_error=err,maximum_rank_error=max(r['maximum_rank_error'] for r in rows),
             maximum_feature_tpr_error=max(r['maximum_tpr_error'] for r in score_errors),
             note='非会員公開特徴の再計算照合。数値差・学習器の同値の差を隠さず保存。')
 (out/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
 print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(score_errors).to_string(index=False))
 u.log(stage='representative_reproduction',seconds=record['seconds'],maximum_rank_error=record['maximum_rank_error'],maximum_tpr_error=record['maximum_feature_tpr_error'])

if __name__=='__main__':main()
