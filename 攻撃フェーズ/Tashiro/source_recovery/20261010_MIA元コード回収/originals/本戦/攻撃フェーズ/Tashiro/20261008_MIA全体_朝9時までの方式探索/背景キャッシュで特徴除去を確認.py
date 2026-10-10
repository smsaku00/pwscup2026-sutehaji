"""47背景の保存特徴で、点密度特徴を外した学習を二方向交差検証する。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import 共通 as u
from 特徴を外して学習を確認 import panel_features,load_truth,crossfit,SEEDS

def aggregate_against(table,reference='gen7_county500'):
 result=[]
 for fpr,group in table.groupby('fpr'):
  baseline=group[group.method==reference].set_index('team').mia_all
  for name,part in group.groupby('method'):
   score=part.set_index('team').mia_all;d=(score-baseline).drop([1,2]);positive=d[d>1e-12].sort_values(ascending=False)
   result.append(dict(method=name,fpr=fpr,all20=float(score.mean()),other18=float(score.drop([1,2]).mean()),delta_vs500=float(d.mean()),wins18=int((d>1e-12).sum()),losses18=int((d< -1e-12).sum()),worst_delta=float(d.min()),gain_after_removing_best2=float(d.drop(positive.index[:2]).sum()),largest_gain_share=float(positive.iloc[0]/positive.sum()) if len(positive) else 0.))
 return pd.DataFrame(result)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'特徴除去の背景結果');args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();y=load_truth(data);source=u.HERE/'背景変更の学習結果'
 status=json.loads((source/'進捗.json').read_text());assert status['state']=='completed' and len(status['completed'])==47
 for f in ['背景キャッシュで特徴除去を確認.py','特徴を外して学習を確認.py']:u.base.INPUT_HASHES[str(u.HERE/f)]=u.sha(u.HERE/f)
 args.output_dir.mkdir();fullpath=source/'full/特徴と順位.npz';full=np.load(fullpath)['features'];indices=[j for j in range(12) if j!=1]
 # 各条件で同じモデル規則を学習し直す。パネルや正則化を条件別に選ばない。
 records=[];completed=[]
 for condition in status['completed']:
  p=source/condition/'特徴と順位.npz';u.base.INPUT_HASHES[str(p)]=u.sha(p);z=np.load(p);assert list(z['teams'])==data['teams'];raw=z['features'];array=panel_features(raw,indices)
  predictions={'gen7_county500':raw[:,:,-1]};splits=[]
  for split in ('fourfold','leave_one_target_out'):
   for strength in (.01,.1,1.) if split=='fourfold' else (.1,):
    draws=[]
    for seed in SEEDS:
     name=f'without_kde_{split}_C{strength:g}_seed{seed}';output,details,_=crossfit(array,y,data['teams'],seed,strength,split)
     predictions[name]=output;draws.append(output)
     if strength==.1:splits.extend(dict(method=name,**r) for r in details)
    predictions[f'without_kde_{split}_C{strength:g}_five_seed_mean']=np.mean(draws,axis=0)
  table,error,_=u.evaluate(data,predictions);agg=aggregate_against(table);out=args.output_dir/condition;out.mkdir()
  table.to_csv(out/'対象別TPR.csv',index=False,float_format='%.17g');agg.to_csv(out/'集約比較.csv',index=False,float_format='%.17g');pd.DataFrame(splits).to_csv(out/'交差検証分割.csv',index=False)
  names=sorted(predictions);np.savez_compressed(out/'内部順位キャッシュ.npz',names=np.array(names),scores=np.stack([predictions[n] for n in names]),teams=np.array(data['teams']),record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
  record=dict(condition=condition,code_sha256=u.sha(__file__),feature_cache_sha256=u.sha(p),label_usage='会員学習18対象のみ。テスト人物・対象・1/2ラベル除外。全20対象で採点。',official_tpr_max_error=error,output_sha256={p.name:u.sha(p) for p in out.iterdir()})
  (out/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');records.append(agg.assign(condition=condition));completed.append(condition)
  (args.output_dir/'進捗.json').write_text(json.dumps(dict(pid=__import__('os').getpid(),completed=completed,total=47,seconds=time.perf_counter()-started,state='running'),ensure_ascii=False,indent=2)+'\n')
  u.log(stage='ablation_background_complete',condition=condition,completed=len(completed),seconds=time.perf_counter()-started)
 pd.concat(records,ignore_index=True).to_csv(args.output_dir/'全条件の比較.csv',index=False,float_format='%.17g')
 for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h,p
 (args.output_dir/'実行記録.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,input_sha256=u.base.INPUT_HASHES,code_sha256=u.sha(__file__),completed=len(completed),adaptive_followup=True),ensure_ascii=False,indent=2)+'\n')
 (args.output_dir/'進捗.json').write_text(json.dumps(dict(pid=__import__('os').getpid(),completed=completed,total=47,seconds=time.perf_counter()-started,state='completed'),ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
