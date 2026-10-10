"""人物分割に加え、uLSIFアンカーの乱数を変えた別系統学習を検査。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import 共通 as u
from 特徴を外して学習を確認 import panel_features,load_truth,crossfit,SEEDS

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'学習の密度比乱数結果');args=parser.parse_args()
 assert not args.output_dir.exists();started=time.perf_counter();data=u.setup()
 ps=[u.HERE/'代表特徴の再現結果/代表特徴.npz',u.HERE/'密度比_属性乱数結果/内部順位キャッシュ.npz']
 for p in ps:u.base.INPUT_HASHES[str(p)]=u.sha(p)
 z=np.load(ps[0]);raw=z['scores'];cache=np.load(ps[1]);assert list(z['teams'])==list(cache['teams'])==data['teams']
 cnames=list(cache['names']);y=load_truth(data);predictions={'gen7_county500':raw[:,:,-1]};specs=[];splits=[];seed_ensembles={}
 for centerseed in range(20261008,20261018):
  changed=raw.copy();changed[:,:,7]=cache['scores'][cnames.index(f'all13_seed{centerseed}__raw__cross0')]
  for panel,indices in [('all12',list(range(12))),('without_kde',[j for j in range(12) if j!=1])]:
   x=panel_features(changed,indices)
   for target_split in ('fourfold','leave_one_target_out'):
    draws=[]
    for personseed in SEEDS:
     name=f'{panel}_{target_split}_center{centerseed}_person{personseed}'
     output,details,_=crossfit(x,y,data['teams'],personseed,split=target_split)
     predictions[name]=output;draws.append(output);splits.extend(dict(method=name,**r) for r in details)
     specs.append(dict(method=name,centerseed=centerseed,personseed=personseed,C=.1,panel=panel,target_split=target_split,center_selection='No choice by membership score; all ten predeclared seeds retained'))
    name=f'{panel}_{target_split}_center{centerseed}_five_person_mean';predictions[name]=np.mean(draws,axis=0)
    seed_ensembles.setdefault((panel,target_split),[]).append(predictions[name])
    specs.append(dict(method=name,centerseed=centerseed,personseeds=';'.join(map(str,SEEDS)),C=.1,panel=panel,target_split=target_split))
  u.log(stage='membership_center_seed_complete',centerseed=centerseed,seconds=time.perf_counter()-started)
 for (panel,target_split),draws in seed_ensembles.items():
  prefix=f'{panel}_{target_split}_ten_centers_five_people'
  predictions[prefix+'_mean']=np.mean(draws,axis=0);predictions[prefix+'_median']=np.median(draws,axis=0)
  specs.extend(dict(method=prefix+'_'+reduce,panel=panel,target_split=target_split,C=.1,center_seeds='20261008..20261017',person_seeds='20261008..20261012',reducer=reduce) for reduce in ('mean','median'))
 for p in [Path(__file__),u.HERE/'特徴を外して学習を確認.py']:u.base.INPUT_HASHES[str(p)]=u.sha(p)
 u.save(args.output_dir,data,predictions,specs,started,{'label_usage':'18対象の公開会員ラベルを学習に使用。各分割でテスト人物・対象・1/2の正解を除外。採点は全20対象。','adaptive_followup':True,'note':'単一uLSIF特徴7のみ中心乱数を変更。特徴8の10中心平均は固定。対象別に最良中心を選ぶ処理はない。'})
 pd.DataFrame(splits).to_csv(args.output_dir/'交差検証分割.csv',index=False)
 rpath=args.output_dir/'実行記録.json';r=json.loads(rpath.read_text());r['label_usage']=r['extra']['label_usage'];r['output_sha256']={p.name:u.sha(p) for p in args.output_dir.iterdir() if p.name!='実行記録.json'};rpath.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
