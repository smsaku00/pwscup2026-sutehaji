"""全対象に同じsoftmax競合規則を適用。人数を強制せず、会員学習は行わない。"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
from scipy.special import logit,logsumexp
from scipy.stats import rankdata
import 共通 as u

SOURCES=[('距離と分布結果','gen7_reference__county500__cross0.1'),
 ('距離と分布結果','q7_cat4_nnratio_k3__raw__cross0'),
 ('混合分布結果','q7_robust_sex1_full_reg0.01_gmm1__county500__cross0.1'),
 ('混合分布結果','q13_robust_sex0_diag_reg0.01_gmm16__raw__cross0.1'),
 ('局所密度結果','q13_robust_lofC_k100__raw__cross0'),
 ('学習の分割感度結果','median20_fourfold_C0.1_five_seed_mean'),
 ('学習の密度比乱数結果','without_kde_fourfold_ten_centers_five_people_median')]

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'滑らかな対象間補正結果');args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();predictions={};specs=[]
 for index,(folder,name) in enumerate(SOURCES):
  p=u.HERE/folder/'内部順位キャッシュ.npz';u.base.INPUT_HASHES[str(p)]=u.sha(p);z=np.load(p);assert list(z['teams'])==data['teams']
  raw=z['scores'][list(z['names']).index(name)];predictions[f'source{index}_unchanged']=raw
  ranked=np.column_stack([rankdata(v,method='average')/875 for v in raw.T]);utility=logit(np.clip(ranked,.5/875,1-.5/875))
  for scope in ('all20','excluding1and2'):
   valid=np.array([j for j,t in enumerate(data['teams']) if scope=='all20' or t not in (1,2)])
   for temperature in (.25,.5,1.,2.,4.,8.):
    scaled=utility*temperature
    for none in (0.,2.,4.):
     # 対象自身を分母から除く。noneは「他のどの対象にも属さない」項。
     competitors=np.column_stack([logsumexp(scaled[:,valid[valid!=j]],axis=1) for j in range(len(data['teams']))])
     score=scaled-np.logaddexp(competitors,none)
     label=f'source{index}_{scope}_temperature{temperature:g}_none{none:g}';predictions[label]=score
     specs.append(dict(method=label,source_folder=folder,source_method=name,competitor_scope=scope,temperature=temperature,none_utility=none,quota_enforced=False))
 u.base.INPUT_HASHES[str(Path(__file__))]=u.sha(__file__)
 u.save(args.output_dir,data,predictions,specs,started,{'label_usage':'この後処理は会員ラベルを使わない。採点は公開正解。source5/6の元の予測は二方向交差検証による18対象の会員学習を使用。','adaptive_followup':True,'note':'全対象共通のスコア規則で、対象別の過去最良回答の選択・統合ではない。固定人数の割当ても行わない。'})
 # 元の学習済みOOF予測を使う処理なので、記録のラベル使用を明確にする。
 import json
 rp=args.output_dir/'実行記録.json';r=json.loads(rp.read_text());r['label_usage']=r['extra']['label_usage'];rp.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
