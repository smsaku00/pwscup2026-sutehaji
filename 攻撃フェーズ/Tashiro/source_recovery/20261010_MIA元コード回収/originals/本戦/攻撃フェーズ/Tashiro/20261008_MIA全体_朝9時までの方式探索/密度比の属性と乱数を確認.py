"""初回で良かった密度比の属性依存・ランダム中心依存を追試する。"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import 共通 as u
from 距離と分布の探索 import transform
from 直接密度比の探索 import kernel_dist

SEEDS=(20261008,20261009,20261010,20261011,20261012,20261013,20261014,20261015,20261016,20261017)
GROUPS=[('all13',u.NUM13,u.CAT13),('without_time',u.NUM9,u.CAT13),
        ('without_outcomes',u.NUM9,u.CAT7+('smoking',)),('quasi9',u.NUM9,u.CAT7),
        ('quasi7',u.NUM7,u.CAT7),('quasi7_time',u.NUM7+('time',),u.CAT7),
        ('numeric8_sexpref',u.NUM13,u.CAT7),('time_categorical',('time',),u.CAT13)]

def one_score(a,c,query,nums,cats,seed,centers=100,h=1.,relative=.1,regularizer=1.):
 ar,cr,x=transform(a,[a,c,query],nums,'robust')
 ii=np.random.default_rng(seed).permutation(len(c))[:centers];f=c.iloc[ii];z=cr[ii]
 ka=np.exp(-.5*kernel_dist(ar,z,a,f,cats)/h**2)
 kc=np.exp(-.5*kernel_dist(cr,z,c,f,cats)/h**2)
 kx=np.exp(-.5*kernel_dist(x,z,query,f,cats)/h**2)
 matrix=(1-relative)*(ka.T@ka/len(a))+relative*(kc.T@kc/len(c))
 beta=np.maximum(0,np.linalg.solve(matrix+regularizer*np.eye(centers),kc.mean(axis=0)))
 return kx@beta

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=u.HERE/'密度比_属性乱数結果')
 args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();banks={};specs=[]
 for p in [Path(__file__),u.HERE/'距離と分布の探索.py',u.HERE/'直接密度比の探索.py']:u.base.INPUT_HASHES[str(p.resolve())]=u.sha(p)
 for label,nums,cats in GROUPS:
  query=data['query'] if label=='quasi7' else data['b'];calibration=label=='quasi7'
  for index,(t,c) in enumerate(data['cs'].items(),1):
   draws=np.stack([one_score(data['a'],c,query,nums,cats,s+t) for s in SEEDS])
   ranks=np.stack([rankdata(x,method='average')/len(x) for x in draws])
   scores={f'{label}_seed{s}':draws[i] for i,s in enumerate(SEEDS)}
   scores.update({f'{label}_raw_mean':draws.mean(axis=0),f'{label}_raw_median':np.median(draws,axis=0),
                  f'{label}_raw_q20':np.quantile(draws,.2,axis=0),f'{label}_raw_mean_minus_sd':draws.mean(axis=0)-draws.std(axis=0),
                  f'{label}_rank_mean':ranks.mean(axis=0),f'{label}_rank_median':np.median(ranks,axis=0)})
   for name,v in u.variants(scores,data,t,calibration).items():banks.setdefault(name,{})[t]=v
   if index==1:
    for name in scores:specs.append(dict(raw_method=name,family='ulsif_ablation_bagging',num_attributes=';'.join(nums),cat_attributes=';'.join(cats),
                                      h=1.,relative=.1,regularizer=1.,centers=100,seeds=';'.join(map(str,SEEDS)),uses_public_null=calibration))
  u.log(stage='ablation_complete',group=label,seconds=time.perf_counter()-started)
 models=u.matrices(banks,data['teams'])
 u.log(stage='predictions_fixed',raw_specs=len(specs),models=len(models),seconds=time.perf_counter()-started)
 table,_=u.save(args.output_dir,data,models,specs,started,{'adaptive_followup':'初回結果で13属性・中心100・h1・lambda1が良かったことを見た後の追試。独立した検証ではない。','seeds':SEEDS})
 # seed20261008の全13属性は先の固有値分解版と同じ数学式。
 old=u.base.read(u.HERE/'直接密度比結果/対象別TPR.csv')
 expected=old[old.method=='q13_robust_ulsif_centers100_h1_a0.1_lambda1__raw__cross0'].set_index(['team','fpr']).mia_all
 actual=table[table.method=='all13_seed20261008__raw__cross0'].set_index(['team','fpr']).mia_all
 error=float(np.max(np.abs(expected-actual)));assert error<1e-12
 u.log(stage='solve_implementation_reproduced',maximum_error=error)
 (args.output_dir/'実装照合.json').write_text(__import__('json').dumps({'maximum_tpr_error':error})+'\n')

if __name__=='__main__':main()
