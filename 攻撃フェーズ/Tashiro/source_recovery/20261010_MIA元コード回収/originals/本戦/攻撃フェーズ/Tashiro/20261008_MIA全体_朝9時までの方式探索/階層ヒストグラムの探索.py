"""属性分布差の地域・性別条件付けと事前分布の強さを比較する。"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
import 共通 as u

STRENGTHS=(1,10,50,100,200,500,1000,5000)

def group_codes(frames,columns):
 if not columns:return [np.zeros(len(f),int) for f in frames],1
 combined=pd.concat([f[list(columns)] for f in frames],ignore_index=True);codes,_=pd.factorize(pd.MultiIndex.from_frame(combined),sort=True)
 result=[];start=0
 for frame in frames:result.append(codes[start:start+len(frame)]);start+=len(frame)
 return result,int(codes.max()+1)

def histograms(a,query,c,nums,cats,bins,group):
 frames=[a,query,c];gg,ng=group_codes(frames,group);result={s:[np.zeros(len(query)),np.zeros(len(query))] for s in STRENGTHS}
 for col in list(nums)+[col for col in cats if col not in group]:
  if col in nums:
   cuts=np.unique(np.quantile(a[col],np.arange(1,bins)/bins));codes=[np.searchsorted(cuts,f[col].to_numpy(),side='right') for f in frames];size=len(cuts)+1
  else:
   levels=sorted(set(a[col])|set(query[col])|set(c[col]));mapping={v:i for i,v in enumerate(levels)};codes=[f[col].map(mapping).to_numpy() for f in frames];size=len(levels)
  ac=np.bincount(gg[0]*size+codes[0],minlength=ng*size).reshape(ng,size);cc=np.bincount(gg[2]*size+codes[2],minlength=ng*size).reshape(ng,size)
  global_prior=(ac.sum(axis=0)+1)/(len(a)+size)
  # Aの小さい群も全体へ50人分寄せる。正解で選ばない固定値。
  expected=(ac+50*global_prior)/(ac.sum(axis=1,keepdims=True)+50)
  p=expected[gg[1],codes[1]];n=cc.sum(axis=1)
  observed=cc[gg[1],codes[1]];expected_count=n[gg[1]]*p
  for strength in STRENGTHS:
   shrunk=(observed+strength*p)/(n[gg[1]]+strength)
   result[strength][0]+=np.log(shrunk/p)
   # 正則化した頻度差。強さを変えると少数群の寄与が変わる。
   result[strength][1]+=(observed-expected_count)/np.sqrt((n[gg[1]]+strength)*p*(1-p)+1e-8)
 assert all(np.isfinite(v).all() for vals in result.values() for v in vals)
 return {f's{strength}_{kind}':values[i] for strength,values in result.items() for i,kind in enumerate(('logratio','zscore'))}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'階層ヒストグラム結果');args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();a=data['a'];banks={};specs=[]
 for label,nums,cats in [('q7',u.NUM7,u.CAT7),('q9',u.NUM9,u.CAT7),('q13',u.NUM13,u.CAT13)]:
  query=data['query'] if label=='q7' else data['b'];calibration=label=='q7'
  for group_label,group in [('all',[]),('sex',['sex']),('pref_sex',['prefecture','sex'])]:
   for bins in (5,10,20,40):
    for t,c in data['cs'].items():
     values=histograms(a,query,c,nums,cats,bins,group)
     scores={f'{label}_{group_label}_bins{bins}_{key}':v for key,v in values.items()}
     for name,score in u.variants(scores,data,t,calibration).items():banks.setdefault(name,{})[t]=score
     if t==data['teams'][0]:specs.extend(dict(raw_method=name,attributes=label,group=group_label,bins=bins,public_null_calibration=calibration) for name in scores)
    u.log(stage='hierarchical_histogram_complete',attributes=label,group=group_label,bins=bins,seconds=time.perf_counter()-started)
 u.base.INPUT_HASHES[str(Path(__file__))]=u.sha(__file__)
 u.save(args.output_dir,data,u.matrices(banks,data['teams']),specs,started,{'note':'数値の境界は公開Aのみで固定。A群の全体縮小は50。C群のAへの縮小は1..5000。会員正解で境界・群・強さを選ばない。','strengths':STRENGTHS,'adaptive_followup':True})

if __name__=='__main__':main()
