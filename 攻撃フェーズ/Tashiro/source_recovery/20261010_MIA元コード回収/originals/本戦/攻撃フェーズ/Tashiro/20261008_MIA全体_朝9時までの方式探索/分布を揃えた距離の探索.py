"""Cの全体的な尺度ずれを公開Aで補正し、距離・密度比を評価する。"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
import 共通 as u
from 距離と分布の探索 import transform,distances,nearest,geometry

def align(a,c,cols,kind,by_sex=False):
 result=c.copy()
 sexes=sorted(c.sex.unique()) if by_sex else [None]
 for sex in sexes:
  mask=np.ones(len(c),dtype=bool) if sex is None else c.sex.to_numpy()==sex
  aa=a if sex is None else a[a.sex==sex]
  if mask.sum()<10 or len(aa)<10:continue
  for col in cols:
   raw=c.loc[mask,col].to_numpy(float);reference=aa[col].to_numpy(float)
   log=col in ('TG','ALT')
   if log:raw=np.log1p(raw);reference=np.log1p(reference)
   if kind=='quantile':
    ordered,counts=np.unique(raw,return_counts=True)
    p=(np.cumsum(counts)-.5*counts)/len(raw)
    probability=np.interp(raw,ordered,p)
    value=np.quantile(reference,probability)
   else:
    if kind=='mean_sd':
     center_c,center_a=raw.mean(),reference.mean();scale_c,scale_a=raw.std(),reference.std()
    else:
     center_c,center_a=np.median(raw),np.median(reference)
     scale_c=np.quantile(raw,.75)-np.quantile(raw,.25);scale_a=np.quantile(reference,.75)-np.quantile(reference,.25)
    value=raw-center_c+center_a if scale_c<1e-8 or scale_a<1e-8 else (raw-center_c)*scale_a/scale_c+center_a
   if log:value=np.expm1(value)
   result.loc[mask,col]=value
 assert np.isfinite(result[list(cols)].to_numpy(float)).all()
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output-dir',type=Path,default=u.HERE/'分布を揃えた距離結果')
 args=parser.parse_args();assert not args.output_dir.exists()
 started=time.perf_counter();data=u.setup();banks={};specs=[]
 for p in [Path(__file__),u.HERE/'距離と分布の探索.py']:u.base.INPUT_HASHES[str(p.resolve())]=u.sha(p)
 settings=[('median_iqr',False),('mean_sd',False),('quantile',False),('median_iqr',True),('quantile',True)]
 for attributes,nums,cats in [('q7',u.NUM7,u.CAT7),('q13',u.NUM13,u.CAT13)]:
  query=data['query'] if attributes=='q7' else data['b'];calibration=attributes=='q7'
  ar,x=transform(data['a'],[data['a'],query],nums,'robust')
  ref_near=nearest(distances(x,ar,query,data['a'],cats,1),150)
  for kind,by_sex in settings:
   for index,(t,c) in enumerate(data['cs'].items(),1):
    aligned=align(data['a'],c,nums,kind,by_sex)
    label=f'{attributes}_{kind}_sex{int(by_sex)}'
    scores={label+'_'+n:v for n,v in geometry(data['a'],query,aligned,nums,cats,'robust',1,ref_near).items()}
    if attributes=='q7':scores[label+'_gen7']=u.base.core(data['a'],query,aligned,u.NUM7,(10,))[10]
    for name,value in u.variants(scores,data,t,calibration).items():banks.setdefault(name,{})[t]=value
    if index==1:
     for name in scores:specs.append(dict(raw_method=name,family='public_distribution_alignment',align=kind,sex_groups=by_sex,attributes=attributes,uses_public_null=calibration,
                                       hypothesis='Released data may have a global numeric shift; alignment may recover proximity but can also remove genuine cohort distribution differences.'))
   u.log(stage='alignment_complete',attributes=attributes,align=kind,sex_groups=by_sex,seconds=time.perf_counter()-started)
 predictions=u.matrices(banks,data['teams'])
 u.log(stage='predictions_fixed',raw_specs=len(specs),models=len(predictions),seconds=time.perf_counter()-started)
 u.save(args.output_dir,data,predictions,specs,started,{'no_membership_labels_in_alignment':True,'categories':'sex/prefecture/smoking/onset/death are unchanged. Numeric TG/ALT alignment is on log scale.'})

if __name__=='__main__':main()
