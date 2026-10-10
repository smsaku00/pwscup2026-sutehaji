"""Report the original .16 anchors unchanged across the seven background pools."""
from pathlib import Path
import json
import pandas as pd
HERE=Path(__file__).resolve().parent
ANCHORS={'histogram':[(7,'q7_sex_bins40_s10_logratio__county500__cross0'),(7,'q7_sex_bins40_s1_logratio__county500__cross0')],'alignment':[(26,'q13_quantile_sex1_kde_ratio_h0.5__raw__cross0')],'local':[(26,'q13_robust_adaptivek5_h0.5__raw__cross0')]}
rows=[]
for family,anchors in ANCHORS.items():
 out=HERE/family/'prelim'
 if not (out/'completion.json').exists():continue
 for bg in ['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']:
  x=pd.read_csv(out/bg/'by_target.csv',float_precision='round_trip')
  for team,method in anchors:
   hit=x[x.team.eq(team)&x.method.eq(method)].copy();assert len(hit)==3
   hit['historical_anchor_all1049_prelim']=.16;rows.append(hit)
pd.concat(rows,ignore_index=True).to_csv(HERE/'historical_016_anchors_backgrounds.csv',index=False,float_format='%.17g')
print(pd.concat(rows,ignore_index=True).query("population=='all_fixed'")[['family','background','method','team','tpr_0.01']].to_string(index=False))
