from pathlib import Path
import json,sys,os
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
M=json.load(open(os.environ.get('MIA_INPUT_MANIFEST','/workspace/shared/mia_bg7_inputs/manifest.json')))
R=lambda p:pd.read_csv(p,float_precision='round_trip')
QI=M['qi7'];FULL=M['features13'];NUM=[c for c in FULL if c!='prefecture'];LAB=['BMI','SBP','TG','HDL','ALT','FPG']
bg=pd.concat([R(v['path']) for v in M['backgrounds'].values()],ignore_index=True)
s=R(HERE/'team_processing_comparison_metrics.csv');res=[]
def rowsig(x,cols,d):
 x=x[cols].copy()
 for c in set(cols)&set(NUM):x[c]=x[c].astype(float).round(d)
 return list(x.itertuples(index=False,name=None))
for _,r in s.iterrows():
 for phase,t in [('prelim',r.prelim_cohort),('main',r.main_cohort)]:
  if pd.isna(t):continue
  p=Path(M['phases'][phase]['attack_dir'])/f'C_{int(t)}.csv'
  if not p.exists():continue
  c=R(p)
  for decimals in [8,6,3]:
   a={'team_name':r.team_name,'phase':phase,'decimals':decimals}
   for cols,label in [(FULL,'full'),(QI,'qi7')]:
    ref=set(rowsig(bg,cols,decimals));a[f'bg_overlap_{label}_n']=sum(x in ref for x in rowsig(c,cols,decimals))
   for col in LAB:a[f'bg_value_reuse_{col}']=c[col].round(decimals).isin(bg[col].round(decimals)).mean()
   res.append(a)
pd.DataFrame(res).to_csv(HERE/'rounding_overlap_robustness.csv',index=False)
# Available exact B comparison for the user's own team only.
a=[]
for phase in ['prelim','main']:
 t=M['phases'][phase]['self_id'];b=R(M['phases'][phase]['b_self']);c=R(Path(M['phases'][phase]['attack_dir'])/f'C_{t}.csv')
 for col in FULL:
  bp=b[col].astype(str).value_counts(normalize=True);cp=c[col].astype(str).value_counts(normalize=True)
  a.append({'team_name':'ステテコは恥だが役に立つ','phase':phase,'column':col,'B_nunique':b[col].nunique(),'C_nunique':c[col].nunique(),'B_C_marginal_TV':bp.subtract(cp,fill_value=0).abs().sum()/2,'B_mean':b[col].mean() if col in NUM else None,'C_mean':c[col].mean() if col in NUM else None})
pd.DataFrame(a).to_csv(HERE/'own_team_exact_B_C_comparison.csv',index=False)
print(pd.DataFrame(res).query('decimals==6')[['team_name','phase','bg_overlap_full_n','bg_overlap_qi7_n']].to_string(index=False))
