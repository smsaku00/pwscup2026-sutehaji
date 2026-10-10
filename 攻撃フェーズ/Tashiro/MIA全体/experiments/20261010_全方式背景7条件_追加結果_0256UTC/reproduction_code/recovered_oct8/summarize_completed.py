"""Descriptive within-family summaries of already frozen, completed grids."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
BG=['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
def main():
 p=argparse.ArgumentParser();p.add_argument('--family',required=True);args=p.parse_args();out=HERE/args.family/'prelim'
 assert json.loads((out/'completion.json').read_text())['complete']
 s=pd.read_csv(out/'summary.csv',float_precision='round_trip');best=[];changes=[]
 for population in ('all_fixed','nonrare','normal'):
  for evaluation in ('all','without1_2'):
   sub=s[(s.population==population)&(s.evaluation==evaluation)]
   for bg in BG:
    v=sub[sub.background==bg].copy();top=v.mean_tpr.max();w=v[np.isclose(v.mean_tpr,top,atol=1e-14,rtol=0)].sort_values('method').iloc[0]
    best.append({'family':args.family,'background':bg,'population':population,'evaluation':evaluation,'candidate_count':len(v),'method':w.method,'mean_tpr':w.mean_tpr,'TPR_ge010':int(w.TPR_ge010),'ties_at_maximum':int(np.isclose(v.mean_tpr,top,atol=1e-14,rtol=0).sum()),'descriptive_selection_not_independent_validation':True})
   ref=sub[sub.background=='prelim'].set_index('method')
   for bg in BG:
    v=sub[sub.background==bg].set_index('method').loc[ref.index];delta=v.mean_tpr-ref.mean_tpr
    changes.append({'family':args.family,'background':bg,'reference_background':'prelim','population':population,'evaluation':evaluation,'settings':len(v),'improved':int((delta>1e-12).sum()),'equal':int((abs(delta)<=1e-12).sum()),'worsened':int((delta< -1e-12).sum()),'mean_recipe_delta':delta.mean(),'min_delta':delta.min(),'max_delta':delta.max()})
 pd.DataFrame(best).to_csv(out/'within_family_maxima_descriptive.csv',index=False,float_format='%.17g')
 pd.DataFrame(changes).to_csv(out/'same_setting_background_changes.csv',index=False,float_format='%.17g')
 if args.family=='mixture':
  diagnostics=[]
  for bg in BG:
   f=pd.read_csv(out/bg/'fit_diagnostics.csv')
   for source,g in f.groupby('source'):
    diagnostics.append({'background':bg,'source':source,'fits':len(g),'nonconverged':int((~g.converged).sum()),'max_iterations':int(g.iterations.max()),'fits_with_warnings':int(g.warnings.fillna('').ne('').sum())})
  pd.DataFrame(diagnostics).to_csv(out/'convergence_summary.csv',index=False)
 print(pd.DataFrame(best).query("population=='all_fixed' and evaluation=='without1_2'").to_string(index=False))
if __name__=='__main__':main()
