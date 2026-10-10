"""Truth-free preliminary score/rank stability against the main-source BG.

These are preliminary candidates with main as a background source, not main
competition candidate scores. No record-level array is exported.
"""
import json
from functools import lru_cache
import numpy as np,pandas as pd
from scipy.stats import spearmanr
import runtime as r
from evaluate_public_policies import ensure_kdeb,metadata

@lru_cache(None)
def context(bg):
 d=r.data(bg);bank=r.load_scores(r.HERE/'cache/p07'/bg/'full');bank['KDEB']=ensure_kdeb(bg,d)
 parts=[]
 for p in (r.HERE/'results_public'/bg).glob('selection_*.csv'):parts.append(pd.read_csv(p))
 choice=pd.concat(parts,ignore_index=True).drop_duplicates(['family','policy','team'])
 return d,bank,choice
@lru_cache(None)
def p03(bg):
 d,_,_=context(bg);ix=[j for j,t in enumerate(d['teams']) if t not in(1,2)]
 cal=r.raw_cal(bg,d);dist=r.p06.dist_to_C(r.u.base,d['b'],d['a'],d['cs'],d['teams'])
 return {t:x for t,x in zip([d['teams'][j] for j in ix],(r.cross(r.base.ranks(cal[d['normal']][:,ix]))+.1*r.base.ranks(-dist[d['normal']][:,ix])).T)}
@lru_cache(None)
def scores(folder):return r.load_scores(r.HERE/folder)
@lru_cache(None)
def frame(path):return pd.read_csv(r.HERE/path)
def output(row,bg):
 d,bank,choice=context(bg);teams=d['teams'];f=row['family'];p=row['policy'];view=row['view'];result={}
 if f in ('p05','p06','p07'):
  z=choice[(choice.family==f)&(choice.policy==p)]
  if f=='p06' and view!='full':
   bank=scores(f'cache/p06/{bg}/{view}');t=frame(f'results_public/{bg}/p06_by_target.csv');eligible=set(t[(t.policy==p)&(t.view==view)].team);z=z[z.team.isin(eligible)]
  for x in z.itertuples():
   result[x.team]=p03(bg)[x.team] if x.selected_family=='P03_native18' else bank[x.selected_family][:,teams.index(x.team)]
 elif f in ('transfer','nested','nested_safety','nested_fixed72'):
  z=frame(f'results_transfer/{bg}/selection_audit.csv');z=z[(z.family==f)&(z.policy==p)];bank=scores(f'cache/fixed22/{bg}/full')
  result={x.team:bank[x.selected_method][:,teams.index(x.team)] for x in z.itertuples()}
 elif f=='p08d_looto':
  z=frame(f'cache/p08d/{bg}/selection_audit.csv');z=z[z.policy==p];bank=scores(f'cache/p08d/{bg}/full');small=[t for t in teams if t not in(1,2)]
  result={x.team:bank[x.chosen_method][:,small.index(x.team)] for x in z.itertuples()}
 elif f=='p07_Q_omission':
  x=scores(f'cache/p07/{bg}/{view}')[p];result=dict(zip(teams,x.T))
 elif f=='p07_Q_omission_composite':
  h=row['parameters']['heldout_Q'];omit=scores(f'cache/p07/{bg}/hold_{h}')['P01'];_,groups,cuts=metadata(bg,False);assign=r.assignments(groups.reset_index(),cuts)
  result={t:(omit if assign[t]=='P01' else bank[assign[t]])[:,teams.index(t)] for t in r.p05.TEAM17}
 else:raise ValueError(f)
 return {t:(x[d['normal']] if len(x)==len(d['b']) else x) for t,x in result.items()}

def run():
 registry=[json.loads(x) for x in (r.HERE/'registry/outputs.jsonl').read_text().splitlines()];rows=[];hashes=[]
 for row in registry:
  reference=output(row,'main')
  for bg in r.BG:
   actual=output(row,bg);common=sorted(set(actual)&set(reference));digest=r.msha(np.column_stack([actual[t] for t in sorted(actual)])) if actual else None
   hashes.append({'config_id':row['config_id'],'background':bg,'targets':';'.join(map(str,sorted(actual))),'normal_matrix_sha256':digest})
   for t in common:
    x,y=actual[t],reference[t];assert len(x)==len(y)==875
    tx=set(np.argsort(-x,kind='stable')[:20]);ty=set(np.argsort(-y,kind='stable')[:20]);ux=len(np.unique(x));uy=len(np.unique(y));rho=float(spearmanr(x,y).statistic) if min(ux,uy)>1 else np.nan
    rows.append({'config_id':row['config_id'],'family':row['family'],'policy':row['policy'],'view':row['view'],'background':bg,'team':t,'candidate_phase':'prelim','reference_background':'main','spearman':rho,'spearman_defined':min(ux,uy)>1,'score_distinct_values':ux,'reference_distinct_values':uy,'top20_boundary_ties':int((x==np.sort(x)[-20]).sum()),'reference_top20_boundary_ties':int((y==np.sort(y)[-20]).sum()),'top20_overlap':len(tx&ty),'native_targets':len(actual),'reference_targets':len(reference),'comparable_targets':len(common)})
 out=r.HERE/'rank_stability';out.mkdir(exist_ok=True);tab=pd.DataFrame(rows);tab.to_csv(out/'by_target.csv',index=False,float_format='%.17g')
 tab.groupby(['config_id','family','policy','view','background'],sort=False).agg(targets=('team','size'),spearman_valid_targets=('spearman','count'),mean_spearman=('spearman','mean'),mean_top20_overlap=('top20_overlap','mean')).reset_index().to_csv(out/'summary.csv',index=False,float_format='%.17g')
 pd.DataFrame(hashes).to_csv(out/'normal_prediction_hashes.csv',index=False)
 r.dump(out/'protocol.json',{'truth_read':False,'candidate_phase':'prelim','reference_BG':'main','not_main_competition_predictions':True,'top20_ties':'stable original B row order','scope':'intersection of native target sets; P06 eligibility may vary by BG','output_registration_count':len(registry),'score_construction':'same sealed original models and frozen choices used for metric reports'})
 print('RANK_STABILITY_DONE',len(rows),len(hashes))
if __name__=='__main__':run()
