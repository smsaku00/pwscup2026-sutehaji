"""Check rank-report reconstruction against the sealed per-target evaluations."""
import json,argparse
import numpy as np,pandas as pd
import runtime as r
import rank_policy_backgrounds as k

def run(backgrounds):
 registry=[json.loads(x) for x in (r.HERE/'registry/outputs.jsonl').read_text().splitlines()];coverage=pd.read_csv(r.HERE/'registry/coverage.csv');cache={};checks=[]
 for bg in backgrounds:
  d,_,_=k.context(bg);y=r.truth(d);errors=[];total=0
  for row in registry:
   path=coverage[(coverage.config_id==row['config_id'])&(coverage.background==bg)].iloc[0].result_path
   if path not in cache:cache[path]=pd.read_csv(r.HERE/path)
   tab=cache[path];vcol='condition' if 'condition' in tab else 'view' if 'view' in tab else 'background';tab=tab[(tab.policy==row['policy'])&(tab[vcol]==row['view'])&(tab.population=='normal')]
   if 'family' in tab:tab=tab[tab.family==row['family']]
   expected=tab.set_index('team')['tpr_0.01'];actual=k.output(row,bg);assert set(actual)==set(expected.index),(row,bg)
   for t,x in actual.items():
    val=r.base.metrics(x,y[d['normal'],d['teams'].index(t)])['tpr_0.01'];errors.append(abs(val-expected.loc[t]));total+=1
  error=max(errors);assert error<1e-12,(bg,error);checks.append({'background':bg,'registrations':len(registry),'team_scores_checked':total,'max_TPR_error':error})
 r.dump(r.HERE/'validation/rank_reconstruction_tests.json',checks);print('RANK_RECONSTRUCTION_OK',backgrounds)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--backgrounds',nargs='+',default=list(r.BG));a=p.parse_args();run(a.backgrounds)
