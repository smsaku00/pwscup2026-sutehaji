#!/usr/bin/env python3
"""Matched-target comparison with published column-top2 and exact baseline reproduction."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd

def run(prior_density,augmentation,calibration,out):
    old=pd.read_csv(augmentation)
    new=pd.read_csv(calibration)
    exact=old[(old.policy=='base_0') & (old.model.isin(['c_bg','c_bg_plus_local']))]
    actual=new[new.policy=='base_0']
    d=actual.merge(exact,on=['team','model'],suffixes=('_new','_prior'),validate='one_to_one')
    assert len(d)==42
    maxerrs={key:float(np.max(np.abs(d[f'{key}_new']-d[f'{key}_prior']))) for key in ('r_all','r_rare','r_max','hit_m','hit_c')}
    assert max(maxerrs.values())<1e-12,maxerrs
    prior=pd.read_csv(prior_density)
    rows=[]
    for scope,excl in [('nonself20',{5}),('processed18',{1,2,5})]:
        prev=prior[(prior.method=='prior_col2') & (prior.variant.isin(['base0','base11'])) & (~prior.team.isin(excl))]
        cal=new[(new.model=='c_bg_plus_local')&(~new.team.isin(excl))]
        for p in ['base_0','base_11','conservative_cap10']:
            for v in ('base0','base11'):
                rr=cal[cal.policy==p].set_index('team').r_all
                qq=prev[prev.variant==v].set_index('team').r_all
                difference=(rr-qq).sort_index()
                assert len(difference)==(20 if scope=='nonself20' else 18)
                rng=np.random.default_rng(20261009)
                ix=rng.integers(0,len(difference),size=(15000,len(difference)))
                lo,hi=np.quantile(difference.to_numpy()[ix].mean(axis=1),[.025,.975])
                rows.append(dict(scope=scope,policy=p,old_variant=v,old_score='prior_col2',
                                 mean_new=float(rr.mean()),mean_old=float(qq.mean()),mean_diff=float(difference.mean()),
                                 ci_low=float(lo),ci_high=float(hi),wins=int((difference>1e-10).sum()),
                                 losses=int((difference< -1e-10).sum()),ties=int((abs(difference)<=1e-10).sum())))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'comparison_prior_col2.csv',index=False)
    (out/'baseline_consistency.json').write_text(json.dumps({'matched_target_model_pairs':42,'max_score_errors':maxerrs,
                                           'published_density_added_baseline_exact':True},indent=2)+'\n')
    print('Baseline equality',maxerrs)
    print(pd.DataFrame(rows).query("scope == 'processed18' and old_variant=='base0'")[['policy','mean_new','mean_old','mean_diff','ci_low','ci_high']].to_string(index=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--prior-density',type=Path,required=True)
    ap.add_argument('--augmentation',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();run(a.prior_density,a.augmentation,a.calibration,a.out)
