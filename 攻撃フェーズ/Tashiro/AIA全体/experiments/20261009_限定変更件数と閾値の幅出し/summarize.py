#!/usr/bin/env python3
"""Generate reviewable, small scope summaries from the sweep without re-optimizing."""
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
SEED=20261009
parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,default=Path(__file__).parent/'results')
P=parser.parse_args().results
a=pd.read_csv(P/'grid_summary.csv');b=pd.read_csv(P/'grid_bootstrap.csv');t=pd.read_csv(P/'per_team_grid.csv.gz');m=pd.read_csv(P/'nested_selection.csv');base=pd.read_csv(P/'baseline.csv')
ss=a[(a.folds==4)&(a.family=='down11')&(a.margin==.02)][['scope','budget','r_all','r_rare','r_max','mean_changed','total_changed','gained_m','lost_m','gained_c','lost_c','delta_r_all','gain_loss_net']]
ss.to_csv(P/'budget_curve.csv',index=False)
a[(a.folds==4)&(a.family=='down11')&(a.budget==10)][['scope','margin','r_all','delta_r_all','mean_changed','gain_loss_net']].to_csv(P/'threshold_curve.csv',index=False)
a[(a.folds==4)&(a.budget.isin([5,10,15]))&(a.margin.isin([0,.02]))][['scope','family','budget','margin','r_all','delta_r_all','mean_changed','gain_loss_net']].to_csv(P/'action_comparison.csv',index=False)
a[(a.family=='down11')&(a.budget.isin([5,8,10,15,20]))&(a.margin==.02)][['scope','folds','budget','r_all','delta_r_all','mean_changed']].to_csv(P/'fold_sensitivity.csv',index=False)
b[(b.family=='down11')&(b.margin==.02)&(b.budget.isin([5,8,10,15,20]))].to_csv(P/'focus_bootstrap.csv',index=False)
focus=t[(t.folds==4)&(t.family=='down11')&(t.margin==.02)&(t.budget.isin([0,8,10,15,20,30,50]))][['team','budget','r_all','r_rare','r_max','changed','hit_m','hit_c','lost_m','lost_c']]
focus.to_csv(P/'per_team_focus.csv',index=False)
meta=[];rng=np.random.default_rng(SEED)
for scope,omit in [('nonself20',{5}),('processed18',{1,2,5})]:
 d=m[m.scope==scope].set_index('team').sort_index()
 ref=base[(base.folds==4)&(~base.team.isin(omit))].set_index('team').loc[d.index]
 delta=(d.r_all-ref.r_all).to_numpy()
 samp=rng.integers(0,len(delta),size=(25000,len(delta)))
 ci=np.quantile(delta[samp].mean(axis=1),[.025,.975])
 meta.append(dict(scope=scope,n=len(delta),policy='inner_oof_LOCO_selection',mean_r=float(d.r_all.mean()),base=float(ref.r_all.mean()),delta=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),wins=int((delta>1e-12).sum()),losses=int((delta< -1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()),mean_changed=float(d.changed.mean()),selected_budget_count=json.dumps({int(k):int(v) for k,v in d.budget.value_counts().items()},ensure_ascii=False)))
pd.DataFrame(meta).to_csv(P/'nested_meta_summary.csv',index=False)
print('BUDGET_CURVE:')
print(ss[(ss.scope=='nonself20')&(ss.budget.isin([0,1,2,5,8,10,15,20,30,50]))].to_string(index=False,float_format=lambda v:f'{v:.6f}'))
print('META:',pd.DataFrame(meta).to_string(index=False,float_format=lambda v:f'{v:.6f}'))
print('FAMILY COMPARISON:')
print(a[(a.scope=='nonself20')&(a.folds==4)&(a.budget==10)&(a.margin.isin([0,.02]))][['family','budget','margin','r_all','delta_r_all','mean_changed','gained_m','lost_m','gained_c','lost_c']].to_string(index=False,float_format=lambda v:f'{v:.5f}'))
