"""Paired descriptive BG comparisons for fully sealed fixed-recipe families."""
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
import runtime as r
from run_family import STAGE
from run_queue import complete,COUNTS

def main():
    frames=[];families=[]
    for family in COUNTS:
        if family in ('learned_main_transfer','learned_multiview_main'):continue
        if not all(complete(family,bg) and (STAGE/family/bg/'completion.json').exists() for bg in r.BG):continue
        families.append(family)
        for bg in r.BG:
            tab=pd.read_csv(STAGE/family/bg/'対象別TPR.csv');tab=tab[tab.fpr==.01].copy();tab['family']=family;tab['background']=bg;frames.append(tab)
    if not frames:return
    d=pd.concat(frames,ignore_index=True);base=d[d.background=='prelim'][['family','method','team','mia_all']].rename(columns={'mia_all':'prelim_TPR'})
    paired=d.merge(base,on=['family','method','team'],validate='many_to_one');paired['delta']=paired.mia_all-paired.prelim_TPR
    out=STAGE/'completed_panel_comparison'
    if out.exists():out=STAGE/('completed_panel_comparison_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()))
    out.mkdir(exist_ok=False)
    crosses=paired[(paired.prelim_TPR<.1-1e-12)&(paired.mia_all>=.1-1e-12)]
    crosses.to_csv(out/'same_recipe_new_010_crossings.csv',index=False,float_format='%.17g')
    rows=[]
    for team,g in d.groupby('team'):
        old=g[g.background=='prelim'].sort_values(['mia_all','family','method'],ascending=[False,True,True]).iloc[0]
        new=g[g.background!='prelim'].sort_values(['mia_all','family','method','background'],ascending=[False,True,True,True]).iloc[0]
        rows.append(dict(team=int(team),old_panel_best_TPR=old.mia_all,old_family=old.family,old_method=old.method,new_panel_best_TPR=new.mia_all,new_family=new.family,new_method=new.method,new_background=new.background,panel_best_change=new.mia_all-old.mia_all,selection='Post-evaluation best within these completed panels; not an all-history record or validated selector'))
    pd.DataFrame(rows).to_csv(out/'per_team_panel_best.csv',index=False,float_format='%.17g')
    methods=[]
    for (family,method,bg),g in d.groupby(['family','method','background']):
        methods.append(dict(family=family,method=method,background=bg,all20=g.mia_all.mean(),without1_2=g[~g.team.isin([1,2])].mia_all.mean()))
    m=pd.DataFrame(methods);m.to_csv(out/'method_background_means.csv',index=False,float_format='%.17g')
    robust=m.groupby(['family','method']).agg(background_conditions=('background','size'),mean_condition_all20=('all20','mean'),minimum_condition_all20=('all20','min'),maximum_condition_all20=('all20','max'),mean_condition_without1_2=('without1_2','mean'),minimum_condition_without1_2=('without1_2','min'),maximum_condition_without1_2=('without1_2','max')).reset_index()
    robust.sort_values(['minimum_condition_without1_2','mean_condition_without1_2'],ascending=False).to_csv(out/'descriptive_bg_robustness.csv',index=False,float_format='%.17g')
    m.groupby('background')[['all20','without1_2']].mean().to_csv(out/'mean_over_fixed_configs.csv',float_format='%.17g')
    record=dict(complete_request=False,complete_families=families,source_recipe_BG_evaluations=len(m),rows=len(d),same_recipe_crossing_targets=sorted(map(int,crosses.team.unique())),snapshot_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),interpretation='Seven overlapping background source conditions, not seven independent replications. Selection across methods/BGs is descriptive and uses preliminary evaluation labels. Historical adaptively selected feature recipes remain exploratory. Old means here use the same current numerical execution protocol and fixed recipe as the changed-BG run; historical catalog comparisons are separate.')
    (out/'comparison_record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record));print(robust.sort_values(['minimum_condition_without1_2','mean_condition_without1_2'],ascending=False).head(3).to_string(index=False));print(pd.DataFrame(rows).query('panel_best_change > 1e-12').to_string(index=False))
if __name__=='__main__':main()
