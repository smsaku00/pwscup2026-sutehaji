from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;C=Path('/workspace/shared/mia_all_bg_recovered_calibration/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_calibration');P=C/'completed_panel_comparison'
read=lambda p:pd.read_csv(p,float_precision='round_trip')
record=json.loads((P/'comparison_record.json').read_text());before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file()}
frames=[]
for family in record['complete_families']:
 for bg in ['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']:
  assert json.loads((C/family/bg/'completion.json').read_text())['source_main_returned']
  d=read(C/family/bg/'対象別TPR.csv');d=d[d.fpr==.01].copy();d['family']=family;d['background']=bg;frames.append(d)
d=pd.concat(frames,ignore_index=True);keys=['family','method','team'];assert not d.duplicated(keys+['background']).any();assert len(d)==record['rows'];look=d.set_index(keys+['background']).mia_all
base=d[d.background=='prelim'].set_index(keys).mia_all
paired=d.join(base.rename('reference'),on=keys)
wanted=paired[(paired.reference<.1-1e-12)&(paired.mia_all>=.1-1e-12)]
x=read(P/'same_recipe_new_010_crossings.csv');assert set(map(tuple,x[keys+['background']].to_numpy()))==set(map(tuple,wanted[keys+['background']].to_numpy()))
errs=[]
for row in x.itertuples():
 k=(row.family,row.method,row.team);expected=look.loc[k+(row.background,)];old=base.loc[k]
 errs.extend([abs(row.mia_all-expected),abs(row.prelim_TPR-old),abs(row.delta-(expected-old))])
best=read(P/'per_team_panel_best.csv');assert set(best.team)==set(d.team)
for row in best.itertuples():
 old=base.xs(row.team,level='team').max();new=d[(d.team==row.team)&(d.background!='prelim')].mia_all.max()
 errs.extend([abs(row.old_panel_best_TPR-old),abs(row.new_panel_best_TPR-new),abs(row.panel_best_change-(new-old)),abs(look.loc[(row.old_family,row.old_method,row.team,'prelim')]-old),abs(look.loc[(row.new_family,row.new_method,row.team,row.new_background)]-new)])
agg=d.groupby(['family','method','background']).mia_all.mean().rename('all20').to_frame().join(d[~d.team.isin([1,2])].groupby(['family','method','background']).mia_all.mean().rename('without1_2'))
m=read(P/'method_background_means.csv').set_index(['family','method','background']);assert set(m.index)==set(agg.index);errs.extend(abs(m.loc[agg.index,['all20','without1_2']].to_numpy()-agg.to_numpy()).ravel())
w=read(P/'descriptive_bg_robustness.csv').set_index(['family','method']);assert len(w)==record['source_recipe_BG_evaluations']//7
for (fam,method),g in agg.groupby(level=[0,1]):
 row=w.loc[(fam,method)];assert row.background_conditions==7
 for suffix,col in [('all20','all20'),('without1_2','without1_2')]:
  for pref,fn in [('mean',np.mean),('minimum',np.min),('maximum',np.max)]:errs.append(abs(row[f'{pref}_condition_{suffix}']-fn(g[col])))
avg=read(P/'mean_over_fixed_configs.csv').set_index('background');expected=agg.groupby(level='background').mean();errs.extend(abs(avg.loc[expected.index].to_numpy()-expected.to_numpy()).ravel())
assert max(errs)<1e-12 and before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file()}
out={'status':'passed','audit_time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'families':record['complete_families'],'source_recipe_background_rows':len(agg),'source_target_rows':len(d),'paired_crossing_rows':len(x),'panel_best_target_rows':len(best),'numeric_values_checked':len(errs),'maximum_absolute_error':float(max(errs)),'scope':'Retrospective within-panel BG comparisons only. Distinct source grids and overlapping background pools are not independent replications; these maxima are not all-history records or a validated selector.','files_sha256':before}
(R/'inventory/completed_panel_comparison_independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files_sha256'},ensure_ascii=False))
