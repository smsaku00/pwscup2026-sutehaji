from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
BASE=Path(__file__).resolve().parent
CAT=Path('/workspace/shared/mia_recovered_sources/攻撃フェーズ/Tashiro/source_recovery/20261010_MIA元コード回収/catalog/方式条件一覧.csv')
files=[p for p in (BASE/'results').glob('*/*/by_target.csv') if (p.parent/'completion.json').exists()]
if not files:raise SystemExit('no completed results')
d=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in files],ignore_index=True)
d['target_phase']='prelim'
d.to_csv(BASE/'by_target.csv',index=False,float_format='%.17g')
coverage=d.groupby(['family','method','background']).agg(n_targets=('team','size')).reset_index();coverage['background_dependent']=~coverage.method.isin(['legacy_library_exact','legacy_redteam_event_time_exact']);coverage.to_csv(BASE/'coverage.csv',index=False)

s=pd.concat([pd.read_csv(p.parent/'summary.csv',float_precision='round_trip') for p in files],ignore_index=True);s['target_phase']='prelim';s.to_csv(BASE/'summary.csv',index=False,float_format='%.17g')
cat=pd.read_csv(CAT,float_precision='round_trip');old=cat[cat.method.isin(d.method.unique())]
checks=[]
for r in old.itertuples():
 teams=[int(t) for t in str(r.evaluated_teams).split(',')]
 rows=d[(d.background=='prelim')&d.method.eq(r.method)&d.team.isin(teams)]
 if len(rows)!=len(teams):continue
 now=rows.tpr.mean();checks.append(dict(method=r.method,condition_id=r.condition_id,source_ids=r.source_ids,n_targets=len(teams),historical_mean_tpr=r.mean_tpr,current_mean_tpr=now,error=now-r.mean_tpr,match=abs(now-r.mean_tpr)<1e-12))
pd.DataFrame(checks).to_csv(BASE/'baseline_validation.csv',index=False,float_format='%.17g')
base=d[d.background=='prelim'][['method','team','tpr','auc']].rename(columns={'tpr':'baseline_tpr','auc':'baseline_auc'})
c=d.merge(base,on=['method','team']);c['tpr_delta']=c.tpr-c.baseline_tpr;c['newly_at_least_010']=(c.baseline_tpr<.1)&(c.tpr>=.1);c.to_csv(BASE/'background_changes.csv',index=False,float_format='%.17g')
c=c[(~c.team.isin([1,2]))&(c.background!='prelim')];c.sort_values(['tpr_delta','tpr'],ascending=False).to_csv(BASE/'non12_same_config_gains.csv',index=False,float_format='%.17g')
status=dict(completed_family_background_jobs=len(files),methods_current=int(d.method.nunique()),target_rows=len(d),baseline_checked=len(checks),baseline_matches=int(sum(r['match'] for r in checks)),baseline_max_abs_error=max([abs(r['error']) for r in checks],default=0),backgrounds=sorted(d.background.unique()),expected_primary_bg_dependent_methods=99,expected_controls=2,completed_method_background_conditions=len(coverage),completed_bg_dependent_conditions=int(coverage.background_dependent.sum()),expected_bg_dependent_conditions=693)
(BASE/'aggregate_status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status));print(pd.DataFrame(checks).query('not match').to_string(index=False) if checks else '')
