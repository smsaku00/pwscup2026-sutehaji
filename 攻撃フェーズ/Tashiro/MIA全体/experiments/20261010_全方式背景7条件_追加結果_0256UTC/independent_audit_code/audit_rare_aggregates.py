"""Independently recompute rare diagnostic aggregates; no membership inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
L=Path('/workspace/shared/mia_all_bg_legacy/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/legacy/aggregate')
read=lambda p:pd.read_csv(p,float_precision='round_trip')
d=read(L/'rare_diagnostics_all.csv')
assert len(d)==11088 and d.background.nunique()==7 and d.method.nunique()==3
assert not d.duplicated(['background','method','team','record_id']).any()
table=read(L/'rare_by_target_aggregate.csv').set_index(['background','method','team'])
errors=[]
for (bg,method,team),g in d.groupby(['background','method','team']):
    c=g[g.role=='candidate'];n=g[g.role!='candidate']
    assert len(c)==11 and len(n)==13
    values={'known_nonmember_n':13,'known_nonmember_mutual_nearest':int((n.C_to_B_rank_min==1).sum()),'candidate_mutual_nearest':int((c.C_to_B_rank_min==1).sum()),'known_nonmember_d1_median':n.d1.median(),'candidate_d1_median':c.d1.median(),'known_nonmember_specificity_max':n.specificity.max(),'candidate_specificity_max':c.specificity.max(),'known_nonmember_closer_than_best_candidate':int((n.d1<=c.d1.min()+1e-12).sum()),'known_nonmember_more_specific_than_best_candidate':int((n.specificity>=c.specificity.max()-1e-12).sum())}
    row=table.loc[(bg,method,team)]
    errors.extend(abs(row[k]-v) for k,v in values.items())
summary=read(L/'rare_summary.csv').set_index(['background','method'])
for key,g in table.reset_index().groupby(['background','method']):
    row=summary.loc[key]
    assert row.targets==22 and not row.membership_inferred
    errors.extend(abs(row[c]-g[c].mean()) for c in values)
change=read(L/'rare_background_changes.csv').set_index(['background','method'])
base=d[d.background=='main'].set_index(['team','record_id','method'])
for key,g in d[d.role=='candidate'].groupby(['background','method']):
    ref=base.loc[list(zip(g.team,g.record_id,g.method))]
    row=change.loc[key];assert row.candidate_target_pairs==242
    for dst,col in [('nearest_C_changed','nearest_C_row0'),('candidate_distance_rank_changed','candidate_distance_rank'),('team_rank_for_candidate_changed','team_distance_rank_for_record')]:
        errors.append(abs(row[dst]-(g[col].to_numpy()!=ref[col].to_numpy()).sum()))
    errors.append(abs(row.mean_abs_d1_change-np.mean(abs(g.d1.to_numpy()-ref.d1.to_numpy()))))
assert max(errors)<1e-12
result={'status':'passed','backgrounds':7,'diagnostic_methods':3,'target_aggregate_rows':len(table),'summary_rows':len(summary),'background_change_rows':len(change),'numeric_values_recomputed':len(errors),'max_abs_error':float(max(errors)),'membership_truth_used':False,'source_roles_preserved':'11 unknown rare candidates and13 known nonmembers; no per-record values included in this audit report'}
(R/'inventory/rare_independent_aggregate_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
