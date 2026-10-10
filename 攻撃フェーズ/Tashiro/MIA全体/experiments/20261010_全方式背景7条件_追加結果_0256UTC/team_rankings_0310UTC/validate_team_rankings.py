from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
P=Path(__file__).parent
ROOT=Path('/workspace/shared/mia_all_bg_delivery/20261010T025623Z/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件')
LIVE=Path('/workspace/shared/mia_all_bg_recovered_calibration/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_calibration')
top=pd.read_csv(P/'top10_per_team_primary_all20_all1049.csv');alltop=pd.read_csv(P/'top10_per_team_by_protocol.csv');th=pd.read_csv(P/'all_results_strictly_gt_010.csv')
assert top.groupby('team').size().eq(10).all() and len(top)==200
teams=[1,2,3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
assert sorted(top.team.unique())==teams
assert top.training_category.eq('no_membership_labels').all()
assert top.inference.eq('all20').all() and top.population.eq('all_fixed').all() and top.n.eq(1049).all()
assert top.target_set.eq(','.join(map(str,teams))).all()
assert th.tpr.gt(.1).all() and not np.isclose(th.tpr,.1,rtol=0,atol=1e-12).any()
assert not th.duplicated(['canonical_id','background','team','inference','population']).any()
assert np.nanmax(abs(th.tpr-th.prelim_bg_tpr-th.delta))<1e-12
# Independent index-and-value rejoin to original CSV rows for every primary top10 row,
# every protocol's maximum, and a deterministic sample from the exhaustive threshold table.
sample=pd.concat([top,alltop[alltop['rank'].eq(1)],th.sample(min(250,len(th)),random_state=20261010)]).drop_duplicates(['canonical_id','background','team','inference','population'])
cache={};checked=0; baseline_checked=0

def select(row,bg):
 rel=row.source_path
 path=(LIVE/rel.removeprefix('learned_completed/')) if rel.startswith('learned_completed/') else (ROOT/rel)
 parts=list(path.parts)
 # Individual BG tables have BG as a directory. Multi-BG aggregate tables carry a column instead.
 if row.background in parts:parts[parts.index(row.background)] = bg;path=Path(*parts)
 if path not in cache:cache[path]=pd.read_csv(path)
 d=cache[path]
 m=d.team.eq(row.team)
 if 'fpr' in d:m &= d.fpr.eq(.01)
 if 'config_id' in d and str(row.source_id).startswith('BG7-'):m &= d.config_id.eq(row.source_id)
 else:
  col='method' if 'method' in d else 'policy'
  m &= d[col].eq(row.method)
 if 'condition' in d:m &= d.condition.eq('full')
 if 'background' in d:m &= d.background.eq(bg)
 if 'background_source' in d:m &= d.background_source.eq(bg)
 if 'population' in d:m &= d.population.eq(row.population)
 if 'inference' in d:m &= d.inference.replace({'legacy18':'without1_2'}).eq(row.inference)
 s=d[m]
 score='tpr_0.01' if 'tpr_0.01' in d else ('mia_all' if 'mia_all' in d else 'tpr')
 assert len(s)>0,('no original row',row.team,row.method,bg,path)
 values=s[score].unique();assert max(values)-min(values)<1e-12,('ambiguous score',path,row.method)
 return float(values[0])
for row in sample.itertuples():
 assert abs(select(row,row.background)-row.tpr)<1e-12,('score mismatch',row)
 checked+=1
 if not pd.isna(row.prelim_bg_tpr):
  assert abs(select(row,'prelim')-row.prelim_bg_tpr)<1e-12,('baseline mismatch',row)
  baseline_checked+=1
# Existing independent official-equivalent audit lineage; inspect the actual audit values.
geo=json.loads((ROOT/'independent_audits/recovered_geometry_independent_audit.json').read_text())
assert geo['status']=='passed' and geo['max_absolute_metric_error']==0
for bg,proof in geo['backgrounds'].items():
 live=Path('/workspace/shared/mia_all_bg_fusion/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_oct8/geometry/prelim')/bg
 f=ROOT/f'recovered_oct8/geometry/prelim/{bg}/by_target.csv'
 assert hashlib.sha256((live/'scores.npz').read_bytes()).hexdigest()==proof['score_file_sha256']
 assert f.read_bytes()==(live/'by_target.csv').read_bytes()
learn_path=ROOT.parents[5] if False else Path('/workspace/shared/mia_all_background_inventory/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/inventory/recovered_learned_independent_audit.json')
learn=json.loads(learn_path.read_text());learned_summary={}
for fam in ['learned_yesterday','learned_bg_transfer','learned_bg_ablation','learned_ablation']:
 a=learn[fam];assert a['all7_complete'] and a['same_config_set']
 assert all(x['maximum_TPR_error']==0 for x in a['jobs'])
 learned_summary[fam]={'backgrounds':len(a['jobs']),'target_FPR_values_checked':sum(x['target_FPR_values_checked'] for x in a['jobs']),'max_TPR_error':0}
result={'status':'passed','primary_rows_checked':200,'total_original_score_rejoins':checked,'regenerated_prelim_baseline_rejoins':baseline_checked,'strict_threshold_verified':len(th),'all20_exact_target_set':teams,'primary_training_class':'no_membership_labels','all20_all1049_all_teams_have_a_value_strictly_above_010':bool(top.groupby('team').tpr.max().gt(.1).all()),'official_equivalent_geometry_audit':{'source':'independent_audits/recovered_geometry_independent_audit.json','matrix_hashes':geo['prediction_matrix_hashes_checked'],'independent_metric_values':geo['metric_values_checked'],'max_absolute_error':geo['max_absolute_metric_error']},'learned_complete_audits':learned_summary,'method_note':'Ranking rejoin is independent of builder; source audits independently recomputed official-equivalent ROC from sealed score arrays.'}
(P/'independent_ranking_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
