from pathlib import Path
import pandas as pd,json,hashlib
OUT=Path(__file__).parent
I=Path('/workspace/shared/mia_all_background_inventory/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件')
R=Path('/workspace/shared/mia_all_bg_delivery/20261010T025623Z/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件')
BGS=['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
TEAMS={20:'さいふぉん-3',7:'神神公社',8:'CyberMantra',12:'k等分の花嫁',6:'Trailblazer',1:'Ritz',9:'あかぎ団'}
d=pd.read_csv(I/'team_rankings_0310UTC/primary_all20_all1049_strictly_gt_010.csv');d=d[d.team.isin(TEAMS)];d['rtpr']=d.tpr.round(12);d['rdelta']=d.delta.round(12)
a=d.sort_values(['team','rtpr','rdelta','canonical_id','background'],ascending=[True,False,False,True,True]).drop_duplicates(['team','canonical_id'])
a['recipe_rank']=a.groupby('team').cumcount()+1
a['team_name']=a.team.map(TEAMS)
a.to_csv(OUT/'all_distinct_recipe_peaks_0310.csv',index=False)
a[a.recipe_rank.le(3)].to_csv(OUT/'strict_top3_distinct_parameter_recipes.csv',index=False)
# Candidate pool: top recipes, every family best, and high-value interpretable variants.
q=a[a.recipe_rank.le(3)].copy()
q=pd.concat([q,a.groupby(['team','method_label']).head(1)])
extra={20:['q7_cat4_nnratio_k30__county500__cross0','q7_robust_adaptivek15_h1__county500__cross0','q13_robust_sex1_full_reg0.01_gmm8__raw__cross0'],7:['q13_robust_sex0_diag_reg0.01_gmm4__raw__cross0','q7_whiten_nnratio_k10__raw__cross0'],8:list(a[a.team.eq(8)].method),12:['q7_robust_ulsif_centers300_h1_a0_lambda0.0001__age100__cross0'],6:['q7_robust_lofC_k100__age100__cross0.1'],9:['q7_quantile_sex1_kde_ratio_h0.5__county500__cross0.1','q9_rank_lof_ratio_k30__raw__cross0.1']}
for t,methods in extra.items():q=pd.concat([q,a[a.team.eq(t)&a.method.isin(methods)]])
q=q.drop_duplicates(['team','canonical_id'])
cache={};manifest=[];allrows=[];summ=[]
def read(rel):
 if rel not in cache:
  p=R/rel;v=pd.read_csv(p);cache[rel]=v
  manifest.append({'source_path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(v)})
 return cache[rel]
for row in q.to_dict('records'):
 sp=row['source_path'];rows=[]
 for bg in BGS:
  if '/prelim/'+row['background']+'/' in sp:rel=sp.replace('/prelim/'+row['background']+'/', '/prelim/'+bg+'/')
  elif '/'+row['background']+'/' in sp:rel=sp.replace('/'+row['background']+'/', '/'+bg+'/')
  else:rel=sp
  z=read(rel);z=z[z.team.eq(row['team'])&z.method.eq(row['method'])]
  if 'background' in z:z=z[z.background.eq(bg)]
  if 'population' in z:z=z[z.population.eq('all_fixed')]
  if 'inference' in z:z=z[z.inference.eq('all20')]
  if 'fpr' in z:z=z[z.fpr.eq(.01)]
  if 'condition' in z:z=z[z.condition.eq('full')]
  if 'canonical_config_id' in z and sp.startswith('legacy/'):
   z=z[z.canonical_config_id.eq(row['canonical_id'].removeprefix('oct07::'))]
  assert len(z)==1,(row['team'],row['method'],bg,len(z))
  z=z.iloc[0];tpr=float(z['tpr_0.01'] if 'tpr_0.01' in z else z['mia_all'])
  rows.append({'team':row['team'],'team_name':row['team_name'],'canonical_id':row['canonical_id'],'method':row['method'],'background':bg,'tpr':tpr,'members':int(z['members']),'source_path':str(R/rel)})
 vals=[r['tpr'] for r in rows];assert abs(max(vals)-row['tpr'])<1e-12
 s=dict(row);s.update({'min_tpr':min(vals),'median_tpr':float(pd.Series(vals).median()),'max_tpr':max(vals),'background_count_gt_010':sum(round(x,12)>.10 for x in vals),'all_best_backgrounds':'|'.join(r['background'] for r in rows if abs(r['tpr']-max(vals))<1e-12)})
 summ.append(s);allrows+=rows
pd.DataFrame(summ).to_csv(OUT/'candidate_pool_with_seven_background_summary.csv',index=False)
pd.DataFrame(allrows).to_csv(OUT/'candidate_pool_seven_backgrounds.csv',index=False)
(OUT/'source_manifest.json').write_text(json.dumps({'ranking_source':str(I/'team_rankings_0310UTC/primary_all20_all1049_strictly_gt_010.csv'),'fpr':.01,'inference':'all20','population':'all_fixed','n':1049,'sources':manifest},ensure_ascii=False,indent=2))
print(pd.DataFrame(summ)[['team','method','max_tpr','min_tpr','median_tpr','background_count_gt_010','all_best_backgrounds']].to_string(index=False))
