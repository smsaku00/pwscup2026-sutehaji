from pathlib import Path
import pandas as pd, numpy as np, json, hashlib, collections, datetime
ROOT=Path('/workspace/shared/mia_all_bg_delivery/20261010T025623Z/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件')
LIVE=Path('/workspace/shared/mia_all_bg_recovered_calibration/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_calibration')
OUT=Path(__file__).parent
CUTOFF='2026-10-10T03:10:00+00:00'
BGS=['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
TEAMS=[1,2,3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
REG={r['registration_id']:r for r in map(json.loads,(ROOT/'inventory/recovered_recipe_catalog.jsonl').read_text().splitlines())}
ALIASES={r['original_config_id']:r['canonical_config_id'] for r in pd.read_csv(ROOT/'inventory/cross_panel_semantic_aliases.csv').to_dict('records')}
LABELS={'hierarchical_histogram':'階層ヒストグラム','geometry_distance':'距離・分布','aligned_distance':'分布整列距離','mixture':'混合分布','local_density':'局所密度','source_direct_ratio':'直接密度比','source_ratio_attribute_seed':'属性・seed密度比','source_ratio_stable_centers':'安定中心密度比','broad_calibration':'広範較正','broad_calibration_mix':'広範較正混合','learned_feature_ablation':'学習特徴除外','learned_legacy_18_check':'旧学習・学習対象感度','learned_background_transfer':'学習背景転移','learned_background_ablation':'学習背景除外','oct07':'10月7日方式','archive':'旧アーカイブ','meta_public_null_strength':'公開非会員較正選択','meta_public_null_nested':'公開非会員入れ子較正','meta_split_A_fixed':'背景分割較正','meta_split_A_gaussian_seed':'Gaussian分割seed','meta_public_Q_owner_removal':'公開Q所有者除外診断','meta_public_C_row_removal':'公開C行除外診断','available':'Gen・TAPAS・公開較正','joint':'共同・排他補正','fusion':'KDE融合'}
CAT={'fixed_formula_public_input_fitted':'no_membership_labels','fixed_formula_input_fitted':'no_membership_labels','source_label_learned_not_MIA_label_learned':'no_membership_labels','source_label_learned':'no_membership_labels','MIA_membership_label_learned':'membership_label_OOF','fixed_formula_with_preserved_historical_state':'historical_state'}
manifest=[]; audit=[]; frames=[]; exceptions=[]; alias_records=[]
seal_hash={r['path']:r['sha256'] for r in json.loads((ROOT/'delivery_sha256.json').read_text())}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p, live=False):
 h=sha(p);rel=('learned_completed/'+str(p.relative_to(LIVE))) if live else str(p.relative_to(ROOT))
 if not live:assert seal_hash[rel]==h,('hash mismatch',rel)
 else:
  comp=json.loads((p.parent/'completion.json').read_text());rec=json.loads((p.parent/'実行記録.json').read_text())
  assert comp['source_main_returned'] and rec['complete']
  assert rec['forest_prediction_execution']=='fixed_tree_order_serial_predict_v1'
  assert p.stat().st_mtime <= datetime.datetime.fromisoformat(CUTOFF).timestamp()
  assert (p.parent/'prediction_hashes.json').exists()
  if 'output_sha256' in rec: assert rec['output_sha256'][p.name]==h
 manifest.append({'source_path':rel,'sha256':h,'bytes':p.stat().st_size,'seal':'completed_before_0310UTC' if live else '025623UTC_snapshot'})
 d=pd.read_csv(p);d['source_path']=rel;return d

def apply_registry(d, ids, family):
 d=d.copy();d['recipe_id']=ids;d['family']=family if isinstance(family,str) else family
 missing=sorted(set(d.recipe_id)-set(REG));assert not missing,('missing registry',missing[:5])
 d['canonical_id']=d.recipe_id.map(lambda x:REG[x]['canonical_config_id'])
 d['training_category']=d.recipe_id.map(lambda x:CAT.get(REG[x]['learning_class'],REG[x]['learning_class']))
 d['source_id']=d.recipe_id.map(lambda x:';'.join(REG[x].get('source_result_ids',[])) or REG[x].get('catalog_condition_id','') or x)
 d['method_label']=d['family'].map(lambda x:LABELS.get(x,x))
 d['caveats']='予備戦での事後比較・独立検証ではない'
 d.loc[d.training_category.eq('membership_label_OOF'),'caveats']='所属ラベルOOF学習;履歴適応選択;過去値ドリフトあり;現環境再生成baseline'
 d.loc[d.training_category.eq('historical_state'),'caveats']='旧提出・問い合わせ状態を保存した履歴依存方式'
 return d

def add(d,panel,default_inference='all20',default_population='all_fixed',extra_caveat=''):
 d=d.copy();d['panel']=panel
 for c,v in [('inference',default_inference),('population',default_population),('n',1049),('training_category','no_membership_labels'),('method_label',LABELS.get(panel,panel)),('source_id',''),('caveats','予備戦での事後比較・独立検証ではない')]:
  if c not in d:d[c]=v
 if 'canonical_id' not in d:d['canonical_id']=d.recipe_id.map(lambda x:ALIASES.get(x,x))
 if 'tpr' not in d:d['tpr']=d['tpr_0.01'] if 'tpr_0.01' in d else d.mia_all
 d['source_id']=d.source_id.where(d.source_id.ne(''),d.recipe_id)
 d['inference']=d.inference.replace({'legacy18':'without1_2'})
 d['n']=d.population.map({'all_fixed':1049,'normal':875,'nonrare':1025,'historical_native':1049}).fillna(d.n).astype(int)
 d['caveats']=d.caveats+('；'+extra_caveat if extra_caveat else '')
 d['team']=d.team.astype(int)
 assert d.tpr.notna().all() and d.tpr.between(0,1).all()
 # Store the exact native target set per recipe AND BG, including varying policy split views.
 gs=['recipe_id','background','population','inference']
 scope=d.groupby(gs).team.agg(lambda x:','.join(map(str,sorted(set(x)))))
 d=d.join(scope.rename('target_set'),on=gs)
 # Baseline joins retain source recipe, inference and population; no historical number is substituted.
 key=['recipe_id','team','population','inference']
 base=d[d.background.eq('prelim')][key+['tpr']].rename(columns={'tpr':'prelim_bg_tpr'})
 assert not base.duplicated(key).any(),('duplicate baseline',panel)
 d=d.merge(base,on=key,how='left',validate='many_to_one')
 missing=int(d.prelim_bg_tpr.isna().sum())
 if missing:d.loc[d.prelim_bg_tpr.isna(),'caveats']+='；prelim同一ビューで対象外のためbaselineなし'
 d['delta']=d.tpr-d.prelim_bg_tpr
 d['tpr_sort']=d.tpr.round(12)
 d['strict_gt_010']=d.tpr_sort.gt(.10)
 d['easy_team']=d.team.isin([1,2])
 d['family_complete_all7']=True
 # Source aliases are proof-based. Equality is required, never used to infer equivalence.
 k=['canonical_id','team','population','inference','background']
 dup=d[d.duplicated(k,False)]
 if len(dup):
  diff=dup.groupby(k).tpr.agg(['min','max']);bad=diff[diff['max']-diff['min']>1e-12]
  if len(bad):raise AssertionError(('alias metrics mismatch',panel,bad.head().to_dict()))
  alias_records.extend(dup[['recipe_id','canonical_id','panel']].drop_duplicates().to_dict('records'))
 d=d.sort_values(['recipe_id']).drop_duplicates(k)
 audit.append({'panel':panel,'rows':len(d),'all7_configurations':d[['canonical_id','background','inference']].drop_duplicates().shape[0], 'strict_gt_010':int(d.strict_gt_010.sum()),'exact_010':int(d.tpr_sort.eq(.1).sum()),'baseline_missing':missing,'teams':sorted(d.team.unique().tolist())})
 # Keep all requested threshold rows, plus at least ten score levels per scope/team for top10.
 group=['training_category','inference','population','team']
 ranks=d.groupby(group).tpr_sort.rank(method='min',ascending=False)
 d=d[d.strict_gt_010 | ranks.le(10)]
 cols=['team','family','method','method_label','source_id','recipe_id','canonical_id','background','tpr','prelim_bg_tpr','delta','training_category','inference','population','n','target_set','easy_team','caveats','source_path','panel','strict_gt_010','tpr_sort']
 frames.append(d[cols]);print(panel,audit[-1],flush=True)

# Primary available table refits TAPAS bins on each BG. Older fixed-bin panel is a separate control.
for variant,rel in [('primary','available/results_prelim_primary/by_target.csv'),('fixed_bins_control','available/results_prelim/by_target.csv')]:
 d=read(ROOT/rel);d['family']='available';d['recipe_id']=['available|prelim|'+variant+'|available|'+i+'|'+m for i,m in zip(d.inference,d.method)]
 d['canonical_id']=d.recipe_id.map(lambda x:ALIASES.get(x,x))
 if variant!='primary':d['training_category']='fixed_bin_control'
 add(d,'available_'+variant)
for panel,rel in [('joint','joint/prelim/by_target.csv'),('fusion','fusion/prelim_by_target.csv')]:
 d=read(ROOT/rel);d['recipe_id']=[f'{panel}|prelim|primary|{f}|{i}|{m}' for f,i,m in zip(d.family,d.inference,d.method)]
 d['canonical_id']=d.recipe_id.map(lambda x:ALIASES.get(x,x));d['training_category']='no_membership_labels'
 d.loc[d.family.eq('posthoc_gates_diagnostic'),'training_category']='posthoc_diagnostic'
 # Fusion threshold-gate family is historically outcome-exposed; retain separately.
 d.loc[d.family.eq('p03_threshold010'),'training_category']='posthoc_diagnostic'
 add(d,panel)
for folder,fam in {'histogram':'hierarchical_histogram','geometry':'geometry_distance','alignment':'aligned_distance','mixture':'mixture','local':'local_density'}.items():
 d=pd.concat([read(ROOT/f'recovered_oct8/{folder}/prelim/{bg}/by_target.csv') for bg in BGS],ignore_index=True)
 d=apply_registry(d,fam+'::'+d.method,fam);add(d,'recovered_'+folder)
for folder,fam in {'direct':'source_direct_ratio','attribute_seeds':'source_ratio_attribute_seed','stable_centers':'source_ratio_stable_centers','diagnostics/q_owner_removal':'meta_public_Q_owner_removal','diagnostics/public_null':'meta_public_null_strength','diagnostics/nested_public_null':'meta_public_null_nested','diagnostics/split_A':'meta_split_A_fixed','diagnostics/gaussian_seeds':'meta_split_A_gaussian_seed','diagnostics/C_row_removal':'meta_public_C_row_removal'}.items():
 parts=[]
 for bg in BGS:
  d=read(ROOT/f'recovered_calibration/{folder}/{bg}/対象別TPR.csv');d=d[d.fpr.eq(.01)].copy();d['background']=bg;parts.append(d)
 d=pd.concat(parts,ignore_index=True);d=apply_registry(d,fam+'::'+d.method,fam)
 if folder.startswith('diagnostics/'):d['training_category']='public_only_diagnostic'
 add(d,folder.replace('/','_'))
d=read(ROOT/'recovered_calibration/broad/prelim/by_target.csv');fam=d.family.map({'broad65':'broad_calibration','broadmix31':'broad_calibration_mix','mix31':'broad_calibration_mix','mixture31':'broad_calibration_mix'})
assert fam.notna().all(),d.family.unique()
d=apply_registry(d,fam+'::'+d.method,fam);add(d,'broad')
d=read(ROOT/'legacy/aggregate/by_target_primary_canonical.csv');d=apply_registry(d,'oct07::'+d.canonical_config_id,'oct07');add(d,'legacy',extra_caveat='HGB3設定は過去値差あり')
d=read(ROOT/'recovered_archive/all_available_by_target.csv');d=apply_registry(d,'archive::'+d.method,'archive');d['inference']='all20';d.loc[d.method.str.startswith('legacy_atk_split_seed1_original_'),'inference']='native_24_26_29'
add(d,'archive',default_population='historical_native',extra_caveat='全1049元確信度変換；all_fixed共通規約とは別表')
# Policy targets and transformed population stay independent of main all20 ranking.
d=read(ROOT/'policy/report/primary_by_target.csv').rename(columns={'background_source':'background','policy':'method'});pr={x['config_id']:x for x in map(json.loads,(ROOT/'policy/registry/outputs.jsonl').read_text().splitlines())}
d['recipe_id']=d.config_id;d['canonical_id']=d.config_id.map(lambda x:pr[x]['semantic_alias_of'] or x);d['source_id']=d.config_id;d['method_label']='公開ゲート／事後方策';d['inference']='policy_native_'+d.native_view;d['training_category']='historically_exposed_public_policy';d.loc[d.family.isin(['transfer','nested','nested_safety','p08d_looto']),'training_category']='supervised_selector';d.loc[d.family.eq('nested_fixed72'),'training_category']='posthoc_policy_screening';d['caveats']='元対象範囲を保持；履歴で予備戦結果を参照；独立検証ではない'
add(d,'policy')
# r2 is the one missing standalone public fixed-formula recipe in the fixed22 support panel.
fixed22=pd.concat([read(ROOT/f'policy/cache/fixed22/{bg}/by_target.csv') for bg in BGS],ignore_index=True)
full=fixed22[fixed22.condition.eq('full')].copy()
full[full['tpr_0.01'].round(12).gt(.1)].to_csv(OUT/'supporting_fixed22_full_gt010_includes_aliases.csv',index=False)
r2=full[full.method.eq('legacy_saved_work_20260901以前__prelim_attack__r2__F_mia')].copy()
assert len(r2)==7*20*3
r2['recipe_id']='policy_fixed22::full::r2_unified_BG';r2['canonical_id']=r2.recipe_id;r2['family']='fixed22_r2';r2['source_id']='r2_unified_BG_fixed22';r2['method_label']='旧r2公開入力固定式';r2['training_category']='no_membership_labels';r2['caveats']='元B_practice2役割を保持；選択BGを全背景役割へ一度ずつ適用；旧二資料役割対照とは別'
add(r2,'fixed22_r2')

# Complete all7 learned families as of 03:10 UTC; no partial family enters the ranking.
for folder,fam in {'learned_yesterday':'learned_legacy_18_check','learned_bg_transfer':'learned_background_transfer','learned_bg_ablation':'learned_background_ablation','learned_ablation':'learned_feature_ablation'}.items():
 parts=[]
 for bg in BGS:
  d=read(LIVE/f'{folder}/{bg}/対象別TPR.csv',True);d=d[d.fpr.eq(.01)].copy();d['background']=bg;parts.append(d)
 d=pd.concat(parts,ignore_index=True);d=apply_registry(d,fam+'::'+d.method,fam);add(d,folder)

D=pd.concat(frames,ignore_index=True)
k=['canonical_id','team','population','inference','background']
dup=D[D.duplicated(k,False)]
if len(dup):
 diff=dup.groupby(k).tpr.agg(['min','max']);bad=diff[diff['max']-diff['min']>1e-12]
 assert not len(bad),('cross-panel alias mismatch',bad.head().to_dict())
 alias_records.extend(dup[['recipe_id','canonical_id','panel']].drop_duplicates().to_dict('records'))
# Prefer canonical source and primary class; aliases do not multiply ranking slots.
D['alias_priority']=D.recipe_id.ne(D.canonical_id).astype(int)
D=D.sort_values(['alias_priority','panel','recipe_id']).drop_duplicates(k)
D['comparison_group']=D.training_category+'|'+D.inference+'|'+D.population
D.loc[D.panel.eq('policy'),'comparison_group']+='|targets='+D.loc[D.panel.eq('policy'),'target_set']
sort=['comparison_group','team','tpr_sort','delta','canonical_id','background']
D=D.sort_values(sort,ascending=[True,True,False,False,True,True],na_position='last')
D['rank']=D.groupby(['comparison_group','team']).cumcount()+1
D['score_rank']=D.groupby(['comparison_group','team']).tpr_sort.rank(method='min',ascending=False).astype(int)
D['above_threshold']=np.where(D.strict_gt_010,'>0.10','0.10以下')
COLS=['team','rank','score_rank','method_label','method','source_id','background','tpr','prelim_bg_tpr','delta','training_category','inference','population','n','target_set','easy_team','above_threshold','caveats','source_path','canonical_id','comparison_group']
primary=D[(D.training_category=='no_membership_labels') & D.inference.eq('all20') & D.population.eq('all_fixed')].copy()
assert sorted(primary.team.unique())==TEAMS
primary10=primary[primary['rank']<=10]
assert len(primary10)==200
D[D['rank']<=10][COLS].to_csv(OUT/'top10_per_team_by_protocol.csv',index=False)
primary10[COLS].to_csv(OUT/'top10_per_team_primary_all20_all1049.csv',index=False)
threshold=D[D.strict_gt_010]
threshold[COLS].to_csv(OUT/'all_results_strictly_gt_010.csv',index=False)
primary[primary.strict_gt_010][COLS].to_csv(OUT/'primary_all20_all1049_strictly_gt_010.csv',index=False)
counts=D.groupby(['comparison_group','team']).agg(threshold_rows=('strict_gt_010','sum'),max_tpr=('tpr','max')).reset_index();counts.to_csv(OUT/'counts_by_team_and_protocol.csv',index=False)
# Split the exhaustive CSV by protocol/training category for manageable downloads.
parts_dir=OUT/'threshold_by_protocol';parts_dir.mkdir(exist_ok=True)
parts=[]
for i,(g,d) in enumerate(threshold.groupby('comparison_group'),1):
 fname=f'{i:02d}_{g.replace("|","_").replace("/","_")}.csv';d[COLS].to_csv(parts_dir/fname,index=False);parts.append({'comparison_group':g,'file':'threshold_by_protocol/'+fname,'rows':len(d)})
(OUT/'threshold_file_index.json').write_text(json.dumps(parts,ensure_ascii=False,indent=2)+'\n')
(OUT/'input_manifest.json').write_text(json.dumps({'cutoff_utc':CUTOFF,'sealed_snapshot_utc':'2026-10-10T02:56:23Z','sources':manifest},ensure_ascii=False,indent=2)+'\n')
pd.DataFrame(alias_records).drop_duplicates().to_csv(OUT/'applied_source_aliases.csv',index=False)
(OUT/'validation.json').write_text(json.dumps({'cutoff_utc':CUTOFF,'sources_sha256_verified':len(manifest),'panels':audit,'retained_rows_after_proof_alias_dedup':len(D),'threshold_rows':len(threshold),'primary_threshold_rows':int(primary.strict_gt_010.sum()),'primary_top10_rows':len(primary10),'baseline_missing_threshold':int(threshold.prelim_bg_tpr.isna().sum()),'threshold_rule':'TPR rounded to 12 decimals > 0.10; numerical representations of exact 0.10 excluded','aliases_metric_tolerance':1e-12,'top10_sort':'TPR desc (12-decimal numerical equality), delta desc, canonical_id, background; score_rank marks equal TPR','excluded_partial_learned_families':['learned_base','learned_split','learned_lowfpr','learned_multiview','learned_multiview_loo','learned_center_seed'],'raw_candidate_data_included':False},ensure_ascii=False,indent=2)+'\n')

lines=['# チーム別 MIA 上位10と TPR > 0.10 全結果','',f'凍結: 2026-10-10 03:10 UTC。02:56:23 UTC封印版と、03:10までに背景7条件すべて完了した学習4パネルが対象。','', '## 読み方','- 主表は所属ラベル未使用、20対象推論、all_fixed 1,049人、公式 TPR@FPR 1%。各チーム上位10件を掲載。','- 同じ方式でもBGが違えば別の結果。ソース同一性が証明済みの別名だけ統合し、数値一致だけでは統合しない。','- prelim欄は今回の同一方式・同一推論・同一母集団で再生成したprelim BG。差分はTPR−prelim。過去保存値との差ではない。','- 同率TPRは差分、方式ID、BG名の順で表示。差分による順番は探索上の表示順で、性能の有意差を意味しない。','- チーム1・2も含めるが、容易な対象として明記。18対象推論・875人・1,025人・旧1049確信度・所属ラベルOOF学習・教師付き選択・公開診断は別表。','- mainは背景資料名。すべて予備戦の評価であり、本戦TPRではない。','- 多数の設定から結果を見て選んだ記述。独立検証済みの採用方式や、全履歴の最高記録とは主張しない。','',f'主表の厳密0.10超え: {int(primary.strict_gt_010.sum()):,}行。区別した全プロトコル合計: {len(threshold):,}行。機械丸めによる0.10±1e-12未満の表現差は12桁へ揃え、0.10ちょうどを除外。','', '## ファイル','- top10_per_team_primary_all20_all1049.csv: 主表200行、全フィールド','- top10_per_team_by_protocol.csv: 別プロトコル・学習区分ごとの上位10（0.10以下も保持）','- primary_all20_all1049_strictly_gt_010.csv: 主表に対応する0.10超え全件','- all_results_strictly_gt_010.csv: 0.10超え全件。母集団・推論・学習区分で必ず絞って比較','- threshold_by_protocol/: 小さく分割した同じ全件。threshold_file_index.jsonに一覧','- input_manifest.json / validation.json: 元表ハッシュ・対象範囲・照合結果','', '## 主表: チーム別上位10']
for team,d in primary10.groupby('team',sort=True):
 lines += ['',f'### チーム {team}'+('（容易な対象）' if team in [1,2] else ''),'','|順位|方式|BG|TPR|prelim|差分|','|---:|---|---|---:|---:|---:|']
 for r in d.itertuples():lines.append(f'|{r.rank}|{r.method_label}: {r.method}|{r.background}|{r.tpr:.6f}|{r.prelim_bg_tpr:.6f}|{r.delta:+.6f}|')
lines += ['', '## 分離した評価の最大値','', '|学習・推論・母集団|0.10超え件数|最大TPR|','|---|---:|---:|']
for g,d in threshold.groupby('comparison_group'):lines.append(f'|{g}|{len(d):,}|{d.tpr.max():.6f}|')
lines += ['', '## 未完了・除外','- 学習base、split、lowfpr、multiviewは背景の一部が完了していても、03:10時点で7背景が揃わないため本順位表から除外。multiview-LOO、center-seedも除外。','- 元生成コードが未回収の提出1・提出2・Zhou KDEは未再現。','- 希少26方式、Chow–Liuは公開診断であり、今回求める全体TPRではないので除外。','- fixed22支持パネルはfullビューのr2だけを主表へ追加。他21は既存方式との横断別名を含むため、supporting_fixed22_full_gt010_includes_aliases.csvを参照（主件数に重複加算しない）。内部7摂動は独立の新BG方式に数えない。','- 元入力・候補ごとの予測配列・個人IDは含めない。','- 過去値ドリフトが残るHGB・LOF・一部密度比中心・学習系は、今回再生成したprelim BGとの比較。評価値の再採点の正しさと過去再現性は区別。','']
(OUT/'チーム別上位10と010超え全結果.md').write_text('\n'.join(lines))
print('FINAL',json.dumps({'threshold':len(threshold),'primary_threshold':int(primary.strict_gt_010.sum()),'top10':len(primary10),'groups':len(parts),'primary_best':primary10.groupby('team').head(1)[['team','method','background','tpr','prelim_bg_tpr','delta']].to_dict('records')},ensure_ascii=False),flush=True)
