from pathlib import Path
import subprocess,sys,json,time,hashlib
import pandas as pd
BASE=Path(__file__).resolve().parent
BGS=['prelim','starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
FAMILIES=['redteam','library','cross','atk','density','virtual','initial_gen','raw_gen','domias','aia2mia','outcome']
assert all((BASE/'results'/bg/f/'completion.json').exists() for bg in BGS for f in FAMILIES),'incomplete primary grid'
for script in ['repair_metadata.py','aggregate_archive.py','audit_historical_archive.py','verify_archive.py','verify_rare_secondary.py']:
 subprocess.run([sys.executable,str(BASE/script)],check=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=json.loads(Path('/workspace/shared/mia_bg7_inputs/manifest.json').read_text());immut=[]
for name,r in inputs['subsets'].items():
 for field,hf in [('path','sha256'),('identity_path','identity_sha256')]:immut.append({'path':r[field],'sha256':sha(r[field]),'expected':r[hf],'match':sha(r[field])==r[hf]})
for r in inputs['files']:
 # Data and original sources are immutable; truth is only hashed after all scores are frozen.
 immut.append({'path':r['path'],'sha256':sha(r['path']),'expected':r['sha256'],'match':sha(r['path'])==r['sha256']})
assert all(r['match'] for r in immut)
(BASE/'input_immutability_qa.json').write_text(json.dumps({'files_checked':len(immut),'all_match':True,'files':immut},ensure_ascii=False,indent=2)+'\n')
qa=json.loads((BASE/'qa.json').read_text());rare=json.loads((BASE/'rare_secondary_qa.json').read_text());old=json.loads((BASE/'baseline_target_validation.json').read_text());c=pd.read_csv(BASE/'coverage.csv');d=pd.read_csv(BASE/'by_target.csv');s=pd.read_csv(BASE/'summary.csv');changes=pd.read_csv(BASE/'background_changes.csv');assert len(c)==707 and int(c.background_dependent.sum())==693 and len(d)==13783
assert qa['status']=='PASS' and rare['status']=='PASS'
assert d.groupby(['method','team']).background.nunique().eq(7).all()
assert c.apply(lambda r:r.n_targets==(3 if 'split_seed1_original' in r.method else 20),axis=1).all()
base_validation=pd.read_csv(BASE/'baseline_method_validation.csv').set_index('method');changes['historical_tpr_exact']=changes.method.map(base_validation.tpr_mismatches.eq(0));changes['historical_tpr_auc_exact']=changes.method.map((base_validation.tpr_mismatches.eq(0)&base_validation.auc_mismatches.eq(0)))
changes.to_csv(BASE/'background_changes.csv',index=False,float_format='%.17g')
non12=changes[(~changes.team.isin([1,2]))&(changes.background!='prelim')].sort_values(['tpr_delta','tpr'],ascending=False)
non12.to_csv(BASE/'non12_same_config_gains.csv',index=False,float_format='%.17g');non12[non12.newly_at_least_010].to_csv(BASE/'non12_new_tpr_at_least_010.csv',index=False,float_format='%.17g')
mx0=d[d.background.eq('prelim')].groupby('team').tpr.max();mx=d.groupby('team').tpr.max();panel=pd.DataFrame({'prelim_panel_max':mx0,'all7_panel_max':mx});panel['delta']=panel.all7_panel_max-panel.prelim_panel_max;panel['newly_at_least_010']=(panel.prelim_panel_max<.1)&(panel.all7_panel_max>=.1);panel.reset_index().to_csv(BASE/'within_panel_team_max.csv',index=False,float_format='%.17g')
means=s[(s.scope=='without1_2')&(s.n_targets==18)].sort_values('tpr',ascending=False);means.to_csv(BASE/'non12_full_roster_method_means.csv',index=False,float_format='%.17g')
manifest={'status':'COMPLETE_AVAILABLE_ARCHIVE_GRID','completed_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'whole_mia_source_recipe_ids':101,'background_dependent_recipes':99,'background_dependent_conditions':693,'background_independent_controls':2,'control_conditions':14,'total_method_background_conditions':707,'completed_family_background_jobs':77,'target_rows':13783,'background_dependent_target_rows':13503,'control_target_rows':280,'full20_recipes':98,'source_native_three_target_recipes':3,'three_target_ids':[24,26,29],'rare_secondary_recipes':26,'rare_secondary_conditions':182,'historical_validation':old,'current_score_qa':qa,'rare_qa':rare,'original_panel_excluded_saved_history_cases':9,'followup_source_recovered_recipes':6,'followup_status_path':'followup/status.json','followup_fixed_input_blocked_composites':0,'remaining_generator_missing_recipes':3,'r2_reconstruction_owned_elsewhere':'fixed22/policy worker','main_target_membership_labels_used':False,'membership_training_labels_used':False,'note':'All selected-BG pools refit original algorithm statistics/hyperparameter rules. No historical-result retuning. MainBG is a reference pool, not main-label evaluation.'}
(BASE/'completion_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
p=BASE/'README.md';text=p.read_text();text=text.replace('実行中。最新の完了件数は `aggregate_status.json`、全体の実行状態は `batch_status.json` を参照。','実行可能な旧アーカイブ99背景依存方式×7条件が完了。背景非依存の対照2方式も7条件で不変性を検証。全707方式条件・13,783対象行を公式再採点した。別途、Team20希少26方式×7条件の副次診断も完了。旧保存履歴9ケースのうち追加6方式の元コードが回収され、別途 followup/ に再実験・不足入力を記録する。生成器が未回収の3ケースは全体台帳に残し、r2は固定22方式パネルで再現する。')
p.write_text(text)
public=[p for p in BASE.iterdir() if p.is_file() and p.suffix in ['.csv','.json','.md','.py'] and p.name not in ['priority_request.json','light_status.json','batch_status.json','delivery_hashes.json']]
files={p.name:sha(p) for p in public};(BASE/'delivery_hashes.json').write_text(json.dumps({'aggregate_only_files':files,'excluded':['results/','rare_secondary/','compat/','locks/','logs/','benchmarks/','followup/results/'],'note':'Frozen candidate arrays and source/input links remain internal.'},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k not in ['historical_validation','current_score_qa','rare_qa']},ensure_ascii=False))
