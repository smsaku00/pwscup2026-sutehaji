"""Historical submission7 source composition with exact fixed query history and caps."""
from pathlib import Path
import ast,copy,json,time,argparse
import numpy as np
import pandas as pd
import run_archive as ar
import run_followup as f
SOURCE=ar.ORIG/'予備戦/攻撃フェーズ/提出記録/7_20260907_GenLRA_Seo/作成時記録/MIA/作成.py'
PRIVATE=Path('/workspace/shared/mia_recovered_validation/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/private_inputs')
BASELINE=PRIVATE/'prelim_submission6_F_mia.csv';SETTINGS=PRIVATE/'prelim_submission7_fixed_settings.json'
BASELINE_SHA='bb71413e27ff0b005c21dfcd52a24c5a30952b1eb552ae5cdeb585cbd66644e8';SETTINGS_SHA='1d7f4d369fecf17f2a059bcbe378ea6b061e9fb29cad250e4a0d94c45393be9b'
METHOD='legacy_submission_7'
def run(bg):
 dest=f.ROOT/'results'/bg/'submission7';rawdir=ar.BASE/'results'/bg/'raw_gen'
 if (dest/'completion.json').exists():return
 assert (rawdir/'completion.json').exists();dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
 assert ar.sha(BASELINE)==BASELINE_SHA and ar.sha(SETTINGS)==SETTINGS_SHA
 setting=json.loads(SETTINGS.read_text());baseline=pd.read_csv(BASELINE,float_precision='round_trip');original=baseline.iloc[:,1:].to_numpy();ids=baseline.record_id.to_numpy()
 audit=ar.module('_s7_public_inputs',ar.WRAPPERS/'official_audit.py');b,private,rare,normal,dist=audit.load_inputs(ar.setup(bg))
 assert b.record_id.tolist()==ids.tolist() and list(baseline.columns)==['record_id']+[str(k) for k in ar.TEAMS]
 assert np.array_equal(private,np.isin(ids,setting['private_ids']));eligible=setting['eligible_ids'];assert set(eligible)==set(ids[rare&~private])
 known={(int(r['team']),int(r['record_id'])) for r in setting['known_negative_pairs']};assert len(known)==48
 stored={int(r['team']):r for r in setting['teams']};pos={int(rid):i for i,rid in enumerate(ids)};answer=original.copy()
 rawmanifest=json.loads((rawdir/'prediction_manifest.json').read_text());assert ar.sha(rawdir/'frozen.npz')==rawmanifest['freeze_sha256'];raws=np.load(rawdir/'frozen.npz')
 tree=ast.parse(SOURCE.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute');loop=next(n for n in main.body if isinstance(n,ast.For) and n.lineno==107)
 body=[copy.deepcopy(n) for n in loop.body if 122<=n.lineno<=143 and n.lineno!=133]
 diagnostics=[]
 for j,k in enumerate(ar.TEAMS):
  env=dict(np=np,pd=pd,original=original,answer=answer,rare=rare,private=private,nonrare_unknown=normal,eligible=eligible,known=known,pos=pos,raw=raws['legacy_raw_qi7_gen'][:,j],raw9=raws['legacy_raw_attrs9'][:,j],j=j,k=k)
  f.run_ast(SOURCE,body,env)
  assert env['cap']==stored[k]['cap'];match=env['pair']==stored[k]['rare_top2'];unique=len(set(env['raw9'][[pos[rid] for rid in env['order']]]))==len(env['order'])
  if bg=='prelim':assert match and unique and env['order']==stored[k]['rare_order']
  diagnostics.append(dict(team=k,rare_top2_matches_historical=match,raw9_remaining_scores_unique=unique,rare_remaining_count=len(env['order']),protected_nonrare_prefix=env['protected_prefix']))
 assert np.isfinite(answer).all() and (answer>=0).all() and (answer<=1).all()
 assert int((answer[rare]==0).sum())==308 and int((answer[rare]>0).sum())==172
 mats={METHOD:answer};np.savez_compressed(dest/'frozen.npz',**mats)
 meta=dict(source=str(SOURCE.relative_to(ar.REC)),source_sha256=ar.sha(SOURCE),source_ast_lines=[122,143],historical_assertion133='recorded as diagnostic under new BG; enforced for prelim baseline',fixed_baseline_sha256=BASELINE_SHA,fixed_settings_sha256=SETTINGS_SHA,raw_freeze_sha256=rawmanifest['freeze_sha256'],historical_negative_pairs_count=48,historical_query_outcomes_used=True,cap_rule='fixed max of rare scores in original submission6 for each target',preserved_nonrare_targets=[1,2],normal_rows=875,applicable_target_ids=ar.TEAMS,diagnostics=diagnostics)
 manifest=dict(background=bg,family='submission7',methods=list(mats),teams=ar.TEAMS,seconds=time.perf_counter()-start,freeze_sha256=ar.sha(dest/'frozen.npz'),predictions_frozen_before_evaluation_labels=True,membership_training_labels_used=False,historical_query_outcomes_used=True,historical_fixed_score_component_used=True,metadata={METHOD:meta})
 (dest/'prediction_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');f.evaluate(bg,mats,dest);(dest/'completion.json').write_text(json.dumps({k:v for k,v in manifest.items() if k!='metadata'},ensure_ascii=False,indent=2)+'\n');print('COMPLETED SUBMISSION7',bg,manifest['seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--wait-primary',action='store_true');args=p.parse_args()
 if args.wait_primary:
  while len(list((ar.BASE/'results').glob('*/*/completion.json')))<77:time.sleep(15)
 for bg in ar.BGS:run(bg)
 f.aggregate()
