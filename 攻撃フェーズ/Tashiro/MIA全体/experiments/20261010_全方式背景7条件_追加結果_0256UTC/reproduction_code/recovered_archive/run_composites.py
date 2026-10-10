"""Exact saved-Gen nonrare composition AST, with verified submission8 fixed cells."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import ast,copy,json,time,argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import run_archive as ar
import run_followup as f
SOURCE=ar.ORIG/'予備戦/結果分析/分析.py'
BASELINE=Path('/workspace/shared/mia_recovered_validation/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/private_inputs/prelim_submission8_F_mia.csv')
BASELINE_SHA='cecc02572e677666d9b42283e25a8a656801b804cd83b25afc799adc700fffa3'
METHODS=['legacy_gen7_nonrare_historical_replace','legacy_gen9_nonrare_historical_replace','legacy_gen10_nonrare_historical_replace']
def run(bg):
 dest=f.ROOT/'results'/bg/'composites';rawdir=ar.BASE/'results'/bg/'raw_gen'
 if (dest/'completion.json').exists():return
 assert (rawdir/'completion.json').exists(),'raw score calculation not yet complete'
 dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter();assert ar.sha(BASELINE)==BASELINE_SHA
 rawmanifest=json.loads((rawdir/'prediction_manifest.json').read_text());assert rawmanifest['freeze_sha256']==ar.sha(rawdir/'frozen.npz')
 raw=np.load(rawdir/'frozen.npz');mf8=pd.read_csv(BASELINE,float_precision='round_trip')
 audit=ar.module('_composite_public_inputs',ar.WRAPPERS/'official_audit.py');b,private,rare,normal,dist=audit.load_inputs(ar.setup(bg))
 assert mf8.record_id.tolist()==b.record_id.tolist();assert list(mf8.columns)==['record_id']+[str(k) for k in ar.TEAMS]
 assert int(private.sum())==163 and int((~private&~rare).sum())==875
 p21={'raw_scores':{str(k):{'qi7_gen':raw['legacy_raw_qi7_gen'][:,j],'attrs10_gen':raw['legacy_raw_attrs10_gen'][:,j]} for j,k in enumerate(ar.TEAMS)}}
 p22={'raw_new_scores':{str(k):{'attrs9':raw['legacy_raw_attrs9'][:,j]} for j,k in enumerate(ar.TEAMS)}}
 mats={}
 def capture(attacker,mf,af,name,label):
  dim=name.split('_')[0][3:];mats['legacy_gen'+dim+'_nonrare_historical_replace']=mf[[str(k) for k in ar.TEAMS]].to_numpy().copy();return []
 tree=ast.parse(SOURCE.read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and n.lineno==190);node=copy.deepcopy(node)
 modes=next(n for n in node.body if isinstance(n,ast.For) and n.lineno==191);modes.iter=ast.copy_location(ast.List(elts=[ast.Constant(value='非希少のみ差替')],ctx=ast.Load()),modes.iter);ast.fix_missing_locations(node)
 env=dict(np=np,rankdata=rankdata,mf8=mf8,rare=rare,private=private,p21=p21,p22=p22,teams=ar.TEAMS,variants=[],candidate_docs=[],evaluate=capture,af8=None)
 f.run_ast(SOURCE,[node],env)
 assert set(mats)==set(METHODS)
 for matrix in mats.values():
  assert np.array_equal(matrix[rare],mf8.iloc[:,1:].to_numpy()[rare]);assert np.array_equal(matrix[:,:2],mf8.iloc[:,1:3].to_numpy())
 np.savez_compressed(dest/'frozen.npz',**mats)
 meta=dict(source=str(SOURCE.relative_to(ar.REC)),source_sha256=ar.sha(SOURCE),source_ast_lines=[190,204],source_mode='非希少のみ差替',fixed_baseline_sha256=BASELINE_SHA,raw_freeze_sha256=rawmanifest['freeze_sha256'],preserved_targets=[1,2],preserved_rare=True,private_mask='exact public own-B/QI intersection, equivalent to historical private member-Q keys; no MIA labels used',replacement_mask_rows=875,applicable_target_ids=ar.TEAMS)
 manifest=dict(background=bg,family='composites',methods=list(mats),teams=ar.TEAMS,seconds=time.perf_counter()-start,freeze_sha256=ar.sha(dest/'frozen.npz'),predictions_frozen_before_evaluation_labels=True,membership_training_labels_used=False,historical_fixed_score_component_used=True,metadata={name:meta for name in mats})
 (dest/'prediction_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');f.evaluate(bg,mats,dest)
 (dest/'completion.json').write_text(json.dumps({k:v for k,v in manifest.items() if k!='metadata'},ensure_ascii=False,indent=2)+'\n');print('COMPLETED COMPOSITES',bg,manifest['seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--available',action='store_true');p.add_argument('--wait-primary',action='store_true');a=p.parse_args()
 if a.wait_primary:
  while len(list((ar.BASE/'results').glob('*/*/completion.json')))<77:time.sleep(20)
 for bg in ar.BGS:
  if (ar.BASE/'results'/bg/'raw_gen/completion.json').exists():run(bg)
 f.aggregate()
