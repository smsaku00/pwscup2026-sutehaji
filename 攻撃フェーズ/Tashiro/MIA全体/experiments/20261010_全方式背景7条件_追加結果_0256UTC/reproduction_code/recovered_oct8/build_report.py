"""Audit and package the completed recovered Oct8 primary preliminary grids."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
FAMILIES=['histogram','geometry','alignment','mixture','local']
COUNTS={'histogram':1920,'geometry':666,'alignment':750,'mixture':960,'local':300}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 coverage=[];summaries=[]
 for family in FAMILIES:
  out=HERE/family/'prelim';complete=json.loads((out/'completion.json').read_text());assert complete['complete'] and complete['settings_per_background']==COUNTS[family]
  for script in ('audit_catalog_baseline.py','audit_target_baseline.py','summarize_completed.py'):
   subprocess.run([sys.executable,str(HERE/script),'--family',family],check=True)
  audit=json.loads((out/'historical_catalog_regression.json').read_text());assert not audit['catalog_without_result']
  if audit['failed']:
   assert family=='local' and audit['failed']==1
   verification=json.loads((out/'lof_source_direct_verification.json').read_text());assert verification['source_direct_verification_passed']
   failures=pd.read_csv(out/'historical_catalog_regression.csv');failures=failures[~failures['pass']]
   assert failures.method.tolist()==['q7_robust_lofC_k10__age100__cross0.1']
  for recpath in out.glob('*/prediction_record.json'):
   rec=json.loads(recpath.read_text());assert not rec['truth_read'] and sha(recpath.parent/'scores.npz')==rec['scores_sha256']
  coverage.append({'family':family,'settings_per_bg':COUNTS[family],'backgrounds':7,'panels':7*COUNTS[family],'membership_scored':True,'baseline_catalog_matches':audit['matched_methods'],'baseline_failed':audit['failed'],'source_output_names_not_in_catalog':len(audit['not_in_catalog'])})
  summaries.append(pd.read_csv(out/'within_family_maxima_descriptive.csv',float_precision='round_trip'))
 chow=json.loads((HERE/'chow/prelim/completion.json').read_text());assert chow['complete'] and not chow['membership_scoring']
 coverage.append({'family':'chow','settings_per_bg':3,'backgrounds':7,'panels':21,'membership_scored':False,'baseline_catalog_matches':0,'baseline_failed':0,'source_output_names_not_in_catalog':0})
 pd.DataFrame(coverage).to_csv(HERE/'family_coverage.csv',index=False)
 pd.concat(summaries,ignore_index=True).to_csv(HERE/'within_family_maxima_descriptive.csv',index=False,float_format='%.17g')
 mpath=Path('/workspace/shared/mia_bg7_inputs/manifest.json');m=json.loads(mpath.read_text());assert sha(mpath)=='95820447e6a4cccb8da8a228ae9d21adc24d35f0b1bb03d71c019c52b597ddda'
 for f in m['files']:assert sha(f['path'])==f['sha256']
 for v in m['subsets'].values():assert sha(v['path'])==v['sha256'] and sha(v['identity_path'])==v['identity_sha256']
 dump(HERE/'completion.json',{'complete':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'families':coverage,'source_settings_per_bg':4599,'membership_settings_per_bg':4596,'source_panels':32193,'membership_scored_panels':32172,'label_free_chow_panels':21,'inference_targets':20,'all7_preliminary_complete':True,'main_projection_deferred':True,'original_canonical_files_unchanged':len(m['files']),'input_manifest_sha256':sha(mpath),'within_family_maxima_are_descriptive':True,'no_source_parameter_pruning':True,'historical_catalog_checks':4592,'historical_catalog_passed':4591,'historical_catalog_failed':1,'all_historical_metrics_reproduced':False,'known_validation_exception':'local q7_robust_lofC_k10__age100__cross0.1 differs from historical catalog by0.002105263157894735 mean; all6 direct-source LOF k10 channels match current saved rerun bitwise; historical runtime/cache cause unresolved'})
 dump(HERE/'delivery_manifest.json',{'files':[{'path':str(p.relative_to(HERE)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json' and '__pycache__' not in p.parts],'input_manifest_sha256':sha(mpath)})
 print('COMPLETE: 32172 membership panels +21 original label-free Chow panels',flush=True)
if __name__=='__main__':main()
