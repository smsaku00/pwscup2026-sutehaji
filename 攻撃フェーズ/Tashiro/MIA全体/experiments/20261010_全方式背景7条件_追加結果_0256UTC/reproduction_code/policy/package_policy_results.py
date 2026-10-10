"""Copy executable dependencies and package aggregate-only results."""
import sys,json,platform,zipfile,hashlib
from pathlib import Path
import numpy,pandas,scipy,sklearn,openpyxl
import runtime as r
import run_fixed22,run_p08d_looto,build_policy_features,evaluate_transfer

def run():
 report=r.HERE/'report';assert (report/'completion.json').exists()
 deps={};external=r.HERE/'code_dependencies';external.mkdir(exist_ok=True)
 candidates=set()
 for m in list(sys.modules.values()):
  p=getattr(m,'__file__',None)
  if p and p.endswith('.py') and p.startswith('/workspace/shared/') and not Path(p).is_relative_to(r.HERE):candidates.add(Path(p))
 candidates.add(run_fixed22.OUTCOME)
 for p in sorted(candidates):
  sha=r.sha(p);dest=external/sha[:16]/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes());deps[str(p)]={'sha256':sha,'bundled_copy':str(dest.relative_to(r.HERE))}
 for name in ('optimization_validation.json','support_code_manifest.json'):
  p=r.GRID/name
  if p.exists():(report/('shared_'+name)).write_bytes(p.read_bytes())
 src=json.loads((r.HERE/'source_manifest.json').read_text())
 for name,x in src.items():assert r.sha(r.HERE/'original'/name)==x['sha256']
 r.dump(report/'dependency_manifest.json',{'python':sys.version,'platform':platform.platform(),'versions':{k:m.__version__ for k,m in [('numpy',numpy),('pandas',pandas),('scipy',scipy),('scikit_learn',sklearn),('openpyxl',openpyxl)]},'external_source_copies':deps,'original_local_source_files':len(src),'canonical_input_manifest':{'path':'/workspace/shared/mia_bg7_inputs/manifest.json','sha256':r.sha('/workspace/shared/mia_bg7_inputs/manifest.json')},'canonical_input_commit':'5b73915671df48a7fe53027977aa1ad4a09e72b2','recovered_original_commit':'7bed499c3a95a939ee4bc73e177ae38bd747fcdd','exact_extra_input_provenance':str(run_fixed22.PRIVATE.parent/'validation_input_provenance.json'),'exact_extra_input_provenance_sha256':r.sha(run_fixed22.PRIVATE.parent/'validation_input_provenance.json'),'source_resolution':'runtime.py uses explicit source locations; bundled copies and SHA manifest can restore those inputs without a source search; raw official inputs remain separately required'})
 # Allowed aggregate schemas only. Model arrays, workbooks, raw records and logs are excluded.
 files=[]
 for p in r.HERE.rglob('*'):
  if not p.is_file() or p.suffix not in ('.py','.md','.csv','.json','.jsonl','.txt'):continue
  rel=p.relative_to(r.HERE)
  if '__pycache__' in rel.parts or p.name in ('file_hashes.json','package_manifest.json','delivery.json'):continue
  if len(rel.parts)==1 and p.suffix not in ('.py','.md','.json'):continue
  if p.suffix=='.csv':
   head=p.read_text().splitlines()[0]
   assert not any(x in head.split(',') for x in ('record_id','pool_id','source_record_key','pooled_row','confidence')),(str(rel),head)
  files.append(p)
 hashes={str(p.relative_to(r.HERE)):r.sha(p) for p in sorted(files)};r.dump(report/'file_hashes.json',hashes);files.append(report/'file_hashes.json')
 record={'included_files':len(files)+1,'included_paths':[str(p.relative_to(r.HERE)) for p in files],'candidate_array_extensions_forbidden':['.npz','.npy','.pkl','.pickle','.xlsx'],'private_raw_input_files_included':False,'all_csv_headers_checked_for_record_identifiers':True};r.dump(report/'package_manifest.json',record);files.append(report/'package_manifest.json')
 archive=r.HERE/'policy_background7_results.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in sorted(files):z.write(p,str(p.relative_to(r.HERE)))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None;assert all(Path(n).suffix not in ('.npz','.npy','.xlsx') for n in z.namelist())
 r.dump(r.HERE/'delivery.json',{'file':str(archive),'bytes':archive.stat().st_size,'sha256':r.sha(archive),'entries':len(files),'aggregate_only':True})
 print('PACKAGED',archive.stat().st_size,r.sha(archive))
if __name__=='__main__':run()
