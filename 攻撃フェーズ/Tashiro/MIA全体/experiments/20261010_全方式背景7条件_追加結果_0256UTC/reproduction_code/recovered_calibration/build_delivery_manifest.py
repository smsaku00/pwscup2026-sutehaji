"""Allowlist sealed result panels without candidate-level or runtime-control files."""
from pathlib import Path
import csv,json,hashlib,time
from run_family import STAGE
from execution_protocol import PROTOCOL
CSV={'対象別TPR.csv','集約比較.csv','方式一覧.csv','交差検証分割.csv','標準化後係数.csv','学習係数.csv','収束の確認.csv','最適化の収束.csv','rank_by_target.csv'}
JSON={'prediction_hashes.json','catalog_regression.json','completion.json','実行記録.json','実装照合.json','numerical_reproduction_assessment.json'}
FORBIDDEN={'record_id','person_id','patient_id','score_row','B_row','person_index','train_indices','test_indices','row_id'}
def main():
 files=[];panels=[]
 for record in STAGE.glob('*/*/実行記録.json'):
  d=record.parent;family=d.parent.name
  if '_initial' in d.name or not (d/'completion.json').exists():continue
  rr=json.loads(record.read_text());seal=json.loads((d/'completion.json').read_text())
  if not seal.get('source_main_returned'):continue
  if (family=='classifiers' or family.startswith('learned') and family!='learned_yesterday') and rr.get('forest_prediction_execution')!=PROTOCOL:continue
  panels.append(dict(family=family,background=d.name,phase=rr.get('phase'),models=rr.get('models')))
  for p in d.rglob('*'):
   if not p.is_file() or p.name not in CSV|JSON:continue
   if p.suffix=='.csv':
    with p.open() as f:header=next(csv.reader(f),[])
    assert not FORBIDDEN.intersection(header),(p,header)
   text=p.read_text()
   if any('"'+k+'"' in text for k in FORBIDDEN):continue
   files.append(dict(path=str(p.relative_to(STAGE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,sanitize_execution_metadata_before_delivery=p.name=='実行記録.json'))
 manifest=dict(snapshot_time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),complete_request=False,panels=panels,files=files,exclusions=['All candidate-level predictions, raw inputs, fit checkpoints and row/fold assignments','All runtime-control *_initial directories','Unsealed running panels','Execution logs and ownership/coordination metadata'],record_sanitization='Remove PID, local paths and execution ownership fields recursively; preserve source hashes, numeric versions, fixed recipe parameters, aggregate counts and validity caveats. Do not modify pinned source code.')
 (STAGE/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(len(panels),'sealed panels',len(files),'allowlisted result files')
if __name__=='__main__':main()
