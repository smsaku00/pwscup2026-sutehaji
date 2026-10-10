from pathlib import Path
import json,numpy as np
BASE=Path(__file__).resolve().parent
n=0
for file in (BASE/'results').glob('*/*/completion.json'):
 folder=file.parent;done=json.loads(file.read_text());bg=done['background'];nbg=5143*len(bg.split('+'));data=np.load(folder/'frozen.npz');teams=done['teams'];ids={name:[teams[j] for j in range(len(teams)) if np.isfinite(data[name][:,j]).all()] for name in data.files}
 done['target_phase']='prelim';done['n_background']=nbg;done['applicable_target_ids']=ids;file.write_text(json.dumps(done,ensure_ascii=False,indent=2)+'\n')
 f=folder/'prediction_manifest.json';m=json.loads(f.read_text());m['target_phase']='prelim';m['n_background']=nbg;m['applicable_target_ids']=ids
 for name,meta in m['metadata'].items():
  if isinstance(meta,dict):
   if 'background' in meta:meta['source_historical_background_label']=meta.pop('background')
   meta['background_condition']=bg;meta['n_selected_background']=nbg
   if name in ids:meta['applicable_target_ids']=ids[name]
 f.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n');n+=1
print('metadata clarified',n)
