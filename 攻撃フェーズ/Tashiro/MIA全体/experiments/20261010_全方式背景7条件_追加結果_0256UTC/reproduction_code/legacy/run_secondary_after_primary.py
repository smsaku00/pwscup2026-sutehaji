from pathlib import Path
import time,subprocess,sys,json
BASE=Path(__file__).resolve().parent
while not (BASE/'batch_status.json').exists():time.sleep(30)
records=[]
for bg in ['main','starter','prelim','starter+prelim','starter+main','prelim+main','starter+prelim+main']:
 with (BASE/'logs'/f'rare__{bg}.log').open('w') as f:
  p=subprocess.run([sys.executable,'-u',str(BASE/'run_rare.py'),'--background',bg],stdout=f,stderr=subprocess.STDOUT)
 records.append({'background':bg,'returncode':p.returncode})
 print(records[-1],flush=True)
(BASE/'rare_batch_status.json').write_text(json.dumps(records,indent=2))
subprocess.run([sys.executable,str(BASE/'collect_results.py')],check=True)
