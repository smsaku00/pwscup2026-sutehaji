from pathlib import Path
import subprocess,concurrent.futures,json,time,sys
BASE=Path(__file__).resolve().parent
BGS=['prelim','starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
FAMILIES=['geometry','additional','attributes','bootstrap','logistic','reference','fixed_sensitivity','qcal','qcal_sensitivity']
logs=BASE/'logs';logs.mkdir(exist_ok=True)
def job(bg):
 records=[]
 for family in FAMILIES:
  path=logs/f'prelim__{bg}__{family}.log'
  start=time.time()
  with path.open('w') as f:
   p=subprocess.run([sys.executable,'-u',str(BASE/'run_legacy.py'),'--background',bg,'--family',family],stdout=f,stderr=subprocess.STDOUT)
  record={'phase':'prelim','background':bg,'family':family,'exitcode':p.returncode,'wall_seconds':time.time()-start,'log':str(path)}
  records.append(record)
  (logs/f'status__{bg}.json').write_text(json.dumps(records,indent=2))
  print(json.dumps(record),flush=True)
 return records
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 results=list(ex.map(job,BGS))
(BASE/'batch_status.json').write_text(json.dumps(results,indent=2))
