from pathlib import Path
import concurrent.futures,sys,subprocess,json,time
BASE=Path(__file__).resolve().parent
FAMILIES=['geometry','additional','attributes','bootstrap','logistic','reference','fixed_sensitivity','qcal','qcal_sensitivity']
def job(bg):
 records=[]
 for fam in FAMILIES:
  path=BASE/'logs'/f'accelerated__{bg}__{fam}.log';start=time.time()
  with path.open('w') as f:p=subprocess.run([sys.executable,'-u',str(BASE/'run_legacy.py'),'--background',bg,'--family',fam],stdout=f,stderr=subprocess.STDOUT)
  row={'background':bg,'family':fam,'exitcode':p.returncode,'elapsed':time.time()-start};records.append(row);print(json.dumps(row),flush=True)
 return records
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:results=list(e.map(job,['starter+main','prelim+main','starter+prelim+main']))
(BASE/'acceleration_status.json').write_text(json.dumps(results,indent=2))
