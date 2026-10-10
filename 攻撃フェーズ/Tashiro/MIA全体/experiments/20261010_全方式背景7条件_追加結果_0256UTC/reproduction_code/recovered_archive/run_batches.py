from pathlib import Path
import sys,subprocess,time,json,concurrent.futures
BASE=Path(__file__).resolve().parent
BGS=['prelim','starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
FAMILIES=['redteam','library','cross','atk','density','virtual','initial_gen','raw_gen','domias','aia2mia','outcome']
JOBS=[(bg,f) for bg in BGS for f in FAMILIES]
START=time.time();STATUS={}
(BASE/'logs').mkdir(exist_ok=True)
def run(job):
 bg,f=job;dest=BASE/'results'/bg/f
 if (dest/'completion.json').exists():return job,0,'cached'
 log=BASE/'logs'/f'{bg}__{f}.log'
 with log.open('w') as out:r=subprocess.run([sys.executable,str(BASE/'run_archive.py'),'--bg',bg,'--family',f],stdout=out,stderr=subprocess.STDOUT)
 return job,r.returncode,str(log)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(run,j) for j in JOBS]
 for fut in concurrent.futures.as_completed(futures):
  (bg,f),rc,log=fut.result();STATUS[bg+'__'+f]={'returncode':rc,'log':log,'finished_at':time.time()}
  (BASE/'batch_status.json').write_text(json.dumps({'started_at':START,'elapsed_seconds':time.time()-START,'done':len(STATUS),'total':len(JOBS),'status':STATUS},indent=2))
  print(bg,f,rc,len(STATUS),len(JOBS),flush=True)
