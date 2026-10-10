from pathlib import Path
import subprocess,sys,time,json
BASE=Path(__file__).resolve().parent
jobs=[(bg,f) for bg in ['starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main'] for f in ['library','cross','domias','aia2mia','outcome']]
status=[];start=time.time()
for bg,f in jobs:
 with (BASE/'logs'/f'light_{bg}__{f}.log').open('w') as log:r=subprocess.run([sys.executable,str(BASE/'run_archive.py'),'--bg',bg,'--family',f],stdout=log,stderr=subprocess.STDOUT)
 status.append({'background':bg,'family':f,'returncode':r.returncode})
 (BASE/'light_status.json').write_text(json.dumps({'done':len(status),'total':len(jobs),'elapsed':time.time()-start,'status':status},indent=2)+'\n')
 print(bg,f,r.returncode,flush=True)
 subprocess.run([sys.executable,str(BASE/'aggregate_archive.py')],stdout=subprocess.DEVNULL)
