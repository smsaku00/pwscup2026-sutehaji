from pathlib import Path
import concurrent.futures,subprocess,sys,json
BASE=Path(__file__).resolve().parent
BGS=['main','starter','prelim','starter+prelim','starter+main','prelim+main','starter+prelim+main']
def run(bg):
 with (BASE/'logs'/f'rare_accelerated__{bg}.log').open('w') as f:p=subprocess.run([sys.executable,'-u',str(BASE/'run_rare.py'),'--background',bg],stdout=f,stderr=subprocess.STDOUT)
 print(bg,p.returncode,flush=True);return {'background':bg,'returncode':p.returncode}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:r=list(e.map(run,BGS))
(BASE/'rare_acceleration_status.json').write_text(json.dumps(r,indent=2))
