from pathlib import Path
import time,subprocess,sys
BASE=Path(__file__).resolve().parent
# Start the final independent background only after the main-only BG releases a slot.
while not (BASE/'results/prelim/main/qcal_sensitivity/completion.json').exists():time.sleep(10)
for family in ['geometry','additional','attributes','bootstrap','logistic','reference','fixed_sensitivity','qcal','qcal_sensitivity']:
 with (BASE/'logs'/f'last_background__{family}.log').open('w') as f:
  p=subprocess.run([sys.executable,'-u',str(BASE/'run_legacy.py'),'--background','starter+prelim+main','--family',family],stdout=f,stderr=subprocess.STDOUT)
 print(family,p.returncode,flush=True)
