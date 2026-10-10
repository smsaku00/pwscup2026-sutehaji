"""Wait for final seals, validate and build the bounded aggregate deliverable."""
import time,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BG=['prelim','starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
while not all((HERE/'results_transfer'/b/'complete.json').exists() and (HERE/'cache/p08d'/b/'complete.json').exists() for b in BG):time.sleep(30)
for script,args in [('validate_policy_results.py',['--all']),('build_policy_registry.py',[]),('rank_policy_backgrounds.py',[]),('build_policy_report.py',[])]:
 print('START',script,flush=True);subprocess.run([sys.executable,script,*args],cwd=HERE,check=True);print('DONE',script,flush=True)
