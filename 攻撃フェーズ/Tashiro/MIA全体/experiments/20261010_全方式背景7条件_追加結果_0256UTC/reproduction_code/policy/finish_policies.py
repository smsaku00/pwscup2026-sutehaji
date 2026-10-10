"""Resume independent downstream stages once their sealed inputs exist."""
import time,subprocess,sys
import pandas as pd
from pathlib import Path
HERE=Path(__file__).resolve().parent
BG=['prelim','starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
def ready(bg):
 p=HERE/'cache/fixed22'/bg
 if not (p/'complete.json').exists() or not (p/'by_target.csv').exists():return False
 try:return len(pd.read_csv(p/'by_target.csv'))==9240
 except (pd.errors.EmptyDataError,pd.errors.ParserError):return False
def run(*args):
 print('START',*args,flush=True);subprocess.run([sys.executable,*args],cwd=HERE,check=True);print('DONE',*args,flush=True)
if sys.argv[1]=='public':
 while not all((HERE/'cache/p06'/b/'complete.json').exists() for b in BG):time.sleep(15)
 run('evaluate_public_policies.py','--families','p06');run('score_counterfactuals.py');run('run_p08d_looto.py')
elif sys.argv[1]=='transfer':
 pending=set(BG)
 while pending:
  for bg in BG:
   if bg in pending and ready(bg):
    run('evaluate_transfer.py','--backgrounds',bg);pending.remove(bg)
  if pending:time.sleep(15)
else:raise ValueError(sys.argv[1])
