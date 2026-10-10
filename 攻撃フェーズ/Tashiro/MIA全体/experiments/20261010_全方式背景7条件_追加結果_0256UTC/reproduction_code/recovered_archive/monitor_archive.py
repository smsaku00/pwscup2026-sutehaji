from pathlib import Path
import subprocess,time,sys,json
BASE=Path(__file__).resolve().parent
last=-1
while True:
 count=len(list((BASE/'results').glob('*/*/completion.json')))
 if count!=last:
  subprocess.run([sys.executable,str(BASE/'aggregate_archive.py')],check=True)
  last=count
  if count==77:
   subprocess.run([sys.executable,str(BASE/'verify_archive.py')],check=True)
   break
 status=BASE/'batch_status.json'
 if status.exists():
  d=json.loads(status.read_text())
  if d['done']==d['total'] and count<77:
   print('BATCH_TERMINAL_WITH_FAILURES',count,flush=True);break
 time.sleep(30)
