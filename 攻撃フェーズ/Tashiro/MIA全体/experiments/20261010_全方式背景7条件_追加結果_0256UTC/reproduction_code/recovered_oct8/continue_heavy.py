"""Serialize recovered heavy families and audit each completed original baseline."""
from pathlib import Path
import json,os,subprocess,sys,time
HERE=Path(__file__).resolve().parent
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
def audit(family):
    subprocess.run([sys.executable,str(HERE/'audit_catalog_baseline.py'),'--family',family],env=ENV,check=True)
    r=json.loads((HERE/family/'prelim/historical_catalog_regression.json').read_text())
    assert r['failed']==0 and not r['catalog_without_result'],r
print('Waiting for geometry completion before alignment/local.',flush=True)
while not (HERE/'geometry/prelim/completion.json').exists():time.sleep(10)
audit('geometry')
for family in ('alignment','local'):
    if not (HERE/family/'prelim/completion.json').exists():
        print('START',family,flush=True)
        with (HERE/f'{family}.log').open('a') as fp:
            subprocess.run([sys.executable,str(HERE/'run_grid.py'),'--family',family],env=ENV,stdout=fp,stderr=subprocess.STDOUT,check=True)
    audit(family)
    print('COMPLETE',family,flush=True)
