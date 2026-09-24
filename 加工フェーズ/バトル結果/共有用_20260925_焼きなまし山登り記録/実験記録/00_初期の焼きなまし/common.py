"""Fixed, auditable inputs for the approved C-versus-background experiment."""
from pathlib import Path
import os, sys, json, hashlib, importlib.util, shutil, subprocess, time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
RUN=HERE.parent
ROOT=RUN.parents[2]
OLD=RUN/'神奈川387人固定_3案評価'
PYTHON=ROOT/'スターターキット/初版/starter/.venv/bin/python'
QI=['age','SBP','TG','HDL','ALT','sex','prefecture']
COLS=['age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG','time','onset','death']
GROUPS=['通常・未発症','通常・発症','希少・未発症','希少・発症']
QUOTA=np.array([138,12,9,4])
def load(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return pd.read_csv(p,float_precision='round_trip')
def out(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(v).to_csv(p,index=False,encoding='utf-8-sig',float_format='%.17g')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def keys(d):return set(d[QI].itertuples(index=False,name=None))
def mod(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def plan():return load(HERE/'計画.json')
def rate(pm,pc):return float(np.clip((pm-pc)/max(1-pc,1e-9),0,1))
def predict_file(p,f):
    d=read(f);d=d[d.target_k.eq(p['target_id'])];q=read(p['challenge'])
    assert d.challenge_row_id.is_unique and set(d.challenge_row_id)==set(q.challenge_row_id)
    x=d.set_index('challenge_row_id').loc[q.challenge_row_id,'time_hat'].to_numpy(float)
    assert np.isfinite(x).all() and (x>=0).all() and (x<=10).all()
    return x
def attack_job(dataset,variant,a,panels,cpath):
    dest=HERE/'work'/dataset/variant/a['id'];meta=dest/'実行記録.json'
    inputs={'C':sha(cpath),'B':sha(OLD/'public/B_attacker.csv'),'A':sha(OLD/'public/A_bg.csv'),
            **{p['id']:sha(p['challenge']) for p in panels},'attack':a['sha256']}
    if meta.exists():
        r=load(meta);assert r['inputs']==inputs and r['returncode']==0 and sha(r['prediction'])==r['prediction_sha256'];return r
    dest.mkdir(parents=True,exist_ok=True);src=RUN/a['path'];script=dest/src.name;shutil.copyfile(src,script)
    assert sha(script)==a['sha256']
    for asset in a['assets']:shutil.copyfile(src.parent/asset['name'],dest/asset['name'])
    targets=[dict(target_k=p['target_id'],c_path=str(cpath),aia_challenge_path=p['challenge']) for p in panels]
    save(dest/'targets.json',targets)
    cmd=[str(PYTHON),'-B',str(script),'--b',str(OLD/'public/B_attacker.csv'),'--a-bg',str(OLD/'public/A_bg.csv'),'--targets',str(dest/'targets.json')]
    start=time.monotonic()
    with (dest/'実行.log').open('w') as log:
        proc=subprocess.run(cmd,cwd=dest,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    pred=dest/('F_aia_'+script.stem+'.csv')
    assert proc.returncode==0 and pred.exists(),(variant,a['name'],str(dest/'実行.log'))
    for p in panels:predict_file(p,pred)
    r=dict(dataset=dataset,variant=variant,attack=a['id'],command=cmd,inputs=inputs,returncode=proc.returncode,
           prediction=str(pred),prediction_sha256=sha(pred),seconds=time.monotonic()-start,panel_count=len(panels))
    save(meta,r);return r
def predictions(dataset,variant,a,p):
    r=load(HERE/'work'/dataset/variant/a['id']/'実行記録.json');assert sha(r['prediction'])==r['prediction_sha256'];return predict_file(p,r['prediction'])
