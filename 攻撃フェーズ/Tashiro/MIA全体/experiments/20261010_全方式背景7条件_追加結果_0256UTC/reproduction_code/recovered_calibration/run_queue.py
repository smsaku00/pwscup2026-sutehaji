"""Local resumable job dispatcher. A worker runs exactly one BLAS1 job at a time."""
from pathlib import Path
import argparse,json,time,os,subprocess,fcntl,hashlib
import runtime as r
from run_family import STAGE,WORK,FOLDERS
from generate_features import VIEWCACHE,CONDITIONS,CACHE
from execution_protocol import PROTOCOL

ROOT=Path('/workspace/shared/mia_recovered_calibration');STATE=ROOT/'queue';STATE.mkdir(exist_ok=True)
COUNTS={'direct':1008,'classifiers':272,'attribute_seeds':320,'stable_centers':240,'learned_base':23,'learned_split':84,'learned_ablation':157,'learned_lowfpr':361,'learned_centers':249,'learned_yesterday':14,'learned_bg_ablation':25,'learned_bg_transfer':37,'learned_multiview':49,'learned_multiview_loo':25,'smooth':259,'learned_main_transfer':10,'learned_multiview_main':49}

def complete(family,bg):
    if family=='main_features':
        p=CACHE/bg/'main/代表特徴.npz';q=p.with_name('record.json')
        return p.exists() and q.exists() and json.loads(q.read_text()).get('forest_prediction_execution')==PROTOCOL
    if family=='views':return all((VIEWCACHE/bg/name/'代表特徴.npz').exists() and (VIEWCACHE/bg/name/'record.json').exists() for name,_,_ in CONDITIONS[1:])
    p=STAGE/family/bg/'実行記録.json';h=p.with_name('prediction_hashes.json')
    if not p.exists() or not h.exists():return False
    try:
        record=json.loads(p.read_text())
        affected=family=='classifiers' or (family.startswith('learned') and family!='learned_yesterday')
        return (not affected or record.get('forest_prediction_execution')==PROTOCOL) and record.get('models')==COUNTS[family] and len(json.loads(h.read_text()))==COUNTS[family]
    except (ValueError,KeyError):return False

def ready(family,bg):
    if family=='main_features':return True
    if family=='learned_main_transfer':return complete('main_features',bg)
    if family=='learned_multiview_main':return complete('main_features',bg) and complete('learned_multiview',bg)
    if family in ('classifiers','attribute_seeds','stable_centers','learned_yesterday'):return True
    if not (CACHE/bg/'代表特徴.npz').exists():return False
    if family in ('learned_base','learned_split'):
        return all(complete(f,bg) for f in ['direct','attribute_seeds','classifiers']) and (ROOT/'cache/source_features'/bg/'geometry.npz').exists()
    if family in ('learned_multiview','learned_multiview_loo'):return complete('views',bg) and complete('attribute_seeds',bg)
    if family=='learned_centers':return complete('attribute_seeds',bg)
    if family=='smooth':return complete('learned_split',bg) and complete('learned_centers',bg) and (ROOT/'cache/source_features'/bg/'smooth_dependencies.json').exists()
    return True

def jobs():
    rows=[]
    plans=[('fast',10,'attribute_seeds'),('fast',20,'stable_centers'),('fast',30,'learned_yesterday'),('fast',40,'learned_bg_transfer'),('fast',45,'learned_bg_ablation'),('fast',50,'learned_split'),('fast',55,'learned_ablation'),('fast',60,'learned_base'),('fast',65,'learned_centers'),('fast',70,'smooth'),('classifiers',10,'classifiers'),('lowfpr',10,'learned_lowfpr'),('augment',10,'views'),('multiview',10,'learned_multiview'),('multiview',20,'learned_multiview_loo'),('main',10,'main_features'),('main',20,'learned_main_transfer'),('main',30,'learned_multiview_main')]
    order=['prelim']+[b for b in r.BG if b!='prelim']
    for lane,priority,family in plans:
        for n,bg in enumerate(order):rows.append(dict(id=family+'__'+bg,lane=lane,priority=priority+n*.01,family=family,bg=bg))
    return sorted(rows,key=lambda j:j['priority'])

def command(job):
    f=job['family'];bg=job['bg']
    if f=='main_features':return ['python','-B',str(ROOT/'generate_features.py'),'--bg',bg,'--phase','main']
    if f=='learned_main_transfer':return ['python','-B',str(ROOT/'run_main_transfer.py'),'--bg',bg]
    if f=='learned_multiview_main':return ['python','-B',str(ROOT/'project_multiview_main.py'),'--bg',bg]
    if f=='views':return ['python','-B',str(ROOT/'generate_features.py'),'--bg',bg,'--views']
    script='run_learned_special.py' if f in ('learned_bg_transfer','learned_bg_ablation','learned_multiview','learned_multiview_loo') else 'run_family.py'
    return ['python','-B',str(ROOT/script),'--family',f,'--bg',bg]

def run(args):
    allowed=set(args.lanes.split(','));plan=jobs()
    if args.backgrounds:
        order=args.backgrounds.split(',');assert set(order).issubset(r.BG)
        plan=[j for j in plan if j['bg'] in order];plan.sort(key=lambda j:(int(j['priority']),order.index(j['bg'])))
    (STATE/'planned_jobs.json').write_text(json.dumps(jobs(),ensure_ascii=False,indent=2)+'\n')
    n=0
    while args.max_jobs is None or n<args.max_jobs:
        chosen=None;pending=False
        with (STATE/'lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for job in plan:
                if job['lane'] not in allowed or complete(job['family'],job['bg']):continue
                pending=True;record=STATE/(job['id']+'.json')
                if record.exists():
                    status=json.loads(record.read_text())
                    if status['state'] in ('running','failed'):continue
                if not ready(job['family'],job['bg']):continue
                chosen=job;record.write_text(json.dumps(dict(**job,state='running',worker=args.worker,started=time.time(),command=command(job)))+'\n');break
        if chosen is None:
            if not pending:print('QUEUE_COMPLETE',args.worker,flush=True);return
            print('QUEUE_WAITING',args.worker,flush=True);time.sleep(30);continue
        job=chosen;log=ROOT/(job['id']+'.log');env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        if job['family']!='attribute_seeds':env['PYTHONPATH']=str(ROOT/'deps_20261010')
        else:env.pop('PYTHONPATH',None)
        print('JOB_START',job['id'],flush=True)
        with log.open('a') as out:result=subprocess.run(command(job),stdout=out,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
        record=STATE/(job['id']+'.json');status=json.loads(record.read_text());status.update(state='complete' if result.returncode==0 and complete(job['family'],job['bg']) else 'failed',returncode=result.returncode,ended=time.time(),log=str(log));record.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        print('JOB_END',job['id'],status['state'],flush=True);n+=1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lanes',required=True);p.add_argument('--worker',required=True);p.add_argument('--max-jobs',type=int);p.add_argument('--backgrounds');run(p.parse_args())
