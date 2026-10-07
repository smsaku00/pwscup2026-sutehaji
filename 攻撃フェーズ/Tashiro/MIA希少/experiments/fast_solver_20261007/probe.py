"""Bounded two-process experiment; no daemon, scheduler, or contest submission."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from model import *
from solver import *
from verify import verify


def worker(seed):
    s=generate(30); engine=ParentScreen(s); rng=np.random.default_rng(seed)
    a=[Q4_30[1]] + list(map(int,rng.choice(1024,15,replace=False)))
    b=[Q4_30[3]] + list(map(int,rng.choice(1024,31,replace=False)))
    tested=0; survivors=[]; firstfail={}; start=time.perf_counter()
    for x in a:
        for y in b:
            q=list(Q4_30);q[1]=x;q[3]=y
            bad,counts=engine.evaluate(q,early=True)
            tested+=1
            if not bad:
                survivors.append({'q4':q,'largest_child':int(counts.max()),
                                  'status':'NECESSARY_CONDITION_ONLY'})
            else:
                code,t,n=bad[0]; key=f'{code}:{t}'
                firstfail[key]=firstfail.get(key,0)+1
    return {'seed':seed,'tested':tested,'survivors':survivors,
            'seconds':time.perf_counter()-start,
            'constraint_cache_hits':engine.hits,'constraint_cache_misses':engine.misses,
            'first_failure_histogram':firstfail}


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();start=time.perf_counter()
    if args.workers==1: results=[worker(seed) for seed in (310071,310072)]
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(worker,(310071,310072)))
    grid_seconds=time.perf_counter()-start
    # Demonstrate the unified bounded solver on six SMALLER children, not a
    # claim that the parent is solved. Save every SAT witness for replay.
    s=generate(30); parts=partition(s,Q4_30); solver=Solver(beam_width=12,query_budget=24)
    leaf_probes=[]
    for code,b in sorted(parts.items(),key=lambda item:(len(item[1]),item[0]))[:6]:
        t=time.perf_counter(); ans=solver.solve(b,2,seconds=5.0)
        row={'q4_code':code,'states':len(b),'status':ans.status.value,
             'seconds':time.perf_counter()-t,'reason':ans.reason}
        if ans.status==Status.SAT:
            row['independently_verified']=verify(b,ans.tree,2)
            cert={'q3_code':30,'q4_query':list(Q4_30),'q4_outcome':code,
                  'depth':2,'state_hash':fingerprint(b),'tree':ans.tree}
            Path(f'certificate_q3_30_q4_{code}.json').write_text(json.dumps(cert,ensure_ascii=False,indent=2))
        leaf_probes.append(row)
    result={'q3_code':30,'states':len(s),'state_hash':fingerprint(s),
            'mode':'bounded two-team grid; necessary-condition pruning',
            'workers':min(args.workers,2),'grid_wall_seconds':grid_seconds,
            'candidate_evaluations':sum(x['tested'] for x in results),
            'survivors':sum(len(x['survivors']) for x in results),
            'overall_status':'UNKNOWN_NOT_A_FULL_DEPTH3_PROOF',
            'worker_results':results,'small_child_probes':leaf_probes,
            'solver_stats':solver.stats}
    Path('probe_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='worker_results'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
