"""Bounded CEGIS query generation from independently checkable UNSAT cores.

Z3 SAT here means only that a QUERY passes learned nogoods. A decision-tree
SAT result still requires all q5/q6 branches and independent full replay.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import numpy as np
import z3
from model import generate, Q4_30, POPC, codes, fingerprint
from solver import ParentScreen, FailureCache, Result, Status
from continuation import read_json, write_json, serializable_cores


def main():
    p = argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--load-cores');p.add_argument('--beam',type=int,default=8)
    p.add_argument('--extra-teams','--changed-singletons',dest='extra_teams',type=int,default=1)
    p.add_argument('--output',default='constrained_results.json.gz');p.add_argument('--seed',type=int,default=310073)
    args = p.parse_args();start = time.perf_counter();states = generate(30)
    cache = FailureCache();engine = ParentScreen(states,cache)
    active = [[i for i in range(11) if np.any(states[:,t] & (1<<i))] for t in range(6)]
    variables = [[z3.Bool(f'q{t}_{i}') if i in active[t][:-1] else z3.BoolVal(False)
                  for i in range(11)] for t in range(6)]
    base = []
    for t in range(6):
        union = sum(1<<i for i in active[t]);q = Q4_30[t] & union
        if q & (1<<active[t][-1]): q ^= union
        base.append(q)
    solver = z3.Solver();solver.set(timeout=2000,random_seed=args.seed)
    changed = []
    for t in (0,2,4,5):
        changed.append(z3.Or([v != z3.BoolVal(bool(base[t]>>i&1)) for i,v in enumerate(variables[t])]))
    # The scalar pair Team25 counts as one extra changed team too.
    solver.add(z3.PbLe([(v,1) for v in changed],args.extra_teams))
    @lru_cache(maxsize=None)
    def count(t,m):
        return z3.Sum([z3.If(variables[t][i],1,0) for i in range(11) if m>>i&1])
    @lru_cache(maxsize=None)
    def split(t,members):
        return z3.Or([count(t,m) != count(t,members[0]) for m in members[1:]])
    seen = set();applied = 0
    def add_core(e):
        projections = tuple(tuple(map(int,np.unique(e['states'][:,t]))) for t in range(6))
        if projections in seen: return
        seen.add(projections)
        solver.add(z3.Or([split(t,ms) for t,ms in enumerate(projections)]))
    if args.load_cores:
        old = read_json(args.load_cores)
        for e in old['failure_cores']:
            cache.add(np.asarray(e['states'],dtype=np.uint16),2,Result(Status.UNSAT,reason=e['reason']))
        for e in cache.entries: add_core(e)
        applied = len(cache.entries)
    evaluated = 0;survivors = [];last = start;constraint_status = 'unknown'
    while time.perf_counter()-start < args.seconds:
        solver.push();batch = []
        for _ in range(args.beam):
            answer = solver.check();constraint_status = str(answer)
            if answer != z3.sat: break
            model = solver.model();q = tuple(sum(1<<i for i,v in enumerate(vs)
                    if z3.is_true(model.eval(v,model_completion=True))) for vs in variables)
            cnt = np.bincount(codes(states,q),minlength=729).astype(np.int64)
            batch.append(((int(cnt.max()),int(cnt@cnt)),q))
            solver.add(z3.Or([v != model.eval(v,model_completion=True) for vs in variables for v in vs
                             if not z3.is_false(v)]))
        solver.pop()
        if batch: constraint_status = 'sat'
        if not batch:
            # Solver timeout, or exhaustion of this RESTRICTED generator. It
            # is not an UNSAT statement about unrestricted depth three.
            break
        for score,q in sorted(batch):
            bad,counts = engine.evaluate(q,early=True);evaluated += 1
            for e in cache.entries[applied:]: add_core(e)
            applied = len(cache.entries)
            if not bad:
                survivors.append({'query':q,'branches':int(np.count_nonzero(counts)),
                                  'largest_child':int(counts.max()),'status':'NECESSARY_CONDITION_ONLY'})
                print(json.dumps({'survivor':survivors[-1]},ensure_ascii=False),flush=True)
        if time.perf_counter()-last >= 5:
            print(json.dumps({'evaluated':evaluated,'cores':len(cache.entries),'clauses':len(seen),
                              'seconds':round(time.perf_counter()-start,3),'constraint_status':constraint_status}),flush=True)
            last = time.perf_counter()
        if survivors: break
    result = {'q3_code':30,'states':len(states),'state_hash':fingerprint(states),
              'evaluated':evaluated,'survivors':survivors,'overall_status':'UNKNOWN',
              'generator_constraint_status':constraint_status,'extra_changed_teams':args.extra_teams,
              'seconds':time.perf_counter()-start,'beam':args.beam,'seed':args.seed,
              'z3_version':z3.get_version_string(),'failure_cores':serializable_cores(cache)}
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='failure_cores'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
