"""Exact-domain, bounded CEGIS for a depth-2 subproblem.

Q5 candidates are constrained by proved one-query UNSAT subsets. Deadline or
any incompletely examined query leaves the overall result UNKNOWN.
"""
from functools import lru_cache
import time
import numpy as np
from model import FULL,codes,partition,canonical_rows
from solver import Solver,Status,Result,depth2_domains,query_options
from solve_all_orbits import Q4Generator


def solve_two(states,seconds=180,beam=16):
    states=canonical_rows(states);deadline=time.perf_counter()+seconds
    domains=depth2_domains(states)
    if any(d==0 for d in domains):return Result(Status.UNSAT,reason='exact depth-2 fiber obstruction')
    gen=Q4Generator(states);z3=gen.z3
    for t in range(6):
        qs=[q for q,_ in query_options(states,t,domains[t])]
        gen.solver.add(z3.Or([z3.And([v==z3.BoolVal(bool(q>>i&1)) for i,v in enumerate(gen.vars[t])
                                    if not z3.is_false(v)]) for q in qs]))
    engine=Solver(node_limit=1000000);unknown_seen=False;trials=0
    while time.perf_counter()<deadline:
        models=gen.models(beam)
        if not models:
            status=gen.solver.check()
            if status==z3.unsat and not unknown_seen:return Result(Status.UNSAT,reason='complete exact-domain CEGIS exclusion')
            return Result(Status.UNKNOWN,reason='CEGIS constraint timeout / incomplete query')
        ranked=[]
        for q in models:
            cnt=np.bincount(codes(states,q),minlength=729).astype(np.int64)
            ranked.append(((int(cnt.max()),int(cnt@cnt)),q))
        for _,q in sorted(ranked):
            if time.perf_counter()>deadline:break
            trials+=1;children={};failed=False
            for c,cell in sorted(partition(states,q).items(),key=lambda kv:-len(kv[1])):
                ans=engine.last_query(cell,min(deadline,time.perf_counter()+5))
                if ans.status==Status.UNKNOWN:
                    unknown_seen=True;gen.block(q);failed=True;break
                if ans.status==Status.UNSAT:
                    core=cell.copy()
                    # Greedy deletion preserves only independently complete
                    # UNSAT judgments; UNKNOWN trials never shrink a core.
                    if len(core)<=32:
                        for j in range(len(core)-1,-1,-1):
                            if time.perf_counter()>deadline:break
                            smaller=np.delete(core,j,axis=0)
                            r=engine.last_query(smaller,min(deadline,time.perf_counter()+0.05))
                            if r.status==Status.UNSAT:core=smaller
                    gen.engine.failure_cache.add(core,1,ans);gen.feedback();failed=True;break
                children[str(c)]=ans.tree
            if not failed:return Result(Status.SAT,{'query':list(q),'children':children},reason=f'CEGIS q5 trials {trials}')
    return Result(Status.UNKNOWN,reason=f'bounded CEGIS deadline; {trials} query trials')
