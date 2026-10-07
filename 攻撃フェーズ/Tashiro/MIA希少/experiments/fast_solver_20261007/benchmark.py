"""Reproducible same-machine benchmarks, run without competing CPU workloads."""
import json
import platform
import statistics
import time
from pathlib import Path
import numpy as np
import reference
from model import *
from solver import *
from verify import verify


def measure(fn, repeat=3):
    timings=[]; values=[]
    for _ in range(repeat):
        t=time.perf_counter(); values.append(fn()); timings.append(time.perf_counter()-t)
    return statistics.median(timings),timings,values[-1]


def main():
    s=generate(30); validate(s)
    p=partition(s,Q4_30); b=p[273]
    report={'model':'ideal count-vector oracle; no official scorer',
            'python':platform.python_version(),'numpy':np.__version__,
            'platform':platform.platform(),'state_hash':fingerprint(s),
            'q3_code':30,'states':len(s),'tests':[]}
    # A: exact depth-2 necessary condition on the known 662-state obstruction.
    times_ref=[]; times_fast=[]
    for _ in range(3):
        reference.one.cache_clear(); reference.valid.cache_clear()
        t=time.perf_counter(); a=reference.domains(b); times_ref.append(time.perf_counter()-t)
        fiber_valid.cache_clear(); scalar_one.cache_clear(); active_query_masks.cache_clear()
        t=time.perf_counter(); z=depth2_domains(b); times_fast.append(time.perf_counter()-t)
        assert a == z
    report['tests'].append({'name':'662_state_depth2_necessary_cold',
        'reference_seconds':statistics.median(times_ref),
        'fast_seconds':statistics.median(times_fast),'reference_runs':times_ref,
        'fast_runs':times_fast,'identical':True,
        'allowed_masks_by_team':[x.bit_count() for x in z]})
    # B: a batch of DIFFERENT q4 candidates, to measure reusable team constraints.
    candidates=[]
    for a in (1598,638,63):
        for bmask in (721,877,147):
            q=list(Q4_30);q[1]=a;q[3]=bmask;candidates.append(q)
    reference.one.cache_clear(); reference.valid.cache_clear()
    fiber_valid.cache_clear(); scalar_one.cache_clear(); active_query_masks.cache_clear()
    t=time.perf_counter(); expect=[reference.screen(s,q) for q in candidates]; ref_cold=time.perf_counter()-t
    t=time.perf_counter(); engine=ParentScreen(s); got=[engine.evaluate(q)[0] for q in candidates]; fast_cold=time.perf_counter()-t
    assert got == expect
    ref_warm,rt,_=measure(lambda:[reference.screen(s,q) for q in candidates])
    fast_warm,ft,_=measure(lambda:[engine.evaluate(q)[0] for q in candidates])
    report['tests'].append({'name':'9_distinct_q4_screens', 'states_each':len(s),
        'reference_cold_total_seconds':ref_cold,'fast_cold_total_seconds':fast_cold,
        'reference_warm_batch_seconds':ref_warm,'fast_warm_batch_seconds':fast_warm,
        'reference_warm_runs':rt,'fast_warm_runs':ft, 'identical':True,
        'candidate_queries':candidates,
        'per_query_bad_child_counts':[len({x[0] for x in bad}) for bad in got],
        'constraint_cache_hits':engine.hits,'constraint_cache_misses':engine.misses})
    # C: last-question witness building, not a cached identical-query lookup.
    rng=np.random.default_rng(442)
    tests=[]
    for n in (4,6,8,10,12):
        for _ in range(4):
            base=p[int(rng.choice(list(p)))]; tests.append(base[rng.choice(len(base),min(n,len(base)),replace=False)])
    def ref_last():return [reference.last_query(x) for x in tests]
    def opt_last():
        e=Solver(); return [e.last_query(x) for x in tests]
    ref_time,rr,ref_ans=measure(ref_last)
    opt_time,ff,opt_ans=measure(opt_last)
    for x,ref,ans in zip(tests,ref_ans,opt_ans):
        assert (ref is not None) == (ans.status==Status.SAT)
        if ans.status==Status.SAT: verify(x,ans.tree,1)
    report['tests'].append({'name':'20_last_question_cells_cold_solver',
        'reference_seconds':ref_time,'fast_seconds':opt_time,'reference_runs':rr,'fast_runs':ff,
        'cells':len(tests),'cell_sizes':[len(x) for x in tests],
        'all_classifications_match':True,'SAT':sum(a.status==Status.SAT for a in opt_ans)})
    report['limitations']=[
        'Reference is a reconstruction of the earlier loop-based method, not old timing logs.',
        'Both versions run on identical inputs on one CPU process; warm and cold are separate.',
        'The benchmark times kernels and screening, not completion of all 53 orbits.',
        'All timings include returned results; imports and generation of the synthetic states are excluded.',
        'No official mean-TPR precision, padding, or CSV end-to-end validation is included.'
    ]
    Path('benchmark_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
