"""q3=30 continuation: proof nogoods + minimax beam/LDS ordering.

A finite restricted grid can be UNSAT; the full depth-three problem remains
UNKNOWN unless a complete witness is independently replayed.
"""
import argparse
import json
import gzip
from pathlib import Path
import time
import numpy as np
from model import generate, canonical_rows, fingerprint, Q4_30, FULL, POPC, MASKS, codes, partition
from solver import ParentScreen, Solver, Status, Result, FailureCache, active_query_masks


def read_json(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as f:
        return json.load(f)


def write_json(path,data):
    with (gzip.open(path,'wt') if str(path).endswith('.gz') else open(path,'w')) as f:
        json.dump(data,f,ensure_ascii=False,separators=(',',':'))


def normalized(query):
    return tuple(min(int(q), FULL ^ int(q)) for q in query)


def mask_order(states, fixed, team):
    # Minimax and sumsq order for the partial question with this team added.
    base = list(fixed); base[team] = 0
    c = codes(states,base)
    score = []
    for q in range(1024):
        x = c + POPC[states[:,team] & q].astype(np.int32) * 3**team
        cnt = np.bincount(x,minlength=729).astype(np.int64)
        score.append((int(cnt.max()), int(cnt@cnt), q))
    return [q for _,_,q in sorted(score)]


def grid_search(states, fixed, varying=(1,3), seconds=300, beam=32, cache=None):
    assert len(varying) == 2
    start = time.perf_counter(); engine = ParentScreen(states,cache)
    order_base = list(fixed)
    for t in varying: order_base[t] = 0
    orders = [mask_order(engine.states,order_base,t) for t in varying]
    # LDS: visit low total rank discrepancies first; beam reranks each block
    # by actual complete-query minimax and sumsq scores.
    i,j = np.indices((1024,1024)); priority = np.lexsort((j.ravel(),i.ravel(),(i+j).ravel()))
    first = np.asarray(orders[0])[i.ravel()[priority]]
    second = np.asarray(orders[1])[j.ravel()[priority]]
    available = np.ones((1024,1024),dtype=bool)
    evaluated = []; survivors = []; cursor = 0; applied = 0; last = start

    def apply_feedback():
        nonlocal applied
        for e in engine.failure_cache.entries[applied:]:
            if e['depth'] != 2: continue
            if not all(e['uniform'][t][fixed[t]] for t in range(6) if t not in varying): continue
            a = np.flatnonzero(e['uniform'][varying[0]])
            b = np.flatnonzero(e['uniform'][varying[1]])
            available[np.ix_(a,b)] = False
        applied = len(engine.failure_cache.entries)

    apply_feedback()
    while cursor < len(first) and time.perf_counter()-start < seconds:
        batch = []
        while cursor < len(first) and len(batch) < beam:
            a,b = int(first[cursor]),int(second[cursor]);cursor += 1
            if not available[a,b]: continue
            q = list(fixed);q[varying[0]] = a;q[varying[1]] = b
            cnt = np.bincount(codes(engine.states,q),minlength=729).astype(np.int64)
            batch.append(((int(cnt.max()),int(cnt@cnt)),q,a,b))
        batch.sort(key=lambda x:x[0])
        for score,q,a,b in batch:
            if not available[a,b]: continue
            bad,counts = engine.evaluate(q,early=True)
            available[a,b] = False
            evaluated.append([a,b])
            if not bad:
                survivors.append({'query':q,'largest_child':int(counts.max()),'branches':int(np.count_nonzero(counts)),
                                  'status':'NECESSARY_CONDITION_ONLY'})
                print(json.dumps({'survivor':survivors[-1]},ensure_ascii=False),flush=True)
            apply_feedback()
        if time.perf_counter()-last >= 5:
            print(json.dumps({'varying':varying,'fixed':fixed,'evaluated':len(evaluated),
                              'excluded':int(available.size-np.count_nonzero(available)),
                              'cores':len(engine.failure_cache.entries),'survivors':len(survivors),
                              'seconds':round(time.perf_counter()-start,3)},ensure_ascii=False),flush=True)
            last = time.perf_counter()
    remaining = int(np.count_nonzero(available))
    restricted = 'UNSAT' if remaining == 0 and not survivors else 'UNKNOWN'
    return {'fixed_query':fixed,'varying_team_indices':list(varying),'representatives_per_team':1024,
            'evaluated':len(evaluated),'excluded_or_evaluated':1024**2-remaining,'remaining':remaining,
            'survivors':survivors,'restricted_status':restricted,'overall_status':'UNKNOWN',
            'seconds':time.perf_counter()-start,'beam_width':beam,'ordering':'minimax beam / rank-sum LDS',
            'core_count_at_end':len(engine.failure_cache.entries),
            'constraint_cache_hits':engine.hits,'constraint_cache_misses':engine.misses},engine.failure_cache


def serializable_cores(cache):
    return [{'depth':e['depth'],'states':e['states'].tolist(),'reason':e['reason']} for e in cache.entries]


def main():
    p = argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--beam',type=int,default=32);p.add_argument('--output',default='continuation_results.json')
    p.add_argument('--fixed',nargs=6,type=int);p.add_argument('--third-team',type=int)
    p.add_argument('--third-masks',nargs='*',type=int)
    p.add_argument('--load-cores')
    p.add_argument('--third-all',action='store_true');p.add_argument('--skip-baseline',action='store_true')
    args = p.parse_args();s = generate(30);fixed = normalized(args.fixed or Q4_30)
    results = [];cache = None
    if args.load_cores:
        old = read_json(args.load_cores);cache = FailureCache()
        for e in old['failure_cores']:
            cache.add(np.asarray(e['states'],dtype=np.uint16),e['depth'],Result(Status.UNSAT,reason=e['reason']))
    if not args.skip_baseline:
        result,cache = grid_search(s,fixed,seconds=args.seconds,beam=args.beam,cache=cache)
        results.append(result)
    if args.third_team is not None:
        masks = args.third_masks or []
        if args.third_all:
            candidates = set(active_query_masks(tuple(map(int,np.unique(s[:,args.third_team])))))
            masks = [q for q in mask_order(s,list(fixed),args.third_team) if q in candidates]
        for m in masks:
            q = list(fixed);q[args.third_team] = min(m,FULL^m)
            r,cache = grid_search(s,q,seconds=args.seconds,beam=args.beam,cache=cache)
            results.append(r)
            write_json(args.output,{'q3_code':30,'states':len(s),'state_hash':fingerprint(s),'results':results,
                       'third_team_index':args.third_team,'third_all_requested':args.third_all,'third_all':False,
                       'failure_cores':serializable_cores(cache),'overall_status':'UNKNOWN'})
            print(json.dumps({'slice_done':m,'restricted_status':r['restricted_status'],
                              'seconds':r['seconds'],'cores':len(cache.entries),'survivors':len(r['survivors'])}),flush=True)
            if r['survivors']: break
    out = {'q3_code':30,'states':len(s),'state_hash':fingerprint(s),'results':results,
           'third_team_index':args.third_team,'third_all_requested':args.third_all,
           'third_all':args.third_all and {r['fixed_query'][args.third_team] for r in results} == set(masks),
           'failure_cores':serializable_cores(cache),'overall_status':'UNKNOWN'}
    write_json(args.output,out)
    print(json.dumps({k:v for k,v in out.items() if k!='failure_cores'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
