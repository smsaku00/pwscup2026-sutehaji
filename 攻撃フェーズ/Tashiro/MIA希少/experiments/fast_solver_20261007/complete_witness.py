"""Construct all q5/q6 witnesses after a q4 passes screening, then transport.

A q4 necessary-condition survivor is not SAT until every branch is solved.
UNKNOWN branches are preserved in a resumable checkpoint.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from model import (generate, partition, fingerprint, state_key, codes, N,
                   verified_q3_automorphisms, transform_code)
from solver import Solver, Status
from verify import verify_certificate, response


def transform_mask(mask,pm):
    return sum(1<<pm[i] for i in range(N) if mask>>i&1)


def transform_rows(states,pm,tm):
    lut = np.array([transform_mask(m,pm) for m in range(2048)],dtype=np.uint16)
    out = np.zeros_like(states)
    for t in range(6): out[:,tm[t]] = lut[states[:,t]]
    return out


def transform_tree(node,pm,tm):
    if node.get('leaf'): return {'leaf':True}
    q = [0]*6
    for t,m in enumerate(node['query']): q[tm[t]] = transform_mask(m,pm)
    if node.get('terminal'): return {'query':q,'terminal':True}
    return {'query':q,'children':{str(transform_code(int(c),tm)):transform_tree(child,pm,tm)
                                 for c,child in node['children'].items()}}


def check_partitions(source,dest,a,b,tm,depth=0):
    # Compare every corresponding state response at every q4/q5 node, not
    # merely a multiset of bucket sizes.
    if a.get('leaf'): return
    ca = [response(row,a['query']) for row in source]
    cb = [response(row,b['query']) for row in dest]
    if any(transform_code(x,tm) != y for x,y in zip(ca,cb)):
        raise ValueError('transported partition mismatch')
    if a.get('terminal'): return
    for c,node in a['children'].items():
        ix = [i for i,x in enumerate(ca) if x == int(c)]
        check_partitions([source[i] for i in ix],[dest[i] for i in ix],node,
                         b['children'][str(transform_code(int(c),tm))],tm,depth+1)


def explicit_q6_witnesses(rows,tree):
    """Also record a concrete zero-mask witness for singleton q5 cells."""
    if tree.get('leaf'):
        return {'query':[0]*6,'children':{'0':{'query':[0]*6,'terminal':True}}}
    if tree.get('terminal'):
        return {'query':tree['query'],'children':{str(c):{'query':[0]*6,'terminal':True}
                                                for c in partition(rows,tree['query'])}}
    return {'query':tree['query'],'children':{c:({'query':[0]*6,'terminal':True} if child.get('leaf') else child)
                                            for c,child in tree['children'].items()}}


def main():
    p = argparse.ArgumentParser();p.add_argument('--query',nargs=6,type=int,required=True)
    p.add_argument('--seconds-per-branch',type=float,default=30);p.add_argument('--sample-queries',type=int,default=1024)
    p.add_argument('--query-budget',type=int,default=64);p.add_argument('--output-dir',default='../../analysis/code30_continuation_20261007')
    args = p.parse_args();out = Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    s = generate(30);parts = partition(s,args.query);checkpoint = out/'q3_30_checkpoint.json'
    if checkpoint.exists():
        data = json.loads(checkpoint.read_text())
        if data['query'] != args.query or data['state_hash'] != fingerprint(s): raise ValueError('checkpoint input mismatch')
    else: data = {'query':args.query,'state_hash':fingerprint(s),'branches':{}}
    seeds = []
    for code in (273,274):
        seeds.extend(json.loads(Path(f'../../analysis/q3_{code}_q5_map.json').read_text()).values())
    seeds = list(dict.fromkeys(map(tuple,seeds)))
    engine = Solver(sample_queries=args.sample_queries,query_budget=args.query_budget,
                    seed_queries=seeds,node_limit=1000000)
    for code,rows in sorted(parts.items(),key=lambda x:(-len(x[1]),x[0])):
        if data['branches'].get(str(code),{}).get('status') == 'SAT': continue
        start = time.perf_counter();ans = engine.solve(rows,2,seconds=args.seconds_per_branch)
        record = {'states':len(rows),'state_hash':fingerprint(rows),'status':ans.status.value,
                  'seconds':time.perf_counter()-start,'reason':ans.reason}
        if ans.status == Status.SAT:
            cert = {'q3_code':30,'q4_query':args.query,'q4_outcome':code,
                    'depth':2,'state_hash':fingerprint(rows),'tree':ans.tree}
            record['independent_replay'] = verify_certificate(cert)
            record['tree'] = ans.tree
        data['branches'][str(code)] = record
        checkpoint.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
        print(json.dumps({k:v for k,v in record.items() if k!='tree'}|{'q4_code':code},ensure_ascii=False),flush=True)
        if ans.status == Status.UNSAT: break
    if len(data['branches']) != len(parts) or any(e['status'] != 'SAT' for e in data['branches'].values()):
        print(json.dumps({'fixed_q4_status':'UNSAT' if any(e['status']=='UNSAT' for e in data['branches'].values()) else 'UNKNOWN',
                          'overall_status':'UNKNOWN','solved_branches':sum(e['status']=='SAT' for e in data['branches'].values()),
                          'reachable_branches':len(parts)}),flush=True);return
    tree = {'query':args.query,'children':{c:explicit_q6_witnesses(parts[int(c)],e['tree'])
                                         for c,e in data['branches'].items()}}
    reports = []
    for target in (30,246,270):
        if target == 30:
            dest = s;target_tree = tree;mapping = None
        else:
            pm,tm = next((pm,tm) for pm,tm in verified_q3_automorphisms() if transform_code(30,tm)==target)
            dest = transform_rows(s,pm,tm);actual = generate(target)
            if state_key(dest) != state_key(actual): raise ValueError('transported state-set mismatch')
            target_tree = transform_tree(tree,pm,tm);check_partitions(s.tolist(),dest.tolist(),tree,target_tree,tm)
            mapping = {'candidate_old_to_new':pm,'team_old_to_new':tm}
        cert = {'q3_code':target,'depth':3,'state_hash':fingerprint(dest),'tree':target_tree,'transport':mapping}
        report = verify_certificate(cert);reports.append(report)
        (out/f'q3_{target}_certificate.json').write_text(json.dumps(cert,ensure_ascii=False,separators=(',',':')))
    (out/'full_replay.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
    print(json.dumps({'overall_status':'SAT','replays':reports},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
