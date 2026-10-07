"""Complete the fixed-q3 depth-four constructive proof over every orbit.

No finite search failure is called global UNSAT. All SAT trees get an
independent integer replay before becoming authoritative certificates.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import gzip
import json
from pathlib import Path
import time
import numpy as np
from model import *
from solver import Solver, ParentScreen, Result, Status, query_options
from complete_witness import explicit_q6_witnesses
from verify import verify_certificate, verify_report

OUT = Path('../../analysis/full_depth4_20261007')
Q4_SEEDS = [(453,461,522,566,1201,1924),(888,211,39,93,724,111),
            (283,482,39,538,724,111),(432,962,939,562,392,951)]
KNOWN = {283:Q4_SEEDS[0],274:Q4_SEEDS[1],273:Q4_SEEDS[2]}


def read(path):
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else open(path)) as f:return json.load(f)


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with (gzip.open(temp,'wt') if str(path).endswith('.gz') else open(temp,'w')) as f:
        json.dump(data,f,ensure_ascii=False,separators=(',',':'))
    temp.replace(path)


def manifest():
    path=OUT/'manifest.json'
    if path.exists():return read(path)
    s=generate();counts=np.bincount(codes(s,Q3),minlength=729);remaining=set(map(int,np.flatnonzero(counts)))
    autos=verified_q3_automorphisms();orbits=[]
    while remaining:
        rep=min(remaining);orbit=sorted({transform_code(rep,tm) for _,tm in autos});remaining-=set(orbit)
        mappings={}
        for c in orbit:
            pm,tm=next((pm,tm) for pm,tm in autos if transform_code(rep,tm)==c)
            mappings[str(c)]={'candidate_old_to_new':pm,'team_old_to_new':tm}
        orbits.append({'representative':rep,'states':int(counts[rep]),'q3_codes':orbit,'mappings':mappings})
    data={'q3_query':list(Q3),'caps':list(CAPS),'candidate_ids':list(IDS),'total_states':len(s),
          'reachable_q3_codes':int(np.count_nonzero(counts)),
          'orbits':sorted(orbits,key=lambda x:(-x['states'],x['representative']))}
    if len(s)!=2494800 or len(orbits)!=53 or data['reachable_q3_codes']!=209:raise ValueError('unexpected full state universe')
    save(path,data);return data


class Q4Generator:
    """CEGIS over six mask columns, fed only proved failure subsets."""
    def __init__(self,states):
        import z3
        self.z3=z3;self.engine=ParentScreen(states);self.solver=z3.Solver()
        self.solver.set(timeout=2000,random_seed=310074)
        active=[]
        for t in range(6):
            union=0
            for m in np.unique(states[:,t]):union|=int(m)
            active.append([i for i in range(11) if union>>i&1])
        self.vars=[[z3.Bool(f'q{t}_{i}') if i in active[t][:-1] else z3.BoolVal(False)
                    for i in range(11)] for t in range(6)]
        self.count_cache={};self.split_cache={};self.applied=0;self.seen=set()
    def count(self,t,m):
        key=(t,m)
        if key not in self.count_cache:
            self.count_cache[key]=self.z3.Sum([self.z3.If(self.vars[t][i],1,0) for i in range(11) if m>>i&1])
        return self.count_cache[key]
    def feedback(self):
        z3=self.z3
        for e in self.engine.failure_cache.entries[self.applied:]:
            ps=tuple(tuple(map(int,np.unique(e['states'][:,t]))) for t in range(6))
            if ps in self.seen:continue
            self.seen.add(ps);clauses=[]
            for t,ms in enumerate(ps):
                key=(t,ms)
                if key not in self.split_cache:
                    self.split_cache[key]=z3.Or([self.count(t,m)!=self.count(t,ms[0]) for m in ms[1:]])
                clauses.append(self.split_cache[key])
            self.solver.add(z3.Or(clauses))
        self.applied=len(self.engine.failure_cache.entries)
    def block(self,q):
        self.solver.add(self.z3.Or([v!=self.z3.BoolVal(bool(q[t]>>i&1))
                                    for t,vs in enumerate(self.vars) for i,v in enumerate(vs)
                                    if not self.z3.is_false(v)]))
    def models(self,beam=16):
        z3=self.z3;self.feedback();self.solver.push();out=[]
        for _ in range(beam):
            if self.solver.check()!=z3.sat:break
            model=self.solver.model();q=tuple(sum(1<<i for i,v in enumerate(vs)
                   if z3.is_true(model.eval(v,model_completion=True))) for vs in self.vars)
            out.append(q);self.block(q)
        self.solver.pop();return out


def q5_seeds():
    seeds=[]
    for code in (273,274):seeds.extend(read(Path(f'../../analysis/q3_{code}_q5_map.json')).values())
    c=read(Path('../../analysis/code30_continuation_20261007/q3_30_certificate.json'))
    seeds.extend(child['query'] for child in c['tree']['children'].values())
    return list(dict.fromkeys(map(tuple,seeds)))


def legacy_branch(rows,q5,engine,deadline):
    children={}
    for c,cell in sorted(partition(rows,q5).items(),key=lambda kv:-len(kv[1])):
        r=engine.last_query(cell,deadline)
        if r.status!=Status.SAT:return r
        children[str(c)]=r.tree
    return Result(Status.SAT,{'query':list(q5),'children':children})


def try_q4(code,states,q4,seeds,seconds=15):
    folder=OUT/'orbits'/str(code);tag='-'.join(map(str,q4));cp=folder/f'q4_{tag}_checkpoint.json.gz'
    parts=partition(states,q4);engine=Solver(sample_queries=1024,query_budget=64,seed_queries=seeds,node_limit=1000000)
    data=read(cp) if cp.exists() else {'q4_query':list(q4),'branches':{}}
    legacy=read(Path(f'../../analysis/q3_{code}_q5_map.json')) if code in (273,274) and tuple(q4)==KNOWN[code] else {}
    start=time.perf_counter();last=start
    for outcome,rows in sorted(parts.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        previous=data['branches'].get(str(outcome),{})
        if previous.get('status')=='SAT':continue
        if previous.get('status')=='UNSAT':return Result(Status.UNSAT,reason='proved fixed-q4 child UNSAT')
        if str(outcome) in legacy:
            # Failure of a proposed q5 says nothing about OTHER q5 choices.
            # It must not become a depth-2 UNSAT result for this q4 child.
            proposed=legacy_branch(rows,legacy[str(outcome)],engine,time.perf_counter()+seconds)
            ans=proposed if proposed.status==Status.SAT else engine.solve(rows,2,seconds=seconds)
        else:
            ans=engine.solve(rows,2,seconds=seconds)
        if ans.status==Status.UNKNOWN:
            engine.sample_queries=4096;engine.query_budget=256
            ans=engine.solve(rows,2,seconds=max(seconds,30));engine.sample_queries=1024;engine.query_budget=64
        if ans.status==Status.UNKNOWN:
            from solve_two_cegis import solve_two
            ans=solve_two(rows,seconds=120)
        if ans.status==Status.SAT:
            verify_report(rows,ans.tree,2,first_round=5)
        data['branches'][str(outcome)]={'status':ans.status.value,'states':len(rows),'tree':ans.tree,'reason':ans.reason}
        save(cp,data)
        if ans.status==Status.UNSAT:return ans
        if time.perf_counter()-last>10:
            print(json.dumps({'rep':code,'q4':q4,'solved_q4_branches':sum(v['status']=='SAT' for v in data['branches'].values()),
                              'reachable_q4_branches':len(parts),'unknown':sum(v['status']=='UNKNOWN' for v in data['branches'].values())}),flush=True)
            last=time.perf_counter()
    if len(data['branches'])!=len(parts) or any(v['status']!='SAT' for v in data['branches'].values()):
        return Result(Status.UNKNOWN,reason='bounded q5 search')
    tree={'query':list(q4),'children':{c:explicit_q6_witnesses(parts[int(c)],v['tree']) for c,v in data['branches'].items()}}
    return Result(Status.SAT,tree)


def worker(orbit,root_seconds=600):
    code=orbit['representative'];folder=OUT/'orbits'/str(code);cert=folder/'certificate.json.gz';start=time.perf_counter()
    if cert.exists():
        data=read(cert);report=verify_certificate(data)
        return {'representative':code,'status':'SAT','replay':report,'q4_query':data['tree']['query'],'resumed':True}
    states=generate(code)
    if code==30:
        data=read(Path('../../analysis/code30_continuation_20261007/q3_30_certificate.json'));save(cert,data)
        return {'representative':code,'status':'SAT','replay':verify_certificate(data),'q4_query':data['tree']['query']}
    seeds=q5_seeds();generator=Q4Generator(states);attempts=[];queue=[]
    candidates=([KNOWN[code]] if code in KNOWN else [])+Q4_SEEDS
    for q in dict.fromkeys(candidates):
        bad,cnt=generator.engine.evaluate(q,early=False)
        attempts.append({'query':q,'necessary_bad':len(bad),'max_bucket':int(cnt.max())})
        if not bad:queue.append((int(cnt.max()),q))
    tested={tuple(a['query']) for a in attempts}
    while time.perf_counter()-start<root_seconds:
        if not queue:
            models=generator.models(16)
            if not models:break
            ranked=[]
            for q in models:
                if q in tested:generator.block(q);continue
                cnt=np.bincount(codes(states,q),minlength=729).astype(np.int64)
                ranked.append(((int(cnt.max()),int(cnt@cnt)),q))
            for score,q in sorted(ranked):
                bad,cnt=generator.engine.evaluate(q,early=True);tested.add(q)
                attempts.append({'query':q,'necessary_bad':len(bad),'max_bucket':int(cnt.max())})
                if not bad:queue.append((int(cnt.max()),q))
            generator.feedback()
            if not queue:continue
        queue.sort();_,q4=queue.pop(0)
        print(json.dumps({'rep':code,'candidate_q4':q4,'max_bucket':int(np.bincount(codes(states,q4),minlength=729).max())}),flush=True)
        ans=try_q4(code,states,q4,seeds)
        attempts.append({'query':q4,'fixed_q4_status':ans.status.value,'reason':ans.reason})
        if ans.status==Status.SAT:
            data={'q3_code':code,'depth':3,'state_hash':fingerprint(states),'tree':ans.tree,'transport':None}
            report=verify_certificate(data);save(cert,data)
            result={'representative':code,'status':'SAT','q4_query':list(q4),'replay':report,'seconds':time.perf_counter()-start}
            save(folder/'search_summary.json',result|{'attempts':attempts});return result
        generator.block(q4)
    result={'representative':code,'status':'UNKNOWN','states':len(states),'seconds':time.perf_counter()-start,
            'attempts':attempts,'proved_failure_cores':len(generator.engine.failure_cache.entries)}
    save(folder/'search_summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--root-seconds',type=float,default=600)
    p.add_argument('--representatives',nargs='*',type=int);args=p.parse_args();data=manifest()
    todo=[o for o in data['orbits'] if not args.representatives or o['representative'] in args.representatives]
    progress_path=OUT/'progress.json';progress=read(progress_path) if progress_path.exists() else {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(worker,o,args.root_seconds):o for o in todo}
        for f in as_completed(futures):
            try:result=f.result()
            except Exception as e:result={'representative':futures[f]['representative'],'status':'ERROR','error':repr(e)}
            progress[str(result['representative'])]=result;save(progress_path,progress)
            solved=[o for o in data['orbits'] if progress.get(str(o['representative']),{}).get('status')=='SAT']
            print(json.dumps({'orbit_done':result,'certified_orbits':len(solved),
                              'certified_q3_branches':sum(len(o['q3_codes']) for o in solved),
                              'certified_states':sum(o['states']*len(o['q3_codes']) for o in solved)},ensure_ascii=False),flush=True)
    print(json.dumps({'all_orbits_status':'SAT' if all(progress.get(str(o['representative']),{}).get('status')=='SAT' for o in data['orbits']) else 'UNKNOWN',
                      'orbit_count':len(data['orbits'])}),flush=True)

if __name__=='__main__':main()
