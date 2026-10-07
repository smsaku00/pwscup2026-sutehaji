"""Independent integer checker for depth-2 UNSAT cores and grid coverage.

No solver/model/reference import. All first and final scalar questions are
explicitly enumerated. A finite slice proof never proves the unrestricted q4
problem UNSAT.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import json
import gzip
from pathlib import Path
import time

CAPS = (1,2,1,2,1,2)
Q3 = (590,436,1547,440,1095,433)
FULL = 2047

@lru_cache(maxsize=None)
def one_question(members):
    if len(members) <= 1: return True
    if len(members) > members[0].bit_count()+1: return False
    return any(len({(m&q).bit_count() for m in members}) == len(members) for q in range(1024))

@lru_cache(maxsize=None)
def two_question_domain(members):
    bits = 0
    for q in range(1024):
        cells = defaultdict(list)
        for m in members: cells[(m&q).bit_count()].append(m)
        if all(one_question(tuple(sorted(v))) for v in cells.values()): bits |= 1 << q
    return bits


def obstruction(rows, team):
    fibers = defaultdict(set)
    for row in rows:
        fibers[tuple(row[t] for t in range(6) if t != team)].add(row[team])
    allowed = (1<<1024)-1
    for members in fibers.values():
        allowed &= two_question_domain(tuple(sorted(members)))
        if allowed == 0: return True
    return False


def check_rows(rows, q3_code):
    if len(rows) != len(set(rows)): raise ValueError('duplicate core states')
    for row in rows:
        if len(row) != 6: raise ValueError('invalid state width')
        used = 0
        for m,cap in zip(row,CAPS):
            if not isinstance(m,int) or not 0 <= m <= FULL or m.bit_count() != cap or used&m:
                raise ValueError('illegal state')
            used |= m
        if sum((m&q).bit_count()*3**t for t,(m,q) in enumerate(zip(row,Q3))) != q3_code:
            raise ValueError('core outside parent q3 branch')


def uniform_masks(rows,t):
    members = set(row[t] for row in rows)
    return [q for q in range(1024) if len({(m&q).bit_count() for m in members}) == 1]


def check_file(data):
    start = time.perf_counter();cores = []
    from verify import generate_states,state_hash
    parent = generate_states(data['q3_code'])
    if data['states'] != len(parent) or data['state_hash'] != state_hash(parent):
        raise ValueError('parent state count/hash mismatch')
    for i,e in enumerate(data['failure_cores']):
        if e['depth'] != 2: raise ValueError('unsupported core depth')
        rows = list(map(tuple,e['states']));check_rows(rows,data['q3_code'])
        # The indicated team is only a hint; its impossibility is recomputed.
        t = int(e['reason'].rsplit(' ',1)[-1])
        if not 0 <= t < 6 or not obstruction(rows,t): raise ValueError(f'invalid UNSAT core {i}')
        cores.append((rows,{}))
    reports = []
    for grid in data.get('results',[]):
        fixed = grid['fixed_query'];a,b = grid['varying_team_indices']
        if len(fixed) != 6 or any(type(q) != int or not 0 <= q <= FULL for q in fixed):
            raise ValueError('invalid fixed query')
        if a == b or not 0 <= a < 6 or not 0 <= b < 6: raise ValueError('invalid varying teams')
        count = grid.get('core_count_at_end',len(cores))
        if not 0 <= count <= len(cores): raise ValueError('invalid core prefix')
        covered = [0]*1024
        for rows,uniform in cores[:count]:
            if not all(len({(row[t]&fixed[t]).bit_count() for row in rows}) == 1
                       for t in range(6) if t not in (a,b)): continue
            for t in (a,b):
                if t not in uniform: uniform[t] = uniform_masks(rows,t)
            bits = sum(1<<q for q in uniform[b])
            for q in uniform[a]: covered[q] |= bits
        excluded = sum(x.bit_count() for x in covered)
        if grid['restricted_status'] == 'UNSAT' and excluded != 1024**2:
            raise ValueError('incomplete restricted grid proof')
        if grid['overall_status'] != 'UNKNOWN': raise ValueError('invalid unrestricted conclusion')
        reports.append({'fixed_query':fixed,'varying_team_indices':[a,b],
                        'independently_excluded':excluded,'total_candidates':1024**2,
                        'restricted_status':grid['restricted_status']})
    family_status = 'UNKNOWN'
    if data.get('third_all'):
        t = data['third_team_index'];union = 0
        first = data['results'][0];varying = first['varying_team_indices']
        if t in varying: raise ValueError('third team overlaps varying teams')
        for grid in data['results']:
            if grid['varying_team_indices'] != varying or any(
                grid['fixed_query'][u] != first['fixed_query'][u] for u in range(6) if u not in varying+[t]):
                raise ValueError('inconsistent fixed query family')
        for row in parent: union |= row[t]
        active = [i for i in range(11) if union>>i&1][:-1]
        expected = {sum(1<<i for j,i in enumerate(active) if bits>>j&1) for bits in range(1<<len(active))}
        actual = {grid['fixed_query'][t] for grid in data['results']}
        if actual != expected or len(actual) != len(data['results']):
            raise ValueError('third-team representative enumeration incomplete')
        if all(grid['restricted_status'] == 'UNSAT' for grid in data['results']): family_status = 'UNSAT'
    return {'valid_cores':len(cores),'parent_states':len(parent),'parent_state_hash':state_hash(parent),
            'grid_replays':reports,'restricted_family_status':family_status,'overall_status':'UNKNOWN',
            'seconds':time.perf_counter()-start}


def main():
    p = argparse.ArgumentParser();p.add_argument('certificate');p.add_argument('--output')
    args = p.parse_args()
    with (gzip.open(args.certificate,'rt') if args.certificate.endswith('.gz') else open(args.certificate)) as f:
        data = json.load(f)
    result = check_file(data)
    if args.output: Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
