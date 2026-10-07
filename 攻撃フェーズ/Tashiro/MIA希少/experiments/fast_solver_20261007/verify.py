"""Independent pure-integer replay checker; no solver/model import."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import struct

CAPS = (1,2,1,2,1,2)
Q3 = (590,436,1547,440,1095,433)


def response(row, query):
    if len(query) != 6 or any(type(q) != int or not 0 <= q < 2048 for q in query):
        raise ValueError('invalid query')
    return sum((int(m)&q).bit_count()*3**t for t,(m,q) in enumerate(zip(row,query)))


def generate_states(code):
    # Independently rebuild the combinatorial parent, with Python integer
    # loops; do not trust a solver-produced row list or count.
    opts = []
    for t,cap in enumerate(CAPS):
        target = (code//3**t)%3
        opts.append([sum(1<<i for i in c) for c in combinations(range(11),cap)
                     if (sum(1<<i for i in c)&Q3[t]).bit_count() == target])
    rows = []
    def rec(t,used,row):
        if t == 6: rows.append(tuple(row));return
        for m in opts[t]:
            if not m&used: rec(t+1,used|m,row+[m])
    rec(0,0,[])
    return rows


def state_hash(rows):
    return hashlib.sha256(b''.join(struct.pack('<6H',*row) for row in sorted(rows))).hexdigest()


def verify_report(states, tree, max_depth, first_round=4):
    rows = [tuple(map(int,row)) for row in states]
    if len(rows) != len(set(rows)): raise ValueError('duplicate states')
    stats = {'verified_states':0,'collisions':0,'max_depth':0,
             'q4_branches':0,'q4_max_bucket':0,'q5_cells':0,'q5_max_cell':0,'q6_queries':0}
    def rec(current,node,remaining,depth):
        stats['max_depth'] = max(stats['max_depth'],depth)
        if node.get('leaf'):
            if len(current) > 1: raise ValueError('non-singleton leaf')
            return len(current)
        if remaining <= 0: raise ValueError('depth exceeded')
        groups = {}
        for row in current: groups.setdefault(response(row,node['query']),[]).append(row)
        stats['max_depth'] = max(stats['max_depth'],depth+1)
        if first_round+depth == 4:
            stats['q4_branches'] = len(groups)
            stats['q4_max_bucket'] = max(map(len,groups.values()),default=0)
        if first_round+depth == 5:
            stats['q5_cells'] += len(groups)
            stats['q5_max_cell'] = max(stats['q5_max_cell'],max(map(len,groups.values()),default=0))
        if node.get('terminal'):
            if any(len(v) != 1 for v in groups.values()): raise ValueError('collision at terminal')
            if first_round+depth == 6: stats['q6_queries'] += 1
            return len(current)
        children = node['children']
        if set(children) != {str(code) for code in groups}: raise ValueError('missing/extra reachable child')
        return sum(rec(group,children[str(code)],remaining-1,depth+1) for code,group in groups.items())
    count = rec(rows,tree,max_depth,0)
    if count != len(rows): raise ValueError('incomplete replay')
    stats['verified_states'] = count
    return stats


def verify(states,tree,max_depth):
    return verify_report(states,tree,max_depth)['verified_states']


def verify_certificate(data):
    rows = generate_states(data['q3_code'])
    if data.get('q4_query') is not None:
        rows = [r for r in rows if response(r,data['q4_query']) == data['q4_outcome']]
    if data.get('state_hash') and data['state_hash'] != state_hash(rows):
        raise ValueError('state fingerprint mismatch')
    result = verify_report(rows,data['tree'],data['depth'],5 if data.get('q4_query') is not None else 4)
    result['q3_code'] = data['q3_code'];result['state_hash'] = state_hash(rows)
    return result


def verify_transport(source,data):
    mapping = data['transport'];pm = mapping['candidate_old_to_new'];tm = mapping['team_old_to_new']
    if sorted(pm) != list(range(11)) or sorted(tm) != list(range(6)):
        raise ValueError('invalid transport permutation')
    if any(CAPS[t] != CAPS[tm[t]] for t in range(6)):
        raise ValueError('transport capacity mismatch')
    if any(((Q3[t]>>i)&1) != ((Q3[tm[t]]>>pm[i])&1) for t in range(6) for i in range(11)):
        raise ValueError('transport q3 incidence mismatch')
    def mask(m): return sum(1<<pm[i] for i in range(11) if m>>i&1)
    def row(x):
        out = [0]*6
        for t,m in enumerate(x): out[tm[t]] = mask(m)
        return tuple(out)
    def code(c): return sum(((c//3**t)%3)*3**tm[t] for t in range(6))
    if code(source['q3_code']) != data['q3_code']: raise ValueError('transport q3 code mismatch')
    original = generate_states(source['q3_code']);moved = [row(x) for x in original]
    if set(moved) != set(generate_states(data['q3_code'])): raise ValueError('transport state-set mismatch')
    checked = [0,0,0]
    def rec(a,b,x,y,depth):
        if x.get('leaf'):
            if not y.get('leaf'): raise ValueError('transport leaf mismatch')
            return
        expected = [0]*6
        for t,m in enumerate(x['query']): expected[tm[t]] = mask(m)
        if expected != y['query']: raise ValueError('transport query mask mismatch')
        groups = {}
        for u,v in zip(a,b):
            ca,cb = response(u,x['query']),response(v,y['query'])
            if code(ca) != cb: raise ValueError('transport partition mismatch')
            groups.setdefault(ca,([],[]));groups[ca][0].append(u);groups[ca][1].append(v)
        checked[depth] += len(a)
        if x.get('terminal'):
            if not y.get('terminal'): raise ValueError('transport terminal mismatch')
            return
        if set(y['children']) != {str(code(int(c))) for c in x['children']}:
            raise ValueError('transport child-key mismatch')
        for c,(aa,bb) in groups.items():rec(aa,bb,x['children'][str(c)],y['children'][str(code(c))],depth+1)
    rec(original,moved,source['tree'],data['tree'],0)
    return {'source_q3_code':source['q3_code'],'target_q3_code':data['q3_code'],
            'state_set_equal':True,'q3_incidence_equal':True,'partition_replayed_states_q4_q5_q6':checked}


def main():
    p = argparse.ArgumentParser();p.add_argument('certificate');p.add_argument('--output');p.add_argument('--source')
    args = p.parse_args();data = json.loads(Path(args.certificate).read_text())
    result = verify_certificate(data)
    if args.source: result['transport_replay'] = verify_transport(json.loads(Path(args.source).read_text()),data)
    if args.output: Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
