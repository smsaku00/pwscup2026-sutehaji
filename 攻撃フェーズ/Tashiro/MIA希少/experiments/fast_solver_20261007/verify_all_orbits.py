"""Independent full-scope audit of a 4-query strategy after the first two rounds.

Standard library only. Does not import model, solver, search, or score search.
Each of all 216 capacity-legal q3 answers is independently regenerated.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations,product
import gzip
import hashlib
import json
from pathlib import Path
import struct

CAPS=(1,2,1,2,1,2)
IDS=(47,76,90,115,169,264,304,383,462,495,675)
Q3=(590,436,1547,440,1095,433)
FULL=2047


def read(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return json.load(f)


def generate(code):
    options=[]
    for t,n in enumerate(CAPS):
        expected=(code//3**t)%3
        options.append([sum(1<<i for i in c) for c in combinations(range(11),n)
                        if (sum(1<<i for i in c)&Q3[t]).bit_count()==expected])
    out=[]
    def dfs(t,used,row):
        if t==6:out.append(tuple(row));return
        for m in options[t]:
            if not m&used:dfs(t+1,used|m,row+[m])
    dfs(0,0,[]);return out


def state_hash(rows):
    h=hashlib.sha256()
    for row in sorted(rows):h.update(struct.pack('<6H',*row))
    return h.hexdigest()


def response(row,q):
    if len(q)!=6 or any(type(m)!=int or not 0<=m<=FULL for m in q):raise ValueError('illegal query')
    return sum((m&x).bit_count()*3**t for t,(m,x) in enumerate(zip(row,q)))


def transform(code,tm):return sum(((code//3**t)%3)*3**tm[t] for t in range(6))


def mask(m,pm):return sum(1<<pm[i] for i in range(11) if m>>i&1)


def move_tree(node,pm,tm):
    if node.get('leaf'):return {'leaf':True}
    q=[0]*6
    for t,m in enumerate(node['query']):q[tm[t]]=mask(m,pm)
    if node.get('terminal'):return {'query':q,'terminal':True}
    return {'query':q,'children':{str(transform(int(c),tm)):move_tree(ch,pm,tm) for c,ch in node['children'].items()}}


def normalize(node):
    if node.get('leaf'):return {'leaf':True}
    old=node['query'];flip=[m.bit_count()>5 for m in old]
    q=[FULL^m if f else m for m,f in zip(old,flip)]
    if node.get('terminal'):return {'query':q,'terminal':True}
    children={}
    for c,ch in node['children'].items():
        hs=[(int(c)//3**t)%3 for t in range(6)]
        nc=sum((CAPS[t]-h if flip[t] else h)*3**t for t,h in enumerate(hs))
        if str(nc) in children:raise ValueError('normalization child collision')
        children[str(nc)]=normalize(ch)
    return {'query':q,'children':children}


def audit_tree(rows,node,totals,remaining=3,round_number=4):
    stats={'states':0,'collisions':0,'max_additional_depth':1,'q4_branches':0,'q4_max_bucket':0,
           'q5_cells':0,'q5_max_cell':0,'q6_nodes':0,'encoded_nodes':0,'all_normalized':True}
    def rec(current,x,left,r):
        if x.get('leaf'):
            if len(current)>1:raise ValueError('non-singleton leaf')
            return len(current)
        if left<=0:raise ValueError('more than four additional queries')
        q=x['query'];groups=defaultdict(list)
        for row in current:groups[response(row,q)].append(row)
        scores={}
        for t,m in enumerate(q):
            size=m.bit_count()
            if size>5 or not 1<=totals[t]-size<=13:raise ValueError('illegal normalized query/padding')
        for c in groups:
            hs=[(c//3**t)%3 for t in range(6)]
            value=sum((Fraction((24-n)*h,100*n*(totals[t]-h)) for t,(n,h) in enumerate(zip(CAPS,hs))),Fraction())/6
            if value in scores:raise ValueError('mean-score collision')
            scores[value]=c
        stats['encoded_nodes']+=1;stats['max_additional_depth']=max(stats['max_additional_depth'],r-2)
        if r==4:
            stats['q4_branches']=len(groups);stats['q4_max_bucket']=max(map(len,groups.values()))
        if r==5:
            stats['q5_cells']+=len(groups);stats['q5_max_cell']=max(stats['q5_max_cell'],max(map(len,groups.values())))
        if r==6:stats['q6_nodes']+=1
        if x.get('terminal'):
            if any(len(g)!=1 for g in groups.values()):raise ValueError('terminal collision')
            return len(current)
        if set(x['children'])!={str(c) for c in groups}:raise ValueError('missing/extra reachable outcome')
        return sum(rec(g,x['children'][str(c)],left-1,r+1) for c,g in groups.items())
    count=rec(rows,node,remaining,round_number)
    if count!=len(rows):raise ValueError('incomplete replay')
    stats['states']=count;return stats


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',default='independent_full_audit.json')
    a=p.parse_args();root=Path(a.directory);manifest=read(root/'manifest.json');encoding=read(root/'universal_score_encoding.json')
    if manifest['q3_query']!=list(Q3) or manifest['caps']!=list(CAPS) or manifest['candidate_ids']!=list(IDS):raise ValueError('wrong model')
    # Independently validate the whole count alphabet, not a supplied SAT flag.
    universal=[]
    for hs in product(*(range(n+1) for n in CAPS)):
        universal.append(sum((Fraction((24-n)*h,100*n*(encoding['total_high'][t]-h)) for t,(n,h) in enumerate(zip(CAPS,hs))),Fraction())/6)
    if len(set(universal))!=216:raise ValueError('universal decoder collision')
    margin=min(y-x for x,y in zip(sorted(universal),sorted(universal)[1:]))
    if margin!=Fraction(encoding['minimum_gap_exact']):raise ValueError('decoder margin mismatch')
    entries={};sources={};source_rows={}
    for orbit in manifest['orbits']:
        rep=orbit['representative'];cert=read(root/'orbits'/str(rep)/'certificate.json.gz')
        if cert['q3_code']!=rep or cert['depth']!=3:raise ValueError('wrong representative certificate')
        original=generate(rep)
        if state_hash(original)!=cert['state_hash']:raise ValueError('representative state hash mismatch')
        sources[rep]=cert
        source_rows[rep]=original
        for c in orbit['q3_codes']:
            if c in entries:raise ValueError('duplicate q3 branch')
            mapping=orbit['mappings'][str(c)];pm=mapping['candidate_old_to_new'];tm=mapping['team_old_to_new']
            if sorted(pm)!=list(range(11)) or sorted(tm)!=list(range(6)):raise ValueError('invalid permutation')
            if any(CAPS[t]!=CAPS[tm[t]] for t in range(6)):raise ValueError('capacity permutation mismatch')
            if any(((Q3[t]>>i)&1)!=((Q3[tm[t]]>>pm[i])&1) for t in range(6) for i in range(11)):
                raise ValueError('q3 incidence permutation mismatch')
            if transform(rep,tm)!=c:raise ValueError('q3 code permutation mismatch')
            entries[c]=(rep,pm,tm)
    reports=[];total=0;empty=[]
    for hs in product(*(range(n+1) for n in CAPS)):
        c=sum(h*3**t for t,h in enumerate(hs));rows=generate(c)
        if not rows:
            if c in entries:raise ValueError('certificate for unreachable branch')
            empty.append(c);continue
        if c not in entries:raise ValueError(f'uncovered q3 branch {c}')
        rep,pm,tm=entries[c]
        # Check the actual transported state SET as well as the incidence
        # proof. This equality does not use model.generate or solver code.
        moved=set()
        lut=[mask(m,pm) for m in range(FULL+1)]
        for row in source_rows[rep]:
            dest=[0]*6
            for t,m in enumerate(row):dest[tm[t]]=lut[m]
            moved.add(tuple(dest))
        if moved!=set(rows):raise ValueError('transported state set mismatch')
        tree=normalize(move_tree(sources[rep]['tree'],pm,tm))
        r=audit_tree(rows,tree,encoding['total_high']);r['q3_code']=c;r['representative']=rep;r['state_hash']=state_hash(rows)
        r['transported_state_set_equal']=True;r['max_q4_to_q6_depth']=r['max_additional_depth']-1
        # Each branch generation is exhaustive; different q3 outcomes are
        # disjoint by definition. Their sum must be the full 11!/16 universe.
        total+=len(rows);reports.append(r)
        if len(reports)%10==0:print(json.dumps({'q3_branches_verified':len(reports),'states_verified':total}),flush=True)
    if total!=2494800 or len(reports)!=209 or len(entries)!=209:raise ValueError('incomplete full universe')
    # Q3 itself uses five candidates in every column and the same encoding.
    if any(m.bit_count()!=5 for m in Q3):raise ValueError('q3 normalization assumption')
    result={'overall_status':'SAT','additional_queries':4,'total_states':total,'q3_query':list(Q3),
            'reachable_q3_branches':len(reports),'verified_orbits':len(manifest['orbits']),
            'q4_branches_total':sum(r['q4_branches'] for r in reports),
            'q4_max_bucket':max(r['q4_max_bucket'] for r in reports),
            'q5_cells_total':sum(r['q5_cells'] for r in reports),'q5_max_cell':max(r['q5_max_cell'] for r in reports),
            'q6_nodes_total':sum(r['q6_nodes'] for r in reports),'collisions':0,
            'max_additional_depth':max(r['max_additional_depth'] for r in reports),
            'max_q4_to_q6_depth':max(r['max_q4_to_q6_depth'] for r in reports),
            'transported_state_sets_equal':len(reports),
            'encoded_nodes_total':1+sum(r['encoded_nodes'] for r in reports),
            'mean_score_total_high':encoding['total_high'],'mean_score_minimum_gap_exact':str(margin),
            'reachable_score_collisions':0,'unreachable_q3_codes':empty,'branches':reports}
    (root/a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='branches'},indent=2),flush=True)

if __name__=='__main__':main()
