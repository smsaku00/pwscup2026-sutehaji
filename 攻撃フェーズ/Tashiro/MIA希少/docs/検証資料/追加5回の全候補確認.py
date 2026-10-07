"""追加5回の保証を、単独Teamの全候補と共通の平均点復号から検査する。

標準ライブラリだけ。選択側は正解を受け取らない。1人の全11状態、
2人の全55状態を独立にreplayし、全経路の質問・最悪深さを記録する。
6Team全配置の保証は合成による証明であり、全2494800状態をこのコードで
replayしたという主張にはしない。4回案のchecker・探索コードはimportしない。
"""
import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import combinations, product
import json
from pathlib import Path

IDS=(47,76,90,115,169,264,304,383,462,495,675)
LETTERS='ABCDEFGHIJK'
FULL=(1<<11)-1
TEAMS=(6,10,11,16,17,25)
CAPS=(1,2,1,2,1,2)
HIGH=(6,7,7,11,10,12)


def names(mask):
    return ''.join(c for i,c in enumerate(LETTERS) if mask>>i&1)


@lru_cache(None)
def select(states):
    cap=states[0].bit_count();best=None
    # The complement has the same partition because cap is known. Fixing
    # the last bit to zero covers all possible partitions once or more.
    for q in range(1,1<<10):
        groups=tuple(tuple(m for m in states if (m&q).bit_count()==h) for h in range(cap+1))
        sizes=tuple(map(len,groups));key=(max(sizes),sum(s*s for s in sizes),q.bit_count(),q)
        if best is None or key<best[0]:best=key,q
    q=best[1]
    if q.bit_count()>5:q^=FULL
    groups=tuple(tuple(m for m in states if (m&q).bit_count()==h) for h in range(cap+1))
    if max(map(len,groups))>=len(states):raise ValueError('non-progressing question')
    return q,groups


def build(states):
    if len(states)==1:return {'member_mask':states[0]}
    q,groups=select(states)
    return {'query_mask':q,'query_letters':names(q),'before':len(states),
            'children':{str(h):build(g) for h,g in enumerate(groups) if g}}


def replay(tree,truth):
    # Direct integer responses, separate from selection and its partitions.
    node=tree;steps=[]
    while 'member_mask' not in node:
        q=node['query_mask']
        if not 0<=q<=FULL or q.bit_count()>5:raise ValueError('illegal normalized query')
        h=(truth&q).bit_count()
        if str(h) not in node['children']:raise ValueError('missing reachable answer')
        child=node['children'][str(h)]
        steps.append({'selected':names(q),'query_mask':q,'response':h,
                      'before':node['before'],'after':child.get('before',1)})
        node=child
        if len(steps)>5:raise ValueError('more than five questions')
    if node['member_mask']!=truth:raise ValueError('wrong singleton')
    return steps


def roc_score(cap,total,h):
    # Independent ROC interpolation, rather than importing the 4-query table.
    x=Fraction(total-h,24-cap);y=Fraction(h,cap)
    if x<=Fraction(1,100):raise ValueError('target FPR not strictly in first segment')
    return y*Fraction(1,100)/x


def compute():
    singles={};policies={}
    for cap,limit in ((1,4),(2,5)):
        states=tuple(sum(1<<i for i in c) for c in combinations(range(11),cap))
        tree=build(states);paths=[replay(tree,m) for m in states]
        depths=Counter(map(len,paths));maxima=[len(states)]
        for depth in range(1,6):
            maxima.append(max(p[depth-1]['after'] if len(p)>=depth else 1 for p in paths))
        if max(depths)!=limit or sum(depths.values())!=len(states):raise ValueError('depth or coverage mismatch')
        singles[str(cap)]={'states_replayed':len(states),'collisions':0,'max_depth':max(depths),
                          'depth_distribution':dict(sorted(depths.items())),
                          'maximum_candidates_after_0_to_5_questions':maxima,
                          'trace_AK':paths[states.index((1<<0)|(1<<10))] if cap==2 else None}
        policies[str(cap)]=tree
    # Four questions on a single 2-member Team would leave at least 28 states
    # after the first answer; three remaining ternary answers have only 27.
    bound_by_size=[max((11-s)*(10-s)//2,s*(11-s),s*(s-1)//2) for s in range(6)]
    if min(bound_by_size)!=28 or 28<=3**3:raise ValueError('lower bound mismatch')
    values={}
    for hs in product(*(range(n+1) for n in CAPS)):
        c=sum(h*3**t for t,h in enumerate(hs))
        values[c]=sum((roc_score(n,HIGH[t],h) for t,(n,h) in enumerate(zip(CAPS,hs))),Fraction())/6
    ordered=sorted(values.values());gap=min(y-x for x,y in zip(ordered,ordered[1:]))
    if len(set(ordered))!=216 or gap!=Fraction(1,54000):raise ValueError('mean decoder collision or margin mismatch')
    for t,total in enumerate(HIGH):
        if any(not 1<=total-size<=13 for size in range(6)):raise ValueError('illegal padding')
    return {'scope':'単独Team全候補のreplayと216平均点の全列挙。6Team全配置はこの2つを合成して保証する。',
            'team_order':TEAMS,'candidate_ids':IDS,'caps':CAPS,'total_high':HIGH,
            'single_team':singles,'single_pair_first_question_lower_bound_by_size_0_to_5':bound_by_size,
            'single_pair_four_questions_impossible':True,'mean_response_patterns':len(values),
            'mean_collisions':0,'mean_minimum_gap_exact':str(gap),
            'compositional_guarantee_max_additional_queries':5,'composition_covers_disjoint_states':2494800,
            'policies':policies}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();result=compute()
    if a.output:Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='policies'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
