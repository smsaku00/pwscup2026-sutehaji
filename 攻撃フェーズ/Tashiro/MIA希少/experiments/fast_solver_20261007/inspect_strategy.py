"""Read-only navigator for the certified strategy; never writes submissions.

Codes supplied here are counts for NORMALIZED questions. Mean values are
decoded against the universal exact table, then checked for reachability.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
from verify_all_orbits import read,generate,move_tree,normalize,response,Q3,IDS,CAPS

TEAMS=(6,10,11,16,17,25)
NEGATIVE_IDS=(126,136,271,273,317,440,480,625,637,643,668,814,841)


def decode(value,encoding):
    values={int(c):Fraction(v,encoding['mean_denominator']) for c,v in encoding['integer_values'].items()}
    ranked=sorted((abs(Fraction(str(value))-v),c) for c,v in values.items())
    if ranked[0][0]>=Fraction(encoding['minimum_gap_exact'])/2:raise ValueError('score too far from exact alphabet')
    if ranked[0][0]==ranked[1][0]:raise ValueError('ambiguous rounded score')
    return ranked[0][1]


def navigate(root,history):
    if len(history)>4:raise ValueError('at most four additional responses')
    encoding=read(root/'universal_score_encoding.json')
    if not history:node={'query':list(Q3)};rows=None
    else:
        manifest=read(root/'manifest.json');c=history[0]
        orbit=next((o for o in manifest['orbits'] if c in o['q3_codes']),None)
        if orbit is None:raise ValueError('unreachable q3 response')
        cert=read(root/'orbits'/str(orbit['representative'])/'certificate.json.gz')
        mapping=orbit['mappings'][str(c)]
        node=normalize(move_tree(cert['tree'],mapping['candidate_old_to_new'],mapping['team_old_to_new']))
        rows=generate(c)
        for c in history[1:]:
            if node.get('leaf'):raise ValueError('response supplied after singleton leaf')
            rows=[row for row in rows if response(row,node['query'])==c]
            if not rows:raise ValueError('unreachable response')
            if node.get('terminal'):
                if len(rows)!=1:raise ValueError('terminal collision')
                node={'leaf':True}
            else:
                if str(c) not in node['children']:raise ValueError('missing reachable child')
                node=node['children'][str(c)]
    if node.get('leaf'):
        if len(rows)!=1:raise ValueError('non-singleton leaf')
        return {'complete':True,'additional_responses':len(history),
                'members_by_team':{str(t):[i for b,i in enumerate(IDS) if m>>b&1] for t,m in zip(TEAMS,rows[0])}}
    q=node['query'];totals=encoding['total_high']
    return {'complete':False,'round':3+len(history),'remaining_states':None if rows is None else len(rows),
            'team_order':TEAMS,'query_masks':q,'total_high':totals,
            'high_confidence':0.4,'low_confidence':0.1,
            'selected_candidate_ids':{str(t):[i for b,i in enumerate(IDS) if m>>b&1] for t,m in zip(TEAMS,q)},
            'known_negative_padding_ids':{str(t):list(NEGATIVE_IDS[:totals[j]-m.bit_count()]) for j,(t,m) in enumerate(zip(TEAMS,q))},
            'minimum_mean_gap_exact':encoding['minimum_gap_exact']}


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');g=p.add_mutually_exclusive_group()
    g.add_argument('--codes',nargs='*',type=int,default=[]);g.add_argument('--means',nargs='*')
    a=p.parse_args();root=Path(a.directory)
    history=[decode(v,read(root/'universal_score_encoding.json')) for v in a.means] if a.means is not None else a.codes
    print(json.dumps(navigate(root,history),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
