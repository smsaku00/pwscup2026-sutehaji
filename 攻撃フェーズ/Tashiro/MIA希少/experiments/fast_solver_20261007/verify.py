"""Independent pure-integer witness checker; never imports solver internals."""
import argparse
import json
from pathlib import Path


def response(row, query):
    if len(query) != 6 or any(not 0 <= int(q) < 2048 for q in query):
        raise ValueError('invalid query')
    return sum((int(m)&int(q)).bit_count()*3**t
               for t,(m,q) in enumerate(zip(row,query)))


def verify(states, tree, max_depth):
    rows = [tuple(map(int,row)) for row in states]
    if len(rows) != len(set(rows)): raise ValueError('duplicate states')
    def rec(current, node, remaining):
        if node.get('leaf'):
            assert len(current) <= 1, 'non-singleton leaf'
            return len(current)
        assert remaining > 0, 'depth exceeded'
        groups = {}
        for row in current: groups.setdefault(response(row,node['query']),[]).append(row)
        if node.get('terminal'):
            assert all(len(v)==1 for v in groups.values()), 'collision at terminal'
            return len(current)
        assert set(map(int,node['children'])) == set(groups), 'missing/extra reachable child'
        return sum(rec(group,node['children'][str(code)],remaining-1)
                   for code,group in groups.items())
    count = rec(rows,tree,max_depth)
    assert count == len(rows)
    return count


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('certificate')
    args=parser.parse_args(); data=json.loads(Path(args.certificate).read_text())
    from model import generate, Q4_30
    rows=generate(data['q3_code'])
    if data.get('q4_query') is not None:
        rows=[r for r in rows if response(r,data['q4_query'])==data['q4_outcome']]
    print(json.dumps({'verified_states':verify(rows,data['tree'],data['depth']),
                      'q3_code':data['q3_code']},ensure_ascii=False))

if __name__=='__main__': main()
