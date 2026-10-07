"""Independent ROC/fraction checker for the universal mean-score encoding.

No import of score_encoding, model or solver. Official scorer comparison is
an optional additional gate; its source SHA-256 is included in the report.
"""
import argparse
import ast
from fractions import Fraction
from itertools import product
import hashlib
import json
from pathlib import Path
import sys

CAPS=(1,2,1,2,1,2)


def exact_roc(n,high,h):
    points=[(Fraction(0),Fraction(0)),(Fraction(high-h,24-n),Fraction(h,n)),(Fraction(1),Fraction(1))]
    target=Fraction(1,100)
    for (x0,y0),(x1,y1) in zip(points,points[1:]):
        if x0<=target<=x1 and x1>x0:return y0+(target-x0)*(y1-y0)/(x1-x0)
    raise ValueError('ROC target outside range')


def check(data,official_src=None):
    totals=data['total_high'];values={};padding_checks=0
    for t,n in enumerate(CAPS):
        for size in range(6):
            pad=totals[t]-size
            if not 1<=pad<=13:raise ValueError('illegal padding')
            for h in range(min(n,size)+1):
                exact_roc(n,totals[t],h);padding_checks+=1
    for row in product(*(range(n+1) for n in CAPS)):
        code=sum(h*3**t for t,h in enumerate(row))
        value=sum((exact_roc(n,totals[t],h) for t,(n,h) in enumerate(zip(CAPS,row))),Fraction())/6
        if value!=Fraction(data['integer_values'][str(code)],data['mean_denominator']):raise ValueError('encoded table mismatch')
        values[code]=value
    ordered=sorted(values.values())
    if len(set(ordered))!=216:raise ValueError('mean-score collision')
    gap=min(b-a for a,b in zip(ordered,ordered[1:]))
    result={'alphabet_responses':216,'exact_collisions':0,'minimum_gap_exact':str(gap),
            'minimum_gap':float(gap),'legal_padding_checks':padding_checks,'total_high':totals}
    if official_src:
        src=Path(official_src).resolve();sys.path.insert(0,str(src))
        from pwscup2026.scoring.attack_score import score_mia
        import numpy as np
        # Execute the exact aggregation functions from the official source
        # without importing unrelated attack pipelines (SciPy etc.).
        aggregation=src/'pwscup2026/kit/codabench/scoring_io.py'
        parsed=ast.parse(aggregation.read_text())
        functions=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in ('_finite','_nanmean')]
        if {node.name for node in functions}!={'_finite','_nanmean'}:raise ValueError('official aggregation functions missing')
        environment={'np':np}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(aggregation),'exec'),environment)
        _nanmean=environment['_nanmean']
        zero=score_mia(np.full(24,0.1),np.zeros(24,dtype=np.int32),
                       {'scoring':{'low_fpr':0.01}},rare_mask=np.ones(24,dtype=bool))
        if np.isfinite(zero.breakdown['tpr_rare']):raise ValueError('zero-positive team must be excluded from rare mean')
        lookup={};maxerror=0.0;checked=0
        for t,n in enumerate(CAPS):
            for size in range(6):
                for h in range(min(n,size)+1):
                    confidence=np.full(24,0.1);confidence[:size]=0.4
                    confidence[11:11+totals[t]-size]=0.4
                    truth=np.zeros(24,dtype=np.int32);truth[:h]=1;truth[size:size+n-h]=1
                    r=score_mia(confidence,truth,{'scoring':{'low_fpr':0.01}},rare_mask=np.ones(24,dtype=bool))
                    got=r.breakdown['tpr_rare'];expected=float(exact_roc(n,totals[t],h))
                    maxerror=max(maxerror,abs(got-expected));checked+=1
                    if abs(got-expected)>1e-14:raise ValueError('official per-team mismatch')
                    lookup[t,h]=got
        for row in product(*(range(n+1) for n in CAPS)):
            code=sum(h*3**t for t,h in enumerate(row))
            got=_nanmean([lookup[t,h] for t,h in enumerate(row)]+[float('nan')]*16)
            expected=float(values[code]);maxerror=max(maxerror,abs(got-expected))
            if abs(got-expected)>1e-14:raise ValueError('official mean mismatch')
            decoded=[c for c,v in values.items() if abs(float(v)-got)<float(gap)/4]
            if decoded!=[code]:raise ValueError('mean-score decoding mismatch')
        result['official_per_team_cases']=checked;result['official_mean_cases']=216
        result['official_zero_positive_rare_is_nan']=True
        result['official_aggregation_mode']='exact _finite/_nanmean functions extracted from official source'
        result['official_max_absolute_error']=maxerror
        result['official_sources']={str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (src/'pwscup2026/scoring/attack_score.py',src/'pwscup2026/kit/codabench/scoring_io.py')}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('encoding');p.add_argument('--official-src');p.add_argument('--output')
    a=p.parse_args();data=json.loads(Path(a.encoding).read_text());result=check(data,a.official_src)
    if a.output:Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
