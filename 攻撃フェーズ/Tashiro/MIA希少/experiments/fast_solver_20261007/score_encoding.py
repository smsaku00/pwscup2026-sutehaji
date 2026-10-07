"""Universal legal mean-TPR encoding after complementing queries to <=5 bits.

Exact integer arithmetic proves the entire 216-response alphabet is distinct.
A zero query is padded by min(total_high,13); its only response is zero.
"""
from itertools import combinations,product
from fractions import Fraction
from math import lcm
import json
from pathlib import Path
import numpy as np

CAPS=(1,2,1,2,1,2)
SCALE=lcm(*range(4,14))
RESPONSES=list(product(*(range(n+1) for n in CAPS)))
CODES=[sum(h*3**t for t,h in enumerate(row)) for row in RESPONSES]


def find_encoding():
    best=None
    for ones in combinations(range(6,15),3):
        for twos in combinations(range(6,15),3):
            totals=(ones[0],twos[0],ones[1],twos[1],ones[2],twos[2])
            table=[]
            for t,n in enumerate(CAPS):
                table.append([0]+[(24-n)*h*SCALE//(n*(totals[t]-h)) for h in range(1,n+1)])
            values=np.array([sum(table[t][h] for t,h in enumerate(row)) for row in RESPONSES],dtype=np.int64)
            gap=int(np.diff(np.sort(values)).min())
            if gap>0 and (best is None or gap>best['integer_gap']):
                best={'total_high':totals,'integer_gap':gap,'integer_values':dict(zip(map(str,CODES),map(int,values)))}
    if best is None:raise ValueError('no universal encoding')
    best['mean_denominator']=600*SCALE
    best['minimum_gap_exact']=str(Fraction(best['integer_gap'],best['mean_denominator']))
    best['minimum_gap']=best['integer_gap']/best['mean_denominator']
    best['responses']=216;best['rare_total']=24;best['mean_teams']=6;best['caps']=CAPS
    best['legal_padding_rule']='For normalized nonzero query: total_high[t]-popcount(query[t]); for zero query: min(total_high[t],13).'
    best['query_normalization']='Complement any 11-bit query having more than 5 selected candidates; relabel its count h to caps[t]-h.'
    return best


def main():
    out=Path('../../analysis/full_depth4_20261007');out.mkdir(parents=True,exist_ok=True)
    result=find_encoding();(out/'universal_score_encoding.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='integer_values'},indent=2),flush=True)

if __name__=='__main__':main()
