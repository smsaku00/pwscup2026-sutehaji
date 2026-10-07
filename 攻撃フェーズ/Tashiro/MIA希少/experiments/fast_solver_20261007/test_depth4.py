"""Regression checks for the new full-scope proof and decoder boundary."""
from fractions import Fraction
from itertools import combinations,product
import unittest
import numpy as np
from solver import Solver,Status
from solve_two_cegis import solve_two
from solve_all_orbits import legacy_branch
from verify import verify
from verify_all_orbits import normalize,response,CAPS,FULL
from verify_score_encoding import exact_roc
from inspect_strategy import decode


class Tests(unittest.TestCase):
    def test_failed_proposed_q5_is_not_two_query_unsat(self):
        # Pair triangle needs two queries, although the proposed zero q5 fails.
        rows=np.array([[8,m,16,96,128,768] for m in (3,5,6)],dtype=np.uint16)
        e=Solver()
        self.assertEqual(legacy_branch(rows,[0]*6,e,float('inf')).status,Status.UNSAT)
        ans=solve_two(rows,seconds=5)
        self.assertEqual(ans.status,Status.SAT);verify(rows,ans.tree,2)
        self.assertEqual(solve_two(rows,seconds=0).status,Status.UNKNOWN)

    def test_depth_two_impossible_scalar_k4(self):
        rows=np.array([[16,(1<<a)|(1<<b),32,192,256,1536] for a,b in combinations(range(4),2)],dtype=np.uint16)
        self.assertEqual(solve_two(rows,seconds=5).status,Status.UNSAT)

    def test_normalization_preserves_partition_and_labels(self):
        rows=[[1,6,8,48,64,384],[2,5,16,40,128,576]]
        q=[FULL,1023,1951,1787,987,1943]
        old={str(response(r,q)):{'leaf':True} for r in rows}
        x=normalize({'query':q,'children':old})
        self.assertTrue(all(m.bit_count()<=5 for m in x['query']))
        for row in rows:self.assertIn(str(response(row,x['query'])),x['children'])
        self.assertEqual(len(x['children']),len(old))

    def test_universal_decoder_and_precision(self):
        totals=[6,7,7,11,10,12];den=216216000
        values={sum(h*3**t for t,h in enumerate(row)):sum((exact_roc(n,totals[t],h) for t,(n,h) in enumerate(zip(CAPS,row))),Fraction())/6
                for row in product(*(range(n+1) for n in CAPS))}
        self.assertEqual(len(set(values.values())),216)
        ordered=sorted(values.values());self.assertEqual(min(b-a for a,b in zip(ordered,ordered[1:])),Fraction(1,54000))
        data={'integer_values':{str(c):int(v*den) for c,v in values.items()},'mean_denominator':den,'minimum_gap_exact':'1/54000'}
        for c,v in values.items():
            self.assertEqual(decode(v,data),c)
            self.assertEqual(decode(f'{float(v):.5f}',data),c)
        with self.assertRaises(ValueError):decode(-1,data)
        with self.assertRaises(ValueError):decode((ordered[0]+ordered[1])/2,data)

if __name__=='__main__':unittest.main(verbosity=2)
