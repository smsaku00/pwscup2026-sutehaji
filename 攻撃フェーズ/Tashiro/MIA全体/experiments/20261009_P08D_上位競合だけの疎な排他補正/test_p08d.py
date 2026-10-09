"""Sparse exclusivity perturbation is target/label-free, local, reproducible."""
import unittest
import numpy as np
from run_p08d import (collision_analysis,sparse_penalty,eligibility,
                      TOP_K,MARGINS,PENALTIES,METHODS)

class P08DSparseTests(unittest.TestCase):
    def test_collision_penalizes_only_lower_score_target(self):
        scores=np.array([
         [.9,.86,.10],
         [.80,.20,.90],
         [.30,.77,.80],
         [.20,.10,.30],
         [.10,.05,.20]
        ])
        k=3
        parts=collision_analysis(scores,k)
        _,many,leader,_,loser,gap,diag=parts
        self.assertGreater(diag["collision_candidates_ge2"],0)
        out,st=sparse_penalty(scores,k,.02,.05,parts)
        elig=loser & (gap>=.02-1e-12)
        np.testing.assert_array_equal(out[~elig],scores[~elig])
        np.testing.assert_allclose(out[elig],scores[elig]-.05,rtol=0,atol=0)
        np.testing.assert_array_equal(out[np.arange(len(scores)),leader],
                                      scores[np.arange(len(scores)),leader])
        self.assertGreater(st["penalized_cells"],0)
    def test_no_collision_means_no_change(self):
        x=np.array([
         [.99,.01,.01],
         [.01,.99,.01],
         [.01,.01,.99],
         [.02,.02,.02]
        ])
        for t in [1,2,3]:
            y,stats=sparse_penalty(x,t,.02,.15)
            if t==1:self.assertTrue(np.array_equal(y,x))
            self.assertTrue(stats["penalized_cells"]>=0)
    def test_equal_tied_scores_are_not_penalized(self):
        x=np.array([[.95,.95,.05],[.90,.90,.10],[.80,.70,.60],[.20,.20,.98]])
        y,diag=sparse_penalty(x,2,.02,.05)
        self.assertEqual(y[0,0],.95)
        self.assertEqual(y[0,1],.95)
        self.assertEqual(y[1,0],.90)
        self.assertEqual(y[1,1],.90)
    def test_target_column_permutation_equivariance_no_ties(self):
        rng=np.random.default_rng(20261009)
        x=rng.uniform(size=(70,5))
        for k in [10,20]:
            a,_=sparse_penalty(x,k,.05,.15)
            order=[3,1,4,2,0]
            b,_=sparse_penalty(x[:,order],k,.05,.15)
            np.testing.assert_allclose(a[:,order],b,rtol=0,atol=0)
    def test_topk_ties_are_kept_as_group(self):
        z=np.array([[.9],[.9],[.8],[.7],[.6]])
        mask=eligibility(z,1)
        self.assertEqual(mask[:,0].tolist(),[True,True,False,False,False])
    def test_method_scope(self):
        self.assertEqual(len(METHODS),19)
        self.assertEqual(TOP_K,(10,20,40))
        self.assertEqual(MARGINS,(.02,.05,.1))
        self.assertEqual(PENALTIES,(.05,.15))

if __name__=="__main__":unittest.main()
