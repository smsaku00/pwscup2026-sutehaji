#!/usr/bin/env python3
"""Independent algebraic, invariance, and score validation tests."""
import unittest
import numpy as np,pandas as pd
from scipy.special import logsumexp
from experiment import (kde_log_ratio, kd_ratio, conditional_weight, metric, top75, soft_group_score)

class Tests(unittest.TestCase):
    def test_kde_direct_sum(self):
        x=np.array([0.,1.2]);c=np.array([-1.,0.,1.,2.]);b=np.array([-2.,-.3,.8,3.,4.]);h=.35
        dC=(x[:,None]-c[None,:])**2;dB=(x[:,None]-b[None,:])**2
        got=kde_log_ratio(dC,dB,len(c),len(b),h)
        exp=np.log(np.exp(-dC/(2*h*h)).mean(axis=1))-np.log(np.exp(-dB/(2*h*h)).mean(axis=1))
        np.testing.assert_allclose(got,exp,atol=1e-12)
    def test_kde_equal(self):
        x=np.array([[0,.3],[1.,1.5]]);c=np.array([[0.,0.],[1.,1.],[2.,2.]]);
        d=((x[:,None,:]-c[None,:,:])**2).sum(axis=2)
        np.testing.assert_allclose(kde_log_ratio(d,d,len(c),len(c),.5),0,atol=1e-13)
    def test_kde_weight_permutations(self):
        dc=np.array([[.01,.7,4.],[1.,.04,1.]]);db=np.array([[1.,.3,.03,.5],[.9,.7,.6,.5]])
        wc=np.array([[.8,.2,0.05],[.3,.5,.02]]);wb=np.array([[.2,.9,.4,.6],[.3,.1,.5,.6]])
        a=kde_log_ratio(dc,db,3,4,.4,wc,wb)
        b=kde_log_ratio(dc[:,[2,0,1]],db[:,[3,0,2,1]],3,4,.4,wc[:,[2,0,1]],wb[:,[3,0,2,1]])
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    def test_knn_ratio_known(self):
        xx=np.array([[0.]]);cx=np.array([[-.1],[.1],[2.]]);bx=np.array([[-.5],[1],[1.5],[2],[3]])
        a=kd_ratio(xx,cx,bx,k=1,dim=1)
        self.assertAlmostEqual(a[0],np.log(5/3)+np.log(.5/.1))
    def test_soft_rank_row_stability(self):
        q=pd.DataFrame({'prefecture':['X']*10, 'sex':[0]*10, 'age':[40]*10,'challenge_row_id':list(range(10))})
        scores=np.arange(10)*.3
        x=soft_group_score(q,scores)
        rng=np.random.default_rng(33);p=rng.permutation(10)
        yy=soft_group_score(q.iloc[p].reset_index(drop=True),scores[p]);np.testing.assert_allclose(x,yy[np.argsort(p)])
    def test_score_formula(self):
        # 163+163 individuals; all member correct, all control wrong -> Rall=1;
        # rare M=13 C=0 -> 1.0 after 1.64SE correction.
        t=pd.DataFrame({'label':['member']*163+['control']*163, 'is_rare':[1]*13+[0]*150+[1]*13+[0]*150,
                        'time':[9.9]*326})
        y=np.array([9.5]*163+[0]*163);m=metric(y,t)
        self.assertAlmostEqual(m['r_all'],1)
        self.assertAlmostEqual(m['r_rare'],1)
    def test_top75(self):
        q=pd.DataFrame({'sex':[0]*40+[1]*20,'challenge_row_id':range(60)})
        a=top75(q,np.arange(60));self.assertEqual(a.sum(),45)

if __name__=='__main__': unittest.main()
