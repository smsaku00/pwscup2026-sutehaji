"""P07 invariants: fixed QI7 rank, all20 peer correction, heldout-only policy."""
import unittest
import numpy as np
import pandas as pd
from run_qcohort_holdout import percentile, two_matrices, choose, METHODS

class Base:
    @staticmethod
    def rivals(v,stat):
        return np.column_stack([stat(np.delete(v,j,axis=1),axis=1)
                                for j in range(v.shape[1])])

class UnitTests(unittest.TestCase):
    def test_holdout_rank(self):
        base=np.arange(10,dtype=float)
        probe=np.array([-1,2,15.],float)
        normal=np.ones(10,dtype=bool)
        rr,pp=percentile(base,probe,normal)
        self.assertAlmostEqual(rr[0],.1)
        np.testing.assert_allclose(pp,[0,.25,1])
    def test_same_qi7_p01_p03_dimensions(self):
        a=np.arange(12*3,dtype=float).reshape(12,3)
        q=np.arange(4*3,dtype=float).reshape(4,3)
        mask=np.r_[np.ones(10,dtype=bool),np.zeros(2,dtype=bool)]
        bv={"P01":a,"P03_Cnn":a}
        nv={"P01":q,"P03_Cnn":q}
        x,y=two_matrices(Base(),bv,nv,mask)
        self.assertEqual(set(x),set(METHODS))
        self.assertEqual(x["P01"].shape,(10,3))
        self.assertEqual(y["P03_Cnn"].shape,(4,3))
    def test_selector_does_not_use_membership_labels(self):
        rows=[]
        for h in [1,2,5,6,7,8]:
            for method,r99,r95 in [("P01",.03,.1),("P03_Cnn",.005,.05)]:
                rows.append({"target":10,"heldout_Q_cohort":h,
                             "method":method,"risk99":r99,"risk95":r95})
        f=pd.DataFrame(rows)
        self.assertEqual(choose(f,10,"P04"),"P01")
        self.assertEqual(choose(f,10,"Q99_delta005"),"P03_Cnn")
        self.assertEqual(choose(f,10,"Q95_delta010"),"P03_Cnn")
        f["hidden_target_truth"]=np.arange(len(f))
        self.assertEqual(choose(f,10,"Q99_delta005"),"P03_Cnn")
    def test_never_call_q_missing_fields_to_kde(self):
        import inspect
        from run_qcohort_holdout import run
        s=inspect.getsource(run)
        self.assertIn("archive.scores_KDEA(common,b,a,cs,teams)",s)
        self.assertNotIn("scores_KDEA(common,selectframe",s)
        self.assertIn("if t==hold:continue",inspect.getsource(__import__("run_qcohort_holdout").evaluate_q_risk))

if __name__=="__main__":
    unittest.main()
