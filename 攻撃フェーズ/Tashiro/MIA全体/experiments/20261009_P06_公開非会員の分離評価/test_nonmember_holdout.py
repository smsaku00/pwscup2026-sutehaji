"""P06 invariants: empirical null ranks and no target-label access in selector."""
import unittest
import numpy as np
import pandas as pd
from run_nonmember_holdout import percentiles, formula, choose, MODELS, POLICIES,SEEDS

class BaseStub:
    @staticmethod
    def rivals(x,stat):
        return np.column_stack([stat(np.delete(x,j,axis=1),axis=1) for j in range(x.shape[1])])

class SelectionTests(unittest.TestCase):
    def test_pctiles_holdout(self):
        b=np.arange(10,dtype=float).reshape(-1,1)
        null=np.array([[-3.],[2.],[10.]])
        mask=np.ones(10,dtype=bool)
        val,probe=percentiles(b,null,mask)
        self.assertAlmostEqual(val[0,0],.1)
        np.testing.assert_allclose(probe[:,0],[0.,.25,1.],atol=1e-15)
    def test_panel_shapes(self):
        normal=np.r_[np.ones(10,dtype=bool),np.zeros(2,dtype=bool)]
        b=np.column_stack([np.arange(12,dtype=float)+j for j in range(3)])
        hold=b[:5]
        values,neg=formula(BaseStub(),b,hold,b,hold,11-b,11-hold,normal)
        self.assertEqual(set(values),set(MODELS))
        for k in MODELS:
            self.assertEqual(values[k].shape,(10,3))
            self.assertEqual(neg[k].shape,(5,3))
    def test_selector_uses_only_public_holdout_risks(self):
        rows=[]
        for seed in SEEDS:
            view=f"split80_seed{seed}"
            for method,r99,r95 in [('KDEA',.020,.10),('P01',.010,.06),('P03_Cnn',.025,.12)]:
                rows.append({'view':view,'source':'A_bg_20pct_heldout',
                             'team':7,'method':method,'risk99':r99,'risk95':r95})
        data=pd.DataFrame(rows)
        self.assertEqual(choose(data,7,'P04'),'KDEA')
        self.assertEqual(choose(data,7,'risk99_0005'),'P01')
        self.assertEqual(choose(data,7,'risk95_0010'),'P01')
        self.assertEqual(choose(data,7,'risk99_all3'),'P01')
        # Changing any unrelated outcome column cannot influence selection.
        data['preliminary_private_truth']=1e9
        self.assertEqual(choose(data,7,'risk99_0005'),'P01')

if __name__=="__main__":unittest.main()
