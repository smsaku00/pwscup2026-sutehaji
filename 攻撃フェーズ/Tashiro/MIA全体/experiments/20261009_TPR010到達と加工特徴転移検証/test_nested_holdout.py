"""Numerical invariants and outer-holdout independence for nested 0.10 selector."""
import unittest
import numpy as np
import pandas as pd
from run_nested_holdout import TGT, METHOD_SPECS, selector, nested_outer_select


class NestedTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(123)
        self.tensor=rng.uniform(0,.2,(17,3,7))
        self.names=["Q_prefecture_sex_shrink500_cross01","kde_a","kde_b"]
        self.bgs=["full","1","2","3","4","5","6"]
        mat=np.abs(np.subtract.outer(np.arange(17),np.arange(17))).astype(float)
        np.fill_diagonal(mat,np.inf)
        self.dist=pd.DataFrame(mat,index=TGT,columns=TGT)
        self.group=[int(i%2==1) for i in range(17)]

    def test_candidates_72(self):
        self.assertEqual(len(METHOD_SPECS),72)

    def test_selector_does_not_read_heldout_truth(self):
        spec=("all",5,"shrunken")
        for test_index in [0,3,9,16]:
            train=[i for i in range(17) if i!=test_index]
            result=selector(self.tensor,self.names,self.bgs,self.dist,
                            self.group,test_index,train,spec)
            changed=self.tensor.copy()
            changed[test_index,:,:]=1e5
            result_after=selector(changed,self.names,self.bgs,self.dist,
                                  self.group,test_index,train,spec)
            self.assertEqual(result,result_after)

    def test_nested_hyperparam_selection_does_not_read_heldout_truth(self):
        test_index=7
        before=nested_outer_select(self.tensor,self.names,self.bgs,
                                   self.dist,self.group,test_index)
        modified=self.tensor.copy()
        modified[test_index,:,:]=1e5
        after=nested_outer_select(modified,self.names,self.bgs,
                                  self.dist,self.group,test_index)
        self.assertEqual(before[0],after[0])
        self.assertEqual(before[1]["hits"],after[1]["hits"])

    def test_low_group_reverts_to_baseline(self):
        i=0
        self.assertFalse(self.group[i])
        for spec in METHOD_SPECS:
            if spec[0]!="high_only":continue
            decision=selector(self.tensor,self.names,self.bgs,self.dist,
                              self.group,i,[j for j in range(17) if j!=i],spec)
            self.assertEqual(decision,0)


if __name__ == "__main__":
    unittest.main()
