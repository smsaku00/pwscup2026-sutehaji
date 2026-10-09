"""Public-feature LOOTO transfer invariants: no target label during selection."""
import unittest
import numpy as np
import pandas as pd
from run_transfer import TARGET_SET, public_distances, neighbor_indices, \
    selected_method, panel_audit, METHODS_N


class ThresholdTransferTests(unittest.TestCase):
    def test_heldout_never_neighbor(self):
        labels = np.abs(np.subtract.outer(np.arange(17), np.arange(17))).astype(float)
        np.fill_diagonal(labels,np.inf)
        d = pd.DataFrame(labels,index=TARGET_SET,columns=TARGET_SET)
        for i in range(17):
            for k in [3,5,7]:
                ns = neighbor_indices(d,i,k)
                self.assertEqual(len(ns),k)
                self.assertNotIn(i,ns)
                rand = neighbor_indices(d,i,k,True)
                self.assertNotIn(i,rand)

    def test_selection_ignores_heldout_truth(self):
        methods=["Q_prefecture_sex_shrink500_cross01","a","b"]
        bgs=["full","x","y","z","a2","b2","c2"]
        values=np.full((4,3,7),0.04)
        # candidate a succeeds for two peer targets
        values[1,1,:]=0.12
        values[2,1,:]=0.15
        choice=selected_method(values,methods,bgs,[1,2,3],"full_hit")
        self.assertEqual(choice,1)
        values[0,:,:]=1.0 # target's labels arbitrarily altered
        self.assertEqual(selected_method(values,methods,bgs,[1,2,3],"full_hit"),choice)

    def test_selection_falls_back_when_no_peer_threshold(self):
        names=["Q_prefecture_sex_shrink500_cross01","candidate"]
        views=["full","seed1"]
        val=np.full((4,2,2),0.02)
        pick=selected_method(val,names,views,[1,2,3],"full_hit")
        self.assertEqual(pick,0)

    def test_target_scope(self):
        self.assertEqual(len(TARGET_SET),17)
        self.assertTrue(1 not in TARGET_SET and 2 not in TARGET_SET and 12 not in TARGET_SET)
        self.assertEqual(METHODS_N,22)

    def test_oracle_audit_is_separate(self):
        names=["Q_prefecture_sex_shrink500_cross01","x"]
        t=np.ones((17,2,7))*0.03
        t[0,1,:]=0.12
        audit=panel_audit(t,names,["full","a","b","c","d","e","f"])
        self.assertEqual(len(audit),17)
        self.assertEqual(int(audit.iloc[0].oracle_fixed22_ge010_all7),1)
        self.assertEqual(int(audit.iloc[1].oracle_fixed22_ge010_all7),0)


if __name__ == "__main__":
    unittest.main()
