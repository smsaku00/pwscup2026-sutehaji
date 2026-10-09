"""Checks for P08 public-copy handling and conditional temporal evidence."""
import unittest
import numpy as np
import pandas as pd
from run_p08_ab import match_public,clean_public_C,time_conditional_signal,METH,TIME_CHANNELS

COLUMNS=("age","sex","prefecture","BMI","SBP","TG","HDL","ALT","smoking","FPG","time","onset","death")
def synthetic(n,shift=0.):
    x=[]
    for i in range(n):
        d={k:1. for k in COLUMNS}
        d.update(age=35+i,sex=i%2,prefecture="a" if i%2 else "b",BMI=20.+i,
          SBP=120.+i,TG=110.+i,HDL=40.+i,ALT=20.+i,smoking=i%2,
          FPG=80.+i,time=2.0+i*.5+shift,onset=i%2,death=0)
        x.append(d)
    return pd.DataFrame(x)

class P08Checks(unittest.TestCase):
    def test_all_copy_falls_back_without_fake_residual(self):
        a=synthetic(20)
        c=a.sample(frac=1,random_state=5).reset_index(drop=True)
        o,stats=clean_public_C({18:c},a,[18])
        self.assertEqual(stats.iloc[0].full_public_A_copy_rows,20)
        self.assertEqual(stats.iloc[0].strict_fallback,1)
        self.assertIs(o["A_full_public_remove"][18],c)
        self.assertTrue(len(o["A_public_keep20"][18])<20)
    def test_exact_copy_and_noncopy(self):
        a=synthetic(20);c=pd.concat([a.iloc[:10],synthetic(20,shift=1.1).iloc[10:]],ignore_index=True)
        copy=match_public(c,a)
        self.assertEqual(int(copy.sum()),10)
        o,s=clean_public_C({29:c},a,[29])
        self.assertEqual(len(o["A_full_public_remove"][29]),10)
        self.assertEqual(int(s.iloc[0].soft_copied_retained)>=0,True)
    def test_temporal_conditional_evidence_finite(self):
        a=synthetic(40);b=synthetic(20);c=synthetic(30,shift=0.33)
        for method in TIME_CHANNELS:
            scores=time_conditional_signal(b,a,c,method)
            self.assertEqual(scores.shape,(20,))
            self.assertTrue(np.isfinite(scores).all())
    def test_exact_matches_are_positive_evidence(self):
        a=synthetic(35,shift=1.7);b=synthetic(20)
        c=synthetic(25)
        sig=time_conditional_signal(b,a,c,"time_exact_age10")
        self.assertTrue(np.any(sig>0))
    def test_fixed_method_count(self):
        self.assertEqual(len(METH),10)
        self.assertEqual(len(set(METH)),10)

if __name__=="__main__":
    unittest.main()
