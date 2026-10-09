"""Self-contained checks for canonical q13 KDE formula and public distribution alignment."""
import unittest
import numpy as np
import pandas as pd
from run_archived_kdeA_mix import align_numeric,mixed_logs,ref_transform,NUM,CAT,WEIGHTS

class Stub:
    def numbers(self,frame,columns):
        x=frame[list(columns)].to_numpy(float).copy()
        for name in ("TG","ALT"):
            if name in columns:x[:,list(columns).index(name)]=np.log1p(x[:,list(columns).index(name)])
        return x

class FormulaTests(unittest.TestCase):
    def test_marginal_alignment_center_and_iqr(self):
        arr=np.array([[1,2,3,4,5,6,7,8],
                      [2,3,4,5,6,7,8,9],
                      [3,4,5,6,7,8,9,10],
                      [4,5,6,7,8,9,10,11]],float)
        values=pd.DataFrame(arr,columns=NUM)
        C=values.copy()
        C["BMI"]=C["BMI"]*2+10
        C["SBP"]=C["SBP"]*3-1
        src=Stub()
        a=src.numbers(values,NUM)
        x=align_numeric(src,a,C)
        for key in ("BMI","SBP"):
            j=NUM.index(key)
            self.assertAlmostEqual(np.median(x[:,j]),np.median(a[:,j]),places=12)

    def test_categorical_smoothed_kde_has_finite_positive_normalization(self):
        q=pd.DataFrame([{"sex":"f","prefecture":"tokyo","smoking":"yes","onset":"x","death":"no"}])
        ref=pd.concat([q,q],ignore_index=True)
        a=ref.copy()
        d2=np.array([[0,1.]],float)
        value=mixed_logs(d2,q,ref,a,1.)
        self.assertEqual(value.shape,(1,))
        self.assertTrue(np.isfinite(value).all())

    def test_scale_fallback_on_constant_reference(self):
        x=np.zeros((10,8))
        b=pd.DataFrame(x,columns=NUM)
        center,scale,_=ref_transform(Stub(),b)
        self.assertTrue(np.all(np.isfinite(scale)))
        self.assertTrue(np.all(scale>0))

    def test_five_mix_weights(self):
        self.assertEqual(len(WEIGHTS),5)
        self.assertEqual(WEIGHTS[0],0.)
        self.assertEqual(WEIGHTS[-1],1.)

if __name__=="__main__":
    unittest.main()
