"""Deterministic public-only KDE numeric/categorical scoring tests."""
import unittest
import numpy as np
from run_kde_rank_fusion import median_iqr_align,log_density,NUM,CAT,ALIGN,BANDWIDTH,WEIGHTS


class KDEFusionTests(unittest.TestCase):
    def test_align_median_even_with_constant_support(self):
        a=np.array([[1.,1.,3.],[2.,2.,3.],[3.,4.,3.],[4.,9.,3.]],float)
        c=np.array([[1.,1.,4.],[1.,1.,4.],[3.,4.,4.],[3.,4.,4.]],float)
        v=median_iqr_align(a,c)
        self.assertEqual(v.shape,c.shape)
        self.assertTrue(np.isfinite(v).all())
        self.assertAlmostEqual(np.median(v[:,2]),3.)
        self.assertTrue(np.all(v[:,1]>=0))

    def test_density_duplicate_reference_normalizes(self):
        d=np.array([[0.0,2.0],[2.0,0.0]])
        ddup=np.repeat(d,2,axis=1)
        for h in BANDWIDTH:
            np.testing.assert_allclose(
                log_density(d,h,2),log_density(ddup,h,4),rtol=0,atol=1e-12)

    def test_density_prefers_nearby_reference(self):
        close=np.full((7,100),.2)
        far=np.full((7,100),4.)
        for h in BANDWIDTH:
            self.assertTrue(np.all(log_density(close,h,100)>
                                   log_density(far,h,100)))

    def test_fixed_candidate_panel(self):
        self.assertEqual(len(NUM)+len(CAT),13)
        self.assertEqual(len(ALIGN)*len(BANDWIDTH)*len(WEIGHTS)+1,13)

if __name__=="__main__":
    unittest.main()
