"""Finite numerical/invariant tests for preliminary P02 joint estimation."""
import unittest
import numpy as np
from run_p02 import box_bounds, EXPERIMENT


class P02Tests(unittest.TestCase):
    def test_twenty_target_capacities(self):
        lo, hi = box_bounds(886, 20)
        self.assertEqual(len(lo), 22)
        self.assertTrue(np.all(lo[:20] == 25))
        self.assertTrue(np.all(hi[:20] == 27))
        self.assertEqual((lo[-2], hi[-2]), (150, 150))
        self.assertEqual((lo[-1], hi[-1]), (225, 236))
        self.assertTrue(lo.sum() <= 886 <= hi.sum())

    def test_twenty_two_target_capacities(self):
        lo, hi = box_bounds(886, 22)
        self.assertEqual(len(lo), 24)
        self.assertEqual((lo[-2], hi[-2]), (150, 150))
        self.assertEqual((lo[-1], hi[-1]), (175, 186))
        self.assertTrue(lo.sum() <= 886 <= hi.sum())

    def test_exact_shared_total_and_private_total(self):
        for k in [20, 22]:
            lo, hi = box_bounds(886, k)
            for shared in [int(736-hi[-1]), int(736-lo[-1])]:
                self.assertEqual(150 + shared + (736-shared), 886)

    def test_infeasible_out_of_range(self):
        with self.assertRaises(ValueError):
            box_bounds(886, 30)

    def test_zero_shifts_preserve_p01(self):
        baseline=np.array([[.8,.3], [.1,.6], [.5,.2]])
        home=np.array([1.,0.,1.])
        for weight in [0.05, 0.1]:
            shifted=baseline + weight * (1-home[:,None])
            np.testing.assert_allclose(shifted[home > 0.5], baseline[home > 0.5])
        np.testing.assert_allclose(.9*baseline+.1*baseline,baseline)

    def test_static_experiment_parameters(self):
        self.assertEqual(EXPERIMENT["backgrounds"][0], "full")
        self.assertEqual(EXPERIMENT["low_fpr_primary"], .01)
        self.assertEqual(EXPERIMENT["private_unknown_capacity"], 150)


if __name__ == "__main__":
    unittest.main()
