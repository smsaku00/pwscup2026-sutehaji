"""Independent numerical and scoring checks: python -m unittest -v test_experiment."""
from __future__ import annotations

import itertools
import json
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

import run_experiment as e


class JointInferenceTests(unittest.TestCase):
    def test_capacity_intersection_prelim_and_main(self):
        for k, expected in [(20, (500, 511)), (22, (550, 561)),
                            (18, (450, 461)), (17, (425, 436))]:
            lo, hi = e.capacity_bounds(886, k)
            self.assertEqual((886 - hi[-1], 886 - lo[-1]), expected)
            self.assertTrue(np.all(lo[:-1] == 25))
            self.assertTrue(np.all(hi[:-1] == 27))
            self.assertLessEqual(lo.sum(), 886)
            self.assertGreaterEqual(hi.sum(), 886)

    def test_own_column_is_excluded_from_competitors(self):
        s = np.array([[.9, .4, .1], [.2, .8, .7]])
        np.testing.assert_allclose(e.rivals(s, np.max), [[.4, .9, .9], [.8, .7, .8]])
        np.testing.assert_allclose(e.rivals(s, np.median), [[.25, .5, .65], [.75, .45, .5]])

    def test_extended_ranks_preserve_normal_ties(self):
        v = np.array([[1., 4.], [2., 2.], [2., 2.], [8., 1.], [1.5, 3.], [12., 0.]])
        normal = np.array([True, True, True, True, False, False])
        private = np.array([False, False, False, False, True, False])
        p = e.extend_ranks(v, normal, ~private)
        np.testing.assert_array_equal(p[normal[~private]], e.ranks(v[normal]))
        self.assertTrue(np.all((0 <= p) & (p <= 1)))

    def test_hard_flow_matches_exhaustive_search(self):
        rng = np.random.default_rng(20261009)
        s = rng.normal(size=(6, 3))
        lo, hi = np.array([1, 1, 2]), np.array([2, 3, 4])
        p, diagnostic = e.hard_assignment(s, lo, hi)
        best = -np.inf
        for assignment in itertools.product(range(3), repeat=6):
            mass = np.bincount(assignment, minlength=3)
            if np.all(mass >= lo) and np.all(mass <= hi):
                best = max(best, sum(s[i, j] for i, j in enumerate(assignment)))
        self.assertAlmostEqual(float((p * s).sum()), best, places=11)
        self.assertEqual(diagnostic['integrality'], 0)
        self.assertEqual(e.feasibility(p, lo, hi), 0)

    def test_soft_dual_matches_independent_primal_solver(self):
        # SLSQP uses all primal entries; production uses the much smaller dual.
        s = np.array([[.1, .3, .7], [.8, .1, .4], [.2, .9, .2], [.6, .4, .1]])
        lo, hi, tau = np.array([1., 1., 1.]), np.array([1., 2., 2.]), .3
        p, diagnostic = e.soft_assignment(s, lo, hi, tau)
        def objective(x):
            q = x.reshape(s.shape)
            return float(-(q * s).sum() + tau * xlogy(q, q).sum())
        constraints = [
            {'type': 'eq', 'fun': lambda x: x.reshape(s.shape).sum(1) - 1},
            {'type': 'eq', 'fun': lambda x: x.reshape(s.shape).sum(0)[0] - 1},
            {'type': 'ineq', 'fun': lambda x: x.reshape(s.shape).sum(0)[1:] - lo[1:]},
            {'type': 'ineq', 'fun': lambda x: hi[1:] - x.reshape(s.shape).sum(0)[1:]},
        ]
        initial = np.tile([.25, .375, .375], (4, 1)).ravel()
        independent = minimize(objective, initial, method='SLSQP',
                               bounds=[(1e-9, 1)] * 12, constraints=constraints,
                               options={'ftol': 1e-12, 'maxiter': 2000})
        self.assertTrue(independent.success, independent.message)
        self.assertAlmostEqual(objective(p.ravel()), independent.fun, places=7)
        np.testing.assert_allclose(p.ravel(), independent.x, atol=2e-5, rtol=0)
        self.assertLess(diagnostic['feasibility'], 1e-6)

    def test_soft_solution_is_row_and_column_equivariant(self):
        rng = np.random.default_rng(28)
        s = rng.normal(size=(12, 4))
        lo, hi = np.array([2, 1, 3, 2]), np.array([3, 4, 4, 5])
        p, _ = e.soft_assignment(s, lo, hi, .2)
        rows, cols = rng.permutation(12), rng.permutation(4)
        q, _ = e.soft_assignment(s[rows][:, cols], lo[cols], hi[cols], .2)
        np.testing.assert_allclose(p[rows][:, cols], q, atol=2e-6, rtol=0)

    def test_score_free_problem_does_not_invent_row_information(self):
        # With no distinguishing score, exchangeable rows retain equal weights.
        s = np.zeros((10, 3))
        lo, hi = np.array([2, 3, 5]), np.array([2, 3, 5])
        p, _ = e.soft_assignment(s, lo, hi, .1)
        np.testing.assert_allclose(p, np.tile([.2, .3, .5], (10, 1)), atol=1e-6)

    def test_confidence_preserves_non_normal_and_ties(self):
        normal = np.array([True, True, True, False, False, False])
        private = np.array([False, False, False, True, True, False])
        rare = np.array([False, False, False, False, True, True])
        s = e.confidence(np.array([[.5, .2], [.5, .9], [.1, .4]]), normal, private, rare)
        np.testing.assert_array_equal(s[private], 0)
        np.testing.assert_array_equal(s[rare & ~private], .4)
        self.assertEqual(s[0, 0], s[1, 0])
        self.assertTrue(np.all((s[normal] > .6) & (s[normal] <= .9)))

    def test_roc_all_ties_and_perfect_and_boundary_ties(self):
        y = np.array([1, 1, 0, 0, 0, 0])
        ties = e.metrics(np.ones(6), y)
        self.assertAlmostEqual(ties['tpr_0.01'], .01)
        self.assertAlmostEqual(ties['auc'], .5)
        perfect = e.metrics(y.astype(float), y)
        self.assertAlmostEqual(perfect['tpr_0.01'], 1)
        # Top-10 is the expected number of positives after random tie-breaking.
        y = np.r_[np.ones(4, dtype=int), np.zeros(16, dtype=int)]
        m = e.metrics(np.ones(20), y)
        self.assertAlmostEqual(m['top10_expected_hits'], 2)

    def test_band_boundaries_are_open_on_lower_end(self):
        # Verify threshold convention without depending on interpolated decimals.
        band = lambda v: 1 + int(v > .25) + int(v > .5) + int(v > .75)
        self.assertEqual([band(v) for v in [0, .25, .2501, .5, .5001, .75, .7501, 1]],
                         [1, 1, 2, 2, 3, 3, 4, 4])


if __name__ == '__main__':
    unittest.main(verbosity=2)
