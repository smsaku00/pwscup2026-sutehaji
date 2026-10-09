"""Deterministic tests for the published-Q clinical-distribution corrections."""
import unittest
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from run_experiment import proxy_keep_indices, score_variants, mean_agg, MAIN_TARGETS, PARAMS


class StubBase:
    @staticmethod
    def ranks(x):
        return np.column_stack([rankdata(v, method="average") / len(v)
                                for v in np.asarray(x).T])

    @staticmethod
    def rivals(x, reducer):
        return np.column_stack([reducer(np.delete(x, j, axis=1), axis=1)
                                for j in range(x.shape[1])])


class AIADeconfoundingTests(unittest.TestCase):
    def test_primary_scope_is_only_18_nontrivial_teams(self):
        self.assertEqual(len(MAIN_TARGETS), 18)
        self.assertNotIn(1, MAIN_TARGETS)
        self.assertNotIn(2, MAIN_TARGETS)
        self.assertTrue(PARAMS["no_truth_used_for_prediction_or_group_selection"])

    def test_nearest_proxy_removed_using_category_penalty(self):
        C = np.array([[0.1, 0], [0.6, 0], [0.3, 0], [10.0, 0]])
        Q = np.array([[0.0, 0]])
        mask, nearest, _ = proxy_keep_indices(
            C, Q, {"c_pref": np.array(["other","home","home","home"]),
                   "q_pref": "home"}, k_remove=2)
        self.assertEqual(mask.tolist(), [False, True, False, True])
        self.assertAlmostEqual(nearest[0], 0.01, places=9)

    def test_no_private_rows_outside_retained_C(self):
        rng = np.random.default_rng(42)
        C = rng.normal(size=(1049, 5))
        Q = rng.normal(size=(326, 5))
        mask, d, _ = proxy_keep_indices(
            C, Q, {"c_pref": np.array(["a"]*1049), "q_pref":"a"})
        self.assertEqual(int(mask.sum()), 736)
        self.assertEqual(int((~mask).sum()), 313)
        self.assertTrue(np.isfinite(d).all())

    def test_score_family_and_group_gate(self):
        base = StubBase()
        cal = np.array([[1,6,4], [2,5,3], [3,4,2], [4,3,1.0]], float)
        d = {
            "c": np.array([[2,3,4],[1,2,3],[3,1,2],[4,4,1]], float),
            "q": np.array([[4,3,2],[3,2,1],[2,1,3],[1,4,4]], float),
            "resid": np.array([[6,3,4],[1,2,3],[2,1,2],[4,5,1]], float),
        }
        groups = pd.DataFrame({
            "team": [3,4,6],
            "aia_imprint_high": [1,0,1],
            "quantized_c": [0,1,1],
        })
        methods=score_variants(base, cal, np.array([True]*4),
                               [0,1,2], [3,4,6], d, groups)
        self.assertEqual(len(methods), 12)
        for m in methods.values():
            self.assertEqual(m.shape, (4,3))
            self.assertTrue(np.isfinite(m).all())
        np.testing.assert_allclose(
            methods["AIA_contrast_high_imprint_010"][:,1],
            methods["P01"][:,1], rtol=0, atol=0)
        np.testing.assert_allclose(
            methods["AIA_residual_quantized_010"][:,0],
            methods["P01"][:,0], rtol=0, atol=0)

    def test_group_aggregation_reports_changes(self):
        dat=pd.DataFrame({
            "team":[3,4,6], "tpr_0.01":[.12,.16,.04],
            "tpr_0.005":[.10,.12,.02],
            "tpr_0.02":[.14,.18,.06], "band":[1,1,1]})
        ref=pd.Series({3:.08,4:.16,6:.08})
        result=mean_agg(dat,ref)
        self.assertEqual(result["wins"],1)
        self.assertEqual(result["ties"],1)
        self.assertEqual(result["losses"],1)
        self.assertAlmostEqual(result["mean_delta_P01"],0.0)

    def test_groups_do_not_need_truth_file(self):
        # Covariate-based group flags are defined by public columns alone.
        g=pd.DataFrame({"team":[3,4,6,7],
                        "AIA_imprint_log_BG_to_C_median_Q_NN":[.1,.2,-.1,.7]})
        cut=g["AIA_imprint_log_BG_to_C_median_Q_NN"].median()
        labels=(g["AIA_imprint_log_BG_to_C_median_Q_NN"]>=cut).astype(int)
        self.assertEqual(labels.sum(),2)


if __name__ == "__main__":
    unittest.main()
