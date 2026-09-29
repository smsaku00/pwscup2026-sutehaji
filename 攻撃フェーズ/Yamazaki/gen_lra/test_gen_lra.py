"""Mathematical and format checks using only artificial data."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from run_gen_lra import gen_lra_from_matrices, run


class GenLRATests(unittest.TestCase):
    def test_fast_identity_matches_explicit_kde_refits(self):
        rng = np.random.default_rng(99)
        ref = rng.normal(size=(37, 3))
        cand = rng.normal(size=(5, 3))
        syn = rng.normal(size=(23, 3))
        h, k = 0.8, 4
        fast = gen_lra_from_matrices(ref, cand, syn, k=k, bandwidth=h, batch_size=3)
        kde_ref = KernelDensity(kernel="gaussian", bandwidth=h).fit(ref)
        expected = []
        for x in cand:
            ids = np.argsort(np.sum((syn - x) ** 2, axis=1))[:k]
            local = syn[ids]
            kde_aug = KernelDensity(kernel="gaussian", bandwidth=h).fit(np.vstack([ref, x]))
            expected.append(np.sum(kde_aug.score_samples(local) - kde_ref.score_samples(local)))
        np.testing.assert_allclose(fast, expected, rtol=0, atol=1e-12)

    def test_memorized_neighborhood_scores_higher(self):
        ref = np.array([[-1.0], [0.0], [1.0], [2.0]])
        syn = np.array([[0.01], [0.02], [0.03]])
        cand = np.array([[0.0], [4.0]])
        scores = gen_lra_from_matrices(ref, cand, syn, k=3, bandwidth=0.3)
        self.assertGreater(scores[0], scores[1])

    def test_submission_uses_only_declared_targets(self):
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            (dist / "targets").mkdir()
            columns = ["record_id", "age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL",
                       "ALT", "smoking", "FPG", "time", "onset", "death"]

            def frame(n):
                a = rng.normal(size=(n, len(columns)))
                df = pd.DataFrame(a, columns=columns)
                df["record_id"] = np.arange(n)
                df["prefecture"] = rng.choice(["A", "B"], size=n)
                return df

            frame(8).to_csv(dist / "B_practice.csv", index=False)
            frame(20).to_csv(dist / "A_bg.csv", index=False)
            frame(8).drop(columns="record_id").to_csv(dist / "targets/C_4.csv", index=False)
            (dist / "targets/mia_columns.json").write_text(json.dumps({"target_columns": ["4"]}))
            out = dist / "F_mia.csv"
            run(dist, out, "all", 3, 1.0, 4)
            result = pd.read_csv(out)
            self.assertEqual(result.columns.tolist(), ["record_id", "4"])
            self.assertEqual(len(result), 8)
            self.assertTrue(result["4"].between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
