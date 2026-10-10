import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
import run_bg7 as e

class ProtocolTests(unittest.TestCase):
    def test_seven_nonempty_subsets(self):
        self.assertEqual(len(set(e.BACKGROUNDS)),7)
        actual={frozenset(bg.split('+')) for bg in e.BACKGROUNDS}
        self.assertEqual(actual,{frozenset(s for j,s in enumerate(('starter','prelim','main')) if bits&(1<<j)) for bits in range(1,8)})

    def test_original_hyperparameters(self):
        self.assertEqual(e.HOLD,(5,7,9,10,13,26))
        self.assertEqual(e.general.CONFIG['shrink'],500)
        self.assertEqual(e.general.CONFIG['ridge_penalty'],.1)
        self.assertEqual(e.tapas.CONFIG['smoothing'],.5)
        self.assertEqual(e.tapas.CONFIG['blend_weight'],.25)
        self.assertEqual(len(e.tapas.PAIRS),7)
        self.assertEqual(len(e.tapas.TRIPLES),7)

    def test_allfixed_convention(self):
        normal=np.array([True,True,False,False,False])
        private=np.array([False,False,True,True,False])
        rare=np.array([False,False,False,True,True])
        actual=e.base.confidence(np.array([[1.,1.],[2.,1.]]),normal,private,rare)
        np.testing.assert_allclose(actual,[[.75,.75],[.9,.75],[0,0],[0,0],[.4,.4]])

    def test_hash_is_platform_fixed_float64(self):
        x=np.array([[1,2],[3,4]],dtype=np.int32)
        self.assertEqual(e.matsha(x),e.matsha(x.astype('>f8')))

    def test_rank_reference_self(self):
        x=np.arange(30)[:,None].astype(float)
        out=e.rank_agreement({('main','all22'):{'P01':x}},{'all22':[0]})
        self.assertEqual(out.top20_overlap.item(),20)
        self.assertAlmostEqual(out.spearman.item(),1)
        self.assertEqual(out.mean_abs_rank_shift.item(),0)

    def test_original_reproductions(self):
        items=json.loads((e.HERE/'baseline_validation.json').read_text())
        self.assertEqual(len(items),7)
        self.assertTrue(all(x['exact_prediction_hash_matches'] for x in items))

    def test_finished_outputs(self):
        for phase,target_count,scope_count,suffix in [('prelim',38,2,''),('main',22,1,''),('prelim',38,2,'_primary'),('main',22,1,'_primary')]:
            d=e.HERE/f'results_{phase}{suffix}'
            self.assertTrue((d/'run_record.json').exists(), str(d))
            ranks=pd.read_csv(d/'rank_agreement.csv')
            self.assertEqual(len(ranks),7*6*target_count)
            self.assertTrue(np.isfinite(ranks.select_dtypes('number')).all().all())
            self.assertEqual(len(json.loads((d/'prediction_hashes.json').read_text())),7*6*scope_count)
            for f in json.loads((d/'null_fits.json').read_text()):
                self.assertFalse(set(f['train_cohorts'])&set(e.HOLD))
                self.assertNotIn(f['team'],f['train_cohorts'])
            if phase=='prelim':
                by=pd.read_csv(d/'by_target.csv')
                self.assertEqual(len(by),7*6*38*2)
                self.assertTrue(all(x['exact_match'] for x in json.loads((d/'adapter_baseline_regression.json').read_text())))
            else:
                self.assertFalse((d/'by_target.csv').exists())

    def test_primary_derived_bins(self):
        for phase in ('prelim','main'):
            d=e.HERE/f'results_{phase}_primary'
            edges=json.loads((d/'bin_edges.json').read_text())
            for bg in e.BACKGROUNDS:
                a=pd.read_csv(f'/workspace/shared/mia_bg7_inputs/bg/subsets/{bg}.csv',float_precision='round_trip')
                for col in e.common.NUM7:
                    np.testing.assert_array_equal(edges[bg][col],np.unique(np.quantile(a[col],np.arange(.1,1,.1))))

    def test_primary_controlled_invariants(self):
        for phase in ('prelim','main'):
            def read(suffix):
                rows=json.loads((e.HERE/f'results_{phase}{suffix}'/'prediction_hashes.json').read_text())
                return {(r['background'],r['inference'],r['method']):r['sha256'] for r in rows}
            fixed,primary=read(''),read('_primary')
            self.assertEqual(fixed.keys(),primary.keys())
            for key,h in primary.items():
                bg,_,method=key
                if bg==phase or method in ('P01','conditional_quartile500','null_copula_ridge01'):
                    self.assertEqual(h,fixed[key],key)

if __name__=='__main__': unittest.main()
