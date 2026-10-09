from __future__ import annotations
import unittest,os
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('aia_change_width',HERE/'width_sweep.py')
w=importlib.util.module_from_spec(spec);sys.modules[spec.name]=w;spec.loader.exec_module(w)

class WidthSweepTests(unittest.TestCase):
    def test_no_edit_is_identity(self):
        base=np.array([True,False,True,False])
        gg=np.array([.1,.4,.02,-.1])
        for family in w.FAMILIES:
            p=w.propose(base,np.arange(4),gg,np.full(4,11.),family,0,-.02)
            np.testing.assert_array_equal(p,[9.5,0,9.5,0])

    def test_budget_nested_single_family(self):
        base=np.array([True,True,False,True,True])
        gg=np.array([.04,.01,-np.inf,.03,.03])
        changed=[]
        for k in [0,1,2,3,4,326]:
            x=w.propose(base,np.arange(5),gg,np.full(5,11.),'down11',k,0)
            mask=x!=np.where(base,9.5,0)
            changed.append(mask)
        self.assertTrue(all(np.all(~a|b) for a,b in zip(changed,changed[1:])))

    def test_threshold_monotonic(self):
        b=np.array([True]*5)
        gg=np.array([-.02,0,.01,.02,.05])
        cnt=[]
        for threshold in [-.03,0,.01,.02,.06]:
            z=w.propose(b,np.arange(5),gg,np.full(5,11.),'down11',5,threshold)
            cnt.append(int(np.sum(z==11)))
        self.assertEqual(cnt,[5,3,2,1,0])

    def test_permutation_invariance_and_ties(self):
        b=np.array([True,True,False,True,True])
        gg=np.array([.1,.1,-np.inf,.1,.01])
        ids=np.array([3,8,9,0,1]);order=np.array([3,2,0,4,1]);inv=np.argsort(order)
        a=w.propose(b,ids,gg,np.full(5,11.),'down11',2,.02)
        z=w.propose(b[order],ids[order],gg[order],np.full(5,11.),'down11',2,.02)
        np.testing.assert_array_equal(a,z[inv])
        self.assertEqual(int((a==11).sum()),2)
        self.assertTrue(a[3]==11 and a[0]==11)

    def test_gains_look_at_no_truth(self):
        p=np.array([.1,.5,.9]);base=np.array([True,False,True]);rates={'m9':.75,'c9':.75,'m0':.02,'c0':.02}
        G,t=w.gains(p,rates,.9,base)
        self.assertEqual(G['down11'].shape,(3,))
        self.assertTrue(np.isneginf(G['down11'][1]))
        self.assertTrue(np.isneginf(G['up11'][0]))
        self.assertTrue(np.all(np.isfinite(t)))

    def test_score_metrics_consistent_with_saved_parent(self):
        cal=Path(os.environ.get('AIA_CAL_DIR','/mnt/data/aia_calibration_20261009'))
        priorpath=Path(os.environ.get('AIA_PRIOR','/mnt/data/aia_density_ratio_20261009/prior'))
        root=Path(os.environ.get('AIA_ROOT','/mnt/data/aia_density_ratio_20261009/inputs'))
        if not (cal/'calibrated_policy.py').exists() or not (priorpath/'experiment.py').exists() or not (root/'公式データ/予備戦').exists():
            self.skipTest('Full external inputs not available')
        c=w.import_module(cal/'calibrated_policy.py','aia_width_cp_test')
        prior=c.import_prior(priorpath)
        _,D=prior.read_data(root)
        for k in (1,5,13,23):
            rng=np.random.default_rng(k)
            p=np.where(rng.uniform(size=326)>.5,9.5,0)
            a=c.score_metrics(p,D[k][2]);b=prior.metrics(p,D[k][2])
            self.assertAlmostEqual(a['r_all'],b['r_all'],places=12)
            self.assertAlmostEqual(a['r_rare'],b['r_rare'],places=12)
            self.assertEqual(a['hit_m'],b['hit_member'])

    def test_stored_parity(self):
        result_dir=Path(os.environ.get('AIA_WIDTH_RESULTS',HERE/'results'))
        cal=Path(os.environ.get('AIA_CAL_DIR','/mnt/data/aia_calibration_20261009'))
        if not (result_dir/'per_team_grid.csv.gz').exists() or not (cal/'results/per_target.csv').exists():
            self.skipTest('Full external results not available')
        df=pd.read_csv(result_dir/'per_team_grid.csv.gz')
        fold=df[(df.folds==4)&(df.family=='down11')&(df.budget==10)&(df.margin==.02)]
        old=pd.read_csv(cal/'results/per_target.csv')
        old=old[(old.model=='c_bg_plus_local')&(old.policy=='conservative_cap10')].set_index('team')
        self.assertEqual(len(fold),21)
        for a,b in [('r_all','r_all'),('r_rare','r_rare'),('changed','changed_vs_base')]:
            delta=(fold.set_index('team')[a]-old[b]).abs()
            self.assertLess(float(delta.max()),1e-12)

if __name__=='__main__':unittest.main(verbosity=2)
