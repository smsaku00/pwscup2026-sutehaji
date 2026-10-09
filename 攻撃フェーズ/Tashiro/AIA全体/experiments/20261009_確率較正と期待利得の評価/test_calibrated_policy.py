from __future__ import annotations
import unittest, tempfile, os
import numpy as np,pandas as pd
from pathlib import Path
import calibrated_policy as cp

class CalibrationDecisionTests(unittest.TestCase):
    def test_projection_half_prior_reached_with_alpha1(self):
        q=pd.DataFrame({'prefecture':['大阪府']*4+['東京都']*3,
                        'sex':[0]*4+[1]*3,'age':[50,51,52,53,60,61,62]})
        p=np.array([.12,.32,.62,.92,.04,.35,.78])
        result=cp.posterior_projection(q,p,alpha=1.0)
        self.assertAlmostEqual(float(result[:4].mean()),.5,places=8)
        np.testing.assert_allclose(result[4:],p[4:],rtol=1e-12,atol=1e-12)

    def test_projection_row_permutation(self):
        q=pd.DataFrame({'prefecture':['大阪府']*6,'sex':[0]*6,'age':[50,51,52,54,50,51]})
        p=np.array([.1,.2,.4,.5,.7,.9]); perm=np.array([3,0,5,2,1,4]);inv=np.argsort(perm)
        a=cp.posterior_projection(q,p)
        b=cp.posterior_projection(q.iloc[perm].reset_index(drop=True),p[perm])[inv]
        np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-11)

    def test_platt_is_monotonic(self):
        x=np.linspace(.05,.95,300)
        y=(x>.45).astype(int)
        a=pd.DataFrame({'score':x,'member':y})
        model=cp.fit_calibrator(a,'platt')
        self.assertTrue(np.all(np.diff(model(x))>0))

    def test_isotonic_tie_break_preserves_original_score_order(self):
        q=pd.DataFrame({'sex':[0]*18+[1]*18, 'challenge_row_id':list(range(36))})
        rng=np.random.default_rng(9);raw=rng.uniform(0,1,36)
        iso=np.round(raw*2)/2  # artificially introduce ties in monotonic transform
        s=cp.top75(q,iso,secondary=raw)
        old=cp.top75(q,raw)
        np.testing.assert_array_equal(s,old)

    def test_action_optimizer_extremes(self):
        p=np.array([.99,.01])
        out,_=cp.optimise_actions(p,.9,.9,.005,.005,lambda0=.8)
        self.assertEqual(out[0],9.5)
        self.assertIn(out[1],(0.,11.))

    def test_safe_optimizer_maintains_unselected_zero(self):
        p=np.array([.99,.8,.05,.1])
        sel=np.array([True,False,True,False]);out,_=cp.optimise_actions(p,.8,.85,.01,.01,lambda0=.8,coverbase=sel,margin=.02)
        np.testing.assert_array_equal(out[~sel],np.zeros(2))

    def test_real_score_consistency_with_older_module(self):
        data=Path(os.environ.get('AIA_ROOT','/mnt/data/aia_density_ratio_20261009/inputs'))
        prior_dir=Path(os.environ.get('AIA_PRIOR','/mnt/data/aia_density_ratio_20261009/prior'))
        if not (data/'公式データ/予備戦').is_dir() or not (prior_dir/'experiment.py').is_file():
            self.skipTest('Set AIA_ROOT and AIA_PRIOR to run scorer concordance check')
        lib=cp.import_prior(prior_dir)
        _,D=lib.read_data(data)
        for k in (1,5,13,23):
            t=D[k][2];p=np.linspace(0,11,326)
            a=cp.score_metrics(p,t);b=lib.metrics(p,t)
            for key in ['r_all','r_rare','r_max']:
                self.assertAlmostEqual(a[key],b[key],places=12)
            self.assertEqual(a['hit_m'],b['hit_member'])
            self.assertEqual(a['hit_c'],b['hit_control'])

    def test_inner_oof_target_truth_not_used(self):
        orig=list(cp.TEAMS)
        try:
            cp.TEAMS=[1,2,3,4,5]
            rng=np.random.default_rng(124)
            features={k:pd.DataFrame({'x':rng.normal(size=326),'row_id':np.arange(326),
                    'sex':np.tile([0,1],163)}) for k in cp.TEAMS}
            truth={k:pd.DataFrame({'label':['member']*163+['control']*163,
                    'time':np.full(326,9.5),'is_rare':np.r_[np.ones(13),np.zeros(150),np.ones(13),np.zeros(150)]}) for k in cp.TEAMS}
            a=cp.inner_oof(features,truth,held_out=3,cols=['x'],folds=4)
            self.assertNotIn(3,a.team.values)
            truth[3]['label']=truth[3].label[::-1].to_numpy()
            truth[3]['time']=0.
            b=cp.inner_oof(features,truth,held_out=3,cols=['x'],folds=4)
            np.testing.assert_allclose(a.score,b.score,rtol=0,atol=0)
        finally:cp.TEAMS=orig

if __name__=='__main__':unittest.main()
