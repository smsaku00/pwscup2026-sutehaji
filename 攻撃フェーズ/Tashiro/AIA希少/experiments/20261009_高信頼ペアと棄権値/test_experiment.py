"""Unit tests; optional direct comparison with the archived official scorer."""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import experiment as ex


def fixture_truth() -> pd.DataFrame:
    return pd.DataFrame({'challenge_row_id':np.arange(326),
        'label':['member']*163+['control']*163,
        'time':np.full(326,9.5),'onset':np.zeros(326,int),
        'is_rare':np.r_[np.ones(13,int),np.zeros(150,int),np.ones(13,int),np.zeros(150,int)]})


def fixture_public() -> pd.DataFrame:
    q=pd.DataFrame({'challenge_row_id':np.arange(326),'age':np.tile(np.arange(40,75),10)[:326],
        'sex':np.arange(326)%2,'prefecture':['OTHER']*326,
        'SBP':np.linspace(70,180,326),'TG':np.linspace(5,500,326),
        'HDL':np.linspace(25,80,326),'ALT':np.linspace(1,100,326)})
    q.loc[:1,['age','sex','prefecture']]=[52,0,'PAIR']
    return q[['challenge_row_id']+ex.QI]


class TestExperiment(unittest.TestCase):
    def test_score_all_196_rare_counts(self):
        t=fixture_truth()
        for hm in range(14):
            for hc in range(14):
                p=np.full(326,11.);p[:hm]=9.5;p[163:163+hc]=9.5
                s=ex.score_prediction(p,t)
                m,c=hm/13,hc/13
                se=np.sqrt((m*(1-m)+c*(1-c))/13)
                expected=np.clip((m-c-1.64*se)/max(1-c,1e-9),0,1)
                self.assertAlmostEqual(s['r_rare'],expected,places=14)
                self.assertEqual(s['hit_m_rare'],hm)
                self.assertEqual(s['hit_c_rare'],hc)
                self.assertAlmostEqual(s['r_max'],max(s['r_all'],s['r_rare']))

    def test_unknown_control_rare_is_not_imputed(self):
        t=fixture_truth().rename(columns={'is_rare':'member_true_rare_control_unknown'})
        t.loc[t.label.eq('control'),'member_true_rare_control_unknown']=-1
        s=ex.score_prediction(np.full(326,9.5),t)
        self.assertIsNone(s['r_rare']); self.assertIsNone(s['r_max'])
        self.assertIsNone(s['hit_c_rare']);self.assertEqual(s['hit_m_rare'],13)

    def test_zero_vs_eleven_boundary(self):
        t=fixture_truth(); t.loc[:3,'time']=[0,.5,.5000001,10]
        z=ex.score_prediction(np.zeros(326),t)
        e=ex.score_prediction(np.full(326,11.),t)
        self.assertEqual(z['hit_m'],2);self.assertEqual(e['hit_m'],0);self.assertEqual(e['hit_c'],0)

    def test_reject_invalid_predictions_and_truth(self):
        t=fixture_truth()
        for p in [np.zeros(325),np.full(326,-1.),np.full(326,np.nan)]:
            with self.assertRaises(ValueError):ex.score_prediction(p,t)
        t.loc[0,'label']='unknown'
        with self.assertRaises(ValueError):ex.score_prediction(np.zeros(326),t)

    def test_inference_rejects_hidden_columns(self):
        q=fixture_public(); scores={m:np.arange(326,dtype=float) for m in ex.METHODS}
        with self.assertRaises(ValueError):ex.predict_masks(q.assign(label='member'),scores,np.linspace(0,1,326),5)
        with self.assertRaises(ValueError):ex.predict_masks(q.assign(is_rare=0),scores,np.linspace(0,1,326),5)
        with self.assertRaises(ValueError):ex.predict_masks(q.assign(pair_id=0),scores,np.linspace(0,1,326),5)

    def test_permutation_and_tie_invariance(self):
        q=fixture_public(); rng=np.random.default_rng(14)
        s={m:rng.integers(0,12,326).astype(float) for m in ex.METHODS}
        prob=rng.integers(0,10,326).astype(float)/10
        base,_=ex.predict_masks(q,s,prob,5)
        perm=rng.permutation(326)
        result,_=ex.predict_masks(q.iloc[perm].reset_index(drop=True),{m:x[perm] for m,x in s.items()},prob[perm],5)
        inv=np.argsort(perm)
        for key in base: np.testing.assert_array_equal(base[key],result[key][inv])

    def test_rejected_pair_preserves_baseline(self):
        q=fixture_public();s={m:np.zeros(326) for m in ex.METHODS}
        masks,a=ex.predict_masks(q,s,np.linspace(0,1,326),5)
        self.assertTrue(len(a)>0);self.assertFalse(any(x['passed'] for x in a))
        for m in ex.METHODS:
            for p in ['visible_pair','stable_rare_pair','combined_pair']:
                np.testing.assert_array_equal(masks[m,'baseline'],masks[m,p])

    def test_gap_can_accept_pair_without_knowing_roles(self):
        q=fixture_public();x=np.linspace(0,1,326);x[0]=2.;x[1]=-1.
        scores={m:x.copy() for m in ex.METHODS};prob=np.linspace(0,1,326)
        masks,a=ex.predict_masks(q,scores,prob,5)
        self.assertTrue(any(z['passed'] and z['source']=='visible' for z in a))
        for m in ex.METHODS:
            self.assertTrue(masks[m,'combined_pair'][0]);self.assertFalse(masks[m,'combined_pair'][1])
        # No assertion that these are actually one member and one control:
        # pair composition is NOT established by this inference function.

    def test_original_model_parameters(self):
        p=ex.rare_model().get_params()
        for k,v in {'max_iter':140,'max_leaf_nodes':15,'min_samples_leaf':35,
                    'l2_regularization':5,'random_state':20261007}.items():
            self.assertEqual(p[k],v)
        self.assertEqual(ex.rare_features(fixture_public()).shape,(326,6))

    def test_official_loader_with_synthetic_fixtures_only(self):
        # This checks file discovery and fold isolation, NOT official performance.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dist=root/'公式データ/予備戦'
            attack=dist/'PWSCup2026_prelim_attack';attack.mkdir(parents=True)
            td=dist/'結果/正解データ';td.mkdir(parents=True)
            common=dist/'PWSCup2026_prelim_common';common.mkdir()
            rng=np.random.default_rng(98)
            c=pd.concat([fixture_public().drop(columns='challenge_row_id')]*4,ignore_index=True).iloc[:1049].copy()
            c['time']=9.5;c['onset']=0
            c.to_csv(common/'A_bg.csv',index=False)
            for team in [1,2,3,4,5,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]:
                q=fixture_public();q['SBP']+=rng.normal(0,.3,len(q))
                q.to_csv(attack/f'aia_challenge_{team}.csv',index=False)
                c.to_csv(attack/f'C_{team}.csv',index=False)
                fixture_truth().to_csv(td/f'aia_truth_{team}.csv',index=False)
            cases,bg,meta=ex.official_inputs(root)
            self.assertEqual(len(cases),21);self.assertEqual(len(meta['folds']),7)
            tested=[]
            for f in meta['folds']:
                self.assertFalse(set(f['train']) & set(f['test']))
                tested.extend(f['test'])
            self.assertEqual(len(set(tested)),21)
            for c in cases:
                p=meta['probabilities'][c.name]
                self.assertEqual(len(p),326);self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((p>=0)&(p<=1)).all())

    @unittest.skipUnless(os.environ.get('PWS_OFFICIAL_SRC'),'Set PWS_OFFICIAL_SRC for official scorer comparison')
    def test_against_archived_official_function(self):
        sys.path.insert(0,os.environ['PWS_OFFICIAL_SRC'])
        official=importlib.import_module('pwscup2026.scoring.attack_score')
        t=fixture_truth(); m=t.label.eq('member').to_numpy(); rare=t.is_rare.to_numpy(bool)
        cfg={'scoring':{'low_fpr':.01,'aia_rare_se_z':1.64},'aia':{'tau_time':.5}}
        rng=np.random.default_rng(73)
        pairs=[]
        for hm in range(14):
            for hc in range(14):
                p=np.full(326,11.);p[:hm]=9.5;p[163:163+hc]=9.5;pairs.append(p)
        pairs.extend(np.where(rng.random((100,326))<.7,9.5,11.))
        for p in pairs:
            a=ex.score_prediction(p,t)
            b=official.score_aia(None,p[m],t.onset.to_numpy()[m],t.time.to_numpy()[m],
                None,p[~m],t.onset.to_numpy()[~m],t.time.to_numpy()[~m],cfg,
                member_rare_mask=rare[m],control_rare_mask=rare[~m])
            self.assertAlmostEqual(a['r_all'],b.breakdown['r_time_all'],places=13)
            self.assertAlmostEqual(a['r_rare'],b.breakdown['r_time_rare'],places=13)
            self.assertAlmostEqual(a['r_max'],b.score,places=13)

if __name__=='__main__':unittest.main(verbosity=2)
