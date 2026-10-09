"""Independent checks on the published candidate sweep and its fixed limited-change rule."""
import unittest
import json
from pathlib import Path
import numpy as np,pandas as pd
import model_sweep as ms
from nested_selection import folds,CANDS
HERE=Path(__file__).resolve().parent
R=HERE/'model_results'

class ModelBreadthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.per=pd.read_csv(R/'per_team.csv')
        cls.extra=pd.read_csv(R/'per_team_extra.csv')
        cls.nested=pd.read_csv(R/'nested_per_team.csv')
        cls.limited=pd.read_csv(R/'limited_model_per_team.csv')
    def test_inputs_and_rows(self):
        self.assertEqual(len(json.loads((HERE/'official_manifest.json').read_text())),65)
        self.assertEqual(len(self.per),len(ms.TEAMS)*len(ms.MODES)*len(ms.REPR))
        self.assertEqual(len(self.extra),len(ms.TEAMS)*4*len(ms.REPR))
        self.assertEqual(self.per.team.nunique(),21)
    def test_no_secret_columns_in_features(self):
        f=pd.read_pickle(HERE/'cache/features_1.pkl')
        for c in ['label','onset','time','death','is_rare']:
            self.assertNotIn(c,f.columns)
    def test_source_baseline_reproduces(self):
        s=pd.read_csv(R/'summary.csv')
        a=s[(s.scope=='nonself20')&(s.features=='c_bg')&(s.model=='lr_c004')].iloc[0]
        b=s[(s.scope=='nonself20')&(s.features=='c_bg_local')&(s.model=='lr_c004')].iloc[0]
        self.assertAlmostEqual(a.r_all,.18886686005164305,places=11)
        self.assertAlmostEqual(b.r_all,.1913993598566548,places=10)
    def test_nested_folds_disjoint(self):
        for n,keys in [(7,ms.TEAMS),(6,[i for i in ms.TEAMS if i not in (1,2,5)])]:
            tests=[]
            for tr,te in folds(keys,n):
                self.assertFalse(set(tr)&set(te))
                tests+=te
            self.assertEqual(sorted(tests),sorted(keys))
        self.assertEqual(len(CANDS),14)
    def test_nested_selection_training_test_disjoint(self):
        d=pd.read_csv(R/'nested_selected_models.csv')
        for _,r in d.iterrows():
            a=set(map(int,str(r.selector_training_teams).split(';')))
            b=set(map(int,str(r.selector_test_teams).split(';')))
            self.assertFalse(a&b)
        self.assertEqual(len(self.nested),78)
    def test_r_score_constraints(self):
        for d in [self.per,self.extra,self.nested,self.limited]:
            self.assertTrue((d.r_all.between(0,1)).all())
            self.assertTrue((d.r_rare.between(0,1)).all())
            self.assertTrue((d.r_max+1e-12>=d.r_all).all())
            self.assertTrue((d.r_max+1e-12>=d.r_rare).all())
    def test_limited_policy_cap_and_shared_incumbent(self):
        self.assertEqual(len(self.limited),21*6)
        self.assertTrue(self.limited.changed.between(0,10).all())
        self.assertTrue(self.limited.groupby('team').baseline_r_all.nunique().eq(1).all())
        self.assertTrue(self.limited.groupby('team').size().eq(6).all())
    def test_limited_baseline_reproduction(self):
        s=pd.read_csv(R/'limited_model_summary.csv')
        v=s[(s.scope=='nonself20')&(s.decision_model=='reference')].iloc[0]
        self.assertAlmostEqual(v.r_all,0.195220690645,places=5)
        self.assertAlmostEqual(v.changed,1.5,places=8)
    def test_summary_vs_raw(self):
        s=pd.read_csv(R/'summary.csv')
        for f,m in [('c_bg','lr_c004'),('c_bg_local','lr_c02'),('c_bg','hgb_shallow')]:
            actual=self.per[(self.per.features==f)&(self.per.model==m)&self.per.team.ne(5)].r_all.mean()
            row=s[(s.scope=='nonself20')&(s.features==f)&(s.model==m)].iloc[0]
            self.assertAlmostEqual(actual,row.r_all,places=12)
    def test_density_reference_inputs(self):
        d=pd.read_csv(HERE/'density/results/density_public_scores.csv.gz')
        self.assertEqual(len(d),6846)
        self.assertEqual(len(d.drop_duplicates(['team','row_id'])),6846)
        self.assertTrue(np.isfinite(d[['knn2_k1','kde4_half']].to_numpy()).all())
if __name__=='__main__': unittest.main(verbosity=2)
