"""Standalone unit checks for the published preliminary cross-team follow-up."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run_followup',HERE/'run_followup.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class DummyCommon:
    def scales(self,a,cols): return np.ones(len(cols))
    def numbers(self,a,cols): return a[list(cols)].to_numpy(dtype=float)
class DummyPrevious:
    @staticmethod
    def ranks(x):return np.column_stack([rankdata(v,method='average')/len(v) for v in x.T])
    @staticmethod
    def rivals(x,stat):return np.column_stack([stat(np.delete(x,j,axis=1),axis=1) for j in range(x.shape[1])])

class TestFollowup(unittest.TestCase):
 def frame(self, sex, pref, age):
    # distance contribution entirely from age in this synthetic unit example.
    return pd.DataFrame({'age':[age], 'SBP':[0.], 'TG':[1.], 'HDL':[0.], 'ALT':[1.], 'sex':[sex], 'prefecture':[pref]})
 def test_category_or_penalty(self):
    b=self.frame(0,'A',3.);a=self.frame(0,'A',3.)
    # C0 has one mismatch, C1 two; each has same 0.25 penalty, not 0.50.
    cs={1:pd.concat([self.frame(1,'A',3.)]*2,ignore_index=True),
        2:pd.concat([self.frame(1,'B',3.)]*2,ignore_index=True)}
    feats=m.public_neighbours(DummyCommon(),b,a,cs,[1,2])
    np.testing.assert_allclose(feats['num5_category025'], [[.25,.25]],rtol=0,atol=0)
 def test_nearest_two_means_squares(self):
    b=self.frame(0,'A',1.)
    c=pd.concat([self.frame(0,'A',2.),self.frame(0,'A',4.)],ignore_index=True)
    f=m.public_neighbours(DummyCommon(),b,b,{1:c},[1])
    np.testing.assert_allclose(f['num5_nearest'],[[1.]])
    np.testing.assert_allclose(f['num5_two'], [[5.]])
 def test_exclusive_comparison_algebra(self):
    rng=np.random.default_rng(42)
    cal=rng.random((8,3));normal=np.array([1,1,0,1,1,0,1,1],dtype=bool)
    public={k:rng.random((8,3)) for k in ['num5_nearest','num5_two','num5_category025']}
    result=m.score_methods(DummyPrevious(),cal,public,normal,[0,1,2])
    self.assertEqual(set(result),set(m.METHODS))
    self.assertEqual(result['P01'].shape,(6,3))
    r=DummyPrevious.ranks(cal[normal]);p=r-.1*DummyPrevious.rivals(r,np.median)
    np.testing.assert_allclose(result['P01'],p,atol=1e-15)
    ff=DummyPrevious.ranks(-public['num5_nearest'][normal]);comp=DummyPrevious.rivals(ff,np.mean)
    np.testing.assert_allclose(result['nearest5_mean0.1'],p+.1*(ff-comp),atol=1e-15)
 def test_overlap(self):
    data=np.stack([np.arange(30)[::-1],np.arange(30)[::-1]],axis=1).astype(float)
    self.assertEqual(m.overlap_top10(data), {'duplicate_records':10,'extra_top10_slots':10})
    data[:,1]*=-1
    self.assertEqual(m.overlap_top10(data), {'duplicate_records':0,'extra_top10_slots':0})
 def test_hash_guard_constant(self):
    self.assertEqual(len(m.PREVIOUS_SHA),64)
 def test_no_tiebreak_by_identity(self):
    # Duplicate scores remain duplicated and are not broken by record_id in candidate construction.
    cal=np.ones((6,3));normal=np.array([1,1,1,1,1,1],dtype=bool)
    p={k:np.ones((6,3)) for k in ['num5_nearest','num5_two','num5_category025']}
    s=m.score_methods(DummyPrevious(),cal,p,normal,[0,1,2])
    for mat in s.values():
        self.assertTrue(np.isfinite(mat).all())
        np.testing.assert_allclose(mat,np.broadcast_to(mat[:1],mat.shape))

if __name__=='__main__':unittest.main()
