"""独立した密度再計算と、順位・非会員処理・公式形式検査を検証。"""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import KernelDensity

from gen_lra import (NUM7, CATS, K, RHO, mixed_scores, numeric, ordinal_core,
                     ordinal_matrices, silverman, split_confidence)


def example(n, seed):
    rng=np.random.default_rng(seed)
    d=pd.DataFrame({c:rng.uniform(1,10,n) for c in NUM7})
    d['sex']=np.arange(n)%2
    d['prefecture']=np.resize(['A','B','C'],n)
    return d


class GenTests(unittest.TestCase):
    def test_mixed_matches_explicit_augmented_density(self):
        r,b,c=example(47,1),example(7,2),example(25,3)
        fast=mixed_scores(r,b,c,NUM7)
        rn=numeric(r,NUM7)
        scale=(np.percentile(rn,75,axis=0)-np.percentile(rn,25,axis=0))/1.349
        rn,bn,cn=rn/scale,numeric(b,NUM7)/scale,numeric(c,NUM7)/scale
        h=silverman(len(r),len(NUM7))
        def kernel(s, point, srow, prow):
            value=np.exp(-np.sum((s-point)**2)/(2*h*h))
            for col in CATS:
                frequency=(int((r[col]==srow[col]).sum())+1)/(len(r)+r[col].nunique())
                value*=RHO*(srow[col]==prow[col])+(1-RHO)*frequency
            return value
        expected=[]
        for i in range(len(b)):
            distances=np.sum((cn-bn[i])**2,axis=1)
            for col in CATS:
                distances+=(c[col].to_numpy()!=b.iloc[i][col])
            nearest=np.argsort(distances,kind='stable')[:K]
            score=0.
            for j in nearest:
                values=[kernel(cn[j],rn[k],c.iloc[j],r.iloc[k]) for k in range(len(r))]
                before=np.mean(values)
                after=np.mean(values+[kernel(cn[j],bn[i],c.iloc[j],b.iloc[i])])
                score+=np.log(after)-np.log(before)
            expected.append(score)
        np.testing.assert_allclose(fast,expected,rtol=0,atol=1e-12)

    def test_mixed_category_names_do_not_create_distance(self):
        r,b,c=example(47,1),example(7,2),example(25,3)
        old=mixed_scores(r,b,c,NUM7)
        for frame in [r,b,c]:
            frame['prefecture']=frame.prefecture.map({'A':'Z','B':'A','C':'Q'})
        np.testing.assert_allclose(old,mixed_scores(r,b,c,NUM7),rtol=0,atol=1e-12)

    def test_ordinal_matches_independent_kde_refits(self):
        rng=np.random.default_rng(99)
        r,b,c=rng.normal(size=(37,3)),rng.normal(size=(5,3)),rng.normal(size=(23,3))
        h,k=.8,4
        actual=ordinal_core(r,b,c,h,k,batch_size=3)
        reference=KernelDensity(bandwidth=h).fit(r)
        expected=[]
        for x in b:
            local=c[np.argsort(np.sum((c-x)**2,axis=1))[:k]]
            aug=KernelDensity(bandwidth=h).fit(np.vstack([r,x]))
            expected.append(np.sum(aug.score_samples(local)-reference.score_samples(local)))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)

    def test_split_keeps_ties_and_known_negatives(self):
        raw=np.array([10.,10.,4.,8.,8.,2.,9.])
        rare=np.array([0,0,0,1,1,1,1],bool)
        private=np.array([0,0,1,0,0,0,1],bool)
        conf=split_confidence(raw,raw,rare,private)
        self.assertEqual(conf[0],conf[1])
        self.assertEqual(conf[3],conf[4])
        self.assertTrue(np.all(conf[private]==0))
        self.assertGreater(conf[~rare&~private].min(),conf[rare].max())
        self.assertGreater(conf[3],conf[5])
        self.assertTrue(np.all((conf>=0)&(conf<=1)))

    def test_schema_rejects_missing_other_team(self):
        from 分析 import validator
        d=pd.DataFrame({'record_id':[0,1],'0':[.1,.2],'1':[.3,.4]})
        ref={'mia_columns':{'target_columns':['0','1','28']},'team_id':28,'attacker_record_ids':{0,1}}
        self.assertTrue(validator.validate_mia_submission(d,{},ref).ok)
        self.assertFalse(validator.validate_mia_submission(d.drop(columns='1'),{},ref).ok)


if __name__=='__main__':
    unittest.main()
