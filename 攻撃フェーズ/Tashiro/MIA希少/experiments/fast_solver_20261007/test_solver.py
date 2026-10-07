import itertools
import random
import unittest
import numpy as np
import reference
from model import *
from solver import *
from verify import verify, response, generate_states, state_hash, verify_certificate

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = generate(30)
        cls.parts = partition(cls.s,Q4_30)

    def test_counts_and_overflow(self):
        self.assertEqual(len(self.s),40428)
        self.assertEqual(len(self.parts),213)
        self.assertEqual(len(self.parts[273]),662)
        self.assertEqual(max(map(len,self.parts.values())),1077)
        fullq=[2047]*6
        self.assertTrue(np.all(codes(self.s,fullq)==637))
        for row in self.s[::791]: self.assertEqual(int(codes(row[None,:],fullq)[0]),response(row,fullq))

    def test_fiber_independent_oracle(self):
        # All nonempty edge subsets on four vertices, including triangles,
        # stars and rectangles. The ref enumerates all 1024 masks separately.
        es = [sum(1<<i for i in e) for e in itertools.combinations(range(4),2)]
        for subset in range(1,1<<6):
            f=tuple(es[i] for i in range(6) if subset>>i&1)
            self.assertEqual(fiber_valid(f),reference.valid(f),f)
        self.assertEqual(fiber_valid(tuple(es)),0)  # K4 needs >=3 tests
        self.assertFalse(scalar_one((3,5,6)))  # triangle is not one-test separable
        self.assertFalse(scalar_one((3,5,9)))  # three-edge star also not
        self.assertTrue(scalar_one((3,6,12)))

    def test_last_query_crosscheck(self):
        rng=np.random.default_rng(331)
        ss=Solver()
        for n in (2,3,4,6,9,12):
            for _ in range(4):
                parent=self.parts[int(rng.choice(list(self.parts)))]
                sample=parent[rng.choice(len(parent),min(n,len(parent)),replace=False)]
                ref=reference.last_query(sample)
                got=ss.last_query(sample)
                self.assertEqual(got.status, Status.SAT if ref is not None else Status.UNSAT)
                if got.status==Status.SAT: verify(sample,got.tree,1)

    def test_two_steps_and_unknown(self):
        # Small exact generic AND/OR problems must give checked witnesses.
        x=self.parts[273][:6]
        r=Solver(exhaustive_product_limit=20000).solve(x,2,seconds=3)
        if r.status==Status.SAT: verify(x,r.tree,2)
        self.assertNotEqual(r.status,Status.UNKNOWN)
        # Deadline must not be called an impossibility proof or enter negative cache.
        engine=Solver(); r=engine.solve(self.s,3,seconds=0)
        self.assertEqual(r.status,Status.UNKNOWN)
        self.assertFalse(engine.cache)
        # Time/node exhaustion at depth one also returns UNKNOWN.
        engine=Solver(node_limit=0)
        self.assertEqual(engine.last_query(x).status,Status.UNKNOWN)
        self.assertFalse(engine.cache)

    def test_baseline_bad_branch(self):
        b=self.parts[273]
        self.assertEqual(depth2_domains(b),reference.domains(b))
        self.assertEqual(Solver().solve(b,2).status,Status.UNSAT)
        e=ParentScreen(self.s)
        failed,c=e.evaluate(Q4_30)
        self.assertEqual(failed,[(273,5,662)])

    def test_complements_and_row_cache(self):
        engine=Solver(); b=self.parts[273][:9]
        ans=engine.last_query(b)
        hits=engine.stats['cache_hits']
        again=engine.last_query(b[::-1])
        self.assertEqual(ans.status,again.status)
        self.assertEqual(engine.stats['cache_hits'],hits+1)
        q=(2,9,77,125,511,729)
        x=codes(b,q); y=codes(b,[FULL^z for z in q])
        self.assertTrue(np.all(x[:,None]==x)==np.all(y[:,None]==y))
        self.assertTrue(np.array_equal(x[:,None]==x, y[:,None]==y))

    def test_automorphisms_checked(self):
        autos=verified_q3_automorphisms()
        self.assertEqual(len(autos),144)
        rng=np.random.default_rng(7)
        sample=self.s[rng.choice(len(self.s),300,replace=False)]
        for pm,tm in autos:
            lut=np.array([sum(1<<pm[i] for i in range(N) if m>>i&1)
                          for m in range(1<<N)],dtype=np.uint16)
            dest=np.zeros_like(sample)
            for t in range(6): dest[:,tm[t]]=lut[sample[:,t]]
            self.assertTrue(np.all(codes(dest,Q3)==transform_code(30,tm)))

    def test_failure_core_and_scope(self):
        e=ParentScreen(self.s);e.evaluate(Q4_30,early=True)
        self.assertEqual(len(e.learned),1)
        core=e.learned[0]
        self.assertEqual(reference.domains(core['states'])[5],0)
        self.assertIsNotNone(e.failure_cache.rejects(Q4_30,2,self.s))
        self.assertIsNone(e.failure_cache.rejects(Q4_30,1,self.s))
        # A core from a different node must never reject a smaller parent.
        self.assertIsNone(e.failure_cache.rejects(Q4_30,2,core['states'][:1]))
        before=len(e.failure_cache.entries)
        e.failure_cache.add(self.s,2,Result(Status.UNKNOWN))
        e.failure_cache.add(self.s,2,Result(Status.SAT))
        self.assertEqual(len(e.failure_cache.entries),before)

    def test_independent_state_generation_and_hash(self):
        independent=generate_states(30)
        self.assertEqual(set(independent),set(map(tuple,self.s.tolist())))
        self.assertEqual(state_hash(independent),fingerprint(self.s))
        b=self.parts[621];engine=Solver();ans=engine.solve(b,2)
        data={'q3_code':30,'q4_query':list(Q4_30),'q4_outcome':621,
              'state_hash':fingerprint(b),'depth':2,'tree':ans.tree}
        self.assertEqual(verify_certificate(data)['verified_states'],len(b))
        data['state_hash']='bad'
        with self.assertRaises(ValueError):verify_certificate(data)
        with self.assertRaises(ValueError):verify(b,{'terminal':True,'query':[0]*6},1)

    def test_sample_generator_unknown_is_not_cached(self):
        b=self.parts[273][:12]
        e=Solver(sample_queries=16,query_budget=1)
        ans=e.solve(b,2,seconds=5)
        if ans.status==Status.SAT:verify(b,ans.tree,2)
        if ans.status==Status.UNKNOWN:
            self.assertNotIn((state_key(b),2),e.cache)

    def test_transport_full_state_sets_and_small_witness(self):
        from complete_witness import transform_rows,transform_tree,check_partitions
        for target in (246,270):
            pm,tm=next((pm,tm) for pm,tm in verified_q3_automorphisms() if transform_code(30,tm)==target)
            dest=transform_rows(self.s,pm,tm)
            self.assertEqual(set(map(tuple,dest.tolist())),set(generate_states(target)))
            b=self.parts[621];ans=Solver().solve(b,2)
            moved=transform_rows(b,pm,tm);tree=transform_tree(ans.tree,pm,tm)
            check_partitions(b.tolist(),moved.tolist(),ans.tree,tree,tm)
            self.assertEqual(verify(moved,tree,2),len(b))

if __name__=='__main__': unittest.main(verbosity=2)
