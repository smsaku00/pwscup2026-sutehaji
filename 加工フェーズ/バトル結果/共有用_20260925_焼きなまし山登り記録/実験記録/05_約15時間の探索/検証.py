"""Behavioral tests for time allocation, continuing after plateaus, and resume boundaries."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import json,unittest
import 時間探索 as t

class SchedulerTests(unittest.TestCase):
    def test_budget_does_not_reset_on_resume(self):
        state={'started_epoch':1000.}
        p={'planned_hours':15,'validation_reserve_minutes':30,'base_order':['Zhang11','Zhang04'],
           'search_shares':{'Zhang11':2,'Zhang04':1}}
        def progress(state,**kw):state.update(kw)
        t.schedule_deadlines(p,state,progress,now=1060.)
        self.assertEqual(state['target_end_epoch'],55000.)
        self.assertEqual(state['base_deadlines']['Zhang04'],53200.)
        self.assertAlmostEqual(state['base_deadlines']['Zhang11'],35820.)
        old=dict(state['base_deadlines']);t.schedule_deadlines(p,state,progress,now=4000.)
        self.assertEqual(old,state['base_deadlines'])

    def exercise(self,limit=5,advance=1,pending=False):
        temp=TemporaryDirectory();root=Path(temp.name);self.addCleanup(temp.cleanup)
        seen_file=root/'previous.json';t.save(seen_file,['old-key'])
        baseline=dict(id='original',base='Test',key='base',ops=[],total_swaps=0,max_AIA=.2,
            utility={'utility':.9},path='not-read-in-this-test')
        state={'base_deadlines':{'Test':limit}}
        p={'base_order':['Test'],'previous_keys':{'Test':str(seen_file)},'seed_parents':{'Test':[]},'beam_width':3,'max_swaps':10}
        clock=[0.];runs=[]
        def progress(s,**kw):s.update(kw)
        def run_batch(pool,jobs,s,stage,base):
            runs.append(list(jobs));clock[0]+=advance
            return [{'id':j['id'],'key':j['key'],'complete':False,'attack_calls':1} for j in jobs]
        ctx=SimpleNamespace(HERE=root,progress=progress,run_batch=run_batch,
            diverse_best=lambda rows,n:sorted(rows,key=lambda r:r['max_AIA'])[:n])
        def factory(ctx,p,base,parents,round_no,floor,seen,stats):
            self.assertIn('old-key',seen)
            key=f'new-{round_no}';self.assertNotIn(key,seen);seen.add(key)
            return [dict(id=key,key=key,ops=[{'attribute':'age'}])],[]
        if pending:
            jp=root/'search/Test/R00001/候補計画.json'
            t.save(jp,[dict(id='pending',key='pending-key',ops=[{'attribute':'age'}])]);t.save(jp.with_name('生成記録.json'),[])
            clock[0]=limit+10
        r=t.continue_search(ctx,p,None,'Test',baseline,state,wall=lambda:clock[0],jobs_factory=factory)
        return r,runs,clock[0]

    def test_plateau_continues_beyond_three_rounds(self):
        r,runs,now=self.exercise(limit=7)
        self.assertEqual(len(runs),7);self.assertEqual(r['new_candidates'],7)
        self.assertEqual(now,7);self.assertIn('時間',r['reason'])

    def test_batch_finishes_after_deadline(self):
        r,runs,now=self.exercise(limit=5,advance=3)
        self.assertEqual(len(runs),2);self.assertEqual(now,6)

    def test_pending_batch_resumes_even_after_deadline(self):
        r,runs,now=self.exercise(limit=5,pending=True)
        self.assertEqual(len(runs),1);self.assertEqual(runs[0][0]['id'],'pending')
        self.assertEqual(r['new_candidates'],1)

    def test_ten_swap_path_does_not_stop_other_paths(self):
        archive=[dict(id='root',key='r',total_swaps=0,max_AIA=.2),
            dict(id='best10',key='10',total_swaps=10,max_AIA=.01),
            dict(id='best9',key='9',total_swaps=9,max_AIA=.03),
            dict(id='less_seen',key='1',total_swaps=1,max_AIA=.1)]
        selected=t.select_parents(archive,{'r':10,'9':20,'1':0},10)
        self.assertEqual([r['id'] for r in selected],['root','best9','less_seen'])

if __name__=='__main__':unittest.main()
