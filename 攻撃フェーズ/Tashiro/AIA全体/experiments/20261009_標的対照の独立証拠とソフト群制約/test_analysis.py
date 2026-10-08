from pathlib import Path
import hashlib,json,unittest,os
import numpy as np,pandas as pd
from experiment import clipping_r,rrare,pairs,top75
from extension import project

BASE=Path(__file__).parent
DATA=Path(os.environ.get("AIA_DATA_ROOT",str(BASE/"input")))
OUT=Path(os.environ.get("AIA_OUTPUT_DIR",str(BASE/"output")))
class Audit(unittest.TestCase):
 def test_metric(self):
  self.assertAlmostEqual(clipping_r(.8,.2),.75)
  self.assertEqual(clipping_r(.2,.8),0)
  self.assertAlmostEqual(rrare(1.,0.),1.)
 def test_quota_and_order(self):
  q=pd.DataFrame({'age':[40,41,42,42,52,53],'sex':[1,1,1,1,0,0],'prefecture':['A']*4+['B']*2,'challenge_row_id':list(range(6))})
  p=np.array([.05,.6,.2,.8,.3,.6])
  self.assertTrue(np.allclose(project(q,p,0),p))
  adj=project(q,p,1)
  for inds in pairs(q,5).values():
   self.assertAlmostEqual(float(adj[inds].sum()),len(inds)/2,places=10)
   self.assertTrue(all((p[a]-p[b])*(adj[a]-adj[b])>=0 for a in inds for b in inds))
 def test_public_predictions_no_truth(self):
  p=OUT/'public_scores.csv';self.assertTrue(p.exists());f=pd.read_csv(p)
  self.assertEqual(len(f),6846)
  self.assertEqual(f.team.nunique(),21)
  self.assertFalse({'time','label','onset','is_rare'} & set(f.columns))
 def test_sha(self):
  m=DATA/'input_sha256.json'
  if m.exists():j=json.loads(m.read_text())
  else:
   official=DATA/'公式データ/予備戦'
   allhash={p:h for h,p in (line.split(maxsplit=1) for line in (official/'SHA256SUMS.txt').read_text().splitlines() if line.strip())}
   j={str((official/p).relative_to(DATA)):h for p,h in allhash.items() if p.endswith('.csv') and (p.startswith('PWSCup2026_prelim_attack/') or p.startswith('結果/正解データ/aia_truth_'))}
   for p in ['PWSCup2026_prelim_common/A_bg.csv','PWSCup2026_prelim_common/schema.json']:
    j[str((official/p).relative_to(DATA))]=allhash[p]
  self.assertEqual(len(j),65)
  for p,s in j.items():self.assertEqual(hashlib.sha256((DATA/p).read_bytes()).hexdigest(),s)
 def test_baseline_saved(self):
  a=pd.read_csv(OUT/'summary_20.csv');m=a.query("policy=='base_0'").set_index('scheme').r_all
  for name,expected in [('near4',.130833),('col4',.158452),('col2',.160199)]:
   self.assertAlmostEqual(float(m[name]),expected,places=5)

if __name__=='__main__':unittest.main()
