"""Report historical numerical drift without changing predictions or fit rules."""
from pathlib import Path
import json
import pandas as pd
from run_family import STAGE,dump

def main():
 for p in STAGE.glob('*/prelim/実行記録.json'):
  folder=p.parent
  if not (folder/'completion.json').exists() or not (folder/'catalog_regression.json').exists():continue
  rec=json.loads(p.read_text());checks=json.loads((folder/'catalog_regression.json').read_text())
  if not checks:continue
  converged=[]
  for q in folder.rglob('最適化の収束.csv'):
   x=pd.read_csv(q);converged.append(dict(table=q.relative_to(folder).as_posix(),original_optimizations=len(x),successful=int(x.success.sum()),iteration_limit_hits=int((x.iterations==200).sum()),max_iter=200,failed_outputs_kept=True))
  d=dict(family=rec.get('family'),background='prelim',comparison_count=len(checks),matching_historical_means=sum(c['within_1e12'] for c in checks),maximum_historical_mean_TPR_difference=max(c['absolute_error'] for c in checks),primary_numeric_runtime={k:rec.get(k) for k in ['numpy','pandas','scipy','sklearn','forest_prediction_execution']},original_experiment_versions_proven=False,interpretation='Historical catalog reproduction and same-protocol BG comparisons are separate. Use the regenerated preliminary-BG baseline when measuring a changed BG. Numerical drift is not tuned away, and its complete causal decomposition has not been established. Learned OOF results retain historical feature-selection/adaptive-exploration caveats.',optimizer_diagnostics=converged)
  dump(folder/'numerical_reproduction_assessment.json',d)
 print('Numerical assessments refreshed only for sealed preliminary baselines')
if __name__=='__main__':main()
