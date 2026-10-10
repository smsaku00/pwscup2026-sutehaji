"""Verify overlapping fixed22 baseline targets at the historical reference FPRs."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
REF=Path('/workspace/shared/mia_all_background_inventory/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/source_snapshot/攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_非学習方式と加工特徴_背景耐性/固定方式_全背景評価.csv')
def main():
 p=argparse.ArgumentParser();p.add_argument('--family',required=True);args=p.parse_args();out=HERE/args.family/'prelim'
 actual=pd.read_csv(out/'prelim/by_target.csv',float_precision='round_trip');actual=actual[actual.population=='all_fixed']
 expected=pd.read_csv(REF,float_precision='round_trip');expected=expected[expected.condition=='full'];methods=sorted(set(actual.method)&set(expected.method));rows=[]
 for fpr,column in [(0.005,'tpr_0.005'),(.01,'tpr_0.01'),(.02,'tpr_0.02'),(.05,'tpr_0.05')]:
  merged=actual[actual.method.isin(methods)][['method','team',column]].merge(expected[expected.method.isin(methods)&expected.fpr.eq(fpr)][['method','team','tpr']],on=['method','team']);merged['fpr']=fpr;merged['error']=abs(merged[column]-merged.tpr);merged=merged.rename(columns={column:'actual','tpr':'expected'});rows.append(merged)
 if not methods:print('No fixed22 overlaps',args.family);return
 report=pd.concat(rows,ignore_index=True);report.to_csv(out/'historical_fixed22_target_regression.csv',index=False,float_format='%.17g')
 result={'family':args.family,'methods':methods,'target_metric_checks':len(report),'fprs_checked':sorted(report.fpr.unique().tolist()),'max_error':float(report.error.max()),'passed':bool((report.error<1e-12).all()),'source_sha256':hashlib.sha256(REF.read_bytes()).hexdigest(),'scope':'original prelim BG; all20 inference; historical19 target report excluding12; all1049; all available overlapping historical FPRs','used_for_tuning':False}
 (out/'historical_fixed22_target_regression.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result);assert result['passed']
if __name__=='__main__':main()
