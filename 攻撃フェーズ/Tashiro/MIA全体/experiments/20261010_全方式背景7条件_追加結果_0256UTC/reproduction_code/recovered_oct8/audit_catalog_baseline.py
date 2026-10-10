"""Compare completed original-prelim-background results to recovered catalog.

Catalog reports19 targets (excluding12), while inference retains all20. This
audit is descriptive verification; never selects or changes source parameters.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
CATALOG=Path('/workspace/shared/mia_recovered_sources/攻撃フェーズ/Tashiro/source_recovery/20261010_MIA元コード回収/catalog/方式条件一覧.csv')
FAMILY={'geometry':'距離と分布結果','alignment':'分布を揃えた距離結果','histogram':'階層ヒストグラム結果','mixture':'混合分布結果','local':'局所密度結果'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--family',choices=FAMILY,required=True);args=p.parse_args();out=HERE/args.family/'prelim'
 actual=pd.read_csv(out/'summary.csv',float_precision='round_trip');actual=actual[(actual.background=='prelim')&(actual.population=='all_fixed')&(actual.evaluation=='without12')]
 catalog=pd.read_csv(CATALOG,float_precision='round_trip');expected=catalog[(catalog.family==FAMILY[args.family])&(catalog.condition=='full')&(catalog.team_count==19)].copy()
 # Canonical catalog names sometimes replace the source output name. Use the
 # catalog's explicit saved_method provenance, never a score-based guess.
 expected['catalog_method']=expected.method
 actual_names=set(actual.method)
 def source_name(row):
  params=json.loads(row.params) if isinstance(row.params,str) else {}
  saved=params.get('saved_method')
  return saved if saved in actual_names else row.method
 expected['method']=expected.apply(source_name,axis=1)
 assert not actual.method.duplicated().any() and not expected.method.duplicated().any()
 joined=actual[['method','mean_tpr']].merge(expected[['condition_id','catalog_method','method','mean_tpr']],on='method',suffixes=('_actual','_expected'),how='inner')
 joined['absolute_error']=abs(joined.mean_tpr_actual-joined.mean_tpr_expected);joined['pass']=joined.absolute_error<1e-12
 joined.to_csv(out/'historical_catalog_regression.csv',index=False,float_format='%.17g')
 report={'family':args.family,'actual_methods':len(actual),'catalog_methods':len(expected),'matched_methods':len(joined),'explicit_catalog_aliases':int((joined.method!=joined.catalog_method).sum()),'passed':int(joined['pass'].sum()),'failed':int((~joined['pass']).sum()),'max_abs_error':float(joined.absolute_error.max()),'not_in_catalog':sorted(set(actual.method)-set(expected.method)),'catalog_without_result':sorted(set(expected.method)-set(actual.method)),'scope':'inference20/evaluation19 excluding12/all1049, original preliminary BG','catalog_sha256':hashlib.sha256(CATALOG.read_bytes()).hexdigest(),'used_for_tuning':False}
 (out/'historical_catalog_regression.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
