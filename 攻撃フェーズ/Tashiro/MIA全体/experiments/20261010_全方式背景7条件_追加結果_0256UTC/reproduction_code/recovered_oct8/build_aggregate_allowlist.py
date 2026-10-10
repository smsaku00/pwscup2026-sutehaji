"""Allowlist aggregate results and source bytes; exclude candidate-level data."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT_FILES={'README.md','completion.json','family_coverage.csv','within_family_maxima_descriptive.csv','historical_016_anchors_backgrounds.csv','runtime_versions.json','support_code_manifest.json','cache_schema.json','optimization_validation.json','denominator_category_coverage.json','scoring_validation.json','benchmark_results.json'}
FAMILY_ROOT_FILES={'completion.json','fixed_protocol.json','summary.csv','historical_anchor_checks.json','historical_catalog_regression.json','historical_catalog_regression.csv','historical_fixed22_target_regression.json','historical_fixed22_target_regression.csv','within_family_maxima_descriptive.csv','same_setting_background_changes.csv','convergence_summary.csv','lof_source_direct_verification.json','fixed40_Team7_backgrounds.csv','best_histogram_by_background_descriptive.csv'}
BG_FILES={'prediction_record.json','prediction_hashes.json','method_specs.csv','by_target.csv','fit_diagnostics.csv'}
CHOW_FILES={'completion.json','fixed_protocol.json','input_provenance.json','background_rank_diagnostics.csv','prediction_record.json','実行記録.json','Arefの固定bin境界.json','Q所有者等重みの診断パネル.csv','batchとsingleの生スコア検証.csv','fold別の課題所有者保留.csv','二fold集約の課題所有者保留.csv','固定三形の非会員診断.csv','対象別の固定三形診断.csv','木構造とPMF正規化.csv','独立形との上位12変化.csv','経験pairMIと木の辺.csv'}
def safe(p):
 r=p.relative_to(HERE)
 if '__pycache__' in r.parts:return False
 if p.suffix=='.py':return True
 if len(r.parts)==1:return p.name in ROOT_FILES
 if r.parts[0]=='chow':return p.name in CHOW_FILES
 if r.parts[0] in {'histogram','geometry','alignment','mixture','local'}:
  return (len(r.parts)==3 and p.name in FAMILY_ROOT_FILES) or (len(r.parts)==4 and p.name in BG_FILES)
 return False
def main():
 assert json.loads((HERE/'completion.json').read_text())['complete']
 included=[];excluded=[]
 for p in sorted(HERE.rglob('*')):
  if not p.is_file() or p.name=='aggregate_only_delivery_manifest.json':continue
  r=str(p.relative_to(HERE))
  if safe(p):included.append({'path':r,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  elif '__pycache__' not in p.parts:excluded.append(r)
 assert not any(x['path'].endswith('.npz') or 'prelim_score_row_mapping.csv' in x['path'] for x in included)
 report={'aggregate_only':True,'base_path':str(HERE),'include':included,'excluded_paths':excluded,'excluded_policy':'Candidate-level score/rank arrays, B record-id row mapping, raw inputs, logs, and unrestricted internal manifest are not included. Target/team-level aggregate metrics and source code are included. Chow tables were inspected: owner/fold/target statistics only, no candidate records.'}
 (HERE/'aggregate_only_delivery_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('Aggregate allowlist:',len(included),'files;',sum(x['bytes'] for x in included),'bytes')
if __name__=='__main__':main()
