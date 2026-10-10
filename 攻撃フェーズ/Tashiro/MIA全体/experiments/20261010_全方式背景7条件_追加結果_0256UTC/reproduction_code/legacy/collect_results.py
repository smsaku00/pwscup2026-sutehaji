"""Build coverage, historical reconciliation, low-FPR aliases and BG sensitivity tables."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'source_oct07'))
sys.path.insert(0,'/workspace/shared/mia_bg7_inputs/official_scoring/src')
import 分析 as common
from pwscup2026.scoring.attack_score import _tpr_at_fpr
INVENTORY=next(Path('/workspace/shared/mia_all_background_inventory').glob('**/inventory/oct07_config_inventory.json'))
INV=json.loads(INVENTORY.read_text());OLD=Path(INV['source_root'])
MAP={'geometry':'分析.py','additional':'追加検討.py','reference':'参照データの検討.py','bootstrap':'ブートストラップの検討.py','logistic':'学習による順位補正の検討.py','attributes':'属性部分集合の検討.py','fixed_sensitivity':'固定候補の感度検証.py','qcal':'非会員を使う順位較正の検討.py','qcal_sensitivity':'非会員較正の感度検証.py','low_fpr':'低FPRと対象偏りの検討.py'}
OLD_FILES={'geometry':'結果/予備戦_対象別比較.csv','additional':'追加結果_最終版/予備戦_追加対象別比較.csv','reference':'参照結果_拡張版/参照_対象別比較.csv','bootstrap':'背景再標本化結果/背景再標本化_対象別比較.csv','logistic':'学習結果_順位校正版/学習_対象別比較.csv','attributes':'属性部分集合結果/部分集合_対象別比較.csv','fixed_sensitivity':'固定候補の感度結果/感度検証_対象別比較.csv','qcal':'非会員較正結果/非会員較正_対象別比較.csv','qcal_sensitivity':'非会員較正の感度結果/較正感度_対象別比較.csv','low_fpr':'低FPR結果/低FPR_対象別比較.csv'}
BGS=['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
def low_alias(bg):
 source=BASE/'frozen/prelim'/bg/'fixed_sensitivity'
 if not (source/'manifest.json').exists():return
 target=BASE/'frozen/prelim'/bg/'low_fpr';target.mkdir(parents=True,exist_ok=True)
 meta=json.loads((source/'manifest.json').read_text());bank=np.load(source/'matrices.npz')
 mats={name.removeprefix('full__'):bank[name] for name in bank.files if name.startswith('full__')}
 assert len(mats)==5
 np.savez_compressed(target/'matrices.npz',**mats)
 meta.update({'family':'low_fpr','methods':list(mats),'alias_of_family':'fixed_sensitivity','alias_method_prefix':'full__','prediction_sha256':{n:hashlib.sha256(m.astype('<f8').tobytes()).hexdigest() for n,m in mats.items()}})
 (target/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 # All underlying predictions already exist and are frozen before pool truth is read.
 b,a,q,p,r,cs,t,self_id=common.phase_inputs(BASE/'compat_root','prelim');normal=~p&~r
 truthdir=BASE/'compat_root/予備戦/配布データ/結果/正解データ'
 own=common.read(truthdir/'pool_ids_5.csv').set_index('record_id').loc[b.record_id]
 rows=[]
 for j,team in enumerate(meta['teams']):
  y=own.pool_id.isin(common.read(truthdir/f'pool_ids_{team}.csv').pool_id).to_numpy(int)
  for name,m in mats.items():
   conf=common.confidence(m[:,j],normal,r,p)
   for fpr in [.005,.01,.02,.05]:
    tpr,auc=_tpr_at_fpr(conf,y,fpr)
    rows.append({'team':team,'method':name,'fpr':fpr,'mia_all':tpr,'auc':auc})
 pd.DataFrame(rows).to_csv(target/'low_fpr_by_target.csv',index=False,float_format='%.17g')
 pd.DataFrame(rows).query('fpr == .01').drop(columns='fpr').to_csv(target/'by_target.csv',index=False,float_format='%.17g')

def collect():
 for bg in BGS:low_alias(bg)
 configs=[c for c in INV['configurations'] if c['source_script'] in MAP.values()]
 assert len(configs)==316
 inverse={v:k for k,v in MAP.items()};coverage=[];frames=[];validation=[];hashrows=[]
 for bg in BGS:
  for fam,script in MAP.items():
   directory=BASE/'frozen/prelim'/bg/fam;path=directory/'by_target.csv'
   done=(BASE/'results/prelim'/bg/fam/'completion.json').exists() if fam!='low_fpr' else path.exists()
   actual=pd.read_csv(path) if path.exists() else None
   for cfg in [c for c in configs if c['source_script']==script]:
    match=actual[actual.method==cfg['historical_method_name']] if actual is not None else None
    replaced=(fam in ['fixed_sensitivity','qcal_sensitivity'] and cfg['historical_method_name'].startswith('fraction'))
    coverage.append({'evaluation_role':'historical_bg_sensitivity_supplement' if replaced else 'primary','background':bg,'family':fam,'config_id':cfg['config_id'],'family_id':cfg['family_id'],'method':cfg['historical_method_name'],'status':'completed' if done and match is not None and len(match)==20 else ('replaced_by_requested_external_background' if replaced and bg not in ['prelim','starter'] else 'prediction_frozen_score_pending' if (directory/'manifest.json').exists() else 'pending'),'target_rows':len(match) if match is not None else 0,'learning_class':cfg['learning_class']})
   if actual is not None:
    actual['evaluation_phase']='prelim';actual['background']=bg;actual['family']=fam;actual['config_id']=script+':'+actual.method
    actual['evaluation_role']=np.where((fam in ['fixed_sensitivity','qcal_sensitivity']) & actual.method.str.startswith('fraction'),'historical_bg_sensitivity_supplement','primary')
    frames.append(actual)
    meta=json.loads((directory/'manifest.json').read_text())
    meta.pop('labels_read_after_prediction_freeze',None);meta['evaluation_labels_read_after_prediction_freeze']=True;meta['training_labels_used']=fam=='logistic';meta['training_label_protocol']='person5_target4_joint_exclusion_OOF' if fam=='logistic' else 'no_MIA_training_labels'
    (directory/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    hashrows += [{'background':bg,'family':fam,'config_id':script+':'+method,'prediction_sha256':h} for method,h in meta['prediction_sha256'].items()]
    if bg=='prelim' and (OLD/OLD_FILES[fam]).exists():
     old=pd.read_csv(OLD/OLD_FILES[fam])
     if 'fpr' in old:old=old[np.isclose(old.fpr,.01)]
     z=old.merge(actual,on=['team','method'],suffixes=['_old','_new'])
     v={'family':fam,'historical_rows':len(old),'new_rows':len(actual),'matched_rows':len(z),'max_abs_mia_all':float((z.mia_all_old-z.mia_all_new).abs().max())}
     if 'auc_old' in z:v['max_abs_auc']=float((z.auc_old-z.auc_new).abs().max())
     v['pass_tolerance_1e12']=len(z)==len(old)==len(actual) and v['max_abs_mia_all']<1e-12 and v.get('max_abs_auc',0)<1e-12
     validation.append(v)
 out=BASE/'aggregate';out.mkdir(exist_ok=True)
 cov=pd.DataFrame(coverage)
 aliasmap=json.loads((BASE/'primary_alias_map.json').read_text())
 canonical={r['original_config_id']:r['canonical_config_id'] for r in aliasmap['map']}
 cov['canonical_config_id']=cov.config_id.map(canonical)
 cov.to_csv(out/'coverage.csv',index=False)
 canonical_cov=cov[(cov.evaluation_role=='primary')&(cov.config_id==cov.canonical_config_id)]
 canonical_cov.to_csv(out/'coverage_primary_canonical.csv',index=False)
 pd.DataFrame(validation).to_csv(out/'historical_validation.csv',index=False,float_format='%.17g')
 pd.DataFrame(hashrows).to_csv(out/'prediction_equivalence.csv',index=False)
 if frames:
  detail=pd.concat(frames,ignore_index=True);detail['canonical_config_id']=detail.config_id.map(canonical);detail.to_csv(out/'by_target_all.csv',index=False,float_format='%.17g')
  detail[detail.evaluation_role=='primary'].to_csv(out/'by_target_primary.csv',index=False,float_format='%.17g')
  detail[detail.config_id==detail.canonical_config_id].to_csv(out/'by_target_primary_canonical.csv',index=False,float_format='%.17g')
  summaries=[]
  for (bg,fam,cfg,method),g in detail.groupby(['background','family','config_id','method'],sort=False):
   for scope,part in [('all20',g),('without1_2',g[~g.team.isin([1,2])])]:
    summaries.append({'evaluation_phase':'prelim','background':bg,'family':fam,'config_id':cfg,'method':method,'target_scope':scope,'evaluation_role':part.evaluation_role.iloc[0],'targets':len(part),'mia_all':part.mia_all.mean(),'auc':part.auc.mean()})
  summary=pd.DataFrame(summaries);summary['canonical_config_id']=summary.config_id.map(canonical);summary.to_csv(out/'summary_all.csv',index=False,float_format='%.17g')
  summary[summary.evaluation_role=='primary'].to_csv(out/'summary_primary.csv',index=False,float_format='%.17g')
  summary[summary.config_id==summary.canonical_config_id].to_csv(out/'summary_primary_canonical.csv',index=False,float_format='%.17g')
  baseline=summary[summary.background=='prelim'][['config_id','target_scope','mia_all']].rename(columns={'mia_all':'prelim_background_mia_all'})
  changes=summary.merge(baseline,on=['config_id','target_scope'],how='left');changes['delta_from_prelim_background']=changes.mia_all-changes.prelim_background_mia_all
  changes.to_csv(out/'background_changes.csv',index=False,float_format='%.17g')
  changes[changes.config_id==changes.canonical_config_id].to_csv(out/'background_changes_primary_canonical.csv',index=False,float_format='%.17g')
  robust=summary.groupby(['config_id','family','method','target_scope']).agg(bg_conditions=('background','nunique'),worst_mia_all=('mia_all','min'),best_mia_all=('mia_all','max'),mean_mia_all=('mia_all','mean')).reset_index();robust['range_mia_all']=robust.best_mia_all-robust.worst_mia_all
  robust.to_csv(out/'robustness.csv',index=False,float_format='%.17g')
  robust['canonical_config_id']=robust.config_id.map(canonical)
  robust[robust.config_id==robust.canonical_config_id].to_csv(out/'robustness_primary_canonical.csv',index=False,float_format='%.17g')
 result={'historical_scenario_identity_backgrounds':2212,'expected_primary_config_backgrounds':1750,'primary_historical_config_ids_per_background':250,'unique_primary_recipes_per_background':206,'unique_primary_recipe_backgrounds':1442,'canonical_primary_completed':int((canonical_cov.status=='completed').sum()),'primary_completed':int(((cov.status=='completed')&(cov.evaluation_role=='primary')).sum()),'completed':int((cov.status=='completed').sum()),'status_counts':cov.status.value_counts().to_dict(),'historical_validation':validation,'excluded_nonprimary':{'AIA_proxy':10,'rare_diagnostics':3},'original_oof_preserved':True,'main_truth_used':False}
 (out/'status.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':collect()
