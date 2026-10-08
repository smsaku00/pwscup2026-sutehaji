"""旧横断検証から、明示した非訓練基準でTPR>.10の全件を再集計する。

評価Cは所属ラベルを予測に使わないことだけを表し、非訓練とは別。
今回の非訓練一覧は、分類器・最適化する密度比/GMM/OCSVM・Cox・
データからの依存木等を除く。参照値の記述統計、固定KDE、近傍/LOF、
単一Gaussianのモーメント、Gaussian copula、頻度表、順位CDFは含む。
fit()というAPI名だけでは除外しない。非訓練は統計推定全般の否定ではない。

採否はcondition metadataと元コードの確認から決め、成績を読まない。
既報件数は採否確定後の検算のみ。元CSV/コード/予測・正解を更新しない。
生成物はこのタスクの集計CSV3件だけ（個票・順位・正解なし）。
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
OLD=HERE.parent/'20261008_帯別横断検証'
SRC='本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索/'
ARC='アーカイブ/予備戦_攻撃フェーズ/work_20260901以前/'
FAMILY_EXCLUSIONS={
 '公開データ分類器結果':('public_classifier','公開C対Aの分類器、IsolationForest等を訓練する。所属正解不使用でも訓練方式。',SRC+'公開データ分類器の探索.py:23'),
 '混合分布結果':('gmm_fit','GaussianMixtureのパラメータをfit、BIC等で成分を選ぶ。',SRC+'混合分布と局所密度の探索.py:14'),
 '直接密度比結果':('optimized_density_ratio','公開C/Aから正則化密度比係数betaを解く。',SRC+'直接密度比の探索.py:23'),
 '密度比_属性乱数結果':('optimized_density_ratio','公開C/Aから正則化密度比係数betaを解く。属性・乱数の変更でも同じ訓練成分。',SRC+'密度比の属性と乱数を確認.py:17'),
 '密度比_中心配置結果':('optimized_density_ratio','密度比係数を最適化する。中心選択にKMeans等も含む。',SRC+'中心配置を安定させた密度比.py:32'),
}
TRAINED_TOKENS=('ocsvm','c2st','classifier','extratrees','randomforest','_extra_',
 '_hgb_','_lr_c','_rf_','gmm','ulsif','rulsif','mi_tree','outcome_ensemble',
 'survival_likelihood','cox_residual')
OLD_DENSITY_EXCLUDE={'legacy_density_d3','legacy_density_d5','legacy_density_d5b','legacy_density_d6','legacy_density_d6b'}
VIRTUAL_ALLOWED={'legacy_virtual_covariate_quick','legacy_virtual_component_wb_recovered_gap'}
EXPECTED=dict(adopted_conditions=4777,above010_conditions=3110,above010_method_names=2456,
              above010_target_rows=6527,above010_teams=19)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def evidence(relative):
 if not relative:return ''
 path,line=relative.rsplit(':',1)
 assert (ROOT/path).is_file(),path
 return str(ROOT/path)+':'+line

def classify(row):
 """入力は方式条件だけ。TPR/成績による分岐はない。"""
 name=row['method'].lower();family=row['family']
 def result(adopted,rule,reason,reference='',training='該当なし'):
  return dict(nontraining_adopted=adopted,decision='採用' if adopted else '除外',
              classification_rule=rule,classification_reason=reason,
              training_assessment=training,classification_code_evidence=evidence(reference))
 if row['evaluation_class']!='C':
  return result(False,'evaluation_scope','評価A/Bを対象外とする。評価Cの所属ラベル非使用を前提に、さらに訓練成分を分類する。',training='採用範囲外')
 if row['selection']!='fixed_rule':
  return result(False,'selection_scope','対象別の履歴/oracle組込方針を対象外とし、全対象共通のfixed_ruleだけを対象にする。',training='採用範囲外')
 if family in FAMILY_EXCLUSIONS:
  rule,reason,ref=FAMILY_EXCLUSIONS[family]
  return result(False,rule,reason,ref,'訓練/係数最適化あり')
 # canonical名のprefixや別名統合によるsource1_unchangedも元aliasで解決。
 if family=='滑らかな対象間補正結果':
  sources=set(re.findall(r'source([0-6])_',row['source_method_aliases'].lower()))
  assert len(sources)==1,('softmax source ambiguous',row['condition_id'],sources)
  source=int(next(iter(sources)))
  if source in (0,1,4):
   return result(True,'softmax_fixed_source',f'源{source}はGen7/近傍比/LOFの固定score。softmax自体は固定温度・固定係数の変換で訓練しない。',SRC+'他対象との競合を滑らかに補正.py:10')
  return result(False,'softmax_trained_source',f'源{source}はGMM（2/3）または会員学習B（5/6）。後処理のラベル非使用で元の訓練成分は消えない。',SRC+'他対象との競合を滑らかに補正.py:10','訓練scoreを使用')
 if family=='離散依存木の密度比結果':
  if 'independent_pmf' in name:
   return result(True,'independent_frequency_pmf','独立PMFはedgesなし。固定bin上の属性別頻度とLaplace平滑化だけを使い、MI依存構造を学習しない。',SRC+'離散依存木の密度比を確認.py:172')
  return result(False,'learned_mi_tree','公開C/AのMIから最大全域木の依存構造を学習する。',SRC+'離散依存木の密度比を確認.py:56','依存構造学習あり')
 if family=='旧仮想大会の公開入力MIA':
  if name in VIRTUAL_ALLOWED:
   if name.endswith('covariate_quick'):
    return result(True,'virtual_fixed_components','covariate_quickのNN gap/kNN/DOMIAS/multiviewは固定参照scoreだけ。',ARC+'virtual_mia_engine.py:136')
   return result(True,'virtual_source_gap','公開source回復gapは分位写像と近傍差だけ。selected-negativeのExtraTrees成分を含めない。',ARC+'virtual_mia_engine.py:257')
  return result(False,'virtual_trained_component','分類器、Cox/生存モデル、またはsource選択ExtraTreesを成分に含むprofile/成分を除外。outcome_quickのclassifier-freeという記載だけで採用しない。',ARC+'virtual_mia_engine.py:143','訓練成分あり')
 if name in OLD_DENSITY_EXCLUDE:
  return result(False,'old_density_trained','旧D3はGMM/BIC、D5/D5bは分類器、D6/D6bはuLSIF/RuLSIF係数最適化。',
                'アーカイブ/予備戦_攻撃フェーズ/過去のコード/densratio/densities.py:238','訓練/係数最適化あり')
 matches=[token for token in TRAINED_TOKENS if token in name]
 if matches and 'independent_pmf' not in name:
  if any(t in name for t in ('survival_likelihood','cox_residual','outcome_ensemble')):
   reference=ARC+'outcome_mia_redteam.py:235';reason='C/AのCox/生存推定器、またはその成分と分類器を含むensemble。'
  elif name.startswith('legacy_redteam_'):
   reference=ARC+('mia_redteam.py:124' if name=='legacy_redteam_shadow_c2st' else 'outcome_mia_redteam.py:182');reason='公開C対AのExtraTrees/HGB分類器を訓練する。'
  else:
   reference='';reason='方式名が既知の分類器・最適化密度比・GMM・依存木・生存訓練成分を示す。元family/source metadataも採否分類CSVへ保持。'
  return result(False,'trained_component_token',reason+' 一致token='+','.join(matches),reference,'訓練成分あり')
 return result(True,'fixed_reference_statistic','既存固定設定のKDE・近傍/LOF・分布/頻度差・モーメント/順位/CDF・公開参照処理。訓練分類器/係数最適化/依存構造学習の成分を含まない。')

def main():
 paths=[OLD/name for name in ['方式条件一覧.csv','全方式評価.csv','結果保存元.csv']]
 input_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
 config,evaluations,sources=[pd.read_csv(p,float_precision='round_trip').fillna('') for p in paths]
 assert config.condition_id.is_unique and sources.source_id.is_unique
 assert not evaluations.duplicated(['condition_id','team']).any()
 assert set(evaluations.condition_id)==set(config.condition_id)
 source_lookup=sources.set_index('source_id')
 decisions=pd.DataFrame([classify(r) for r in config.to_dict('records')])
 classified=pd.concat([config,decisions],axis=1)
 classified['source_paths']=[json.dumps([str(ROOT/str(source_lookup.loc[s,'path']).split('#',1)[0])+('#'+str(source_lookup.loc[s,'path']).split('#',1)[1] if '#' in str(source_lookup.loc[s,'path']) else '') for s in ids.split(';')],ensure_ascii=False) for ids in config.source_ids]
 classified['catalog_source_sha256']=[json.dumps({s:source_lookup.loc[s,'sha256'] for s in ids.split(';')},ensure_ascii=False) for ids in config.source_ids]
 classified['inventory_input_sha256']=json.dumps(input_hashes,ensure_ascii=False,sort_keys=True)
 classified['inventory_code_sha256']=sha(Path(__file__))
 adopted=classified[classified.nontraining_adopted].copy()
 # 方式の採否を全件確定した後、TPRの閾値を適用する。
 eligible=evaluations[evaluations.condition_id.isin(adopted.condition_id)].copy()
 assert np.isfinite(eligible.tpr).all() and eligible.tpr.between(0,1).all()
 above=eligible[eligible.tpr>.10].copy()
 evaluation_fields=['condition_id','team','tpr','band','delta_p01','members','tp_equivalent']
 complete=above[evaluation_fields].merge(adopted,on='condition_id',validate='many_to_one')
 complete['threshold']=.10;complete['threshold_rule']='TPR生値>0.10（等号なし、表示丸めを使わない）'
 complete=complete.sort_values(['team','tpr','method','condition_id'],ascending=[True,False,True,True])
 summary=[]
 for team,part in eligible.groupby('team',sort=True):
  good=above[above.team.eq(team)];maximum=float(part.tpr.max());best=part[part.tpr.eq(maximum)]
  p01=part.tpr-part.delta_p01;assert p01.max()-p01.min()<1e-12
  summary.append(dict(team=int(team),members=int(part.members.iloc[0]),eligible_conditions=part.condition_id.nunique(),
    above010_conditions=good.condition_id.nunique(),above010_method_names=good.method.nunique(),above010_target_rows=len(good),
    maximum_tpr=maximum,p01_tpr=float(p01.iloc[0]),maximum_delta_p01=float(part.delta_p01.max()),
    maximum_condition_ids=';'.join(best.condition_id),maximum_methods=';'.join(sorted(set(best.method))),
    above010_families=';'.join(sorted(set(complete[complete.team.eq(team)].family))),
    interpretation='既存結果を見た後の候補集合内一覧。未知対象の期待TPRや方法の因果的優位性を示さない。'))
 summary=pd.DataFrame(summary)
 actual=dict(adopted_conditions=len(adopted),above010_conditions=above.condition_id.nunique(),
             above010_method_names=complete.method.nunique(),above010_target_rows=len(complete),above010_teams=complete.team.nunique())
 comparison={key:dict(expected=value,actual=int(actual[key]),match=int(actual[key])==value) for key,value in EXPECTED.items()}
 # 件数との不一致は出力・報告する。数合わせの採否変更はしない。
 classified['previous_count_comparison']=json.dumps(comparison,ensure_ascii=False,sort_keys=True)
 classified.to_csv(HERE/'非訓練方式_採否分類.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
 complete.to_csv(HERE/'非訓練方式_0.10超_全件.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
 summary.to_csv(HERE/'非訓練方式_チーム別集約.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
 for path,expected_hash in input_hashes.items():assert sha(ROOT/path)==expected_hash,path
 print(json.dumps(dict(count_comparison=comparison,total_classified=len(classified),
     decision_counts=classified.decision.value_counts().to_dict(),team_maximum=summary.set_index('team').maximum_tpr.to_dict(),
     definition='評価C/fixed_ruleに加え、訓練成分を元コードから除外。fit API名だけでは分類しない。'),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
