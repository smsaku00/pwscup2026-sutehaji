"""Keep published20-target counterfactual convention separate from score inference."""
import json,argparse
from pathlib import Path
import numpy as np,pandas as pd
import runtime as r

WB=Path('/workspace/shared/mia_recovered_validation/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/private_inputs/公開用結果表_20260909.xlsx')
SOURCES={'p05':('20261009_P05_損失防止と公開特徴ゲート/results/summary.csv','rule','prelim_MIA_all_mean_20','simulated_attack_power','prelim_attack_010_hits_20'),
'p06':('20261009_P06_公開非会員の分離評価/results/counterfactual_summary.csv','policy','mean_MIA_full_20','hypothetical_attack_power','hits_ge010_20'),
'p07':('20261009_P07_公開AIAコホート保留と方式選択/results/counterfactual_summary.csv','policy','MIA_full_avg20','hypothetical_prelim_attack','TPR010_hits20')}

def run(bg,historical=False):
 out=r.HERE/'results_public'/('historical_reference_roles' if historical else bg);d=r.data(bg);r.p05.OFFICIAL=str(WB)
 _,mia,aia,oldattack=r.p05.official_excel(Path('/'));y=r.truth(d);full=r.load_scores(r.HERE/'cache/p07'/bg/'full')
 p01conf=r.base.confidence(full['P01'],d['normal'],d['private'],d['rare']);t12=r.base.metrics(p01conf[:,d['teams'].index(12)],y[:,d['teams'].index(12)])['tpr_0.01'];rows=[]
 for family in SOURCES:
  p=out/f'{family}_by_target.csv'
  if not p.exists():continue
  by=pd.read_csv(p);native=by[(by.view=='full')&(by.population=='historical_native')]
  for policy,g in native.groupby('policy',sort=False):
   vals=g.set_index('team')['tpr_0.01'].to_dict();assert set(vals)==set(r.p05.TEAM17)
   vals.update({1:mia[1]['old_tpr'],2:mia[2]['old_tpr'],12:t12})
   rows.append({'outer_background':bg,'family':family,'policy':policy,'population':'historical20_counterfactual','native_scored_targets':17,'fixed_official_targets':'1;2','P01_completion_target':12,'mean20':np.mean(list(vals.values())),'mean18_without_fixed1_2':np.mean([v for t,v in vals.items() if t not in(1,2)]),'TPR_ge010':sum(v>=.1-1e-12 for v in vals.values()),'hypothetical_attack_power':r.p05.attack_power(vals,mia,aia),'old_official_attack_power':oldattack})
 table=pd.DataFrame(rows);table.to_csv(out/'counterfactual20.csv',index=False,float_format='%.17g');r.dump(out/'counterfactual_protocol.json',{'official_workbook_sha256':r.sha(WB),'not_twenty_regenerated_predictions':True,'official_MIA_1_2_frozen':True,'official_rare_MIA_and_AIA_frozen':True,'team12_regenerated_P01':True,'normal_counterfactual20':'not applicable: official1/2 record-level scores absent; no invented normal scores'})
 if historical:
  checks=[]
  for family,(path,key,mean,power,hits) in SOURCES.items():
   if family not in set(table.family):continue
   old=pd.read_csv(r.SNAP/path).set_index(key);actual=table[table.family==family].set_index('policy')
   for policy,x in actual.iterrows():
    checks.append({'family':family,'policy':policy,'mean20_error':abs(x.mean20-old.loc[policy,mean]),'attack_power_error':abs(x.hypothetical_attack_power-old.loc[policy,power]),'hits_equal':bool(x.TPR_ge010==old.loc[policy,hits])})
  r.dump(out/'historical_counterfactual_regression.json',checks);assert all(max(x['mean20_error'],x['attack_power_error'])<1e-12 and x['hits_equal'] for x in checks),checks
  pp=out/'p06_by_target.csv'
  if pp.exists():
   p=pd.read_csv(pp);p=p[(p.population=='historical_native')&(p.view!='full')];actual=p.groupby(['policy','view'])['tpr_0.01'].mean();old=pd.read_csv(r.SNAP/'20261009_P06_公開非会員の分離評価/results/matched_background_summary.csv').set_index(['policy','view']).selected_mean;err=float((actual-old).abs().max());r.dump(out/'historical_p06_split_regression.json',{'max_error':err,'rows':len(actual),'expected':len(old)});assert len(actual)==len(old) and err<1e-12
 print('COUNTERFACTUALS_SCORED',bg,historical,len(table),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--backgrounds',nargs='+',default=list(r.BG));p.add_argument('--historical',action='store_true');a=p.parse_args()
 for bg in a.backgrounds:run(bg,a.historical)
