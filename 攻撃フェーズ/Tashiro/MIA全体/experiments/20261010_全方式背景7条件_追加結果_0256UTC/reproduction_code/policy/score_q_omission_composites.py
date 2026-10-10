"""Replay all six original P04 Q-omission composite reports, without choosing a best fold."""
import argparse,json
import pandas as pd,numpy as np
import runtime as r
from evaluate_public_policies import metadata,ensure_kdeb
from score_counterfactuals import WB

def run(bg,historical=False):
 key='historical_reference_roles' if historical else bg;out=r.HERE/'results_public'/key;d=r.data(bg)
 _,groups,cuts=metadata(bg,historical);assign=r.assignments(groups.reset_index(),cuts)
 full=r.load_scores(r.HERE/'cache/p07'/bg/'full');full['KDEB']=ensure_kdeb(bg,d)
 probes=json.loads((r.HERE/'cache/p07'/bg/'complete.json').read_text())['folds'];bank={}
 for h in probes:
  score=r.load_scores(r.HERE/'cache/p07'/bg/f'hold_{h}')['P01'].copy()
  for t in r.p05.TEAM17:
   if assign[t]!='P01':score[:,d['teams'].index(t)]=full[assign[t]][:,d['teams'].index(t)]
  bank[h]=score
 r.dump(out/'q_omission_composites_freeze.json',{'all_six_reported':True,'prediction_hashes':{str(h):r.msha(v) for h,v in bank.items()},'decisions_use_membership_truth':False,'scope':'P04 public grouping; Q-omitted P01 only in native P01 group and team12; original official1/2 completion separate'})
 y=r.truth(d);rows=[];counter=[];r.p05.OFFICIAL=str(WB);_,mia,aia,_=r.p05.official_excel(r.HERE)
 for h,score in bank.items():
  met=pd.DataFrame(r.metrics_rows(score,d,y,family='p07_Q_omission_composite',policy=f'omit_Q_{h}',view='full',outer_background=bg))
  rows.extend(met[met.team.isin(r.p05.TEAM17)].to_dict('records'))
  vals=met[(met.population=='historical_native')&(~met.team.isin([1,2]))].set_index('team')['tpr_0.01'].to_dict();vals.update({t:mia[t]['old_tpr'] for t in (1,2)})
  counter.append({'outer_background':bg,'excluded_public_Q_cohort':h,'population':'historical20_counterfactual','native_scored_targets':17,'fixed_official_targets':'1;2','P01_completion_target':12,'mean20':np.mean(list(vals.values())),'mean18_without_fixed1_2':np.mean([v for t,v in vals.items() if t not in(1,2)]),'TPR_ge010':sum(v>=.1-1e-12 for v in vals.values()),'hypothetical_attack_power':r.p05.attack_power(vals,mia,aia)})
 pd.DataFrame(rows).to_csv(out/'p07_omission_composite_by_target.csv',index=False,float_format='%.17g');tab=pd.DataFrame(counter);tab.to_csv(out/'p07_omission_counterfactual20.csv',index=False,float_format='%.17g')
 if historical:
  old=pd.read_csv(r.SNAP/'20261009_P07_公開AIAコホート保留と方式選択/results/Q_cohort_omission_counterfactual.csv');z=tab.merge(old,on='excluded_public_Q_cohort');err=float((z.hypothetical_attack_power-z.P04_Q_omission_hypothetical_attack).abs().max());eq=bool((z.TPR_ge010==z.TPR010_count).all())
  r.dump(out/'historical_q_omission_composite_regression.json',{'rows':len(z),'max_attack_power_error':err,'hits_equal':eq});assert len(z)==6 and err<1e-12 and eq
 print('Q_OMISSION_COMPOSITES',bg,historical,len(tab),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--backgrounds',nargs='+',default=list(r.BG));p.add_argument('--historical',action='store_true');a=p.parse_args()
 for bg in a.backgrounds:run(bg,a.historical)
