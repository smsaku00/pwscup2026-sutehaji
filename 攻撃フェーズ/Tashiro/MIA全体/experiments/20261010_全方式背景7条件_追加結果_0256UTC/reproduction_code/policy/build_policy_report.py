"""Aggregate-only report builder and delivery manifest."""
import json,sys,platform,zipfile,hashlib
from pathlib import Path
import numpy as np,pandas as pd,scipy,sklearn,openpyxl
import runtime as r

def run():
 out=r.HERE/'report';out.mkdir(exist_ok=True);reg=pd.read_csv(r.HERE/'registry/coverage.csv').fillna('');assert not (reg.status=='pending').any()
 frames={};details=[]
 for x in reg.itertuples():
  if x.status!='complete':continue
  p=r.HERE/x.result_path
  if p not in frames:frames[p]=pd.read_csv(p)
  tab=frames[p];vcol='condition' if 'condition' in tab else 'view' if 'view' in tab else 'background';z=tab[(tab.policy==x.policy)&(tab[vcol]==x.view)].copy()
  if 'family' in z:z=z[z.family==x.family]
  z['config_id']=x.config_id;z['background_source']=x.background;z['native_view']=x.view;z['family']=x.family;z['semantic_alias_of']=x.semantic_alias_of;details.append(z[['config_id','family','policy','native_view','background_source','team','population','n','members','tpr_0.01','semantic_alias_of']])
 detail=pd.concat(details,ignore_index=True);detail.to_csv(out/'primary_by_target.csv',index=False,float_format='%.17g');summary=[]
 for labels,z in detail.groupby(['config_id','family','policy','native_view','background_source','population'],sort=False):
  vals=z['tpr_0.01'];ids=set(z.team);summary.append({**dict(zip(['config_id','family','policy','native_view','background_source','population'],labels)),'targets':len(z),'mean_native':vals.mean(),'TPR_ge010':int((vals>=.1-1e-12).sum()),'mean20':vals.mean() if len(ids)==20 else np.nan,'mean18_without1_2':z[~z.team.isin([1,2])]['tpr_0.01'].mean() if len(ids-set([1,2]))==18 else np.nan})
 summ=pd.DataFrame(summary);summ.to_csv(out/'primary_summary.csv',index=False,float_format='%.17g')
 ref=detail[detail.background_source=='main'][['config_id','population','team','tpr_0.01']].rename(columns={'tpr_0.01':'main_BG_tpr'});paired=detail.merge(ref,on=['config_id','population','team']);paired['delta_vs_main_BG']=paired['tpr_0.01']-paired.main_BG_tpr
 paired.groupby(['config_id','family','policy','native_view','background_source','population'],sort=False).agg(comparable_targets=('team','size'),mean_paired_delta=('delta_vs_main_BG','mean'),improved_targets=('delta_vs_main_BG',lambda x:int((x>1e-12).sum())),worsened_targets=('delta_vs_main_BG',lambda x:int((x< -1e-12).sum()))).reset_index().to_csv(out/'paired_vs_main_BG.csv',index=False,float_format='%.17g')
 counter=pd.concat([pd.read_csv(r.HERE/'results_public'/bg/'counterfactual20.csv') for bg in r.BG],ignore_index=True);counter.to_csv(out/'historical20_counterfactual_summary.csv',index=False,float_format='%.17g')
 fixed=[]
 for bg in r.BG:
  z=pd.read_csv(r.HERE/'cache/fixed22'/bg/'by_target.csv')
  for labels,h in z.groupby(['method','condition','population'],sort=False):
   for scope,part in [('all20',h),('without1_2',h[~h.team.isin([1,2])]),('native17',h[~h.team.isin([1,2,12])])]:fixed.append(dict(zip(['method','inner_view','population'],labels),background_source=bg,target_scope=scope,targets=len(part),mean_tpr=part['tpr_0.01'].mean(),TPR_ge010=int((part['tpr_0.01']>=.1-1e-12).sum())))
 pd.DataFrame(fixed).to_csv(out/'fixed22_summary.csv',index=False,float_format='%.17g')
 # Full selection context stays explicit in the report, without a best-only claim.
 full=summ[(summ.population=='all_fixed')&(summ.native_view=='full')];cols=['family','policy','background_source','targets','mean_native','TPR_ge010']
 overview=full[(full.family.isin(['p05','p06','p07','transfer','nested','nested_safety','p08d_looto']))].copy();overview[cols].to_csv(out/'policy_overview.csv',index=False,float_format='%.17g')
 counts=json.loads((r.HERE/'registry/counts.json').read_text());r.dump(out/'completion.json',{**counts,'coverage_complete':int((reg.status=='complete').sum()),'preliminary_explicit_not_applicable':int((reg.status!='complete').sum()),'main_not_applicable_registrations':139,'per_target_rows':len(detail),'summary_rows':len(summ),'main_policy_scores':'not applicable; original public-only P04 family projection has no score implementation','candidate_arrays_delivered':False})
 print('REPORT_COMPLETE',len(detail),len(summ))

if __name__=='__main__':run()
