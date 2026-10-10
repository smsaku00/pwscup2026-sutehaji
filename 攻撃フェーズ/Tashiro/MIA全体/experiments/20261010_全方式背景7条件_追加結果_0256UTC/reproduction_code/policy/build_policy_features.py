"""Regenerate all background-dependent policy metadata without membership truth."""
import numpy as np,pandas as pd
import runtime as r

def generate(bg,historical=False):
 d=r.data(bg);a1=d['a'];a2=pd.read_csv('/workspace/shared/mia_bg7_inputs/bg/subsets/starter.csv',float_precision='round_trip') if historical else a1
 public=pd.concat([a1,a2],ignore_index=True) if historical else a1
 scale=a1[list(r.u.NUM7)].std(ddof=1).to_numpy();pk={key:r.public.keys(public,cols) for key,cols in [('clinical6',r.public.CLINICAL),('full13',r.public.FULL)]}
 a1keys=r.public.keys(a1,r.public.FULL);a2keys=r.public.keys(a2,r.public.FULL);rows=[]
 for t in d['teams']:
  if t==12:continue
  c,q=d['cs'][t],d['qs'][t];cq=r.public.nearest_rms(q,c,scale);ca=r.public.nearest_rms(a1,c,scale)
  row=dict(team=t,rows=len(c),numeric7_mean_abs_spearman=r.public.mean_abs_spearman(c,r.public.NUMERIC),QI5_mean_abs_spearman=r.public.mean_abs_spearman(c,list(r.u.NUM7)),A1_numeric7_mean_abs_spearman=r.public.mean_abs_spearman(a1,r.public.NUMERIC),public_A_clinical6_joint_rows=int(r.public.keys(c,r.public.CLINICAL).isin(pk['clinical6']).sum()),public_A_full13_rows=int(r.public.keys(c,r.public.FULL).isin(pk['full13']).sum()),A1_full13_rows=int(r.public.keys(c,r.public.FULL).isin(a1keys).sum()),A2_full13_rows=int(r.public.keys(c,r.public.FULL).isin(a2keys).sum()),A2_except_prefecture_rows=int(r.public.keys(c,[x for x in r.public.FULL if x!='prefecture']).isin(r.public.keys(a2,[x for x in r.public.FULL if x!='prefecture'])).sum()),full13_duplicates=int(c[r.public.FULL].round(8).duplicated().sum()),Q_rows=len(q),Q_sexmatched_QI5_RMS_lt001=int((cq<.01).sum()),Q_sexmatched_QI5_RMS_lt005=int((cq<.05).sum()),A1_rows=len(a1),A1_sexmatched_QI5_RMS_lt005=int((ca<.05).sum()),Q_sexmatched_QI5_RMS_median=float(np.median(cq)),A1_sexmatched_QI5_RMS_median=float(np.median(ca)))
  for col in r.public.NUMERIC+['time','smoking','onset','death']:row[col+'_unique']=int(c[col].nunique())
  for col in r.public.CLINICAL:row[col+'_public_value_fraction']=float(c[col].round(8).isin(public[col].round(8)).mean())
  rows.append(row)
 features=pd.DataFrame(rows);_,groups,cuts=r.group_features(d);out=r.HERE/'cache/metadata'/('historical_reference_roles' if historical else bg);out.mkdir(parents=True,exist_ok=True)
 features.to_csv(out/'public_features.csv',index=False,float_format='%.17g');groups.to_csv(out/'groups.csv',index=False,float_format='%.17g');r.dump(out/'cuts.json',cuts)
 r.dump(out/'p04_assignment.json',r.assignments(groups,cuts));r.dump(out/'protocol.json',{'background':bg,'historical_reference_roles':historical,'A1_role':'prelim' if historical else 'selected pool','A2_role':'starter' if historical else 'same selected pool alias for original feature names; not concatenated twice','public_A_reference':'prelim+starter' if historical else 'selected pool exactly once','B_practice_used':False,'truth_read':False})
 if historical:
  expected=pd.read_csv(r.SNAP/'20261008_非学習方式と加工特徴_背景耐性/公開加工特徴.csv').set_index('team');actual=features.set_index('team');cols=list(actual.select_dtypes('number').columns)
  err=float(np.nanmax(abs(actual[cols].sort_index().to_numpy()-expected.loc[actual.index,cols].sort_index().to_numpy())))
  old=pd.read_csv(r.SNAP/'20261009_AIA臨床分布と加工特徴の群別補正/results/public_target_groups.csv').set_index('team');cur=groups.set_index('team');gc=list(cur.select_dtypes('number').columns);gerr=float(np.nanmax(abs(cur[gc].sort_index().to_numpy()-old.loc[cur.index,gc].sort_index().to_numpy())))
  r.dump(out/'historical_regression.json',{'public_features_max_error':err,'group_features_max_error':gerr,'passed':bool(max(err,gerr)<1e-12)});assert max(err,gerr)<1e-12,(err,gerr)
 print('FEATURES_FROZEN',bg,historical,flush=True)

if __name__=='__main__':
 generate('prelim',True)
 for bg in r.BG:generate(bg)
