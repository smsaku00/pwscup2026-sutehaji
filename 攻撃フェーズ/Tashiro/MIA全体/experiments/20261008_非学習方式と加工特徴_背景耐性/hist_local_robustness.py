"""固定した既存hist/LOF/adaptive KDE 9設定の背景感度。

元探索のpure関数だけを呼ぶ。全予測完成後にだけ採点する。
個人別予測・順位・正解はメモリだけに保持し、保存しない。
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
SOURCE=ROOT/'本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索'
SETTINGS=[
 ('q7_all_bins20_s1000_logratio','county500',.1,23,'hist','q7','all',20,'s1000_logratio'),
 ('q7_all_bins20_s1000_zscore','county500',.1,29,'hist','q7','all',20,'s1000_zscore'),
 ('q7_sex_bins40_s10_logratio','county500',0.,7,'hist','q7','sex',40,'s10_logratio'),
 ('q13_pref_sex_bins10_s10_logratio','raw',.1,8,'hist','q13','pref_sex',10,'s10_logratio'),
 ('q13_robust_lofC_k100','raw',0.,3,'local','q13','robust','lofC',100),
 ('q7_robust_lofC_k100','age100',0.,6,'local','q7','robust','lofC',100),
 ('q7_rank_lof_ratio_k100','age100',.1,17,'local','q7','rank','lof_ratio',100),
 ('q7_robust_adaptivek15_h1','age100',.1,20,'local','q7','robust','adaptive',(15,1.)),
 ('q13_robust_adaptivek5_h0.5','raw',0.,26,'local','q13','robust','adaptive',(5,.5)),
]

def method_name(setting):
 return f'{setting[0]}__{setting[1]}__cross{setting[2]:g}'

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
 return value

def prediction_modules(u=None):
 if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
 if u is None:u=module('共通',SOURCE/'共通.py')
 else:sys.modules['共通']=u
 dist=module('距離と分布の探索',SOURCE/'距離と分布の探索.py')
 hist=module('_background_hist',SOURCE/'階層ヒストグラムの探索.py')
 local=module('_background_local',SOURCE/'混合分布と局所密度の探索.py')
 return u,dist,hist,local

def predict_condition(data,background,label,modules):
 """全20対象で設定を共通に固定。所属ラベルを受け取らない。"""
 u,dist,hist,local=modules
 banks={setting[0]+'__'+setting[1]:{} for setting in SETTINGS}
 transforms={};den={};lof_a={};started=time.monotonic()
 # 同じA/同じ尺度の参照部分は一度だけ計算。
 for view,kind in [('q7','robust'),('q7','rank'),('q13','robust')]:
  nums=u.NUM7 if view=='q7' else u.NUM13;cats=u.CAT7 if view=='q7' else u.CAT13
  query=data['query'] if view=='q7' else data['b']
  ar,x=dist.transform(background,[background,query],nums,kind)
  transforms[view,kind]=(ar,x,query,nums,cats)
  if kind=='rank':
   fit=LocalOutlierFactor(n_neighbors=100,novelty=True,n_jobs=2);fit.fit(ar)
   lof_a[view,kind]=fit.score_samples(x)
  else:
   k,multiplier=(15,1.) if view=='q7' else (5,.5)
   ka=min(len(background)-1,round(k*len(background)/1049))
   da=cdist(x,ar,'sqeuclidean');daa=cdist(ar,ar,'sqeuclidean')
   den[view,kind]=local.adaptive_log_density(da,daa,query,background,background,cats,ka,multiplier,len(nums))
   del da,daa
 for j,team in enumerate(data['teams'],1):
  c=data['cs'][team]
  raw={}
  for view,group,bins in [('q7','all',20),('q7','sex',40),('q13','pref_sex',10)]:
   nums=u.NUM7 if view=='q7' else u.NUM13;cats=u.CAT7 if view=='q7' else u.CAT13
   query=data['query'] if view=='q7' else data['b']
   columns=[] if group=='all' else ['sex'] if group=='sex' else ['prefecture','sex']
   result=hist.histograms(background,query,c,nums,cats,bins,columns)
   for setting in SETTINGS:
    if setting[4]=='hist' and (setting[5],setting[6],setting[7])==(view,group,bins):raw[setting[0]]=result[setting[8]]
  for view,kind in [('q7','robust'),('q7','rank'),('q13','robust')]:
   ar,x,query,nums,cats=transforms[view,kind]
   _,_,cr=dist.transform(background,[background,query,c],nums,kind)
   fit=LocalOutlierFactor(n_neighbors=100,novelty=True,n_jobs=2);fit.fit(cr);value=fit.score_samples(x)
   raw[f'{view}_{kind}_lofC_k100']=value
   if kind=='rank':raw[f'{view}_{kind}_lof_ratio_k100']=value-lof_a[view,kind]
   else:
    k,multiplier=(15,1.) if view=='q7' else (5,.5)
    dc=cdist(x,cr,'sqeuclidean');dcc=cdist(cr,cr,'sqeuclidean')
    value=local.adaptive_log_density(dc,dcc,query,c,background,cats,k,multiplier,len(nums))
    raw[f'{view}_{kind}_adaptivek{k}_h{multiplier:g}']=value-den[view,kind]
    del dc,dcc
  for setting in SETTINGS:
   stem,cal=setting[:2];variants=u.variants({stem:raw[stem]},data,team,setting[5]=='q7')
   key=stem+'__'+cal;banks[key][team]=variants[key]
  if j%4==0 or j==len(data['teams']):
   print(f'hist/local {label}: {j}/{len(data["teams"])} targets ({time.monotonic()-started:.0f}s)',flush=True)
 models={}
 for setting in SETTINGS:
  key=setting[0]+'__'+setting[1]
  name=method_name(setting)
  models[name]=u.matrices({key:banks[key]},data['teams'],cross=(setting[2],))[name]
 assert len(models)==9
 return models

def saved_rank_check(data,models):
 """保存通常順位との照合には正解を読まない。"""
 checks=[]
 for folder,family in [('階層ヒストグラム結果','hist'),('局所密度結果','local')]:
  path=SOURCE/folder/'内部順位キャッシュ.npz'
  with np.load(path,allow_pickle=False) as saved:
   team_indices=[saved['teams'].tolist().index(t) for t in data['teams']]
   assert np.array_equal(saved['record_ids'],data['b'].loc[data['normal'],'record_id'].to_numpy())
   names=saved['names'].tolist();saved_scores=saved['scores']
   for setting in SETTINGS:
    if setting[4]!=family:continue
    name=method_name(setting);error=float(np.max(np.abs(models[name]-saved_scores[names.index(name)][:,team_indices])))
    assert error<1e-12,(name,error)
    checks.append(dict(method=name,saved_source=str(path.relative_to(ROOT)),maximum_rank_error=error))
 return checks

def main():
 bc=module('_background_hist_local_support',HERE/'background_common.py')
 started=time.monotonic();data=bc.setup();modules=prediction_modules(bc.u)
 source_paths=[Path(__file__),SOURCE/'階層ヒストグラムの探索.py',SOURCE/'混合分布と局所密度の探索.py',SOURCE/'距離と分布の探索.py']
 source_paths.extend(SOURCE/folder/file for folder in ['階層ヒストグラム結果','局所密度結果'] for file in ['対象別TPR.csv','内部順位キャッシュ.npz'])
 original_hashes={str(p.relative_to(ROOT)):bc.sha(p) for p in source_paths}
 predictions={};checks=[]
 with threadpool_limits(limits=1):
  for label,changed in bc.conditions(data):
   models=predict_condition(changed,changed['a'],label,modules)
   if label=='full':checks=saved_rank_check(changed,models)
   for name,mat in models.items():predictions[label,name]=mat
   bc.log(stage='background_predictions_complete',family='hist_local',condition=label,background_rows=len(changed['a']),methods=len(models),seconds=time.monotonic()-started)
 # 所属ラベルはここまで一度も読んでいない。
 assert len(predictions)==7*9
 for p,h in original_hashes.items():assert bc.sha(ROOT/p)==h,p
 specs=[dict(method=method_name(s),family=s[4],attributes=s[5],scale_or_group=s[6],
             bins_or_component=s[7],fixed_parameter=s[8],calibration=s[1],cross=s[2],
             original_highlight_team=s[3],settings_fixed_before_background_test=True) for s in SETTINGS]
 bc.score_and_save(data,predictions,'hist_local_results.csv','hist_local_run.json',specs,started,
   extra=dict(saved_rank_checks=checks,source_and_baseline_sha256=original_hashes,
              age100_cuts='full Aで定義したage_bin境界を全7背景で固定',
              local_background_k='元adaptive定義通りmin(nA-1,round(k*nA/1049))',
              original_functions='histograms, transform, adaptive_log_density, variants, matrices;元main不実行'))

if __name__=='__main__':main()
