"""方式探索用の入力・較正・採点。公開所属正解は評価関数でのみ読む。"""
from __future__ import annotations
from pathlib import Path
import hashlib, importlib.util, json, platform, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OLD=HERE.parent/'20261007_MIA全体改善の検討'
CAL=HERE.parent/'20261008_MIA全体_較正の広範比較'
FPRS=(.005,.01,.02,.05)
NUM7=('age','SBP','TG','HDL','ALT')
NUM9=('age','BMI','SBP','TG','HDL','ALT','FPG')
NUM13=NUM9+('time',)
CAT7=('sex','prefecture')
CAT13=CAT7+('smoking','onset','death')

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m
base=module('分析',OLD/'分析.py')
cal=module('overnight_cal',CAL/'分析.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def log(**message):print(json.dumps(message,ensure_ascii=False),flush=True)

def setup(phase='prelim'):
 b,a,_,private,rare,cs,targets,self_id=base.phase_inputs(ROOT,phase)
 b=b.copy();a=a.copy();teams=list(cs)
 qs={t:base.read(targets/f'aia_challenge_{t}.csv') for t in sorted(set(cs)|{self_id})}
 query=pd.concat([b[list(base.QI7)]]+[q[list(base.QI7)] for q in qs.values()],ignore_index=True)
 owner=np.r_[np.full(len(b),-1),*[np.full(len(q),t) for t,q in qs.items()]]
 controls={t:np.flatnonzero((owner!=-1)&(owner!=t)) for t in teams}
 cuts=np.unique(np.quantile(a.age,(.2,.4,.6,.8)))
 query['age_bin']=np.searchsorted(cuts,query.age,side='right')
 b['age_bin']=query.age_bin.iloc[:len(b)].to_numpy()
 a['age_bin']=np.searchsorted(cuts,a.age,side='right')
 normal=~private&~rare
 for t,ii in controls.items():
  assert not pd.MultiIndex.from_frame(b.loc[normal,list(base.QI7)]).isin(pd.MultiIndex.from_frame(query.iloc[ii][list(base.QI7)])).any()
 assert int(normal.sum())==875
 return dict(phase=phase,b=b,a=a,private=private,rare=rare,normal=normal,cs=cs,teams=teams,
             query=query,controls=controls,age_cuts=cuts,self_id=self_id,targets=targets,qs=qs)

def rank(values,normal):return rankdata(np.asarray(values)[normal],method='average')/normal.sum()

def variants(scores,data,team,calibration=True):
 """全対象共通の較正条件。scoresはB、またはB+公開課題の順。"""
 b=data['b'];n=len(b);out={}
 for name,value in scores.items():
  assert np.isfinite(value).all(),(name,team)
  out[name+'__raw']=rank(value[:n],data['normal'])
  if calibration:
   ii=data['controls'][team];null=value[ii];q=data['query'].iloc[ii]
   for label,cols,s in [('county500',['prefecture','sex'],500),('age100',['sex','age_bin'],100)]:
    local,count=cal.group_stats(value[:n],b,null,q,cols)
    global_cdf=cal.empirical_cdf(value[:n],null)
    transformed=.999*cal.blend(local,count,global_cdf,s)+.001*rankdata(value[:n],method='average')/n
    out[name+'__'+label]=rank(transformed,data['normal'])
 return out

def matrices(banks,teams,cross=(0.,.1)):
 out={}
 for name,by_target in banks.items():
  mat=np.column_stack([by_target[t] for t in teams])
  assert mat.shape==(875,len(teams)) and np.isfinite(mat).all()
  med=np.column_stack([np.median(np.delete(mat,j,axis=1),axis=1) for j in range(len(teams))])
  for a in cross:out[name+f'__cross{a:g}']=mat-a*med
 return out

def evaluate(data,models):
 """全モデルの全対象予測を確定してから呼ぶ。"""
 b=data['b'];teams=data['teams']
 truthdir=base.child(base.child(ROOT/'予備戦','配布データ')/'結果','正解データ')
 own=base.read(truthdir/'pool_ids_5.csv').set_index('record_id').loc[b.record_id]
 truth=np.column_stack([own.pool_id.isin(base.read(truthdir/f'pool_ids_{t}.csv').pool_id).to_numpy(int) for t in teams])
 assert (truth.sum(axis=1)<=1).all() and not truth[data['private']].any()
 kit=base.child(base.child(ROOT/'本戦','配布データ'),'pwscup2026-kit-main_revised0912')
 sys.path.insert(0,str(kit/'starter/src'))
 from pwscup2026.scoring.attack_score import _tpr_at_fpr
 import pwscup2026.scoring.attack_score as official
 base.INPUT_HASHES[str(Path(official.__file__).resolve())]=sha(official.__file__)
 rows=[];err=0.
 for name,mat in models.items():
  for j,t in enumerate(teams):
   conf=base.confidence(mat[:,j],data['normal'],data['rare'],data['private'])
   fp,tp,_=roc_curve(truth[:,j],conf,drop_intermediate=False)
   for point in FPRS:
    score=float(np.interp(point,fp,tp));expected,_=_tpr_at_fpr(conf,truth[:,j],point)
    err=max(err,abs(score-expected));assert abs(score-expected)<1e-12
    rows.append(dict(method=name,team=t,fpr=point,mia_all=score,members=int(truth[:,j].sum())))
 return pd.DataFrame(rows),err,truth

def summaries(table,condition='full'):
 refpath=CAL/'結果/対象別TPR.csv';base.INPUT_HASHES[str(refpath.resolve())]=sha(refpath)
 refs=pd.read_csv(refpath)
 refs=refs[(refs.method=='flat_pref_sex_s500')&(refs.condition==condition)]
 rows=[]
 for (name,fpr),part in table.groupby(['method','fpr']):
  s=part.set_index('team').mia_all;other=s.drop([1,2])
  ref=refs[refs.fpr==fpr].set_index('team').mia_all
  d=other-ref.loc[other.index];positive=d[d>1e-12].sort_values(ascending=False)
  rows.append(dict(method=name,fpr=fpr,all20=float(s.mean()),other18=float(other.mean()),
                   delta_vs500=float(d.mean()),wins18=int((d>1e-12).sum()),losses18=int((d< -1e-12).sum()),
                   worst_delta=float(d.min()),gain_after_removing_best2=float(d.drop(positive.index[:2]).sum()),
                   largest_gain_share=float(positive.iloc[0]/positive.sum()) if len(positive) else 0.))
 return pd.DataFrame(rows).sort_values(['fpr','other18'],ascending=[True,False])

def save(out,data,models,specs,started,extra=None):
 if out.exists():raise FileExistsError(out)
 # 予測配列を作り終えた後、採点関数で初めて所属正解を読む。
 table,err,truth=evaluate(data,models)
 aggregate=summaries(table)
 out.mkdir(parents=True)
 table.to_csv(out/'対象別TPR.csv',index=False,float_format='%.17g')
 aggregate.to_csv(out/'集約比較.csv',index=False,float_format='%.17g')
 # 次段階の学習・独立再採点用。公式提出形式ではなく、内部の順位配列。
 names=sorted(models);arrays=np.stack([models[n] for n in names])
 np.savez_compressed(out/'内部順位キャッシュ.npz',names=np.array(names),teams=np.array(data['teams']),scores=arrays,
                     record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
 pd.DataFrame(specs).to_csv(out/'方式一覧.csv',index=False)
 for p,h in base.INPUT_HASHES.items():assert sha(p)==h,p
 now=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
 record=dict(completed_jst=now,seconds=time.perf_counter()-started,python=sys.executable,python_version=platform.python_version(),
             models=len(models),raw_specs=len(specs),official_tpr_max_error=err,input_sha256=base.INPUT_HASHES,
             support_code_sha256={str(p):sha(p) for p in [Path(__file__),OLD/'分析.py',CAL/'分析.py',HERE/'実験方針.json']},
             output_sha256={p.name:sha(p) for p in out.iterdir()},label_usage='予測確定後の採点のみ',
             private_and_rare_scores='既知非共有者0、未知希少0.4、通常者0.6より上。全方式共通。',
             exploratory=True,extra=extra or {})
 (out/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 log(stage='completed',output=str(out),models=len(models),seconds=record['seconds'],official_error=err)
 print(aggregate[aggregate.fpr==.01].head(15).to_string(index=False),flush=True)
 return table,aggregate
