"""Exact P06/P07 public holdout mechanics, replayed on each source pool.

Only mathematical source helpers are invoked. Published truth is not read
here. Frozen scores/decisions are consumed later by the policy evaluator.
"""
import argparse,time,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
import runtime as r

def p07(bg):
 d=r.data(bg);out=r.HERE/'cache/p07'/bg
 if (out/'complete.json').exists():return
 out.mkdir(parents=True,exist_ok=True);start=time.time()
 b,a,qs,cs,teams,normal=d['b'],d['a'],d['qs'],d['cs'],d['teams'],d['normal'];n=len(b)
 probes=sorted({5,*np.random.default_rng(r.p07.SEED).choice([t for t in sorted(qs) if t!=5],size=r.p07.N_Q_OTHER,replace=False).tolist()})
 raw=r.cached_raw(bg,d);offset=n;qidx={}
 for t in sorted(qs):qidx[t]=slice(offset,offset+len(qs[t]));offset+=len(qs[t])
 frame=pd.concat([b[list(r.u.base.QI7)]]+[qs[t][list(r.u.base.QI7)] for t in probes],ignore_index=True)
 nn=r.p07.nearest_both(r.u.base,frame,a,cs,teams);nnB=nn[:n];nnQ={};offset=n
 for h in probes:nnQ[h]=nn[offset:offset+len(qs[h])];offset+=len(qs[h])
 fullcal=r.cal_from_raw(raw,d);kde,_=r.archive.scores_KDEA(r.u.base,b,a,cs,teams)
 full,_=r.p07.two_matrices(r.base,{'P01':fullcal,'P03_Cnn':nnB},{'P01':fullcal[d['private']],'P03_Cnn':nnB[d['private']]},normal)
 full['KDEA']=r.cross(r.base.ranks(kde[normal]));r.seal_scores(out/'full',full,d)
 risks=[];discards=[]
 for h in probes:
  cb=[];cq=[]
  for j,t in enumerate(teams):
   leave=[k for k in sorted(qs) if k!=t and k!=h]
   controls=pd.concat([qs[k][list(r.u.base.QI7)] for k in leave],ignore_index=True)
   values=np.concatenate([raw[qidx[k],j] for k in leave])
   keep=~pd.MultiIndex.from_frame(controls).isin(pd.MultiIndex.from_frame(qs[h][list(r.u.base.QI7)]))
   discards.append({'target':t,'heldout':h,'cross_cohort_duplicate_QI7_removed':int((~keep).sum()),'null_train_cohorts':len(leave),'null_train_rows':int(keep.sum())})
   controls=controls.iloc[np.flatnonzero(keep)];values=values[keep];assert len(values)>=5000
   cb.append(r.calib.calibrate(raw[:n,j],b,values,controls,['prefecture','sex'],500))
   cq.append(r.calib.calibrate(raw[qidx[h],j],qs[h],values,controls,['prefecture','sex'],500))
  bs,hs=r.p07.two_matrices(r.base,{'P01':np.column_stack(cb),'P03_Cnn':nnB},{'P01':np.column_stack(cq),'P03_Cnn':nnQ[h]},normal)
  risks.extend(r.p07.evaluate_q_risk(bs,hs,normal,teams,h));r.seal_scores(out/f'hold_{h}',bs,d)
 pd.DataFrame(risks).to_csv(out/'risk.csv',index=False,float_format='%.17g');pd.DataFrame(discards).to_csv(out/'Q_exclusion_audit.csv',index=False)
 r.dump(out/'complete.json',{'background':bg,'folds':probes,'seconds':time.time()-start,'frozen_utc':datetime.now(timezone.utc).isoformat(),'truth_read':False})
 print('P07_FROZEN',bg,round(time.time()-start,2),flush=True)

def p06(bg):
 d=r.data(bg);out=r.HERE/'cache/p06'/bg
 if (out/'complete.json').exists():return
 out.mkdir(parents=True,exist_ok=True);start=time.time()
 b,a,qs,cs,teams,normal=d['b'],d['a'],d['qs'],d['cs'],d['teams'],d['normal'];n=len(b)
 risks=[];hashes=[];splits=[]
 for seed in r.p06.SEEDS:
  key=f'split80_seed{seed}';p=out/key
  rng=np.random.default_rng(seed);held=np.sort(rng.choice(len(a),len(a)//5,replace=False));train=np.setdiff1d(np.arange(len(a)),held)
  assert not set(held)&set(train)
  splits.append({'view':key,'fit_rows':len(train),'heldout_rows':len(held),'fit_pooled_row_sha256':r.msha(train),'hold_pooled_row_sha256':r.msha(held),'identity':'source+record_id via canonical pooled_row; no bare record_id joins'})
  if (p/'prediction_record.json').exists() and (p/'risk.csv').exists():
   models=r.load_scores(p);risks.extend(pd.read_csv(p/'risk.csv').to_dict('records'))
  else:
   aa=a.iloc[train].copy();hh=a.iloc[held].copy()
   cb,ch=r.p06.p01_public_holdout(r.u.base,r.calib,b,hh,aa,qs,cs,teams)
   combo=pd.concat([b,hh],ignore_index=True);kde,_=r.archive.scores_KDEA(r.u.base,combo,aa,cs,teams);dist=r.p06.dist_to_C(r.u.base,combo,aa,cs,teams)
   models,nulls=r.p06.formula(r.base,cb,ch,kde[:n],kde[n:],dist[:n],dist[n:],normal)
   table=r.p06.score_risk(models,nulls,key,'A_bg_20pct_heldout',teams);r.seal_scores(p,models,d)
   table.to_csv(p/'risk.csv',index=False,float_format='%.17g');risks.extend(table.to_dict('records'))
  hashes.extend({'view':key,'method':m,'sha256':r.msha(v)} for m,v in models.items())
  print('P06_SPLIT_FROZEN',bg,key,round(time.time()-start,2),flush=True)
 # Full primary scores are identical to P07's unchanged full20 construction.
 if not (r.HERE/'cache/p07'/bg/'complete.json').exists():p07(bg)
 full=r.load_scores(r.HERE/'cache/p07'/bg/'full');r.seal_scores(out/'full',full,d)
 hashes.extend({'view':'full','method':m,'sha256':r.msha(v)} for m,v in full.items())
 pd.DataFrame(risks).to_csv(out/'risk.csv',index=False,float_format='%.17g');r.dump(out/'prediction_hashes.json',hashes);r.dump(out/'split_audit.json',splits)
 if bg=='prelim':
  old=json.loads((r.SNAP/'20261009_P06_公開非会員の分離評価/results/prediction_hashes.json').read_text());expected={(x['view'],x['method']):x['sha256'] for x in old}
  checks=[dict(**v,exact_match=v['sha256']==expected[v['view'],v['method']]) for v in hashes]
  r.dump(out/'historical_prediction_regression.json',checks);assert all(x['exact_match'] for x in checks),checks
 r.dump(out/'complete.json',{'background':bg,'seconds':time.time()-start,'frozen_utc':datetime.now(timezone.utc).isoformat(),'truth_read':False})
 print('P06_FROZEN',bg,round(time.time()-start,2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--family',choices=['p06','p07'],required=True);p.add_argument('--backgrounds',nargs='+',default=list(r.BG));a=p.parse_args()
 for bg in a.backgrounds:globals()[a.family](bg)
