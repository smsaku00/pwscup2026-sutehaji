"""P09: two fixed label-free conditional-null corrections; never submits.
Public Q cohorts 5,7,9,10,13,26 are wholly withheld from new-model fitting.
Released membership truth is opened only after all four predictions freeze.
Repeatedly researched preliminary cohort: NOT external validation.
"""
import argparse,hashlib,importlib.util,json,sys,time,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.stats import norm,rankdata

BASE='攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py'
HOLD=(5,7,9,10,13,26)
BGS=('full','sub80_seed20261009','sub80_seed20261010','sub80_seed20261011')
METHODS=('P01','conditional_quartile500','null_copula_ridge01')
CONFIG={'methods':METHODS,'holdout_Q_cohorts':HOLD,'backgrounds':BGS,'quartiles':4,'shrink':500,'ridge_penalty':.1,'infer_and_evaluate':'18 targets excluding 1/2','selection':'none; all two variants reported, no optimization','truth_used_for_prediction':False,'independent_external_validation':False,'private_rare_output':'unchanged private=0 rare=.4, normal rank .6-.9','baseline':'exact P01 uses full non-target Q; proposed variants use training Q only'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cdf(v,ref):
 s=np.sort(ref);return (np.searchsorted(s,v,'left')+np.searchsorted(s,v,'right')+1)/(2*(len(s)+1))
def rankref(v,ref):
 s=np.sort(ref);return (np.searchsorted(s,v,'left')+np.searchsorted(s,v,'right')+1)/(2*len(s))

def fit_transform(raw,source,nb,normal,teams,frames,calib,base):
 # raw contains B plus all Q, same public feature columns throughout.
 qsource=source[nb:]; n=len(raw); k=len(teams); train=~np.isin(qsource,HOLD)
 cal=np.empty_like(raw); z=np.empty_like(raw)
 for j,t in enumerate(teams):
  valid=qsource!=t; rr=raw[nb:,j]
  cal[:nb,j]=calib.calibrate(raw[:nb,j],frames.iloc[:nb],rr[valid],frames.iloc[nb:].loc[valid],['prefecture','sex'],500)
  cal[nb:,j]=calib.calibrate(rr,frames.iloc[nb:],rr[valid],frames.iloc[nb:].loc[valid],['prefecture','sex'],500)
  ref=rr[valid&train];p=np.clip(cdf(raw[:,j],ref),.5/(len(ref)+1),1-.5/(len(ref)+1));z[:,j]=norm.ppf(p)
 # Original calibrated ranks use only normal B rows, P01 inference excludes1/2.
 ranks=np.column_stack([rankref(cal[:,j],cal[:nb,j][normal]) for j in range(k)])
 p01=ranks-.1*base.rivals(ranks,np.median)
 assert np.allclose(ranks[:nb][normal],base.ranks(cal[:nb][normal]),rtol=0,atol=1e-15)
 scores={'P01':p01}; quart=np.empty_like(raw); resid=np.empty_like(raw);diags=[]
 for j,t in enumerate(teams):
  other=np.arange(k)!=j; med=np.median(z[:,other],axis=1)
  # For known Q rows, exclude the row's own team from rival propensity;
  # Q from its own cohort may be a member. Never use that score as a null.
  for s in np.unique(qsource):
   mask=np.flatnonzero(source==s); use=other & (np.array(teams)!=s)
   med[mask]=np.median(z[mask][:,use],axis=1)
  fit=np.r_[np.zeros(nb,dtype=bool),train&(qsource!=t)]
  edges=np.quantile(med[fit],[.25,.5,.75]); bins=np.searchsorted(edges,med,side='right')
  globalp=cdf(raw[:,j],raw[fit,j]);result=globalp.copy()
  for g in range(4):
   local=fit&(bins==g);idx=bins==g;nlocal=int(local.sum())
   result[idx]=(nlocal*cdf(raw[idx,j],raw[local,j])+500*globalp[idx])/(nlocal+500)
  # Tiny raw rank restores deterministic tail ordering, no individual ID tie break.
  quart[:,j]=.999*result+.001*rankref(raw[:,j],raw[:nb,j][normal])
  x=med[fit];y=z[fit,j];mx=x.mean();my=y.mean();vx=np.mean((x-mx)**2)
  beta=np.mean((x-mx)*(y-my))/(vx+.1)
  resid[:,j]=z[:,j]-beta*(med-mx)+1e-6*rankref(raw[:,j],raw[:nb,j][normal])
  diags.append({'team':t,'train_nonmembers':int(fit.sum()),'beta':float(beta),'rival_variance':float(vx),'bin_edges':edges.tolist(),'train_cohorts':sorted(set(qsource[train&(qsource!=t)].tolist()))})
 scores['conditional_quartile500']=quart;scores['null_copula_ridge01']=resid
 return scores,diags

def main(root,out):
 out.mkdir(parents=True,exist_ok=False);dump(out/'fixed_protocol.json',CONFIG);start=time.time()
 sp=importlib.util.spec_from_file_location('base',root/BASE);base=importlib.util.module_from_spec(sp);sp.loader.exec_module(base)
 common,calib,folder,paths,b,a,qs,cs,allteams,private,rare,normal=base.load_inputs(root)
 teams=[t for t in allteams if t not in(1,2)]; allq=pd.concat([qs[t][list(common.QI7)] for t in sorted(qs)],ignore_index=True)
 qsources=np.concatenate([np.repeat(t,len(qs[t])) for t in sorted(qs)]);source=np.r_[np.repeat(-1,len(b)),qsources]
 frames=pd.concat([b[list(common.QI7)],allq],ignore_index=True)
 frozen=[];hashes=[];fitstats=[];risk=[]
 for bg in BGS:
  if bg=='full':aa=a
  else:
   rng=np.random.default_rng(int(bg.split('seed')[1]));aa=a.iloc[np.sort(rng.choice(len(a),int(.8*len(a)),replace=False))]
  columns=[]
  for t in teams:
   columns.append(common.core(aa,frames,cs[t],common.NUM7,(10,))[10]);print(bg,t,round(time.time()-start,2),flush=True)
  raw=np.column_stack(columns);scores,stats=fit_transform(raw,source,len(b),normal,teams,frames,calib,base)
  fitstats.extend([{'background':bg,**s} for s in stats]); frozen.append((bg,{m:s[:len(b)][normal].copy() for m,s in scores.items()}))
  for m,s in scores.items():
   hashes.append({'background':bg,'method':m,'sha256':hashlib.sha256(np.ascontiguousarray(s[:len(b)][normal],dtype='<f8').tobytes()).hexdigest()})
   for j,t in enumerate(teams):
    ref=s[:len(b),j][normal];cut=np.quantile(ref,.99)
    for cohort in HOLD:
     if cohort==t:continue
     vals=s[len(b):,j][qsources==cohort]
     risk.append({'background':bg,'method':m,'team':t,'heldout_cohort':cohort,'n':len(vals),'above_B_normal_p99':float(np.mean(vals>cut)),'scope':'candidate-threshold exceedance; NOT official FPR'})
 dump(out/'prediction_hashes.json',hashes);dump(out/'null_fits.json',fitstats);pd.DataFrame(risk).to_csv(out/'heldout_Q_diagnostics.csv',index=False)
 freeze=datetime.now(timezone.utc).isoformat();print('ALL PREDICTIONS FROZEN; NOW OPEN RELEASED TRUTH',flush=True)
 truth,audit,truthpaths=base.audit_truth(folder,b,private,rare,allteams);rows=[]
 for bg,methods in frozen:
  for m,s in methods.items():
   confidence=base.confidence(s,normal,private,rare)
   for j,t in enumerate(teams):rows.append({'background':bg,'method':m,'team':t,**base.metrics(confidence[:,j],truth[:,allteams.index(t)])})
 by=pd.DataFrame(rows);by.to_csv(out/'by_target.csv',index=False,float_format='%.17g');summ=[]
 rng=np.random.default_rng(20261010)
 for bg,g in by.groupby('background',sort=False):
  ref=g[g.method=='P01'].set_index('team')['tpr_0.01']
  for m,h in g.groupby('method',sort=False):
   v=h.set_index('team')['tpr_0.01'];d=v-ref;boot=d.to_numpy()[rng.integers(len(d),size=(10000,len(d)))].mean(1);ci=np.quantile(boot,[.025,.975])
   summ.append({'background':bg,'method':m,'n_targets':len(v),'mean_tpr':v.mean(),'delta_P01':d.mean(),'worst_team_delta':d.min(),'best_team_delta':d.max(),'wins':int((d>1e-12).sum()),'losses':int((d< -1e-12).sum()),'TPR_ge010':int((v>=.1-1e-12).sum()),'new_ge010':int(((v>=.1-1e-12)&(ref<.1-1e-12)).sum()),'lost_ge010':int(((v<.1-1e-12)&(ref>=.1-1e-12)).sum()),'band2plus':int((h.band>=2).sum()),'mean_tpr_0005':h['tpr_0.005'].mean(),'mean_tpr_002':h['tpr_0.02'].mean(),'paired_target_bootstrap_lo':ci[0],'paired_target_bootstrap_hi':ci[1]})
 summary=pd.DataFrame(summ);summary.to_csv(out/'summary.csv',index=False,float_format='%.17g')
 actual=summary[(summary.background=='full')&(summary.method=='P01')].mean_tpr.item();assert abs(actual-.025495409939854385)<1e-12,actual
 stability=by.assign(hit=by['tpr_0.01']>=.1-1e-12).groupby(['team','method']).agg(min_tpr=('tpr_0.01','min'),mean_tpr=('tpr_0.01','mean'),hit_backgrounds=('hit','sum')).reset_index();stability.to_csv(out/'stability.csv',index=False)
 dump(out/'run_record.json',{'frozen_utc':freeze,'elapsed_seconds':time.time()-start,'baseline_regression_error':actual-.025495409939854385,'n_rows':len(by),'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},'input_sha256':{str(p.relative_to(root)):sha(p) for p in paths+[root/BASE]},'truth_sha256':{str(p.relative_to(root)):sha(p) for p in truthpaths},'script_sha256':sha(__file__),'truth_audit':audit,'limits':['same preliminary truth repeatedly studied','backgrounds are correlated resamples','Q cohorts not B sharing distribution','bootstrap descriptive targets exchangeability not guaranteed','no main data no submission no individual outputs']})
 print(summary.to_string(index=False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
