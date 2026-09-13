"""seo / 案01 / 順位混合Cox調整 / v1.0 / 2026-09-13.

個票入力は --b と --a-bg のみ。公式採点・攻撃課題・既存Cは読みません。
仕様の出典: PWS Cup 2026 main-process-20260912, §§3,5.1,6.1.2.
値域: 同版 participant_data/schema.json。下記は公式の共通スキーマのみ。
公式の全チーム共通の希少検出器をコードに記録。個人の希少正解は使いません。
A_bg引数は共通インターフェースとして検証しますが、生成計算には使いません。
"""
from __future__ import annotations

import os
# Set before numerical-library imports: fixed, single-threaded numerical execution.
for _var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning

# Algorithm settings, NOT official obligations. Counts are derived from inputs.
SEED = 9203
DONORS = 3
OUTCOME_SEED_OFFSET = 50000
CALIBRATION_SEED_OFFSET = 100003
STEPS = 30000
LAB_JITTER_IQR = 0.012
EVENT_JITTER_YEARS = 0.015
RISK_BINS = 64
RISK_PENALIZER = 0.05
START_TEMPERATURE = 0.012
END_TEMPERATURE = 0.0001

# Official common schema, frozen to main-process-20260912 (not learned constants).
HORIZON = 10.0
AGE_RANGE = (40,74)
RANGES = {'BMI':(10.,55.),'SBP':(40.,250.),'TG':(1.,3000.),
          'HDL':(10.,160.),'ALT':(.1,700.),'FPG':(40.,400.)}
COLS = ['age','sex','prefecture','BMI','SBP','TG','HDL','ALT',
        'smoking','FPG','time','onset','death']
CONT = ['age','BMI','SBP','TG','HDL','ALT','FPG']
COX = ['age','sex','BMI','SBP','log_TG','HDL','log_ALT','FPG','smoking']
INTEGERS = ['age','sex','smoking','onset','death']

# PUBLIC, all-team constants: main-process-20260912 participant_data/schema.json
# rare_detector. These are NOT team05 B statistics or individual/hidden labels.
# Order: age, BMI, SBP, TG, HDL, ALT, FPG. Frozen version; not a universal medical reference.
PUBLIC_OUTLIER_LO = 4.0
PUBLIC_MU = np.array([56.77566349411114,23.294812108472467,123.99248206704331,
    98.66064284138636,47.273300160874896,18.72786345814881,91.99791095874004])
PUBLIC_SIGMA = np.array([
    [104.60649311894663,-.8789019242323519,57.00514018313898,59.47191157451019,.7750840541190347,-14.643245558651431,33.39906681671122],
    [-.8789019242323519,15.689843536286972,29.030760348444648,94.2970738843124,-.8920293565829541,23.064103632916726,15.311644318894949],
    [57.00514018313898,29.030760348444648,365.903631028708,300.8876636296543,11.885497463006608,59.819670509128606,87.40558091814135],
    [59.47191157451019,94.2970738843124,300.8876636296543,5401.9165088216305,-35.78518366477272,379.716021717396,212.23925504519636],
    [.7750840541190347,-.8920293565829541,11.885497463006608,-35.78518366477272,69.65428549543113,19.237832999465834,15.435874265394421],
    [-14.643245558651431,23.064103632916726,59.819670509128606,379.716021717396,19.237832999465834,290.7075680483161,40.809523864100086],
    [33.39906681671122,15.311644318894949,87.40558091814135,212.23925504519636,15.435874265394421,40.809523864100086,232.01504023996264]])


def validate(df, label, expected=None):
    if list(df.columns) != COLS:
        raise ValueError(f'{label}: 出力列の不一致')
    if expected is not None and len(df) != expected:
        raise ValueError(f'{label}: 行数不一致')
    if df.empty or df.isna().any().any():
        raise ValueError(f'{label}: 空表または欠測')
    for col in COLS:
        if col == 'prefecture':
            if not df[col].map(lambda v: isinstance(v,str) and bool(v.strip())).all():
                raise ValueError(f'{label}: prefectureは空でない文字列が必要')
            continue
        vals=df[col].to_numpy(float)
        if not np.isfinite(vals).all():
            raise ValueError(f'{label}: {col}に非有限値')
        if col in INTEGERS and not np.equal(vals,np.rint(vals)).all():
            raise ValueError(f'{label}: {col}に非整数')
        if col in ['sex','smoking','onset','death'] and not np.isin(vals,[0,1]).all():
            raise ValueError(f'{label}: {col}に0/1以外')
        lo,hi=RANGES.get(col, AGE_RANGE if col=='age' else (-np.inf,np.inf))
        if ((vals<lo)|(vals>hi)).any():
            raise ValueError(f'{label}: {col}の値域[{lo},{hi}]外')
    if ((df.time<=0)|(df.time>HORIZON)).any():
        raise ValueError(f'{label}: timeは0より大きく10以下が必要')
    if ((df.onset==1)&(df.death==1)).any():
        raise ValueError(f'{label}: onsetとdeathは同時に1にできません')
    cens=(df.onset==0)&(df.death==0)
    if cens.any() and (abs(df.loc[cens,'time']-HORIZON)>.01).any():
        raise ValueError(f'{label}: 非発症・非死亡のtimeが追跡終了と不整合')


def read_input(path, label):
    # Do not silently ignore labels, hidden truth, or extra columns.
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        header=next(csv.reader(f),[])
    if len(header)!=len(set(header)):
        raise ValueError(f'{label}: 重複列名')
    df=pd.read_csv(path,encoding='utf-8-sig',float_precision='round_trip',dtype={'prefecture':str})
    if set(df.columns) not in [set(COLS),set(COLS+['record_id'])]:
        raise ValueError(f'{label}: 13属性と任意のrecord_idのみ使用可。B_selfや真値列は渡さないでください')
    if 'record_id' in df:
        rid=pd.to_numeric(df.record_id,errors='raise').to_numpy(float)
        if not np.isfinite(rid).all() or not np.equal(rid,np.rint(rid)).all() or len(np.unique(rid))!=len(rid):
            raise ValueError(f'{label}: record_idの欠測・非整数・重複')
    df=df[COLS].copy()
    for col in COLS:
        if col!='prefecture': df[col]=pd.to_numeric(df[col],errors='raise')
    validate(df,label)
    for col in INTEGERS: df[col]=df[col].astype(int)
    # Preserve specified input order for reproducibility with the development
    # implementation. IDs never affect transformations; no ID-specific rules.
    return df.reset_index(drop=True)


def cox_frame(d):
    x=d.copy()
    x['log_TG']=np.log(x.TG.to_numpy(float))
    x['log_ALT']=np.log(x.ALT.to_numpy(float))
    return x[COX]


def fit_cox(d,penalizer=0.):
    frame=cox_frame(d)
    frame['time']=d.time.to_numpy(float)
    frame['onset']=d.onset.to_numpy(int)
    model=CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(frame,duration_col='time',event_col='onset')
    if not np.isfinite(model.params_).all() or not (model.standard_errors_>0).all():
        raise ValueError('Cox推定が退化しました')
    return model


def public_detector():
    # Public schema only. B_self and utility_ref are never read by this file.
    inv=np.linalg.pinv(PUBLIC_SIGMA)
    def distance(d):
        q=d[CONT].to_numpy(float)-PUBLIC_MU
        return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',q,inv,q),0.))
    return lambda d: distance(d)>=PUBLIC_OUTLIER_LO


def mix_numeric(b,detect,rng):
    n=len(b); x=b[CONT].to_numpy(float)
    z=np.column_stack([norm.ppf(pd.Series(x[:,j]).rank(method='average').to_numpy()/(n+1)) for j in range(len(CONT))])
    groups=[np.flatnonzero(~detect(b)),np.flatnonzero(detect(b))]
    if any(0<len(g)<DONORS for g in groups):
        raise ValueError('外れ度群の人数が混合数未満です。この入力では本方式を適用できません')
    mixed=np.empty_like(z)
    for group in groups:
        if not len(group): continue
        mixed[group]=sum(z[rng.permutation(group)] for _ in range(DONORS))/np.sqrt(DONORS)
    c=b.copy()
    for j,col in enumerate(CONT):
        order=np.argsort(mixed[:,j],kind='stable')
        values=np.empty(n); values[order]=np.sort(x[:,j])
        if col!='age':
            iqr=float(np.subtract(*np.quantile(x[:,j],[.75,.25])))
            values+=rng.normal(0.,LAB_JITTER_IQR*iqr,n)
            values=np.clip(values,*RANGES[col])
        c[col]=values
    c['age']=np.clip(np.rint(c.age),*AGE_RANGE).astype(int)
    sd=x.std(0)
    if (sd<=0).any(): raise ValueError('Bの連続列に定数列があります')
    cost=(((c[CONT].to_numpy(float)[:,None,:]-x[None,:,:])/sd)**2).sum(axis=2)
    rows,donors=linear_sum_assignment(cost)
    cats=['sex','smoking','prefecture']
    c.loc[rows,cats]=b.iloc[donors][cats].to_numpy()
    for col in ['sex','smoking']: c[col]=c[col].astype(int)
    return c


def initial_outcomes(c,b,model,rng):
    c=c.copy(); n=len(b)
    rb=np.log(np.clip(model.predict_partial_hazard(cox_frame(b)).to_numpy(float),1e-12,None))
    rc=np.log(np.clip(model.predict_partial_hazard(cox_frame(c)).to_numpy(float),1e-12,None))
    tri=np.empty((n,3)); raw=b[['time','onset','death']].to_numpy(float)
    bins=min(RISK_BINS,n)
    for ib,ic in zip(np.array_split(np.argsort(rb),bins),np.array_split(np.argsort(rc),bins)):
        donors=ib.copy(); rng.shuffle(donors)
        tri[ic]=raw[donors]
    event=(tri[:,1]+tri[:,2])>0
    tri[event,0]=np.clip(tri[event,0]+rng.normal(0.,EVENT_JITTER_YEARS,int(event.sum())),1/365.25,HORIZON-.02)
    cens=b.loc[(b.onset==0)&(b.death==0),'time']
    if cens.empty: raise ValueError('非発症・非死亡が0人の入力には対応しません')
    tri[~event,0]=float(cens.median())
    c[['time','onset','death']]=tri
    return c


def calibrate(c,b,model,detect,rng):
    bx=cox_frame(b).to_numpy(float); sd=bx.std(0)
    if (sd<=0).any(): raise ValueError('Cox共変量に定数列があります')
    x=(cox_frame(c).to_numpy(float)-bx.mean(0))/sd
    beta=model.params_.to_numpy()*sd
    weights=np.exp(np.clip(x@beta,-25.,25.))
    cov=model.variance_matrix_.to_numpy()*np.outer(sd,sd)
    tri=c[['time','onset','death']].to_numpy(float)
    slots=np.argsort(tri[:,0],kind='stable'); fixed=tri[slots].copy(); perm=slots.copy()
    event_idx=np.flatnonzero(fixed[:,1]==1)
    # Efron gradient handles tied onset times (including clipping ties).
    event_times,starts,counts=np.unique(fixed[event_idx,0],return_index=True,return_counts=True)
    risk_start=np.searchsorted(fixed[:,0],event_times,side='left')
    rep=np.repeat(np.arange(len(counts)),counts)
    frac=np.concatenate([np.arange(k)/k for k in counts])
    slot_rare=detect(c)[perm]
    groups=[np.flatnonzero(~slot_rare),np.flatnonzero(slot_rare)]
    movable=np.flatnonzero((fixed[:,1]+fixed[:,2])>0)

    def loss():
        w=weights[perm]; xx=x[perm]; wx=w[:,None]*xx
        rs=np.cumsum(w[::-1])[::-1][risk_start]
        rx=np.cumsum(wx[::-1],axis=0)[::-1][risk_start]
        ew=np.add.reduceat(w[event_idx],starts)
        ex=np.add.reduceat(wx[event_idx],starts,axis=0)
        denom=rs[rep]-frac*ew[rep]
        expected=(rx[rep]-frac[:,None]*ex[rep])/denom[:,None]
        gradient=xx[event_idx].sum(0)-expected.sum(0)
        return float(gradient@cov@gradient)

    current=loss(); best=current; best_perm=perm.copy()
    for step in range(STEPS):
        i=int(rng.choice(movable)); j=int(rng.choice(groups[int(slot_rare[i])]))
        if i==j: continue
        perm[i],perm[j]=perm[j],perm[i]
        new=loss()
        temp=START_TEMPERATURE*(END_TEMPERATURE/START_TEMPERATURE)**(step/max(STEPS-1,1))
        if new<current or rng.random()<np.exp(min(0.,(current-new)/temp)):
            current=new
            if new<best: best=new; best_perm=perm.copy()
        else: perm[i],perm[j]=perm[j],perm[i]
    out=c.copy(); dest=np.empty_like(tri); dest[best_perm]=fixed
    out[['time','onset','death']]=dest
    for col in INTEGERS: out[col]=out[col].astype(int)
    return out,best


def generate(b,a,seed=SEED):
    if len(b)<30 or len(a)<30 or int(b.onset.sum())<10:
        raise ValueError('B/A_bgは各30行以上、Bの発症は10行以上が本方式の必要条件です')
    detect=public_detector(); model=fit_cox(b)
    # One frozen development configuration, no score/label-based candidate search.
    # Ridge Cox is used only to initialize coarse risk bins, not the target fit.
    risk_model=fit_cox(b,penalizer=RISK_PENALIZER)
    c=mix_numeric(b,detect,np.random.default_rng(seed))
    c=initial_outcomes(c,b,risk_model,np.random.default_rng(seed+OUTCOME_SEED_OFFSET))
    best,loss=calibrate(c,b,model,detect,np.random.default_rng(seed+CALIBRATION_SEED_OFFSET))
    trials=[{'seed':seed,'cox_gradient_loss':loss,'selection':'fixed configuration; minimum gradient loss over 30000 swaps'}]
    print(json.dumps(trials[0],ensure_ascii=False),flush=True)
    best=best.iloc[np.random.default_rng(seed+999983).permutation(len(b))].reset_index(drop=True)
    validate(best,'最終C',len(b))
    return best,trials


def csv_bytes(df):
    stream=io.StringIO(newline='')
    writer=csv.writer(stream,lineterminator='\n')
    writer.writerow(COLS)
    for row in df[COLS].itertuples(index=False,name=None):
        writer.writerow([str(v) if col=='prefecture' else str(int(v)) if col in INTEGERS else format(float(v),'.17g') for col,v in zip(COLS,row)])
    return stream.getvalue().encode('utf-8')


def peak_memory_mib():
    """Process peak working set on Windows; diagnostic only, not a stopping rule."""
    if os.name!='nt': return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
            (name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    psapi=ctypes.WinDLL('psapi'); kernel=ctypes.WinDLL('kernel32')
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    ok=psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb)
    return round(counters.PeakWorkingSetSize/1024**2,2) if ok else None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--b',required=True,type=Path)
    ap.add_argument('--a-bg',required=True,type=Path)
    ap.add_argument('--seed',type=int,default=SEED,help='安定性検証用。通常は既定値を使います')
    args=ap.parse_args(); start=time.perf_counter()
    try:
        if args.seed<0: raise ValueError('seedは0以上')
        b=read_input(args.b,'B'); a=read_input(args.a_bg,'A_bg')
        c,trials=generate(b,a,args.seed)
        data=csv_bytes(c)
        dest=Path(__file__).resolve().with_name('C_'+Path(__file__).stem+'.csv')
        if dest in [args.b.resolve(),args.a_bg.resolve()]: raise ValueError('入力と出力が同じパスです')
        if dest.exists() and dest.read_bytes()!=data:
            old=dest.read_bytes(); backup=dest.parent/'versions'
            backup.mkdir(exist_ok=True)
            archived=backup/(dest.stem+'-'+hashlib.sha256(old).hexdigest()[:12]+'.csv')
            if not archived.exists(): archived.write_bytes(old)
        dest.write_bytes(data)
        loaded=pd.read_csv(dest,float_precision='round_trip',dtype={'prefecture':str})
        validate(loaded,'保存後C',len(b))
        if csv_bytes(loaded)!=data: raise ValueError('保存後の数値再読込が一致しません')
        print(json.dumps({'output':dest.name,'rows':len(c),'columns':len(COLS),
            'sha256':hashlib.sha256(data).hexdigest(),'seconds':round(time.perf_counter()-start,3),
            'peak_working_set_mib':peak_memory_mib(),
            'official_utility':'not computed by generator'},ensure_ascii=False))
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
