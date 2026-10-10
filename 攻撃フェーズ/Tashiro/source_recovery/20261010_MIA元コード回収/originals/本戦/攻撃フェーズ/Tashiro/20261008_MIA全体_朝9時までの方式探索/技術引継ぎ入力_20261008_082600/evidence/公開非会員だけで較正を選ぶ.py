"""会員正解を一切読まず、保留した既知非会員のCRPSで較正強度を選ぶ。

既存Gen-LRA7の式は固定。強度は対象共通、フェーズ別に選ぶ。
MIA採点・公式回答CSV・提出ZIPは作らない。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import 共通 as u

STRENGTHS = np.array([0,25,50,100,200,300,500,750,1000,2000,5000,np.inf])
SEED = 20261008
FOLDS = 5
PRIMARY = 'owner_one_se'
SOURCE = 'https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf'


def strength_name(s):
    return 'global' if np.isinf(s) else str(int(s))


def interval_cdf(knots, null):
    """各開区間内でのCDFの両端極限。最小・最大点での跳びも扱う。"""
    values, counts = np.unique(null, return_counts=True)
    cdf = (np.cumsum(counts)-.5*counts)/len(null)
    mid = (knots[:-1]+knots[1:])/2
    low = np.zeros(len(mid)); high = np.zeros(len(mid))
    above = mid > values[-1]
    low[above] = high[above] = 1.
    inside = (mid > values[0]) & (mid < values[-1])
    if inside.any():
        low[inside] = np.interp(knots[:-1][inside],values,cdf)
        high[inside] = np.interp(knots[1:][inside],values,cdf)
    return low,high


def prefix_at(knots, low, high, observed):
    width=np.diff(knots)
    prefix=np.r_[0.,np.cumsum(width*(low+high)/2)]
    clipped=np.clip(observed,knots[0],knots[-1])
    index=np.clip(np.searchsorted(knots,clipped,side='right')-1,0,len(width)-1)
    delta=clipped-knots[index]
    at=low[index]+(high[index]-low[index])*delta/width[index]
    return prefix[index]+delta*(low[index]+at)/2,prefix[-1],clipped


def crps_coefficients(global_null, local_null, observed):
    """CRPS(Fglobal+w*(Flocal-Fglobal),z)=A(z)+2wB(z)+w²C。"""
    observed=np.asarray(observed,float)
    if len(local_null)==0: local_null=global_null
    knots=np.unique(np.r_[global_null,local_null])
    if len(knots)==1:
        return np.abs(observed-knots[0]),np.zeros(len(observed)),0.
    gl,gh=interval_cdf(knots,global_null)
    ll,lh=interval_cdf(knots,local_null)
    dl,dh=ll-gl,lh-gh
    width=np.diff(knots)
    gp,gt,z=prefix_at(knots,gl,gh,observed)
    dp,dt,_=prefix_at(knots,dl,dh,observed)
    global_square=np.sum(width*(gl*gl+gl*gh+gh*gh)/3)
    product=np.sum(width*(2*gl*dl+gl*dh+gh*dl+2*gh*dh)/6)
    difference_square=np.sum(width*(dl*dl+dl*dh+dh*dh)/3)
    tail=np.maximum(knots[0]-observed,0)+np.maximum(observed-knots[-1],0)
    a=global_square+knots[-1]-z-2*(gt-gp)+tail
    b=product-(dt-dp)
    return a,b,float(difference_square)


def verify_integral():
    """別計算の区間ごとGauss積分で、跳び・同点・外挿を検査する。"""
    rng=np.random.default_rng(SEED)
    cases=[(np.array([0.,0.,1.,3.]),np.array([1.,1.,4.])),
           (np.array([2.,2.]),np.array([2.])),
           (rng.normal(size=17),rng.normal(size=7)),
           (np.array([0.]),np.array([-1.,1.]))]
    nodes,weights=np.polynomial.legendre.leggauss(3)
    maximum=0.; comparisons=0
    for global_null,local_null in cases:
        observations=np.r_[np.min(global_null)-3.,np.max(local_null)+3.,0.,2.]
        a,b,c=crps_coefficients(global_null,local_null,observations)
        for w in (0.,.2,.8,1.):
            expected=a+2*w*b+w*w*c
            for j,z in enumerate(observations):
                knots=np.unique(np.r_[global_null,local_null,z])
                low=knots[:-1,None];high=knots[1:,None]
                query=((high-low)*nodes+(high+low))/2
                f=(1-w)*u.cal.empirical_cdf(query.ravel(),global_null)+w*u.cal.empirical_cdf(query.ravel(),local_null)
                integrand=(f.reshape(query.shape)-(query>=z))**2
                measured=np.sum((high-low)/2*integrand*weights)
                maximum=max(maximum,abs(float(expected[j])-float(measured)))
                comparisons+=1
    assert maximum<1e-11,maximum
    return dict(comparisons=comparisons,maximum_absolute_error=maximum)


def public_folds(data):
    owner=np.r_[np.full(len(data['b']),-1),*[np.full(len(q),t) for t,q in data['qs'].items()]]
    ii=np.flatnonzero(owner!=-1)
    keys=pd.MultiIndex.from_frame(data['query'].iloc[ii][list(u.base.QI7)])
    ids,unique=pd.factorize(keys,sort=True)
    fold_unique=np.empty(len(unique),int)
    fold_unique[np.random.default_rng(SEED).permutation(len(unique))]=np.arange(len(unique))%FOLDS
    folds=np.full(len(owner),-1);folds[ii]=fold_unique[ids]
    # 完全一致する公開行は必ず同じfoldに置く。
    assert pd.DataFrame({'id':ids,'fold':folds[ii]}).groupby('id').fold.nunique().max()==1
    return owner,folds,int(len(ii)-len(unique))


def loss_for_target(raw,data,team,owner,folds):
    ii=data['controls'][team]
    values=raw[ii]; q=data['query'].iloc[ii].reset_index(drop=True)
    group_codes,_=pd.factorize(pd.MultiIndex.from_frame(q[['prefecture','sex']]),sort=True)
    scale=np.quantile(values,.75)-np.quantile(values,.25)
    if scale<=np.finfo(float).eps: scale=max(float(np.std(values)),1.)
    values=values/scale
    losses=np.full((len(ii),len(STRENGTHS)),np.nan)
    pits=np.full_like(losses,np.nan)
    counts=np.zeros(len(ii),int)
    for fold in range(FOLDS):
        train=folds[ii]!=fold;test=~train
        global_null=values[train]
        for group in np.unique(group_codes[test]):
            held=np.flatnonzero(test&(group_codes==group))
            local=values[train&(group_codes==group)]
            a,b,c=crps_coefficients(global_null,local,values[held])
            w=np.divide(len(local),len(local)+STRENGTHS,out=np.zeros(len(STRENGTHS)),where=len(local)+STRENGTHS>0)
            losses[held]=a[:,None]+2*w*b[:,None]+w*w*c
            g=u.cal.empirical_cdf(values[held],global_null)
            l=u.cal.empirical_cdf(values[held],local) if len(local) else g
            pits[held]=g[:,None]+(l-g)[:,None]*w
            counts[held]=len(local)
    assert np.isfinite(losses).all() and np.isfinite(pits).all()
    assert (losses>=-1e-10).all()
    rows=[]
    for source in np.unique(owner[ii]):
        mask=owner[ii]==source
        for k,s in enumerate(STRENGTHS):
            rows.append(dict(phase=data['phase'],target=team,null_owner=int(source),strength=strength_name(s),
                             crps=float(losses[mask,k].mean()),heldout_rows=int(mask.sum()),
                             pit_above_99=float((pits[mask,k]>.99).mean()),
                             pit_above_95=float((pits[mask,k]>.95).mean()),
                             local_n_min=int(counts[mask].min()),local_n_median=float(np.median(counts[mask]))))
    return rows


def select_strength(table):
    # 同じ公開Q行を対象ごとに独立な患者とは数えず、Q所有者でまとめる。
    panel=table.groupby(['null_owner','strength']).crps.mean().unstack('strength')
    columns=[strength_name(s) for s in STRENGTHS]
    panel=panel[columns]
    mean=panel.mean(axis=0);minimum_index=int(np.argmin(mean.to_numpy()))
    best=columns[minimum_index]
    se=float(panel[best].std(ddof=1)/np.sqrt(len(panel)))
    permitted=np.flatnonzero(mean.to_numpy()<=float(mean[best])+se)
    primary=int(permitted[-1])
    return dict(primary_strength=columns[primary],minimum_crps_strength=best,
                primary_index=primary,minimum_index=minimum_index,
                owner_standard_error=se,owners=len(panel),
                one_se_is_heuristic_not_transfer_confidence_interval=True),panel,mean


def calibrated(raw,data,team,strength):
    n=len(data['b']);ii=data['controls'][team]
    v=raw[:n];null=raw[ii];q=data['query'].iloc[ii]
    local,count=u.cal.group_stats(v,data['b'],null,q,['prefecture','sex'])
    global_cdf=u.cal.empirical_cdf(v,null)
    answer=u.cal.blend(local,count,global_cdf,strength)
    return u.rank(.999*answer+.001*rankdata(v,method='average')/n,data['normal'])


def run_phase(phase,out,qa):
    started=time.perf_counter(); data=u.setup(phase);owner,folds,duplicates=public_folds(data)
    raw={};rows=[]
    for index,(team,c) in enumerate(data['cs'].items(),1):
        raw[team]=u.base.core(data['a'],data['query'],c,u.NUM7,(10,))[10]
        rows.extend(loss_for_target(raw[team],data,team,owner,folds))
        u.log(stage='public_null_calibration',phase=phase,completed_targets=index,targets=len(data['teams']))
    table=pd.DataFrame(rows)
    selection_table=table[~table.target.isin([1,2])] if phase=='prelim' else table
    selection,panel,mean=select_strength(selection_table)
    selection['selection_targets']=sorted(int(t) for t in selection_table.target.unique())
    primary=STRENGTHS[selection['primary_index']]
    banks={'fixed500':{},PRIMARY:{}}
    for team in data['teams']:
        banks['fixed500'][team]=calibrated(raw[team],data,team,500)
        banks[PRIMARY][team]=calibrated(raw[team],data,team,primary)
    matrices=u.matrices(banks,data['teams'],cross=(.1,))
    reference=matrices['fixed500__cross0.1'];current=matrices[PRIMARY+'__cross0.1']
    diagnosis=[]
    for j,team in enumerate(data['teams']):
        for top in (12,20):
            old=set(np.flatnonzero(rankdata(-reference[:,j],method='min')<=top))
            new=set(np.flatnonzero(rankdata(-current[:,j],method='min')<=top))
            diagnosis.append(dict(phase=phase,team=team,top=top,old_count=len(old),new_count=len(new),
                                  added=len(new-old),removed=len(old-new),membership_accuracy_unknown=True,
                                  rank_correlation=float(np.corrcoef(rankdata(reference[:,j]),rankdata(current[:,j]))[0,1])))
    if phase=='main':
        path=u.HERE.parent/'20261008_第3回MIA_固定5bitと非会員較正/結果/F_mia.csv'
        submitted=u.base.read(path).set_index('record_id').loc[data['b'].record_id]
        error=0.
        for j,team in enumerate(data['teams']):
            confidence=.6+.3*rankdata(reference[:,j],method='min')/875
            error=max(error,float(np.max(np.abs(confidence-submitted.loc[data['normal'],str(team)].to_numpy()))))
        assert error<1e-14,error
        selection['round3_reference_max_error']=error
    selection['duplicate_public_qi7_rows']=duplicates
    dest=out/phase;dest.mkdir()
    table.to_csv(dest/'所有者別の保留非会員評価.csv',index=False,float_format='%.17g')
    panel.to_csv(dest/'所有者単位のCRPS.csv',float_format='%.17g')
    pd.DataFrame({'strength':mean.index,'mean_crps':mean.to_numpy()}).to_csv(dest/'強度選択の曲線.csv',index=False,float_format='%.17g')
    pd.DataFrame(diagnosis).to_csv(dest/'基準との順位差.csv',index=False,float_format='%.17g')
    names=sorted(matrices)
    np.savez_compressed(dest/'内部順位だけ.npz',names=np.array(names),teams=np.array(data['teams']),scores=np.stack([matrices[n] for n in names]),
                        record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
    (dest/'選択結果.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h,p
    assert not any('pool_ids' in p or '正解データ' in p for p in u.base.INPUT_HASHES)
    record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,
                code_path=str(Path(__file__).resolve()),code_sha256=u.sha(__file__),input_sha256=dict(u.base.INPUT_HASHES),
                support_code_sha256={str(p):u.sha(p) for p in [u.HERE/'共通.py',u.OLD/'分析.py',u.CAL/'分析.py']},
                output_sha256={p.name:u.sha(p) for p in dest.iterdir()},membership_labels_used=False,
                python=sys.executable,python_version=platform.python_version(),numerical_qa=qa,
                primary=PRIMARY,selection=selection,source=SOURCE,
                interpretation='非会員分布の予測損失。MIA成績・本戦への転移性能は評価していない。')
    (dest/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    u.log(stage='public_null_selected',phase=phase,selection=selection,seconds=record['seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=u.HERE/'公開非会員の較正結果')
    parser.add_argument('--qa-only',action='store_true')
    args=parser.parse_args();qa=verify_integral()
    if args.qa_only:u.log(numerical_qa=qa);return
    assert not args.output_dir.exists();args.output_dir.mkdir()
    plan=dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),code_sha256=u.sha(__file__),
              strengths=[strength_name(s) for s in STRENGTHS],folds=FOLDS,seed=SEED,
              prediction='固定Gen-LRA7;県×性別のCDF縮小;他対象中央値補正0.1',
              selection='非会員CRPSをQ所有者単位で集約し、最小値の1SE以内で最大の縮小強度を選ぶ。フェーズ内対象共通。予備戦の選択対象から1・2を除外。',
              primary=PRIMARY,membership_labels_used=False,metric_grid_chosen_from_existing_20261008_calibration_grid=True,
              no_mia_score_or_per_team_answer_selection=True,source=SOURCE,numerical_qa=qa)
    (args.output_dir/'実験方針.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for phase in ('prelim','main'):run_phase(phase,args.output_dir,qa)


if __name__=='__main__':main()
