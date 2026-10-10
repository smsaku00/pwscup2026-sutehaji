"""非会員CDFの広い集団への退避を、課題所有者を外した入れ子CVで検証。

六つの固定CDF混合案を公開非会員CRPSだけで選ぶ。会員正解は読まない。
"""
from __future__ import annotations
import argparse,json,platform,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import 共通 as u

pub=u.module('frozen_public_calibration',u.HERE/'公開非会員だけで較正を選ぶ.py')
CONFIGS=[
    dict(name='flat500',local=500,parent=None,parent_strength=None),
    dict(name='flat300',local=300,parent=None,parent_strength=None),
    dict(name='county300_sex500',local=300,parent='sex',parent_strength=500),
    dict(name='county300_sex100',local=300,parent='sex',parent_strength=100),
    dict(name='county300_sex_age500',local=300,parent='sex_age',parent_strength=500),
    dict(name='county300_sex_age100',local=300,parent='sex_age',parent_strength=100),
]


def crps_basis(samples,observed):
    """Fglobalからの三差分に関する正確な二次式。"""
    observed=np.asarray(observed,float)
    samples=[samples[0] if len(s)==0 else s for s in samples]
    knots=np.unique(np.concatenate(samples))
    dimension=len(samples)-1
    if len(knots)==1:
        return np.abs(observed-knots[0]),np.zeros((len(observed),dimension)),np.zeros((dimension,dimension))
    boundaries=[pub.interval_cdf(knots,s) for s in samples]
    gl,gh=boundaries[0];width=np.diff(knots)
    gp,gt,z=pub.prefix_at(knots,gl,gh,observed)
    a=np.sum(width*(gl*gl+gl*gh+gh*gh)/3)+knots[-1]-z-2*(gt-gp)
    a+=np.maximum(knots[0]-observed,0)+np.maximum(observed-knots[-1],0)
    dl=np.column_stack([low-gl for low,_ in boundaries[1:]])
    dh=np.column_stack([high-gh for _,high in boundaries[1:]])
    b=[]
    for j in range(dimension):
        dp,dt,_=pub.prefix_at(knots,dl[:,j],dh[:,j],observed)
        product=np.sum(width*(2*gl*dl[:,j]+gl*dh[:,j]+gh*dl[:,j]+2*gh*dh[:,j])/6)
        b.append(product-(dt-dp))
    c=(2*dl.T@(width[:,None]*dl)+dl.T@(width[:,None]*dh)+dh.T@(width[:,None]*dl)+2*dh.T@(width[:,None]*dh))/6
    return a,np.column_stack(b),c


def mixture_weights(counts):
    # 差分の順は県×性別、性別、性別×年齢。残りの重みは全体CDF。
    result=[]
    for config in CONFIGS:
        wl=counts[0]/(counts[0]+config['local']);v=np.array([wl,0.,0.])
        if config['parent']:
            j=1 if config['parent']=='sex' else 2
            wp=counts[j]/(counts[j]+config['parent_strength']);v[j]=(1-wl)*wp
        result.append(v)
    result=np.stack(result)
    assert (result>=0).all() and (result.sum(axis=1)<=1+1e-14).all()
    return result


def numerical_qa():
    rng=np.random.default_rng(20261008);nodes,weights=np.polynomial.legendre.leggauss(3)
    maximum=0.;comparisons=0
    for global_null in [np.array([0.,0.,1.,3.,4.]),rng.normal(size=19),np.array([2.,2.,2.])]:
        samples=[global_null,global_null[::2],global_null[1::2],global_null[-2:]]
        observed=np.r_[global_null[0],-5.,5.,.5]
        a,b,c=crps_basis(samples,observed)
        mixtures=[np.array([.2,.3,.1]),np.array([1.,0.,0.]),np.zeros(3)]
        for v in mixtures:
            loss=a+2*b@v+v@c@v
            for j,z in enumerate(observed):
                knots=np.unique(np.r_[global_null,z]);lo=knots[:-1,None];hi=knots[1:,None]
                x=((hi-lo)*nodes+(hi+lo))/2
                g=u.cal.empirical_cdf(x.ravel(),samples[0])
                f=g.copy()
                for k in range(3):f+=v[k]*(u.cal.empirical_cdf(x.ravel(),samples[k+1])-g)
                integral=float(np.sum((hi-lo)/2*(f.reshape(x.shape)-(x>=z))**2*weights))
                maximum=max(maximum,abs(float(loss[j])-integral));comparisons+=1
    assert maximum<1e-11,maximum
    return dict(comparisons=comparisons,maximum_absolute_error=maximum)


def evaluate_target(raw,data,target,owner,folds,excluded=None,outer=False):
    ii=data['controls'][target];q=data['query'].iloc[ii].reset_index(drop=True)
    source=owner[ii];values=raw[ii].copy()
    eligible=np.ones(len(ii),bool) if excluded is None else source!=excluded
    # 外側テスト所有者を使わず、内側fold間でも同じ尺度を固定する。
    scale=np.quantile(values[eligible],.75)-np.quantile(values[eligible],.25)
    if scale<=np.finfo(float).eps:scale=max(float(np.std(values[eligible])),1.)
    values/=scale
    losses=np.full((len(ii),len(CONFIGS)),np.nan);pits=np.full_like(losses,np.nan)
    county=q.prefecture.to_numpy();sex=q.sex.to_numpy();age=q.age_bin.to_numpy()
    cases=[(eligible,~eligible)] if outer else [(eligible&(folds[ii]!=f),eligible&(folds[ii]==f)) for f in range(pub.FOLDS)]
    for train,test in cases:
        if not test.any():continue
        global_null=values[train];held=np.flatnonzero(test)
        for (pref,sx,ag),local_indices in q.iloc[held].groupby(['prefecture','sex','age_bin'],sort=True).indices.items():
            observed_indices=held[local_indices];observed=values[observed_indices]
            masks=[(county==pref)&(sex==sx),sex==sx,(sex==sx)&(age==ag)]
            samples=[global_null]+[values[train&mask] for mask in masks]
            mix=mixture_weights([len(s) for s in samples[1:]])
            a,b,c=crps_basis(samples,observed)
            losses[observed_indices]=a[:,None]+2*b@mix.T+np.einsum('ki,ij,kj->k',mix,c,mix)[None,:]
            g=u.cal.empirical_cdf(observed,global_null)
            diff=np.column_stack([u.cal.empirical_cdf(observed,s)-g if len(s) else np.zeros(len(g)) for s in samples[1:]])
            pits[observed_indices]=g[:,None]+diff@mix.T
    mask=~eligible if outer else eligible
    assert np.isfinite(losses[mask]).all() and np.isfinite(pits[mask]).all()
    assert (losses[mask]>=-1e-9).all()
    rows=[]
    for heldowner in np.unique(source[mask]):
        held=mask&(source==heldowner)
        for k,config in enumerate(CONFIGS):
            rows.append(dict(phase=data['phase'],target=target,null_owner=int(heldowner),method=config['name'],
                             crps=float(losses[held,k].mean()),rows=int(held.sum()),
                             pit_above_99=float((pits[held,k]>.99).mean()),pit_above_95=float((pits[held,k]>.95).mean()),
                             outer_heldowner=excluded,outer_test=outer,training_iqr_scale=float(scale)))
    return pd.DataFrame(rows)


def choose(table,selection_targets):
    # 共有するQ患者を対象ごとに独立標本と数えず、所有者を等重みとする。
    mean=table[table.target.isin(selection_targets)].groupby(['null_owner','method']).crps.mean().groupby('method').mean()
    values=np.array([mean[c['name']] for c in CONFIGS])
    # 同点時は事前に固定した列順。候補の複雑化で同点を選ばない。
    best=int(np.flatnonzero(values<=values.min()+1e-12)[0])
    return best,mean


def predict(raw,data,target,config):
    ii=data['controls'][target];q=data['query'].iloc[ii];v=raw[:len(data['b'])];null=raw[ii]
    global_cdf=u.cal.empirical_cdf(v,null)
    local,n=u.cal.group_stats(v,data['b'],null,q,['prefecture','sex'])
    wl=n/(n+config['local']);fallback=global_cdf
    if config['parent']:
        cols=['sex'] if config['parent']=='sex' else ['sex','age_bin']
        parent,np_=u.cal.group_stats(v,data['b'],null,q,cols)
        wp=np_/(np_+config['parent_strength']);fallback=wp*parent+(1-wp)*global_cdf
    value=wl*local+(1-wl)*fallback
    return u.rank(.999*value+.001*rankdata(v,method='average')/len(v),data['normal'])


def phase_run(phase,out,qa):
    started=time.perf_counter();data=u.setup(phase);owner,folds,duplicates=pub.public_folds(data)
    cache_path=u.HERE/'非会員課題の丸ごと保留結果'/phase/'候補と公開課題の生スコア.npz'
    u.base.INPUT_HASHES[str(cache_path)]=u.sha(cache_path);cache=np.load(cache_path)
    assert list(cache['teams'])==data['teams'] and np.array_equal(cache['query_owner'],owner)
    assert cache['raw'].shape==(len(data['query']),len(data['teams']))
    raw={t:cache['raw'][:,j] for j,t in enumerate(data['teams'])}
    selection_targets=[t for t in data['teams'] if phase!='prelim' or t not in (1,2)]
    full=pd.concat([evaluate_target(raw[t],data,t,owner,folds) for t in data['teams']],ignore_index=True)
    best,mean=choose(full,selection_targets)
    dest=out/phase;dest.mkdir();full.to_csv(dest/'全公開非会員の内側保留.csv',index=False,float_format='%.17g')
    nested=[];selection=[]
    public_owners=sorted(data['qs'])
    for index,heldowner in enumerate(public_owners,1):
        targets=[t for t in selection_targets if t!=heldowner]
        inner=pd.concat([evaluate_target(raw[t],data,t,owner,folds,heldowner) for t in targets],ignore_index=True)
        selected,inner_mean=choose(inner,targets)
        assert not (inner.null_owner==heldowner).any()
        method=CONFIGS[selected]['name']
        outer=pd.concat([evaluate_target(raw[t],data,t,owner,folds,heldowner,outer=True) for t in targets],ignore_index=True)
        outer=outer[outer.method.isin([method,'flat500'])].copy()
        outer['selected_method']=method;nested.append(outer)
        selection.append(dict(phase=phase,heldowner=heldowner,selected_method=method,selection_targets=';'.join(map(str,targets)),
                              outer_owner_in_inner_fit=False,minimum_inner_crps=float(inner_mean[method])))
        u.log(stage='nested_public_calibration',phase=phase,completed_owners=index,owners=len(public_owners),selected_method=method,seconds=time.perf_counter()-started)
    outer_table=pd.concat(nested,ignore_index=True)
    outer_table.to_csv(dest/'課題所有者の外側保留.csv',index=False,float_format='%.17g')
    pd.DataFrame(selection).to_csv(dest/'外側保留ごとの内側選択.csv',index=False,float_format='%.17g')
    banks={c['name']:{t:predict(raw[t],data,t,c) for t in data['teams']} for c in CONFIGS}
    banks['public_selected']=banks[CONFIGS[best]['name']]
    matrices=u.matrices(banks,data['teams'],cross=(.1,))
    old_path=u.HERE/'公開非会員の較正結果'/phase/'内部順位だけ.npz'
    u.base.INPUT_HASHES[str(old_path)]=u.sha(old_path);old=np.load(old_path)
    baseline=matrices['flat500__cross0.1'];j=list(old['names']).index('fixed500__cross0.1')
    error=float(np.max(np.abs(baseline-old['scores'][j])));assert error<1e-14,error
    diagnosis=[]
    for j,t in enumerate(data['teams']):
        before=set(np.flatnonzero(rankdata(-baseline[:,j],method='min')<=12))
        for name,matrix in matrices.items():
            after=set(np.flatnonzero(rankdata(-matrix[:,j],method='min')<=12))
            diagnosis.append(dict(phase=phase,team=t,method=name,top=12,new_candidates=len(after-before),old_candidates=len(before-after),
                                  rank_correlation=float(np.corrcoef(rankdata(baseline[:,j]),rankdata(matrix[:,j]))[0,1]),membership_accuracy_unknown=True))
    pd.DataFrame(diagnosis).to_csv(dest/'基準との上位順位差.csv',index=False,float_format='%.17g')
    names=sorted(matrices);np.savez_compressed(dest/'内部順位だけ.npz',names=np.array(names),teams=np.array(data['teams']),scores=np.stack([matrices[n] for n in names]))
    final_selection=dict(phase=phase,primary_method=CONFIGS[best]['name'],inner_crps={k:float(v) for k,v in mean.items()},
                         outer_selection_counts=pd.Series([s['selected_method'] for s in selection]).value_counts().to_dict(),
                         selection_targets=selection_targets,membership_labels_used=False)
    (dest/'選択結果.json').write_text(json.dumps(final_selection,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h,p
    assert not any('pool_ids' in p or '正解データ' in p for p in u.base.INPUT_HASHES)
    record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,
                code_path=str(Path(__file__).resolve()),code_sha256=u.sha(__file__),input_sha256=dict(u.base.INPUT_HASHES),
                support_code_sha256={str(p):u.sha(p) for p in [u.HERE/'共通.py',u.HERE/'公開非会員だけで較正を選ぶ.py',u.OLD/'分析.py',u.CAL/'分析.py']},
                output_sha256={p.name:u.sha(p) for p in dest.iterdir()},numerical_qa=qa,membership_labels_used=False,
                full_selected_method=CONFIGS[best]['name'],baseline_rank_max_error=error,duplicate_public_qi7_rows=duplicates,
                python=sys.executable,python_version=platform.python_version(),
                scale_protocol='Outer training controls only; scale frozen across inner folds.',
                aggregation_protocol='Inner: mean over eligible targets within Q owner, then equal mean over owners. Outer: equal Q-owner weight.',
                sex_age_parent='県×性別と厳密な階層ではなく、重なる広い群へのCDF退避混合。',
                meaning='非会員分布較正の入れ子CV。会員判別・本戦のMIA転移性能は評価しない。')
    (dest/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    u.log(stage='nested_public_calibration_complete',phase=phase,selection=final_selection,seconds=record['seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=u.HERE/'広い集団への非会員較正結果');parser.add_argument('--qa-only',action='store_true')
    args=parser.parse_args();qa=numerical_qa()
    if args.qa_only:u.log(numerical_qa=qa);return
    assert not args.output_dir.exists();args.output_dir.mkdir()
    plan=dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),configs=CONFIGS,inner_folds=pub.FOLDS,seed=pub.SEED,
              code_sha256=u.sha(__file__),selection='六つの事前固定CDF混合案から公開非会員CRPSだけで対象共通選択。',
              validation='全Q所有者の外側保留。外側所有者は内側選択から完全除外。尺度は外側学習のみ。',
              aggregation='Q所有者内で対象平均、その後Q所有者を等重みで集約。外側評価も所有者を等重みにする。',
              preliminary_selection_excludes_targets=[1,2],membership_labels_used=False,mia_score_selection=False,numerical_qa=qa)
    (args.output_dir/'実験方針.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for phase in ('prelim','main'):phase_run(phase,args.output_dir,qa)


if __name__=='__main__':main()
