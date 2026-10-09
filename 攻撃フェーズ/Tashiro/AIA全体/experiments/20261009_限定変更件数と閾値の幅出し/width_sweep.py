#!/usr/bin/env python3
"""PWS CUP 2026 AIA: width sweep for limited edits using held-out-cohort models.

Prediction inputs: public Q/C/background, plus trained models using OTHER publicly
released preliminary cohorts only.  Target label/time/is_rare never enter decisions.
Truth is used solely in the evaluation functions, and inner training labels for
meta-selection NEVER include the outer target cohort.  This is exploratory reuse
of previously analysed preliminary cohorts, not proof of main-phase performance.
"""
from __future__ import annotations
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, hashlib, importlib.util, json, math, time, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

SEED=20261009
TEAMS=[1,2,3,4,5,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
BUDGETS=(0,1,2,3,5,8,10,15,20,30,50,80,160,326)
MARGINS=(-.02,0,.005,.01,.02,.03,.05,.08,.12)
FAMILIES=('down11','down0','up11','mixed')
FIXED=('down11',10,.02)
REFERENCE_MEAN_R={'nonself20':.19139935985665485,'processed18':.149846}


def import_module(path: Path, name: str):
    spec=importlib.util.spec_from_file_location(name, path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def gains(p, rates, lam, base):
    p=np.asarray(p,float)
    out={}
    g11= -(p*rates['m9']-lam*(1-p)*rates['c9'])
    g0 = g11+p*rates['m0']-lam*(1-p)*rates['c0']
    gup= - (p*rates['m0']-lam*(1-p)*rates['c0'])
    out['down11']=np.where(base,g11,-np.inf)
    out['down0']=np.where(base,g0,-np.inf)
    out['up11']=np.where(~base,gup,-np.inf)
    out['mixed']=np.where(base,np.maximum(g11,g0),gup)
    return out, np.where(g11>=g0,11.,0.)


def propose(base, row_ids, g, edit_target, family, budget, margin):
    """Only strictly greater estimated gain than margin, capped at budget.

    base: Boolean mask for 9.5-year; other rows are 0-year.  For mixed,
    target is 11/0 for selected rows and 11 for unselected rows.
    Ties by challenge_row_id.  No truth is accepted by this function.
    """
    base=np.asarray(base,bool)
    if budget<0:raise ValueError('negative budget')
    g=np.asarray(g,float)
    idx=np.flatnonzero(g>margin)
    if not budget or not len(idx):
        return np.where(base,9.5,0.)
    order=np.lexsort((np.asarray(row_ids,int)[idx],-g[idx]))
    chosen=idx[order[:budget]]
    pred=np.where(base,9.5,0.)
    if family=='down11' or family=='up11':pred[chosen]=11.
    elif family=='down0':pred[chosen]=0.
    elif family=='mixed':pred[chosen]=edit_target[chosen]
    else:raise ValueError(family)
    return pred


def data_source(args,cp):
    prior=cp.import_prior(args.prior)
    D,F,T=cp.data(args.root,prior,args.density_scores)
    scores=pd.read_csv(args.outer_scores)
    scores=scores[scores.model=='c_bg_plus_local'].copy()
    assert len(scores)==21*326
    idx=scores.set_index(['team','row_id'])['score']
    return D,F,T,idx


def prepare_outer(k,cp,D,F,T,idx,folds):
    inner=cp.inner_oof(F,T,k,cp.SPECS['c_bg_plus_local'],folds)
    calibrator=cp.fit_calibrator(inner,'platt')
    lam=cp.expected_lambda(inner)
    rates=cp.expected_hit_rates(inner)
    q=D[k][0]
    raw=idx.loc[[(k,int(z)) for z in q.challenge_row_id]].to_numpy(float)
    assert len(raw)==326
    cal=calibrator(raw)
    base=cp.top75(q,raw)
    G,which=gains(cal,rates,lam,base)
    return dict(q=q,cal=cal,base=base,original=np.where(base,9.5,0.),gain=G,action=which,
                inner=inner,calibrator=calibrator,lambda_est=lam,rates=rates)


def eval_policy(cp,pred,truth,original=None):
    d=cp.score_metrics(pred,truth)
    if original is not None:
        h0=np.abs(original-truth.time.to_numpy(float))<=.5
        h1=np.abs(pred-truth.time.to_numpy(float))<=.5
        member=truth.label.eq('member').to_numpy(bool)
        rare=truth.is_rare.to_numpy(bool)
        d.update(changed=int(np.sum(pred!=original)),
                 gained_m=int(np.sum(~h0&h1&member)),lost_m=int(np.sum(h0&~h1&member)),
                 gained_c=int(np.sum(~h0&h1&~member)),lost_c=int(np.sum(h0&~h1&~member)),
                 gained_rare_m=int(np.sum(~h0&h1&member&rare)),lost_rare_m=int(np.sum(h0&~h1&member&rare)),
                 gained_rare_c=int(np.sum(~h0&h1&~member&rare)),lost_rare_c=int(np.sum(h0&~h1&~member&rare)))
    return d


def evaluate_grid(cp,outer,truth):
    q,base,orig=outer['q'],outer['base'],outer['original']
    ids=q.challenge_row_id.to_numpy(int)
    result=[]
    for fam in FAMILIES:
        g=outer['gain'][fam]
        for budget in BUDGETS:
            for margin in MARGINS:
                pred=propose(base,ids,g,outer['action'],fam,budget,margin)
                d=eval_policy(cp,pred,truth,orig)
                result.append(dict(family=fam,budget=budget,margin=margin,**d))
    return result


def aggregate(raw,base_per_team):
    scopes={'nonself20': {5},'processed18':{1,2,5}}
    summaries=[]
    for scope,omit in scopes.items():
        df=raw[~raw.team.isin(omit)]
        grouped=df.groupby(['folds','family','budget','margin'],sort=True)
        a=grouped.agg(n=('team','size'),r_all=('r_all','mean'),r_rare=('r_rare','mean'),r_max=('r_max','mean'),rare_extra=('rare_extra','mean'),
            mean_changed=('changed','mean'),total_changed=('changed','sum'),
            hit_m=('hit_m','mean'),hit_c=('hit_c','mean'),
            gained_m=('gained_m','sum'),lost_m=('lost_m','sum'),gained_c=('gained_c','sum'),lost_c=('lost_c','sum'),
            hit_m_rare=('hit_m_rare','mean'),hit_c_rare=('hit_c_rare','mean')).reset_index()
        a.insert(0,'scope',scope)
        a['delta_r_all']=a['r_all']-a['r_all'].where(a.budget.eq(0)).groupby([a['folds'],a['family'],a['margin']]).transform('max').fillna(0)
        # Preserve reference from stored per-target, not the potentially reordered grid.
        a['delta_r_all']=a.apply(lambda r:r.r_all-base_per_team[(r.folds,scope)],axis=1)
        a['gain_loss_net']=(a['gained_m']-a['lost_m'])-(a['gained_c']-a['lost_c'])
        summaries.append(a)
    return pd.concat(summaries,ignore_index=True)


def bootstrap_for(df,base_df,folds=4,nsample=15000):
    rng=np.random.default_rng(SEED)
    out=[]
    scopes={'nonself20':{5},'processed18':{1,2,5}}
    for scope,omit in scopes.items():
        selected=df[(df.folds==folds)&(~df.team.isin(omit))]
        baseline=base_df[(base_df.folds==folds)&(~base_df.team.isin(omit))].set_index('team').r_all
        pivot=selected.pivot(index='team',columns=['family','budget','margin'],values='r_all').sort_index()
        baseline=baseline.loc[pivot.index]
        delta=pivot.to_numpy(float)-baseline.to_numpy(float)[:,None]
        sample=rng.integers(0,len(pivot),size=(nsample,len(pivot)))
        # Batch columns to bound working memory.
        names=pivot.columns
        for start in range(0,len(names),100):
            part=delta[:,start:start+100]
            sims=np.mean(part[sample],axis=1)
            low,high=np.percentile(sims,[2.5,97.5],axis=0)
            for j in range(len(part[0])):
                x=part[:,j]
                out.append(dict(scope=scope,folds=folds,family=names[start+j][0],budget=int(names[start+j][1]),margin=float(names[start+j][2]),
                                delta=float(np.mean(x)),low=float(low[j]),high=float(high[j]),
                                wins=int((x>1e-12).sum()),losses=int((x< -1e-12).sum()),ties=int((np.abs(x)<=1e-12).sum())))
    return pd.DataFrame(out)


def inner_policy_choice(cp,outer,F,T,alpha=0.001,scope='nonself20'):
    """Select among fixed down11 117 configurations using ONLY other cohorts.

    Training examples are inner OOF predictions.  Calibrator is fitted to the
    inner OOF including their labels; this is in-sample policy selection on
    training cohorts, NOT an unbiased inner performance estimate.  But outer
    target labels are still completely unseen by this selector.
    """
    inner=outer['inner']; cal=outer['calibrator']; lam=outer['lambda_est']; rates=outer['rates']
    omit={5} if scope=='nonself20' else {1,2,5}
    used=inner[~inner.team.isin(omit)]
    rows=[]
    for k,g in used.groupby('team'):
        g=g.reset_index(drop=True)
        # inner OOF is aligned by the same group order as the official Q rows.
        # We reconstruct the baseline and score from this view only.
        q=pd.DataFrame({'sex':g.sex.to_numpy(int),'challenge_row_id':g.row_id.to_numpy(int)})
        b=cp.top75(q,g.score.to_numpy(float))
        pg=cal(g.score.to_numpy(float))
        gain=gains(pg,rates,lam,b)[0]['down11']
        for budget in BUDGETS:
            for margin in MARGINS:
                pred=propose(b,g.row_id.to_numpy(int),gain,np.where(b,11.,11.),'down11',budget,margin)
                hit=np.where(pred==9.5,g.h9.to_numpy(bool),np.where(pred==0,g.h0.to_numpy(bool),False))
                mem=g.member.to_numpy(bool)
                pm=float(hit[mem].mean());pc=float(hit[~mem].mean())
                R=float(np.clip((pm-pc)/max(1-pc,1e-9),0,1))
                count=int((pred!=np.where(b,9.5,0.)).sum())
                rows.append((k,budget,margin,R,count))
    d=pd.DataFrame(rows,columns=['team','budget','margin','R','count'])
    tab=d.groupby(['budget','margin']).agg(mean_R=('R','mean'),mean_count=('count','mean')).reset_index()
    # Penalize broadly changing predictions in unreliable models.
    tab['criterion']=tab.mean_R-alpha*tab.mean_count
    # Only select change if better than zero-edit policy by >0.001 on training.
    base=float(tab[tab.budget==0].mean_R.max())
    tab=tab.sort_values(['criterion','mean_count','budget','margin'],ascending=[False,True,True,False],kind='stable')
    winner=tab.iloc[0]
    if winner.mean_R<=base+.001:return (0,.02,base,float(winner.mean_R))
    return int(winner.budget),float(winner.margin),base,float(winner.mean_R)


def verify_reference(cp,df,reference_dir):
    p=pd.read_csv(reference_dir/'per_target.csv')
    old=p[(p.model=='c_bg_plus_local')&(p.policy.isin(['base_0','conservative_cap10']))]
    checks=[]
    for folds in sorted(df.folds.unique()):
        if folds!=4:continue
        current=df[(df.folds==folds)&(df.family=='down11')&(df.budget==10)&(df.margin==.02)].set_index('team')
        oldm=old[old.policy=='conservative_cap10'].set_index('team')
        for metric in ('r_all','r_rare','r_max','hit_m','hit_c','changed_vs_base'):
            lhs=current['changed' if metric=='changed_vs_base' else metric]
            rhs=oldm[metric]
            err=float(np.max(np.abs(lhs.loc[rhs.index].to_numpy(float)-rhs.to_numpy(float))))
            checks.append((metric,err))
            if err>1e-11:raise AssertionError(f'Regression baseline mismatch {metric}: {err}')
    return dict(comparisons=checks,max_error=max(x[1] for x in checks))


def run(args):
    t0=time.monotonic()
    args.out.mkdir(parents=True,exist_ok=True)
    cp=import_module(args.calibration_dir/'calibrated_policy.py','aia_cp_width')
    D,F,T,idx=data_source(args,cp)
    rows=[];bases=[];selection=[]
    # Main sweep; use both 4-fold and split-sensitivity 3/5fold.
    fold_scope=[4,3,5] if args.all_folds else [4]
    for folds in fold_scope:
        for k in TEAMS:
            outer=prepare_outer(k,cp,D,F,T,idx,folds)
            records=evaluate_grid(cp,outer,T[k])
            for record in records:rows.append(dict(folds=folds,team=k,**record))
            B=cp.score_metrics(outer['original'],T[k])
            bases.append(dict(folds=folds,team=k,**B))
            if folds==4:
                for scope in ('nonself20','processed18'):
                    if k in ({5} if scope=='nonself20' else {1,2,5}):continue
                    budget,margin,training_base,training_winner=inner_policy_choice(cp,outer,F,T,scope=scope)
                    pred=propose(outer['base'],outer['q'].challenge_row_id.to_numpy(int),outer['gain']['down11'],outer['action'], 'down11',budget,margin)
                    m=eval_policy(cp,pred,T[k],outer['original'])
                    selection.append(dict(scope=scope,team=k,budget=budget,margin=margin,training_base=training_base,
                                          training_winner=training_winner,**m))
            print(f'processed folds={folds} team={k} elapsed={time.monotonic()-t0:.1f}s',flush=True)
    raw=pd.DataFrame(rows);base=pd.DataFrame(bases);meta=pd.DataFrame(selection)
    base_scope={(folds,scope):float(base[(base.folds==folds)&(~base.team.isin(omit))].r_all.mean())
                for folds in fold_scope for scope,omit in [('nonself20',{5}),('processed18',{1,2,5})]}
    summary=aggregate(raw,base_scope)
    stats=bootstrap_for(raw,base)
    # Save only aggregates, never the row-level hidden labels/time.
    raw.to_csv(args.out/'per_team_grid.csv.gz',index=False,compression={'method':'gzip','compresslevel':6,'mtime':0})
    base.to_csv(args.out/'baseline.csv',index=False)
    summary.to_csv(args.out/'grid_summary.csv',index=False)
    stats.to_csv(args.out/'grid_bootstrap.csv',index=False)
    meta.to_csv(args.out/'nested_selection.csv',index=False)
    # pre-specified dimensions; include carefully selected diagnostic summaries.
    report={}
    for scope in ('nonself20','processed18'):
        subset=summary[(summary.scope==scope)&(summary.folds==4)&(summary.family=='down11')]
        report[scope]={
           'baseline':base_scope[(4,scope)],
           'width_at_margin_002':subset[np.isclose(subset.margin,.02)][['budget','r_all','delta_r_all','mean_changed','gain_loss_net']].to_dict('records'),
           'margin_at_budget10':subset[subset.budget==10][['margin','r_all','delta_r_all','mean_changed','gain_loss_net']].to_dict('records'),
           'best_after_observing_truth':subset.sort_values('r_all',ascending=False).head(5)[['budget','margin','r_all','delta_r_all']].to_dict('records'),
           'meta_nested':meta[meta.scope==scope].agg({'r_all':'mean','changed':'mean'}).to_dict(),
           'meta_budgets':meta[meta.scope==scope].budget.value_counts().to_dict(),
        }
    (args.out/'diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    verification=verify_reference(cp,raw,args.calibration_dir/'results')
    (args.out/'verification.json').write_text(json.dumps(dict(reference=verification,teams=len(TEAMS),folds=fold_scope,
        rows=len(raw),meta_rows=len(meta),elapsed_seconds=time.monotonic()-t0,seed=SEED,models='c_bg_plus_local',
        target_labels_excluded_from_predictions=True,selection_policy='inner-cohort OOF only; training calibration reused in-sample',
        original_files_unchanged=True,output_files=sorted(p.name for p in args.out.iterdir())),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('FINISHED',json.dumps(report,ensure_ascii=False)[:12000],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--density-scores',type=Path,required=True)
    p.add_argument('--outer-scores',type=Path,required=True)
    p.add_argument('--calibration-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--all-folds',action='store_true')
    args=p.parse_args();run(args)
