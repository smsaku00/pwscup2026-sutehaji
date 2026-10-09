#!/usr/bin/env python3
"""PWS CUP 2026 AIA: nested held-out-cohort calibration and rational time answers.

All policy and calibration training excludes the evaluated team's labels and time.
Inner cross-fitting excludes the outer target team from every training fold.
No private team data, CodaBench submission, or hidden target truth is used for prediction.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, importlib.util, json, math, time, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit,logit
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

SEED=20261009
TEAMS=[1,2,3,4,5,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
CLIN=['SBP','TG','HDL','ALT']
COL_M=['member_near4','member_knn4','member_near5','member_partial2','rank_member_near4','rank_member_near5','rank_member_partial2']+[f'c_{n.lower()}' for n in CLIN]
COL_BG=['bg_near4','bg_knn4','bg_near5','rank_bg_near4','rank_bg_knn4']+[f'bg_{n.lower()}' for n in CLIN]
COL_RATIO=['contrast_near4','contrast_knn4','contrast_near5','col4','col2','col1','col4_min','rank_contrast_near4','rank_col4','rank_col2','rank_col1']+[f'ratio_{n.lower()}' for n in CLIN]
SPECS={'c_bg':COL_M+COL_BG+COL_RATIO,
       'c_bg_plus_local':COL_M+COL_BG+COL_RATIO+['knn2_k1','kde4_half']}
POLICIES=['base_0','base_11','platt_rank75_0','isotonic_rank75_0',
          'platt_r_0','platt_r_11','platt_r_allactions','platt_r_safe002',
          'platt_r_safe005','platt_r_soft5','platt_r_soft5_allactions','training_fraction_11']


def import_prior(path:Path):
    sp=importlib.util.spec_from_file_location('cal_prior_experiment',str(path/'experiment.py'))
    obj=importlib.util.module_from_spec(sp);sys.modules[sp.name]=obj;sp.loader.exec_module(obj)
    return obj


def data(root,prior,density_path):
    bg,D=prior.read_data(root)
    features,truths=prior.feature_table(bg,D)
    density=pd.read_csv(density_path).set_index(['team','row_id'])
    for k in TEAMS:
        q=D[k][0]
        lookup=density.loc[[(k,int(i)) for i in q.challenge_row_id]]
        for n in ['knn2_k1','kde4_half']: features[k][n]=lookup[n].to_numpy(float)
        assert len(features[k])==len(truths[k])==326
    return D,features,truths


def model_train(x,y):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.04,max_iter=1500,solver='lbfgs'))
    model.fit(x,y)
    return model


def inner_oof(features,truth,held_out,cols,folds=4):
    """For outer target k, build calibration evidence from j!=k using models
    that were never fit to data from k, even when predicting inner j. """
    other=[j for j in TEAMS if j!=held_out]
    groups=np.concatenate([np.full(326,j,dtype=int) for j in other])
    x=np.concatenate([features[j][cols].to_numpy(float) for j in other])
    y=np.concatenate([truth[j].label.eq('member').to_numpy(int) for j in other])
    h9=np.concatenate([(abs(truth[j].time.to_numpy(float)-9.5)<=.5).astype(int) for j in other])
    h0=np.concatenate([(truth[j].time.to_numpy(float)<=.5).astype(int) for j in other])
    rare=np.concatenate([truth[j].is_rare.to_numpy(int) for j in other])
    sexes=np.concatenate([features[j]['sex'].to_numpy(int) for j in other])
    row_ids=np.concatenate([features[j]['row_id'].to_numpy(int) for j in other])
    probabilities=np.full(len(y),np.nan)
    for tr,va in GroupKFold(n_splits=folds).split(x,y,groups=groups):
        model=model_train(x[tr],y[tr])
        probabilities[va]=model.predict_proba(x[va])[:,1]
    assert np.isfinite(probabilities).all()
    return pd.DataFrame({'team':groups,'score':probabilities,'member':y,'h9':h9,'h0':h0,'rare':rare,'sex':sexes,'row_id':row_ids})


def fit_calibrator(train:pd.DataFrame,scheme='platt'):
    x=logit(np.clip(train.score.to_numpy(),1e-5,1-1e-5))
    y=train.member.to_numpy(int)
    if scheme=='platt':
        model=LogisticRegression(C=1.0,max_iter=500,solver='lbfgs')
        model.fit(x[:,None],y)
        return lambda p: model.predict_proba(logit(np.clip(np.asarray(p),1e-5,1-1e-5))[:,None])[:,1]
    iso=IsotonicRegression(y_min=.01,y_max=.99,out_of_bounds='clip')
    iso.fit(train.score.to_numpy(float),y)
    return lambda p: iso.predict(np.asarray(p,float))


def posterior_projection(q:pd.DataFrame,p,alpha=.5,age_band=5):
    p=np.asarray(p,dtype=float)
    logit0=logit(np.clip(p,.02,.98));logitp=logit0.copy()
    groups={}
    for i,(_,r) in enumerate(q.iterrows()):
        key=(r.prefecture,int(r.sex),int(r.age)//age_band)
        groups.setdefault(key,[]).append(i)
    for ids in groups.values():
        if len(ids)<2 or len(ids)%2:continue
        g=np.array(ids,dtype=int);v=logit0[g];lo,hi=-20.,20.
        for _ in range(40):
            mid=(lo+hi)*.5
            if expit(v-mid).mean()>.5:lo=mid
            else:hi=mid
        logitp[g]-=alpha*(lo+hi)*.5
    return expit(logitp)


def top75(q,p,secondary=None):
    p=np.asarray(p,float);ret=np.zeros(len(p),bool)
    for s in (0,1):
        ix=np.flatnonzero(q.sex.to_numpy(int)==s)
        if secondary is None:
            rank=np.lexsort((q.challenge_row_id.to_numpy(int)[ix],-p[ix]))
        else:
            rank=np.lexsort((q.challenge_row_id.to_numpy(int)[ix],-np.asarray(secondary,float)[ix],-p[ix]))
        ret[ix[rank[:math.ceil(.75*len(ix))]]]=True
    return ret


def optimise_actions(p,hm9,hc9,hm0,hc0,lambda0=.8,coverbase=None,margin=0):
    """Approximate expected R optimisation by Dinkelbach iteration.
    Expected member hit = p * class-dependent P(hit), expected control hit=(1-p)*P(hit).
    AIA member and control populations have exactly 163 rows each.
    Separate outcome models are trained ONLY from outer-training cohorts.
    """
    p=np.asarray(p,float);n=len(p)
    gainm=np.column_stack([p*hm0,p*hm9,np.zeros(n)])
    gainc=np.column_stack([(1-p)*hc0,(1-p)*hc9,np.zeros(n)])
    if coverbase is not None:
        available=np.zeros((n,3),bool)
        available[:,0]=True
        available[:,1]=coverbase
        available[:,2]=coverbase
        # For selected 9.5 rows, allow switch to 0 or 11 only when advantageous by margin.
        # Nonselected rows MUST keep their baseline 0 answer.
    else:available=np.ones((n,3),bool)
    lam=lambda0
    for _ in range(15):
        value=gainm-lam*gainc
        value[~available]=-np.inf
        choice=np.argmax(value,axis=1)
        if coverbase is not None:
            orig=np.where(coverbase,1,0)
            incumbent=value[np.arange(n),orig]
            competitor=value[np.arange(n),choice]
            choice=np.where(competitor>=incumbent+margin,choice,orig)
        pm=float(np.sum(gainm[np.arange(n),choice])/163)
        pc=float(np.sum(gainc[np.arange(n),choice])/163)
        R=float(np.clip((pm-pc)/max(1-pc,1e-8),0,1))
        new_lambda=1-R
        if abs(new_lambda-lam)<1e-8:break
        lam=new_lambda
    pred=np.array([0.,9.5,11.])[choice]
    return pred,dict(predicted_R=R,predicted_pm=pm,predicted_pc=pc,lambda_final=new_lambda)


def expected_hit_rates(inner):
    """Beta(25*.85,25*.15) shrinkage; no target data.
    Uniform hit9,hit0 within each membership class is a small, transparent model.
    """
    mask=inner.member.to_numpy(int).astype(bool)
    h9=inner.h9.to_numpy(int);h0=inner.h0.to_numpy(int)
    rates={}
    for key,sub in [('m',mask),('c',~mask)]:
        n=sub.sum()
        rates[key+'9']=(h9[sub].sum()+25*.85)/(n+25)
        rates[key+'0']=(h0[sub].sum()+25*.015)/(n+25)
    return rates


def expected_lambda(inner):
    """Estimate 1-R using only inner held-out predictions/truth and top75 decisions."""
    M,C=0,0;count=0
    for _,g in inner.groupby('team'):
        chosen=np.zeros(len(g),bool)
        for sex,sub in g.groupby('sex'):
            rank=sub.sort_values(['score','row_id'],ascending=[False,True],kind='stable')
            chosen[g.index.get_indexer(rank.index[:math.ceil(.75*len(sub))])]=True
        d=g.iloc[np.flatnonzero(chosen)]
        M+=((d.member==1)&(d.h9==1)).sum()
        C+=((d.member==0)&(d.h9==1)).sum()
        count+=1
    pm=M/(163*count);pc=C/(163*count)
    return (1-pm)/max(1-pc,1e-9)


def score_metrics(pred,truth):
    truth=truth.reset_index(drop=True)
    p=np.asarray(pred,float)
    member=truth.label.eq('member').to_numpy(bool)
    rare=truth.is_rare.eq(1).to_numpy(bool)
    actual=truth.time.to_numpy(float)
    hit=np.abs(p-actual)<=.5
    pm=float(hit[member].mean());pc=float(hit[~member].mean())
    hmr=int(hit[member&rare].sum());hcr=int(hit[(~member)&rare].sum())
    pr=hmr/13;qr=hcr/13;se=np.sqrt((pr*(1-pr)+qr*(1-qr))/13)
    R=float(np.clip((pm-pc)/max(1-pc,1e-9),0,1))
    RR=float(np.clip((pr-qr-1.64*se)/max(1-qr,1e-9),0,1))
    return dict(r_all=R,r_rare=RR,r_max=max(R,RR),rare_extra=max(R,RR)-R,
                hit_m=int(hit[member].sum()),hit_c=int(hit[~member].sum()),
                hit_m_rare=hmr,hit_c_rare=hcr,n_95=int((p==9.5).sum()),n_0=int((p==0).sum()),n_11=int((p==11).sum()))


def train_best_fraction(inner):
    # Fit on the inner-cohort OOF scores and outcomes. No outer-target data enters selection.
    fracs=[.55,.60,.65,.70,.75,.80,.85,.90,1.0]
    rewards=[]
    for frac in fracs:
        Rs=[]
        for _,g in inner.groupby('team'):
            g=g.reset_index(drop=True)
            selected=np.zeros(len(g),bool)
            for sex,part in g.groupby('sex'):
                rank=part.sort_values(['score','row_id'],ascending=[False,True],kind='stable')
                selected[rank.index[:math.ceil(frac*len(part))]]=True
            # 0 for excluded. All 326 rows; membership/time from other known cohorts only.
            valid=np.where(selected,g.h9.to_numpy(int),g.h0.to_numpy(int)).astype(bool)
            member=g.member.to_numpy(bool)
            pm=float(valid[member].mean());pc=float(valid[~member].mean())
            Rs.append(np.clip((pm-pc)/max(1-pc,1e-9),0,1))
        rewards.append(float(np.mean(Rs)))
    # Occam tie-break: prefer original 75% on near-tie (0.005 difference).
    best=max(rewards)
    valid=[i for i,v in enumerate(rewards) if v>=best-0.005]
    i=min(valid,key=lambda i:(abs(fracs[i]-.75),i))
    return fracs[i],dict(candidates=fracs,scores=rewards)


def run(root,prior_dir,features_csv,outer_csv,out,targets=None,folds=4):
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    prior=import_prior(prior_dir)
    D,F,T=data(root,prior,features_csv)
    outer=pd.read_csv(outer_csv)
    outer=outer[outer.model.isin(SPECS)]
    index=outer.set_index(['team','row_id','model'])['score']
    needed=TEAMS if targets is None else targets
    rows=[];checks=[];bin_rows=[];training_summaries=[]
    for k in needed:
        for method,cols in SPECS.items():
            inner=inner_oof(F,T,k,cols,folds)
            calib=fit_calibrator(inner,'platt')
            iso=fit_calibrator(inner,'isotonic')
            q=D[k][0];target=T[k]
            pp=index.loc[[(k,int(i),method) for i in q.challenge_row_id]].to_numpy(float)
            assert len(pp)==326
            # Crossfit side effect check: explicit outer refit matches published probabilities.
            other=[j for j in TEAMS if j!=k];xtrain=pd.concat([F[j][cols] for j in other],ignore_index=True).to_numpy(float)
            ytrain=np.concatenate([T[j].label.eq('member').to_numpy(int) for j in other])
            pred=model_train(xtrain,ytrain).predict_proba(F[k][cols].to_numpy(float))[:,1]
            err=float(np.max(np.abs(pred-pp)))
            if err>2e-10:raise ValueError(f'Outer baseline changed k={k} method={method} err={err}')
            base=top75(q,pp);priorR=expected_lambda(inner)
            rates=expected_hit_rates(inner)
            p_cal=np.clip(calib(pp),.001,.999);p_iso=np.clip(iso(pp),.001,.999)
            p_proj=posterior_projection(q,p_cal,alpha=.5)
            fraction,train_reward=train_best_fraction(inner)
            train_score=inner.score.to_numpy(float);train_cal=np.clip(calib(train_score),.001,.999)
            truth=target.label.eq('member').to_numpy(int)
            check=dict(team=k,model=method,outer_max_error=err,inner_team_count=int(inner.team.nunique()),
                       inner_derived_from_non_target_labels=True,train_lambda=float(priorR),
                       inner_calib_brier=float(brier_score_loss(inner.member,train_cal)),
                       inner_raw_brier=float(brier_score_loss(inner.member,train_score)),
                       outer_raw_brier=float(brier_score_loss(truth,pp)),
                       outer_platt_brier=float(brier_score_loss(truth,p_cal)),
                       outer_isotonic_brier=float(brier_score_loss(truth,p_iso)),
                       outer_raw_auc=float(roc_auc_score(truth,pp)),
                       p_member_range=[float(p_cal.min()),float(p_cal.max())],
                       train_selected_fraction=fraction)
            checks.append(check)
            for m in [0,1]:
                for j in range(10):
                    ids=np.flatnonzero((truth==m)&(p_cal>=j/10)&(p_cal<(j+1)/10))
                    if len(ids):bin_rows.append(dict(team=k,model=method,member=m,bin=j,n=len(ids),mean_p=float(p_cal[ids].mean())))
            answer={
                'base_0':np.where(base,9.5,0),
                'base_11':np.where(base,9.5,11),
                'platt_rank75_0':np.where(top75(q,p_cal),9.5,0),
                'isotonic_rank75_0':np.where(top75(q,p_iso,secondary=pp),9.5,0),
            }
            common=(rates['m9'],rates['c9'],rates['m0'],rates['c0'])
            p1,diagnostics=optimise_actions(p_cal,*common,lambda0=priorR)
            p1c,_=optimise_actions(p_cal,*common,lambda0=priorR,coverbase=base,margin=.02)
            p1d,_=optimise_actions(p_cal,*common,lambda0=priorR,coverbase=base,margin=.05)
            p2,_=optimise_actions(p_proj,*common,lambda0=priorR)
            p3,_=optimise_actions(p_proj,*common,lambda0=priorR,coverbase=base,margin=0)
            # Conservative pointwise edits: keep the original 75% answers and touch
            # only a few rows with positive, training-derived expected R contribution.
            # No per-target truth enters the gain calculation or the selected budgets.
            def capped_modifications(calibrated, count, gap=0.02, alt=11., per_sex=False):
                gain11=-(calibrated*rates['m9']-priorR*(1-calibrated)*rates['c9'])
                gain0=gain11+calibrated*rates['m0']-priorR*(1-calibrated)*rates['c0']
                gain=gain11 if alt==11 else gain0
                chosen=np.zeros(len(q),bool)
                if per_sex:
                    sectors=[np.flatnonzero(q.sex.to_numpy(int)==j) for j in (0,1)]
                else:sectors=[np.arange(len(q))]
                for sector in sectors:
                    ids=sector[base[sector] & (gain[sector]>gap)]
                    rank=np.lexsort((q.challenge_row_id.to_numpy(int)[ids],-gain[ids]))
                    chosen[ids[rank[:count]]]=True
                return np.where(chosen,alt,np.where(base,9.5,0.)),int(chosen.sum())
            limited={}
            for budget in (2,5,10,20):
                ans,_=capped_modifications(p_cal,budget,.02,11)
                answer_key='conservative_cap'+str(budget)
                # Store below after the answer dictionary is initialised.
                limited[answer_key]=ans
            x,_=capped_modifications(p_cal,3,.02,11,per_sex=True)
            limited['conservative_sex3']=x
            x,_=capped_modifications(p_proj,5,.02,11)
            limited['conservative_proj5']=x
            x,_=capped_modifications(p_cal,5,.02,0)
            limited['conservative_cap5_to0']=x
            for budget in (2,5,10):
                x,_=capped_modifications(p_iso,budget,.02,11)
                limited['isotonic_cap'+str(budget)]=x
            # A deliberately stringent agreement variant.  Different calibrators
            # must both signal improvement on the same original 9.5-year row.
            gainp=-(p_cal*rates['m9']-priorR*(1-p_cal)*rates['c9'])
            gaini=-(p_iso*rates['m9']-priorR*(1-p_iso)*rates['c9'])
            agreeing=np.where((gainp>.02)&(gaini>.02),np.minimum(gainp,gaini),-np.inf)
            best=np.flatnonzero(base & np.isfinite(agreeing))
            order=np.lexsort((q.challenge_row_id.to_numpy(int)[best],-agreeing[best]))
            chosen=best[order[:10]]
            x=np.where(base,9.5,0.).astype(float);x[chosen]=11.
            limited['dual_calibrator_cap10']=x
            answer.update(limited)
            answer['platt_r_allactions']=p1
            answer['platt_r_safe002']=p1c
            answer['platt_r_safe005']=p1d
            answer['platt_r_soft5_allactions']=p2
            answer['platt_r_soft5']=p3
            answer['platt_r_0']=np.where(p1==11,0,p1)
            answer['platt_r_11']=np.where(p1==0,11,p1)
            ix=np.zeros(326,bool)
            for sex in (0,1):
                sids=np.flatnonzero(q.sex.to_numpy(int)==sex)
                order=np.lexsort((q.challenge_row_id.to_numpy(int)[sids],-pp[sids]));ix[sids[order[:math.ceil(fraction*len(sids))]]]=True
            answer['training_fraction_11']=np.where(ix,9.5,11)
            for name,values in answer.items():
                metrics=score_metrics(values,target)
                base_answer=answer['base_0']
                rows.append(dict(team=k,model=method,policy=name,changed_vs_base=int(np.sum(values!=base_answer)),
                                 selected_fraction=float((values==9.5).mean()),
                                 **metrics))
            training_summaries.append(dict(team=k,model=method,lambda_approx=priorR,**rates,
                                          frac=fraction,optimistic_train_R=float(max(train_reward['scores'])),
                                          predicted_R=diagnostics['predicted_R']))
        print('finished target',k,'seconds',round(time.monotonic()-start,2),flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(out/'per_target.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'nested_calibration.csv',index=False)
    pd.DataFrame(bin_rows).to_csv(out/'calibration_bins.csv',index=False)
    pd.DataFrame(training_summaries).to_csv(out/'training_params.csv',index=False)
    group=[]
    for scope,omit in [('nonself20',{5}),('processed18',{1,2,5})]:
        selected=raw[~raw.team.isin(omit)]
        if len(selected)==0:continue
        g=selected.groupby(['model','policy']).agg(n=('team','count'),r_all=('r_all','mean'),r_rare=('r_rare','mean'),r_max=('r_max','mean'),
                        rare_extra=('rare_extra','mean'),m_hit=('hit_m','mean'),c_hit=('hit_c','mean'),
                        rare_m_hit=('hit_m_rare','mean'),rare_c_hit=('hit_c_rare','mean'),changed=('changed_vs_base','mean'),
                        selected_frac=('selected_fraction','mean')).reset_index()
        g.insert(0,'scope',scope);group.append(g)
    total=pd.concat(group,ignore_index=True);total.to_csv(out/'summary.csv',index=False)
    test=[];rng=np.random.default_rng(SEED)
    for scope,omit in [('nonself20',{5}),('processed18',{1,2,5})]:
        sel=raw[~raw.team.isin(omit)]
        for method in SPECS:
            s=sel[sel.model==method].pivot(index='team',columns='policy',values='r_all')
            base=s['base_0'];n=len(s)
            sample=rng.integers(0,n,size=(15000,n))
            for policy in s.columns:
                if policy=='base_0':continue
                delta=(s[policy]-base).to_numpy(float)
                boot=np.quantile(delta[sample].mean(axis=1),[.025,.975])
                test.append(dict(scope=scope,method=method,policy=policy,mean_change=float(delta.mean()),low=float(boot[0]),high=float(boot[1]),
                                 wins=int((delta>1e-10).sum()),losses=int((delta<-1e-10).sum()),ties=int((abs(delta)<=1e-10).sum())))
    pd.DataFrame(test).to_csv(out/'paired_bootstrap.csv',index=False)
    program=Path(__file__).read_bytes()
    (out/'manifest.json').write_text(json.dumps(dict(seed=SEED,nested_outer_excluded_target=True,inner_folds=folds,
                                 sha256=hashlib.sha256(program).hexdigest(),n_teams=len(needed),n_rows=len(raw),
                                 features=list(SPECS.keys()),policies=sorted(raw.policy.unique().tolist()),scoring='official prelim raw R & rare z=1.64; no phi',
                                 time_seconds=time.monotonic()-start),ensure_ascii=False,indent=2)+'\n')
    print(total[total.scope=='nonself20'].sort_values(['model','r_all'],ascending=[True,False]).round(5).to_string(index=False),flush=True)
    return raw,total


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--density-scores',type=Path,required=True)
    ap.add_argument('--outer-scores',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--targets',type=int,nargs='*',default=None)
    ap.add_argument('--folds',type=int,default=4)
    args=ap.parse_args()
    run(args.root,args.prior,args.density_scores,args.outer_scores,args.out,args.targets,args.folds)
