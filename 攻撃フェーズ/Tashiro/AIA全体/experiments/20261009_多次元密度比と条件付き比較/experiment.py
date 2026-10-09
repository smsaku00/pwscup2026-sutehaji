#!/usr/bin/env python3
"""AIA joint/conditional density-ratio benchmark; official preliminary datasets only.

Only Q, C, A_bg are used to produce unsupervised density scores. True membership,
rare flags and time enter isolated evaluation and other-cohort supervised fitting.
Do not infer that C/BG density ratio is a calibrated membership probability.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import logsumexp, expit, logit
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

SEED=20261009
TEAMS=[1,2,3,4,5,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
COLS=['SBP','TG','HDL','ALT']
PAIRS=list(combinations(range(4),2))
H_CONFIG={'kde_1d':1.0,'kde_2d':1.0,'kde_4d':1.0,'kde_4d_half':0.5,'kde_4d_double':2.0}
# kNN zero distance is capped, identical arbitrary records generate strong local evidence.
EPS=1e-5

def csv_read(root,k):
    d=root/'公式データ'/'予備戦'
    q=pd.read_csv(d/f'PWSCup2026_prelim_attack/aia_challenge_{k}.csv').sort_values('challenge_row_id').reset_index(drop=True)
    c=pd.read_csv(d/f'PWSCup2026_prelim_attack/C_{k}.csv')
    t=pd.read_csv(d/f'結果/正解データ/aia_truth_{k}.csv').sort_values('challenge_row_id').reset_index(drop=True)
    assert len(q)==326 and len(c)==1049 and len(t)==326 and q.challenge_row_id.equals(t.challenge_row_id)
    assert (t.label=='member').sum()==163 and (t.label=='control').sum()==163
    assert ((t.label=='member') & (t.is_rare==1)).sum()==13 and ((t.label=='control') & (t.is_rare==1)).sum()==13
    return q,c,t

def matrix(df):
    arr=df[COLS].to_numpy(float).copy()
    arr[:,1]=np.log1p(np.maximum(arr[:,1],0))
    arr[:,3]=np.log1p(np.maximum(arr[:,3],0))
    return arr

def kd_ratio(x, cx, bx,k,dim):
    if len(cx)<k or len(bx)<k:return np.zeros(len(x))
    a=cKDTree(cx).query(x,k=k,workers=1)[0]
    b=cKDTree(bx).query(x,k=k,workers=1)[0]
    if k>1:a=a[:,-1];b=b[:,-1]
    # Same k on both sources, include sample-size correction for true density estimator.
    return np.clip(np.log(len(bx)/len(cx))+dim*(np.log(np.maximum(b,EPS))-np.log(np.maximum(a,EPS))),-30,30)

def kde_log_ratio(d2c,d2b,nc,nb,h,weightsC=None,weightsB=None):
    # Weights implement conditional density within fixed sex; normalize by effective mass.
    vc=-d2c/(2*h*h)
    vb=-d2b/(2*h*h)
    if weightsC is None:
        c=logsumexp(vc,axis=1)-np.log(nc)
        b=logsumexp(vb,axis=1)-np.log(nb)
    else:
        c=logsumexp(vc+np.log(np.maximum(weightsC,1e-12)),axis=1)-logsumexp(np.log(np.maximum(weightsC,1e-12)),axis=1)
        b=logsumexp(vb+np.log(np.maximum(weightsB,1e-12)),axis=1)-logsumexp(np.log(np.maximum(weightsB,1e-12)),axis=1)
    return np.clip(c-b,-30,30)

def conditional_weight(q,source,age_h=5,other_pref=.15):
    delta=(q.age.to_numpy(float)[:,None]-source.age.to_numpy(float)[None,:])/age_h
    pref=np.where(q.prefecture.to_numpy()[:,None]==source.prefecture.to_numpy()[None,:],1.,other_pref)
    return np.exp(-.5*delta*delta)*pref

def density_features(q,c,bg,scale):
    qx=matrix(q)/scale;cx=matrix(c)/scale;bx=matrix(bg)/scale
    outputs={name:np.zeros(len(q),float) for name in [
        'kde1_mean','kde1_top2','kde2_mean6','kde2_top2','kde2_fixed2',
        'kde4_joint','kde4_half','kde4_double','kde4_cond5','kde4_cond10',
        'kde2_cond_mean6','knn2_k1','knn2_k5','knn4_k1','knn4_k5']}
    sexq=q.sex.to_numpy(int);sexc=c.sex.to_numpy(int);sexb=bg.sex.to_numpy(int)
    for s in (0,1):
        iq=np.flatnonzero(sexq==s);ic=np.flatnonzero(sexc==s);ib=np.flatnonzero(sexb==s)
        if not len(iq):continue
        assert len(ic)>50 and len(ib)>200, (s,len(ic),len(ib))
        qs=q.iloc[iq];cs=c.iloc[ic];bs=bg.iloc[ib]
        X=qx[iq];C=cx[ic];B=bx[ib]
        distc=(X[:,None,:]-C[None,:,:])**2
        distb=(X[:,None,:]-B[None,:,:])**2
        dC4=distc.sum(axis=2);dB4=distb.sum(axis=2)
        # Same bandwidth for C and BG in a sex group avoids arbitrary ratio scaling.
        h1=len(ic)**(-1/5);h2=len(ic)**(-1/6);h4=len(ic)**(-1/8)
        ratios1=[];ratios2=[];knn1=[];knn5=[];cond2=[]
        for j in range(4):
            ratios1.append(kde_log_ratio(distc[:,:,j],distb[:,:,j],len(ic),len(ib),h1))
        r1=np.column_stack(ratios1)
        outputs['kde1_mean'][iq]=r1.mean(axis=1)
        outputs['kde1_top2'][iq]=np.sort(r1,axis=1)[:,-2:].mean(axis=1)
        wc=conditional_weight(qs,cs);wb=conditional_weight(qs,bs)
        for j,k in PAIRS:
            ac=distc[:,:,j]+distc[:,:,k];ab=distb[:,:,j]+distb[:,:,k]
            ratios2.append(kde_log_ratio(ac,ab,len(ic),len(ib),h2))
            cond2.append(kde_log_ratio(ac,ab,len(ic),len(ib),h2,wc,wb))
            knn1.append(kd_ratio(X[:,[j,k]],C[:,[j,k]],B[:,[j,k]],1,2))
            knn5.append(kd_ratio(X[:,[j,k]],C[:,[j,k]],B[:,[j,k]],5,2))
        r2=np.column_stack(ratios2)
        outputs['kde2_mean6'][iq]=r2.mean(axis=1)
        outputs['kde2_top2'][iq]=np.sort(r2,axis=1)[:,-2:].mean(axis=1)
        outputs['kde2_fixed2'][iq]=(r2[:,PAIRS.index((0,1))]+r2[:,PAIRS.index((2,3))])/2
        outputs['kde2_cond_mean6'][iq]=np.column_stack(cond2).mean(axis=1)
        outputs['knn2_k1'][iq]=np.column_stack(knn1).mean(axis=1)
        outputs['knn2_k5'][iq]=np.column_stack(knn5).mean(axis=1)
        outputs['kde4_joint'][iq]=kde_log_ratio(dC4,dB4,len(ic),len(ib),h4)
        outputs['kde4_half'][iq]=kde_log_ratio(dC4,dB4,len(ic),len(ib),h4*.5)
        outputs['kde4_double'][iq]=kde_log_ratio(dC4,dB4,len(ic),len(ib),h4*2)
        outputs['kde4_cond5'][iq]=kde_log_ratio(dC4,dB4,len(ic),len(ib),h4,wc,wb)
        w10c=conditional_weight(qs,cs,age_h=10,other_pref=.3)
        w10b=conditional_weight(qs,bs,age_h=10,other_pref=.3)
        outputs['kde4_cond10'][iq]=kde_log_ratio(dC4,dB4,len(ic),len(ib),h4,w10c,w10b)
        outputs['knn4_k1'][iq]=kd_ratio(X,C,B,1,4)
        outputs['knn4_k5'][iq]=kd_ratio(X,C,B,5,4)
    for k,v in outputs.items():assert np.isfinite(v).all(),k
    return pd.DataFrame(outputs)

def direct_ratio(q,c,bg,scale):
    """Discriminate public C rows vs BG rows; does NOT train on AIA membership."""
    xc=matrix(c)/scale;xb=matrix(bg)/scale;xq=matrix(q)/scale
    outcome={'direct_lr':np.zeros(len(q)),'direct_hgb':np.zeros(len(q))}
    for s in (0,1):
        ic=np.flatnonzero(c.sex.to_numpy(int)==s);ib=np.flatnonzero(bg.sex.to_numpy(int)==s);iq=np.flatnonzero(q.sex.to_numpy(int)==s)
        if not len(iq):continue
        pref=str(q.prefecture.iloc[iq[0]])
        def f(df, mat, idx):
            rows=df.iloc[idx]
            return np.column_stack([mat[idx], rows.age.to_numpy(float)/10,
                                    (rows.prefecture.to_numpy()==pref).astype(float)])
        C=f(c,xc,ic);B=f(bg,xb,ib);Q=f(q,xq,iq)
        X=np.vstack([C,B]);y=np.concatenate([np.ones(len(C)),np.zeros(len(B))]);wt=np.concatenate([np.full(len(C),.5/len(C)),np.full(len(B),.5/len(B))]);wt*=len(y)
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=1000))
        model.fit(X,y,logisticregression__sample_weight=wt)
        p=model.predict_proba(Q)[:,1]
        outcome['direct_lr'][iq]=logit(np.clip(p,1e-5,1-1e-5))
        hgb=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=12,min_samples_leaf=40,
                l2_regularization=10,learning_rate=.05,random_state=SEED)
        hgb.fit(X,y,sample_weight=wt)
        p=hgb.predict_proba(Q)[:,1]
        outcome['direct_hgb'][iq]=logit(np.clip(p,1e-5,1-1e-5))
    return pd.DataFrame(outcome)

def soft_group_score(q,score,alpha=.5):
    # Rank -> logit, centered to group expectation 50% only for even groups >=4.
    score=np.asarray(score,float)
    ranks=np.zeros(len(score))
    for sex in (0,1):
        ix=np.flatnonzero(q.sex.to_numpy(int)==sex)
        ranks[ix]=(rankdata(score[ix],method='average')-.5)/len(ix)
    logits=logit(np.clip(ranks,.01,.99));out=logits.copy()
    groups={}
    for i,r in q.iterrows():groups.setdefault((str(r.prefecture),int(r.sex),int(r.age//5)),[]).append(i)
    for g in groups.values():
        if len(g)<4 or len(g)%2:continue
        a=np.array(g,dtype=int);v=logits[a]
        lo,hi=-20.,20.
        for _ in range(42):
            mid=(lo+hi)/2
            if expit(v-mid).mean()>.5:lo=mid
            else:hi=mid
        out[a]=v-alpha*(lo+hi)/2
    return out

def top75(q,score):
    selected=np.zeros(len(q),bool)
    for s in (0,1):
        ix=np.flatnonzero(q.sex.to_numpy(int)==s)
        order=np.lexsort((q.challenge_row_id.to_numpy(int)[ix],-np.asarray(score)[ix]))
        selected[ix[order[:math.ceil(.75*len(ix))]]]=True
    return selected

def metric(pred,t):
    assert len(pred)==len(t)
    member=t.label.eq('member').to_numpy();control=~member;rare=t.is_rare.to_numpy(bool)
    hit=np.abs(pred-t.time.to_numpy(float))<=.5
    mm=hit[member].sum();cc=hit[control].sum();mrm=hit[member&rare].sum();crc=hit[control&rare].sum()
    pm,pc=mm/163,cc/163;pmr,pcr=mrm/13,crc/13
    rall=float(np.clip((pm-pc)/max(1-pc,1e-9),0,1));se=np.sqrt((pmr*(1-pmr)+pcr*(1-pcr))/13)
    rr=float(np.clip((pmr-pcr-1.64*se)/max(1-pcr,1e-9),0,1))
    return dict(r_all=rall,r_rare=rr,r_max=max(rall,rr),rare_extra=max(0,rr-rall),p_m=pm,p_c=pc,
                hit_m=int(mm),hit_c=int(cc),hit_m_rare=int(mrm),hit_c_rare=int(crc))

def heldout_classifiers(data,truths):
    # Input features ONLY independent of membership labels. The heldout cohort does not train its model.
    specs={'density_logistic':(['kde2_mean6','kde4_joint','kde4_cond5','knn2_k5','knn4_k5','direct_lr','direct_hgb'], 'lr'),
           'distance_density_logistic':(['prior_near4','prior_col4','prior_col2','kde2_mean6','kde4_joint','kde4_cond5','knn2_k5','direct_lr'], 'lr'),
           'distance_density_hgb':(['prior_near4','prior_col4','prior_col2','kde2_mean6','kde4_joint','kde4_cond5','knn2_k5','direct_lr'], 'hgb')}
    pred={k:{} for k in data}
    for name,(cols,kind) in specs.items():
        for k in data:
            train=[i for i in data if i!=k]
            xx=pd.concat([data[i][cols] for i in train],ignore_index=True).to_numpy()
            yy=np.concatenate([truths[i].label.eq('member').to_numpy(int) for i in train])
            assert len(xx)==326*(len(data)-1)
            if kind=='lr':model=make_pipeline(StandardScaler(),LogisticRegression(C=.04,max_iter=2000,random_state=SEED))
            else:model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=10,min_samples_leaf=85,
                            l2_regularization=20,learning_rate=.045,random_state=SEED)
            model.fit(xx,yy)
            pred[k][name]=model.predict_proba(data[k][cols].to_numpy())[:,1]
    return pred,specs

def auc_safe(y,s):
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else np.nan

def contrast_2persons(q,s,t):
    group={}
    for i,r in q.iterrows():group.setdefault((str(r.prefecture),int(r.sex),int(r.age//5)),[]).append(i)
    correct=0;checked=0;ambiguous=0;actual_groups=0
    for ix in group.values():
        if len(ix)!=2:continue
        checked+=1
        if (t.label.iloc[ix]=='member').sum()!=1:ambiguous+=1;continue
        actual_groups+=1
        high=ix[int(s[ix[1]]>s[ix[0]])]
        correct+=(t.label.iloc[high]=='member')
    return dict(pairs_found=checked,member_control_pairs=actual_groups,orientation_correct=int(correct),pairs_unbalanced=ambiguous)

def bootstrap_differences(table,a,b,col='r_all',teams=None,seed=SEED,n=10000):
    teams=teams if teams is not None else [k for k in TEAMS if k!=5]
    sub=table[table.team.isin(teams)].pivot(index='team',columns='method',values=col)
    diff=sub[a].to_numpy()-sub[b].to_numpy();rng=np.random.default_rng(seed)
    ix=rng.integers(0,len(diff),size=(n,len(diff)))
    lo,hi=np.quantile(diff[ix].mean(axis=1),[.025,.975]);return dict(delta_mean=float(diff.mean()),CI95=[float(lo),float(hi)],wins=int((diff>1e-10).sum()),losses=int((diff< -1e-10).sum()),ties=int((abs(diff)<=1e-10).sum()))

def main(root,out):
    tic=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    d=root/'公式データ'/'予備戦'
    bg=pd.read_csv(d/'PWSCup2026_prelim_common/A_bg.csv')
    prior=out/'prior_public_scores.csv'
    if not prior.exists(): raise FileNotFoundError('Put existing OOF public_scores.csv as out/prior_public_scores.csv before running')
    ps=pd.read_csv(prior).set_index(['team','row_id'])
    scalemat=matrix(bg);scale=np.maximum(np.quantile(scalemat,.75,axis=0)-np.quantile(scalemat,.25,axis=0),1e-5)
    data={}; truths={};qs={};source_hash={};directs={}
    for k in TEAMS:
        q,c,t=csv_read(root,k);truths[k]=t;qs[k]=q
        paths=[d/f'PWSCup2026_prelim_attack/aia_challenge_{k}.csv',d/f'PWSCup2026_prelim_attack/C_{k}.csv',d/f'結果/正解データ/aia_truth_{k}.csv']
        for p in paths:source_hash[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
        feat=density_features(q,c,bg,scale)
        direct=direct_ratio(q,c,bg,scale)
        feat=pd.concat([feat,direct],axis=1)
        priorrows=ps.loc[[(k,int(i)) for i in q.challenge_row_id]]
        feat['prior_near4']=priorrows.near4.to_numpy(float)
        feat['prior_col4']=priorrows.col4.to_numpy(float)
        feat['prior_col2']=priorrows.col2.to_numpy(float)
        assert np.isfinite(feat.to_numpy()).all()
        data[k]=feat
        print(f'Density/Direct fitted team {k} ({len(feat.columns)} evidence scores), {time.monotonic()-tic:.1f}s',flush=True)
    bg_path=d/'PWSCup2026_prelim_common/A_bg.csv';source_hash[str(bg_path.relative_to(root))]=hashlib.sha256(bg_path.read_bytes()).hexdigest()
    supervised,specs=heldout_classifiers(data,truths)
    print(f'All heldout estimators complete, {time.monotonic()-tic:.1f}s',flush=True)
    for k in TEAMS:
        for name,prob in supervised[k].items():data[k][name]=prob
    methods=list(data[TEAMS[0]].columns)
    perrow=[];met=[];auc=[];orient=[];selectionMasks={}
    for k in TEAMS:
        q=qs[k];t=truths[k];member=t.label.eq('member').to_numpy();israre=t.is_rare.eq(1).to_numpy()
        selectionMasks[k]={}
        for method in methods:
            s=data[k][method].to_numpy(float)
            auc.append(dict(team=k,method=method,auc=auc_safe(member,s),auc_rare=auc_safe(member[israre],s[israre])))
            entry=dict(team=k,method=method,**contrast_2persons(q,s,t));orient.append(entry)
            selectionMasks[k][method]=top75(q,s)
            for response in ('0','11'):
                pred=np.where(selectionMasks[k][method],9.5,0 if response=='0' else 11).astype(float)
                met.append(dict(team=k,method=method,variant='base'+response,**metric(pred,t)))
            if method in ('density_logistic','distance_density_logistic','distance_density_hgb','kde2_mean6','kde4_joint','kde4_cond5','knn4_k5','prior_col2'):
                soft=soft_group_score(q,s)
                pred=np.where(top75(q,soft),9.5,0)
                met.append(dict(team=k,method=method,variant='soft05_0',**metric(pred,t)))
                pred=np.where(top75(q,soft),9.5,11)
                met.append(dict(team=k,method=method,variant='soft05_11',**metric(pred,t)))
        df=data[k].copy();df.insert(0,'row_id',q.challenge_row_id.to_numpy());df.insert(0,'team',k)
        perrow.append(df)
    result=pd.DataFrame(met);adf=pd.DataFrame(auc);orient=pd.DataFrame(orient)
    allscores=pd.concat(perrow,ignore_index=True)
    allscores.to_csv(out/'density_public_scores.csv',index=False)
    result.to_csv(out/'per_team_metrics.csv',index=False)
    adf.to_csv(out/'per_team_auc.csv',index=False)
    orient.to_csv(out/'pair_orientation.csv',index=False)
    summaries=[]
    for label,keys in [('20target',[k for k in TEAMS if k!=5]),('processed18',[k for k in TEAMS if k not in (1,2,5)])]:
        sub=result[result.team.isin(keys)];p=sub.groupby(['method','variant'],sort=True).agg(
            r_all=('r_all','mean'),r_rare=('r_rare','mean'),r_max=('r_max','mean'),rare_extra=('rare_extra','mean'),
            p_m=('p_m','mean'),p_c=('p_c','mean'),hit_m=('hit_m','mean'),hit_c=('hit_c','mean'),n=('team','size')).reset_index()
        p.insert(0,'scope',label);summaries.append(p)
    summary=pd.concat(summaries,ignore_index=True);summary.to_csv(out/'summary.csv',index=False)
    aucSummary=[]
    for label,keys in [('20target',[k for k in TEAMS if k!=5]),('processed18',[k for k in TEAMS if k not in (1,2,5)])]:
        o=adf[adf.team.isin(keys)].groupby('method').agg(auc=('auc','mean'),auc_rare=('auc_rare','mean'),n=('team','size')).reset_index()
        o.insert(0,'scope',label);aucSummary.append(o)
    pd.concat(aucSummary).to_csv(out/'auc_summary.csv',index=False)
    # Calibration is important: density logits cannot be read as membership probabilities.
    calibration=[]
    for name in ('density_logistic','distance_density_logistic','distance_density_hgb'):
        for k in TEAMS:
            if k==5:continue
            prob=data[k][name].to_numpy()
            label=truths[k].label.eq('member').to_numpy()
            for low,up in zip([0,.1,.2,.3,.4,.5,.6,.7,.8,.9],[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.01]):
                idx=(prob>=low)&(prob<up)
                if idx.sum():calibration.append(dict(method=name,team=k,low=low,high=up,n=int(idx.sum()),member=int(label[idx].sum())))
    pd.DataFrame(calibration).to_csv(out/'calibration_counts.csv',index=False)
    checks={}
    base=result[result.variant=='base0'];soft=result[result.variant=='soft05_0']
    for a,b,variant in [('kde2_mean6','prior_col2','base0'),('kde4_joint','prior_col2','base0'),
                        ('kde4_cond5','kde4_joint','base0'),('knn4_k5','prior_col2','base0'),
                        ('direct_hgb','prior_col2','base0'),('distance_density_logistic','prior_col2','base0')]:
        checks[f'{a}_vs_{b}_{variant}']={scope:bootstrap_differences(result[result.variant==variant],a,b,teams=keys) for scope,keys in [('20target',[k for k in TEAMS if k!=5]),('processed18',[k for k in TEAMS if k not in (1,2,5)])]}
    for a in ('density_logistic','distance_density_logistic','distance_density_hgb','kde4_cond5'):
        s=result[result.method==a]
        p=s.pivot(index='team',columns='variant',values='r_all')
        for label,keys in [('20target',[k for k in TEAMS if k!=5]),('processed18',[k for k in TEAMS if k not in(1,2,5)])]:
            delta=p.loc[keys,'soft05_0']-p.loc[keys,'base0']; rng=np.random.default_rng(SEED)
            idx=rng.integers(0,len(delta),size=(10000,len(delta)));ci=np.quantile(delta.to_numpy()[idx].mean(axis=1),[.025,.975])
            checks[f'{a}_soft05_vs_base_{label}']={'mean':float(delta.mean()),'CI95':ci.tolist(),'wins':int((delta>1e-10).sum()),'losses':int((delta<-1e-10).sum())}
    (out/'paired_statistics.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False)+'\n')
    (out/'manifest.json').write_text(json.dumps({'seed':SEED,'teams':TEAMS,'primary_cohorts':[k for k in TEAMS if k!=5],
        'processed18':[k for k in TEAMS if k not in(1,2,5)],'input_sha256':source_hash,
        'baseline':'sex-specific ceil75% ->9.5, else 0 or11','score_methods':methods,'supervised_models':specs,
        'seconds':time.monotonic()-tic,'submission_made':False,'member_truth_used_for_unsupervised_scores':False,
        'label_used_only_for_other_cohort_supervised_training_and_metric':True,
        'true_rare_used_only_for_metrics':True,'study_exploratory':True},indent=2,ensure_ascii=False)+'\n')
    print('Completed. AUC leaderboard:',flush=True)
    print(pd.concat(aucSummary).query("scope == '20target'").sort_values('auc',ascending=False).head(20).to_string(index=False),flush=True)
    print('Rall leaderboard:',flush=True)
    print(summary.query("scope == '20target' and variant == 'base0'").sort_values('r_all',ascending=False).head(20).to_string(index=False),flush=True)
    print('seconds',time.monotonic()-tic,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();main(args.root.resolve(),args.out.resolve())
