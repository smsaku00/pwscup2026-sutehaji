#!/usr/bin/env python3
"""Direct member-control pair comparisons in public 5-year strata, cohort-held-out."""
from __future__ import annotations
from pathlib import Path
import itertools,math,argparse
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.special import expit
from scipy.stats import rankdata
from experiment import read_data,feature_table, TEAMS,SEED,top75,pairs,metrics

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).parent/'input');parser.add_argument('--out',type=Path,default=Path(__file__).parent/'output');args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True);bg,D=read_data(args.root);f,truth=feature_table(bg,D)
cols=['member_near4','member_knn4','member_near5','member_partial2',
      'bg_near4','bg_knn4','contrast_near4','contrast_near5','contrast_otherc',
      'col4','col2','col1','col4_min']+[f'ratio_{col}'for col in ['sbp','tg','hdl','alt']]
feat={k:f[k][cols].to_numpy(float) for k in TEAMS}
# Using genuine group memberships of OTHER cohorts to construct training pairs.
rng=np.random.default_rng(SEED); pairsTrain={}
for k in TEAMS:
    arr=[]
    Q=D[k][0]; t=truth[k]
    for inds in pairs(Q,5).values():
        mm=[i for i in inds if t.label.iloc[i]=='member']; cc=[i for i in inds if t.label.iloc[i]=='control']
        comb=list(itertools.product(mm,cc))
        if len(comb)>120:
            selected=rng.choice(len(comb),size=120,replace=False)
            comb=[comb[int(i)] for i in selected]
        if comb:
            a=np.stack([feat[k][i]-feat[k][j] for i,j in comb])
            arr.append(a)
    pairsTrain[k]=np.vstack(arr)
result=[];predictions=[];orientation=[]
for k in TEAMS:
    training=np.vstack([pairsTrain[j] for j in TEAMS if j!=k])
    tr=np.vstack([training,-training]);ytr=np.concatenate([np.ones(len(training)),np.zeros(len(training))]).astype(int)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.02,fit_intercept=False,max_iter=800))
    model.fit(tr,ytr)
    # Score for each individual: projection onto learned pair-comparison directions.
    scale=model.named_steps['standardscaler'];coeff=model.named_steps['logisticregression'].coef_[0]
    # On matched centered strata, intercept cancels; use centered features to reduce cohort shifts.
    Q=D[k][0];g=pairs(Q,5);X=feat[k]
    cen=X.copy()
    for inds in g.values(): cen[inds]-=X[inds].mean(axis=0)
    scores=cen/scale.scale_@coeff
    assert np.isfinite(scores).all()
    t=truth[k];labels=t.label.eq('member').to_numpy(bool)
    selected=top75(Q,scores)
    base=metrics(np.where(np.isin(np.arange(326),list(selected)),9.5,0.),t)
    near=f[k].col2.to_numpy(float)
    # Compare large-contrast within-group low candidates only; no forced half of Q.
    edits=[];values=np.where(np.isin(np.arange(326),list(selected)),9.5,0.)
    for inds in g.values():
        if len(inds)%2 or len(inds)<2:continue
        ordered=sorted(inds,key=lambda i:scores[i]);m=len(inds)//2
        for i in range(m):
            lo,hi=ordered[i],ordered[-1-i]
            # Pairwise logistic comparison score is NOT a calibrated individual P(member).
            pr=expit(scores[hi]-scores[lo]);
            if pr<.80 or lo not in selected:continue
            if near[hi]<=near[lo]:continue
            values[lo]=11.;edits.append(lo)
    guarded=metrics(values,t)
    orientationValid=[];calib=[]
    for inds in g.values():
        if len(inds)!=2:continue
        i,j=inds
        decision=np.argmax(scores[inds]);correct=int(labels[inds[decision]])
        p=expit(abs(scores[i]-scores[j]));
        orientationValid.append((correct,p))
        orientation.append({'team':k,'correct':correct,'p_direction':p,'direction_margin':abs(scores[i]-scores[j])})
    result.append({'team':k,'pair_score_auc':roc_auc_score(labels,scores),
                   'baseline_r_all':base['r_all'],'guarded_r_all':guarded['r_all'],
                   'baseline_r_rare':base['r_rare'],'guarded_r_rare':guarded['r_rare'],
                   'baseline_rmax':base['r_max'],'guarded_rmax':guarded['r_max'],
                   'pair_groups':len(orientationValid),'pair_accuracy':np.mean([i[0] for i in orientationValid]) if orientationValid else np.nan,
                   'guarded_changes':len(edits),
                   'guarded_lost_member_hit':int(sum((t.label.iloc[i]=='member') and abs(9.5-t.time.iloc[i])<=.5 for i in edits)),
                   'guarded_removed_control_hit':int(sum((t.label.iloc[i]=='control') and abs(9.5-t.time.iloc[i])<=.5 for i in edits)),})
    predictions.extend([{'team':k,'row_id':int(qid),'pair_relative_score':float(p)} for qid,p in zip(Q.challenge_row_id,scores)])
res=pd.DataFrame(result);res.to_csv(out/'pairwise_target.csv',index=False)
pd.DataFrame(orientation).to_csv(out/'pairwise_orientation.csv',index=False)
pd.DataFrame(predictions).to_csv(out/'pairwise_public_scores.csv',index=False)
a=res[res.team.ne(5)];b=res[~res.team.isin([1,2,5])]
print('Pairwise model:')
for name,z in [('20',a),('18',b)]:
 print(name,'AUC',z.pair_score_auc.mean(),'Rall before/after',z.baseline_r_all.mean(),z.guarded_r_all.mean(),
       'Rrare before/after',z.baseline_r_rare.mean(),z.guarded_r_rare.mean(),
       'changes',z.guarded_changes.sum(),'actual outcome: lostM',z.guarded_lost_member_hit.sum(),'removedC',z.guarded_removed_control_hit.sum())
print('orientation pairs:',a.pair_groups.sum(),'accuracy',pd.DataFrame(orientation).query('team!=5').correct.mean())
for cut in [0.5,0.65,0.7,0.8,0.9]:
 x=pd.DataFrame(orientation).query('team!=5 and p_direction>=@cut');print('cut',cut,'n',len(x),'accuracy',x.correct.mean())
