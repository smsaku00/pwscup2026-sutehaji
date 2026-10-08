#!/usr/bin/env python3
"""Soft-balanced group probabilities, model agreement, and revised answer decision.
Reads ONLY public per-row predictions + public Q until final evaluation against truth.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import rankdata
from experiment import TEAMS, top75, pairs, metrics, SEED

SCORE_METHODS=['near4','col2','c_bg','c_bg_hgb','ensemble_lr_hgb','ensemble_with_col2']
ALPHAS=[0.0,.25,.5,1.0]

def project(q,p,alpha):
    out=p.copy()
    for g in pairs(q,5).values():
        if len(g)<2 or len(g)%2:continue
        z=logit(np.clip(p[g],1e-6,1-1e-6))
        lo,hi=-20.,20.
        for _ in range(50):
            mid=(lo+hi)/2
            if expit(z+mid).mean()<.5:lo=mid
            else:hi=mid
        # A soft log-odds offset, preserving order inside every group.
        out[g]=expit(z+alpha*(lo+hi)/2)
    return out

def conservative(q,score,member_prob,typ,near,col,base):
    pred=np.full(len(q),0.0)
    pred[list(base)]=9.5
    edits=[]
    groups=pairs(q,5)
    for g in groups.values():
        if len(g)%2 or len(g)<2:continue
        lo=sorted(g,key=lambda i:(member_prob[i],i));m=len(g)//2
        for j in range(m):
            low,high=lo[j],lo[-j-1]
            if low not in base:continue
            if typ=='pair_consensus' and len(g)!=2:continue
            if typ in ('pair_consensus','group_consensus'):
                if member_prob[low]>.45 or member_prob[high]<.55 or member_prob[high]-member_prob[low]<.10:continue
                if near[high]<=near[low] or col[high]<=col[low]:continue
            elif typ=='group_soft':
                if member_prob[low]>.40 or member_prob[high]<.60 or member_prob[high]-member_prob[low]<.15:continue
            elif typ=='group_strict':
                if member_prob[low]>.25 or member_prob[high]<.75 or member_prob[high]-member_prob[low]<.30:continue
            else:raise ValueError(typ)
            pred[low]=11;edits.append((low,high))
    return pred,edits

def main(root, score_csv, out):
    out.mkdir(parents=True,exist_ok=True)
    public=pd.read_csv(score_csv)
    assert len(public)==21*326 and public.team.nunique()==21
    dataset={}
    for k in TEAMS:
        dr=root/'公式データ'/'予備戦'
        q=pd.read_csv(dr/f'PWSCup2026_prelim_attack/aia_challenge_{k}.csv').sort_values('challenge_row_id').reset_index(drop=True)
        t=pd.read_csv(dr/f'結果/正解データ/aia_truth_{k}.csv').sort_values('challenge_row_id').reset_index(drop=True)
        assert q.challenge_row_id.equals(t.challenge_row_id)
        f=public[public.team==k].sort_values('row_id').reset_index(drop=True)
        assert q.challenge_row_id.equals(f.row_id)
        dataset[k]=(q,t,f)
    results=[];orient=[];audit=[];tally=[]
    for k,(q,t,f) in dataset.items():
        n=len(q)
        logi=f.c_bg.to_numpy(float);hgb=f.c_bg_hgb.to_numpy(float)
        near=f.near4.to_numpy(float);col=f.col2.to_numpy(float)
        # NOTE: col2 ranks are not calibrated posterior probabilities.
        colrank=(rankdata(col,method='average')-.5)/n
        score={nm:f[nm].to_numpy(float) for nm in ['near4','col2','c_bg','c_bg_hgb']}
        score['ensemble_lr_hgb']=(logi+hgb)/2
        score['ensemble_with_col2']=.65*((logi+hgb)/2)+.35*colrank
        for nm in SCORE_METHODS:
            source=score[nm]
            rawp=np.clip(source,1e-6,1-1e-6) if nm not in ('near4','col2') else (.04+.92*(rankdata(source,method='average')-.5)/n)
            for a in ALPHAS:
                p=project(q,rawp,a)
                top=top75(q,p)
                predict=np.where(np.isin(np.arange(n),list(top)),9.5,0.0)
                m=metrics(predict,t)
                res={'team':k,'score':nm,'alpha':a,'mode':'top75_rebalanced',
                    'changed':0 if a==0 else None,'baseline0':False,**m}
                results.append(res)
            baseline=top75(q,source)
            basepred=np.where(np.isin(np.arange(n),list(baseline)),9.5,0.0)
            for ptype in ('pair_consensus','group_consensus','group_soft','group_strict'):
                pred,edits=conservative(q,source,score['ensemble_lr_hgb'],ptype,near,col,baseline)
                assert np.all(pred[np.setdiff1d(np.arange(n),[p[0] for p in edits])]==basepred[np.setdiff1d(np.arange(n),[p[0] for p in edits])])
                res={'team':k,'score':nm,'alpha':None,'mode':ptype,'changed':len(edits),'baseline0':False,**metrics(pred,t)}
                results.append(res)
                for low,high in edits:
                    audit.append({'team':k,'score':nm,'policy':ptype,'low_id':int(q.challenge_row_id.iloc[low]),
                        'high_id':int(q.challenge_row_id.iloc[high]),
                        'low_is_control':bool(t.label.iloc[low]=='control'),
                        'low_was_correct':bool(abs(basepred[low]-t.time.iloc[low])<=.5),
                        'low_now_correct':bool(abs(pred[low]-t.time.iloc[low])<=.5)})
        for g in pairs(q,5).values():
            if len(g)<2:continue
            mm=int((t.label.iloc[g]=='member').sum());cc=len(g)-mm
            model=score['ensemble_lr_hgb'];l=sorted(g,key=lambda i:(model[i],i))
            d={ 'team':k,'group_size':len(g),'balanced':mm==cc,'model_gap':model[l[-1]]-model[l[0]],
                'top_half_member_rate':float((t.label.iloc[l[len(g)//2:]]=='member').mean()),
                'bottom_half_control_rate':float((t.label.iloc[l[:len(g)//2]]=='control').mean())}
            orient.append(d)
            if len(g)==2:
                a,b=g;true_member=t.label.iloc[a]=='member'
                # Binary agreement of separately-motivated scores.
                comps={nam:(val[a]>val[b]) for nam,val in score.items() if val[a]!=val[b]}
                agree=(int(comps.get('near4',False))==int(comps.get('col2',False))==int(comps.get('c_bg',False)))
                tally.append({'team':k,'same_direction_near_col_model':agree,'correct_if_agree':int(comps['c_bg']==true_member) if agree else np.nan})
    results=pd.DataFrame(results)
    results.to_csv(out/'extension_per_target.csv',index=False)
    for name,exclude in [('20',[5]),('18',[1,2,5])]:
        d=results[~results.team.isin(exclude)].groupby(['score','mode','alpha'],dropna=False).agg(
            n=('team','nunique'),r_all=('r_all','mean'),r_rare=('r_rare','mean'),r_max=('r_max','mean'),
            pm=('pm','mean'),pc=('pc','mean'),changed=('changed','sum')).reset_index()
        d.to_csv(out/f'extension_summary_{name}.csv',index=False)
    pd.DataFrame(orient).to_csv(out/'group_direction.csv',index=False)
    pd.DataFrame(tally).to_csv(out/'consensus_pair.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'group_changes.csv',index=False)
    z=pd.read_csv(out/'extension_summary_20.csv')
    print('Groups balanced:',pd.DataFrame(orient).query('team != 5').groupby('group_size').balanced.agg(['size','mean']).head(20).to_string(),flush=True)
    print('Soft quotas 20:')
    print(z[(z['mode']=='top75_rebalanced') & z.score.isin(['c_bg','c_bg_hgb','ensemble_lr_hgb','ensemble_with_col2','col2'])].pivot(index='score',columns='alpha',values='r_all').round(4).to_string(),flush=True)
    print('Best conditional policies 20:')
    print(z[(z['mode'].isin(['group_consensus','group_soft','group_strict','pair_consensus'])) & z.score.isin(['c_bg','ensemble_lr_hgb','col2'])][['score','mode','r_all','r_rare','r_max','changed']].round(4).to_string(index=False),flush=True)
    c=pd.DataFrame(tally);c=c[c.team!=5]; print('Pair consensus:',len(c),c.same_direction_near_col_model.sum(),c.loc[c.same_direction_near_col_model,'correct_if_agree'].mean(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).parent/'input');p.add_argument('--scores',type=Path,default=Path(__file__).parent/'output'/'public_scores.csv');p.add_argument('--out',type=Path,default=Path(__file__).parent/'output');a=p.parse_args();main(a.root,a.scores,a.out)
