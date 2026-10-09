#!/usr/bin/env python3
"""AIA target-cohort-held-out controlled addition of unsupervised density features to frozen C+BG models."""
from __future__ import annotations
import argparse,importlib.util,json,time,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from experiment import TEAMS, top75, metric, SEED, bootstrap_differences,soft_group_score,csv_read

def main(root,prior_path,out):
    t0=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('prior_experiment',prior_path/'experiment.py');prior=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prior;spec.loader.exec_module(prior)
    bg=pd.read_csv(root/'公式データ'/'予備戦'/'PWSCup2026_prelim_common/A_bg.csv')
    D={k:csv_read(root,k) for k in TEAMS}
    f,truths=prior.feature_table(bg,D)
    density=pd.read_csv(out/'density_public_scores.csv').set_index(['team','row_id'])
    old_scores=pd.read_csv(prior_path/'output/public_scores.csv').set_index(['team','row_id'])
    member=['member_near4','member_knn4','member_near5','member_partial2','rank_member_near4','rank_member_near5','rank_member_partial2']+[f'c_{n.lower()}' for n in ['SBP','TG','HDL','ALT']]
    bge=['bg_near4','bg_knn4','bg_near5','rank_bg_near4','rank_bg_knn4']+[f'bg_{n.lower()}' for n in ['SBP','TG','HDL','ALT']]
    contrast=['contrast_near4','contrast_knn4','contrast_near5','col4','col2','col1','col4_min','rank_contrast_near4','rank_col4','rank_col2','rank_col1']+[f'ratio_{n.lower()}' for n in ['SBP','TG','HDL','ALT']]
    fixed=member+bge+contrast
    extras={
      'base':[],
      'plus_local':['knn2_k1','kde4_half'],
      'plus_smooth':['kde2_mean6','kde4_joint','kde4_cond5'],
      'plus_direct':['direct_lr','direct_hgb'],
      'plus_all':['knn2_k1','kde4_half','kde2_mean6','kde4_joint','kde4_cond5','direct_lr','direct_hgb']
    }
    for k in TEAMS:
        q=D[k][0]
        x=density.loc[[(k,int(i)) for i in q.challenge_row_id]]
        for new in sorted(set(sum(extras.values(),[]))): f[k][new]=x[new].to_numpy(float)
    res=[];prob_byteam=[];scorecols={}
    for key,more in extras.items():
        cols=fixed+more
        scorecols[key]=cols
        for k in TEAMS:
            train=[j for j in TEAMS if j!=k]
            xtrain=pd.concat([f[j][cols] for j in train],ignore_index=True).to_numpy(float)
            ytrain=np.concatenate([truths[j].label.eq('member').to_numpy(int) for j in train])
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.04,max_iter=1500,solver='lbfgs'))
            model.fit(xtrain,ytrain)
            p=model.predict_proba(f[k][cols].to_numpy(float))[:,1]
            if key=='base':
                ref=old_scores.loc[[(k,int(i)) for i in D[k][0].challenge_row_id]].c_bg.to_numpy(float)
                err=np.max(np.abs(p-ref))
                if err>1e-10:raise ValueError(f'Original baseline not recreated team {k} max error {err}')
            score='c_bg'+('' if key=='base' else '_'+key)
            q,t=D[k][0],truths[k];label=t.label.eq('member').to_numpy();rare=t.is_rare.eq(1).to_numpy()
            masks={'base0':top75(q,p),'soft05_0':top75(q,soft_group_score(q,p))}
            for policy,mask in masks.items():
                for value in [0,11]:
                    pred=np.where(mask,9.5,value).astype(float)
                    met=metric(pred,t)
                    res.append(dict(team=k,model=score,policy=('base' if policy=='base0' else 'soft05')+'_'+str(value),
                            auc=float(roc_auc_score(label,p)),auc_rare=float(roc_auc_score(label[rare],p[rare])),
                            **met))
            prob_byteam.append(pd.DataFrame({'team':k,'row_id':q.challenge_row_id,'model':score,'score':p}))
        print('finished',key,round(time.monotonic()-t0,1),'s',flush=True)
    raw=pd.DataFrame(res);raw.to_csv(out/'augment_target_metrics.csv',index=False)
    probs=pd.concat(prob_byteam,ignore_index=True);probs.to_csv(out/'augment_public_predictions.csv',index=False)
    group=[]
    for label,keys in [('20target',[k for k in TEAMS if k!=5]),('processed18',[k for k in TEAMS if k not in (1,2,5)])]:
        s=raw[raw.team.isin(keys)].groupby(['model','policy']).agg(
            auc=('auc','mean'),auc_rare=('auc_rare','mean'),r_all=('r_all','mean'),r_rare=('r_rare','mean'),r_max=('r_max','mean'),
            rare_extra=('rare_extra','mean'),hit_m=('hit_m','mean'),hit_c=('hit_c','mean'),n=('team','count')).reset_index()
        s.insert(0,'scope',label);group.append(s)
    pd.concat(group).to_csv(out/'augment_summary.csv',index=False)
    paired=[]
    for label,keys in [('20target',[k for k in TEAMS if k!=5]),('processed18',[k for k in TEAMS if k not in (1,2,5)])]:
        s=raw[(raw.team.isin(keys)) & (raw.policy=='base_0')]
        pivot=s.pivot(index='team',columns='model',values='r_all')
        base=pivot['c_bg']
        for m in pivot:
            if m=='c_bg':continue
            delta=pivot[m]-base;rng=np.random.default_rng(SEED)
            ri=rng.integers(0,len(delta),size=(20000,len(delta)))
            ci=np.quantile(delta.to_numpy()[ri].mean(axis=1),[.025,.975])
            paired.append(dict(scope=label,model=m,delta_mean=float(delta.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
                        wins=int((delta>1e-10).sum()),losses=int((delta<-1e-10).sum()),ties=int((abs(delta)<=1e-10).sum())))
    pd.DataFrame(paired).to_csv(out/'augment_paired_tests.csv',index=False)
    (out/'augment_design.json').write_text(json.dumps({'prior_baseline_recomputed_exact':True,'heldout_cohorts':21,
        'base_features':fixed,'density_feature_sets':extras,'target_predictions_from_heldout_cohort_labels':False,
        'logistic_C':0.04,'seconds':time.monotonic()-t0},ensure_ascii=False,indent=2)+'\n')
    print(pd.concat(group).query("scope=='20target' and policy=='base_0'").sort_values('r_all',ascending=False).to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();main(args.root,args.prior,args.out)
