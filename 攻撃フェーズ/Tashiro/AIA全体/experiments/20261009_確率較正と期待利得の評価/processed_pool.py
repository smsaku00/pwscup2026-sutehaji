#!/usr/bin/env python3
"""Sensitivity study: train membership model only on processed18 cohorts, target held out.

This training-set selection is POST-HOC, based on previously published outcomes.
It is not an independent assessment of a pre-specified grouping policy.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
import calibrated_policy as cp

def run(root,prior_dir,density,out):
    lib=cp.import_prior(prior_dir)
    D,F,T=cp.data(root,lib,density)
    processed=[k for k in cp.TEAMS if k not in (1,2,5)]
    rows=[]
    for team in processed:
        for method,cols in cp.SPECS.items():
            train=[k for k in processed if k!=team]
            x=np.concatenate([F[j][cols].to_numpy(float) for j in train]);y=np.concatenate([T[j].label.eq('member').to_numpy(int) for j in train])
            model=cp.model_train(x,y)
            p=model.predict_proba(F[team][cols].to_numpy(float))[:,1]
            label=T[team].label.eq('member').to_numpy(int);rare=T[team].is_rare.eq(1).to_numpy(bool)
            selected=cp.top75(D[team][0],p)
            for miss in (0,11):
                met=cp.score_metrics(np.where(selected,9.5,miss),T[team])
                rows.append(dict(team=team,model=method,miss=miss,auc=roc_auc_score(label,p),
                                 auc_rare=roc_auc_score(label[rare],p[rare]),
                                 brier=brier_score_loss(label,p),n_training_cohorts=len(train),**met))
    out.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out,index=False)
    print(df.groupby(['model','miss'])[['auc','brier','r_all','r_rare','r_max']].mean().round(6).to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--density-scores',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.prior,a.density_scores,a.out)
