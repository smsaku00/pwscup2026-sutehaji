#!/usr/bin/env python3
"""BG-disjoint C placebo comparisons for density ratio evidence; no labels used to fit scores."""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from experiment import TEAMS, matrix, density_features, direct_ratio, top75, metric, csv_read, SEED

def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    d=root/'公式データ'/'予備戦'
    BG=pd.read_csv(d/'PWSCup2026_prelim_common/A_bg.csv')
    rng=np.random.default_rng(SEED+447)
    perm=rng.permutation(len(BG));pool=BG.iloc[perm[:2571]].reset_index(drop=True)
    ref=BG.iloc[perm[2571:]].reset_index(drop=True)
    arr=matrix(ref);scale=np.maximum(np.quantile(arr,.75,axis=0)-np.quantile(arr,.25,axis=0),1e-5)
    rows=[];method_names=['kde1_mean','kde2_mean6','kde2_cond_mean6','kde4_half','kde4_joint','kde4_cond5','knn2_k1','knn4_k5','direct_hgb','direct_lr']
    for k in TEAMS:
        if k==5:continue
        q,C,truth=csv_read(root,k);m=truth.label.eq('member').to_numpy()
        ix=[]
        for s in (0,1):
            src=np.flatnonzero(pool.sex.to_numpy()==s)
            nc=int((C.sex==s).sum())
            ix.extend(rng.choice(src,size=nc,replace=False).tolist())
        CN=pool.iloc[ix].reset_index(drop=True)
        assert len(CN)==len(C) and set(CN.index)==set(range(len(C)))
        for source,cc in [('real',C),('BG_placebo',CN)]:
            scores=pd.concat([density_features(q,cc,ref,scale),direct_ratio(q,cc,ref,scale)],axis=1)
            for method in method_names:
                s=scores[method].to_numpy()
                mask=top75(q,s)
                stats=metric(np.where(mask,9.5,0),truth)
                rows.append(dict(team=k,source=source,method=method,auc=float(roc_auc_score(m,s)),
                    r_all=stats['r_all'],r_rare=stats['r_rare'],r_max=stats['r_max']))
        print('placebo complete',k,flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'placebo_team.csv',index=False)
    agg=df.groupby(['source','method'],as_index=False)[['auc','r_all','r_max','r_rare']].mean()
    agg.to_csv(out/'placebo_summary.csv',index=False)
    print(agg.to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();main(a.root,a.out)
