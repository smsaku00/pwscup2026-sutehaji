#!/usr/bin/env python3
"""Fixed-reference 50% background sensitivity. No attack labels enter density estimation."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from experiment import TEAMS,SEED, matrix,density_features,top75,metric,csv_read

def main(root,out):
    d=root/'公式データ'/'予備戦';BG=pd.read_csv(d/'PWSCup2026_prelim_common/A_bg.csv')
    allX=matrix(BG);iqr=np.maximum(np.quantile(allX,.75,axis=0)-np.quantile(allX,.25,axis=0),1e-5)
    targets={k:csv_read(root,k) for k in TEAMS if k!=5}
    kinds=['kde2_mean6','kde2_cond_mean6','kde4_joint','kde4_half','kde4_cond5','knn2_k1','knn2_k5','knn4_k5']
    result=[]
    for seed in [202610090,202610091,202610092,202610093]:
        rng=np.random.default_rng(seed)
        ref=BG.iloc[rng.permutation(len(BG))[:2572]].reset_index(drop=True)
        for k,(q,c,t) in targets.items():
            outscore=density_features(q,c,ref,iqr)
            y=t.label.eq('member').to_numpy();rare=t.is_rare.eq(1).to_numpy()
            for method in kinds:
                sc=outscore[method].to_numpy(float);mask=top75(q,sc)
                mm=metric(np.where(mask,9.5,0),t)
                result.append({'split_seed':seed,'team':k,'method':method,'auc':roc_auc_score(y,sc),
                    'auc_rare':roc_auc_score(y[rare],sc[rare]),'r_all':mm['r_all'],'r_max':mm['r_max']})
        print('Reference-split',seed,'complete',flush=True)
    raw=pd.DataFrame(result);raw.to_csv(out/'background_split_detail.csv',index=False)
    agg=raw.groupby(['split_seed','method'],as_index=False)[['auc','auc_rare','r_all','r_max']].mean()
    agg.to_csv(out/'background_split_summary.csv',index=False)
    print(agg.to_string(index=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();main(a.root,a.out)
