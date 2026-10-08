from pathlib import Path
import numpy as np,pandas as pd,argparse
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
D=Path(__file__).parent; parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=D/'input');parser.add_argument('--out',type=Path,default=D/'output');args=parser.parse_args();r=args.out;r.mkdir(parents=True,exist_ok=True);src=args.root/'公式データ'/'予備戦'
p=pd.read_csv(r/'public_scores.csv');out=[];bp=[]
for k,g in p.groupby('team'):
    t=pd.read_csv(src/f'結果/正解データ/aia_truth_{k}.csv').sort_values('challenge_row_id').reset_index(drop=True)
    g=g.sort_values('row_id').reset_index(drop=True)
    assert (t.challenge_row_id==g.row_id).all()
    y=t.label.eq('member').to_numpy(int);rare=t.is_rare.to_numpy(bool)
    for score in ['near4','col4','col2','q_only','bg_only','c_only','c_bg','c_bg_other','c_bg_hgb']:
        s=g[score].to_numpy(float)
        out.append(dict(team=k,score=score,auc_all=roc_auc_score(y,s),auc_rare=roc_auc_score(y[rare],s[rare]),
                        auc_common=roc_auc_score(y[~rare],s[~rare]),rare_count=int(rare.sum())))
    # Detailed calibration of the held-out c_bg score (post-hoc diagnostic only).
    prob=g.c_bg.to_numpy(float)
    for i,(lo,hi) in enumerate(zip([0,.1,.2,.3,.4,.5,.6,.7,.8,.9],[.1,.2,.3,.4,.5,.6,.7,.8,.9,1])):
        m=(prob>=lo)&(prob<hi if hi!=1 else prob<=hi)
        if m.any():bp.append({'team':k,'bin':i,'lo':lo,'hi':hi,'n':int(m.sum()),'mean_pred':float(prob[m].mean()),'true_member_fraction':float(y[m].mean())})
out=pd.DataFrame(out);out.to_csv(r/'rare_AUC_diagnostics.csv',index=False)
cal=pd.DataFrame(bp);cal.to_csv(r/'probability_reliability.csv',index=False)
print(out.query('team!=5').groupby('score')[['auc_all','auc_common','auc_rare']].mean().round(4))
print('Calibration pooled:');print(cal.query('team!=5').groupby('bin')[['n','mean_pred','true_member_fraction']].agg({'n':'sum','mean_pred':'mean','true_member_fraction':'mean'}).round(4))
for typ in ['auc_all','auc_rare']:
    d=out.query('team!=5').pivot(index='team',columns='score',values=typ);a=(d.c_bg-d.c_only).to_numpy();rng=np.random.default_rng(20261009)
    b=a[rng.integers(0,len(a),(20000,len(a)))].mean(axis=1)
    print(typ,'∆c_bg vs c_only',a.mean(), 'CI',np.quantile(b,[.025,.975]),'wins',int((a>0).sum()))
