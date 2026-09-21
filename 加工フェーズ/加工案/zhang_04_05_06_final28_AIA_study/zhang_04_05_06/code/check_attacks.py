"""公式参照MIA/AIAをB内holdoutで検証する（公式protectionではない）。
モデル・設定の固定後に選んだ分割で評価。QIによる対応対照も診断する。
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest,mannwhitneyu,fisher_exact
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
from generate_three import generate,METHODS
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'official_scoring/src'))
from pwscup2026.kit.selfscore import load_local_reference
from pwscup2026.attack.mia import overlap_mia
from pwscup2026.attack.aia import aia_predict

def bootstrap_auc(y,score,seed):
 rng=np.random.default_rng(seed);pos=score[y==1];neg=score[y==0];values=[]
 for _ in range(300):
  p=rng.choice(pos,len(pos));n=rng.choice(neg,len(neg));values.append(roc_auc_score(y,np.r_[p,n]))
 return np.quantile(values,[.025,.975]).tolist()

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dist',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
 args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 b,ref,cfg,_=load_local_reference(args.dist);a=ref.a_bg;s=json.loads((args.dist/'schema.json').read_text());rare=ref.b_is_rare
 cfg['aia'].update({'qi_continuous':['age','SBP','TG','HDL','ALT'],'block_keys':['sex','prefecture'],'k_nn_onset':5})
 rows=[];matched=[]
 for split in [9501,9502,9503,9504]:
  strata=rare.astype(int)*2+b.onset.to_numpy(int)
  ti,vi=train_test_split(np.arange(len(b)),test_size=.3,stratify=strata,random_state=split)
  train=b.iloc[ti].reset_index(drop=True);test=b.iloc[vi].reset_index(drop=True);q=pd.concat([train,test],ignore_index=True)
  y=np.r_[np.ones(len(train)),np.zeros(len(test))]
  # 属性対応を付ける追加診断。年齢帯=10歳。公式の対照データそのものではない。
  tr=train.assign(age_band=train.age//10);te=test.assign(age_band=test.age//10)
  keys=['sex','prefecture','age_band','onset'];gt=tr.groupby(keys).indices;gv=te.groupby(keys).indices
  rng=np.random.default_rng(split);pairs=[]
  for key in sorted(gt.keys() & gv.keys()):
   n=min(len(gt[key]),len(gv[key]));pairs.extend(zip(rng.permutation(gt[key])[:n],rng.permutation(gv[key])[:n]))
  pi,pj=np.array(pairs).T
  for method in METHODS:
   c,meta=generate(train,a,s,method)
   score=overlap_mia(c,a,q,cfg);fpr,tpr,th=roc_curve(y,score);valid=np.flatnonzero(fpr<=.01);j=valid[np.argmax(tpr[valid])]
   ci=bootstrap_auc(y,score,split)
   _,hm=aia_predict(c,train[s['aia_qi_columns']],a,cfg);_,hc=aia_predict(c,test[s['aia_qi_columns']],a,cfg)
   sm=abs(hm-train.time.to_numpy())<=.5;sc=abs(hc-test.time.to_numpy())<=.5
   pos=sm[pi];neg=sc[pj];wins=int((pos&~neg).sum());losses=int((~pos&neg).sum());p_aia=binomtest(wins,wins+losses,.5,alternative='greater').pvalue if wins+losses else 1.
   kr=np.r_[rare[ti],rare[vi]]
   row={'split':split,'method':method,'KDE_AUC':roc_auc_score(y,score),'KDE_AUC_CI_low':ci[0],'KDE_AUC_CI_high':ci[1],
        'KDE_TPR01':float(tpr[j]),'realized_FPR':float(fpr[j]),'KDE_rare_AUC':roc_auc_score(y[kr],score[kr]),
        'MIA_AUC_one_sided_p':mannwhitneyu(score[y==1],score[y==0],alternative='greater').pvalue,
        'aia_member_all':sm.mean(),'aia_control_all':sc.mean(),
        'aia_member_onset':sm[train.onset==1].mean(),'aia_control_onset':sc[test.onset==1].mean(),
        'constant10_member':float((abs(train.time.to_numpy()[pi]-10)<=.5).mean()),'constant10_control':float((abs(test.time.to_numpy()[pj]-10)<=.5).mean()),'matched_pairs':len(pi),'aia_matched_member':pos.mean(),'aia_matched_control':neg.mean(),'aia_matched_difference':pos.mean()-neg.mean(),
        'aia_matched_one_sided_p':p_aia,'matched_member_only_correct':wins,'matched_control_only_correct':losses,
        'n_train':len(train),'n_test':len(test),'rare_train':int(rare[ti].sum()),'rare_test':int(rare[vi].sum())}
   rows.append(row);print(json.dumps(row),flush=True)
   pd.DataFrame({'train_row':ti[pi],'holdout_row':vi[pj],'member_hit':pos,'control_hit':neg}).to_csv(args.out/f'pairs_{split}_{method}.csv',index=False)
   np.savez(args.out/f'predictions_{split}_{method}.npz',mia=score,y=y,train_index=ti,holdout_index=vi,member_time_hat=hm,control_time_hat=hc)
  pd.DataFrame(rows).to_csv(args.out/'confirmation_attacks.csv',index=False)
 # Holm correction across 12 AUC tests and 12 matched AIA tests; no inference of equivalence.
 df=pd.DataFrame(rows);pv=np.r_[df.MIA_AUC_one_sided_p,df.aia_matched_one_sided_p];order=np.argsort(pv)
 adj=np.empty(len(pv));adj[order]=np.minimum(1,np.maximum.accumulate(pv[order]*(len(pv)-np.arange(len(pv)))))
 df['MIA_AUC_Holm_p']=adj[:len(df)];df['AIA_matched_Holm_p']=adj[len(df):]
 df.to_csv(args.out/'confirmation_attacks.csv',index=False)
 (args.out/'protocol.json').write_text(json.dumps({'model_frozen_before_splits':True,'splits':[9501,9502,9503,9504],
  'marginal_stratification':'is_rare + onset for evaluation splitting only','paired_controls':'same sex, prefecture, 10-year age band, onset; no replacement',
  'auc_ci':'300 within-split bootstrap resamples; descriptive, model-training uncertainty not included','multiplicity':'Holm across 24 AUC and paired-AIA tests',
  'not_official':'Holdout queries are from B, not the nonshared block; no official phi correction. Constant10 baseline ignores C, and its excess cannot be corrected by synthetic data.',
  'reuse':'4 splits reuse the same B; not independent cohorts'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
