"""Frozen generators: local holdout diagnostics, not server protection.
Four overlapping B splits. No parameter tuning uses these outcomes.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest,mannwhitneyu
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from generate_three import generate,METHODS
from previous_baseline import generate as previous_generate,METHODS as PREVIOUS
from decoy import decoy_swaps
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'official_scoring/src'))
from pwscup2026.kit.selfscore import load_local_reference
from pwscup2026.attack.mia import overlap_mia
from pwscup2026.attack.aia import aia_predict

def alternate_aia(c,q,a):
 cols=['age','SBP','TG','HDL','ALT'];sd=a[cols].std().to_numpy()
 d=cdist(q[cols]/sd,c[cols]/sd,'sqeuclidean');out={k:np.empty(len(q)) for k in ['knn5_median','sex_only_1nn']}
 for i,row in q.reset_index(drop=True).iterrows():
  sex=np.flatnonzero(c.sex.to_numpy()==row.sex);both=sex[c.prefecture.to_numpy()[sex]==row.prefecture]
  if not len(sex):sex=np.arange(len(c))
  if not len(both):both=sex
  ids=both[np.argsort(d[i,both])[:5]];out['knn5_median'][i]=np.median(c.time.to_numpy()[ids])
  ids=sex[d[i,sex]<=d[i,sex].min()];out['sex_only_1nn'][i]=np.mean(c.time.to_numpy()[ids])
 return out

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--dist',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 b,ref,cfg,_=load_local_reference(args.dist);a=ref.a_bg;s=json.loads((args.dist/'schema.json').read_text());rare=ref.b_is_rare
 cfg['aia'].update({'qi_continuous':['age','SBP','TG','HDL','ALT'],'block_keys':['sex','prefecture'],'k_nn_onset':5})
 rows=[];attacks=[];metas=[]
 for split in [9601,9602,9603,9604]:
  ti,vi=train_test_split(np.arange(len(b)),test_size=.3,stratify=rare.astype(int)*2+b.onset.to_numpy(int),random_state=split)
  train=b.iloc[ti].reset_index(drop=True);test=b.iloc[vi].reset_index(drop=True);q=pd.concat([train,test],ignore_index=True);y=np.r_[np.ones(len(train)),np.zeros(len(test))];n=len(train)
  tr=train.assign(age_band=train.age//10);te=test.assign(age_band=test.age//10);keys=['sex','prefecture','age_band','onset'];gt=tr.groupby(keys).indices;gv=te.groupby(keys).indices;rng=np.random.default_rng(split);pairs=[]
  for key in sorted(gt.keys() & gv.keys()):
   count=min(len(gt[key]),len(gv[key]));pairs.extend(zip(rng.permutation(gt[key])[:count],rng.permutation(gv[key])[:count]))
  pi,pj=np.array(pairs).T
  c07=None
  for method in [PREVIOUS[0],*METHODS,'zhang_07_d01_prefecture_decoy']:
   if method==PREVIOUS[0]:c,meta=previous_generate(train,a,s,method)
   elif method.endswith('decoy'):c,meta=decoy_swaps(c07,train,a)
   else:c,meta=generate(train,a,s,method)
   if method==METHODS[0]:c07=c.copy()
   metas.append({'split':split,'method':method,**meta})
   score=overlap_mia(c,a,q,cfg);auc=roc_auc_score(y,score);p=mannwhitneyu(score[y==1],score[y==0],alternative='greater').pvalue
   kr=np.r_[rare[ti],rare[vi]]
   _,official=aia_predict(c,q[s['aia_qi_columns']],a,cfg)
   predictions={'official_1nn':official,**alternate_aia(c,q,a),'constant10':np.full(len(q),10.)}
   for attack,hat in predictions.items():
    hit=abs(hat-q.time.to_numpy())<=.5;hm,hc=hit[:n],hit[n:];pos,neg=hm[pi],hc[pj];wins=int((pos&~neg).sum());losses=int((~pos&neg).sum());diff=pos.astype(float)-neg.astype(float)
    boot=np.random.default_rng(split).choice(diff,(1000,len(diff)),replace=True).mean(1)
    row={'split':split,'method':method,'attack':attack,'member_accuracy':hm.mean(),'control_accuracy':hc.mean(),'matched_pairs':len(pi),'matched_member':pos.mean(),'matched_control':neg.mean(),'matched_difference':diff.mean(),'difference_CI_low':np.quantile(boot,.025),'difference_CI_high':np.quantile(boot,.975),'one_sided_p':binomtest(wins,wins+losses,.5,alternative='greater').pvalue if wins+losses else 1.,'member_onset_accuracy':hm[train.onset==1].mean(),'control_onset_accuracy':hc[test.onset==1].mean(),'member_rare_accuracy':hm[rare[ti]].mean(),'control_rare_accuracy':hc[rare[vi]].mean()}
    attacks.append(row)
   early_auc=roc_auc_score(y,-official)
   row={'split':split,'method':method,'KDE_AUC':auc,'KDE_rare_AUC':roc_auc_score(y[kr],score[kr]),'KDE_one_sided_p':p,'predicted_early_time_MIA_AUC':early_auc,'predicted_early_time_MIA_p':mannwhitneyu(-official[:n],-official[n:],alternative='greater').pvalue,'official_aia_matched_difference':attacks[-4]['matched_difference'],'official_aia_matched_p':attacks[-4]['one_sided_p']}
   rows.append(row);print(json.dumps(row),flush=True)
   np.savez(args.out/f'predictions_{split}_{method}.npz',mia=score,y=y,train_index=ti,holdout_index=vi,paired_train=pi,paired_holdout=pj,**predictions)
  pd.DataFrame(rows).to_csv(args.out/'mia.csv',index=False);pd.DataFrame(attacks).to_csv(args.out/'aia.csv',index=False)
  (args.out/'generation.json').write_text(json.dumps(metas,indent=2))
 df=pd.DataFrame(rows);pv=np.r_[df.KDE_one_sided_p,df.predicted_early_time_MIA_p,df.official_aia_matched_p];order=np.argsort(pv);adj=np.empty(len(pv));adj[order]=np.minimum(1,np.maximum.accumulate(pv[order]*(len(pv)-np.arange(len(pv)))))
 for k,col in enumerate(['KDE_Holm_p','early_time_MIA_Holm_p','official_aia_Holm_p']):df[col]=adj[k*len(df):(k+1)*len(df)]
 df.to_csv(args.out/'mia.csv',index=False)
 (args.out/'protocol.json').write_text(json.dumps({'splits':[9601,9602,9603,9604],'frozen_before_evaluation':True,'train_test':[734,315],'stratification':'true rare + onset, evaluation only','matching':'sex, prefecture, 10-year age band, onset; no replacement','uncertainty':'1000 paired bootstrap draws per split; 60-test Holm for primary MIA / early-time MIA / official AIA','scope':'Overlapping B splits, not independent trials. No official hidden controls or phi correction. Non-significance does not prove privacy.','selection':'All three methods reported. The exploratory decoy variant is reported separately.'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
