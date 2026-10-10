from pathlib import Path
import os,sys,json,hashlib,unicodedata
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from scipy.stats import wasserstein_distance,ks_2samp
HERE=Path(__file__).resolve().parent
M=json.load(open(os.environ.get('MIA_INPUT_MANIFEST','/workspace/shared/mia_bg7_inputs/manifest.json')))
U=Path(os.environ.get('MIA_UTILITY_DIR','/workspace/shared/mia_utility_stage/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_U有用性と方式選択'))
CACHED=Path(os.environ.get('MIA_CFEATURE_DIR','/workspace/shared/mia_c_only_stage/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_C単独選択とアンサンブル'))
NUM=['age','BMI','SBP','TG','HDL','ALT','FPG','time'];LAB=['BMI','SBP','TG','HDL','ALT','FPG'];CAT=['sex','prefecture','smoking','onset','death'];QI=M['qi7'];FULL=M['features13']
read=lambda p:pd.read_csv(p,float_precision='round_trip')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
registry={str(Path(x['path'])):x for x in M['files']}
def load(p):
 x=read(p)
 if 'prefecture' in x:x['prefecture']=x['prefecture'].astype(str).map(lambda v:unicodedata.normalize('NFC',v))
 return x
BG={p:load(M['backgrounds'][p]['path']) for p in ('starter','prelim','main')}
B={p:load(M['phases'][p]['b_self']) for p in ('prelim','main')}
ALLBG=pd.concat(BG.values(),ignore_index=True)
MAP=read(U/'phase_cohort_crosswalk.csv')
CACHE=read(CACHED/'C_only_features.csv').set_index('team')
BIND=read(U/'utility_C_binding_audit.csv')
MAINU=read(os.environ.get('MIA_MAIN_UTILITY_CSV','/workspace/shared/攻撃フェーズ/Tashiro/MIA全体/本戦公式有用性_20261010/main_utility_scores.csv'))

def sig(x,cols):
 # Convert all numeric columns to float, NFC for prefecture; no record_id, no tolerance.
 y=x[cols].copy()
 for col in cols:
  if col!='prefecture':y[col]=pd.to_numeric(y[col]).astype(float)
 return list(y.itertuples(index=False,name=None))
def overlap(x,y,cols):
 target=set(sig(y,cols));s=sig(x,cols);return sum(t in target for t in s)
def tv(x,y):
 a=x.astype(str).value_counts(normalize=True);b=y.astype(str).value_counts(normalize=True)
 return float((a.subtract(b,fill_value=0).abs()).sum()/2)
def residcat(xp,xm,bp,bm):
 a=xm.astype(str).value_counts(normalize=True).subtract(xp.astype(str).value_counts(normalize=True),fill_value=0)
 b=bm.astype(str).value_counts(normalize=True).subtract(bp.astype(str).value_counts(normalize=True),fill_value=0)
 return float(a.subtract(b,fill_value=0).abs().sum()/2)
def describe(x,phase,team,kind):
 base={'phase':phase,'cohort':team,'kind':kind,'n':len(x),'duplicate_full_n':int(x[FULL].duplicated().sum()),'duplicate_qi7_n':int(x[QI].duplicated().sum())}
 vals=[]
 for c in NUM:
  v=x[c].to_numpy(float); vc=x[c].value_counts();d={**base,'column':c,'nunique':len(vc),'repeated_excess_fraction':1-len(vc)/len(x),'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1)),'min':float(np.min(v)),'max':float(np.max(v)),'mode':float(vc.index[0]),'mode_fraction':float(vc.iloc[0]/len(v))}
  for q in [.01,.05,.25,.5,.75,.95,.99]:d[f'q{int(q*100):02}']=float(np.quantile(v,q))
  for u in [1e-6,.001,.1,1.,5.,10.]:d[f'grid_{u:g}']=float(np.isclose(v/u,np.round(v/u),atol=1e-7,rtol=0).mean())
  for s,bg in {**BG,'all':ALLBG}.items():d[f'value_reuse_bg_{s}']=float(x[c].isin(bg[c]).mean())
  if phase in B:d['value_reuse_B_self']=float(x[c].isin(B[phase][c]).mean())
  vals.append(d)
 cats=[]
 for c in CAT:
  for value,n in x[c].astype(str).value_counts().items():cats.append({**base,'column':c,'value':value,'count':n,'fraction':n/len(x)})
 return vals,cats

numeric=[];categorical=[];datasets={};inputs=[];profiles=[]
for phase in ['prelim','main']:
 ids=sorted(int(p.stem.split('_')[1]) for p in Path(M['phases'][phase]['attack_dir']).glob('C_*.csv'))
 for id in ids:
  p=Path(M['phases'][phase]['attack_dir'])/f'C_{id}.csv';x=load(p);datasets[(phase,id)]=x
  a,b=describe(x,phase,id,'C');numeric+=a;categorical+=b
  reg=registry.get(str(p),{});inputs.append({'phase':phase,'cohort':id,'path':str(p),'sha256':sha(p),'manifest_sha256_matches':sha(p)==reg.get('sha256'),'source_url':reg.get('source_url'),'rows':len(x)})
  z={'phase':phase,'cohort':id,'rows':len(x),'dup_full_n':int(x[FULL].duplicated().sum()),'dup_qi7_n':int(x[QI].duplicated().sum())}
  for s,bg in {**BG,'all':ALLBG}.items():
   for cols,label in [(FULL,'full'),(QI,'qi7')]:z[f'overlap_bg_{s}_{label}_n']=overlap(x,bg,cols)
  z['overlap_B_self_full_n']=overlap(x,B[phase],FULL);z['overlap_B_self_qi7_n']=overlap(x,B[phase],QI)
  # Prior aggregate features reused verbatim for 20 preliminary target Cs.
  if phase=='prelim' and id in CACHE.index:
   z.update({f'cached_{k}':v for k,v in CACHE.loc[id].to_dict().items()})
  profiles.append(z)
for ph in ['prelim','main']:
 for kind,x in [('BG',BG[ph]),('B_self',B[ph])]:
  a,b=describe(x,ph,M['phases'][ph]['self_id'],kind);numeric+=a;categorical+=b
N=pd.DataFrame(numeric);N.to_csv(HERE/'numeric_signatures_long.csv',index=False)
pd.DataFrame(categorical).to_csv(HERE/'category_distributions_long.csv',index=False)
P=pd.DataFrame(profiles).set_index(['phase','cohort']);P.to_csv(HERE/'C_profile_phase.csv')
A=N[N.kind=='C'].set_index(['phase','cohort','column'])
rows=[];contrasts=[];corrs=[]
for _,r in MAP.iterrows():
 main=int(r.main_cohort);pre=int(r.prelim_cohort) if pd.notna(r.prelim_cohort) else None;name=r.team_name
 row={'team_name':name,'prelim_cohort':pre,'main_cohort':main,'main_submission_id':int(r.main_submission_id),'mapping_source':r.source,'identity_method':r.identity_method,'main_username':MAINU.set_index('cohort').loc[main,'codabench_display_name']}
 valid=pre is not None and ('prelim',pre) in datasets
 row['comparison_available']=valid
 row['scope_note']='two distributed C files' if valid else ('no preliminary C file' if pre is not None else 'no preliminary roster correspondence')
 for ph,id in [('prelim',pre),('main',main)]:
  if id is None or (ph,id) not in datasets:continue
  for k,v in P.loc[(ph,id)].to_dict().items():row[f'{ph}_{k}']=v
  for col in ['age','BMI','SBP','TG','HDL','ALT','FPG','time']:
   for field in ['nunique','mean','sd','mode','mode_fraction','grid_1e-06','grid_0.001','grid_0.1','grid_1','value_reuse_bg_all']:
    row[f'{ph}_{col}_{field}']=A.loc[(ph,id,col),field]
  bind=BIND[(BIND.phase==ph)&(BIND.team==id)]
  if len(bind):row[f'{ph}_selected_submission_id']=bind.iloc[0].official_submission_id
 if valid:
  p=datasets[('prelim',pre)];m=datasets[('main',main)]
  row['crossphase_C_exact_full_n']=overlap(m,p,FULL);row['crossphase_C_exact_qi7_n']=overlap(m,p,QI)
  for c in NUM:
   scale=max(float(pd.concat([BG['prelim'][c],BG['main'][c]]).std()),1e-12)
   wd=wasserstein_distance(p[c],m[c])/scale;bgwd=wasserstein_distance(BG['prelim'][c],BG['main'][c])/scale
   delta=(m[c].mean()-p[c].mean())/scale;bgdelta=(BG['main'][c].mean()-BG['prelim'][c].mean())/scale
   contrasts.append({'team_name':name,'prelim_cohort':pre,'main_cohort':main,'column':c,'C_wasserstein_in_BGsd':wd,'BG_wasserstein_in_BGsd':bgwd,'C_mean_shift_in_BGsd':delta,'BG_mean_shift_in_BGsd':bgdelta,'C_minus_BG_mean_shift':delta-bgdelta,'C_KS':ks_2samp(p[c],m[c]).statistic,'BG_KS':ks_2samp(BG['prelim'][c],BG['main'][c]).statistic,'C_unique_change':m[c].nunique()-p[c].nunique(),'grid_1e-6_change':A.loc[('main',main,c),'grid_1e-06']-A.loc[('prelim',pre,c),'grid_1e-06']})
  for c in ['age']+CAT:
   row[f'{c}_C_TV']=tv(p[c],m[c]);row[f'{c}_BG_TV']=tv(BG['prelim'][c],BG['main'][c]);row[f'{c}_C_shift_minus_BG_shift_TV']=residcat(p[c],m[c],BG['prelim'][c],BG['main'][c])
  cp=p[NUM].corr(method='spearman');cm=m[NUM].corr(method='spearman');bp=BG['prelim'][NUM].corr(method='spearman');bm=BG['main'][NUM].corr(method='spearman')
  cd=[];res=[]
  for j,a in enumerate(NUM):
   for b in NUM[j+1:]:
    dc=cm.loc[a,b]-cp.loc[a,b];db=bm.loc[a,b]-bp.loc[a,b];cd.append(abs(dc));res.append(abs(dc-db));corrs.append({'team_name':name,'column1':a,'column2':b,'prelim_C_spearman':cp.loc[a,b],'main_C_spearman':cm.loc[a,b],'C_change':dc,'BG_change':db,'C_minus_BG_change':dc-db})
  row['spearman_pair_abs_change_mean']=np.nanmean(cd);row['spearman_pair_abs_change_max']=np.nanmax(cd);row['spearman_BGadjusted_abs_change_mean']=np.nanmean(res)
 rows.append(row)
S=pd.DataFrame(rows);S.to_csv(HERE/'team_processing_comparison_metrics.csv',index=False)
pd.DataFrame(contrasts).to_csv(HERE/'numeric_distribution_contrasts.csv',index=False)
pd.DataFrame(corrs).to_csv(HERE/'correlation_contrasts.csv',index=False)
MAP.to_csv(HERE/'verified_team_crosswalk.csv',index=False)
json.dump({'inputs':inputs,'pinned_source_commit':M['pinned_commit'],'cached_C_feature_source':str(CACHED/'C_only_features.csv'),'cached_C_features_sha256':sha(CACHED/'C_only_features.csv'),'cached_rows_reused':20,'attack_models_rerun':0,'identity':'official phase-specific team names, username and selected main submission ID; never number equality','normalization':'NFC prefecture; exact float64 numeric values using round-trip CSV parse; no row identifier, no rounding for exact overlap','B_available':{p:M['phases'][p]['b_self'] for p in B},'limitations':['Only the user team B available in both phases; phase A_bg is a population proxy, not other teams original B.','C shape changes cannot identify algorithm names or separate random seed, parameter, bug fix, dataset, and serialization changes.','23 main teams; 21 paired C comparisons; bot035 preliminary C absent; HiddenMetrics has no mapped preliminary participant.','Main utility ledger is saved 2026-09-24T23:55:23Z (September 25 JST) matching selected submission IDs; not live/latest scoring.','One main submission reportedly padded 1048 to 1049 rows; team not confirmed, duplicated-row findings alone are not definitive.']},open(HERE/'analysis_provenance.json','w'),ensure_ascii=False,indent=2)
print(S[['team_name','prelim_dup_full_n','main_dup_full_n','prelim_time_nunique','main_time_nunique','prelim_time_mode_fraction','main_time_mode_fraction','spearman_pair_abs_change_mean']].to_string(index=False))
