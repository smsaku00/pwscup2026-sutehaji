from run_archive import *
import math
from scipy.stats import rankdata
p=ORIG/'予備戦/攻撃フェーズ/分析/20_Team20希少のGenLRA検証/分析.py'
gen=extract(p,['numeric','distance2','logsumexp','category_frequencies','log_category','log_kernel','influence','ordered','minimum_rank','compute_feature'],dict(np=np,pd=pd,math=math,RHO=.98,FACTORS=[.5,1.,2.],KS=[1,10,20],PRIMARY='qi7_h1_k10'))
rows=[];started=time.time();matrices={}
for bg in BGS:
 root=setup(bg);audit=module('_rare_archive_inputs',root/'wrappers/official_audit.py');b,private,rare,normal,dist=audit.load_inputs(root)
 rr=b.loc[rare].sort_values('record_id').reset_index(drop=True);ids=rr.record_id.astype(int).tolist();priv=set(b.loc[private&rare,'record_id'].astype(int));eligible=[rid for rid in ids if rid not in priv];remaining=[rid for rid in eligible if rid not in [201,238]]
 assert len(ids)==24 and len(eligible)==11 and len(remaining)==9
 a=pd.read_csv(root/'予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv',float_precision='round_trip');c=pd.read_csv(dist/'C_20.csv',float_precision='round_trip')
 out=BASE/'rare_secondary'/bg;out.mkdir(parents=True,exist_ok=True);mats={};metadata=[]
 for name,nums,cats in [('qi7',['age','SBP','TG','HDL','ALT'],['sex','prefecture']),('attrs10',NUM7,['sex','prefecture','smoking'])]:
  cfg,rrr,settings,contributions,baselines=gen.compute_feature(name,nums,cats,a,c,rr,ids,eligible,remaining)
  metadata.append(cfg)
  table=pd.DataFrame(rrr)
  for setting,g in table.groupby('setting'):mats[setting]=g.set_index('record_id').loc[ids,'influence'].to_numpy()
  table=pd.DataFrame(baselines)
  for factor,g in table.groupby('bandwidth_factor'):
   mats[f'{name}_point_h{factor:g}']=g.set_index('record_id').loc[ids,'pointwise_log_density_ratio'].to_numpy()
   if factor==.5:mats[f'{name}_DCR']=g.set_index('record_id').loc[ids,'dcr_score'].to_numpy()
  (out/f'{name}_candidate_diagnostics.json').write_text(json.dumps({'rows':rrr,'settings':settings,'baselines':baselines},ensure_ascii=False,indent=2))
 assert len(mats)==26
 np.savez_compressed(out/'frozen.npz',**mats);matrices[bg]=mats
 eligible_idx=[ids.index(r) for r in eligible];remaining_idx=[ids.index(r) for r in remaining]
 for name,s in mats.items():
  assert np.isfinite(s).all()
  for scope,ix in [('all24',range(24)),('eligible11',eligible_idx),('remaining9_historical',remaining_idx)]:
   ss=s[list(ix)];rows.append(dict(background=bg,team=20,method=name,scope=scope,n_candidates=len(ss),score_min=float(ss.min()),score_max=float(ss.max()),score_mean=float(ss.mean()),n_top_ties=int((ss==ss.max()).sum()),score_std=float(ss.std())))
 (out/'completion.json').write_text(json.dumps({'n_methods':26,'source_sha256':sha(p),'frozen_sha256':sha(out/'frozen.npz'),'prediction_only':True,'membership_truth_read':False,'historical_remaining_mask':'11public-eligible minus source-explicit201/238; no posterior reconstructed','n_background':len(a),'preprocessing':metadata},ensure_ascii=False,indent=2)+'\n')
 print('RARE',bg,26,'elapsed',time.time()-started,flush=True)
pd.DataFrame(rows).to_csv(BASE/'rare_secondary_summary.csv',index=False,float_format='%.17g')
changes=[]
for bg,mats in matrices.items():
 for name,s in mats.items():
  t=matrices['prelim'][name]
  for scope,ix in [('all24',range(24)),('eligible11',eligible_idx),('remaining9_historical',remaining_idx)]:
   ix=list(ix);rx=rankdata(t[ix]);ry=rankdata(s[ix]);changes.append(dict(background=bg,method=name,scope=scope,n_candidates=len(ix),spearman_prelim=float(np.corrcoef(rx,ry)[0,1]),top_tied_set_equal=bool(np.array_equal(t[ix]==t[ix].max(),s[ix]==s[ix].max())),max_abs_score_change=float(np.max(np.abs(s[ix]-t[ix])))))
pd.DataFrame(changes).to_csv(BASE/'rare_secondary_background_changes.csv',index=False,float_format='%.17g')
(BASE/'rare_secondary_completion.json').write_text(json.dumps({'methods':26,'backgrounds':7,'conditions':182,'aggregate_rows':len(rows),'elapsed_seconds':time.time()-started,'historical_posterior':'not_recomputed_missing_history_state','score_only_secondary':True},indent=2)+'\n')
