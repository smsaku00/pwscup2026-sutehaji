"""Nested center-seed experiment; original fit equation, two fixed panels, all20 targets.
Only source-equivalent public-data kernel sufficient-statistic caching is added.
No membership labels are read until all target predictions for a checkpoint freeze.
"""
from pathlib import Path
import os,sys,time,json,hashlib,shutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import roc_curve
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/workspace/shared/mia_recovered_calibration')
import runtime as r
sys.path.insert(0,str(r.SRC))
u=r.import_path('共通',r.SRC/'共通.py');u.ROOT=r.ROOT;u._setup_original=u.setup
# Resolve only the recovered official-kit path; scorer and original evaluate stay unchanged.
_original_child=u.base.child
def _resolve_child(parent,child_name):
 if child_name=='pwscup2026-kit-main_revised0912':return r.RECOVERY/'originals/本戦/配布データ/pwscup2026-kit-main_revised0912'
 return _original_child(parent,child_name)
u.base.child=_resolve_child
orig=r.import_path('original_attribute_seeds',r.SRC/'密度比の属性と乱数を確認.py')
BGS=['starter+prelim+main','prelim'];NS=[10,30,100,1000]
PANELS=[('numeric8_sexpref',u.NUM13,u.CAT7,'mean',.1,12),('without_outcomes',u.NUM9,u.CAT7+('smoking',),'median',0.,3)]
START=time.perf_counter();HASHES=[];SCORES=[];TIMES=[];EQUIV=[]
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def log(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True);dump(OUT/'progress.json',dict(elapsed_seconds=time.perf_counter()-START,**kw))
def name(panel):return f'{panel[0]}_rank_{panel[3]}__raw__cross{panel[4]:g}'
def cachepaths(bg,panel,t):
 root=OUT/'cache'/bg/panel[0];root.mkdir(parents=True,exist_ok=True);return root/f't{t}_sufficient.npz',root/f't{t}_ranks.npy',root/f't{t}_progress.json'
def prepare(data,bg,panel,t,c):
 p,rp,meta=cachepaths(bg,panel,t)
 if p.exists():return
 st=time.perf_counter();a=data['a'];q=data['b'];nums,cats=panel[1:3]
 ar,cr,x=orig.transform(a,[a,c,q],nums,'robust')
 ka=np.exp(-.5*orig.kernel_dist(ar,cr,a,c,cats));kc=np.exp(-.5*orig.kernel_dist(cr,cr,c,c,cats));kx=np.exp(-.5*orig.kernel_dist(x,cr,q,c,cats))
 gram=.9*(ka.T@ka/len(a))+.1*(kc.T@kc/len(c));mean=kc.mean(axis=0)
 # Check exact original scoring/ranking for first ten seeds, not merely final TPR.
 errors=[];ranks=[]
 for seed in range(20261008,20261018):
  ii=np.random.default_rng(seed+t).permutation(len(c))[:100]
  beta=np.maximum(0,np.linalg.solve(gram[np.ix_(ii,ii)]+np.eye(100),mean[ii]));v=kx[:,ii]@beta
  original=orig.one_score(a,c,q,nums,cats,seed+t)
  rr=rankdata(original,method='average')/len(q);fast=rankdata(v,method='average')/len(q)
  error=float(np.max(abs(original-v)));assert error<1e-12,(bg,panel,t,error)
  assert np.array_equal(rr,fast),(bg,panel,t,seed,'rank mismatch')
  errors.append(error);ranks.append(rr)
 np.savez(p,gram=gram,mean=mean,kx=kx)
 arr=np.lib.format.open_memmap(rp,mode='w+',dtype='float64',shape=(1000,len(q)));arr[:10]=np.stack(ranks);arr.flush();del arr
 record=dict(background=bg,panel=panel[0],team=t,completed=10,seconds=time.perf_counter()-st,original_first10_max_raw_error=max(errors),original_first10_all_ranks_exact=True)
 dump(meta,record);log(stage='prepared',**record)
def extend(data,bg,panel,t,c,n):
 p,rp,mp=cachepaths(bg,panel,t);meta=json.loads(mp.read_text());old=meta['completed']
 if old>=n:return
 st=time.perf_counter();z=np.load(p);gram=z['gram'];mean=z['mean'];kx=z['kx'];q=len(data['b']);arr=np.load(rp,mmap_mode='r+')
 for i in range(old,n):
  seed=20261008+i;ii=np.random.default_rng(seed+t).permutation(len(c))[:100]
  beta=np.maximum(0,np.linalg.solve(gram[np.ix_(ii,ii)]+np.eye(100),mean[ii]));value=kx[:,ii]@beta
  arr[i]=rankdata(value,method='average')/q
  if (i+1)%100==0:arr.flush();meta['completed']=i+1;dump(mp,meta)
 arr.flush();meta['completed']=n;meta.setdefault('extensions',[]).append(dict(start=old+1,end=n,seconds=time.perf_counter()-st));dump(mp,meta)
def models_from(data,bg,start,end):
 result={}
 for panel in PANELS:
  cols=[]
  for t in data['teams']:
   _,rp,mp=cachepaths(bg,panel,t);assert json.loads(mp.read_text())['completed']>=end
   ranks=np.load(rp,mmap_mode='r')[start:end]
   v=ranks.mean(axis=0) if panel[3]=='mean' else np.median(ranks,axis=0)
   cols.append(u.rank(v,data['normal']))
  mat=np.column_stack(cols);med=np.column_stack([np.median(np.delete(mat,j,axis=1),axis=1) for j in range(len(data['teams']))])
  result[name(panel)]=mat-panel[4]*med
 return result
def freeze(data,bg,n,models):
 root=OUT/'predictions'/bg;root.mkdir(parents=True,exist_ok=True)
 for method,mat in models.items():
  path=root/f'N{n}_{method}.npz';np.savez_compressed(path,scores=mat,teams=np.asarray(data['teams']),record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy(),full_confidence=np.column_stack([u.base.confidence(mat[:,j],data['normal'],data['rare'],data['private']) for j in range(len(data['teams']))]))
  HASHES.append(dict(background=bg,N=n,method=method,score_sha256=r.matsha(mat),file_sha256=r.sha(path),file=str(path.relative_to(OUT))))
 dump(OUT/'prediction_hashes.json',HASHES)
def evaluate(data,bg,n,models):
 table,err,truth=u.evaluate(data,models);assert err<1e-12
 table.insert(0,'N',n);table.insert(0,'background',bg);SCORES.append(table);pd.concat(SCORES).to_csv(OUT/'scores_all_targets.csv',index=False,float_format='%.17g')
 if n==10:
  oldpath=Path('/workspace/shared/mia_recovered_calibration/work')/bg/'密度比_属性乱数結果/内部順位キャッシュ.npz';z=np.load(oldpath);names=list(z['names']);teams=list(z['teams']);assert teams==data['teams']
  checks=[]
  for method,mat in models.items():
   old=z['scores'][names.index(method)];assert np.array_equal(mat,old),(bg,method,'historical prediction mismatch',np.max(abs(mat-old)))
   checks.append(dict(method=method,all20_predictions_bitwise_equal=True,max_error=float(abs(mat-old).max())))
  focus=table[(table.fpr==.01)&(((table.team==12)&(table.method==name(PANELS[0])))|((table.team==3)&(table.method==name(PANELS[1]))))]
  expected={12:4/26,3:.12}
  for _,row in focus.iterrows():assert abs(row.mia_all-expected[row.team])<1e-12,(bg,row.to_dict())
  dump(OUT/f'N10_regression_{bg}.json',dict(previous_cache=str(oldpath),cache_sha256=r.sha(oldpath),checks=checks,focus=focus.to_dict('records')))
 rows=table[(table.fpr==.01)&(((table.team==12)&(table.method==name(PANELS[0])))|((table.team==3)&(table.method==name(PANELS[1]))))].to_dict('records');log(stage='checkpoint_scored',background=bg,N=n,focus=rows,official_max_error=err)
 return truth

def loadmodels(bg,n):
 return {name(p):np.load(OUT/'predictions'/bg/f'N{n}_{name(p)}.npz')['scores'] for p in PANELS}
def diagnostics(data,bg,truth):
 comparison=[];boundaries=[];blocks=[]
 models={n:loadmodels(bg,n) for n in NS}
 for n,m in models.items():
  for method,mat in m.items():
   for j,t in enumerate(data['teams']):
    conf=u.base.confidence(mat[:,j],data['normal'],data['rare'],data['private']);fp,tp,th=roc_curve(truth[:,j],conf,drop_intermediate=False)
    low=np.flatnonzero(fp<=.01)[-1];up=np.flatnonzero(fp>.01)[0]
    boundaries.append(dict(background=bg,N=n,method=method,team=t,members=int(truth[:,j].sum()),nonmembers=int((1-truth[:,j]).sum()),low_fpr=float(fp[low]),high_fpr=float(fp[up]),low_tpr=float(tp[low]),high_tpr=float(tp[up]),low_threshold=float(th[low]),high_threshold=float(th[up]),low_fp=float(fp[low]*(1-truth[:,j]).sum()),high_fp=float(fp[up]*(1-truth[:,j]).sum()),low_tp=float(tp[low]*truth[:,j].sum()),high_tp=float(tp[up]*truth[:,j].sum())))
 for na,nb in [(10,30),(30,100),(100,1000),(10,1000),(30,1000)]:
  for method in models[na]:
   for j,t in enumerate(data['teams']):
    a=models[na][method][:,j];b=models[nb][method][:,j];row=dict(background=bg,method=method,team=t,N_from=na,N_to=nb,spearman=float(spearmanr(a,b).statistic))
    for top in [10,20,50]:row[f'top{top}_overlap']=len(set(np.argsort(-a,kind='stable')[:top])&set(np.argsort(-b,kind='stable')[:top]))
    comparison.append(row)
 for blocksize,total in [(10,100),(100,1000)]:
  for start in range(0,total,blocksize):
   bm=models_from(data,bg,start,start+blocksize)
   # Whole all20 block predictions are frozen to an audit cache before evaluation.
   root=OUT/'blocks'/bg;root.mkdir(parents=True,exist_ok=True);np.savez_compressed(root/f'block{blocksize}_{start//blocksize}.npz',**bm)
   table,_,_=u.evaluate(data,bm)
   for panel in PANELS:
    method=name(panel);t=panel[-1];j=data['teams'].index(t);a=bm[method][:,j];b=models[total][method][:,j]
    score=float(table[(table.method==method)&(table.team==t)&(table.fpr==.01)].mia_all.iloc[0])
    blocks.append(dict(background=bg,method=method,team=t,blocksize=blocksize,block=start//blocksize+1,first_seed=20261008+start,last_seed=20261008+start+blocksize-1,tpr_001=score,spearman_vs_cumulative=float(spearmanr(a,b).statistic),top20_overlap_vs_cumulative=len(set(np.argsort(-a,kind='stable')[:20])&set(np.argsort(-b,kind='stable')[:20]))))
 return comparison,boundaries,blocks

def run():
 source=OUT/'source'/'originals';source.mkdir(exist_ok=True)
 for p in [r.SRC/'密度比の属性と乱数を確認.py',r.SRC/'共通.py',r.SRC/'距離と分布の探索.py',r.SRC/'直接密度比の探索.py',u.OLD/'分析.py',u.CAL/'分析.py',Path(r.__file__)]:
  dest=source/(str(p.parent.name)+'__'+p.name);shutil.copy2(p,dest)
 pd.DataFrame([dict(repeat=i+1,base_seed=20261008+i,target=t,actual_seed=20261008+i+t) for i in range(1000) for t in [1,10,12,13,17,18,19,2,20,22,23,24,26,29,3,4,6,7,8,9]]).to_csv(OUT/'seeds.csv',index=False)
 datas={bg:r.data_for(u,bg) for bg in BGS}
 for bg,data in datas.items():assert (len(data['b']),int(data['normal'].sum()),int((data['private'] & ~data['rare']).sum()),int(data['rare'].sum()))==(1049,875,150,24)
 for n in NS:
  for bg,data in datas.items():
   st=time.perf_counter()
   for panel in PANELS:
    for t,c in data['cs'].items():
     prepare(data,bg,panel,t,c);extend(data,bg,panel,t,c,n)
   models=models_from(data,bg,0,n);freeze(data,bg,n,models);truth=evaluate(data,bg,n,models)
   TIMES.append(dict(background=bg,N=n,stage_seconds=time.perf_counter()-st,cumulative_wall_seconds=time.perf_counter()-START));dump(OUT/'timings.json',TIMES)
 comparisons=[];boundaries=[];blocks=[]
 for bg,data in datas.items():
  _,_,truth=u.evaluate(data,loadmodels(bg,1000));a,b,c=diagnostics(data,bg,truth);comparisons+=a;boundaries+=b;blocks+=c
 pd.DataFrame(comparisons).to_csv(OUT/'rank_stability.csv',index=False,float_format='%.17g');pd.DataFrame(boundaries).to_csv(OUT/'fpr1_boundaries.csv',index=False,float_format='%.17g');pd.DataFrame(blocks).to_csv(OUT/'seed_block_stability.csv',index=False,float_format='%.17g')
 dump(OUT/'run_record.json',dict(complete=True,seconds=time.perf_counter()-START,python=sys.executable,numpy=np.__version__,pandas=pd.__version__,backgrounds={bg:r.MANIFEST['subsets'][bg] for bg in BGS},source_sha256={str(p.relative_to(OUT)):r.sha(p) for p in (OUT/'source').rglob('*') if p.is_file()},input_sha256=u.base.INPUT_HASHES,protocol=dict(seeds='20261008+i+target, i=0..999',checkpoints=NS,centers=100,h=1,relative=.1,regularizer=1,rank_scope=1049,normal_rerank=875,private_confidence=0,rare_confidence=.4,training='A and C only, no membership-label fit',query='B only, never in fit',cross='target-excluded median of same-family all20 reranks per N',optimization='Precomputed full public C center kernel Gram; first10 raw predictions within1e-12 and all per-seed ranks exactly verified against original at all80 BG-panel-target cases',evaluation='Official pwscup2026 _tpr_at_fpr, after checkpoint predictions frozen',exploratory=True)))
 log(stage='complete',seconds=time.perf_counter()-START)
if __name__=='__main__':run()
