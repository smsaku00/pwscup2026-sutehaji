"""Original learned recipes with the old external BG axis replaced by BG7.

Intrinsic training views, person folds, target folds, weights, regularization,
and max_iter are retained. Only the selected outer BG's full output is scored.
Private per-fit checkpoints make the expensive multiview jobs resumable.
"""
from pathlib import Path
import argparse,json,time,sys,hashlib,resource
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib,sklearn,scipy,platform
import runtime as r
from run_family import configure,WORK,STAGE,SCRIPTS,FOLDERS,dump
from generate_features import CACHE,VIEWCACHE,CONDITIONS
from execution_protocol import PROTOCOL

def read_raw(bg,label='full',phase='prelim'):
    folder=CACHE/bg if label=='full' else VIEWCACHE/bg/label
    if phase!='prelim':folder=folder/phase
    p=folder/'代表特徴.npz';z=np.load(p)
    meta=json.loads((folder/'record.json').read_text());assert meta['background_sha256']==r.MANIFEST['subsets'][bg]['sha256'];assert meta['score_sha256']==r.matsha(z['scores']);assert meta.get('forest_prediction_execution')==PROTOCOL
    return z['scores'],p,z['teams']

def save_outputs(u,a,work,data,models,specs,start,extra,details=None):
    out=work/FOLDERS[a.family]
    u.save(out,data,models,specs,start,extra)
    if details is not None:
        for filename,rows in details.items():
            pd.DataFrame(rows).to_csv(out/filename,index=False)
            pd.DataFrame(rows).to_csv(STAGE/a.family/a.bg/filename,index=False)

def background(u,a,work,data,raw,start):
    f=r.import_path('special_ablation',r.SRC/'特徴を外して学習を確認.py')
    y=f.load_truth(data);splits=[];specs=[];models={'gen7_county500':raw[:,:,-1]}
    indices=[j for j in range(12) if j!=1] if a.family=='learned_bg_ablation' else list(range(12))
    array=f.panel_features(raw,indices)
    settings=[('fourfold',v) for v in (.01,.1,1.)]
    if a.family=='learned_bg_ablation':settings += [('leave_one_target_out',.1)]
    for split,strength in settings:
        draws=[]
        for seed in f.SEEDS:
            output,details,_=f.crossfit(array,y,data['teams'],seed,strength,split)
            draws.append(output)
            names=[f'without_kde_{split}_C{strength:g}_seed{seed}'] if a.family=='learned_bg_ablation' else [f'{context}_C{strength:g}_seed{seed}' for context in ('full','same_background')]
            for name in names:
                models[name]=output;specs.append(dict(method=name,seed=seed,C=strength,split=split))
                splits.extend(dict(method=name,**row) for row in details)
        names=[f'without_kde_{split}_C{strength:g}_five_seed_mean'] if a.family=='learned_bg_ablation' else [f'{context}_C{strength:g}_five_seed_mean' for context in ('full','same_background')]
        for name in names:models[name]=np.mean(draws,axis=0);specs.append(dict(method=name,seeds=list(f.SEEDS),C=strength,split=split))
        print('BG_LEARNING',a.family,a.bg,split,strength,time.perf_counter()-start,flush=True)
    assert len(models)==(25 if a.family=='learned_bg_ablation' else 37)
    extra={'label_usage':'Original 18-target person/target OOF training. Adaptive historical exploration, not independent validation.','old_external_BG_conditions_replaced_by_seven':True,'full_and_same_background_rebased_to_selected_BG':a.family=='learned_bg_transfer','intrinsic_sampling_changed':False}
    save_outputs(u,a,work,data,models,specs,start,extra,{'交差検証分割.csv':splits})

def multiview(u,a,work,data,raw,start):
    f=r.import_path('special_multiview_ablation',r.SRC/'特徴を外して学習を確認.py')
    views=[];hashes={}
    for label,_,_ in CONDITIONS:
        v,p,teams=read_raw(a.bg,label);assert list(teams)==data['teams'];views.append(v);hashes[str(p)]=r.sha(p)
    views=np.stack(views)
    p=WORK/a.bg/'密度比_属性乱数結果/内部順位キャッシュ.npz';hashes[str(p)]=r.sha(p);z=np.load(p);assert list(z['teams'])==data['teams'];cn=list(z['names']);centers=[]
    for seed in range(20261008,20261018):
        changed=raw.copy();changed[:,:,7]=z['scores'][cn.index(f'all13_seed{seed}__raw__cross0')];centers.append(changed)
    y=f.load_truth(data);valid=np.array([j for j,t in enumerate(data['teams']) if t not in (1,2)])
    tfold=np.array([sorted(data['teams']).index(t)%4 for t in data['teams']])
    loo=a.family=='learned_multiview_loo'
    groups=[np.array([j]) for j in valid]+[np.array([j for j,t in enumerate(data['teams']) if t in (1,2)])] if loo else [np.flatnonzero(tfold==tf) for tf in range(4)]
    panels={'without_kde':[j for j in range(12) if j!=1]} if loo else {'all12':list(range(12)),'without_kde':[j for j in range(12) if j!=1]}
    cp=Path('/workspace/shared/mia_recovered_calibration/cache/fits')/a.family/a.bg;cp.mkdir(parents=True,exist_ok=True)
    policy={'input_hashes':hashes,'source_sha256':r.sha(r.SRC/SCRIPTS[a.family]),'teams':data['teams'],'person_seeds':list(f.SEEDS),'C':.1,'max_iter':1500,'background_views':47,'center_views':10,'all_views_share_people_and_target_holdouts':True,'numeric_versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__},'forest_prediction_execution':PROTOCOL,'adapter_sha256':{p.name:r.sha(p) for p in [Path(__file__),Path(__file__).with_name('runtime.py'),Path(__file__).with_name('run_family.py'),Path(__file__).with_name('execution_protocol.py')]},'original_helper_sha256':{name:r.sha(r.SRC/name) for name in ['代表特徴を計算.py','特徴を外して学習を確認.py','背景キャッシュで特徴除去を確認.py','共通.py']}}
    if (cp/'policy.json').exists():assert json.loads((cp/'policy.json').read_text())==policy
    else:dump(cp/'policy.json',policy)
    models={'gen7_county500':raw[:,:,-1]};specs=[];splits=[];convergence=[]
    fit_count=0;new_fit_count=0;new_fit_seconds=0.;first_context_fits={}
    for panel,indices in panels.items():
        x=np.stack([f.panel_features(v,indices) for v in views]);cx=np.stack([f.panel_features(v,indices) for v in centers])
        settings=[('full_only',x[[0]],np.ones(1)),('background47',x,np.full(47,1/47)),('center10',cx,np.full(10,1/10)),('balanced_background_and_centers',np.concatenate([x,cx]),np.r_[np.full(47,.5/47),np.full(10,.5/10)])]
        for context,train_views,view_weight in settings:
            draws=[]
            for seed in f.SEEDS:
                name=f'without_kde_LOO_{context}_seed{seed}' if loo else f'{panel}_{context}_seed{seed}'
                folder=cp/name;folder.mkdir(exist_ok=True);output=np.full(y.shape,np.nan)
                fold=np.empty(875,int);fold[np.random.default_rng(seed).permutation(875)]=np.arange(875)%5
                for pf in range(5):
                    trp=np.flatnonzero(fold!=pf);tep=np.flatnonzero(fold==pf)
                    for gi,tet in enumerate(groups):
                        trt=np.setdiff1d(valid,tet);nc=x.shape[3]
                        assert not np.intersect1d(trp,tep).size and not np.intersect1d(trt,tet).size and all(data['teams'][j] not in (1,2) for j in trt)
                        fitpath=folder/f'person{pf}_target{gi}.joblib'
                        fit_started=time.perf_counter();new_fit=not fitpath.exists()
                        if not new_fit:scaler,fit=joblib.load(fitpath)
                        else:
                            train=train_views[:,trp,:,:][:,:,trt,:].reshape(-1,nc);truth=np.tile(y[np.ix_(trp,trt)].ravel(),len(train_views));weight=np.repeat(view_weight,len(trp)*len(trt))
                            scaler=StandardScaler().fit(train,sample_weight=weight)
                            fit=LogisticRegression(C=.1,class_weight='balanced',max_iter=1500);fit.fit(scaler.transform(train),truth,sample_weight=weight)
                            joblib.dump((scaler,fit),fitpath.with_suffix('.tmp'));fitpath.with_suffix('.tmp').replace(fitpath)
                        fit_count+=1
                        if new_fit:
                            duration=time.perf_counter()-fit_started;new_fit_count+=1;new_fit_seconds+=duration
                            if panel+'_'+context not in first_context_fits:
                                first_context_fits[panel+'_'+context]=dict(seconds=duration,training_rows=len(train),columns=nc,views=len(train_views),iterations=int(fit.n_iter_[0]),process_peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                                dump(cp/'first_fit_by_context.json',first_context_fits)
                        if fit_count%20==0:
                            pending=cp/'fit_progress.pending.json';dump(pending,dict(completed_original_fits=fit_count,total_original_fits=1900 if loo else 800,new_fit_count=new_fit_count,new_fit_seconds=new_fit_seconds,elapsed_seconds=time.perf_counter()-start,last_method=name,last_person_fold=pf,last_target_group=gi,process_peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)));pending.replace(cp/'fit_progress.json')
                        decision=fit.decision_function(scaler.transform(x[0,:,tet,:].transpose(1,0,2).reshape(-1,nc))).reshape(875,len(tet))
                        ranked=np.column_stack([rankdata(col,method='average')/875 for col in decision.T]);assert np.isnan(output[np.ix_(tep,tet)]).all();output[np.ix_(tep,tet)]=ranked[tep]
                        splits.append(dict(method=name,person_fold=pf,target_group=gi,training_views=len(train_views),effective_training_cells=float(view_weight.sum()*len(trp)*len(trt)),train_targets=';'.join(str(data['teams'][j]) for j in trt),test_targets=';'.join(str(data['teams'][j]) for j in tet),train_people=len(trp),test_people=len(tep)))
                        convergence.append(dict(method=name,person_fold=pf,target_group=gi,iterations=int(fit.n_iter_[0]),max_iter=1500))
                assert np.isfinite(output).all();draws.append(output);models[name]=output
                specs.append(dict(method=name,panel=panel,training_context=context,seed=seed,C=.1,view_weight_sum=float(view_weight.sum())))
                np.save(folder/'output.npy',output)
                print('MULTIVIEW_SEED',a.family,a.bg,name,time.perf_counter()-start,flush=True)
                dump(cp/'progress.json',dict(last_completed=name,elapsed_seconds=time.perf_counter()-start,complete=False))
            name=f'without_kde_LOO_{context}_five_seed_mean' if loo else f'{panel}_{context}_five_seed_mean';models[name]=np.mean(draws,axis=0);specs.append(dict(method=name,panel=panel,training_context=context,C=.1,seeds=list(f.SEEDS)))
        del x,cx,settings,train_views
        try:del train,truth,weight
        except UnboundLocalError:pass
    assert len(models)==(25 if loo else 49)
    save_outputs(u,a,work,data,models,specs,start,{'label_usage':'Original 18-target OOF training, identical held-out people and targets across all57 views. Historical adaptive exploration.','intrinsic_background_views':47,'intrinsic_center_views':10,'predicted_external_BG_output':'Full selected BG; old46 external subsample output projections are not multiplied into the seven-condition axis.','main_projection':'Fold fits retained privately for original main-only rank projection; no main truth.'},{'交差検証分割.csv':splits,'収束の確認.csv':convergence})
    dump(cp/'progress.json',dict(elapsed_seconds=time.perf_counter()-start,complete=True))

def run(a):
    u=r.initialize();work=configure(u,a);start=time.perf_counter();data=u.setup()
    raw,p,teams=read_raw(a.bg);assert list(teams)==data['teams']
    assert not (work/FOLDERS[a.family]).exists()
    if a.family in ('learned_bg_ablation','learned_bg_transfer'):background(u,a,work,data,raw,start)
    else:multiview(u,a,work,data,raw,start)
    dump(STAGE/a.family/a.bg/'completion.json',{'source_main_returned':True,'family':a.family,'background':a.bg,'adapted_original_recipe_returned':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['learned_bg_ablation','learned_bg_transfer','learned_multiview','learned_multiview_loo'],required=True);p.add_argument('--bg',choices=r.BG,required=True);run(p.parse_args())
