"""Apply the source-defined 20-fold multiview ensemble to main; no main labels."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import joblib,sklearn,scipy
import runtime as r
from run_family import STAGE,WORK,dump,SCRIPTS
from run_learned_special import read_raw
from execution_protocol import PROTOCOL

CONTEXTS=('full_only','background47','center10','balanced_background_and_centers')

def run(bg):
    u=r.initialize();data=r.data_for(u,bg,'main');raw,featurepath,teams=read_raw(bg,phase='main');assert list(teams)==data['teams']
    f=r.import_path('main_projection_features',r.SRC/'特徴を外して学習を確認.py')
    cp=Path('/workspace/shared/mia_recovered_calibration/cache/fits/learned_multiview')/bg
    assert json.loads((cp/'progress.json').read_text())['complete']
    policy=json.loads((cp/'policy.json').read_text());assert policy['C']==.1 and policy['max_iter']==1500 and policy['background_views']==47 and policy['center_views']==10
    for p,h in policy['input_hashes'].items():assert r.sha(p)==h,p
    training=STAGE/'learned_multiview'/bg
    assert json.loads((training/'実行記録.json').read_text())['forest_prediction_execution']==PROTOCOL
    out=WORK/bg/'learned_runtime/複数ビューの本戦順位診断';dest=STAGE/'learned_multiview_main'/bg
    assert not out.exists();out.mkdir(parents=True);dest.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();models={'gen7_county500':raw[:,:,-1]};checkpoint_hashes={}
    for panel,indices in {'all12':list(range(12)),'without_kde':[j for j in range(12) if j!=1]}.items():
        x=f.panel_features(raw,indices);nc=x.shape[-1]
        for context in CONTEXTS:
            draws=[]
            for seed in f.SEEDS:
                name=f'{panel}_{context}_seed{seed}';fits=[]
                for pf in range(5):
                    for gi in range(4):
                        p=cp/name/f'person{pf}_target{gi}.joblib';scaler,fit=joblib.load(p);checkpoint_hashes[str(p)]=r.sha(p)
                        assert fit.C==.1 and fit.max_iter==1500 and fit.class_weight=='balanced'
                        decision=fit.decision_function(scaler.transform(x.reshape(-1,nc))).reshape(x.shape[:2])
                        fits.append(np.column_stack([rankdata(col,method='average')/875 for col in decision.T]))
                score=np.mean(fits,axis=0);draws.append(score);models[name]=score
            models[f'{panel}_{context}_five_seed_mean']=np.mean(draws,axis=0)
    assert len(models)==49 and all(v.shape==(875,22) and np.isfinite(v).all() for v in models.values())
    names=sorted(models);p=out/'内部順位キャッシュ.npz';np.savez_compressed(p,names=names,scores=np.stack([models[n] for n in names]),teams=data['teams'],record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
    hashes=[dict(method=n,shape=[875,22],sha256=r.matsha(models[n])) for n in names];dump(dest/'prediction_hashes.json',hashes)
    rows=[];control=models['gen7_county500']
    for name,mat in models.items():
        for j,t in enumerate(data['teams']):
            a,b=control[:,j],mat[:,j];ra,rb=rankdata(a),rankdata(b)
            rows.append(dict(method=name,team=t,top12_overlap_control=len(set(np.argsort(-a,kind='stable')[:12])&set(np.argsort(-b,kind='stable')[:12])),top20_overlap_control=len(set(np.argsort(-a,kind='stable')[:20])&set(np.argsort(-b,kind='stable')[:20])),spearman_control=float(np.corrcoef(ra,rb)[0,1]),main_truth_used=False))
    pd.DataFrame(rows).to_csv(dest/'rank_by_target.csv',index=False,float_format='%.17g')
    dump(out/'private_fit_provenance.json',dict(fit_sha256=checkpoint_hashes,feature_file_sha256=r.sha(featurepath),training_policy_sha256=r.sha(cp/'policy.json')))
    dump(dest/'実行記録.json',dict(complete=True,phase='main',family='learned_multiview_main',background=bg,models=49,source='複数背景と乱数で学習.py',source_sha256=r.sha(r.SRC/'複数背景と乱数で学習.py'),background_sha256=r.MANIFEST['subsets'][bg]['sha256'],main_truth_used=False,rank_only=True,training='Original preliminary18 OOF fits,20 folds per seed and5 fixed seeds',training_protocol_modified=False,forest_prediction_execution=PROTOCOL,fit_count=800,new_fit_count=0,feature_file_sha256=r.sha(featurepath),prediction_cache_sha256=r.sha(p),seconds=time.perf_counter()-start,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__))
    dump(dest/'completion.json',dict(source_main_returned=True,models=49,original_main_projection_complete=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bg',choices=r.BG,required=True);run(p.parse_args().bg)
