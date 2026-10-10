"""Refresh only the forest feature affected by parallel summation order."""
from pathlib import Path
import json,time
import numpy as np
from scipy.special import logit
import runtime as r
from execution_protocol import PROTOCOL

def ensure(u,bg,phase='prelim',label='full',fraction=1.,seed=None):
    base=Path('/workspace/shared/mia_recovered_calibration/cache')
    folder=base/'representative'/bg if label=='full' else base/'multiview'/bg/label
    if phase!='prelim':folder=folder/phase
    p=folder/'代表特徴.npz';mp=folder/'record.json'
    if not p.exists() or not mp.exists():return False
    meta=json.loads(mp.read_text())
    if meta.get('forest_prediction_execution')==PROTOCOL:return True
    if bg=='prelim' and phase=='prelim' and label=='full':
        progress=Path('/workspace/shared/mia_recovered_calibration/work/prelim/learned_runtime/低FPR学習結果/進捗.json')
        frozen=Path('/workspace/shared/mia_all_bg_recovered_calibration/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_calibration/learned_lowfpr/prelim/実行記録.json')
        while progress.exists() and not frozen.exists():
            print('WAIT_FOR_PARALLEL_BASELINE_TO_FREEZE',flush=True);time.sleep(15)
    assert meta['background_sha256']==r.MANIFEST['subsets'][bg]['sha256']
    z=np.load(p);raw=z['scores'];assert r.matsha(raw)==meta['score_sha256'];teams=z['teams'];ids=z['record_ids']
    data=r.data_for(u,bg,phase);assert list(teams)==data['teams']
    a=data['a'] if label=='full' else u.cal.stratified_subset(data['a'],fraction,seed)
    m=r.import_path('serial_representative_refresh',r.SRC/'代表特徴を計算.py')
    start=time.perf_counter()
    for j,t in enumerate(data['teams']):
        ar,cr,x=m.encode(a,data['cs'][t],data['query'],u.NUM7,u.CAT7,'robust')
        model=dict((name,model) for name,model,_ in m.source_models(20261008))['extra_leaf1']
        model.fit(np.vstack([ar,cr]),np.r_[np.zeros(len(ar),int),np.ones(len(cr),int)])
        score=logit(np.clip(model.predict_proba(x)[:,1],1e-8,1-1e-8))
        raw[:,j,9]=m.conditional_rank(score,data,t,county=True)
    temp=folder/'serial_refresh.pending.npz';np.savez_compressed(temp,scores=raw,teams=teams,record_ids=ids)
    old_hash=meta['score_sha256'];meta.update(forest_prediction_execution=PROTOCOL,parallel_control_score_sha256=old_hash,score_sha256=r.matsha(raw),file_sha256=r.sha(temp),serial_refresh_seconds=time.perf_counter()-start,serial_refreshed_channel_index=9)
    backup=folder/'代表特徴_parallel_initial.npz';bm=folder/'record_parallel_initial.json';assert not backup.exists() and not bm.exists()
    mp.replace(bm);p.replace(backup);temp.replace(p)
    pending=folder/'record.pending.json';pending.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');pending.replace(mp)
    print('SERIAL_FEATURE_REFRESH',bg,label,meta['serial_refresh_seconds'],flush=True)
    return True
