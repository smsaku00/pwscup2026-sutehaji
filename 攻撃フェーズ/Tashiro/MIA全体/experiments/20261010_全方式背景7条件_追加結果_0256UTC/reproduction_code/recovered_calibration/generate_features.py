"""Generate the original frozen representative feature recipes, without labels."""
from pathlib import Path
import argparse,json,time,sys
import numpy as np
import pandas as pd
import scipy,sklearn
import resource
import runtime as r
from run_family import dump,WORK
from execution_protocol import PROTOCOL

CACHE=Path('/workspace/shared/mia_recovered_calibration/cache/representative')
VIEWCACHE=Path('/workspace/shared/mia_recovered_calibration/cache/multiview')
CONDITIONS=[('full',1.,None)]+[(f'fraction{f:g}_seed{s}',f,s) for f in (.9,.8,.5) for s in (20261007,20261008)]
CONDITIONS += [(f'fraction{f:g}_seed{s}',f,s) for f in (.95,.9,.8,.7,.5) for s in range(20261009,20261017)]
assert len(CONDITIONS)==47

def run(bg,views=False,phase='prelim',limit=None):
    u=r.initialize();u.HERE=r.SRC;data=r.data_for(u,bg,phase)
    source=r.import_path('representative_generation',r.SRC/'代表特徴を計算.py')
    from bounded_features import install
    install(source,r.SRC/'代表特徴を計算.py')
    from bounded_core import install as install_core
    install_core(u.base)
    common={'background':bg,'background_sha256':r.MANIFEST['subsets'][bg]['sha256'],'source_sha256':r.sha(r.SRC/'代表特徴を計算.py'),'teams':data['teams'],'phase':phase,'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'age_cuts':data['age_cuts'].tolist(),'truth_read':False,'forest_prediction_execution':PROTOCOL,'memory_adapter':'Row-independent BG KNN and KDE reference calculations in blocks256; full-query category frequencies unchanged; Gen-LRA C-row blocks128 with full reference-axis reduction and final stable sort','memory_adapter_sha256':r.sha(Path('/workspace/shared/mia_recovered_calibration/bounded_features.py')),'core_memory_adapter_sha256':r.sha(Path('/workspace/shared/mia_recovered_calibration/bounded_core.py'))}
    for label,fraction,seed in (CONDITIONS if views else CONDITIONS[:1])[:limit]:
        out=(CACHE/bg if label=='full' else VIEWCACHE/bg/label)
        if phase!='prelim':out=out/phase
        out.mkdir(parents=True,exist_ok=True);p=out/'代表特徴.npz';meta=out/'record.json'
        if p.exists() and not meta.exists():
            p.replace(out/f'uncommitted_{int(time.time())}.npz')
        if p.exists():
            from deterministic_features import ensure
            ensure(u,bg,phase,label,fraction,seed)
            z=np.load(p);assert z['scores'].shape==(875,len(data['teams']),14) and list(z['teams'])==data['teams']
            if meta.exists():
                existing=json.loads(meta.read_text())
                for key in ['background_sha256','source_sha256','phase','sklearn']:assert existing[key]==common[key],(p,key)
            assert r.matsha(z['scores'])==existing['score_sha256']
            print('CACHED',bg,label,flush=True);continue
        aa=data['a'] if label=='full' else u.cal.stratified_subset(data['a'],fraction,seed)
        start=time.perf_counter();raw=source.calculate(data,aa,f'{bg}/{label}')
        temp=out/'temporary.npz';np.savez_compressed(temp,scores=raw,teams=data['teams'],record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy());temp.replace(p)
        record={**common,'condition':label,'fraction':fraction,'intrinsic_seed':seed,'rows':len(aa),'score_sha256':r.matsha(raw),'file_sha256':r.sha(p),'seconds':time.perf_counter()-start,'process_peak_rss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),'internal_age_bins':'Original augmentation uses outer BG-fitted age cuts; unchanged'}
        commit=out/'record.pending.json';dump(commit,record);commit.replace(meta);print('FEATURES_SAVED',bg,label,record['seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bg',choices=r.BG,required=True);p.add_argument('--views',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--phase',choices=['prelim','main'],default='prelim');a=p.parse_args()
    if a.bg!='prelim' and (VIEWCACHE/'prelim'/CONDITIONS[2][0]/'代表特徴.npz').exists() and not Path('/workspace/shared/mia_recovered_calibration/multiview_cost_probe.json').exists():
        from benchmark_multiview import main
        main()
    run(a.bg,a.views,a.phase,a.limit)
    if a.bg=='prelim' and a.views and a.limit==3:
        from benchmark_multiview import main
        main()
