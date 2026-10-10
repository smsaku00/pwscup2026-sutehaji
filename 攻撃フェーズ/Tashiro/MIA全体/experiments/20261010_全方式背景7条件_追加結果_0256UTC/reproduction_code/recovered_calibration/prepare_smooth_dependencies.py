"""Verify and select exact source-grid channels for original smooth postprocessing."""
from pathlib import Path
import argparse,json
import numpy as np
import runtime as r
from run_family import WORK,dump

DATA=Path('/workspace/shared/mia_all_bg_fusion/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_oct8')
POLICY=Path('/workspace/shared/mia_all_bg_policy/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/policy/cache/early_lof')

def run(bg):
    u=r.initialize();data=r.data_for(u,bg);m=r.import_path('smooth_definitions',r.SRC/'他対象との競合を滑らかに補正.py');sources={};provenance=[]
    for folder,name in m.SOURCES:
        if folder=='距離と分布結果':p=DATA/'geometry/prelim'/bg/'scores.npz'
        elif folder=='混合分布結果':p=DATA/'mixture/prelim'/bg/'scores.npz'
        elif folder=='局所密度結果':p=DATA/'local/prelim'/bg/'scores.npz'
        else:p=WORK/bg/'learned_runtime'/folder/'内部順位キャッシュ.npz'
        assert p.exists(),p;z=np.load(p)
        if folder in ('距離と分布結果','混合分布結果','局所密度結果'):
            rec=json.loads(p.with_name('prediction_record.json').read_text());assert r.sha(p)==rec['scores_sha256'];names=list(z['names']);mat=z['scores'][names.index(name)];teams=list(z['teams'])
        elif folder=='局所密度結果':
            rec=json.loads(p.with_name('prediction_record.json').read_text());assert r.sha(p)==rec['file_sha256'];mat=z['x0'];teams=rec['teams'];assert name in rec['methods']
        else:names=list(z['names']);mat=z['scores'][names.index(name)];teams=list(z['teams'])
        mat=mat[:,[teams.index(t) for t in data['teams']]];sources.setdefault(folder,{})[name]=mat
        provenance.append(dict(source=str(p),sha256=r.sha(p),method=name,matrix_sha256=r.matsha(mat)))
    for folder,models in sources.items():
        out=WORK/bg/folder/'内部順位キャッシュ.npz';out.parent.mkdir(exist_ok=True,parents=True)
        # This is a selected-channel smooth-only cache. Full source outputs live
        # in their original worker/private directories and are never overwritten.
        if out.exists() and folder not in ('距離と分布結果','混合分布結果','局所密度結果'):
            old=np.load(out);assert all(name in list(old['names']) for name in models);continue
        names=sorted(models);np.savez_compressed(out,names=names,scores=np.stack([models[name] for name in names]),teams=data['teams'],record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
    dump(Path('/workspace/shared/mia_recovered_calibration/cache/source_features')/bg/'smooth_dependencies.json',provenance)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bg',choices=r.BG,required=True);a=p.parse_args();run(a.bg)
