"""Read-only recovered-source bridge to canonical phase inputs and BG subsets."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
import pandas as pd

RECOVERY=Path('/workspace/shared/mia_recovered_sources/攻撃フェーズ/Tashiro/source_recovery/20261010_MIA元コード回収')
SRC=RECOVERY/'originals/本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索'
INPUT=Path('/workspace/shared/mia_bg7_inputs')
MANIFEST=json.loads((INPUT/'manifest.json').read_text())
ROOT=Path('/workspace/shared/mia_recovered_runtime_root')
BG=tuple(MANIFEST['subsets'])

def import_path(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def initialize():
    from execution_protocol import install
    install()
    sys.path.insert(0,str(SRC))
    u=import_path('共通',SRC/'共通.py')
    for folder,target in [(ROOT/'予備戦/配布データ',INPUT/'root/公式データ/予備戦'),(ROOT/'本戦/配布データ',INPUT/'root/公式データ'),(ROOT/'本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力',INPUT/'root/公式データ/PWSCup2026_final_attack')]:
        folder.parent.mkdir(parents=True,exist_ok=True)
        if not folder.exists():folder.symlink_to(target,target_is_directory=True)
    u.ROOT=ROOT
    u._setup_original=u.setup
    return u

def data_for(u,bg,phase='prelim'):
    original=u.base.phase_inputs
    def inputs(root,requested_phase):
        b,a,q,private,rare,cs,targets,self_id=original(ROOT,requested_phase)
        a=pd.read_csv(MANIFEST['subsets'][bg]['path'],float_precision='round_trip')
        return b,a,q,private,rare,cs,targets,self_id
    u.base.phase_inputs=inputs
    try:data=u._setup_original(phase)
    finally:u.base.phase_inputs=original
    return data

def raw_cache(data,bg):
    h=Path('/workspace/shared/mia_all_bg_available/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/available')/f"results_{data['phase']}"
    p=h/f'raw_cache_{bg}.npz';meta=json.loads(p.with_suffix('.json').read_text())
    assert meta['background_sha256']==MANIFEST['subsets'][bg]['sha256']
    z=np.load(p);numeric=sorted(data['teams'])
    raw=z['raw'][:,[numeric.index(t) for t in data['teams']]]
    assert raw.shape==(len(data['query']),len(data['teams']))
    return raw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def matsha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<f8').tobytes()).hexdigest()
