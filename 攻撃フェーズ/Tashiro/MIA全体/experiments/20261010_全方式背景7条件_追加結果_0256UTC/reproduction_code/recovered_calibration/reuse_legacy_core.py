"""Read-only reuse of byte-identical Oct7 core outputs for original contexts."""
from pathlib import Path
import inspect,json
import numpy as np
from joblib import Memory,load
import runtime as r

LEGACY=Path('/workspace/shared/mia_all_bg_legacy/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/legacy')

def install(u,record_path,bg):
    assert r.sha(u.OLD/'分析.py')==r.sha(LEGACY/'source_oct07/分析.py')
    original=u.base.core;memory=Memory(Path('/workspace/shared/mia_recovered_calibration/cache/legacy_read_only_hashing'),verbose=0);cached=memory.cache(original)
    function_dir=LEGACY/'cache/分析/core'
    stored=(function_dir/'func_code.py').read_text().split('\n',1)[1]
    assert stored==inspect.getsource(original)
    fresh_a=u.base.read(Path(r.MANIFEST['subsets'][bg]['path']))
    fresh_b,*_=u.base.phase_inputs(r.ROOT,'prelim')
    records=[];verified=set()
    def original_frame(frame,fresh):
        clean=frame.drop(columns=['age_bin'],errors='ignore')
        contexts=[fresh,fresh.iloc[::2],fresh.iloc[1::2]] if fresh is fresh_a else [fresh]
        result=next((v for v in contexts if clean.index.equals(v.index)),None)
        if result is None:result=fresh.loc[clean.index]
        assert list(result.columns)==list(clean.columns) and result.equals(clean)
        return result
    def core(a,b,c,cols,ks=(1,3,10),whiten=False):
        aa=original_frame(a,fresh_a);bb=original_frame(b,fresh_b);cc=c
        canonical=tuple(sorted({1,3,10}|set(ks)));args=(aa,bb,cc,tuple(cols),canonical,whiten)
        key=cached._get_args_id(*args);path=function_dir/key/'output.pkl'
        if path.exists() and (path.parent/'metadata.json').exists():
            values=load(path)
            assert set(canonical).issubset(values) and all(np.isfinite(values[k]).all() and len(values[k])==len(b) for k in canonical)
            # Verify the first hit for each numeric/whitening/row-size shape
            # against the untouched current-runtime core before reusing it.
            check=(tuple(cols),whiten,len(a),len(b),len(c))
            if check not in verified:
                fresh=original(a,b,c,cols,ks,whiten)
                assert all(np.array_equal(fresh[k],values[k]) for k in ks),(path,check)
                verified.add(check)
            records.append(dict(cache_key=key,file_sha256=r.sha(path),numeric_columns=list(cols),ks=list(ks),whiten=whiten,reference_rows=len(a),query_rows=len(b),synthetic_rows=len(c),cache_hit=True))
            out={k:values[k] for k in ks}
        else:
            out=original(a,b,c,cols,ks,whiten)
            records.append(dict(cache_key=key,numeric_columns=list(cols),ks=list(ks),whiten=whiten,reference_rows=len(a),query_rows=len(b),synthetic_rows=len(c),cache_hit=False))
        record_path.parent.mkdir(parents=True,exist_ok=True);record_path.write_text(json.dumps({'read_only_legacy_cache':True,'records':records,'verified_raw_shape_classes':len(verified)},ensure_ascii=False,indent=2)+'\n')
        return out
    u.base.core=core
