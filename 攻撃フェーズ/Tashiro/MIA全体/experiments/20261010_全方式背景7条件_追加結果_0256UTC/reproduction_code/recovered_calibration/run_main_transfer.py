"""Run the original prelim18 → main22 transfer, with rank-only reporting."""
from pathlib import Path
import argparse,ast,json,time,shutil,sys
import numpy as np
import runtime as r
from run_family import configure,STAGE,FOLDERS,dump
from generate_features import CACHE

BASELINE=Path('/workspace/shared/mia_recovered_validation/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/private_inputs/main_round3_F_mia.csv')
BASELINE_SHA='6ca9604a8936fb47255ae0bd05c9f3958368f0e196bc2af258f8255cf39c08eb'

def run(a):
    a.family='learned_main_transfer';u=r.initialize();work=configure(u,a)
    source=r.SRC/'本戦で順位を確認.py';m=r.import_path('original_main_transfer',source)
    p=CACHE/a.bg/'main/代表特徴.npz';assert p.exists(),p
    z=np.load(p)
    seal=json.loads(p.with_name('record.json').read_text());assert seal['score_sha256']==r.matsha(z['scores']) and seal['background_sha256']==r.MANIFEST['subsets'][a.bg]['sha256']
    from execution_protocol import PROTOCOL
    assert seal['forest_prediction_execution']==PROTOCOL
    def calculate(data,aa,label):
        assert data['phase']=='main' and list(z['teams'])==data['teams']
        assert len(aa)==r.MANIFEST['subsets'][a.bg]['rows'] if 'rows' in r.MANIFEST['subsets'][a.bg] else True
        return z['scores'].copy()
    m.calculate=calculate
    assert r.sha(BASELINE)==BASELINE_SHA
    original=u.base.read
    def read(path):
        path=Path(path)
        return original(BASELINE if path.name=='F_mia.csv' and '第3回MIA' in str(path) else path)
    u.base.read=read
    # Only the original main BG is asserted identical to its historical F_mia.
    # Other BG controls are newly recomputed and their rank drift is reported.
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    for i,n in enumerate(fn.body):
        if isinstance(n,ast.Assert) and ast.unparse(n.test)=='err < 1e-12':
            if a.bg!='main':fn.body[i]=ast.Pass()
    ast.fix_missing_locations(fn);exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),m.__dict__)
    sys.argv=[str(source),"--output-dir",str(work/FOLDERS[a.family])]
    m.main()
    out=work/FOLDERS[a.family];dest=STAGE/a.family/a.bg;dest.mkdir(parents=True,exist_ok=True)
    for path in out.iterdir():
        if path.suffix in ('.csv','.json'):shutil.copy2(path,dest/path.name)
    cache=np.load(out/'内部特徴と順位.npz');names=list(cache['names']);assert len(names)==10
    hashes=[dict(method=name,shape=list(mat.shape),sha256=r.matsha(mat),phase='main',diagnostic_only=True) for name,mat in zip(names,cache['scores'])]
    dump(dest/'prediction_hashes.json',hashes)
    record=json.loads((dest/'実行記録.json').read_text());record.update(phase='main',background=a.bg,models=10,forest_prediction_execution=PROTOCOL,source_sha256=r.sha(source),main_truth_used=False,rank_only=True,original_historical_control_assertion_applied=a.bg=='main',historical_control_comparison='Old main BG baseline; drift is expected for a different BG',training_protocol_modified=False,candidate_cache_in_report_tree=False)
    dump(dest/'実行記録.json',record);dump(dest/'completion.json',dict(complete=True,models=10,source_main_returned=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bg',choices=r.BG,required=True);run(p.parse_args())
