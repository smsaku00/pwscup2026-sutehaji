"""Replay exact original full-background baseline, without changing source files."""
from pathlib import Path
import argparse, importlib.util, json

HERE=Path(__file__).resolve().parent

def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    checks=[]
    for family in ('general','tapas'):
        spec=importlib.util.spec_from_file_location(family,HERE/'original'/f'{family}.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.BGS=('full',)  # Select original full condition; no parameters changed.
        module.main(root,out/f'baseline_{family}')
        historical=json.loads((HERE/'original'/f'historical_{family}_prediction_hashes.json').read_text())
        expected={x['method']:x['sha256'] for x in historical if x['background']=='full'}
        actual=json.loads((out/f'baseline_{family}'/'prediction_hashes.json').read_text())
        for item in actual:
            checks.append({'family':family,'method':item['method'],
                           'exact_prediction_hash_matches':item['sha256']==expected[item['method']]})
    (out/'baseline_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert all(x['exact_prediction_hash_matches'] for x in checks),checks

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();main(args.root,args.out)
