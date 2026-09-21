"""本番team28の配布ZIPから、3案のC・提出ZIP・公式ローカル有用性を再現。
python code/reproduce.py --team-zip /path/to/final_team28.zip --out reproduced
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse, hashlib, json, re, sys, zipfile
from pathlib import Path
import pandas as pd
from generate_three import METHODS, SEEDS, generate, csv_bytes

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'official_scoring/src'))
from pwscup2026.kit.selfscore import load_local_reference, self_score
from pwscup2026.kit.validate import validate_submission_zip
from pwscup2026.scoring.result import json_safe

def pack(path,c_bytes,token):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in [('C.csv',c_bytes),('token.txt',token)]:
            info=zipfile.ZipInfo(name,date_time=(2026,9,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--team-zip',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    dist=out/'private_input';dist.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.team_zip) as z:
        readme=z.read('README.txt').decode('utf-8')
        match=re.search(r'コホート番号\s*[:：]\s*(\d+)',readme)
        if '本戦' not in readme or match is None or int(match.group(1))!=28:
            raise ValueError('本番team28の配布ZIPを指定してください')
        for name in ['B.csv','B_self.csv','utility_ref.json','A_bg.csv','schema.json','token.txt','README.txt']:
            (dist/name).write_bytes(z.read(name))
    (dist/'scoring_config.yaml').write_bytes((ROOT/'official_scoring/scoring_config.yaml').read_bytes())
    schema=json.loads((dist/'schema.json').read_text())
    b=pd.read_csv(dist/'B.csv',float_precision='round_trip')
    bref,ref,cfg,info=load_local_reference(dist)
    results=[];manifest={'round':'final','cohort':28,'kit':'main-process-20260912',
        'B_sha256':hashlib.sha256((dist/'B.csv').read_bytes()).hexdigest(), 'files':{}}
    for method in METHODS:
        folder=out/method;folder.mkdir(exist_ok=True)
        c,gen_meta=generate(b,ref.a_bg,schema,method);data=csv_bytes(c);(folder/'C.csv').write_bytes(data)
        zp=out/(method+'.zip');pack(zp,data,(dist/'token.txt').read_bytes())
        valid=validate_submission_zip(zp,cfg,dist)
        if not valid.ok:raise ValueError(str(valid.errors))
        loaded=pd.read_csv(folder/'C.csv',float_precision='round_trip')
        score=self_score(loaded,bref,ref,cfg)
        (folder/'utility.json').write_text(json.dumps(json_safe(score),ensure_ascii=False,indent=2,allow_nan=False))
        row={'method':method,'seed':SEEDS[method],'U':score['utility'],**score['facets'],'validated':True}
        results.append(row);print(json.dumps(row),flush=True)
        manifest['files'][method]={'seed':SEEDS[method],'settings':gen_meta,'C_sha256':hashlib.sha256(data).hexdigest(),'zip_sha256':hashlib.sha256(zp.read_bytes()).hexdigest()}
    pd.DataFrame(results).to_csv(out/'utility_summary.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
