"""Optional 07-d01 experiment. The holdout found additional membership leakage.
Not recommended for submission; generates the experimental ZIP for reproducibility.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from decoy import decoy_swaps
from reproduce import pack
from generate_three import csv_bytes
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'official_scoring/src'))
from pwscup2026.kit.selfscore import load_local_reference,self_score
from pwscup2026.kit.validate import validate_submission_zip
from pwscup2026.scoring.result import json_safe
from pwscup2026.attack.aia import aia_predict

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 for key in ['dist','c','out']:ap.add_argument('--'+key,type=Path,required=True)
 args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 b,ref,cfg,_=load_local_reference(args.dist);a=ref.a_bg
 c=pd.read_csv(args.c,float_precision='round_trip');d,meta=decoy_swaps(c,b,a)
 cfg['aia'].update({'qi_continuous':['age','SBP','TG','HDL','ALT'],'block_keys':['sex','prefecture'],'k_nn_onset':5})
 schema=json.loads((args.dist/'schema.json').read_text());_,h0=aia_predict(c,b[schema['aia_qi_columns']],a,cfg);_,h1=aia_predict(d,b[schema['aia_qi_columns']],a,cfg)
 hit0=abs(h0-b.time.to_numpy())<=.5;hit1=abs(h1-b.time.to_numpy())<=.5
 meta.update({'official_training_hits_before':int(hit0.sum()),'official_training_hits_after':int(hit1.sum()),'example_predictions':[{'true_time':float(b.time.iloc[i]),'before_time':float(h0[i]),'after_time':float(h1[i])} for i in np.flatnonzero(hit0&~hit1)[:3]],'status':'Experimental only; holdout revealed membership leakage from predicted time.'})
 data=csv_bytes(d);zp=args.out/'zhang_07_d01_prefecture_decoy.zip';pack(zp,data,(args.dist/'token.txt').read_bytes())
 valid=validate_submission_zip(zp,cfg,args.dist)
 if not valid.ok:raise ValueError(str(valid.errors))
 loaded=pd.read_csv(__import__('io').BytesIO(data),float_precision='round_trip');score=self_score(loaded,b,ref,cfg)
 meta.update({'U':score['utility'],'facets':score['facets'],'zip_sha256':hashlib.sha256(zp.read_bytes()).hexdigest(),'validated':True})
 (args.out/'decoy_experiment.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));(args.out/'decoy_utility.json').write_text(json.dumps(json_safe(score),ensure_ascii=False,indent=2))
 print(json.dumps(meta,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
