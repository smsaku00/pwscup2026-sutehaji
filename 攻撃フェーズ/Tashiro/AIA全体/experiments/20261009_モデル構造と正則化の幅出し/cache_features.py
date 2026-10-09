#!/usr/bin/env python3
"""Rebuild public-Q/C/BG features locally; official truths are read only for scoring later."""
import argparse,hashlib,json,time,sys
from pathlib import Path
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'prior'))
import experiment as prior

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
 root=a.root.resolve();manifest=json.loads((HERE/'official_manifest.json').read_text())
 assert len(manifest)==65
 for p,digest in manifest.items():
  candidate=root/p
  if not candidate.exists():raise FileNotFoundError(candidate)
  if hashlib.sha256(candidate.read_bytes()).hexdigest()!=digest:raise ValueError('SHA256 mismatch: '+p)
 print('VERIFIED_OFFICIAL_INPUTS',len(manifest),flush=True)
 start=time.time();bg,D=prior.read_data(root);features,truth=prior.feature_table(bg,D)
 out=HERE/'cache';out.mkdir(parents=True,exist_ok=True)
 for k in prior.TEAMS:
  features[k].to_pickle(out/f'features_{k}.pkl')
  truth[k].to_pickle(out/f'truth_{k}.pkl')
  D[k][0][['challenge_row_id','age','sex','prefecture']].to_pickle(out/f'q_{k}.pkl')
 print('CACHED',len(features),'COHORTS',round(time.time()-start,3),'SEC',flush=True)
if __name__=='__main__':main()
