#!/usr/bin/env python3
"""Reproduce the earlier public-scores baseline using its own frozen source, no target labels in target models."""
from __future__ import annotations
import argparse,importlib.util,sys
from pathlib import Path
import pandas as pd

def main(root,prior,out):
    spec=importlib.util.spec_from_file_location('existing_aia_prior',prior/'experiment.py')
    if spec is None or spec.loader is None:raise FileNotFoundError(prior/'experiment.py')
    lib=importlib.util.module_from_spec(spec);sys.modules[spec.name]=lib;spec.loader.exec_module(lib)
    bg,D=lib.read_data(root)
    feats,truths=lib.feature_table(bg,D)
    probs,_=lib.modeling(feats,truths)
    frames=[]
    for k,(q,_,_) in D.items():
        f=feats[k]
        z=pd.DataFrame({'team':k,'row_id':q.challenge_row_id,'sex':q.sex,'prefecture':q.prefecture,'age':q.age,
                        'near4':f['member_near4'],'col4':f['col4'],'col2':f['col2']})
        for model,p in probs[k].items():z[model]=p
        frames.append(z)
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.concat(frames,ignore_index=True).to_csv(out,index=False)
    print('Prepared prior public scores',out,'n=',sum(map(len,frames)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();main(args.root,args.prior,args.out)
