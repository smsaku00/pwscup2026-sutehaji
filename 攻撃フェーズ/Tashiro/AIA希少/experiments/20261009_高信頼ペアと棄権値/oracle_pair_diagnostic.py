#!/usr/bin/env python3
"""MOCK ONLY: answer the counterfactual 'What if the true pairs were known?'.

The pairing ID is hidden information, deliberately used ONLY in this diagnostic.
Orientation never sees member/control labels. Truth is used only after prediction.
This is not an implementable contest attack, or an independent holdout experiment.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import experiment as ex


def run(work: Path, out: Path) -> None:
    out.mkdir(parents=True,exist_ok=True)
    if (out/'oracle_manifest.json').exists():raise FileExistsError('Completed run exists')
    ex.dump(out/'oracle_plan.json',dict(kind='Post-hoc mechanism diagnostic',
        actual_pair_id_used=True,membership_used_to_orient=False,gap=ex.HIGH_GAP,
        modes=['true_pair_all','true_pair_gated'],miss_values=[0.,11.],
        assumptions='Only archived mock pairing is known. No official pair IDs are available.'))
    cases,bg,_=ex.mock_inputs(work)
    rows=[];pair_rows=[]
    for case in cases:
        q=case.q[['challenge_row_id']+ex.QI].copy()
        scores=ex.evidence(q,case.c,bg);ranks=ex.within_sex_ranks(q,scores)
        # One-to-one alignment checked explicitly; no source record IDs copied.
        mapping=case.pairmap.set_index('challenge_row_id').loc[q.challenge_row_id]
        truth=case.truth.set_index('challenge_row_id').loc[q.challenge_row_id].reset_index()
        for meth in ex.METHODS:
            base=ex.baseline_mask(q,scores[meth]);allmask=base.copy();gatedmask=base.copy()
            for pair_id in sorted(mapping.pair_id.unique()):
                ind=np.flatnonzero(mapping.pair_id.to_numpy()==pair_id)
                if len(ind)!=2:raise ValueError('Mock pair is not size 2')
                a,b=map(int,ind)
                diff=float(ranks[meth][a]-ranks[meth][b]);direction=np.sign(diff)
                hi,lo=(a,b) if diff>=0 else (b,a)
                agree=bool(direction!=0 and
                    np.sign(ranks['near4'][a]-ranks['near4'][b])==direction and
                    np.sign(ranks['col4_simple'][a]-ranks['col4_simple'][b])==direction)
                accept=bool(abs(diff)>=ex.HIGH_GAP and agree)
                allmask[hi]=True;allmask[lo]=False
                if accept:gatedmask[hi]=True;gatedmask[lo]=False
                # Evaluation side, after orientation. Never changes masks.
                iscorrect=bool(truth.label.iloc[hi]=='member')
                if truth.label.iloc[a]==truth.label.iloc[b]:raise ValueError('Broken oracle pair map')
                pair_rows.append(dict(dataset=case.name,panel=case.panel,control_case=case.control_case,
                    method=meth,rank_gap=abs(diff),accepted=accept,correct_orientation=iscorrect))
            for policy,mask in [('baseline',base),('true_pair_all',allmask),('true_pair_gated',gatedmask)]:
                for miss in [0.,11.]:
                    p=np.where(mask,ex.HIT_TIME,miss);bp=np.where(base,ex.HIT_TIME,miss)
                    sc=ex.score_prediction(p,truth);bsc=ex.score_prediction(bp,truth)
                    oldhit=np.abs(bp-truth.time.to_numpy())<=.5;newhit=np.abs(p-truth.time.to_numpy())<=.5
                    m=truth.label.eq('member').to_numpy()
                    rows.append(dict(dataset=case.name,panel=case.panel,control_case=case.control_case,
                        method=meth,policy=policy,miss_value=miss,changed_rows=int((p!=bp).sum()),
                        delta_r_all=sc['r_all']-bsc['r_all'],
                        member_hit_gain=int((m&~oldhit&newhit).sum()),member_hit_loss=int((m&oldhit&~newhit).sum()),
                        control_hit_gain=int((~m&~oldhit&newhit).sum()),control_hit_loss=int((~m&oldhit&~newhit).sum()),
                        **sc))
    tab=pd.DataFrame(rows);pa=pd.DataFrame(pair_rows)
    tab.to_csv(out/'oracle_case_metrics.csv',index=False,float_format='%.17g')
    parts=[];orientation=[]
    for name,sel in [('mock3C_5panels',~tab.control_case),('CONTROL_identity',tab.dataset.eq('CONTROL_identity')),
                     ('CONTROL_background_only',tab.dataset.eq('CONTROL_background_only'))]:
        agg=tab[sel].groupby(['method','policy','miss_value']).agg(
            r_all=('r_all','mean'),delta_r_all=('delta_r_all','mean'),
            changed_rows=('changed_rows','mean'),hit_m=('hit_m','mean'),hit_c=('hit_c','mean'),
            improves=('delta_r_all',lambda x:int((x>1e-12).sum())),
            worsens=('delta_r_all',lambda x:int((x< -1e-12).sum()))).reset_index()
        agg.insert(0,'evaluation_set',name);parts.append(agg)
        subset=pa[~pa.control_case] if name=='mock3C_5panels' else pa[pa.dataset.eq(name)]
        for m in ex.METHODS:
            p=subset[subset.method.eq(m)];a=p[p.accepted]
            orientation.append(dict(evaluation_set=name,method=m,total_comparisons=len(p),
                correct_all=int(p.correct_orientation.sum()),accepted=len(a),correct_accepted=int(a.correct_orientation.sum()),
                orientation_accuracy=float(p.correct_orientation.mean()),
                accepted_accuracy=None if not len(a) else float(a.correct_orientation.mean())))
    pd.concat(parts).to_csv(out/'oracle_summary.csv',index=False,float_format='%.17g')
    pd.DataFrame(orientation).to_csv(out/'oracle_orientation.csv',index=False,float_format='%.17g')
    ex.dump(out/'oracle_manifest.json',dict(mock_only=True,official_experiment_run=False,
        hidden_pair_id_used=True,membership_used_to_predict=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=ex.HASHES,
        caveat='Post-hoc mechanism diagnostic, not a deployable policy. No Rrare due to unknown control rare labels.'))
    print(pd.concat(parts).query("evaluation_set=='mock3C_5panels'").to_string(index=False))
    print(pd.DataFrame(orientation).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.out.resolve())
