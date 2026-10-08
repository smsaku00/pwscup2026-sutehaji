#!/usr/bin/env python3
"""Summarize saved results without changing predictions or fitting new models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import experiment as ex


def run(work: Path, results: Path, out: Path) -> None:
    out.mkdir(parents=True,exist_ok=True)
    tbl=pd.read_csv(results/'case_metrics.csv'); pairs=pd.read_csv(results/'pair_audit.csv')
    contrast=pd.read_csv(results/'miss_value_contrast.csv')
    real=tbl[~tbl.control_case & tbl.width.eq(10)]
    outrows=[]
    for method in ex.METHODS:
        s=contrast[~contrast.control_case & contrast.width.eq(10)&contrast.method.eq(method)&contrast.policy.eq('baseline')]
        outrows.append(dict(method=method,cases=len(s),mean_delta_r_all=float(s.eleven_minus_zero_r_all.mean()),
            lost_member_hits=-int(s.eleven_minus_zero_hit_m.sum()),
            lost_control_hits=-int(s.eleven_minus_zero_hit_c.sum()),
            improved_cases=int((s.eleven_minus_zero_r_all>1e-12).sum()),
            worsened_cases=int((s.eleven_minus_zero_r_all< -1e-12).sum())))
    pd.DataFrame(outrows).to_csv(out/'baseline_0_to_11.csv',index=False,float_format='%.17g')
    realpairs=pairs[~pairs.dataset.str.startswith('CONTROL_')]
    coverage=[]
    for width in [5,10]:
        for method in ex.METHODS:
            p=realpairs[realpairs.width.eq(width)&realpairs.method.eq(method)]
            coverage.append(dict(width=width,method=method,candidates=len(p),passes=int(p.passed.sum()),
                one_member_one_control=int(p.one_member_one_control.sum()),
                true_pairs=int(p.true_pair_match.sum()),
                pass_with_mixed_roles=int((p.passed&p.one_member_one_control).sum())))
    pd.DataFrame(coverage).to_csv(out/'pair_coverage.csv',index=False)
    d=pd.read_csv(results/'changed_rows_diagnostics.csv')
    d=d[~d.dataset.str.startswith('CONTROL_') & d.policy.eq('combined_pair') & d.miss_value.eq(0)]
    # No raw clinical values or true time copied to the distributable audit.
    d.drop(columns=['true_time']).to_csv(out/'pair_changes_without_time.csv',index=False)
    members=[];controls=[];rarehits=[]
    cases,bg,meta=ex.mock_inputs(work)
    bypanel={c.panel:c for c in cases}
    for panel,c in bypanel.items():
        mapping=c.pairmap.set_index('challenge_row_id')
        t=c.truth.set_index('challenge_row_id');q=c.q.set_index('challenge_row_id')
        merged=mapping.join(t[['label','time','member_true_rare_control_unknown']])
        members.extend(merged[merged.label.eq('member')].source_row_index.astype(int).tolist())
        controls.extend(merged[merged.label.eq('control')].source_row_index.astype(int).tolist())
        for _,g in merged.groupby('pair_id'):
            idx=g.index
            assert len(g)==2 and set(g.label)=={'member','control'}
            assert q.loc[idx].prefecture.nunique()==1
            assert q.loc[idx].sex.nunique()==1
            assert (q.loc[idx].age//10).nunique()==1
            assert c.truth.set_index('challenge_row_id').loc[idx].onset.nunique()==1
        prob=meta['probabilities'][(c.name,panel)]
        chosen=ex.top_indices(prob,c.q.challenge_row_id.to_numpy(int),26)
        tr=c.truth.iloc[chosen]
        h=int(tr.member_true_rare_control_unknown.eq(1).sum())
        rarehits.append(dict(panel=panel,top26_known_rare_member_hits=h,
            known_rare_member_recall=h/13,control_rare_unknown=True,
            probability_at26=float(sorted(prob,reverse=True)[25])))
    pd.DataFrame(rarehits).to_csv(out/'mock_detector_member_only.csv',index=False,float_format='%.17g')
    ex.dump(out/'dataset_audit.json',dict(mock_panels=5,members_per_panel=163,controls_per_panel=163,
        total_member_appearances=len(members),unique_member_source_rows=len(set(members)),
        total_control_appearances=len(controls),unique_control_source_rows=len(set(controls)),
        mock_pairs_checked=815,known_key='prefecture, sex, floor(age/10), onset',
        control_true_is_rare_known=False,all_primary_candidate_pairs=len(realpairs[realpairs.width.eq(10)]),
        note='Source-row uniqueness counts separately within main B and preliminary BG, not across data worlds.'))
    print(pd.DataFrame(outrows).to_string(index=False)); print(pd.DataFrame(coverage).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.results.resolve(),a.out.resolve())
