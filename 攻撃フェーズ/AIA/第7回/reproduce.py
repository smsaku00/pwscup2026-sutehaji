#!/usr/bin/env python3
"""Create round7_2 from frozen AIA answers, then compare non-group rows to rounds1..6."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
KEY=['target_k','challenge_row_id']
BASE_HASH='59ed6173457ef20e643e20d6becd322b02e0a9a84b22d3463c613406bd89acb5'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def frame(path):
    d=pd.read_csv(path,float_precision='round_trip').sort_values(KEY).reset_index(drop=True)
    assert list(d.columns)==KEY+['time_hat'] and len(d)==7172
    assert not d.duplicated(KEY).any() and d.groupby('target_k').size().eq(326).all()
    assert np.isfinite(d.time_hat).all()
    return d

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attack-dir',type=Path,required=True,help='Official public main attack directory, for validation')
    p.add_argument('--out',type=Path,default=HERE/'results')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    src=HERE/'inputs';base=frame(src/'base_round7_F_aia.csv')
    assert sha(src/'base_round7_F_aia.csv')==BASE_HASH
    previous={i:frame(src/f'round{i}_F_aia.csv') for i in range(1,7)}
    for d in previous.values():assert d[KEY].equals(base[KEY])
    groups=pd.read_csv(src/'group_assignments.csv');assert len(groups)==181 and not groups.duplicated(KEY).any()
    assert groups.groupby(['target_k','group_id']).ngroups==57
    gkeys=set(map(tuple,groups[KEY].to_numpy()))
    mask=np.array([tuple(k) in gkeys for k in base[KEY].to_numpy()])
    assert mask.sum()==181
    for (k,gid),g in groups.groupby(['target_k','group_id']):
        q=pd.read_csv(a.attack_dir/f'aia_challenge_{k}.csv');q['age_band']=(q.age//5).astype(int)
        rows=q[q.challenge_row_id.isin(g.challenge_row_id)]
        assert len(rows)==len(g) and len(g) in [2,3,4]
        assert len(rows[['prefecture','sex','age_band']].drop_duplicates())==1
        key=rows[['prefecture','sex','age_band']].iloc[0]
        assert set(q.loc[(q[['prefecture','sex','age_band']]==key).all(axis=1),'challenge_row_id'])==set(g.challenge_row_id)
    scores={i:json.loads((src/f'round{i}_aia_scores.json').read_text())['aia_per_target'] for i in [5,6]}
    new=base.copy();choices=[]
    for k in sorted(base.target_k.unique()):
        s5=float(scores[5][str(k)]);s6=float(scores[6][str(k)])
        chosen=5 if s5>s6 else 6 if s6>s5 else 7
        team=base.target_k.eq(k).to_numpy();inside=team&mask
        if chosen!=7:new.loc[inside,'time_hat']=previous[chosen].loc[inside,'time_hat']
        assert np.array_equal(new.time_hat[inside],(previous[chosen] if chosen!=7 else base).time_hat[inside])
        choices.append({'target_k':int(k),'score5':s5,'score6':s6,'group_source':chosen,
            'group_rows':int(inside.sum()),'changed_from_base7':int(new.time_hat[inside].ne(base.time_hat[inside]).sum())})
    assert np.array_equal(new.time_hat[~mask],base.time_hat[~mask])
    assert new.time_hat.isin([9.5,11]).all()
    raw=new.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()
    (a.out/'F_aia.csv').write_bytes(raw)
    pd.DataFrame(choices).to_csv(a.out/'group_selection.csv',index=False)
    comparison=[]
    for k in sorted(base.target_k.unique()):
        outside=(base.target_k.eq(k).to_numpy())&~mask
        for i,old in previous.items():
            nv=new.time_hat[outside].to_numpy();ov=old.time_hat[outside].to_numpy()
            same=nv==ov;selection=(nv==9.5)==(ov==9.5)
            comparison.append({'target_k':int(k),'previous_round':i,'non_group_rows':len(nv),
                'numeric_same':int(same.sum()),'numeric_changed':int((~same).sum()),
                'selection95_same':int(selection.sum()),'selection95_changed':int((~selection).sum()),
                'numeric_identical':bool(same.all()),'selection95_identical':bool(selection.all())})
    comp=pd.DataFrame(comparison);comp.to_csv(a.out/'non_group_comparison.csv',index=False)
    sums=comp.groupby('previous_round')[['non_group_rows','numeric_same','numeric_changed','selection95_same','selection95_changed']].sum()
    sums.to_csv(a.out/'non_group_summary.csv')
    detail=base[KEY].copy();detail['small_group']=mask;detail['base7']=base.time_hat;detail['round7_2']=new.time_hat
    for i,old in previous.items():detail[f'round{i}']=old.time_hat
    detail.to_csv(a.out/'row_comparison.csv',index=False)
    sys.path.insert(0,str(HERE/'official_validator_snapshot'))
    from pwscup2026.common.config import load_config
    from pwscup2026.kit.validate import load_dist,validate_aia_submission
    cfg=load_config(HERE/'official_validator_snapshot/pwscup2026/kit/kit_config.yaml')
    check=validate_aia_submission(pd.read_csv(a.out/'F_aia.csv'),cfg,load_dist(a.attack_dir,team_id=28))
    assert check.ok,check.errors
    audit={'format_validation':'PASS','scope':'AIA only','base7_sha256':BASE_HASH,
        'new_aia_sha256':hashlib.sha256(raw).hexdigest(),'input_sha256':{f.name:sha(f) for f in sorted(src.iterdir()) if f.is_file()},
        'rows':7172,'groups':57,'group_rows':181,'non_group_rows':6991,
        'changed_from_base7':int(new.time_hat.ne(base.time_hat).sum()),
        'changed_outside_groups':int(new.time_hat[~mask].ne(base.time_hat[~mask]).sum()),
        'group_choices':{str(i):[r['target_k'] for r in choices if r['group_source']==i] for i in [5,6,7]},
        'answer_counts':{str(v):int(n) for v,n in new.time_hat.value_counts().items()},
        'numeric_identical_team_rounds':comp.loc[comp.numeric_identical,['target_k','previous_round']].to_dict('records'),
        'selection95_identical_team_rounds':comp.loc[comp.selection95_identical,['target_k','previous_round']].to_dict('records'),
        'main_B_used':False,'main_truth_used':False,'official_score_obtained':False,
        'base7_note':'Frozen original candidate; model training is not repeated by this script.',
        'selection_note':'Whole-team official scores choose group values; group causality is not established.',
        'comparison_note':'Exact numeric equality and 9.5-selection equality are separate metrics. Selection equality alone does not imply equal scoring.'}
    (a.out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='input_sha256'},ensure_ascii=False,indent=2))
    print(sums.to_string())

if __name__=='__main__':main()
