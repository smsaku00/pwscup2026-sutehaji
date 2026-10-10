#!/usr/bin/env python3
"""Rebuild Team28 round6 column-top2, then exactly reverse 5_2 small groups.
Only public A_bg/C/challenge and fixed 5_2 assignment IDs are prediction inputs.
Writes F_aia.csv and an audit JSON; validates only the AIA answer.
"""
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
CLIN=['SBP','TG','HDL','ALT']
EXPECTED_BASE='9964ccdef6eab7f8b90a5422a52c833b10a68149f3f9b090a7496ac031e0bc3b'
EXPECTED_NEW='47900c8b65e29fd3f9c790d8cb52e9b204269d57b5e5a275267fdea23fb629be'

def transform(d):
    x=d[CLIN].to_numpy(dtype=float).copy()
    for name in ['TG','ALT']:
        j=CLIN.index(name);x[:,j]=np.log1p(np.maximum(0,x[:,j]))
    return x

def near_1d(source,values):
    s=np.sort(np.asarray(source,float));values=np.asarray(values,float)
    if not len(s):return np.full(len(values),1e7)
    k=np.searchsorted(s,values,side='left');out=np.full(len(values),np.inf)
    ok=k<len(s);out[ok]=np.abs(values[ok]-s[k[ok]])
    ok=k>0;out[ok]=np.minimum(out[ok],np.abs(values[ok]-s[k[ok]-1]))
    return out

def base_prediction(q,c,bg,b,iqr):
    x=transform(q);cx=transform(c);dc=np.zeros((len(q),4));db=dc.copy()
    qs=q.sex.to_numpy(int);cs=c.sex.to_numpy(int);bs=bg.sex.to_numpy(int)
    for sex in [0,1]:
        ii=np.flatnonzero(qs==sex)
        if not len(ii):continue
        for j in range(4):
            dc[ii,j]=near_1d(cx[cs==sex,j]/iqr[j],x[ii,j]/iqr[j])
            db[ii,j]=near_1d(b[bs==sex,j]/iqr[j],x[ii,j]/iqr[j])
    evidence=np.log((db+0.001)/(dc+0.001))
    score=np.sort(evidence,axis=1)[:,-2:].mean(axis=1)
    keep=np.zeros(len(q),bool)
    for sex in sorted(q.sex.unique()):
        ix=np.flatnonzero(qs==sex)
        order=np.lexsort((q.challenge_row_id.to_numpy()[ix],-score[ix]))
        keep[ix[order[:math.ceil(.75*len(ix))]]]=True
    return np.where(keep,9.5,0.).astype(float)

def csv_bytes(d):
    return d.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()

def validate(aia_path,attack_dir):
    sys.path.insert(0,str(HERE/'official_validator_snapshot'))
    from pwscup2026.common.config import load_config
    from pwscup2026.kit.validate import load_dist,validate_aia_submission
    cfg=load_config(HERE/'official_validator_snapshot/pwscup2026/kit/kit_config.yaml')
    dist=load_dist(attack_dir,team_id=28)
    result=validate_aia_submission(pd.read_csv(aia_path),cfg,dist)
    if not result.ok:raise ValueError(result.errors)
    return 'PASS'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attack-dir',type=Path,required=True)
    p.add_argument('--a-bg',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('round6_reproduced'))
    a=p.parse_args()
    bg=pd.read_csv(a.a_bg);b=transform(bg)
    iqr=np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0),1e-5)
    mc=json.loads((a.attack_dir/'mia_columns.json').read_text(encoding='utf-8-sig'))
    targets=sorted(set(map(int,mc['target_columns']))-{28})
    groups=pd.read_csv(HERE/'参照/5_2_group_assignments.csv')
    assert len(targets)==22 and len(groups)==181 and not groups.duplicated(['target_k','challenge_row_id']).any()
    assert groups.groupby(['target_k','group_id']).ngroups==57
    bases=[];news=[];source_hashes={'A_bg.csv':hashlib.sha256(a.a_bg.read_bytes()).hexdigest()}
    for k in targets:
        qp=a.attack_dir/f'aia_challenge_{k}.csv';cp=a.attack_dir/f'C_{k}.csv'
        q=pd.read_csv(qp).sort_values('challenge_row_id').reset_index(drop=True);c=pd.read_csv(cp)
        assert len(q)==326 and q.challenge_row_id.is_unique
        v=base_prediction(q,c,bg,b,iqr)
        base=pd.DataFrame({'target_k':k,'challenge_row_id':q.challenge_row_id,'time_hat':v})
        new=base.copy();index=dict(zip(q.challenge_row_id,range(len(q))))
        team_groups=groups[groups.target_k.eq(k)]
        q['age_band']=(q.age//5).astype(int)
        for _,g in team_groups.groupby('group_id'):
            n=len(g);assert n in [2,3,4] and g.group_size.eq(n).all()
            assert int(g.time_hat.eq(9.5).sum())==(n+1)//2
            ids=set(g.challenge_row_id);sub=q[q.challenge_row_id.isin(ids)]
            assert len(sub)==n and len(sub[['prefecture','sex','age_band']].drop_duplicates())==1
            key=sub[['prefecture','sex','age_band']].iloc[0]
            assert set(q.loc[(q[['prefecture','sex','age_band']]==key).all(axis=1),'challenge_row_id'])==ids
            for row in g.itertuples():
                assert row.time_hat in [9.5,11]
                new.loc[index[row.challenge_row_id],'time_hat']=11. if row.time_hat==9.5 else 9.5
        nongroup=~q.challenge_row_id.isin(team_groups.challenge_row_id)
        assert np.array_equal(new.time_hat[nongroup],base.time_hat[nongroup])
        bases.append(base);news.append(new)
        for f in [qp,cp]:source_hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    base=pd.concat(bases,ignore_index=True);new=pd.concat(news,ignore_index=True)
    old_raw=csv_bytes(base);raw=csv_bytes(new)
    assert hashlib.sha256(old_raw).hexdigest()==EXPECTED_BASE,'Original round6 did not reproduce'
    assert hashlib.sha256(raw).hexdigest()==EXPECTED_NEW,'Round6_2 did not reproduce'
    assert set(new.time_hat.unique())=={0.,9.5,11.} and len(new)==7172
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'F_aia.csv').write_bytes(raw)
    status=validate(a.out/'F_aia.csv',a.attack_dir)
    audit={'format_validation':status,'validation_scope':'AIA only','base_aia_sha256':EXPECTED_BASE,'new_aia_sha256':EXPECTED_NEW,
           'rows':len(new),'targets':len(targets),'groups':57,'group_rows':181,
           'changed_from_original_round6':int(new.time_hat.ne(base.time_hat).sum()),
           'source_hashes':source_hashes,'main_B_used':False,'main_truth_used':False,'official_score_obtained':False}
    (a.out/'generation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='source_hashes'},ensure_ascii=False))

if __name__=='__main__':main()
