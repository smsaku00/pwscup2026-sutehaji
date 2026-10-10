"""Reverse round-5_2 small groups on a frozen round-6 answer and compare prior answers.
Usage: python round6_compare.py --inputs INPUT_DIR --round6 ROUND6_ZIP --out OUTPUT_DIR
INPUT_DIR: r3.csv, r4.csv, r5_2.csv, group_assignments.csv, r3.py, r4.py,
           public_attack/aia_challenge_<k>.csv (to verify public grouping).
Never logs or models the submission token; MIA/token are copied unchanged.
"""
import argparse, ast, hashlib, io, json, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['target_k', 'challenge_row_id']

def load_frame(source):
    d = pd.read_csv(source).sort_values(KEY).reset_index(drop=True)
    assert list(d.columns) == KEY + ['time_hat']
    assert not d.duplicated(KEY).any()
    assert len(d) == 7172 and len(d.target_k.unique()) == 22
    assert d.groupby('target_k').size().eq(326).all()
    assert np.isfinite(d.time_hat).all()
    return d

def constant(path, name):
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise ValueError(name)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--attack-dir', type=Path, required=True)
    p.add_argument('--round6', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.round6) as z:
        base_raw=z.read('F_aia.csv'); mia=z.read('F_mia.csv'); token=z.read('token.txt')
    base=load_frame(io.BytesIO(base_raw))
    old={r:load_frame(a.inputs/f'r{r}.csv') for r in [3,4]}
    r5=load_frame(a.inputs/'r5_2.csv')
    for d in [*old.values(),r5]: assert d[KEY].equals(base[KEY])
    assert set(base.time_hat.unique()) == {0,9.5}
    groups=pd.read_csv(a.inputs/'group_assignments.csv')
    assert len(groups)==181 and not groups.duplicated(KEY).any()
    assert groups.groupby(['target_k','group_id']).ngroups==57
    lookup=base.set_index(KEY); prev=r5.set_index(KEY)
    for (k,gid),g in groups.groupby(['target_k','group_id']):
        n=len(g);assert n in [2,3,4] and g.group_size.eq(n).all()
        assert int(g.time_hat.eq(9.5).sum()) == (n+1)//2
        q=pd.read_csv(a.attack_dir/f'aia_challenge_{k}.csv')
        q['age_band']=(q.age//5).astype(int)
        ids=set(g.challenge_row_id.astype(int));sub=q[q.challenge_row_id.isin(ids)]
        assert len(sub)==n and len(sub[['prefecture','sex','age_band']].drop_duplicates())==1
        key=sub[['prefecture','sex','age_band']].iloc[0]
        allsame=(q[['prefecture','sex','age_band']]==key).all(axis=1)
        assert set(q.loc[allsame,'challenge_row_id'].astype(int))==ids
        for row in g.itertuples():
            assert prev.loc[(row.target_k,row.challenge_row_id),'time_hat']==row.time_hat
            lookup.loc[(row.target_k,row.challenge_row_id),'time_hat']=11. if row.time_hat==9.5 else 9.5
    new=lookup.reset_index().sort_values(KEY).reset_index(drop=True)
    gkeys=set(map(tuple,groups[KEY].to_numpy())); mask=np.array([tuple(x) in gkeys for x in base[KEY].to_numpy()])
    assert np.array_equal(new.time_hat[~mask],base.time_hat[~mask])
    assert new.time_hat[mask].isin([9.5,11]).all()
    assert (new.time_hat[mask].to_numpy()!=r5.time_hat[mask].to_numpy()).all()
    r3teams=constant(a.inputs/'r3.py','P01_TEAMS');r4route=constant(a.inputs/'r4.py','ROUTE')
    detail=new[KEY].copy(); detail['small_group']=mask
    detail['r3_time_hat']=old[3].time_hat;detail['r4_time_hat']=old[4].time_hat
    detail['r5_2_time_hat']=r5.time_hat;detail['original_r6_time_hat']=base.time_hat;detail['r6_2_time_hat']=new.time_hat
    detail.to_csv(a.out/'row_comparison.csv',index=False)
    rows=[]
    for k,d in detail.groupby('target_k'):
        record={'target_k':int(k),'r3_method':'P01' if k in r3teams else 'other', 'r4_method':r4route[k],
                'small_group_rows':int(d.small_group.sum()),'r6_2_n95':int(d.r6_2_time_hat.eq(9.5).sum()),
                'r6_group_changed':int(d.r6_2_time_hat.ne(d.original_r6_time_hat).sum())}
        for r in [3,4]:
            v=d[f'r{r}_time_hat'];n=d.r6_2_time_hat
            eq=np.isclose(v,n,rtol=0,atol=1e-9)
            switch=v.eq(9.5)!=n.eq(9.5)
            record.update({f'r{r}_numeric_changed':int((~eq).sum()), f'r{r}_95_selection_changed':int(switch.sum()),
                           f'r{r}_95_to_non95':int((v.eq(9.5)&n.ne(9.5)).sum()),
                           f'r{r}_non95_to_95':int((v.ne(9.5)&n.eq(9.5)).sum()),
                           f'r{r}_selection_changed_outside_groups':int((switch&~d.small_group).sum()),
                           f'r{r}_zero_to_11_only':int((v.eq(0)&n.eq(11)).sum())})
        rows.append(record)
    summary=pd.DataFrame(rows);summary.to_csv(a.out/'per_target_comparison.csv',index=False)
    r4teams={k for k,v in r4route.items() if v.startswith(('P01_','P05_'))}
    aggregates={}
    for name,r,teams in [('round3_P01',3,r3teams),('round4_P01_P05',4,r4teams),('round4_all_partial',4,{k for k,v in r4route.items() if v.startswith(('P01_','P03_','P04_','P05_'))})]:
        s=summary[summary.target_k.isin(teams)]
        aggregates[name]={'teams':sorted(teams),'rows':326*len(teams),
            'numeric_changed':int(s[f'r{r}_numeric_changed'].sum()),
            '95_selection_changed':int(s[f'r{r}_95_selection_changed'].sum()),
            'selection_changed_outside_groups':int(s[f'r{r}_selection_changed_outside_groups'].sum()),
            'zero_to_11_only':int(s[f'r{r}_zero_to_11_only'].sum())}
    raw=new.to_csv(index=False,lineterminator='\n',float_format='%.17g').encode()
    (a.out/'F_aia.csv').write_bytes(raw)
    for no_token in [False,True]:
        suffix='_no_token' if no_token else ''
        path=a.out/f'zhang_T28_test06_2_column_top2_group_reverse{suffix}.zip'
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            z.writestr('F_mia.csv',mia);z.writestr('F_aia.csv',raw)
            if not no_token:z.writestr('token.txt',token)
        with zipfile.ZipFile(path) as z:
            assert z.read('F_mia.csv')==mia and z.read('F_aia.csv')==raw
            if not no_token:assert z.read('token.txt')==token
    audit={'aia_rows':7172,'opponents':22,'group_rows':181,'groups':57,
        'group_reverse_relative_to':'5_2, not a new rank by round6 scores',
        'three_person_rule':'1 answers 9.5, 2 answer 11 (exact reversal of 5_2)',
        'modified_from_original_round6':int(new.time_hat.ne(base.time_hat).sum()),
        'selection_modified_from_original_round6':int(new.time_hat.eq(9.5).ne(base.time_hat.eq(9.5)).sum()),
        'answer_counts':{str(k):int(v) for k,v in new.time_hat.value_counts().items()},
        'aggregates':aggregates,'F_aia_sha256':hashlib.sha256(raw).hexdigest(),
        'base_round6_sha256':hashlib.sha256(base_raw).hexdigest(),
        'MIA_token_copied_byte_for_byte':True,'official_score_obtained':False,
        'note':'0 is not guaranteed to miss; 11 is outside true 0..10 range. 95-selection metrics apply as binary comparison only for P01/P05.'}
    (a.out/'comparison_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps(audit,ensure_ascii=False,indent=2));print(summary.to_string(index=False))

if __name__=='__main__':main()
