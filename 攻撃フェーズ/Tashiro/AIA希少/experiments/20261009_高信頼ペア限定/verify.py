"""Independent checks of saved counts, baseline reproduction and fold separation."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import raw_score_counts,write_json,sha

HERE=Path(__file__).resolve().parent

def run(root:Path,out:Path):
    read=lambda p:pd.read_csv(p,float_precision='round_trip')
    f=read(out/'per_target.csv');ch=read(out/'changed_rows.csv');b=read(out/'baseline_reproduction.csv')
    cfg=json.loads((HERE/'protocol.json').read_text());meta=json.loads((out/'run.json').read_text())
    assert meta['protocol_sha256']==sha(HERE/'protocol.json')
    assert meta['script_sha256']==sha(HERE/'analyze.py')
    assert all(sha(root/n)==s for n,s in meta['input_sha256'].items())
    keys=['repeat','team','method','policy','low_value']
    assert not f.duplicated(keys).any() and len(f)==3*20*3*13
    assert set(f.team)==set(b.team)-{5}
    base=b.set_index(['team','method']);changes={k:g for k,g in ch.groupby(keys)}
    maxerr=0.
    for r in f.itertuples(index=False):
        key=tuple(getattr(r,k) for k in keys);bc=base.loc[(r.team,r.method)]
        c=changes.get(key,pd.DataFrame(columns=ch.columns))
        assert len(c)==r.changed and c.challenge_row_id.is_unique
        expected={m:int(bc[m]) for m in ['hit_m','hit_c','hit_m_rare','hit_c_rare']}
        for x in c.itertuples(index=False):
            assert x.old_time in [0.,9.5] and x.new_time==r.low_value and x.new_time!=x.old_time
            d=int(x.new_hit)-int(x.old_hit)
            expected['hit_m' if x.is_member else 'hit_c']+=d
            if x.is_rare:expected['hit_m_rare' if x.is_member else 'hit_c_rare']+=d
        assert all(getattr(r,k)==v for k,v in expected.items())
        all_=raw_score_counts(expected['hit_m'],expected['hit_c'])
        rare=raw_score_counts(expected['hit_m_rare'],expected['hit_c_rare'],13,1.64)
        maxerr=max(maxerr,abs(r.r_all-all_),abs(r.r_rare-rare),abs(r.r_max-max(all_,rare)))
    assert maxerr<1e-12
    src=root/'攻撃フェーズ/Tashiro/AIA全体/experiments/20261007/集計/同点修正_対象別.csv'
    old=read(src).query("method=='stable_matching/sex_quota/long75'")
    cmp=old.merge(b[b.method.eq('P01')],on='team',suffixes=('_old','_new'))
    berr=max((cmp[k+'_old']-cmp[k+'_new']).abs().max() for k in ['r_all','r_rare','r_max','p_m','p_c'])
    assert berr<1e-12
    old=read(root/'攻撃フェーズ/Tashiro/AIA全体/experiments/20261008/属性値集合による選別/集計/予備戦_平均.csv')
    old=old[(old.context=='full_reference_overlap')&(old.source=='real_C')]
    names={'P01':'baseline_partial2_matching','all4':'column_all4_mean','top2':'column_top2_mean'}
    for group,exclude in [('prelim20',[5]),('processed18',[1,2,5])]:
        nb=b[~b.team.isin(exclude)].groupby('method')[['r_all','r_rare','r_max','p_m','p_c']].mean()
        for method,nm in names.items():
            v=old[(old.group==group)&(old.method==nm)]
            assert len(v)==1
            for k in nb.columns:berr=max(berr,abs(float(v.iloc[0][k])-float(nb.loc[method,k])))
    assert berr<1e-12
    folds=json.loads((out/'folds.json').read_text())
    for rep in range(3):
        fs=[x for x in folds if x['repeat']==rep]
        assert len(fs)==7
        test=[t for x in fs for t in x['test']]
        assert len(set(test))==len(test)==21
        assert all(set(x['train']).isdisjoint(x['test']) for x in fs)
    q=[]
    for k in sorted(b.team.unique()):
        q.append(read(root/f'公式データ/予備戦/PWSCup2026_prelim_attack/aia_challenge_{k}.csv'))
    q=pd.concat(q,ignore_index=True)
    cols=['age','SBP','TG','HDL','ALT','sex','prefecture']
    duplicate_qi=int(q.duplicated(cols).sum())
    assert duplicate_qi==0
    pairs=read(out/'candidate_pairs.csv')
    assert (pairs.high_id!=pairs.low_id).all() and (pairs.gap>0).all()
    for _,g in pairs.groupby(['repeat','team','method','policy']):
        assert len(set(g.high_id)|set(g.low_id))==2*len(g)
    result=dict(passed=True,per_target_cases_checked=len(f),changed_rows_checked=len(ch),
                score_maximum_error=maxerr,original_baseline_maximum_error=berr,
                crossfit_train_test_disjoint=True,heldout_teams_per_repeat=21,
                global_QI7_duplicate_rows=duplicate_qi,original_input_hashes_verified=len(meta['input_sha256']),
                attempt_not_equal_changed=True,no_live_submission=True)
    write_json(out/'verification.json',result)
    write_json(out/'SHA256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256.json'})
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=HERE.parents[5]);p.add_argument('--out',type=Path,default=HERE/'results')
    a=p.parse_args();run(a.repo.resolve(),a.out.resolve())
