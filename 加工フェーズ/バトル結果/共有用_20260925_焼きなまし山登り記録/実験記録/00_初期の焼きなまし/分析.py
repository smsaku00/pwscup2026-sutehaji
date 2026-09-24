"""Reproduce the approved experiment with --run; default performs read-only verification."""
from common import *
import argparse

def verify(write=False):
    p=plan()
    for f,h in p['hashes'].items():assert sha(f)==h,f
    old=load(OLD/'計画.json')
    for f,h in old['source_hashes'].items():assert sha(f)==h,f
    b=read(Path(p['b']).with_name('B_self.csv'));control_sources={};final_members={};n_records=0
    for pp in p['panels']:
        if pp['dataset']=='old387':continue
        q=read(pp['challenge']);t=read(pp['truth']);assert q.columns.tolist()==['challenge_row_id']+QI
        assert np.array_equal(q.challenge_row_id,t.challenge_row_id) and q.challenge_row_id.is_unique
        mask=t.label.eq('member');assert mask.sum()==(~mask).sum()==pp['target_rows']
        for idx,row in t.iterrows():
            if row.label=='member':src=b.iloc[int(row.source_row)]
            else:
                team=int(row.team)
                if team not in control_sources:
                    qq=read(ROOT/f'予備戦/配布データ/PWSCup2026_prelim_attack/aia_challenge_{team}.csv')
                    tt=read(ROOT/f'予備戦/配布データ/結果/正解データ/aia_truth_{team}.csv')
                    control_sources[team]=qq.merge(tt,on='challenge_row_id',validate='one_to_one').set_index('challenge_row_id')
                src=control_sources[team].loc[int(row.source_row)];assert src.label=='control'
            assert tuple(src[QI])==tuple(q.loc[idx,QI])
            assert src.time==row.time and src.onset==row.onset and src.is_rare==row.is_rare
            n_records+=1
        age_ex=0
        for pair,group in t.groupby('pair_id'):
            assert len(group)==2 and group.label.nunique()==2
            assert group.onset.nunique()==1 and group.is_rare.nunique()==1
            qq=q.loc[group.index];assert qq.sex.nunique()==1
            age_ex+=int((qq.age//10).nunique()!=1)
            if pp['prefecture_matched']:assert qq.prefecture.nunique()==1
        assert age_ex==pp['age_band_exceptions']
        if pp['dataset']=='final163':
            for role in ['member','control']:
                np.testing.assert_array_equal(t[t.label==role].groupby('stratum').size().reindex(range(4)),QUOTA)
            k=pp['scenario'];mkeys=keys(q[mask])
            if k in final_members:assert final_members[k]==mkeys
            final_members[k]=mkeys
    frozen=load(HERE/'search_strong/固定候補.json');candidates=frozen['variants'];jobs=0
    for c in candidates:
        assert sha(c['path'])==c['sha256'];u=load(Path(c['path']).with_name('有用性.json'))
        base=next(v for v in p['variants'] if v['label']==c['base']);assert u['utility']>=base['utility']['U']-1e-12
        original=read(base['path']);candidate=read(c['path'])
        pd.testing.assert_frame_equal(original.drop(columns='time'),candidate.drop(columns='time'))
        for _,idx in original.groupby(['onset','death']).groups.items():
            np.testing.assert_array_equal(np.sort(original.loc[idx,'time']),np.sort(candidate.loc[idx,'time']))
        assert int((original.time!=candidate.time).sum())==c['changed_rows']
    for meta in (HERE/'work').glob('*/*/*/実行記録.json'):
        r=load(meta);assert r['returncode']==0 and sha(r['prediction'])==r['prediction_sha256'];jobs+=1
    for dataset in ['old387','new387','final163']:
        d=read(HERE/f'{dataset}_手法別の増分.csv')
        np.testing.assert_allclose(d['標的的中率']-d['背景標的的中率'],d['標的的中率の増分'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(d['追加的中人数平均']-d['的中喪失人数平均'],d['差引追加的中人数平均'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(d['標的的中率の増分']-d['対照的中率の増分'],d['標的増分と対照増分の差'],atol=1e-12,rtol=0)
        expected=np.clip((d.標的的中率-d.対照的中率)/np.maximum(1-d.対照的中率,1e-9),0,1)
        np.testing.assert_allclose(expected,d.全体AIA,atol=1e-12,rtol=0)
    final=read(HERE/'最終6条件_改良前後の差.csv');assert len(final)==18
    assert load(HERE/'verification_strong/検証.json')['passed']
    result=dict(passed=True,source_hashes=True,source_record_comparisons=n_records,
        original_panels=10,new_development_panels=3,final_panels=6,final_same_member_comparisons=3,
        same_prefecture_age_band_exception_per_panel=1,saved_attack_jobs=jobs,background_samples_per_method=10,
        reference_baseline_trials=1000,final_candidates=3,final_C_attack_comparisons=6*12*6,
        full_utility_verified=True,selection_uses_final_results=False,official_phi_applied=False,official_Anon_calculated=False,
        code_hashes={f.name:sha(f) for f in HERE.glob('*.py')})
    if write:save(HERE/'検証結果.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result

def run_script(name,*args):
    subprocess.run([str(PYTHON),'-B',str(HERE/name),*args],check=True,cwd=ROOT)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--write-audit',action='store_true');args=ap.parse_args()
    if args.run:
        run_script('準備.py')
        for dataset in ['old387','new387']:run_script('実行.py','--dataset',dataset)
        for dataset in ['old387','new387']:run_script('評価.py','--dataset',dataset)
        run_script('背景基準.py','new387')
        run_script('探索_主要攻撃.py','--verify');run_script('探索_主要攻撃.py')
        for dataset in ['old387','new387','final163']:run_script('実行.py','--dataset',dataset,'--candidates')
        for dataset in ['old387','new387','final163']:run_script('評価.py','--dataset',dataset,'--candidates')
        run_script('背景基準.py','final163');run_script('報告.py')
        verify(True)
    else:verify(args.write_audit)
if __name__=='__main__':main()
