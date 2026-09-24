"""Freeze new panels, original C and programs; no previous attack scores are reused."""
from 採点共通 import *
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PILOT = HERE.parent / '初回450候補_ベース順次_16並列'
QI = ['age','SBP','TG','HDL','ALT','sex','prefecture']
MATCH = ['sex','prefecture','age_band','onset']
QUOTA = np.array([138,12,9,4])

def keys(d): return set(d[QI].itertuples(index=False, name=None))

def prepare():
    dest = HERE / '計画.json'
    if dest.exists():
        p = load(dest)
        for f,h in p['hashes'].items(): assert sha(f)==h, f
        return p
    old = load(PILOT/'計画.json'); hashes = {}
    def freeze(src, dst):
        hashes[str(src)] = sha(src)
        dst = copy(src, HERE/dst); hashes[dst]=sha(dst); return dst
    p = dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), kit=old['kit'],
        workers=16, base_order=['Zhang11','Zhang04'], attrs=['time','age','SBP','TG','HDL','ALT'],
        initial_per_attribute=500, expansion_per_parent=600, beam_width=3, max_swaps=10,
        planned_hours=15, validation_reserve_minutes=30, batch_per_parent=600,
        search_shares={'Zhang11':2,'Zhang04':1}, stop_on_plateau=False, hard_deadline=False,
        previous_run=str(HERE.parent/'15時間探索_2ベース_3パネル'), seed_parents={}, previous_keys={},
        seed=202609241502, development_seed=202609240915, old_predictions_reused=False, phi_applied=False,
        objective='max weighted AIA over dev1/dev2 x 12 attacks; strict decrease',
        utility_floor='full official U of own original C', variants=[], attacks=[], panels=[], mia_panels=[])
    for key,name in [('b','utility_inputs/B.csv'),('a_bg','public/A_bg.csv'),('attack_b','public/B_cli.csv')]:
        p[key] = freeze(old[key],name)
    for name in ['B_self.csv','utility_ref.json']:
        freeze(Path(old['b']).with_name(name), 'utility_inputs/'+name)
    # Original distribution metadata lives beside the official B, not the copied pilot B.
    hp = load(HERE.parent/'山登り_time_3ベース/計画.json')
    schema_path = Path(hp['b']).with_name('schema.json')
    schema = load(freeze(schema_path,'utility_inputs/schema.json'))
    for v in old['variants']:
        if v['label'] in p['base_order']:
            p['variants'].append(dict(label=v['label'],path=freeze(v['path'],'bases/'+v['label']+'/C.csv')))
    for a in old['attacks']:
        src=Path(a['local_path']); cp=freeze(src,'programs/'+a['id']+'/'+src.name)
        for asset in a['assets']: freeze(src.parent/asset['name'],'programs/'+a['id']+'/'+asset['name'])
        p['attacks'].append(dict(a,local_path=cp))
    mia4 = ROOT/'本戦/バトル/0923/programs/attack/seo-04a-多重縮小生成比MIA'
    p['mia4'] = freeze(mia4/(mia4.name+'.py'),'programs/M04/'+mia4.name+'.py')
    for f in mia4.glob('*.py'):
        if f.name!=mia4.name+'.py': freeze(f,'programs/M04/'+f.name)
    b=read(Path(p['b']).with_name('B_self.csv')); bg=read(p['a_bg'])
    pools={'予備戦':ROOT/'予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv',
           '練習用':Path(p['kit'])/'participant_data/A_bg.csv'}
    frames=[]; p['control_pools']={}
    for name,src in pools.items():
        fp=freeze(src,'sources/'+name+'.csv');p['control_pools'][name]=fp
        d=read(fp); assert len(d)==5143 and not keys(d)&(keys(b)|keys(bg))
        d['origin']=name;d['source_row_index']=np.arange(len(d));frames.append(d)
    assert not keys(frames[0])&keys(frames[1])
    pool=pd.concat(frames,ignore_index=True)
    b['source_row_index']=np.arange(len(b));b['origin']='本戦B'
    targets=b[b.prefecture.eq('神奈川県')].copy().reset_index(drop=True)
    assert len(targets)==387 and len(bg)==5143
    rare=schema['rare_detector']
    for d in [targets,pool]:
        d['age_band']=(d.age//10).astype(int)
        x=d[rare['columns']].to_numpy(float)-np.array(rare['mu'])
        d['proxy']=(np.einsum('ij,jk,ik->i',x,np.linalg.pinv(rare['sigma']),x)>=rare['band']['lo']**2).astype(int)
    targets['stratum']=targets.is_rare.astype(int)*2+targets.onset.astype(int)
    assert targets.groupby('stratum').size().tolist()==[338,35,10,4]
    overview=[]; control_sets={}
    for number,(pid,role,n) in enumerate([('dev1','探索用',387),('dev2','探索用',387),
            ('holdout','確認用・探索禁止',387),('small1','163人確認用',163),
            ('small2','163人確認用',163),('small3','163人確認用',163)]):
        seed=(p['development_seed'] if number<2 else p['seed'])+number; rng=np.random.default_rng(seed)
        if n==387: t=targets.copy()
        else:
            selected=np.concatenate([rng.choice(targets.index[targets.stratum.eq(s)],int(QUOTA[s]),replace=False) for s in range(4)])
            t=targets.loc[selected].copy().reset_index(drop=True)
        counts=t.groupby('stratum').size().reindex(range(4)).to_numpy()
        paired=np.full(n,-1,int)
        for key,idx in t.groupby(MATCH).groups.items():
            idx=np.asarray(idx); ai=np.asarray(pool.groupby(MATCH).groups[key]); assert len(ai)>=len(idx)
            mismatch=t.loc[idx,'proxy'].to_numpy()[:,None]!=pool.loc[ai,'proxy'].to_numpy()[None,:]
            gap=abs(t.loc[idx,'age'].to_numpy()[:,None]-pool.loc[ai,'age'].to_numpy()[None,:])
            cost=100*mismatch+rng.random((len(idx),len(ai)))+.001*gap
            rr,cc=linear_sum_assignment(cost);paired[idx[rr]]=ai[cc]
        assert (paired>=0).all() and len(set(paired))==n
        c=pool.loc[paired].reset_index(drop=True)
        np.testing.assert_array_equal(t[MATCH].to_numpy(),c[MATCH].to_numpy())
        source=pd.concat([t,c],ignore_index=True);order=rng.permutation(2*n)
        q=source.iloc[order][QI].reset_index(drop=True);q.insert(0,'challenge_row_id',np.arange(2*n))
        truth=pd.DataFrame(dict(challenge_row_id=np.arange(2*n),
            label=np.array(['member']*n+['control']*n)[order],time=source.time.to_numpy()[order],
            onset=source.onset.to_numpy()[order],pair_id=np.tile(np.arange(n),2)[order],
            weight_163=np.tile((QUOTA/counts)[t.stratum.to_numpy()],2)[order],
            paired_member_stratum=np.tile(t.stratum.to_numpy(),2)[order],
            member_true_rare_control_unknown=np.r_[t.is_rare.to_numpy(),np.full(n,-1)][order],
            origin=source.origin.to_numpy()[order],source_row_index=source.source_row_index.to_numpy()[order]))
        qp=HERE/'public'/pid/'aia_challenge.csv';tp=HERE/'evaluation_private'/pid/'truth.csv'
        out(qp,q);out(tp,truth)
        p['panels'].append(dict(id=pid,role=role,n=n,seed=seed,target_id=9800+number,challenge=str(qp),truth=str(tp)))
        hashes[str(qp)]=sha(qp);hashes[str(tp)]=sha(tp);control_sets[pid]=set(paired.tolist())
        overview.append(dict(パネル=pid,用途=role,標的=n,対照=n,重み付き人数=163,
            通常未発症=int(counts[0]),通常発症=int(counts[1]),希少未発症=int(counts[2]),希少発症=int(counts[3]),
            予備戦対照=int(c.origin.eq('予備戦').sum()),練習用対照=int(c.origin.eq('練習用').sum()),
            外れ値代理判定不一致=int((t.proxy.to_numpy()!=c.proxy.to_numpy()).sum())))
    out(HERE/'パネル構成.csv',overview)
    out(HERE/'パネル間の対照重複.csv',[dict(パネル1=a,パネル2=b,共通対照人数=len(control_sets[a]&control_sets[b]))
        for i,a in enumerate(control_sets) for b in list(control_sets)[i+1:]])
    # Membership evaluation uses outside-B/background records, with no AIA controls.
    available=np.array(sorted(set(range(len(pool)))-set.union(*control_sets.values())))
    cols=read(p['b']).columns.tolist()
    assert 'record_id' in cols and len(available)>=1023
    for i in range(3):
        rng=np.random.default_rng(p['seed']+100+i)
        mi=rng.choice(len(b),26,replace=False);ni=rng.choice(available,1023,replace=False)
        d=pd.concat([b.iloc[mi][cols],pool.iloc[ni][cols]],ignore_index=True)
        order=rng.permutation(1049);d=d.iloc[order].reset_index(drop=True);d['record_id']=np.arange(1,1050)
        truth=pd.DataFrame(dict(record_id=d.record_id,is_member=np.r_[np.ones(26,int),np.zeros(1023,int)][order],
            source_row_index=np.r_[mi,pool.iloc[ni].source_row_index.to_numpy()][order],
            origin=np.r_[np.repeat('本戦B',26),pool.iloc[ni].origin.to_numpy()][order]))
        qp=HERE/'public'/f'mia{i+1}'/'B.csv';tp=HERE/'evaluation_private'/f'mia{i+1}'/'truth.csv'
        out(qp,d);out(tp,truth);hashes[str(qp)]=sha(qp);hashes[str(tp)]=sha(tp)
        p['mia_panels'].append(dict(id=f'mia{i+1}',seed=p['seed']+100+i,b=str(qp),truth=str(tp),members=26,controls=1023))
    # Import only development-stage paths and operations; do not select from held-out results.
    previous=Path(p['previous_run'])
    for panel in p['panels'][:2]:
        assert sha(panel['challenge'])==sha(previous/'public'/panel['id']/'aia_challenge.csv')
        assert sha(panel['truth'])==sha(previous/'evaluation_private'/panel['id']/'truth.csv')
    for base in p['base_order']:
        result_path=previous/'search'/base/'完了.json'
        rows=load(result_path)['kept'];hashes[str(result_path)]=sha(result_path)
        p['seed_parents'][base]=[]
        for i,r in enumerate(rows):
            cp=freeze(r['path'],f'seed_inputs/{base}/{i:02d}/C.csv')
            p['seed_parents'][base].append(dict(id=f'seed-{base}-{i:02d}',base=base,ops=r['ops'],
                key=r['key'],input_path=cp,source_id=r['id']))
        seen=set()
        for jp in sorted((previous/'search'/base).glob('S*/候補計画.json')):
            hashes[str(jp)]=sha(jp)
            seen.update(j['key'] for j in load(jp))
        kp=HERE/'既評価候補'/f'{base}.json';save(kp,sorted(seen))
        hashes[str(kp)]=sha(kp);p['previous_keys'][base]=str(kp)
    for f in [previous/'採点共通.py',previous/'事前検証.json']:
        hashes[str(f)]=sha(f)
    # Freeze the official scoring implementation and the complete local orchestration.
    for f in list((Path(p['kit'])/'starter/src/pwscup2026').rglob('*.py'))+[Path(p['kit'])/'participant_data/scoring_config.yaml']+list(HERE.glob('*.py')):
        hashes[str(f)]=sha(f)
    p['hashes']=hashes
    verify(p)
    save(dest,p);save(HERE/'入力検証.json',dict(passed=True,new_panels=4,fixed_development_panels=2,mia_panels=3,targets=387,background=5143,old_predictions_reused=False))
    return p

def verify(p):
    sources={'本戦B':read(Path(p['b']).with_name('B_self.csv')),**{k:read(v) for k,v in p['control_pools'].items()}}
    for panel in p['panels']:
        q=read(panel['challenge']);t=read(panel['truth']);n=panel['n'];m=t.label.eq('member').to_numpy()
        assert len(q)==2*n and len(keys(q))==2*n and q.columns.tolist()==['challenge_row_id']+QI
        assert int(m.sum())==n and q.challenge_row_id.equals(t.challenge_row_id)
        for origin,rows in t.groupby('origin').groups.items():
            src=sources[origin].iloc[t.loc[rows,'source_row_index'].to_numpy(int)]
            pd.testing.assert_frame_equal(q.loc[rows,QI].reset_index(drop=True),src[QI].reset_index(drop=True))
            np.testing.assert_array_equal(t.loc[rows,'time'],src.time)
        np.testing.assert_allclose([t.loc[m,'weight_163'].sum(),t.loc[~m,'weight_163'].sum()],[163,163],atol=1e-10,rtol=0)
        assert t.groupby('pair_id').size().eq(2).all()
    for panel in p['mia_panels']:
        q=read(panel['b']);t=read(panel['truth']);assert len(q)==1049 and t.is_member.sum()==26
        assert q.record_id.equals(t.record_id) and q.record_id.is_unique
        for origin,rows in t.groupby('origin').groups.items():
            src=sources[origin].iloc[t.loc[rows,'source_row_index'].to_numpy(int)]
            cols=q.columns.drop('record_id')
            pd.testing.assert_frame_equal(q.loc[rows,cols].reset_index(drop=True),src[cols].reset_index(drop=True))

if __name__=='__main__': prepare()
