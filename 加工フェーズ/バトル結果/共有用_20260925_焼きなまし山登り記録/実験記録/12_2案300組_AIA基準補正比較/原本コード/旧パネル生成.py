"""Fixed-C confirmation panels; deterministic generation, no tuning on outcomes."""
from 採点共通 import *
from scipy.optimize import linear_sum_assignment
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PREVIOUS=HERE.parents[1]/'0924/有力3候補と元案_zhang11_seo比較'
LATEST=HERE.parents[1]/'0924/U低下許容_20分探索_20260924'
QI=['age','SBP','TG','HDL','ALT','sex','prefecture']
MATCH=['sex','prefecture','age_band','onset']
QUOTA=np.array([138,12,9,4])
def keys(d):return set(d[QI].itertuples(index=False,name=None))

def sources(p):
    b=read(Path(p['b']).with_name('B_self.csv'));b['source_row_index']=np.arange(len(b));b['origin']='本戦B'
    frames=[]
    for origin,f in p['control_pools'].items():
        d=read(f);d['origin']=origin;d['source_row_index']=np.arange(len(d));frames.append(d)
    pool=pd.concat(frames,ignore_index=True);targets=b[b.prefecture.eq('神奈川県')].copy().reset_index(drop=True)
    rare=load(p['schema'])['rare_detector']
    for d in [targets,pool]:
        d['age_band']=(d.age//10).astype(int)
        x=d[rare['columns']].to_numpy(float)-np.array(rare['mu'])
        d['proxy']=(np.einsum('ij,jk,ik->i',x,np.linalg.pinv(rare['sigma']),x)>=rare['band']['lo']**2).astype(int)
    targets['stratum']=targets.is_rare.astype(int)*2+targets.onset.astype(int)
    assert targets.groupby('stratum').size().tolist()==[338,35,10,4]
    assert len(pool)==10286 and not keys(pool)&(keys(b)|keys(read(p['a_bg'])))
    assert len(keys(pool))==len(pool)
    return b,pool,targets

def aia_panel(pool,targets,n,seed):
    rng=np.random.default_rng(seed)
    if n==387:t=targets.copy()
    else:
        selected=np.concatenate([rng.choice(targets.index[targets.stratum.eq(s)],int(QUOTA[s]),replace=False) for s in range(4)])
        t=targets.loc[selected].copy().reset_index(drop=True)
    counts=t.groupby('stratum').size().reindex(range(4)).to_numpy();paired=np.full(n,-1,int)
    groups=pool.groupby(MATCH).groups
    for key,idx in t.groupby(MATCH).groups.items():
        idx=np.asarray(idx);ai=np.asarray(groups[key]);assert len(ai)>=len(idx)
        mismatch=t.loc[idx,'proxy'].to_numpy()[:,None]!=pool.loc[ai,'proxy'].to_numpy()[None,:]
        gap=abs(t.loc[idx,'age'].to_numpy()[:,None]-pool.loc[ai,'age'].to_numpy()[None,:])
        rr,cc=linear_sum_assignment(100*mismatch+rng.random((len(idx),len(ai)))+.001*gap);paired[idx[rr]]=ai[cc]
    assert (paired>=0).all() and len(set(paired))==n
    c=pool.loc[paired].reset_index(drop=True);np.testing.assert_array_equal(t[MATCH].to_numpy(),c[MATCH].to_numpy())
    src=pd.concat([t,c],ignore_index=True);order=rng.permutation(2*n)
    q=src.iloc[order][QI].reset_index(drop=True);q.insert(0,'challenge_row_id',np.arange(2*n))
    truth=pd.DataFrame(dict(challenge_row_id=np.arange(2*n),label=np.array(['member']*n+['control']*n)[order],time=src.time.to_numpy()[order],
        onset=src.onset.to_numpy()[order],pair_id=np.tile(np.arange(n),2)[order],weight_163=np.tile((QUOTA/counts)[t.stratum.to_numpy()],2)[order],
        paired_member_stratum=np.tile(t.stratum.to_numpy(),2)[order],member_true_rare_control_unknown=np.r_[t.is_rare.to_numpy(),np.full(n,-1)][order],
        origin=src.origin.to_numpy()[order],source_row_index=src.source_row_index.to_numpy()[order]))
    overview=dict(n=n,normal_no_onset=int(counts[0]),normal_onset=int(counts[1]),rare_no_onset=int(counts[2]),rare_onset=int(counts[3]),
        control_prelim=int(c.origin.eq('予備戦').sum()),control_practice=int(c.origin.eq('練習用').sum()),proxy_mismatches=int((t.proxy.to_numpy()!=c.proxy.to_numpy()).sum()))
    return q,truth,overview

def prepare():
    dest=HERE/'計画.json'
    if dest.exists():
        p=load(dest)
        for f,h in p['hashes'].items():assert sha(f)==h,f
        return p
    old=load(PREVIOUS/'計画.json');latest=load(LATEST/'計画.json');p={k:old[k] for k in ['kit','b','a_bg','attacks','mia4','phi']}
    p.update(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),cases=old['cases'],workers=16,base_order=['Zhang04','Zhang11','Seo03'],panels=[],mia_panels=[],aia_references=[],mia_references=[],control_pools={},schema=str(LATEST/'utility_inputs/schema.json'))
    p.update(new_panels_per_kind=20,seed_scheme={'387':202609250100,'163':202609250200,'MIA':202609250300},purpose='confirmation only; no C optimization',previous_result=str(PREVIOUS/'比較結果.json'))
    for c in p['cases']:
        c['path']=copy(c['path'],HERE/'C'/Path(c['path']).name);assert sha(c['path'])==c['sha256']
    for key,rel in [('b','inputs/utility/B.csv'),('a_bg','inputs/A_bg.csv'),('schema','inputs/utility/schema.json')]:p[key]=copy(p[key],HERE/rel)
    for f in ['B_self.csv','utility_ref.json']:copy(Path(old['b']).with_name(f),HERE/'inputs/utility'/f)
    for origin,src in latest['control_pools'].items():p['control_pools'][origin]=copy(src,HERE/'inputs/pool'/Path(src).name)
    save(HERE/'結果を見る前の固定方針.json',{k:p[k] for k in ['created','cases','new_panels_per_kind','seed_scheme','purpose','base_order']})
    b,pool,targets=sources(p);overview=[]
    # Check exact compatibility with the old panel generator, before making new panels.
    for n in [387,163]:
        oldq=next(q for q in latest['panels'] if q['id']==('holdout' if n==387 else 'small1'))
        q,t,_=aia_panel(pool,targets,n,oldq['seed'])
        pd.testing.assert_frame_equal(q,read(oldq['challenge']));pd.testing.assert_frame_equal(t,read(oldq['truth']),check_dtype=False,atol=1e-14,rtol=0)
    for n in [387,163]:
        for i in range(1,21):
            pid=f'N{n}_{i:02d}';seed=p['seed_scheme'][str(n)]+i;q,t,info=aia_panel(pool,targets,n,seed)
            qp=HERE/'public'/pid/'aia_challenge.csv';tp=HERE/'evaluation_private'/pid/'truth.csv';out(qp,q);out(tp,t)
            p['panels'].append(dict(id=pid,label=f'追加{n}人・組{i:02d}',n=n,seed=seed,challenge=str(qp),truth=str(tp),set_id='new',role='候補固定後の追加確認'))
            overview.append(dict(panel=pid,seed=seed,**info))
    # Keep the MIA eligibility pool identical to the preceding 20-minute experiment.
    # Excluding all 40 new AIA panels would change its population composition.
    excluded=set()
    for q in latest['panels']:
        t=read(q['truth']);excluded.update(t.loc[t.label.eq('control'),['origin','source_row_index']].itertuples(index=False,name=None))
    available=np.array([i for i,r in pool.iterrows() if (r.origin,int(r.source_row_index)) not in excluded]);assert len(available)>=1023
    p['mia_pool_note']='Same eligible records as 0924 U低下許容 experiment; new AIA/MIA panels may reuse control identities across tests.'
    p['mia_eligible_count']=len(available);out(HERE/'inputs/MIA対象候補一覧.csv',pool.iloc[available][['origin','source_row_index']].to_dict('records'))
    cols=read(p['b']).columns.tolist()
    for i in range(1,21):
        pid=f'NM_{i:02d}';seed=p['seed_scheme']['MIA']+i;rng=np.random.default_rng(seed)
        mi=rng.choice(len(b),26,replace=False);ni=rng.choice(available,1023,replace=False)
        d=pd.concat([b.iloc[mi][cols],pool.iloc[ni][cols]],ignore_index=True);order=rng.permutation(1049);d=d.iloc[order].reset_index(drop=True);d['record_id']=np.arange(1,1050)
        t=pd.DataFrame(dict(record_id=d.record_id,is_member=np.r_[np.ones(26,int),np.zeros(1023,int)][order],source_row_index=np.r_[mi,pool.iloc[ni].source_row_index.to_numpy()][order],origin=np.r_[np.repeat('本戦B',26),pool.iloc[ni].origin.to_numpy()][order]))
        qp=HERE/'public'/pid/'B.csv';tp=HERE/'evaluation_private'/pid/'truth.csv';out(qp,d);out(tp,t)
        p['mia_panels'].append(dict(id=pid,label=f'追加MIA・組{i:02d}',seed=seed,b=str(qp),truth=str(tp),members=26,controls=1023,set_id='new'))
    out(HERE/'パネル構成.csv',overview)
    verify(p)
    files=[str(f) for f in (HERE/'inputs').rglob('*') if f.is_file()]+[c['path'] for c in p['cases']]+[p['previous_result']]
    for q in p['panels']:files.extend([q['challenge'],q['truth']])
    for q in p['mia_panels']:files.extend([q['b'],q['truth']])
    for a in p['attacks']:files.extend(str(f) for f in Path(a['local_path']).parent.glob('*.py'))
    files.extend(str(f) for f in Path(p['mia4']).parent.glob('*.py'))
    files.extend(str(f) for f in (Path(p['kit'])/'starter/src/pwscup2026').rglob('*.py'))
    files.append(str(Path(p['kit'])/'participant_data/scoring_config.yaml'))
    p['hashes']={f:sha(f) for f in files};save(dest,p)
    save(HERE/'入力検証.json',dict(passed=True,new_aia_panels=40,new_mia_panels=20,old_generator_reproduction=2,fixed_cases=7,mia_eligible_count=len(available),seeds_fixed_before_results=True))
    return p

def verify(p):
    b,pool,targets=sources(p);ss={'本戦B':b,**{o:read(f) for o,f in p['control_pools'].items()}}
    sets=[]
    for q in p['panels']:
        x,t=read(q['challenge']),read(q['truth']);n=q['n'];m=t.label.eq('member')
        assert len(x)==2*n and len(keys(x))==2*n and m.sum()==n and x.columns.tolist()==['challenge_row_id']+QI
        assert x.challenge_row_id.equals(t.challenge_row_id)
        np.testing.assert_allclose([t.loc[m,'weight_163'].sum(),t.loc[~m,'weight_163'].sum()],[163,163],atol=1e-10,rtol=0)
        for o,idx in t.groupby('origin').groups.items():
            src=ss[o].iloc[t.loc[idx,'source_row_index'].to_numpy(int)]
            pd.testing.assert_frame_equal(x.loc[idx,QI].reset_index(drop=True),src[QI].reset_index(drop=True));np.testing.assert_array_equal(t.loc[idx,'time'],src.time)
        identity=frozenset(t.loc[~m,['origin','source_row_index']].itertuples(index=False,name=None));assert identity not in [z['controls'] for z in sets]
        sets.append(dict(id=q['id'],n=n,controls=identity,targets=set(t.loc[m,'source_row_index'])))
    overlaps=[]
    for i,a in enumerate(sets):
        for b in sets[i+1:]:
            if a['n']==b['n']:overlaps.append(dict(panel1=a['id'],panel2=b['id'],n=a['n'],shared_targets=len(a['targets']&b['targets']),shared_controls=len(a['controls']&b['controls'])))
    out(HERE/'パネル間の重複.csv',overlaps)
    for q in p['mia_panels']:
        x,t=read(q['b']),read(q['truth']);assert len(x)==1049 and t.is_member.sum()==26 and x.record_id.is_unique and x.record_id.equals(t.record_id)
        for o,idx in t.groupby('origin').groups.items():
            src=ss[o].iloc[t.loc[idx,'source_row_index'].to_numpy(int)];cols=x.columns.drop('record_id')
            pd.testing.assert_frame_equal(x.loc[idx,cols].reset_index(drop=True),src[cols].reset_index(drop=True))

if __name__=='__main__':prepare()
