from common import *

def main():
    if (HERE/'計画.json').exists():
        p=plan()
        for f,h in p['hashes'].items():assert sha(f)==h,f
        print('prepared',len(p['panels']));return
    old=load(OLD/'計画.json')
    for bucket in ['source_hashes','generated_hashes']:
        for f,h in old[bucket].items():assert sha(f)==h,f
    b=read(Path(old['b']).with_name('B_self.csv'));bg=read(old['a_bg']);b['source_row']=np.arange(len(b))
    m=b[b.prefecture.eq('神奈川県')].copy().reset_index(drop=True);assert len(m)==387
    m['source']='本戦B';m['team']=28;m['is_rare']=m.is_rare.astype(int)
    allc=[];sources=[]
    for tp in sorted((ROOT/'予備戦/配布データ/結果/正解データ').glob('aia_truth_*.csv')):
        k=int(tp.stem.split('_')[-1]);qp=ROOT/f'予備戦/配布データ/PWSCup2026_prelim_attack/aia_challenge_{k}.csv'
        x=read(qp).merge(read(tp),on='challenge_row_id',validate='one_to_one');x=x[x.label.eq('control')].copy()
        x['source']='予備戦公式対照';x['source_row']=x.challenge_row_id;x['team']=k;allc.append(x);sources.extend([str(tp),str(qp)])
    controls=pd.concat(allc,ignore_index=True);assert len(controls)==3423 and len(keys(controls))==3423
    excluded=keys(b)|keys(bg)|keys(read(OLD/'public/B_attacker.csv'))
    for c in old['C']:excluded|=keys(read(c['path']))
    overlap=controls[QI].apply(tuple,axis=1).isin(excluded);exclusions=int(overlap.sum())
    controls=controls[~overlap].reset_index(drop=True)
    for d in [m,controls]:d['stratum']=(d.is_rare*2+d.onset).astype(int);d['band']=(d.age//10).astype(int)
    audit={'controls_before':3423,'excluded_QI_overlap':exclusions,'controls_after':len(controls),
           'same_prefecture':int(controls.prefecture.eq('神奈川県').sum()),'target_groups':m.groupby('stratum').size().to_dict()}
    panels=[]
    for op in old['panels']:
        panels.append(dict(op,id=f"old{op['panel']:02d}",dataset='old387',split='development',scenario='既存387人',true_control_rare=False))
    def write_panel(pid,targets,ctrl,seed,scenario,split,dataset,county):
        assert len(targets)==len(ctrl) and len(keys(ctrl))==len(ctrl)
        assert not keys(targets)&keys(ctrl)
        assert np.array_equal(targets[['sex','onset','is_rare']].to_numpy(),ctrl[['sex','onset','is_rare']].to_numpy())
        if county:assert np.array_equal(targets.prefecture.to_numpy(),ctrl.prefecture.to_numpy())
        age_exceptions=int((targets.band.to_numpy()!=ctrl.band.to_numpy()).sum())
        if not county:assert age_exceptions==0
        rng=np.random.default_rng(seed+1);n=len(targets);src=pd.concat([targets,ctrl],ignore_index=True);order=rng.permutation(2*n)
        q=src.iloc[order][QI].reset_index(drop=True);q.insert(0,'challenge_row_id',np.arange(2*n))
        t=src.iloc[order][['time','onset','is_rare','stratum','source','source_row','team']].reset_index(drop=True)
        t.insert(0,'challenge_row_id',np.arange(2*n));t['label']=np.array(['member']*n+['control']*n)[order];t['pair_id']=np.tile(np.arange(n),2)[order]
        counts=targets.groupby('stratum').size().reindex(range(4)).to_numpy();t['weight_163']=(QUOTA/counts)[t.stratum]
        qp=HERE/'public'/pid/'aia_challenge.csv';tp=HERE/'evaluation_private'/pid/'truth.csv';out(qp,q);out(tp,t)
        p=dict(id=pid,dataset=dataset,split=split,scenario=scenario,seed=seed,target_id=10000+len(panels),challenge=str(qp),truth=str(tp),true_control_rare=True,
               target_rows=n,control_rows=n,prefecture_matched=county,age_band_exceptions=age_exceptions,
               cross_prefecture_pairs=int((targets.prefecture.to_numpy()!=ctrl.prefecture.to_numpy()).sum()),
               member_sex0=int(targets.sex.eq(0).sum()),control_sex0=int(ctrl.sex.eq(0).sum()))
        panels.append(p)
    match=['stratum','sex','band']
    def pair(targets,pool,rng,strict=True):
        used=set();ids=[]
        for _,row in targets.iterrows():
            z=pool[(pool.stratum==row.stratum)&(pool.sex==row.sex)&~pool.index.isin(used)]
            exact=z[z.band==row.band]
            if len(exact):z=exact
            elif strict:raise AssertionError(('no control',row[match].tolist()))
            else:
                # Explicit limited age-band relaxation; never condition on time or attack scores.
                z=z[(z.age-row.age).abs()==(z.age-row.age).abs().min()]
            idx=int(rng.choice(z.index));ids.append(idx);used.add(idx)
        return pool.loc[ids].reset_index(drop=True)
    for rep in range(3):
        seed=2026092400+rep;rng=np.random.default_rng(seed);ctrl=pair(m,controls,rng)
        write_panel(f'dev387_{rep+1}',m,ctrl,seed,'全県対照・387人固定','development','new387',False)
    # Predeclare six final panels, never used by the annealing objective or candidate selection.
    # These are sensitivity conditions, not a claim that the hidden official target has these sex ratios.
    allocations={'性別ほぼ均等':[[69,69],[9,3],[5,4],[2,2]],
                 '性別0が多い':[[131,7],[9,3],[5,4],[2,2]],
                 '性別1が多い':[[7,131],[9,3],[5,4],[2,2]]}
    local=controls[controls.prefecture.eq('神奈川県')]
    for si,(scenario,counts) in enumerate(allocations.items()):
        target_rng=np.random.default_rng(2026092500+si*10);parts=[]
        for g,pair_counts in enumerate(counts):
            for sex,n in enumerate(pair_counts):
                z=m[(m.stratum==g)&(m.sex==sex)]
                if g!=3:
                    permitted=[]
                    for band,zz in z.groupby('band'):
                        cap=len(local[(local.stratum==g)&(local.sex==sex)&(local.band==band)])
                        permitted.extend(target_rng.choice(zz.index,min(len(zz),cap),replace=False).tolist())
                    z=z.loc[permitted]
                assert len(z)>=n,(scenario,g,sex,len(z),n)
                parts.append(z.loc[target_rng.choice(z.index,n,replace=False)])
        target=pd.concat(parts).reset_index(drop=True);assert len(target)==163
        for county in [False,True]:
            seed=2026092500+si*10+int(county);rng=np.random.default_rng(seed+100)
            ctrl=pair(target,local if county else controls,rng,strict=not county)
            write_panel(f"final163_{si+1}_{'samepref' if county else 'allpref'}",target,ctrl,seed,scenario,'holdout','final163',county)
    variants=[]
    for c in old['C']:variants.append(dict(label=c['label'],path=c['path'],utility=c['utility']))
    rng=np.random.default_rng(20260923387)
    for i in range(10):
        idx=rng.choice(len(bg),1049,replace=False);path=HERE/'public'/f'background_{i+1:02d}.csv';out(path,bg.iloc[idx][COLS])
        variants.append(dict(label=f'BG{i+1:02d}',path=str(path),seed=20260923387,sample_index=i+1))
    hashes={**{f:sha(f) for f in sources},str(Path(old['b']).with_name('B_self.csv')):sha(Path(old['b']).with_name('B_self.csv')),
            **{str(v['path']):sha(v['path']) for v in variants},**{p[k]:sha(p[k]) for p in panels for k in ['challenge','truth']}}
    p=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_plan=str(OLD/'計画.json'),kit=old['kit'],b=old['b'],a_bg=old['a_bg'],
           variants=variants,attacks=old['attacks'],panels=panels,background_replicates=10,hashes=hashes,control_audit=audit,
           search=dict(bases=['Zhang11','Zhang04','Seo03'],attacks=['A03','A06','A07'],panels=['dev387_1','dev387_2','dev387_3'],chains=2,steps=80,
                       utility_floor='each original saved C utility',validation_not_for_selection=True),
           limits=['allpref panels are cross-prefecture sensitivity tests','samepref final panels relax age band when no exact rare-onset control exists',
                   'targets and rare patients overlap across development and final panels; not independent persons',
                   'official phi is not applied; official P membership is unknown','final163 truth and outcomes must not select annealing candidates'])
    p['search_strong']={**p['search'],'attacks':['A08','A10','A11'],'panels':['old01','old02','dev387_1'],
        'revision_reason':'Development reassessment: original three attacks have almost no positive target lift; select three additional attacks with positive lift, before any final163 results. Initial exploratory candidates are not used.'}
    save(HERE/'計画.json',p);save(HERE/'対照候補監査.json',audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    print('prepared panels:',[(x['id'],x.get('age_band_exceptions')) for x in panels],flush=True)

if __name__=='__main__':main()
