"""Fixed 2x2 comparison for three bases; 6-row/90% smoothing, no search."""
from 採点共通 import *
import zipfile
import 評価部品 as parts
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PREV=HERE.parent/'有力3候補と元案_zhang11_seo比較'
PYTHON=ROOT/'スターターキット/初版/starter/.venv/bin/python'
ORDER=['Zhang04','Zhang11','Seo03']
TYPES=['original','hill','rare','combined']
TITLES={'original':'元案','hill':'山登りのみ','rare':'希少対策のみ','combined':'山登り＋希少対策'}
AIDS=['A04','A05','A06','M04']

def differences(before,after):
    assert before.shape==after.shape and list(before)==list(after)
    rows=np.zeros(len(before),bool);cols=[];cells=0
    for k in before:
        equal=np.isclose(before[k],after[k],atol=1e-10,rtol=0) if pd.api.types.is_numeric_dtype(before[k]) else before[k].to_numpy()==after[k].to_numpy()
        if not equal.all():cols.append(k);rows|=~equal;cells+=int((~equal).sum())
    return dict(rows=(np.flatnonzero(rows)+1).tolist(),attributes=cols,cells=cells)

def prepare():
    previous=load(PREV/'計画.json');p={k:previous[k] for k in ['kit','phi']}
    p.update(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),cases=[],panels=[],mia_panels=[],attacks=[],aia_references=[],mia_references=[],workers=16,base_order=ORDER,new_search=False,rare_seed=202609242358)
    for k,rel in [('b','inputs/utility/B.csv'),('a_bg','inputs/A_bg.csv')]:p[k]=copy(previous[k],HERE/rel)
    for f in ['B_self.csv','utility_ref.json']:copy(Path(previous['b']).with_name(f),HERE/'inputs/utility'/f)
    p['mia4']=copy(previous['mia4'],HERE/'programs/M04'/Path(previous['mia4']).name)
    for f in Path(previous['mia4']).parent.glob('*.py'):copy(f,HERE/'programs/M04'/f.name)
    for a in previous['attacks']:
        local=copy(a['local_path'],HERE/'programs'/a['id']/Path(a['local_path']).name)
        for f in Path(a['local_path']).parent.glob('*.py'):copy(f,HERE/'programs'/a['id']/f.name)
        p['attacks'].append(dict(a,local_path=local))
    for q in previous['panels']:
        p['panels'].append(dict(q,**{k:copy(q[k],HERE/'inputs/aia'/q['id']/Path(q[k]).name) for k in ['challenge','truth']}))
    for q in previous['mia_panels']:
        p['mia_panels'].append(dict(q,**{k:copy(q[k],HERE/'inputs/mia'/q['id']/Path(q[k]).name) for k in ['b','truth']}))
    kitview=HERE/'sources/実行用キット';(kitview/'starter').mkdir(parents=True,exist_ok=True)
    if not (kitview/'starter/src').exists():(kitview/'starter/src').symlink_to(Path(p['kit'])/'starter/src',target_is_directory=True)
    for f in (PREV/'sources/実行用キット/participant_data').glob('*'):copy(f,kitview/'participant_data'/f.name)
    script=Path(copy(PREV/'sources/zhang11_seo/zhang11_rare_smooth.py',HERE/'sources/zhang11_rare_smooth.py'))
    schema=load(kitview/'participant_data/schema.json')['rare_detector'];ref=load(Path(p['b']).with_name('utility_ref.json'))
    assert schema['mu']==ref['mu'] and schema['sigma']==ref['sigma']
    original_ids={'Zhang04':('C02','C27'),'Zhang11':('C01','C26'),'Seo03':('C03','C21')}
    for base in ORDER:
        orig,hill=[next(c for c in previous['cases'] if c['id']==cid) for cid in original_ids[base]]
        for typ in TYPES:
            cid=base+'_'+typ;cp=HERE/'C'/f'{cid}.csv';source=hill if typ in ['hill','combined'] else orig
            c=dict(id=cid,name=base+'｜'+TITLES[typ],base=base,type=typ,path=str(cp),target_id=9951+len(p['cases']),ops=source.get('ops',[]),hill_name=hill['name'],previous_id=None)
            if typ in ['original','hill']:
                copy(source['path'],cp);c['previous_id']=source['id']
            elif base=='Zhang11' and typ=='rare':
                saved=next(x for x in previous['cases'] if x['id']=='new_seo');copy(saved['path'],cp);c['previous_id']='new_seo'
            elif not cp.exists():
                zp=HERE/'sources/input_C'/f'{cid}.zip';zp.parent.mkdir(parents=True,exist_ok=True);cp.parent.mkdir(parents=True,exist_ok=True)
                with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:z.write(source['path'],'C.csv')
                cmd=[str(PYTHON),'-B',str(script),'--kit',str(kitview),'--zhang11-zip',str(zp),'--out-c',str(cp)]
                log=HERE/'generation'/f'{cid}.log';log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('w') as f:r=subprocess.run(cmd,cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
                assert r.returncode==0 and cp.exists(),str(log)
                say('作成 '+c['name'])
            c['sha256']=sha(cp);c['changes_from_original']=differences(read(orig['path']),read(cp))
            if typ in ['rare','combined']:
                c['smoothing_changes']=differences(read(source['path']),read(cp))
                assert len(c['smoothing_changes']['rows'])==6
                assert set(c['smoothing_changes']['attributes'])<=set(['age','BMI','SBP','TG','HDL','ALT','FPG'])
                assert 'time' not in c['smoothing_changes']['attributes']
            p['cases'].append(c)
        # Test whether the selected swaps and smoothing commute for this fixed pair.
        rare=read(next(c['path'] for c in p['cases'] if c['base']==base and c['type']=='rare'))
        for op in hill['ops']:
            i,j,k=op['i'],op['j'],op['attribute'];rare.loc[[i,j],k]=rare.loc[[j,i],k].to_numpy()
        combined=next(c for c in p['cases'] if c['base']==base and c['type']=='combined')
        cdelta=differences(rare,read(combined['path']));combined['order_commutes']=not cdelta['rows'];assert combined['order_commutes']
        swaprows={o[k]+1 for o in hill['ops'] for k in ['i','j']}
        combined['swap_smooth_row_overlap']=sorted(swaprows&set(combined['smoothing_changes']['rows']));assert not combined['swap_smooth_row_overlap']
    assert len(p['cases'])==12 and sum(c['previous_id'] is None for c in p['cases'])==5
    rare_panel(p)
    files=list((HERE/'inputs').rglob('*.csv'))+list((HERE/'inputs').rglob('*.json'))+list((HERE/'C').glob('*.csv'))+list((HERE/'programs').rglob('*.py'))+[script]
    files+=list((kitview/'participant_data').glob('*'))+list((Path(p['kit'])/'starter/src/pwscup2026').rglob('*.py'))+[Path(p['kit'])/'participant_data/scoring_config.yaml']
    p['hashes']={str(f):sha(f) for f in files};save(HERE/'計画.json',p)
    save(HERE/'事前固定_希少MIA条件.json',dict(seed=p['rare_seed'],panel=p['rare_panel'],cases=[dict(id=c['id'],sha256=c['sha256']) for c in p['cases']],attacks=AIDS,primary='true rare TPR at FPR=0.01; 24 positive, 48 negative',sensitivity=['source-specific controls','FPR=0.05','AUC'],no_parameter_search=True))
    return p

def rare_panel(p):
    b=read(p['b']);bs=read(Path(p['b']).with_name('B_self.csv'));bg=read(p['a_bg']);cols=[k for k in b if k!='record_id']
    assert bs.record_id.equals(b.record_id) and int(bs.is_rare.sum())==24
    paths=[('予備戦',next((ROOT/'予備戦/配布データ').glob('PWSCup2026_prelim_team05*/B_self.csv'))),('練習用',Path(p['kit'])/'participant_data/B_self.csv')]
    sources=[];audit=[]
    for label,path in paths:
        fp=copy(path,HERE/'inputs/rare_sources'/f'{label}_B_self.csv');a=read(fp);assert len(a)==1049 and int(a.is_rare.sum())==24
        assert set(a.is_rare.unique())<=set([0,1]);assert not a[cols].duplicated().any()
        overlap=len(a.merge(b[cols],on=cols));background_overlap=len(a.merge(bg[cols],on=cols));assert overlap==0 and background_overlap==0
        a['origin']=label;a['source_row_index']=np.arange(len(a));a['is_member']=0;sources.append(a)
        audit.append(dict(origin=label,path=fp,rows=len(a),rare=24,exact_overlap_B=overlap,exact_overlap_background=background_overlap))
    negative=pd.concat(sources,ignore_index=True);assert not negative[cols].duplicated().any()
    rng=np.random.default_rng(p['rare_seed']);rare_neg=negative[negative.is_rare.eq(1)];assert len(rare_neg)==48
    # Keep CLI query size at 1049: all rare targets/controls plus fixed ordinary padding.
    normals=negative[negative.is_rare.eq(0)];neg_pad=normals.iloc[rng.choice(len(normals),975,replace=False)]
    pos_idx=np.flatnonzero(bs.is_rare.to_numpy(bool));norm_idx=np.flatnonzero(~bs.is_rare.to_numpy(bool));pos_idx=np.r_[pos_idx,rng.choice(norm_idx,2,replace=False)]
    positive=bs.iloc[pos_idx].copy();positive['origin']='本戦B';positive['source_row_index']=pos_idx;positive['is_member']=1
    allrows=pd.concat([positive,rare_neg,neg_pad],ignore_index=True).iloc[rng.permutation(1049)].reset_index(drop=True)
    allrows['record_id']=np.arange(1,1050);assert not allrows[cols].duplicated().any()
    truthcols=['record_id','is_member','is_rare','origin','source_row_index'];out(HERE/'inputs/rare_diagnostic/B.csv',allrows[['record_id']+cols]);out(HERE/'inputs/rare_diagnostic/truth.csv',allrows[truthcols])
    assert allrows.groupby(['is_member','is_rare']).size().to_dict()=={(0,0):975,(0,1):48,(1,0):2,(1,1):24}
    p['rare_panel']=dict(id='rare_diagnostic',b=str(HERE/'inputs/rare_diagnostic/B.csv'),truth=str(HERE/'inputs/rare_diagnostic/truth.csv'),members=26,controls=1023,rare_members=24,rare_controls=48,seed=p['rare_seed'],audit=audit,diagnostic_only=True)

def init(p):parts.HERE=HERE;parts.init(p)

def run():
    p=load(HERE/'計画.json');assert all(sha(f)==h for f,h in p['hashes'].items());start=time.monotonic();ar=[];mr=[];done=0;total=144+120
    for base in ORDER:
        cases=[c for c in p['cases'] if c['base']==base]
        with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'),initializer=init,initargs=(p,)) as pool:
            for f in as_completed([pool.submit(parts.aia_job,c,[q['id']]) for c in cases for q in p['panels']]):
                ar.append(f.result());done+=1;say(bar(done,total)+' AIA '+base)
        parts.HERE=HERE
        with ThreadPoolExecutor(max_workers=16) as pool:
            batch=dict(p,cases=cases,batch_label=base)
            for f in as_completed([pool.submit(parts.mia_job,batch,q,a) for q in p['mia_panels']+[p['rare_panel']] for a in AIDS]):
                mr.append(f.result());done+=1;say(bar(done,total)+' MIA '+base)
    aggregate(p,ar,mr,time.monotonic()-start)

def aggregate(p,ar,mr,seconds):
    sys.path.insert(0,str(Path(p['kit'])/'starter/src'))
    from pwscup2026.scoring.attack_score import _tpr_at_fpr,score_mia
    aia=[x for r in ar for x in r['rows']];allmia=[x for r in mr for x in r['rows']];mia=[x for x in allmia if x['panel']!='rare_diagnostic'];rare=[]
    assert len(aia)==1728 and len(mia)==432 and len(allmia)==480
    truth=read(p['rare_panel']['truth']);assert len(truth)==1049
    cfg=yaml.safe_load((Path(p['kit'])/'participant_data/scoring_config.yaml').read_text())
    for result in [r for r in mr if r['panel']=='rare_diagnostic']:
        predframe=read(result['prediction']).set_index('record_id').loc[truth.record_id]
        for c in p['cases']:
            col=str(c['target_id'])
            if col not in predframe:continue
            values=predframe[col].to_numpy(float)
            for origin in ['両方','予備戦','練習用']:
                keep=truth.is_rare.eq(1)&(truth.is_member.eq(1)|truth.origin.eq(origin) if origin!='両方' else True)
                v=values[keep];y=truth.loc[keep,'is_member'].to_numpy(int);n_pos=int(y.sum());n_neg=len(y)-n_pos
                assert n_pos==24 and n_neg==(48 if origin=='両方' else 24)
                tpr,auc=_tpr_at_fpr(v,y,.01);tpr5,_=_tpr_at_fpr(v,y,.05);tpr0,_=_tpr_at_fpr(v,y,0.)
                rare.append(dict(case=c['id'],attack=result['attack'],controls=origin,n_pos=n_pos,n_neg=n_neg,TPR01=tpr,TPR05=tpr5,TPR0=tpr0,AUC=auc,positive_above_all_controls=int((v[y==1]>v[y==0].max()).sum())))
                if origin=='両方':
                    official=score_mia(values,truth.is_member.to_numpy(int),cfg,rare_mask=truth.is_rare.to_numpy(bool))
                    assert abs(official.breakdown['tpr_rare']-tpr)<1e-12
    assert len(rare)==144
    assert abs(_tpr_at_fpr(np.full(72,.5),np.r_[np.ones(24,int),np.zeros(48,int)],.01)[0]-.01)<1e-12
    prior=load(PREV/'比較結果.json');checks=0
    oldmaps={kind:{(r['case'],r['panel'],r['attack']):r for r in prior[kind]} for kind in ['aia','mia']}
    for kind,records,metric in [('aia',aia,'AIA'),('mia',mia,'TPR_at_FPR01')]:
        mapping={c['id']:c['previous_id'] for c in p['cases']}
        for r in records:
            if mapping[r['case']] is not None:
                expected=oldmaps[kind][mapping[r['case']],r['panel'],r['attack']][metric];assert abs(r[metric]-expected)<1e-12;checks+=1
    s=Scorer(dict(p,panel=p['panels'][0]));summaries=[]
    for c in p['cases']:
        c['utility']=s.full_score(read(c['path']),s.b,s.ref,s.cfg)
        c['aia_by_panel']={q['id']:max(r['AIA'] for r in aia if r['case']==c['id'] and r['panel']==q['id']) for q in p['panels']}
        c['aia387']=[c['aia_by_panel'][q] for q in ['P1','P2','P3']];c['max387']=max(c['aia387'])
        c['max163']=max(c['aia_by_panel'][q['id']] for q in p['panels'] if q['n']==163)
        c['max_mia']=max(r['TPR_at_FPR01'] for r in mia if r['case']==c['id'])
        c['rare_by_source']={source:max(r['TPR01'] for r in rare if r['case']==c['id'] and r['controls']==source) for source in ['両方','予備戦','練習用']}
        c['rare_by_attack']={a:next(r['TPR01'] for r in rare if r['case']==c['id'] and r['controls']=='両方' and r['attack']==a) for a in AIDS}
        c['rare_max05']=max(r['TPR05'] for r in rare if r['case']==c['id'] and r['controls']=='両方')
        if c['previous_id']:
            expected=next(x for x in prior['cases'] if x['id']==c['previous_id'])['utility']['utility'];assert abs(c['utility']['utility']-expected)<1e-12;checks+=1
    by={(c['base'],c['type']):c for c in p['cases']};effects=[]
    for base in ORDER:
        for frm,to in [('original','hill'),('original','rare'),('hill','combined'),('rare','combined'),('original','combined')]:
            b=by[base,frm];c=by[base,to]
            r=dict(base=base,before_type=frm,after_type=to,delta_U=c['utility']['utility']-b['utility']['utility'],delta_rare_max=c['rare_by_source']['両方']-b['rare_by_source']['両方'],delta_max387=c['max387']-b['max387'],delta_mia_max=c['max_mia']-b['max_mia'],delta_aia387=[x-y for x,y in zip(c['aia387'],b['aia387'])])
            r['aia163_worse']=sum(c['aia_by_panel'][q['id']]>b['aia_by_panel'][q['id']]+1e-12 for q in p['panels'] if q['n']==163)
            r['mia_worse']=sum(next(x['TPR_at_FPR01'] for x in mia if x['case']==c['id'] and x['panel']==q['id'] and x['attack']==a)>next(x['TPR_at_FPR01'] for x in mia if x['case']==b['id'] and x['panel']==q['id'] and x['attack']==a)+1e-12 for q in p['mia_panels'] for a in AIDS)
            r['rare_attack_deltas']={a:c['rare_by_attack'][a]-b['rare_by_attack'][a] for a in AIDS};effects.append(r)
    result=dict(cases=p['cases'],aia=aia,mia=mia,rare=rare,effects=effects,panels=p['panels'],mia_panels=p['mia_panels'],rare_panel=p['rare_panel'],attacks=p['attacks'],baselines=prior['baselines'],phi=p['phi'],seconds=seconds)
    assert all(sha(f)==h for f,h in p['hashes'].items());save(HERE/'比較結果.json',result)
    save(HERE/'検証結果.json',dict(passed=True,cases=12,new_candidates=5,aia_scores=1728,mia_scores=432,rare_attack_predictions=48,rare_metrics=144,matched_prior_checks=checks,rare_controls_true_labels=True,rare_source_exact_overlap_B=0,rare_source_exact_overlap_background=0,swap_smoothing_commutes_all_bases=True,input_hashes_unchanged=True,seconds=seconds))
    for c in p['cases']:say(json.dumps(dict(name=c['name'],U=c['utility']['utility'],rare=c['rare_by_attack'],rare_max=c['rare_by_source']['両方'],aia=c['aia387'],mia=c['max_mia']),ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.prepare and not a.run:a.prepare=not (HERE/'計画.json').exists();a.run=True
    if a.prepare:prepare()
    if a.run:run()
