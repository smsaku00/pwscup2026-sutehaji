"""Seven fixed C files, identical AIA/MIA panels. No optimization or submission."""
from 採点共通 import *
import 評価部品 as parts
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OLD=HERE.parent/'山登り実験一覧_有用性と387人AIA3組'
FAIR=HERE.parent/'公平比較_初回合格案と長時間5案_20260924'
LATEST=HERE.parent/'U低下許容_20分探索_20260924'
ORDER=['Zhang04','Zhang11','Seo03']

def prepare():
    old=load(OLD/'計画.json');fair=load(FAIR/'計画.json');latest=load(LATEST/'計画.json')
    p={k:old[k] for k in ['kit','b','a_bg','attacks','phi']}
    p.update(cases=[],panels=old['panels'].copy(),mia_panels=[],mia4=fair['mia4'],aia_references=[],mia_references=[],workers=16)
    for cid in ['C02','C27','C01','C26','C03','C21']:
        c=next(x.copy() for x in old['cases'] if x['id']==cid)
        c['source']=c['path'];c['path']=copy(c['source'],HERE/'C'/f'{cid}.csv');p['cases'].append(c)
    p['cases'].insert(4,dict(id='new_seo',name='zhang11_seo',base='Zhang11',path=str(HERE/'C/zhang11_seo_team28.csv'),swaps=None,ops=[],stage='GitHub追加案',source='GitHub main@1d87a2fdcdea31f21bf9045e7b4bb1e0874662fd'))
    for i,c in enumerate(p['cases']):c['target_id']=9911+i;c['sha256']=sha(c['path'])
    p['panels'] += [q.copy() for q in fair['panels'] if q['n']==163]
    p['panels'] += [dict(q,id='latest_'+q['id']) for q in latest['panels'] if q['n']==163]
    p['mia_panels'] = fair['mia_panels']+[dict(q,id='latest_'+q['id']) for q in latest['mia_panels']]
    assert len(p['panels'])==12 and len(p['mia_panels'])==9
    for key,rel in [('b','inputs/utility/B.csv'),('a_bg','inputs/A_bg.csv')]:p[key]=copy(p[key],HERE/rel)
    for f in ['B_self.csv','utility_ref.json']:copy(Path(old['b']).with_name(f),HERE/'inputs/utility'/f)
    for q in p['panels']:
        for key in ['challenge','truth']:q[key]=copy(q[key],HERE/'inputs/aia'/q['id']/Path(q[key]).name)
    for q in p['mia_panels']:
        for key in ['b','truth']:q[key]=copy(q[key],HERE/'inputs/mia'/q['id']/Path(q[key]).name)
    files=[p['b'],p['a_bg'],str(Path(p['b']).with_name('B_self.csv')),str(Path(p['b']).with_name('utility_ref.json'))]+[c['path'] for c in p['cases']]
    for q in p['panels']:files += [q['challenge'],q['truth']]
    for q in p['mia_panels']:files += [q['b'],q['truth']]
    for a in p['attacks']:files += [str(f) for f in Path(a['local_path']).parent.glob('*.py')]
    files += [str(f) for f in Path(p['mia4']).parent.glob('*.py')]
    files += [str(f) for f in (Path(p['kit'])/'starter/src/pwscup2026').rglob('*.py')]
    files += [str(Path(p['kit'])/'participant_data/scoring_config.yaml')]
    files += [str(f) for f in (HERE/'sources/実行用キット/participant_data').glob('*')]+[str(HERE/'sources/zhang11_seo/zhang11_rare_smooth.py')]
    p['hashes']={f:sha(f) for f in files};save(HERE/'計画.json',p)
    return p

def init(p):
    parts.HERE=HERE;parts.init(p)

def run():
    p=load(HERE/'計画.json');start=time.monotonic();ar=[];mr=[];done=0;total=84+108
    for base in ORDER:
        cases=[c for c in p['cases'] if c['base']==base]
        with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'),initializer=init,initargs=(p,)) as pool:
            for f in as_completed([pool.submit(parts.aia_job,c,[q['id']]) for c in cases for q in p['panels']]):
                ar.append(f.result());done+=1;say(bar(done,total)+' AIA '+base)
        parts.HERE=HERE
        with ThreadPoolExecutor(max_workers=16) as pool:
            batch=dict(p,cases=cases,batch_label=base)
            for f in as_completed([pool.submit(parts.mia_job,batch,q,a) for q in p['mia_panels'] for a in ['A04','A05','A06','M04']]):
                mr.append(f.result());done+=1;say(bar(done,total)+' MIA '+base)
    aggregate(p,ar,mr,time.monotonic()-start)

def aggregate(p,ar,mr,seconds):
    aia=[x for r in ar for x in r['rows']];mia=[x for r in mr for x in r['rows']]
    assert len(aia)==1008 and len(mia)==252
    s=Scorer(dict(p,panel=p['panels'][0]));constants=[]
    for q in p['panels'][:3]:
        ss=Scorer(dict(p,panel=q))
        constants.append(dict(panel=q['id'],constant10=ss.metrics(np.full(len(ss.q),10.)),background1NN=ss.metrics(ss.predict('A12',p['a_bg']))))
    old=load(OLD/'比較結果.json');fair=load(FAIR/'比較集計.json');checks=0
    for c in p['cases']:
        c['utility']=s.full_score(read(c['path']),s.b,s.ref,s.cfg)
        c['max_aia']={q['id']:max(x['AIA'] for x in aia if x['case']==c['id'] and x['panel']==q['id']) for q in p['panels']}
        c['max_387']=max(c['max_aia'][q] for q in ['P1','P2','P3'])
        c['max_163']=max(c['max_aia'][q['id']] for q in p['panels'] if q['n']==163)
        c['max_mia']=max(x['TPR_at_FPR01'] for x in mia if x['case']==c['id'])
        if c['id']!='new_seo':
            prev=next(x for x in old['cases'] if x['id']==c['id'])
            assert abs(c['utility']['utility']-prev['utility']['utility'])<1e-12
            for q in ['P1','P2','P3']:
                past=np.load(OLD/'results'/c['id']/q/'予測.npz');fresh=np.load(HERE/'aia'/c['id']/q/'予測.npz')
                for a in p['attacks']:
                    np.testing.assert_allclose(fresh[q+'__'+a['id']],past[a['id']],atol=1e-10,rtol=0);checks+=1
    baseids={'Zhang04':'C02','Zhang11':'C01','Seo03':'C03'}
    for c in p['cases']:
        b=next(x for x in p['cases'] if x['id']==baseids[c['base']]);c['delta_U']=c['utility']['utility']-b['utility']['utility']
        c['delta_387']=[c['max_aia'][q]-b['max_aia'][q] for q in ['P1','P2','P3']]
        c['delta_constant']=[c['max_aia'][q['panel']]-q['constant10']['AIA'] for q in constants]
        c['delta_background']=[c['max_aia'][q['panel']]-q['background1NN']['AIA'] for q in constants]
        c['bad_163']=sum(c['max_aia'][q['id']]>b['max_aia'][q['id']]+1e-12 for q in p['panels'] if q['n']==163)
        bm={(r['panel'],r['attack']):r['TPR_at_FPR01'] for r in mia if r['case']==b['id']}
        c['bad_mia']=sum(r['TPR_at_FPR01']>bm[r['panel'],r['attack']]+1e-12 for r in mia if r['case']==c['id'])
        ca,cb=read(c['path']),read(b['path']);changed=~np.isclose(ca.to_numpy(float),cb.to_numpy(float),atol=1e-10,rtol=0) if all(pd.api.types.is_numeric_dtype(ca[k]) for k in ca) else None
        attrs=[];rows=np.zeros(len(ca),bool);cells=0
        for k in ca:
            eq=np.isclose(ca[k],cb[k],atol=1e-10,rtol=0) if pd.api.types.is_numeric_dtype(ca[k]) else ca[k].to_numpy()==cb[k].to_numpy()
            if not eq.all():attrs.append(k);rows|=~eq;cells+=int((~eq).sum())
        c['changed_rows']=int(rows.sum());c['changed_attributes']=attrs;c['changed_cells']=cells
    new=next(c for c in p['cases'] if c['id']=='new_seo');assert new['changed_rows']==6 and 'time' not in new['changed_attributes']
    mapping={'C02':'base_Zhang04','C27':'long5','C01':'base_Zhang11','C26':'long1'}
    for kind,freshs,pasts,keys in [('AIA',aia,fair['aia'],['AIA']),('MIA',mia,fair['mia'],['TPR_at_FPR01','AUC'])]:
        ref={(r['case'],r['panel'],r['attack']):r for r in pasts}
        for r in freshs:
            k=(mapping.get(r['case']),r['panel'],r['attack'])
            if k in ref:
                for v in keys:assert abs(r[v]-ref[k][v])<1e-12,(kind,k,v)
                checks+=1
    assert all(sha(f)==h for f,h in p['hashes'].items())
    result=dict(cases=p['cases'],panels=p['panels'],mia_panels=p['mia_panels'],attacks=p['attacks'],aia=aia,mia=mia,baselines=constants,phi=p['phi'],seconds=seconds)
    save(HERE/'比較結果.json',result);out(HERE/'集計/AIA_全手法.csv',aia);out(HERE/'集計/MIA_全手法.csv',mia)
    save(HERE/'検証結果.json',dict(passed=True,cases=7,aia_scores=1008,mia_scores=252,matched_checks=checks,input_hashes_unchanged=True,new_changed_rows=new['changed_rows'],seconds=seconds))
    say(json.dumps([dict(name=c['name'],U=c['utility']['utility'],aia=[c['max_aia'][q] for q in ['P1','P2','P3']],bad163=c['bad_163'],badMIA=c['bad_mia']) for c in p['cases']],ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');args=ap.parse_args()
    if not args.prepare and not args.run:
        args.prepare=not (HERE/'計画.json').exists();args.run=True
    if args.prepare:prepare()
    if args.run:run()
