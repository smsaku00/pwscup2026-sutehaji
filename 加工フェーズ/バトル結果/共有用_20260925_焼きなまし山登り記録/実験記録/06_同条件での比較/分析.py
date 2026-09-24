"""Eight fixed C files, both confirmation sets, identical attacks and acceptance rules."""
from 採点共通 import *
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PYTHON=ROOT/'スターターキット/初版/starter/.venv/bin/python'
RUNS={'first':HERE.parent/'15時間探索_2ベース_3パネル','long':HERE.parent/'15時間継続探索_時間配分方式_v2'}
SET_NAMES={'train':'共通の探索用','first':'初回の確認セット','long':'長時間版の確認セット'}
P=None;SC={}
EPS=1e-12

def key(c):return hashlib.sha256(pd.util.hash_pandas_object(c,index=True).to_numpy().tobytes()).hexdigest()

def prepare():
    pp=HERE/'計画.json'
    if pp.exists():
        p=load(pp)
        for f,h in p['hashes'].items():assert sha(f)==h,f
        return p
    sources={k:load(v/'計画.json') for k,v in RUNS.items()};new=sources['long'];hashes={}
    def freeze(src,rel):
        hashes[str(src)]=sha(src);dst=copy(src,HERE/rel);hashes[dst]=sha(dst);return dst
    for k in ['b','a_bg','attack_b']:
        assert sha(sources['first'][k])==sha(new[k]),k
    p=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),kit=new['kit'],cases=[],panels=[],mia_panels=[],attacks=[],aia_references=[],mia_references=[])
    for k,rel in [('b','utility_inputs/B.csv'),('a_bg','public/A_bg.csv'),('attack_b','public/B_cli.csv')]:p[k]=freeze(new[k],rel)
    for f in ['B_self.csv','utility_ref.json']:freeze(Path(new['b']).with_name(f),'utility_inputs/'+f)
    for a in new['attacks']:
        old=next(x for x in sources['first']['attacks'] if x['id']==a['id'])
        assert sha(a['local_path'])==sha(old['local_path'])
        src=Path(a['local_path']);cp=freeze(src,'programs/'+a['id']+'/'+src.name)
        for asset in a['assets']:
            assert sha(src.parent/asset['name'])==sha(Path(old['local_path']).parent/asset['name'])
            freeze(src.parent/asset['name'],'programs/'+a['id']+'/'+asset['name'])
        p['attacks'].append(dict(a,local_path=cp))
    p['mia4']=freeze(new['mia4'],'programs/M04/'+Path(new['mia4']).name)
    for src in Path(new['mia4']).parent.glob('*.py'):freeze(src,'programs/M04/'+src.name)
    for base in ['Zhang11','Zhang04']:
        v=next(v for v in new['variants'] if v['label']==base)
        p['cases'].append(dict(id='base_'+base,label=base+'（元のC）',name=base+'（元のC）',base=base,origin='元の案',ops=[],swaps=0,source=v['path']))
    first=next(x for x in load(RUNS['first']/'確認前の候補固定.json') if x['display_name']=='Zhang04-HC1')
    p['cases'].append(dict(id='first_pass',label='初回合格案',name='Zhang04-HC1',base='Zhang04',origin='初回約1時間',ops=first['ops'],swaps=1,source=first['path']))
    selected=sorted(load(RUNS['long']/'確認前の候補固定.json'),key=lambda x:x['display_name'])
    selected=sorted(selected,key=lambda x:(0 if x['base']=='Zhang11' else 1,x['display_name']))
    for i,r in enumerate(selected,1):
        p['cases'].append(dict(id=f'long{i}',label=f'長時間案{i}',name=r['display_name'],base=r['base'],origin='長時間探索の最終候補',ops=r['ops'],swaps=r['total_swaps'],source=r['path']))
    for index,c in enumerate(p['cases']):
        c['target_id']=9901+index;c['path']=freeze(c['source'],'C/'+c['name']+'.csv');c['key']=key(read(c['path']));c['sha256']=sha(c['path'])
        base=next(x for x in p['cases'] if x['id']=='base_'+c['base'])
        expected=read(base['path'])
        for op in c['ops']:
            i,j,attr=op['i'],op['j'],op['attribute']
            expected.loc[[i,j],attr]=expected.loc[[j,i],attr].to_numpy()
        pd.testing.assert_frame_equal(read(c['path']),expected)
    assert len(p['cases'])==8 and len({c['key'] for c in p['cases']})==8
    for group,srcp in sources.items():
        for panel in srcp['panels']:
            if panel['id'].startswith('dev'):
                if group=='long':
                    old=next(x for x in sources['first']['panels'] if x['id']==panel['id'])
                    assert sha(panel['challenge'])==sha(old['challenge']) and sha(panel['truth'])==sha(old['truth'])
                    continue
                pid=panel['id'];set_id='train';title='探索用'+pid[-1]
            else:
                pid=group+'_'+panel['id'];set_id=group
                title='387人' if panel['id']=='holdout' else '163人-'+panel['id'][-1]
            q=freeze(panel['challenge'],'public/'+pid+'/aia_challenge.csv')
            t=freeze(panel['truth'],'evaluation_private/'+pid+'/truth.csv')
            p['panels'].append(dict(panel,id=pid,source_pid=panel['id'],set_id=set_id,label=title,challenge=q,truth=t))
        for panel in srcp['mia_panels']:
            pid=group+'_'+panel['id'];q=freeze(panel['b'],'public/'+pid+'/B.csv');t=freeze(panel['truth'],'evaluation_private/'+pid+'/truth.csv')
            p['mia_panels'].append(dict(panel,id=pid,source_pid=panel['id'],set_id=group,b=q,truth=t,label='MIA-'+panel['id'][-1]))
    # Saved results are references for checking new calculations, never the calculation source.
    for group,srcroot in RUNS.items():
        refs=[]
        for rid in ['original-Zhang11','original-Zhang04']+[r['id'] for r in load(srcroot/'確認前の候補固定.json')]:
            f=srcroot/'results'/(rid+'-confirm')/'結果.json'
            if f.exists():refs.append(load(f));hashes[str(f)]=sha(f)
        for c in p['cases']:
            matching=[r for r in refs if r['key']==c['key']]
            if not matching:continue
            r=matching[0];assert sha(r['predictions'])==r['predictions_sha256'];hashes[r['predictions']]=sha(r['predictions'])
            for panel in p['panels']:
                if panel['set_id'] not in ['train',group]:continue
                p['aia_references'].append(dict(case=c['id'],panel=panel['id'],prefix=panel['source_pid'],predictions=r['predictions']))
            for panel in p['mia_panels']:
                if panel['set_id']!=group:continue
                for aid in ['A04','A05','A06','M04']:
                    f=srcroot/'mia'/r['id']/panel['source_pid']/aid/'結果.json'
                    assert f.exists();rr=load(f);assert sha(rr['prediction'])==rr['prediction_sha256']
                    hashes[str(f)]=sha(f);hashes[rr['prediction']]=sha(rr['prediction'])
                    p['mia_references'].append(dict(case=c['id'],panel=panel['id'],attack=aid,prediction=rr['prediction']))
    for f in list((Path(p['kit'])/'starter/src/pwscup2026').rglob('*.py'))+[Path(p['kit'])/'participant_data/scoring_config.yaml']+list(HERE.glob('*.py')):hashes[str(f)]=sha(f)
    p['hashes']=hashes;save(pp,p);return p

def init(p):
    global P
    P=p
def scorer(pid):
    if pid not in SC:
        panel=next(x for x in P['panels'] if x['id']==pid)
        SC[pid]=Scorer(dict(P,panel=panel))
    return SC[pid]

def aia_job(case,pids):
    folder=HERE/'aia'/case['id']/('_'.join(pids));folder.mkdir(parents=True,exist_ok=True)
    meta=folder/'結果.json'
    if meta.exists():
        r=load(meta);assert r['C_sha256']==sha(case['path']) and sha(r['predictions'])==r['predictions_sha256'];return r
    pred={};rows=[];checks=0
    for pid in pids:
        s=scorer(pid);s.shared_c={};s.cost=None
        for a in P['attacks']:
            aid=a['id'];value=s.predict(aid,case['path']);pred[pid+'__'+aid]=value
            rows.append(dict(case=case['id'],panel=pid,attack=aid,**s.metrics(value)))
            for ref in P['aia_references']:
                if ref['case']!=case['id'] or ref['panel']!=pid:continue
                with np.load(ref['predictions']) as z:expected=z[ref['prefix']+'__'+aid]
                np.testing.assert_allclose(value,expected,atol=1e-10,rtol=0)
                np.testing.assert_array_equal(abs(value-s.times)<=.5,abs(expected-s.times)<=.5);checks+=1
    fp=folder/'予測.npz';np.savez_compressed(fp,**pred)
    result=dict(case=case['id'],panels=pids,C_sha256=sha(case['path']),rows=rows,reference_checks=checks,predictions=str(fp),predictions_sha256=sha(fp))
    save(meta,result);return result

def mia_job(p,panel,aid):
    folder=HERE/'mia'/panel['id']/aid;folder.mkdir(parents=True,exist_ok=True);meta=folder/'結果.json'
    if meta.exists():
        r=load(meta);assert sha(r['prediction'])==r['prediction_sha256'];return r
    if aid=='M04':src=Path(p['mia4']);assets=[x for x in src.parent.glob('*.py') if x!=src]
    else:
        a=next(x for x in p['attacks'] if x['id']==aid);src=Path(a['local_path']);assets=[src.parent/x['name'] for x in a['assets']]
    script=Path(copy(src,folder/src.name))
    for asset in assets:copy(asset,folder/asset.name)
    save(folder/'targets.json',[dict(target_k=c['target_id'],c_path=c['path'],aia_challenge_path=p['panels'][0]['challenge']) for c in p['cases']])
    cmd=[str(PYTHON),'-B',str(script),'--b',panel['b'],'--a-bg',p['a_bg'],'--targets',str(folder/'targets.json')]
    with (folder/'実行.log').open('w') as log:r=subprocess.run(cmd,cwd=folder,stdout=log,stderr=subprocess.STDOUT)
    fp=folder/('F_mia_'+script.stem+'.csv');assert r.returncode==0 and fp.exists(),str(folder)
    d=read(fp);t=read(panel['truth']);assert len(d)==1049 and d.record_id.is_unique and set(d.record_id)==set(t.record_id)
    assert set(d.columns)=={'record_id'}|{str(c['target_id']) for c in p['cases']}
    sys.path.insert(0,str(Path(p['kit'])/'starter/src'))
    from pwscup2026.scoring.attack_score import _tpr_at_fpr
    rows=[];checks=0
    for c in p['cases']:
        pred=d.set_index('record_id').loc[t.record_id,str(c['target_id'])].to_numpy(float)
        assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all()
        tpr,auc=_tpr_at_fpr(pred,t.is_member.to_numpy(int),.01)
        rows.append(dict(case=c['id'],panel=panel['id'],attack=aid,TPR_at_FPR01=tpr,AUC=auc))
        for ref in p['mia_references']:
            if (ref['case'],ref['panel'],ref['attack'])!=(c['id'],panel['id'],aid):continue
            original=read(ref['prediction']);col=next(x for x in original.columns if x!='record_id')
            expected=original.set_index('record_id').loc[t.record_id,col].to_numpy(float)
            np.testing.assert_allclose(pred,expected,atol=1e-10,rtol=0);checks+=1
    result=dict(panel=panel['id'],attack=aid,rows=rows,reference_checks=checks,prediction=str(fp),prediction_sha256=sha(fp),command=cmd)
    save(meta,result);return result

def aggregate(p,ar,mr,utilities):
    aia=[x for r in ar for x in r['rows']];mia=[x for r in mr for x in r['rows']]
    assert len(aia)==960 and len(mia)==192
    assert len({(r['case'],r['panel'],r['attack']) for r in aia})==960
    assert len({(r['case'],r['panel'],r['attack']) for r in mia})==192
    cm={c['id']:c for c in p['cases']};pm={x['id']:x for x in p['panels']};mm={x['id']:x for x in p['mia_panels']}
    attack_names={a['id']:a['name'] for a in p['attacks']};attack_names['M04']='seo-04a-多重縮小生成比MIA'
    snapshots=[];decisions=[];details=[]
    for c in p['cases']:
        for panel in p['panels']:
            values=[r['AIA'] for r in aia if r['case']==c['id'] and r['panel']==panel['id']]
            before=[r['AIA'] for r in aia if r['case']=='base_'+c['base'] and r['panel']==panel['id']]
            snapshots.append(dict(case=c['id'],panel=panel['id'],set_id=panel['set_id'],n=panel['n'],
                minimum=min(values),median=float(np.median(values)),maximum=max(values),baseline_max=max(before),delta=max(values)-max(before)))
        if not c['ops']:continue
        for group in ['first','long']:
            hold=next(x for x in snapshots if x['case']==c['id'] and x['panel']==group+'_holdout')
            small=[x for x in snapshots if x['case']==c['id'] and x['set_id']==group and x['n']==163]
            bad_mia=[]
            for r in mia:
                if r['case']!=c['id'] or mm[r['panel']]['set_id']!=group:continue
                original=next(z for z in mia if z['case']=='base_'+c['base'] and z['panel']==r['panel'] and z['attack']==r['attack'])
                delta=r['TPR_at_FPR01']-original['TPR_at_FPR01']
                if delta>EPS:bad_mia.append(dict(panel=r['panel'],attack=r['attack'],before=original['TPR_at_FPR01'],after=r['TPR_at_FPR01'],delta=delta))
            train=max(x['maximum'] for x in snapshots if x['case']==c['id'] and x['set_id']=='train')
            train_before=max(x['baseline_max'] for x in snapshots if x['case']==c['id'] and x['set_id']=='train')
            uok=utilities[c['id']]['utility']>=utilities['base_'+c['base']]['utility']-EPS
            row=dict(case=c['id'],set_id=group,U=utilities[c['id']]['utility'],U_ok=uok,train_max=train,train_before=train_before,
                train_ok=train<train_before-EPS,hold_before=hold['baseline_max'],hold_after=hold['maximum'],hold_ok=hold['delta']<-EPS,
                small_ok=all(x['delta']<=EPS for x in small),small_worse_count=sum(x['delta']>EPS for x in small),
                mia_ok=not bad_mia,mia_worse_count=len(bad_mia))
            row['pass']=all(row[x] for x in ['U_ok','train_ok','hold_ok','small_ok','mia_ok']);decisions.append(row)
            for x in small:
                if x['delta']>EPS:details.append(dict(case=c['id'],set_id=group,test='AIA',panel=x['panel'],attack='12手法の最大値',before=x['baseline_max'],after=x['maximum'],delta=x['delta']))
            for x in bad_mia:details.append(dict(case=c['id'],set_id=group,test='MIA',**x))
    share=HERE/'チーム共有';share.mkdir(exist_ok=True)
    out(share/'判定一覧.csv',[dict(候補=cm[r['case']]['label'],正式名=cm[r['case']]['name'],元の案=cm[r['case']]['base'],交換回数=cm[r['case']]['swaps'],確認セット=SET_NAMES[r['set_id']],**{k:v for k,v in r.items() if k not in ['case','set_id']}) for r in decisions])
    out(share/'AIA_パネル別の要約.csv',[dict(候補=cm[r['case']]['label'],正式名=cm[r['case']]['name'],確認セット=SET_NAMES[r['set_id']],パネル=pm[r['panel']]['label'],**r) for r in snapshots])
    out(share/'AIA_全12手法の詳細.csv',[dict(候補=cm[r['case']]['label'],正式名=cm[r['case']]['name'],確認セット=SET_NAMES[pm[r['panel']]['set_id']],パネル=pm[r['panel']]['label'],手法=attack_names[r['attack']],**r) for r in aia])
    out(share/'MIA_全4手法の詳細.csv',[dict(候補=cm[r['case']]['label'],正式名=cm[r['case']]['name'],確認セット=SET_NAMES[mm[r['panel']]['set_id']],パネル=mm[r['panel']]['label'],手法=attack_names[r['attack']],**r) for r in mia])
    out(share/'悪化した箇所.csv',[dict(候補=cm[r['case']]['label'],正式名=cm[r['case']]['name'],確認セット=SET_NAMES[r['set_id']],**r) for r in details])
    baseline_refs=[]
    for panel in p['panels']:
        s=scorer(panel['id']);s.shared_c={};s.cost=None
        baseline_refs.append(dict(panel=panel['id'],constant10=s.metrics(np.full(2*panel['n'],10.))['AIA'],background1NN=s.metrics(s.predict('A12',p['a_bg']))['AIA']))
    bundle=dict(cases=p['cases'],panels=p['panels'],mia_panels=p['mia_panels'],aia=aia,mia=mia,snapshots=snapshots,decisions=decisions,violations=details,utilities=utilities,baselines=baseline_refs)
    save(HERE/'比較集計.json',bundle)
    save(HERE/'検証結果.json',dict(passed=True,cases=8,new_AIA_evaluations=960,new_MIA_evaluations=192,
        AIA_reference_prediction_checks=sum(r['reference_checks'] for r in ar),MIA_reference_prediction_checks=sum(r['reference_checks'] for r in mr),
        all_panels_same_for_all_candidates=True,previous_C_preserved=True,official_anonymity_predicted=False))
    return bundle

def main():
    p=prepare();init(p);start=time.time();ar=[];mr=[]
    utilities={};s=scorer('dev1')
    for c in p['cases']:utilities[c['id']]=s.full_score(read(c['path']),s.b,s.ref,s.cfg)
    save(HERE/'有用性.json',utilities)
    groups=[['dev1','dev2'],['first_holdout','first_small1'],['first_small2','first_small3'],['long_holdout','long_small1'],['long_small2','long_small3']]
    with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'),initializer=init,initargs=(p,)) as pool:
        for base in ['Zhang11','Zhang04']:
            fs=[pool.submit(aia_job,c,ids) for c in p['cases'] if c['base']==base for ids in groups]
            for f in as_completed(fs):
                ar.append(f.result());say(f'AIA計算 {len(ar)}/40件（1件＝候補1つ×パネル2組×12攻撃）')
    with ThreadPoolExecutor(max_workers=16) as pool:
        fs=[pool.submit(mia_job,p,panel,aid) for panel in p['mia_panels'] for aid in ['A04','A05','A06','M04']]
        for f in as_completed(fs):mr.append(f.result());say(f'MIA計算 {len(mr)}/24件（1件＝候補8つ×1組×1攻撃）')
    result=aggregate(p,ar,mr,utilities)
    for f,h in p['hashes'].items():assert sha(f)==h,f
    say(f'全比較完了 {time.time()-start:.1f}秒')
    for r in result['decisions']:say(f'{r["case"]} / {r["set_id"]}: 合格={r["pass"]} / AIA悪化{r["small_worse_count"]}組 / MIA悪化{r["mia_worse_count"]}件')

if __name__=='__main__':main()
