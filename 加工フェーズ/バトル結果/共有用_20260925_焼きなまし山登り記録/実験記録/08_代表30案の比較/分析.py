"""Saved candidate comparison; no new search, no official submission."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys, json, hashlib, shutil, importlib.util, time, argparse, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parent
OLD = R.parent/'0923/AIA増分評価と改善'
LATEST = R/'U低下許容_20分探索_20260924'
FAIR = R/'公平比較_初回合格案と長時間5案_20260924'
BASE_ORDER = ['Zhang04','Zhang11','Seo03']
def load(p): return json.loads(Path(p).read_text())
def read(p): return pd.read_csv(p, float_precision='round_trip')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(c): return hashlib.sha256(pd.util.hash_pandas_object(c,index=True).to_numpy().tobytes()).hexdigest()
def save(p, x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n'); t.replace(p)
def copy(a,b):
    b=Path(b);b.parent.mkdir(parents=True,exist_ok=True)
    if b.exists(): assert sha(a)==sha(b)
    else: shutil.copyfile(a,b)
    return str(b)
def module():
    spec=importlib.util.spec_from_file_location('shared_scoring',HERE/'採点共通.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
def rows(x): return x if isinstance(x,list) else x['candidates']
def difference(a,b):
    assert a.shape==b.shape and list(a)==list(b)
    eq=np.ones(a.shape,dtype=bool)
    for j,c in enumerate(a):
        if pd.api.types.is_numeric_dtype(a[c]): eq[:,j]=np.isclose(a[c],b[c],atol=1e-10,rtol=0,equal_nan=True)
        else: eq[:,j]=a[c].to_numpy()==b[c].to_numpy()
    return {'changed_rows':int((~eq).any(axis=1).sum()),'changed_cells':int((~eq).sum()),'attributes':[c for j,c in enumerate(a) if (~eq[:,j]).any()]}

def prepare():
    p=load(LATEST/'計画.json')
    plan={k:p[k] for k in ['kit','b','a_bg','attacks']}
    plan.update(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),workers=16,base_order=BASE_ORDER,phi=0.089047,new_search=False)
    plan['panels']=[]
    for i,(folder,label) in enumerate([
        ('15時間探索_2ベース_3パネル','初回実験の確認組'),
        ('15時間継続探索_時間配分方式_v2','長時間実験の確認組'),
        ('U低下許容_20分探索_20260924','20分実験の確認組')],1):
        z=next(x for x in load(R/folder/'計画.json')['panels'] if x['id']=='holdout')
        panel=dict(z,id=f'P{i}',label=label,source_experiment=folder)
        for col in ['challenge','truth']:panel[col]=copy(z[col],HERE/'inputs'/f'P{i}'/Path(z[col]).name)
        plan['panels'].append(panel)
    copy(FAIR/'採点共通.py',HERE/'採点共通.py')
    for filename in ['B.csv','B_self.csv','utility_ref.json']:
        dest=copy(Path(p['b']).with_name(filename),HERE/'inputs/utility'/filename)
        if filename=='B.csv': plan['b']=dest
    plan['a_bg']=copy(p['a_bg'],HERE/'inputs/A_bg.csv')
    cases=[];known={};orig={}
    def add(name,base,path,stage,order,swaps=0,ops=None,aliases=None,source_id='',note=''):
        c=read(path);k=key(c)
        aliases=(aliases or [])+[name]
        if k in known:
            known[k]['aliases']=list(dict.fromkeys(known[k]['aliases']+aliases))
            known[k]['origins']=list(dict.fromkeys(known[k]['origins']+[stage]))
            return
        case=dict(id=f'C{len(cases)+1:02d}',name=name,base=base,source=str(path),stage=stage,order=order,swaps=swaps,
                  ops=ops or [],aliases=aliases,origins=[stage],source_id=source_id,note=note,key=k)
        case.update(difference(orig[base],c))
        case['path']=copy(path,HERE/'C'/f'{case["id"]}.csv');case['sha256']=sha(case['path'])
        cases.append(case);known[k]=case
    sa=load(R/'SA1_3候補_追加AIA比較/候補の命名.json')
    for x in sa:
        orig[x['base']]=read(x['original'])
        add(x['base'],x['base'],x['original'],'元の加工案',0)
    for x in sa:add(x['name'],x['base'],x['candidate'],'timeのみの山登り（SA1）',3,x['accepted'],note='SA1という名称だが、実際の方法は山登り。')
    for sub,title,order in [('search','初期の焼きなまし',1),('search_strong','主要攻撃で焼きなまし',2)]:
        frozen=load(OLD/sub/'固定候補.json');selected={x['path'] for x in frozen['variants']}
        for path in sorted((OLD/sub).glob('*/結果.json')):
            x=load(path);hist=read(path.parent/'探索履歴.csv');best=x['best_step']
            trace=hist[(hist.step<best)&hist.accepted.astype(bool)]
            if best:trace=pd.concat([trace,hist[hist.step==best]])
            ops=[dict(attribute='time',i=int(z.i),j=int(z.j),row1=int(z.i)+1,row2=int(z.j)+1) for z in trace.itertuples()]
            reconstructed=orig[x['base']].copy()
            for op in ops:
                i,j=op['i'],op['j'];reconstructed.loc[[i,j],'time']=reconstructed.loc[[j,i],'time'].to_numpy()
            assert difference(reconstructed,read(x['path']))['changed_cells']==0,(sub,x['base'])
            name=f'{x["base"]}-焼きなまし{"初期" if sub=="search" else "主要"}{x["chain"]}'
            add(name,x['base'],x['path'],title,order,len(ops),ops,source_id=f'{sub}/{x["base"]}_{x["chain"]}',
                note='当時の代表候補' if x['path'] in selected else '当時の別試行の最良候補')
    pilot=R/'初回450候補_ベース順次_16並列'
    cp=read(pilot/'候補計画.csv')
    for x in read(pilot/'ベース別まとめ.csv').to_dict('records'):
        v=cp[cp.id==x['best_candidate']].iloc[0]
        op=dict(attribute=v.attribute,row1=int(v.row1),row2=int(v.row2),i=int(v.row1)-1,j=int(v.row2)-1)
        add(f'{v.base}-1回交換最良',v.base,v.path,'450候補テスト',4,1,[op],[v.id],v.id)
    for folder,stage,order in [('15時間探索_2ベース_3パネル','初回の山登り',5),('15時間継続探索_時間配分方式_v2','長時間の山登り',6)]:
        for x in rows(load(R/folder/'確認前の候補固定.json')):
            add(x['display_name'],x['base'],x['path'],stage,order,x['total_swaps'],x['ops'],source_id=x['id'])
    for x in rows(load(LATEST/'U低下案の追加確認_候補固定.json')):
        add('Zhang11-U低下参考案',x['base'],x['path'],'Uを少し下げる探索',7,x['total_swaps'],x['ops'],[x['id']],x['id'],note='今回の一覧で使う短い表示名。')
    cases.sort(key=lambda x:(BASE_ORDER.index(x['base']),x['order'],x['name']))
    plan['cases']=cases
    paths={plan['b'],plan['a_bg'],str(HERE/'採点共通.py'),str(Path(plan['b']).with_name('B_self.csv')),str(Path(plan['b']).with_name('utility_ref.json')),str(Path(plan['kit'])/'participant_data/scoring_config.yaml')}
    for z in cases:paths.update([z['source'],z['path']])
    for z in plan['panels']:paths.update([z['challenge'],z['truth']])
    for z in plan['attacks']:paths.update(str(f) for f in Path(z['local_path']).parent.glob('*.py'))
    plan['hashes']={f:sha(f) for f in sorted(paths)}
    save(HERE/'計画.json',plan)
    print(json.dumps([{k:c[k] for k in ['id','name','base','swaps','changed_rows','aliases']} for c in cases],ensure_ascii=False,indent=2),flush=True)
    print(f'{len(cases)} unique C x 3 panels x 12 attacks = {len(cases)*36}',flush=True)

def evaluate(job):
    plan,case,panel=job
    s=module().Scorer(dict(plan,panel=panel));path=HERE/'results'/case['id']/panel['id'];path.mkdir(parents=True,exist_ok=True)
    records=[];preds={}
    for a in plan['attacks']:
        pred=s.predict(a['id'],case['path']);preds[a['id']]=pred
        records.append(dict(case=case['id'],panel=panel['id'],attack=a['id'],**s.metrics(pred)))
    result=dict(case=case['id'],panel=panel['id'],aia=records)
    if panel['id']=='P1':result['utility']=s.full_score(read(case['path']),s.b,s.ref,s.cfg)
    np.savez_compressed(path/'予測.npz',**preds);save(path/'結果.json',result)
    return result

def baseline(job):
    plan,panel=job;s=module().Scorer(dict(plan,panel=panel))
    constant=s.metrics(np.full(len(s.q),10.))
    background=s.metrics(s.predict('A12',plan['a_bg']))
    t=s.truth;m=s.member
    records=[]
    for label,mask in [('member',m),('control',~m)]:
        records.append(dict(label=label,n=int(mask.sum()),weight_sum=float(t.loc[mask,'weight_163'].sum()),
                            onset=int(t.loc[mask,'onset'].sum()),mean_time=float(t.loc[mask,'time'].mean()),
                            time_ge_9_5=int((t.loc[mask,'time']>=9.5).sum()),
                            origins=t.loc[mask,'origin'].value_counts().to_dict(),
                            strata=t.loc[mask,'paired_member_stratum'].value_counts().sort_index().to_dict()))
    return dict(panel=panel['id'],constant10=constant,background1NN=background,composition=records)

def run():
    p=load(HERE/'計画.json');start=time.monotonic();out=[];done=0;total=len(p['cases'])*3
    ctx=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=3,mp_context=ctx) as pool:
        baselines=list(pool.map(baseline,[(p,q) for q in p['panels']]))
    for base in BASE_ORDER:
        jobs=[(p,c,q) for c in p['cases'] if c['base']==base for q in p['panels']]
        with ProcessPoolExecutor(max_workers=16,mp_context=ctx) as pool:
            for f in as_completed([pool.submit(evaluate,j) for j in jobs]):
                out.append(f.result());done+=1
                filled=int(done/total*24)
                print('['+'#'*filled+'-'*(24-filled)+f'] {done}/{total} ({100*done/total:.1f}%) {base}',flush=True)
    aggregate(p,out,baselines,time.monotonic()-start)

def aggregate(p,out,baselines,seconds):
    aia=[z for x in out for z in x['aia']];utilities={x['case']:x['utility'] for x in out if 'utility'in x}
    # Check fresh results against matched earlier full predictions, never using old scores as inputs.
    oldp=load(FAIR/'計画.json');checks=0;maxdiff=0.
    for old in oldp['cases']:
        found=[c for c in p['cases'] if c['key']==old['key']]
        if not found:continue
        c=found[0]
        for newpid,oldpid in [('P1','first_holdout'),('P2','long_holdout')]:
            files=list((FAIR/'aia'/old['id']).glob(f'{oldpid}_*/予測.npz'));assert len(files)==1
            past=np.load(files[0]);fresh=np.load(HERE/'results'/c['id']/newpid/'予測.npz')
            panel=next(x for x in p['panels'] if x['id']==newpid);times=read(panel['truth']).time.to_numpy()
            for a in p['attacks']:
                x=fresh[a['id']];y=past[f'{oldpid}__{a["id"]}']
                np.testing.assert_allclose(x,y,atol=1e-10,rtol=0);np.testing.assert_array_equal(abs(x-times)<=.5,abs(y-times)<=.5)
                maxdiff=max(maxdiff,float(abs(x-y).max()));checks+=1
    maxes={(c['id'],q['id']):max(z['AIA'] for z in aia if z['case']==c['id'] and z['panel']==q['id']) for c in p['cases'] for q in p['panels']}
    for c in p['cases']:
        c['utility']=utilities[c['id']];c['max_aia']=[maxes[c['id'],q['id']] for q in p['panels']]
        c['max_3']=max(c['max_aia']);c['median_3']=float(np.median(c['max_aia']));c['min_3']=min(c['max_aia'])
        c['worst_attacks']=[' / '.join(a['name'] for a in p['attacks'] if any(z['case']==c['id'] and z['panel']==q['id'] and z['attack']==a['id'] and abs(z['AIA']-maxes[c['id'],q['id']])<1e-12 for z in aia)) for q in p['panels']]
        assert len([z for z in aia if z['case']==c['id']])==36
    bmap={c['base']:c for c in p['cases'] if c['stage']=='元の加工案'}
    for c in p['cases']:
        b=bmap[c['base']];c['delta_aia']=[x-y for x,y in zip(c['max_aia'],b['max_aia'])]
        c['delta_U']=c['utility']['utility']-b['utility']['utility'];c['delta_max']=c['max_3']-b['max_3']
        c['worse_panels']=sum(x>1e-12 for x in c['delta_aia'])
    assert all(sha(f)==h for f,h in p['hashes'].items())
    result=dict(created=p['created'],cases=p['cases'],panels=p['panels'],attacks=p['attacks'],baselines=baselines,aia=aia,phi=p['phi'],seconds=seconds)
    save(HERE/'比較結果.json',result)
    save(HERE/'検証結果.json',dict(passed=True,cases=len(p['cases']),aia_scores=len(aia),panels=3,matched_prediction_checks=checks,max_prediction_difference=maxdiff,input_hashes_unchanged=True,seconds=seconds))
    print(json.dumps({'passed':True,'cases':len(p['cases']),'aia_scores':len(aia),'matched_checks':checks,'seconds':seconds,'constants':[b['constant10']['AIA'] for b in baselines]},ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.prepare:prepare()
    if a.run:run()
