"""Additional fixed-panel comparison, all seven C files unchanged."""
from 採点共通 import *
from パネル生成 import prepare,PREVIOUS
import 評価部品 as parts
HERE=Path(__file__).resolve().parent
BASE={'Zhang04':'C02','Zhang11':'C01','Seo03':'C03'}
def init(p):parts.HERE=HERE;parts.init(p)
def aia_batch(cases,pid):return [parts.aia_job(c,[pid]) for c in cases]
def baseline(job):
    p,q=job;s=Scorer(dict(p,panel=q))
    return dict(panel=q['id'],constant10=s.metrics(np.full(len(s.q),10.)),background1NN=s.metrics(s.predict('A12',p['a_bg'])))
def stats(v):
    return dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),minimum=float(np.min(v)),maximum=float(np.max(v)),p90=float(np.quantile(v,.90)))
def delta_stats(a,b):
    v=np.array(a)-np.array(b)
    return dict(mean_delta=float(v.mean()),better=int((v<-1e-12).sum()),ties=int((abs(v)<=1e-12).sum()),worse=int((v>1e-12).sum()),max_increase=float(v.max()),max_decrease=float(v.min()))
def simple_constant(panel):
    t=read(panel['truth']);m=t.label.eq('member');hit=abs(t.time.to_numpy()-10)<=.5
    pm=np.average(hit[m],weights=t.loc[m,'weight_163']);pc=np.average(hit[~m],weights=t.loc[~m,'weight_163'])
    return float(np.clip((pm-pc)/max(1-pc,1e-9),0,1))

def aggregate(p,ar,mr,baselines,seconds):
    aia=[x for r in ar for x in r['rows']];mia=[x for r in mr for x in r['rows']]
    assert len(aia)==3360 and len(mia)==560
    assert len({(r['case'],r['panel'],r['attack']) for r in aia})==3360
    assert len({(r['case'],r['panel'],r['attack']) for r in mia})==560
    old=load(p['previous_result']);aa=old['aia']+aia;mm=old['mia']+mia;qs=old['panels']+p['panels'];ms=old['mia_panels']+p['mia_panels']
    const={b['panel']:b['constant10']['AIA'] for b in old['baselines']+baselines}
    for q in qs:
        if q['id'] not in const:const[q['id']]=simple_constant(q)
    assert len(qs)==52 and len(ms)==29
    summary=[];panelrows=[];mia_summary=[];miarows=[];s=Scorer(dict(p,panel=p['panels'][0]));utility={}
    for c in p['cases']:
        utility[c['id']]=s.full_score(read(c['path']),s.b,s.ref,s.cfg)
        prev=next(x for x in old['cases'] if x['id']==c['id'])
        assert abs(utility[c['id']]['utility']-prev['utility']['utility'])<1e-12
        c['utility']=utility[c['id']]
        for q in qs:
            v=[r['AIA'] for r in aa if r['case']==c['id'] and r['panel']==q['id']]
            b=[r['AIA'] for r in aa if r['case']==BASE[c['base']] and r['panel']==q['id']];assert len(v)==len(b)==12
            panelrows.append(dict(case=c['id'],name=c['name'],base=c['base'],panel=q['id'],n=q['n'],set='new' if q in p['panels'] else 'old',maximum=max(v),attack_median=float(np.median(v)),attack_minimum=min(v),constant10=const[q['id']],delta_constant=max(v)-const[q['id']],constant_inclusive_excess=max(0,max(v)-const[q['id']]),base_maximum=max(b),delta=max(v)-max(b),worst_attacks=[r['attack'] for r in aa if r['case']==c['id'] and r['panel']==q['id'] and abs(r['AIA']-max(v))<1e-12]))
        for q in ms:
            vr=[r for r in mm if r['case']==c['id'] and r['panel']==q['id']];br=[r for r in mm if r['case']==BASE[c['base']] and r['panel']==q['id']];assert len(vr)==len(br)==4
            v=[r['TPR_at_FPR01'] for r in vr];b=[r['TPR_at_FPR01'] for r in br];bm={r['attack']:r['TPR_at_FPR01'] for r in br}
            miarows.append(dict(case=c['id'],name=c['name'],base=c['base'],panel=q['id'],set='new' if q in p['mia_panels'] else 'old',maximum=max(v),base_maximum=max(b),delta=max(v)-max(b),worse_attacks=sum(r['TPR_at_FPR01']>bm[r['attack']]+1e-12 for r in vr),better_attacks=sum(r['TPR_at_FPR01']<bm[r['attack']]-1e-12 for r in vr)))
        for group in ['old','new','all']:
            for n in [387,163]:
                rr=[r for r in panelrows if r['case']==c['id'] and r['n']==n and (group=='all' or r['set']==group)]
                v=[r['maximum'] for r in rr];b=[r['base_maximum'] for r in rr]
                summary.append(dict(case=c['id'],name=c['name'],base=c['base'],group=group,target_n=n,U=utility[c['id']]['utility'],**stats(v),**delta_stats(v,b),constant_mean=float(np.mean([r['constant10'] for r in rr])),mean_constant_excess=float(np.mean([r['constant_inclusive_excess'] for r in rr]))))
            rr=[r for r in miarows if r['case']==c['id'] and (group=='all' or r['set']==group)];v=[r['maximum'] for r in rr];b=[r['base_maximum'] for r in rr]
            mia_summary.append(dict(case=c['id'],name=c['name'],base=c['base'],group=group,**stats(v),**delta_stats(v,b),worse_attacks=sum(r['worse_attacks'] for r in rr),better_attacks=sum(r['better_attacks'] for r in rr),attack_comparisons=4*len(rr)))
    assert all(sha(f)==h for f,h in p['hashes'].items())
    # Cross-check summary deltas via separately grouped records.
    for r in summary:
        b=next(x for x in summary if x['case']==BASE[r['base']] and x['group']==r['group'] and x['target_n']==r['target_n'])
        assert abs((r['mean']-b['mean'])-r['mean_delta'])<1e-12
        assert r['better']+r['ties']+r['worse']==r['n']
    result=dict(cases=p['cases'],new_panels=p['panels'],new_mia_panels=p['mia_panels'],attacks=p['attacks'],new_baselines=baselines,aia=aia,mia=mia,summary=summary,mia_summary=mia_summary,panel_results=panelrows,mia_panel_results=miarows,seconds=seconds,phi=p['phi'])
    save(HERE/'比較結果.json',result)
    out(HERE/'集計/AIA_要約.csv',summary);out(HERE/'集計/MIA_要約.csv',mia_summary);out(HERE/'集計/AIA_パネル別.csv',panelrows);out(HERE/'集計/MIA_パネル別.csv',miarows)
    out(HERE/'集計/追加AIA_全手法.csv',aia);out(HERE/'集計/追加MIA_全手法.csv',mia)
    save(HERE/'検証結果.json',dict(passed=True,new_AIA_scores=3360,new_MIA_scores=560,existing_AIA_scores=len(old['aia']),existing_MIA_scores=len(old['mia']),U_checks=7,unchanged_inputs=True,seconds=seconds))
    print(json.dumps([r for r in summary if r['group']=='new' and r['target_n']==387],ensure_ascii=False,indent=2),flush=True)

def run(p):
    start=time.monotonic();ctx=multiprocessing.get_context('spawn');ar=[];mr=[];done=0;total=4000
    with ProcessPoolExecutor(max_workers=16,mp_context=ctx) as pool:
        bs=[]
        for f in as_completed([pool.submit(baseline,(p,q)) for q in p['panels']]):bs.append(f.result());done+=2;say(bar(done,total)+' 基準値')
    for base in p['base_order']:
        cases=[c for c in p['cases'] if c['base']==base]
        with ProcessPoolExecutor(max_workers=16,mp_context=ctx,initializer=init,initargs=(p,)) as pool:
            for f in as_completed([pool.submit(aia_batch,cases,q['id']) for q in p['panels']]):ar.extend(f.result());done+=12*len(cases);say(bar(done,total)+' AIA '+base)
        parts.HERE=HERE;batch=dict(p,cases=cases,batch_label=base)
        with ThreadPoolExecutor(max_workers=16) as pool:
            for f in as_completed([pool.submit(parts.mia_job,batch,q,a) for q in p['mia_panels'] for a in ['A04','A05','A06','M04']]):mr.append(f.result());done+=len(cases);say(bar(done,total)+' MIA '+base)
    assert done==total
    aggregate(p,ar,mr,bs,time.monotonic()-start)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    plan=prepare()
    if a.run or not a.prepare:
        with (HERE/'実行.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(plan)
