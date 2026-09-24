from common import *
from 評価 import metrics,truth,CFG
import math,yaml
from concurrent.futures import ProcessPoolExecutor,as_completed
from pwscup2026.kit.selfscore import self_score,_canonicalize_c
from pwscup2026.scoring import utility as official_u

P=plan()
ATKS={a['id']:a for a in P['attacks']}
PANELS=[p for p in P['panels'] if p['id'] in P['search']['panels']]

def utility_context():
    dist=Path(P['b']).parent;bself=read(dist/'B_self.csv');b=read(P['b']);r=load(dist/'utility_ref.json')
    ref=official_u.UtilityReference(mu=np.array(r['mu']),sigma=np.array(r['sigma']),band=r['band'],b_is_rare=bself.is_rare.to_numpy(bool),
        km_signal_D=r['km_signal_D'],km_floor=r['km_floor'],a_bg=read(P['a_bg']))
    return b,ref

class Attack:
    def __init__(self,base,need_background=True):
        self.mods={i:mod('search_'+i,RUN/ATKS[i]['path']) for i in P['search']['attacks']}
        self.tt=self.mods['A03'];self.core=self.mods['A06'];self.relaxed=self.mods['A07']
        self.q=[self.core.read_table(Path(p['challenge']),'challenge') for p in PANELS]
        self.bg=self.core.read_table(Path(P['a_bg']),'A');self.transform=self.core.BackgroundTransform(self.bg)
        self.ttbg=self.tt.read_table(Path(P['a_bg']),'background')[0];self.scale=self.tt.scale_from_background(self.ttbg)
        self.ttq=pd.concat([self.tt.read_table(Path(p['challenge']),'challenge')[0] for p in PANELS],ignore_index=True)
        self.truth=[truth(p) for p in PANELS]
        self.baseline={}
        if need_background:
            for aid in P['search']['attacks']:
                self.baseline[aid]=[]
                for p,t in zip(PANELS,self.truth):
                    preds=[predictions('new387',f'BG{i:02d}',ATKS[aid],p) for i in range(1,11)]
                    zs=[metrics(pr,t,'weighted',True)[0] for pr in preds]
                    hit=np.abs(np.array(preds)-t.time.to_numpy())<=.5;m=t.label.eq('member').to_numpy()
                    self.baseline[aid].append(dict(pm=float(np.mean([z['pm'] for z in zs])),R=float(np.mean([z['R_all'] for z in zs])),
                        groups=[float(hit[:,m&t.stratum.eq(s).to_numpy()].mean()) for s in range(4)]))
    def predictions(self,c):
        tc=c[COLS].copy()
        for col in COLS:
            if col!='prefecture':tc[col]=pd.to_numeric(tc[col].map(lambda x:format(x,'.17g'))).to_numpy(float)
        tc=tc.sort_values(COLS,kind='stable').reset_index(drop=True)
        pred=self.tt.predict(tc,self.ttq,self.ttbg,self.scale)[0]
        pred=np.array([float(format(v,self.tt.FLOAT_FORMAT)) for v in pred])
        sizes=np.cumsum([len(q) for q in self.q])[:-1]
        result={'A03':list(np.split(pred,sizes))}
        result['A06']=[self.core.aia_times(q,tc,self.transform)[0] for q in self.q]
        result['A07']=[self.relaxed.relaxed_times(q,tc,self.transform,2)[0] for q in self.q]
        return result
    def evaluate(self,c):
        preds=self.predictions(c);rows=[];all_lift=[];group_lift=[];rlift=[]
        for aid,prs in preds.items():
            for j,(pr,t) in enumerate(zip(prs,self.truth)):
                z,h=metrics(pr,t,'weighted',True);b=self.baseline[aid][j];m=t.label.eq('member').to_numpy()
                lift=z['pm']-b['pm'];gl=[float(h[m&t.stratum.eq(s).to_numpy()].mean())-b['groups'][s] for s in range(4)]
                dl=z['R_all']-b['R'];all_lift.append(max(0,lift));group_lift.extend(max(0,x) for x in gl);rlift.append(max(0,dl))
                rows.append(dict(attack=aid,panel=PANELS[j]['id'],weighted_member_lift=lift,group_lifts=gl,weighted_R_lift=dl))
        # Fixed before running: protect the weighted overall target and the weakest stratum,
        # while retaining a control-adjusted diagnostic. This is not official phi or Anon.
        energy=max(all_lift)+.25*max(group_lift)+.25*max(rlift)+.1*float(np.mean(all_lift))
        return dict(energy=energy,member_lift_max=max(all_lift),group_lift_max=max(group_lift),R_lift_max=max(rlift),records=rows)

def pairs(c):
    ps=[]
    for key,idx in c.groupby(['onset','death']).groups.items():
        if key==(0,0):continue
        idx=np.asarray(idx);i,j=np.triu_indices(len(idx),1);ps.extend(zip(idx[i],idx[j]))
    return np.array(ps)

def verify():
    b,ref=utility_context();records=[]
    for v in P['variants'][:3]:
        c=read(v['path']);u=self_score(c,b,ref,CFG)
        np.testing.assert_allclose(u['utility'],v['utility']['U'],atol=1e-12,rtol=0)
        attack=Attack(c,False);pred=attack.predictions(c)
        for aid,prs in pred.items():
            for p,x in zip(PANELS,prs):
                expected=predictions('new387',v['label'],ATKS[aid],p);times=truth(p).time.to_numpy()
                np.testing.assert_allclose(x,expected,atol=1e-10,rtol=0)
                np.testing.assert_array_equal(abs(x-times)<=.5,abs(expected-times)<=.5)
        cand=c.copy();i,j=pairs(c)[23];cand.loc[[i,j],'time']=cand.loc[[j,i],'time'].to_numpy()
        dst=HERE/'verification'/v['label'];out(dst/'swap.csv',cand)
        u2=self_score(cand,b,ref,CFG)
        for k in ['U_gen','U_rare']:np.testing.assert_allclose(u['facets'][k],u2['facets'][k],atol=1e-12,rtol=0)
        cp=official_u._prepare(_canonicalize_c(cand));bp=official_u._prepare(b)
        for key,value in [('U_spec',official_u._score_u_spec(cp,bp)[0]),('U_valid',official_u._score_u_valid(cp,bp,ref,CFG)[0])]:
            np.testing.assert_allclose(value,u2['facets'][key],atol=1e-12,rtol=0)
        pred2=attack.predictions(cand)
        for aid,prs in pred2.items():
            attack_job('verification',v['label'],ATKS[aid],PANELS,str(dst/'swap.csv'))
            for p,x in zip(PANELS,prs):
                expected=predictions('verification',v['label'],ATKS[aid],p);times=truth(p).time.to_numpy()
                np.testing.assert_allclose(x,expected,atol=1e-10,rtol=0)
                np.testing.assert_array_equal(abs(x-times)<=.5,abs(expected-times)<=.5)
        records.append(dict(base=v['label'],U=u['utility'],prediction_checks=18,utility_facets_invariant=['U_gen','U_rare']))
    save(HERE/'verification/検証.json',dict(passed=True,records=records));print('search implementation verified',flush=True)

def chain(job):
    name,rep=job;dst=HERE/'search'/f'{name}_{rep}';meta=dst/'結果.json'
    if meta.exists():return load(meta)
    v=next(v for v in P['variants'] if v['label']==name);base=read(v['path']);b,ref=utility_context();bp=official_u._prepare(b)
    ubase=self_score(base,b,ref,CFG);floor=ubase['utility'];attack=Attack(base);ps=pairs(base)
    seed=2026092600+P['search']['bases'].index(name)*10+rep;rng=np.random.default_rng(seed)
    cur=base.copy();cur_a=attack.evaluate(cur);best=cur.copy();best_a=cur_a;best_u=ubase;best_step=0;logs=[]
    initial_a=cur_a;save(dst/'初期値.json',cur_a)
    steps=P['search']['steps'];start=time.monotonic()
    for step in range(1,steps+1):
        i,j=map(int,ps[rng.integers(len(ps))]);cand=cur.copy();cand.loc[[i,j],'time']=cand.loc[[j,i],'time'].to_numpy()
        cp=official_u._prepare(_canonicalize_c(cand));fac=ubase['facets'].copy()
        fac['U_spec']=official_u._score_u_spec(cp,bp)[0];fac['U_valid']=official_u._score_u_valid(cp,bp,ref,CFG)[0]
        u={'utility':min(fac.values()),'facets':fac};a=None;accept=False;reason='utility_reject'
        temp=.02*(.001/.02)**((step-1)/(steps-1))
        if u['utility']>=floor-1e-12:
            a=attack.evaluate(cand);delta=a['energy']-cur_a['energy'];accept=delta<=0 or rng.random()<math.exp(-delta/temp)
            reason='accepted' if accept else 'attack_reject'
            if accept:cur,cur_a=cand,a
            key=lambda c,u,a:(a['energy'],int((c.time!=base.time).sum()),-u['utility'])
            if key(cand,u,a)<key(best,best_u,best_a):best,best_u,best_a,best_step=cand.copy(),u,a,step
        logs.append(dict(step=step,i=i,j=j,temperature=temp,U=u['utility'],energy=None if a is None else a['energy'],accepted=accept,reason=reason,best_energy=best_a['energy']))
        if step%10==0:
            save(dst/'進捗.json',dict(step=step,total=steps,best_energy=best_a['energy'],seconds=time.monotonic()-start))
            print(name,rep,step,'best',round(best_a['energy'],6),flush=True)
    assert base.drop(columns='time').equals(best.drop(columns='time'))
    for _,idx in base.groupby(['onset','death']).groups.items():np.testing.assert_array_equal(np.sort(base.loc[idx,'time']),np.sort(best.loc[idx,'time']))
    out(dst/'C.csv',best);saved=read(dst/'C.csv');full=self_score(saved,b,ref,CFG)
    assert full['utility']>=floor-1e-12
    for key in full['facets']:np.testing.assert_allclose(full['facets'][key],best_u['facets'][key],atol=1e-12,rtol=0)
    check=attack.evaluate(saved);np.testing.assert_allclose(check['energy'],best_a['energy'],atol=1e-12,rtol=0)
    save(dst/'有用性.json',full);save(dst/'攻撃増分.json',check);out(dst/'探索履歴.csv',logs)
    result=dict(label=f'{name}_探索候補',base=name,chain=rep,seed=seed,path=str(dst/'C.csv'),sha256=sha(dst/'C.csv'),initial_energy=initial_a['energy'],energy=check['energy'],
        U=full['utility'],changed_rows=int((base.time!=saved.time).sum()),best_step=best_step,seconds=time.monotonic()-start)
    save(meta,result);return result

def run():
    assert load(HERE/'verification/検証.json')['passed']
    # Freeze implementation and settings before any chain or final-panel attack is run.
    freeze=HERE/'search/探索条件固定.json'
    state=dict(settings=P['search'],objective='max positive weighted target lift + .25 max group lift + .25 max weighted R lift + .1 mean positive weighted target lift',
               source_hash=sha(__file__),final_results_used=False)
    if freeze.exists():assert load(freeze)==state
    else:save(freeze,state)
    jobs=[(b,i) for b in P['search']['bases'] for i in range(1,P['search']['chains']+1)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        results=[f.result() for f in as_completed([pool.submit(chain,j) for j in jobs])]
    selected=[]
    for b in P['search']['bases']:
        z=min([r for r in results if r['base']==b],key=lambda x:(x['energy'],x['changed_rows'],-x['U'],x['chain']))
        selected.append(z)
    save(HERE/'search/固定候補.json',dict(variants=selected,all_chains=results,selected_without_final_results=True,frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    print(json.dumps(selected,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    verify() if args.verify else run()
