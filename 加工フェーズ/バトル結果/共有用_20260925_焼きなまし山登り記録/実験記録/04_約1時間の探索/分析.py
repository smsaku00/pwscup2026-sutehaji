"""Resumable, standalone local hill climbing. No network or language-model API calls."""
from 採点共通 import *
from 準備 import prepare, verify
from concurrent.futures import wait, FIRST_COMPLETED
import signal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PYTHON=ROOT/'スターターキット/初版/starter/.venv/bin/python'
EPS=1e-12
SCORERS={}; PLAN=None; ORIGINAL={}

def initialize(p):
    global PLAN
    PLAN=p

def scorer(pid):
    if pid not in SCORERS:
        panel=next(x for x in PLAN['panels'] if x['id']==pid)
        SCORERS[pid]=Scorer(dict(PLAN,panel=panel))
    return SCORERS[pid]

def original(base):
    if base not in ORIGINAL: ORIGINAL[base]=read(next(v['path'] for v in PLAN['variants'] if v['label']==base))
    return ORIGINAL[base]

def replay(base,ops):
    c=original(base).copy()
    for op in ops:
        attr,i,j=op['attribute'],op['i'],op['j']
        if attr=='time':
            assert tuple(c.loc[i,['onset','death']])==tuple(c.loc[j,['onset','death']])!=(0,0)
        c.loc[[i,j],attr]=c.loc[[j,i],attr].to_numpy()
    return c

def content_key(c):
    return hashlib.sha256(pd.util.hash_pandas_object(c,index=True).to_numpy().tobytes()).hexdigest()

def normalize_ops(base,ops):
    """After deleting a swap, recompute later before-values and remove no-op operations."""
    stacks={};removed=set()
    for index,op in enumerate(ops):
        stack=stacks.setdefault(op['attribute'],[]);pair=tuple(sorted([op['i'],op['j']]))
        if stack and stack[-1][1]==pair:
            old_index,_=stack.pop();removed.update([old_index,index])
        else:stack.append((index,pair))
    ops=[op for index,op in enumerate(ops) if index not in removed]
    c=original(base).copy();clean=[]
    for old in ops:
        attr,i,j=old['attribute'],old['i'],old['j']
        if c.at[i,attr]==c.at[j,attr]:continue
        clean.append(dict(attribute=attr,i=i,j=j,row1=i+1,row2=j+1,old1=float(c.at[i,attr]),old2=float(c.at[j,attr])))
        c.loc[[i,j],attr]=c.loc[[j,i],attr].to_numpy()
    return clean

def work(job):
    """U first, then a safe lower bound on max AIA. Partial results never rank candidates."""
    folder=HERE/'results'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    result_path=folder/'結果.json'
    if result_path.exists():
        r=load(result_path);assert r['job']==job
        if r.get('complete'): assert sha(r['path'])==r['C_sha256'] and sha(r['predictions'])==r['predictions_sha256']
        return r
    start=time.monotonic()
    c=read(job['input_path']) if job.get('input_path') else replay(job['base'],job['ops'])
    assert content_key(c)==job['key']
    cp=folder/'C.csv';out(cp,c)
    s=scorer('dev1');u=s.full_score(read(cp),s.b,s.ref,s.cfg)
    assert np.isfinite(u['utility']) and np.isfinite(list(u['facets'].values())).all()
    r=dict(id=job['id'],job=job,base=job['base'],ops=job['ops'],total_swaps=len(job['ops']),
        key=job['key'],utility=u,complete=False,attacks={},attack_calls=0,path=str(cp),C_sha256=sha(cp))
    if u['utility']<job.get('floor',0)-EPS:
        r['outcome']='U不足'
    else:
        predictions={};stop=False
        for pid in job['panels']:
            ss=scorer(pid);ss.shared_c={};ss.cost=None
        order=job.get('order') or [(pid,a['id']) for pid in job['panels'] for a in PLAN['attacks']]
        assert set(map(tuple,order))=={(pid,a['id']) for pid in job['panels'] for a in PLAN['attacks']}
        for pid,aid in order:
            ss=scorer(pid);pred=ss.predict(aid,cp);metric=ss.metrics(pred)
            r['attacks'].setdefault(pid,{})[aid]=metric;r['attack_calls']+=1
            predictions[pid+'__'+aid]=pred
            # Strict improvement cannot happen if one component already fails it.
            if job.get('threshold') is not None and metric['AIA']>=job['threshold']-EPS:
                r['outcome']='AIA改善なし（途中で確定）';stop=True;break
        if not stop:
            r['complete']=True;r['outcome']='全攻撃を採点済み'
            r['max_AIA']=max(x['AIA'] for d in r['attacks'].values() for x in d.values())
            r['panel_max']={pid:max(x['AIA'] for x in d.values()) for pid,d in r['attacks'].items()}
            z=folder/'予測.npz';np.savez_compressed(z,**predictions)
            r['predictions']=str(z);r['predictions_sha256']=sha(z)
    r['seconds']=time.monotonic()-start
    if not r['complete']: cp.unlink()
    save(result_path,r);return r

def order_for(r):
    rows=[(pid,aid,x['AIA']) for pid,d in r['attacks'].items() for aid,x in d.items()]
    return [[p,a] for p,a,_ in sorted(rows,key=lambda x:(-x[2],x[0],x[1]))]

def base_job(base,pid_list=('dev1','dev2')):
    c=original(base)
    return dict(id='original-'+base,base=base,ops=[],key=content_key(c),panels=list(pid_list),floor=0)

def cli_check(p,r,pid,aid,root='cli検算'):
    panel=next(x for x in p['panels'] if x['id']==pid);a=next(x for x in p['attacks'] if x['id']==aid)
    folder=HERE/root/r['id']/pid/aid;folder.mkdir(parents=True,exist_ok=True)
    meta=folder/'検証.json'
    if meta.exists():
        old=load(meta);assert old['C_sha256']==sha(r['path']) and sha(old['prediction'])==old['prediction_sha256'];return old
    script=Path(copy(a['local_path'],folder/Path(a['local_path']).name))
    for asset in a['assets']: copy(Path(a['local_path']).parent/asset['name'],folder/asset['name'])
    save(folder/'targets.json',[dict(target_k=panel['target_id'],c_path=r['path'],aia_challenge_path=panel['challenge'])])
    command=[str(PYTHON),'-B',str(script),'--b',p['attack_b'],'--a-bg',p['a_bg'],'--targets',str(folder/'targets.json')]
    with (folder/'実行.log').open('w') as f:
        run=subprocess.run(command,cwd=folder,stdout=f,stderr=subprocess.STDOUT)
    pred_path=folder/('F_aia_'+script.stem+'.csv');assert run.returncode==0 and pred_path.exists(),str(folder)
    d=read(pred_path);q=read(panel['challenge']);t=read(panel['truth'])
    assert set(d.columns)=={'target_k','challenge_row_id','time_hat'} and len(d)==2*panel['n']
    assert d.target_k.eq(panel['target_id']).all() and d.challenge_row_id.is_unique and set(d.challenge_row_id)==set(q.challenge_row_id)
    actual=d.set_index('challenge_row_id').loc[q.challenge_row_id,'time_hat'].to_numpy(float)
    expected=np.load(r['predictions'])[pid+'__'+aid]
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
    np.testing.assert_array_equal(abs(actual-t.time.to_numpy())<=.5,abs(expected-t.time.to_numpy())<=.5)
    record=dict(passed=True,C_sha256=sha(r['path']),prediction=str(pred_path),prediction_sha256=sha(pred_path),command=command)
    save(meta,record);return record

def progress(state,**kwargs):
    state.update(kwargs,updated_epoch=time.time());save(HERE/'状態.json',state)

def run_batch(pool,jobs,state,stage,base):
    results=[];pending={};index=0;done=0;calls=0;rejections={};last=0
    progress(state,status='running',stage=stage,base=base,batch_done=0,batch_total=len(jobs))
    while index<len(jobs) or pending:
        while index<len(jobs) and len(pending)<32:
            j=jobs[index];pending[pool.submit(work,j)]=j;index+=1
        finished,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
        for f in finished:
            pending.pop(f);r=f.result();results.append(r);done+=1;calls+=r['attack_calls']
            rejections[r['outcome']]=rejections.get(r['outcome'],0)+1
        if time.monotonic()-last>=3 or done==len(jobs):
            progress(state,batch_done=done,attack_calls_batch=calls,batch_outcomes=rejections,
                latest=f'{stage}: {done}/{len(jobs)}')
            last=time.monotonic()
        if done and (done%100==0 or done==len(jobs)) and finished:
            say(f'{base} {stage} {bar(done,len(jobs))} / AIA実行 {calls}回 / 全採点 {rejections.get("全攻撃を採点済み",0)}件')
    return sorted(results,key=lambda r:r['id'])

def proposal_jobs(p,base,parents,depth,batch,floor,seen,stats):
    jobs=[];attrs=p['attrs']
    for parent_index,parent in enumerate(parents):
        c=read(parent['path']);rng=np.random.default_rng(p['seed']+p['base_order'].index(base)*100000+depth*1000+batch*100+parent_index)
        n=p['initial_per_attribute']*6 if depth==1 else p['expansion_per_parent']
        # Half uniform attribute choice; half allocated from observed improvements, with pseudocounts.
        gain=np.array([1+stats.get(a,0) for a in attrs],float);prob=.5/6+.5*gain/gain.sum()
        row_gain=np.ones(len(c))
        for row,count in stats.get('rows',{}).items():row_gain[int(row)]+=count
        row_prob=.5/len(c)+.5*row_gain/row_gain.sum()
        chosen=np.repeat(np.arange(6),p['initial_per_attribute']) if depth==1 else rng.choice(6,n,p=prob)
        local_seen=set();failures=0
        for ai in chosen:
            attr=attrs[int(ai)]
            while True:
                i,j=sorted(map(int,rng.choice(len(c),2,replace=False,p=row_prob)))
                if c.at[i,attr]==c.at[j,attr] or (attr,i,j) in local_seen:
                    failures+=1;continue
                if attr=='time' and (tuple(c.loc[i,['onset','death']])!=tuple(c.loc[j,['onset','death']]) or tuple(c.loc[i,['onset','death']])==(0,0)):
                    failures+=1;continue
                local_seen.add((attr,i,j));changed=c.copy();changed.loc[[i,j],attr]=c.loc[[j,i],attr].to_numpy()
                key=content_key(changed)
                if key in seen: failures+=1;continue
                seen.add(key);break
            op=dict(attribute=attr,i=i,j=j,row1=i+1,row2=j+1,old1=float(c.at[i,attr]),old2=float(c.at[j,attr]))
            cid=f'{base}-S{depth:02d}-B{batch:02d}-P{parent_index+1}-{len(jobs)+1:05d}'
            jobs.append(dict(id=cid,base=base,ops=parent['ops']+[op],key=key,panels=['dev1','dev2'],floor=floor,
                threshold=parent['max_AIA'],order=order_for(parent),parent=parent['id']))
        say(f'{base} {depth}交換目 / 親{parent_index+1}: 有効候補{n}件 / 無変更・重複・条件外を{failures}件除外')
    return jobs

def diverse_best(rows,n=3):
    ranked=sorted(rows,key=lambda r:(r['max_AIA'],r['total_swaps'],r['id']));chosen=[];sets=[]
    for r in ranked:
        ops={(x['attribute'],x['i'],x['j']) for x in r['ops']}
        if all(len(ops&s)/max(1,len(ops|s))<.8 for s in sets):chosen.append(r);sets.append(ops)
        if len(chosen)==n:return chosen
    for r in ranked:
        if r['id'] not in {x['id'] for x in chosen}:chosen.append(r)
        if len(chosen)==n:break
    return chosen

def search(p,pool,base,baseline,state):
    dst=HERE/'search'/base;dst.mkdir(parents=True,exist_ok=True)
    final=dst/'完了.json'
    if final.exists():return load(final)
    parents=[baseline];floor=baseline['utility']['utility'];seen={baseline['key']};stats={};kept=[]
    started=time.monotonic();used_before=0.;reason='最大10交換に到達'
    for depth in range(1,p['max_swaps']+1):
        promising=[]
        for batch in range(1,(1 if depth==1 else p['plateau_batches'])+1):
            folder=dst/f'S{depth:02d}-B{batch:02d}';folder.mkdir(parents=True,exist_ok=True)
            jp=folder/'候補計画.json';rp=folder/'集計.json'
            if jp.exists():
                jobs=load(jp);seen.update(j['key'] for j in jobs)
            else:
                jobs=proposal_jobs(p,base,parents,depth,batch,floor,seen,stats);save(jp,jobs)
            progress(state,phase_index=1+p['base_order'].index(base),swaps=depth,best_AIA=min(x['max_AIA'] for x in parents),U_floor=floor)
            if rp.exists():
                summary=load(rp);results=[load(HERE/'results'/j['id']/'結果.json') for j in jobs]
                used_before+=summary['seconds']
            else:
                t=time.monotonic();results=run_batch(pool,jobs,state,f'{depth}交換目・試行{batch}',base)
                summary=dict(seconds=time.monotonic()-t,candidates=len(jobs),attack_calls=sum(r['attack_calls'] for r in results),
                    improving=sum(r['complete'] for r in results));save(rp,summary)
            promising.extend(r for r in results if r['complete'])
            for r in results:
                if r['complete']:
                    assert r['max_AIA']<r['job']['threshold']-EPS and r['utility']['utility']>=floor-EPS
                    attr=r['ops'][-1]['attribute'];stats[attr]=stats.get(attr,0)+1
                    counts=stats.setdefault('rows',{})
                    for row in [r['ops'][-1]['i'],r['ops'][-1]['j']]:counts[str(row)]=counts.get(str(row),0)+1
            if promising:break
            if used_before+time.monotonic()-started>=p['search_hours'][base]*3600:break
        if not promising:
            reason=f'{depth}交換目の予定候補で厳密な改善なし（局所最適の証明ではない）';break
        parents=diverse_best(promising,p['beam_width']);kept.extend(parents)
        save(dst/f'交換数{depth:02d}の保存候補.json',parents)
        say(f'{base}: {depth}交換まで保存、最良AIA={parents[0]["max_AIA"]:.6f}')
        if parents[0]['max_AIA']<=EPS:
            reason='探索指標が下限0に到達';break
        # Soft allocation: finish a whole batch; never abort a candidate or final checks.
        if used_before+time.monotonic()-started>=p['search_hours'][base]*3600:
            reason='探索時間の配分目安に到達。完了した候補を保存して確認工程へ';break
    result=dict(base=base,reason=reason,kept=kept,attribute_improvements=stats,seconds=used_before+time.monotonic()-started)
    save(final,result);return result

def shortlist(searches):
    rows=[]
    for result in searches:
        candidates=result['kept']
        if not candidates:continue
        # Prespecified selection on development panels only: best, fewest swaps, and one middle point.
        best=min(candidates,key=lambda r:(r['max_AIA'],r['total_swaps'],r['id']))
        few=min(candidates,key=lambda r:(r['total_swaps'],r['max_AIA'],r['id']))
        chosen=[best,few]
        front=[r for r in candidates if not any(x['total_swaps']<=r['total_swaps'] and x['max_AIA']<=r['max_AIA'] and
            (x['total_swaps']<r['total_swaps'] or x['max_AIA']<r['max_AIA']) for x in candidates)]
        if front:
            mid=sorted(front,key=lambda r:(r['total_swaps'],r['max_AIA']))[len(front)//2];chosen.append(mid)
        unique={r['key']:r for r in chosen};rows.extend(unique.values())
    return rows

def prune(p,pool,rows,state):
    kept=[]
    for r in rows:
        best=r;round_no=0
        simplified=normalize_ops(r['base'],r['ops'])
        if len(simplified)<len(r['ops']):
            c=replay(r['base'],simplified)
            assert content_key(c)==r['key']
            job=dict(id=r['id']+'-cancel',base=r['base'],ops=simplified,key=r['key'],panels=['dev1','dev2'],floor=r['job']['floor'])
            best=pool.submit(work,job).result();assert best['complete'] and best['max_AIA']<=r['max_AIA']+EPS
        while len(best['ops'])>1:
            round_no+=1;jobs=[]
            for i in range(len(best['ops'])):
                ops=normalize_ops(best['base'],best['ops'][:i]+best['ops'][i+1:]);c=replay(best['base'],ops)
                jobs.append(dict(id=r['id']+f'-prune{round_no}-{i}',base=r['base'],ops=ops,key=content_key(c),
                    panels=['dev1','dev2'],floor=r['job']['floor']))
            results=run_batch(pool,jobs,state,'不要な交換を削減',r['base'])
            valid=[x for x in results if x['complete'] and x['max_AIA']<=best['max_AIA']+EPS]
            if not valid:break
            best=min(valid,key=lambda x:(x['max_AIA'],x['id']))
        kept.append(best)
    return list({r['key']:r for r in kept}.values())

def mia_job(p,r,panel,aid):
    """Use the unmodified attack CLI, so actual saved C is tested without adapters."""
    folder=HERE/'mia'/r['id']/panel['id']/aid;folder.mkdir(parents=True,exist_ok=True);meta=folder/'結果.json'
    if meta.exists():
        old=load(meta);assert old['C_sha256']==sha(r['path']) and sha(old['prediction'])==old['prediction_sha256'];return old
    if aid=='M04':
        src=Path(p['mia4']);assets=[x for x in src.parent.glob('*.py') if x!=src]
    else:
        a=next(x for x in p['attacks'] if x['id']==aid);src=Path(a['local_path']);assets=[src.parent/x['name'] for x in a['assets']]
    script=Path(copy(src,folder/src.name))
    for asset in assets:copy(asset,folder/asset.name)
    first=p['panels'][0]
    save(folder/'targets.json',[dict(target_k=first['target_id'],c_path=r['path'],aia_challenge_path=first['challenge'])])
    cmd=[str(PYTHON),'-B',str(script),'--b',panel['b'],'--a-bg',p['a_bg'],'--targets',str(folder/'targets.json')]
    with (folder/'実行.log').open('w') as log: run=subprocess.run(cmd,cwd=folder,stdout=log,stderr=subprocess.STDOUT)
    fp=folder/('F_mia_'+script.stem+'.csv');assert run.returncode==0 and fp.exists(),str(folder)
    d=read(fp);t=read(panel['truth']);assert len(d)==1049 and d.record_id.is_unique and set(d.record_id)==set(t.record_id)
    value_col=str(first['target_id'])
    assert set(d.columns)=={'record_id',value_col}
    pred=d.set_index('record_id').loc[t.record_id,value_col].to_numpy(float)
    assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all()
    sys.path.insert(0,str(Path(p['kit'])/'starter/src'))
    from pwscup2026.scoring.attack_score import _tpr_at_fpr
    tpr,auc=_tpr_at_fpr(pred,t.is_member.to_numpy(int),.01)
    result=dict(candidate=r['id'],base=r['base'],panel=panel['id'],attack=aid,TPR_at_FPR01=tpr,AUC=auc,
        members=26,controls=1023,rare_evaluated=False,C_sha256=sha(r['path']),prediction=str(fp),prediction_sha256=sha(fp),command=cmd)
    save(meta,result);return result

def report(p,baselines,selected,validation,mia,searches):
    names={a['id']:a['name'] for a in p['attacks']};aia_rows=[];summary=[]
    labels={r['id']+'-confirm':r['display_name'] for r in selected}
    labels.update({'original-'+base+'-confirm':base+'（元のC）' for base in p['base_order']})
    baseline_map={r['base']:r for r in validation if not r['ops']}
    reference={}
    for panel in p['panels']:
        s=scorer(panel['id']);s.shared_c={};s.cost=None
        reference[panel['id']]={'定数10':s.metrics(np.full(2*panel['n'],10.)),
            '背景だけの公式1NN':s.metrics(s.predict('A12',p['a_bg']))}
    save(HERE/'比較基準.json',reference)
    for r in validation:
        base=baseline_map[r['base']]
        for pid,d in r['attacks'].items():
            for aid,m in d.items():
                aia_rows.append(dict(候補=labels[r['id']],内部ID=r['id'],ベース=r['base'],総交換回数=r['total_swaps'],パネル=pid,
                    攻撃=names[aid],攻撃区分=next(a['group'] for a in p['attacks'] if a['id']==aid),**m,
                    元案AIA=base['attacks'][pid][aid]['AIA'],差=m['AIA']-base['attacks'][pid][aid]['AIA']))
            vals=[x['AIA'] for x in d.values()]
            summary.append(dict(候補=labels[r['id']],内部ID=r['id'],ベース=r['base'],総交換回数=r['total_swaps'],パネル=pid,U=r['utility']['utility'],
                AIA最小=min(vals),AIA中央値=float(np.median(vals)),AIA最大=max(vals),
                元案AIA最大=base['panel_max'][pid],最大値の差=max(vals)-base['panel_max'][pid],
                定数10=reference[pid]['定数10']['AIA'],背景1NN=reference[pid]['背景だけの公式1NN']['AIA'],
                最大値ひく背景1NN=max(vals)-reference[pid]['背景だけの公式1NN']['AIA']))
    out(HERE/'AIA_各攻撃と各パネル.csv',aia_rows);out(HERE/'AIA_要約.csv',summary)
    out(HERE/'MIA_4手法と3組.csv',[dict(x,候補=labels[x['candidate']],攻撃=names.get(x['attack'],'seo-04a-多重縮小生成比MIA')) for x in mia])
    decisions=[]
    for r in validation:
        if not r['ops']:continue
        base=baseline_map[r['base']];train=max(r['panel_max'][x] for x in ['dev1','dev2'])
        orig_train=max(base['panel_max'][x] for x in ['dev1','dev2'])
        mia_pairs=[(x,next(z for z in mia if z['candidate']==base['id'] and z['panel']==x['panel'] and z['attack']==x['attack'])) for x in mia if x['candidate']==r['id']]
        mia_ok=len(mia_pairs)==12 and all(x['TPR_at_FPR01']<=z['TPR_at_FPR01']+EPS for x,z in mia_pairs)
        small_ok=all(r['panel_max'][x]<=base['panel_max'][x]+EPS for x in ['small1','small2','small3'])
        hold_ok=r['panel_max']['holdout']<base['panel_max']['holdout']-EPS
        u_ok=r['utility']['utility']>=base['utility']['utility']-EPS
        decision=dict(候補=labels[r['id']],内部ID=r['id'],ベース=r['base'],総交換回数=r['total_swaps'],U維持=u_ok,
            探索2組で改善=train<orig_train-EPS,確認387人で改善=hold_ok,確認163人3組で悪化なし=small_ok,
            MIA全12比較で悪化なし=mia_ok,採用条件合格=u_ok and train<orig_train-EPS and hold_ok and small_ok and mia_ok)
        decisions.append(decision)
    out(HERE/'採用条件.csv',decisions)
    # Select at most three passing candidates, favoring fewer swaps. No forced winner.
    passing=sorted([d for d in decisions if d['採用条件合格']],key=lambda d:(d['総交換回数'],d['ベース'],d['候補']))[:3]
    for i,d in enumerate(passing,1):
        r=next(x for x in validation if x['id']==d['内部ID'])
        copy(r['path'],HERE/'採用候補'/f'{i}_{r["base"]}_総{r["total_swaps"]}交換.csv')
    save(HERE/'最終検証.json',dict(passed=True,candidates_checked=len(selected),all_final_C_cli_verified=True,
        official_anonymity_predicted=False,eligible=len([d for d in decisions if d['採用条件合格']]),recommended=passing))
    text=['# 探索結果（ローカル模擬実験）','',
        '- 元の Zhang11 → Zhang04 の順に探索しました。各ベース内は16並列です。',
        '- 探索中に使ったのは新規 dev1 / dev2 の2組です。Zhangさん作成パネルではありません。',
        '- 387人全員を使い、通常・希少 × 発症の構成が163人分になるよう重みを付けています。',
        '- AIAの値は低い方が良い指標です。2組×12攻撃の最大値を下げました。',
        '- holdout と163人3組、MIA3組は交換の選択に使っていません。',
        '- 同じ有限データから作ったため、各パネルは独立な本番の再現ではありません。',
        '- MIAは現在のCに4攻撃を実行。各組26標的・1023対照、全体TPR@FPR=1%です。真の希少対照ラベルがないため公式の希少MIAは未評価です。',
        '- 「最大値−背景1NN」は参考差分です。φ補正や本番の匿名性得点ではありません。','',
        '| ベース | 探索終了の理由 |','|---|---|']
    text += [f'| {s["base"]} | {s["reason"]} |' for s in searches]
    text += ['','| 候補 | 交換回数 | U維持 | 別387人組 | 163人3組 | MIA | 条件合格 |','|---|---:|---|---|---|---|---|']
    yn=lambda x:'○' if x else '×'
    text += [f'| {d["候補"]} | {d["総交換回数"]} | {yn(d["U維持"])} | {yn(d["確認387人で改善"])} | {yn(d["確認163人3組で悪化なし"])} | {yn(d["MIA全12比較で悪化なし"])} | {yn(d["採用条件合格"])} |' for d in decisions]
    text += ['',f'- 採用候補フォルダへの保存：{len(passing)}件。条件を満たさなければ無理に採用しません。',
        '- 詳細：`AIA_要約.csv`、`AIA_各攻撃と各パネル.csv`、`MIA_4手法と3組.csv`、`採用条件.csv`。',
        '- 探索済みの範囲での結果です。最小交換回数の証明や未知攻撃への保証はありません。']
    (HERE/'結果の読み方.md').write_text('\n'.join(text)+'\n')

def main():
    lock=(HERE/'実行.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=prepare();verify(p);initialize(p)
    state=load(HERE/'状態.json') if (HERE/'状態.json').exists() else dict(started_epoch=time.time())
    if state.get('status')=='complete':say('完了済みです。結果の読み方.md を確認してください。');return
    progress(state,status='running',stage='開始前の検算',phase_index=0,phase_total=6,pid=os.getpid())
    ctx=multiprocessing.get_context('spawn')
    try:
        with ProcessPoolExecutor(max_workers=p['workers'],mp_context=ctx,initializer=initialize,initargs=(p,)) as pool:
            # Validate all attacks on both development panels before trusting early pruning.
            baselines=[]
            for base in p['base_order']:
                r=pool.submit(work,base_job(base)).result();baselines.append(r)
                expected=next(v for v in load(HERE.parent/'初回450候補_ベース順次_16並列/計画.json')['variants'] if v['label']==base)['utility']['U']
                np.testing.assert_allclose(r['utility']['utility'],expected,atol=EPS,rtol=0)
                with ThreadPoolExecutor(max_workers=8) as threads:
                    fs=[threads.submit(cli_check,p,r,pid,a['id']) for pid in ['dev1','dev2'] for a in p['attacks']]
                    for f in as_completed(fs):f.result()
                say(f'事前検算OK {base}: 2組×12攻撃が元CLIと一致 / U={r["utility"]["utility"]:.6f} / 最大AIA={r["max_AIA"]:.6f}')
            if not (HERE/'事前検証.json').exists():
                # Exercise changed attributes and cache reuse, using original function bodies as the oracle.
                for r in baselines:
                    rng=np.random.default_rng(p['seed']+999)
                    for attr in p['attrs']:
                        c=original(r['base']).copy()
                        while True:
                            i,j=sorted(map(int,rng.choice(len(c),2,replace=False)))
                            if c.at[i,attr]==c.at[j,attr]:continue
                            if attr=='time' and (tuple(c.loc[i,['onset','death']])!=tuple(c.loc[j,['onset','death']]) or tuple(c.loc[i,['onset','death']])==(0,0)):continue
                            break
                        c.loc[[i,j],attr]=c.loc[[j,i],attr].to_numpy()
                        cp=HERE/'事前検算用C'/f'{r["base"]}_{attr}.csv';out(cp,c)
                        for pid in ['dev1','dev2']:
                            ss=scorer(pid);ss.shared_c={};ss.cost=None
                            for aid in ['A05','A08','A09','A10','A11']:
                                actual=ss.predict(aid,cp);m=ss.mods[aid]
                                bg=m.read_table(Path(p['a_bg']),'A' if aid=='A05' else 'background')
                                q=m.read_table(Path(ss.p['panel']['challenge']),'challenge');cc=m.read_table(cp,'C')
                                if aid=='A05':expected=m.aia_times(q,bg,cc,m.Scaler(bg))
                                else:
                                    expected=m.Attack(bg).predict(cc,q)[0]
                                    expected=m.exact_override(q,cc,np.asarray(expected,float))[0]
                                np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
                                np.testing.assert_array_equal(abs(actual-ss.times)<=.5,abs(expected-ss.times)<=.5)
                        say(f'交換後の検算OK {r["base"]} / {attr}')
                # Parse and score all four MIA output formats before a long search can start.
                with ThreadPoolExecutor(max_workers=4) as threads:
                    fs=[threads.submit(mia_job,p,baselines[0],p['mia_panels'][0],aid) for aid in ['A04','A05','A06','M04']]
                    for f in as_completed(fs):f.result()
                save(HERE/'事前検証.json',dict(passed=True,cli_comparisons=48,cache_comparisons=120,mia_cli_checks=4,exact_hit_masks=True))
            if '--preflight' in sys.argv:
                progress(state,status='ready',stage='事前検算完了');return
            searches=[search(p,pool,b,r,state) for b,r in zip(p['base_order'],baselines)]
            progress(state,phase_index=3,stage='交換の削減')
            selected=prune(p,pool,shortlist(searches),state)
            naming=[]
            for base in p['base_order']:
                group=sorted([r for r in selected if r['base']==base],key=lambda r:(r['total_swaps'],r['max_AIA']))
                for index,r in enumerate(group,1):
                    r['display_name']=f'{base}-HC{index}'
                    cp=copy(r['path'],HERE/'確認候補'/(r['display_name']+'.csv'))
                    pd.testing.assert_frame_equal(read(cp),replay(base,r['ops']))
                    out(HERE/'確認候補'/(r['display_name']+'_交換履歴.csv'),r['ops'])
                    naming.append(dict(候補=r['display_name'],ベース=base,総交換回数=r['total_swaps'],探索AIA=r['max_AIA'],U=r['utility']['utility'],ファイル=cp,内部ID=r['id']))
            out(HERE/'候補の対応表.csv',naming)
            frozen=HERE/'確認前の候補固定.json'
            if frozen.exists():assert load(frozen)==selected
            else:save(frozen,selected)
            progress(state,phase_index=4,stage='別パネルとMIAの確認')
            validation=[]
            # Sequential bases are also respected during final AIA evaluation.
            for base in p['base_order']:
                rows=[r for r in baselines+selected if r['base']==base];jobs=[]
                for r in rows:
                    jobs.append(dict(id=r['id']+'-confirm',base=base,ops=r['ops'],key=r['key'],input_path=r['path'],
                        panels=[x['id'] for x in p['panels']],floor=0))
                validation.extend(run_batch(pool,jobs,state,'6組で全12攻撃を確認',base))
        # Free all heavyweight candidate processes before the unmodified CLIs.
        for r in validation:
            with ThreadPoolExecutor(max_workers=8) as threads:
                fs=[threads.submit(cli_check,p,r,pid,a['id'],'最終cli検算') for pid in ['dev1','holdout','small1'] for a in p['attacks']]
                for f in as_completed(fs):f.result()
            say(f'最終CLI検算OK {r["id"]}: 387人・163人の予測一致')
        mia=[]
        for base in p['base_order']:
            with ThreadPoolExecutor(max_workers=8) as threads:
                fs=[threads.submit(mia_job,p,r,panel,aid) for r in validation if r['base']==base
                    for panel in p['mia_panels'] for aid in ['A04','A05','A06','M04']]
                for f in as_completed(fs):
                    mia.append(f.result());progress(state,stage='MIAを確認',mia_done=len(mia),mia_total=len(validation)*12)
        progress(state,phase_index=5,stage='資料と最終確認')
        report(p,baselines,selected,validation,mia,searches)
        for f,h in p['hashes'].items():assert sha(f)==h,f
        progress(state,status='complete',phase_index=6,stage='完了',elapsed_seconds=time.time()-state['started_epoch'])
        say('全工程が完了しました。結果の読み方.md を確認してください。')
    except BaseException:
        progress(state,status='failed',error=traceback.format_exc(),elapsed_seconds=time.time()-state['started_epoch'])
        raise

if __name__=='__main__':main()
