"""Keep proposing unseen swaps until the time allocation ends; never stop for a plateau."""
from 採点共通 import *

EPS=1e-12

def schedule_deadlines(p,state,progress,now=None):
    if 'base_deadlines' in state:return
    now=time.time() if now is None else now
    target=state['started_epoch']+p['planned_hours']*3600
    end=target-p['validation_reserve_minutes']*60
    remaining=max(0,end-now);total=sum(p['search_shares'].values());cumulative=0
    deadlines={}
    for base in p['base_order']:
        cumulative+=p['search_shares'][base]
        deadlines[base]=now+remaining*cumulative/total
    progress(state,target_end_epoch=target,search_end_epoch=end,base_deadlines=deadlines,
        schedule_created_epoch=now,previous_search_candidates=0,new_search_candidates=0)

def trim_archive(ctx,rows,width):
    by_key={r['key']:r for r in rows};selected=[]
    for depth in sorted({r['total_swaps'] for r in by_key.values()}):
        group=[r for r in by_key.values() if r['total_swaps']==depth]
        selected.extend(ctx.diverse_best(group,width))
    return selected

def select_parents(archive,attempts,max_swaps):
    eligible=[r for r in archive if r['total_swaps']<max_swaps and r['max_AIA']>EPS]
    if not eligible:return []
    # Always search for small changes again. Explore the current best and a less visited path too.
    chosen=[]
    roots=[r for r in eligible if r['total_swaps']==0]
    if roots:chosen.append(roots[0])
    remaining=[r for r in eligible if r not in chosen]
    if remaining:chosen.append(min(remaining,key=lambda r:(r['max_AIA'],r['total_swaps'],r['id'])))
    remaining=[r for r in eligible if r not in chosen]
    if remaining:chosen.append(min(remaining,key=lambda r:(attempts.get(r['key'],0),r['total_swaps'],r['max_AIA'],r['id'])))
    return chosen

def make_jobs(ctx,p,base,parents,round_no,floor,seen,stats):
    jobs=[];attrs=p['attrs'];max_proposals=p['batch_per_parent'];diagnostics=[]
    for parent_index,parent in enumerate(parents):
        c=read(parent['path']);n=len(c)
        seed=np.random.SeedSequence([p['seed'],p['base_order'].index(base),round_no,parent_index])
        rng=np.random.default_rng(seed)
        gain=np.array([1+stats.get(a,0) for a in attrs],float);prob=.5/len(attrs)+.5*gain/gain.sum()
        row_gain=np.ones(n)
        for row,count in stats.get('rows',{}).items():row_gain[int(row)]+=count
        row_prob=.5/n+.5*row_gain/row_gain.sum()
        # Precompute the legal time pairs once per parent, avoiding mostly-invalid random draws.
        ii,jj=np.triu_indices(n,1)
        legal=(c.onset.to_numpy()[ii]==c.onset.to_numpy()[jj]) & (c.death.to_numpy()[ii]==c.death.to_numpy()[jj])
        legal &= (c.onset.to_numpy()[ii]!=0)|(c.death.to_numpy()[ii]!=0)
        legal &= c.time.to_numpy()[ii]!=c.time.to_numpy()[jj]
        time_pairs=np.column_stack([ii[legal],jj[legal]])
        time_weights=None
        if len(time_pairs):
            w=row_prob[time_pairs[:,0]]*row_prob[time_pairs[:,1]]
            time_weights=.5/len(w)+.5*w/w.sum()
        local_seen=set();skipped=0;made=0;trials=0
        # A bounded generator cannot get stuck when a neighborhood is nearly exhausted.
        while made<max_proposals and trials<max_proposals*100:
            trials+=1;attr=attrs[int(rng.choice(len(attrs),p=prob))]
            if attr=='time':
                if not len(time_pairs):skipped+=1;continue
                i,j=map(int,time_pairs[int(rng.choice(len(time_pairs),p=time_weights))])
            else:i,j=sorted(map(int,rng.choice(n,2,replace=False,p=row_prob)))
            if c.at[i,attr]==c.at[j,attr] or (attr,i,j) in local_seen:skipped+=1;continue
            local_seen.add((attr,i,j))
            changed=c.copy();changed.loc[[i,j],attr]=c.loc[[j,i],attr].to_numpy()
            key=ctx.content_key(changed)
            if key in seen:skipped+=1;continue
            seen.add(key);made+=1
            op=dict(attribute=attr,i=i,j=j,row1=i+1,row2=j+1,old1=float(c.at[i,attr]),old2=float(c.at[j,attr]))
            cid=f'{base}-R{round_no:05d}-P{parent_index+1}-{made:04d}'
            jobs.append(dict(id=cid,base=base,ops=parent['ops']+[op],key=key,panels=['dev1','dev2'],floor=floor,
                threshold=parent['max_AIA'],order=ctx.order_for(parent),parent=parent['id']))
        diagnostics.append(dict(parent=parent['id'],total_swaps=parent['total_swaps']+1,generated=made,skipped=skipped,trials=trials))
    return jobs,diagnostics

def continue_search(ctx,p,pool,base,baseline,state,wall=None,jobs_factory=None):
    wall=time.time if wall is None else wall;jobs_factory=make_jobs if jobs_factory is None else jobs_factory
    dst=ctx.HERE/'search'/base;dst.mkdir(parents=True,exist_ok=True)
    final=dst/'完了.json'
    if final.exists():return load(final)
    deadline=state['base_deadlines'][base];floor=baseline['utility']['utility'];checkpoint=dst/'継続状態.json'
    seen=set(load(p['previous_keys'][base]));seen.add(baseline['key'])
    for jp in sorted(dst.glob('R*/候補計画.json')):seen.update(j['key'] for j in load(jp))
    if checkpoint.exists():
        saved=load(checkpoint);archive=saved['archive'];stats=saved['stats'];attempts=saved['attempts']
        round_no=saved['round_no'];completed=saved['completed'];calls=saved['attack_calls'];no_gain=saved['no_gain_rounds']
    else:
        seed_jobs=[dict(id=x['id'],base=base,ops=x['ops'],key=x['key'],input_path=x['input_path'],
            panels=['dev1','dev2'],floor=floor) for x in p['seed_parents'][base]]
        seeds=ctx.run_batch(pool,seed_jobs,state,'前回の途中案を新たに採点',base) if seed_jobs else []
        for r in seeds:seen.add(r['key'])
        archive=trim_archive(ctx,[baseline]+[r for r in seeds if r['complete'] and r['max_AIA']<baseline['max_AIA']-EPS],p['beam_width'])
        stats={};attempts={};round_no=0;completed=0;calls=0;no_gain=0
    for r in archive:seen.add(r['key'])
    inherited_count=len(load(p['previous_keys'][base]))
    reason='探索時間の目安に到達。現在の候補のまとまりを完了して次の工程へ'
    # Check time only between batches. A plateau is deliberately not a stopping condition.
    while wall()<deadline or (dst/f'R{round_no+1:05d}'/'候補計画.json').exists():
        parents=select_parents(archive,attempts,p['max_swaps'])
        if not parents:
            reason='探索可能な全経路が指標の理論下限に到達';break
        next_round=round_no+1;folder=dst/f'R{next_round:05d}';folder.mkdir(parents=True,exist_ok=True)
        jp=folder/'候補計画.json'
        if jp.exists():
            jobs=load(jp);diag=load(folder/'生成記録.json')
        else:
            jobs,diag=jobs_factory(ctx,p,base,parents,next_round,floor,seen,stats)
            save(jp,jobs);save(folder/'生成記録.json',diag)
        ctx.progress(state,phase_index=1+p['base_order'].index(base),round=next_round,
            swaps=' / '.join(map(str,sorted({len(j['ops']) for j in jobs}))),base_deadline_epoch=deadline,
            best_AIA=min(r['max_AIA'] for r in archive),U_floor=floor,base_candidates=completed,
            previous_search_candidates=inherited_count,no_gain_rounds=no_gain,active_parents=[r['id'] for r in parents])
        before=wall()
        results=ctx.run_batch(pool,jobs,state,f'継続探索 第{next_round}組',base) if jobs else []
        improving=[r for r in results if r['complete']]
        for r in improving:
            assert r['max_AIA']<r['job']['threshold']-EPS and r['utility']['utility']>=floor-EPS
            attr=r['ops'][-1]['attribute'];stats[attr]=stats.get(attr,0)+1
            counts=stats.setdefault('rows',{})
            for row in [r['ops'][-1]['i'],r['ops'][-1]['j']]:counts[str(row)]=counts.get(str(row),0)+1
        no_gain=0 if improving else no_gain+1
        for parent in parents:attempts[parent['key']]=attempts.get(parent['key'],0)+1
        archive=trim_archive(ctx,archive+improving,p['beam_width'])
        round_no=next_round;completed+=len(results);actual_calls=sum(r['attack_calls'] for r in results);calls+=actual_calls
        save(folder/'集計.json',dict(candidates=len(jobs),completed=len(results),improving=len(improving),
            attack_calls=actual_calls,seconds=wall()-before,no_gain_rounds=no_gain))
        save(checkpoint,dict(round_no=round_no,archive=archive,stats=stats,attempts=attempts,
            completed=completed,attack_calls=calls,no_gain_rounds=no_gain,deadline=deadline))
        per_base=dict(state.get('per_base_candidates',{}));per_base[base]=completed
        ctx.progress(state,base_candidates=completed,per_base_candidates=per_base,
            new_search_candidates=sum(per_base.values()),best_AIA=min(r['max_AIA'] for r in archive),no_gain_rounds=no_gain)
        say(f'{base} 第{round_no}組完了 / 新規累計{completed}候補 / 改善{len(improving)}件 / 改善なし連続{no_gain}組でも探索継続')
    result=dict(base=base,reason=reason,kept=[r for r in archive if r['ops']],attribute_improvements=stats,
        new_candidates=completed,attack_calls=calls,rounds=round_no,deadline=deadline,ended_epoch=wall())
    save(final,result);return result
