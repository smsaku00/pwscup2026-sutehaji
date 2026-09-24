"""Twenty-minute utility-budget pilot. Fixed inputs, shared scoring, untouched holdouts."""
from 採点共通 import *
import 探索部品 as engine
from 準備 import prepare
from 時間探索 import make_jobs
from MIA確認 import mia_job
from datetime import datetime
from collections import Counter

HERE=Path(__file__).resolve().parent
EPS=1e-12
START=datetime.fromisoformat('2026-09-24T13:23:05+00:00').timestamp()
DEADLINE=START+20*60
COMPARISON=HERE.parent/'公平比較_初回合格案と長時間5案_20260924'

def budgets(original):return [('現状維持',original),('U≧0.915',.915),('U≧0.910',.910),('U≧0.900',.900)]
def winners(rows,floor):
    eligible=[r for r in rows if r['utility']['utility']>=floor-EPS]
    return sorted(eligible,key=lambda r:(r['max_AIA'],r['total_swaps'],-r['utility']['utility'],r['id']))
def trim(rows,original):
    unique={r['key']:r for r in rows};keep={}
    for _,floor in budgets(original):
        for r in winners(unique.values(),floor)[:3]:keep[r['key']]=r
    for r in unique.values():
        if not r['ops']:keep[r['key']]=r
    return list(keep.values())

def search(p,pool,base,initial,state,end):
    archive=initial[:];floor=initial[0]['utility']['utility'];seen={r['key'] for r in initial}
    attempts={};stats={};counts=Counter();round_no=0;completed=0;all_complete=[]
    while time.time()<end-8:
        eligible=[r for r in archive if r['total_swaps']<p['max_swaps']]
        parents={r['key']:r for r in eligible if not r['ops']}
        for _,u in budgets(floor):
            ranked=winners(eligible,u)
            if ranked:parents[ranked[0]['key']]=ranked[0]
        if eligible:
            r=min(eligible,key=lambda x:(attempts.get(x['key'],0),x['max_AIA'],x['id']));parents[r['key']]=r
        round_no+=1
        jobs,diag=make_jobs(engine,p,base,list(parents.values()),round_no,.900,seen,stats)
        if not jobs:break
        target=HERE/'search'/base/f'R{round_no:04d}'
        save(target/'候補計画.json',jobs);save(target/'生成記録.json',diag)
        engine.progress(state,base_deadline_epoch=end,round=round_no,U_floor=.900,
            swaps='1〜10回',new_search_candidates=state.get('finished_previous_base',0)+completed)
        results=engine.run_batch(pool,jobs,state,f'{base} 有用性の4条件を共通候補で比較',base)
        completed+=len(results);counts.update(r['outcome'] for r in results)
        added=[r for r in results if r['complete']]
        for r in added:
            assert r['utility']['utility']>=.900-EPS and r['max_AIA']<r['job']['threshold']-EPS
            attr=r['ops'][-1]['attribute'];stats[attr]=stats.get(attr,0)+1
            all_complete.append(r)
        archive=trim(archive+added,floor)
        for r in parents.values():attempts[r['key']]=attempts.get(r['key'],0)+1
        summary=[dict(condition=label,floor=u,U=winners(archive,u)[0]['utility']['utility'],
            AIA=winners(archive,u)[0]['max_AIA'],candidate=winners(archive,u)[0]['id']) for label,u in budgets(floor)]
        save(target/'集計.json',dict(completed=len(results),outcomes=dict(Counter(r['outcome'] for r in results)),best=summary))
        save(HERE/'search'/base/'継続状態.json',dict(completed=completed,outcomes=dict(counts),archive=archive,best=summary))
        engine.progress(state,new_search_candidates=state.get('finished_previous_base',0)+completed,
            utility_budgets=summary,best_AIA=min(r['max_AIA'] for r in archive))
        say(f'{base}: {completed}候補 / U不足 {counts["U不足"]} / '+str([(x['condition'],round(x['U'],6),round(x['AIA'],6)) for x in summary]))
    result=dict(base=base,completed=completed,outcomes=dict(counts),archive=archive,fully_scored=all_complete)
    save(HERE/'search'/base/'完了.json',result)
    state['finished_previous_base']=state.get('finished_previous_base',0)+completed
    return result

def verify_initial(p,records):
    checks=0
    for r in records:
        if not r['ops']:
            fp=COMPARISON/'aia'/('base_'+r['base'])/'dev1_dev2'/'予測.npz'
        else:
            previous=load(Path(p['previous_run'])/'search'/r['base']/'完了.json')['kept']
            old=next(x for x in previous if x['key']==r['key']);fp=Path(old['predictions'])
        with np.load(fp) as old,np.load(r['predictions']) as new:
            for k in new.files:
                np.testing.assert_allclose(new[k],old[k],atol=1e-10,rtol=0);checks+=1
    save(HERE/'事前検証.json',dict(passed=True,full_prediction_checks=checks,old_scores_used_for_selection=False))

def report(p,selection,validated,mia,searches):
    cases={r['id']:r for r in validated};originals={r['base']:r for r in validated if not r['ops']}
    detail=[];aia=[];summary=[];changes=[]
    mia_rows=[r for x in mia for r in x['rows']]
    mi={(r['case'],r['panel'],r['attack']):r['TPR_at_FPR01'] for r in mia_rows}
    for r in validated:
        original=originals[r['base']]
        for pid,attacks in r['attacks'].items():
            values=[x['AIA'] for x in attacks.values()];old=original['panel_max'][pid]
            summary.append(dict(候補=r['id'],ベース=r['base'],U=r['utility']['utility'],パネル=pid,
                最大AIA=max(values),元案の最大AIA=old,差=max(values)-old,中央値=float(np.median(values)),最小値=min(values)))
            for aid,value in attacks.items():
                name=next(a['name'] for a in p['attacks'] if a['id']==aid)
                aia.append(dict(候補=r['id'],パネル=pid,手法=name,**value))
    for choice in selection:
        r=cases[choice['case']];old=originals[r['base']]
        bad_small=[pid for pid in ['small1','small2','small3'] if r['panel_max'][pid]>old['panel_max'][pid]+EPS]
        bad_mia=[(panel['id'],aid) for panel in p['mia_panels'] for aid in ['A04','A05','A06','M04']
            if mi[r['id'],panel['id'],aid]>mi[old['id'],panel['id'],aid]+EPS]
        for pid in bad_small:changes.append(dict(条件=choice['condition'],候補=r['id'],種類='AIA',パネル=pid,手法='12手法の最大',
            元案=old['panel_max'][pid],候補の値=r['panel_max'][pid],差=r['panel_max'][pid]-old['panel_max'][pid]))
        for pid,aid in bad_mia:changes.append(dict(条件=choice['condition'],候補=r['id'],種類='MIA',パネル=pid,手法=aid,
            元案=mi[old['id'],pid,aid],候補の値=mi[r['id'],pid,aid],差=mi[r['id'],pid,aid]-mi[old['id'],pid,aid]))
        row=dict(ベース=r['base'],条件=choice['condition'],U下限=choice['floor'],候補=r['id'],有用性U=r['utility']['utility'],
            総交換回数=r['total_swaps'],探索最大AIA=choice['train_max'],確認387人AIA=r['panel_max']['holdout'],
            元案の確認387人AIA=old['panel_max']['holdout'],確認163人の最大AIA=max(r['panel_max'][x] for x in ['small1','small2','small3']),
            AIA悪化組数=len(bad_small),MIA悪化比較数=len(bad_mia))
        row['条件を満たす']=bool(r['utility']['utility']>=choice['floor']-EPS and choice['train_max']<choice['original_train']-EPS
            and r['panel_max']['holdout']<old['panel_max']['holdout']-EPS and not bad_small and not bad_mia)
        detail.append(row)
    out(HERE/'共有用/条件別比較.csv',detail);out(HERE/'共有用/AIA_パネル別.csv',summary)
    out(HERE/'共有用/AIA_全手法.csv',aia);out(HERE/'共有用/MIA_全手法.csv',mia_rows)
    if changes:out(HERE/'共有用/悪化箇所.csv',changes)
    save(HERE/'集計.json',dict(comparison=detail,aia_summary=summary,mia=mia_rows,searches=searches,selection=selection))
    lines=['# 有用性を少し下げる探索：20分試行','',
        '根拠：本戦Bによるローカル模擬実験。有用性は公式コードで自己採点。','',
        '| ベース | U条件 | 実測U | 探索AIA | 別の387人パネル | 163人AIA悪化組数 | MIA悪化数 | 条件を満たす |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in detail:lines.append(f'| {r["ベース"]} | {r["条件"]} | {r["有用性U"]:.6f} | {r["探索最大AIA"]:.6f} | {r["確認387人AIA"]:.6f} | {r["AIA悪化組数"]}/3 | {r["MIA悪化比較数"]}/12 | {"はい" if r["条件を満たす"] else "いいえ"} |')
    lines+=['','- Uは高い方、AIAとMIAは低い方が良い値です。',
        '- 4条件は同じ探索候補群で比較しました。計算を4重に実行していません。',
        '- 各条件の候補は探索用の結果だけで選び、その後に初めて新しい確認パネルを採点しました。',
        '- 既存の最良探索案を出発点に含めていますが、今回の攻撃予測は全て新規計算です。',
        '- 同じ候補が複数条件で最良になる場合があります。Uを下げること自体は目的にしていません。',
        '- 最終条件はU下限、探索AIA改善、確認387人AIA改善、163人3組とMIA12比較で悪化なしです。',
        '- 確認用の標的はBと重複します。未使用の抽出乱数による再抽出であり、独立の新規母集団ではありません。',
        '- 希少MIAと本番の匿名性得点は未評価。φ補正は適用していません。',
        f'- 新規候補数：'+str({x['base']:x['completed'] for x in searches})]
    (HERE/'共有用/結果の説明.md').write_text('\n'.join(lines)+'\n')
    return detail

def main():
    lock=(HERE/'実行.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',started_epoch=START,target_end_epoch=DEADLINE,search_end_epoch=DEADLINE-300,
        phase_index=0,stage='準備と既存結果との照合',finished_previous_base=0)
    engine.progress(state)
    try:
        p=prepare();engine.initialize(p)
        jobs=[]
        for base in p['base_order']:
            jobs.append(engine.base_job(base))
            for x in p['seed_parents'][base]:jobs.append(dict(id=x['id'],base=base,ops=x['ops'],key=x['key'],input_path=x['input_path'],panels=['dev1','dev2'],floor=0))
        with ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'),initializer=engine.initialize,initargs=(p,)) as pool:
            initial=engine.run_batch(pool,jobs,state,'元案と既存の探索最良案を再採点','両ベース')
            verify_initial(p,initial)
            remaining=max(0,DEADLINE-300-time.time());mid=time.time()+remaining/2
            searches=[]
            for index,base in enumerate(p['base_order']):
                engine.progress(state,phase_index=index+1)
                rows=sorted([r for r in initial if r['base']==base],key=lambda r:r['total_swaps'])
                searches.append(search(p,pool,base,rows,state,mid if index==0 else DEADLINE-300))
            selection=[];unique={r['id']:r for r in initial if not r['ops']}
            for result in searches:
                original=next(r for r in initial if r['base']==result['base'] and not r['ops'])
                for label,floor in budgets(original['utility']['utility']):
                    r=winners(result['archive'],floor)[0];unique[r['id']]=r
                    selection.append(dict(base=r['base'],condition=label,floor=floor,case=r['id'],train_max=r['max_AIA'],original_train=original['max_AIA']))
            save(HERE/'確認前の候補固定.json',dict(created_epoch=time.time(),selection=selection,candidates=list(unique.values())))
            engine.progress(state,phase_index=3,stage='未使用の確認パネルでAIAを評価')
            confirm_jobs=[dict(id=r['id']+'-confirm',base=r['base'],ops=r['ops'],key=r['key'],input_path=r['path'],
                panels=['holdout','small1','small2','small3'],floor=0) for r in unique.values()]
            validated=engine.run_batch(pool,confirm_jobs,state,'新しい確認パネル4組×12攻撃','確認用')
        for r in validated:r['id']=r['id'].removesuffix('-confirm')
        cp=dict(p,cases=[dict(id=r['id'],path=r['path'],target_id=9901+i) for i,r in enumerate(validated)],mia_references=[])
        engine.progress(state,phase_index=4,stage='新しいMIA3組×4手法',mia_done=0,mia_total=12)
        mia=[]
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures=[pool.submit(mia_job,cp,panel,aid) for panel in p['mia_panels'] for aid in ['A04','A05','A06','M04']]
            for f in as_completed(futures):
                mia.append(f.result());engine.progress(state,mia_done=len(mia),stage=f'MIA確認 {len(mia)}/12')
        engine.progress(state,phase_index=5,stage='集計と入力の不変確認')
        result=report(p,selection,validated,mia,searches)
        for f,h in p['hashes'].items():assert sha(f)==h,f
        for r in validated:
            assert sha(r['path'])==r['C_sha256']
            if r['ops']:copy(r['path'],HERE/'候補C'/f'{r["id"]}.csv')
        save(HERE/'検証結果.json',dict(passed=True,unique_confirmed_C=len(validated),AIA_confirmation_evaluations=len(validated)*48,
            MIA_confirmation_evaluations=len(validated)*12,frozen_inputs_unchanged=True,unused_panel_seed=p['seed'],
            selected_before_holdout=True,elapsed_seconds=time.time()-START))
        engine.progress(state,status='complete',phase_index=6,stage='完了',elapsed_seconds=time.time()-START)
        say('完了: '+str(result))
    except BaseException as e:
        engine.progress(state,status='failed',error=traceback.format_exc(),elapsed_seconds=time.time()-START)
        raise

if __name__=='__main__':main()
