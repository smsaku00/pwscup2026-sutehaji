"""Secondary check of the best actually-lower-U candidate, fixed before reading holdouts."""
from 採点共通 import *
import 探索部品 as engine
import MIA確認 as mm
HERE=Path(__file__).resolve().parent

def main():
    p=load(HERE/'計画.json');fixed=load(HERE/'U低下案の追加確認_候補固定.json')
    engine.initialize(p);selected=fixed['candidates'];validated=[]
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=engine.initialize,initargs=(p,)) as pool:
        for r in selected:
            job=dict(id=r['id']+'-extra-confirm',base=r['base'],ops=r['ops'],key=r['key'],input_path=r['path'],
                panels=['holdout','small1','small2','small3'],floor=0)
            z=pool.submit(engine.work,job).result();validated.append(z);say('U低下案のAIA確認完了 '+r['base'])
    while load(HERE/'状態.json')['status']=='running':time.sleep(1)
    assert load(HERE/'状態.json')['status']=='complete'
    mm.HERE=HERE/'追加確認'
    cp=dict(p,cases=[dict(id=r['id'],path=r['path'],target_id=9901+i) for i,r in enumerate(validated)],mia_references=[])
    results=[]
    if validated:
        with ThreadPoolExecutor(max_workers=12) as pool:
            fs=[pool.submit(mm.mia_job,cp,panel,aid) for panel in p['mia_panels'] for aid in ['A04','A05','A06','M04']]
            for f in as_completed(fs):results.append(f.result());say(f'U低下案のMIA確認 {len(results)}/12')
    main_result=load(HERE/'集計.json');mia=[r for batch in results for r in batch['rows']]
    all_mia=main_result['mia']+mia
    index={(r['case'],r['panel'],r['attack']):r['TPR_at_FPR01'] for r in all_mia}
    rows=[];panel_rows=[]
    for old,z in zip(selected,validated):
        original=load(HERE/'results'/('original-'+z['base']+'-confirm')/'結果.json')
        strict=next(x for x in main_result['selection'] if x['base']==z['base'] and x['condition']=='現状維持')
        high=load(HERE/'results'/(strict['case']+'-confirm')/'結果.json')
        mia_bad=lambda cid:sum(index[z['id'],t['id'],a]>index[cid,t['id'],a]+1e-12 for t in p['mia_panels'] for a in ['A04','A05','A06','M04'])
        row=dict(候補=old['id'],ベース=z['base'],有用性U=z['utility']['utility'],総交換数=z['total_swaps'],
            探索AIA=old['max_AIA'],U維持案の探索AIA=strict['train_max'],確認387人AIA=z['panel_max']['holdout'],
            U維持案の確認387人AIA=high['panel_max']['holdout'],元案の確認387人AIA=original['panel_max']['holdout'],
            元案比163人AIA悪化組数=sum(z['panel_max'][pid]>original['panel_max'][pid]+1e-12 for pid in ['small1','small2','small3']),
            U維持案比163人AIA悪化組数=sum(z['panel_max'][pid]>high['panel_max'][pid]+1e-12 for pid in ['small1','small2','small3']),
            元案比MIA悪化数=mia_bad('original-'+z['base']),U維持案比MIA悪化数=mia_bad(strict['case']))
        rows.append(row)
        for pid,value in z['panel_max'].items():panel_rows.append(dict(候補=old['id'],パネル=pid,AIA=value,元案のAIA=original['panel_max'][pid],U維持案のAIA=high['panel_max'][pid]))
        copy(z['path'],HERE/'参考C'/f'{old["id"]}.csv')
    if rows:
        out(HERE/'共有用/U低下案_追加確認.csv',rows);out(HERE/'共有用/U低下案_AIAパネル別.csv',panel_rows)
        out(HERE/'共有用/U低下案_MIA全手法.csv',mia)
    save(HERE/'追加確認結果.json',dict(passed=True,selection=fixed,comparison=rows,aia=validated,mia=mia,
        ended_epoch=time.time(),panels_used_for_selection=False))
    say(str(rows))
if __name__=='__main__':main()
