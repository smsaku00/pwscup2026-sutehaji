"""Fixed, fresh 300-panel comparison. No search, MIA, network, or model API."""
from 採点共通 import *
from scipy import stats
import gc
import platform
import statistics

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent/'7案_追加20組比較'
GEN = module('frozen_panel_generator', HERE/'旧パネル生成.py')
COUNT = 300
SEEDS = list(range(202609251001, 202609251301))
TOTAL = COUNT + COUNT*26 + 2
STATE = {}
START = time.monotonic()
EPS = 1e-12

def progress(done, phase):
    elapsed = time.monotonic()-START
    state = dict(done=done,total=TOTAL,percent=100*done/TOTAL,phase=phase,
                 elapsed_seconds=elapsed,updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    save(HERE/'進捗.json',state)
    say(bar(done,TOTAL,32)+f' {phase} / 経過{elapsed/60:.1f}分')

def validate_panel(q,t,pool,b):
    m=t.label.eq('member')
    assert len(q)==326 and m.sum()==163 and (~m).sum()==163
    assert q.columns.tolist()==['challenge_row_id']+GEN.QI and q.challenge_row_id.equals(t.challenge_row_id)
    assert len(GEN.keys(q))==326
    assert t.loc[m,'paired_member_stratum'].value_counts().sort_index().tolist()==[138,12,9,4]
    np.testing.assert_allclose(t.weight_163,1,atol=1e-12,rtol=0)
    for label,g in t.groupby('label'):
        assert len(set(zip(g.origin,g.source_row_index)))==163
    for o,idx in t.groupby('origin').groups.items():
        source=b if o=='本戦B' else pool[pool.origin.eq(o)].set_index('source_row_index',drop=False)
        s=source.loc[t.loc[idx,'source_row_index'].to_numpy(int)]
        pd.testing.assert_frame_equal(q.loc[idx,GEN.QI].reset_index(drop=True),s[GEN.QI].reset_index(drop=True))
        np.testing.assert_array_equal(t.loc[idx,'time'],s.time)
    return frozenset(t.loc[m,'source_row_index'].tolist()),frozenset(t.loc[~m,['origin','source_row_index']].itertuples(index=False,name=None))

def prepare():
    if (HERE/'計画.json').exists():
        p=load(HERE/'計画.json')
        for f,h in p['hashes'].items(): assert sha(f)==h,f
        return p
    old=load(PREVIOUS/'計画.json')
    p={k:old[k] for k in ['kit','b','a_bg','schema','phi']}
    p.update(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),workers=16,panels=[],cases=[],attacks=[],control_pools={},source_hashes={})
    def freeze(src,relative):
        p['source_hashes'][str(src)]=sha(src)
        return copy(src,HERE/relative)
    for k,rel in [('b','inputs/utility/B.csv'),('a_bg','inputs/A_bg.csv'),('schema','inputs/utility/schema.json')]:
        p[k]=freeze(old[k],rel)
    for f in ['B_self.csv','utility_ref.json']:freeze(Path(old['b']).with_name(f),'inputs/utility/'+f)
    for origin,src in old['control_pools'].items():p['control_pools'][origin]=freeze(src,'inputs/pool/'+Path(src).name)
    for cid in ['C27','C26']:
        c=next(c for c in old['cases'] if c['id']==cid)
        p['cases'].append({k:c[k] for k in ['id','name','base','sha256','swaps','ops']})
        p['cases'][-1]['path']=freeze(c['path'],'C/'+c['name']+'.csv')
        assert sha(p['cases'][-1]['path'])==c['sha256']
    for a in old['attacks']:
        dest=HERE/'programs'/a['id']
        for f in Path(a['local_path']).parent.glob('*.py'):freeze(f,'programs/'+a['id']+'/'+f.name)
        p['attacks'].append(dict(a,local_path=str(dest/Path(a['local_path']).name)))
    p['policy']=dict(panel_count=COUNT,seeds=SEEDS,cases=p['cases'],case_order=['C27','C26'],
        primary='mean paired difference: max(0,max_12_AIA-max(constant10,background1NN)); Zhang11 minus Zhang04',
        alpha=.05,test='two-sided paired t, 95% CI; paired panel bootstrap sensitivity only',
        bootstrap_replicates=20000,bootstrap_seed=202609251999,
        secondary=['raw','minus_constant','minus_background','minus_phi'],
        exploratory_tail_thresholds=[.05,.1,.2],early_stopping=False,old_results_in_primary=False,
        scope='163 targets and 163 controls, fixed finite pools, generator unchanged; no MIA or C changes',
        inference_scope='Monte Carlo uncertainty conditional on this panel generator and fixed inputs, not official anonymity')
    save(HERE/'結果を見る前の固定方針.json',p['policy'])
    b,pool,targets=GEN.sources(p)
    old_targets=set()
    for q in old['panels']:
        if q['n']==163:
            t=read(q['truth']);old_targets.add(frozenset(t.loc[t.label.eq('member'),'source_row_index']))
    older=load(old['previous_result'])
    for q in older['panels']:
        if q['n']==163:
            t=read(q['truth']);old_targets.add(frozenset(t.loc[t.label.eq('member'),'source_row_index']))
    oldq=next(q for q in old['panels'] if q['id']=='N163_01')
    q,t,_=GEN.aia_panel(pool,targets,163,oldq['seed'])
    pd.testing.assert_frame_equal(q,read(oldq['challenge']))
    pd.testing.assert_frame_equal(t,read(oldq['truth']),check_dtype=False,atol=1e-14,rtol=0)
    seen_targets=set();seen_controls=set();overview=[]
    for i,seed in enumerate(SEEDS,1):
        pid=f'F163_{i:03d}';q,t,info=GEN.aia_panel(pool,targets,163,seed)
        ti,ci=validate_panel(q,t,pool,b)
        assert ti not in seen_targets and ti not in old_targets and ci not in seen_controls
        seen_targets.add(ti);seen_controls.add(ci)
        qp=HERE/'public'/pid/'aia_challenge.csv';tp=HERE/'evaluation_private'/pid/'truth.csv'
        out(qp,q);out(tp,t)
        p['panels'].append(dict(id=pid,n=163,seed=seed,challenge=str(qp),truth=str(tp)))
        overview.append(dict(panel=pid,seed=seed,**info))
        progress(i,f'パネル作成 {i}/{COUNT}')
    out(HERE/'パネル構成.csv',overview)
    p['reference_panel']=oldq
    files=list((HERE/'inputs').rglob('*'))+list((HERE/'programs').rglob('*.py'))+list((HERE/'C').glob('*.csv'))+list(HERE.glob('*.py'))
    files += [HERE/'結果を見る前の固定方針.json']
    for q in p['panels']:files += [Path(q['challenge']),Path(q['truth'])]
    files += list((Path(p['kit'])/'starter/src/pwscup2026').rglob('*.py'))+[Path(p['kit'])/'participant_data/scoring_config.yaml']
    p['hashes']={str(f):sha(f) for f in files if f.is_file()}
    p['fingerprint']=hashlib.sha256(json.dumps(p['hashes'],sort_keys=True).encode()).hexdigest()
    p['environment']=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__)
    save(HERE/'計画.json',p)
    save(HERE/'入力検証.json',dict(passed=True,new_panels=COUNT,unique_target_sets=COUNT,unique_control_sets=COUNT,
        old_target_sets_excluded=len(old_targets),generator_reproduction=True,target_strata=[138,12,9,4],seeds_fixed_before_scores=True))
    return p

def cleanup(pid):
    for key in list(sys.modules):
        if key.startswith('search_'+pid+'_'):sys.modules.pop(key,None)
    gc.collect()

def reference_check(p):
    q=p['reference_panel'];old=load(PREVIOUS/'比較結果.json');s=Scorer(dict(p,panel=q));checks=0
    utilities={}
    for c in p['cases']:
        s.shared_c={};s.cost=None
        with np.load(PREVIOUS/'aia'/c['id']/q['id']/'予測.npz') as predictions:
            for a in p['attacks']:
                v=s.predict(a['id'],c['path']);expected=predictions[q['id']+'__'+a['id']]
                np.testing.assert_allclose(v,expected,atol=1e-10,rtol=0)
                np.testing.assert_array_equal(abs(v-s.times)<=.5,abs(expected-s.times)<=.5)
                row=next(r for r in old['aia'] if (r['case'],r['panel'],r['attack'])==(c['id'],q['id'],a['id']))
                assert abs(s.metrics(v)['AIA']-row['AIA'])<EPS;checks+=1
        utilities[c['id']]=s.full_score(read(c['path']),s.b,s.ref,s.cfg)
        assert abs(utilities[c['id']]['utility']-.9219987812309569)<EPS
    baseline=next(r for r in old['new_baselines'] if r['panel']==q['id'])
    assert abs(s.metrics(np.full(326,10.))['AIA']-baseline['constant10']['AIA'])<EPS
    assert abs(s.metrics(s.predict('A12',p['a_bg']))['AIA']-baseline['background1NN']['AIA'])<EPS
    save(HERE/'事前検証.json',dict(passed=True,prediction_checks=checks,baseline_checks=2,utility_checks=2))
    save(HERE/'有用性.json',utilities)
    del s;cleanup(q['id'])

def worker_init(p):
    global STATE
    STATE=p

def score_job(cid,pid):
    p=STATE;c=next(x for x in p['cases'] if x['id']==cid);q=next(x for x in p['panels'] if x['id']==pid)
    folder=HERE/'results'/cid/pid;meta=folder/'結果.json'
    if meta.exists():
        r=load(meta);assert r['fingerprint']==p['fingerprint'] and sha(r['predictions'])==r['predictions_sha256']
        return r
    start=time.monotonic();s=Scorer(dict(p,panel=q));pred={};rows=[];baseline=None
    if cid=='C27':
        pred['constant10']=np.full(326,10.);pred['background1NN']=s.predict('A12',p['a_bg'])
        baseline={k:s.metrics(v) for k,v in pred.items()}
    s.shared_c={};s.cost=None
    for a in p['attacks']:
        v=s.predict(a['id'],c['path']);pred[a['id']]=v
        rows.append(dict(case=cid,panel=pid,attack=a['id'],**s.metrics(v)))
    folder.mkdir(parents=True,exist_ok=True);fp=folder/'予測.npz';np.savez_compressed(fp,**pred)
    r=dict(case=cid,panel=pid,rows=rows,baseline=baseline,seconds=time.monotonic()-start,
           fingerprint=p['fingerprint'],predictions=str(fp),predictions_sha256=sha(fp))
    save(meta,r);del s;cleanup(pid);return r

def summary(v):
    v=np.asarray(v,float)
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),minimum=float(v.min()),maximum=float(v.max()),
                p90=float(np.quantile(v,.9)),p95=float(np.quantile(v,.95)),zero_count=int((abs(v)<EPS).sum()))

def aggregate(p,results):
    results=sorted(results,key=lambda r:(r['case'],r['panel']))
    allrows=[x for r in results for x in r['rows']]
    assert len(allrows)==7200 and len({(r['case'],r['panel'],r['attack']) for r in allrows})==7200
    bm={r['panel']:r['baseline'] for r in results if r['case']=='C27'}
    panelrows=[];baselines=[]
    for q in p['panels']:
        pid=q['id'];constant=bm[pid]['constant10']['AIA'];bg=bm[pid]['background1NN']['AIA'];floor=max(constant,bg)
        baselines.append(dict(panel=pid,constant10=constant,background1NN=bg,baseline_max=floor))
        for c in p['cases']:
            rr=[r for r in allrows if (r['case'],r['panel'])==(c['id'],pid)];assert len(rr)==12
            a=max(r['AIA'] for r in rr)
            panelrows.append(dict(case=c['id'],name=c['name'],panel=pid,raw=a,constant10=constant,background1NN=bg,baseline_max=floor,
                adjusted=max(0,a-floor),minus_constant=max(0,a-constant),minus_background=max(0,a-bg),minus_phi=max(0,a-p['phi']),
                worst_attacks=' / '.join(r['attack'] for r in rr if abs(r['AIA']-a)<EPS)))
    summaryrows=[];paired=[];tests={};tails=[]
    metrics=['raw','adjusted','minus_constant','minus_background','minus_phi']
    index={(r['case'],r['panel']):r for r in panelrows}
    for c in p['cases']:
        for metric in metrics:
            vals=[index[c['id'],q['id']][metric] for q in p['panels']]
            summaryrows.append(dict(case=c['id'],name=c['name'],metric=metric,**summary(vals)))
        for threshold in p['policy']['exploratory_tail_thresholds']:
            k=sum(index[c['id'],q['id']]['adjusted']>threshold for q in p['panels'])
            ci=stats.binomtest(k,COUNT).proportion_ci(confidence_level=.95,method='exact')
            tails.append(dict(case=c['id'],name=c['name'],threshold=threshold,count=k,total=COUNT,rate=k/COUNT,ci95_low=float(ci.low),ci95_high=float(ci.high)))
    for q in p['panels']:
        a=index['C27',q['id']];b=index['C26',q['id']]
        paired.append(dict(panel=q['id'],Zhang04_adjusted=a['adjusted'],Zhang11_adjusted=b['adjusted'],
            difference=b['adjusted']-a['adjusted'],raw_difference=b['raw']-a['raw']))
    for metric in metrics:
        diffs=np.array([index['C26',q['id']][metric]-index['C27',q['id']][metric] for q in p['panels']])
        mean=float(diffs.mean());se=float(stats.sem(diffs));half=float(stats.t.ppf(.975,COUNT-1)*se)
        rng=np.random.default_rng(p['policy']['bootstrap_seed'])
        boot=diffs[rng.integers(0,COUNT,size=(p['policy']['bootstrap_replicates'],COUNT))].mean(axis=1)
        tests[metric]=dict(mean_difference_11_minus_04=mean,sd=float(diffs.std(ddof=1)),standard_error=se,
            ci95_low=mean-half,ci95_high=mean+half,p_two_sided=float(stats.ttest_1samp(diffs,0).pvalue),
            bootstrap95_low=float(np.quantile(boot,.025)),bootstrap95_high=float(np.quantile(boot,.975)),
            Zhang11_better=int((diffs<-EPS).sum()),ties=int((abs(diffs)<=EPS).sum()),Zhang04_better=int((diffs>EPS).sum()),
            primary=metric=='adjusted')
    attackrows=[]
    for c in p['cases']:
        for a in p['attacks']:
            vals=[r['AIA'] for r in allrows if r['case']==c['id'] and r['attack']==a['id']]
            attackrows.append(dict(case=c['id'],name=c['name'],attack=a['id'],attack_name=a['name'],**summary(vals)))
    result=dict(cases=p['cases'],panels=p['panels'],summary=summaryrows,paired_tests=tests,tail_rates=tails,baselines=baselines,
        panel_results=panelrows,aia=allrows,paired_differences=paired,attack_summary=attackrows,policy=p['policy'],elapsed_seconds=time.monotonic()-START)
    save(HERE/'比較結果.json',result)
    for name,rows in [('候補別要約',summaryrows),('パネル別比較',panelrows),('2案の対応差',paired),('各パネルの基準値',baselines),
                      ('全7200採点',allrows),('手法別要約',attackrows),('高スコアの発生率',tails)]:out(HERE/'集計'/(name+'.csv'),rows)
    return result

def verify(p,results,d):
    checked=0
    for q in p['panels']:
        t=read(q['truth']);truth=t.time.to_numpy(float);m=t.label.eq('member').to_numpy();weights=t.weight_163.to_numpy(float)
        for cid in ['C27','C26']:
            r=next(r for r in results if (r['case'],r['panel'])==(cid,q['id']))
            assert sha(r['predictions'])==r['predictions_sha256']
            with np.load(r['predictions']) as z:
                for aid in z.files:
                    pred=z[aid];assert pred.shape==(326,) and np.isfinite(pred).all() and ((pred>=0)&(pred<=10)).all()
                    hit=abs(pred-truth)<=.5;pm=float(np.average(hit[m],weights=weights[m]));pc=float(np.average(hit[~m],weights=weights[~m]))
                    value=float(np.clip((pm-pc)/max(1-pc,1e-9),0,1))
                    saved=r['baseline'][aid] if aid in ['constant10','background1NN'] else next(x for x in r['rows'] if x['attack']==aid)
                    assert abs(value-saved['AIA'])<EPS
                    assert int(hit[m].sum())==saved['target_hits'] and int(hit[~m].sum())==saved['control_hits'];checked+=1
    assert checked==7800
    for f,h in {**p['source_hashes'],**p['hashes']}.items():assert sha(f)==h,f
    a=next(x for x in d['summary'] if x['case']=='C27' and x['metric']=='adjusted')
    b=next(x for x in d['summary'] if x['case']=='C26' and x['metric']=='adjusted')
    assert abs((b['mean']-a['mean'])-d['paired_tests']['adjusted']['mean_difference_11_minus_04'])<EPS
    return dict(passed=True,panels=300,cases=2,attack_scores=7200,baseline_scores=600,independently_recomputed_scores=checked,
        all_input_hashes_unchanged=True,reference_prediction_checks=24,reference_baseline_checks=2,
        old_scores_not_in_primary=True,official_anonymity_score=False,elapsed_seconds=time.monotonic()-START)

def report(p,d,v):
    x=d['paired_tests']['adjusted'];a=next(r for r in d['summary'] if r['case']=='C27' and r['metric']=='adjusted');b=next(r for r in d['summary'] if r['case']=='C26' and r['metric']=='adjusted')
    conclusion=('Zhang11-HCv2-1の平均が低い（事前指定の両側5%検定で有意）。' if x['ci95_high']<0 else
                'Zhang04-HCv2-3の平均が低い（事前指定の両側5%検定で有意）。' if x['ci95_low']>0 else '平均差の95%信頼区間が0を含むため、今回の300組では優劣を確定しない。')
    lines=['# 163人パネル300組：2案のAIA比較','',f'**{conclusion}**','',
        '## 前提','',
        '- 2案のC・12攻撃・本戦B・A_bgを固定。新しい163標的＋163対照を300組作成。MIAの追加計算・Cの変更は行わない。【実行記録】',
        '- 標的は通常未発症138・通常発症12・希少未発症9・希少発症4。希少発症4人は毎回同じ。対照生成は前回と同じ。【模擬実験の設計】',
        '- Uは2案とも0.9219987812309569。今回の自己採点と以前のCodabench提出値が一致し、CのSHA-256も一致。【公式スコア・実行記録】','',
        '## 仮説','',
        '- 主比較は「Zhang11案−Zhang04案」の基準差引後AIAの平均差。負なら11案が良い。仮説・300組・seed・集計法を新規採点前に固定した。',
        '- 以前の20組は候補を選ぶために見た結果なので、今回の主比較に混ぜない。','',
        '## 分析方法','',
        '- 各パネルの基準値＝max（定数10回答のAIA、A_bgだけによる公式参照1NNのAIA）。',
        '- 基準差引後AIA＝max（0、12攻撃の最大AIA−そのパネルの基準値）。低いほど良い。公式φとは別の比較用基準。',
        '- 同じパネルの2案を対応付け、差の平均・両側95% t信頼区間・両側p値を算出。パネル単位20,000回の対応ブートストラップは感度確認。',
        '- 分布の中央値・最小・最大・90/95パーセンタイル、0.05/0.10/0.20超の頻度は副指標。副指標で主判定を後から置き換えない。',
        '- 16並列、Zhang04→Zhang11。読込・距離など12攻撃の共通処理を共有。基準値は1回だけ計算して2案で共用。',
        '- 別々の乱数で繰り返すことで測るのは、この有限の入力・生成方法に条件付けた変動。人の再登場はあり、新しい独立した300集団ではない。',
        '- 本番の標的選定・非公開対照・希少AIA・未知攻撃の違いは含まない。ここから本番の匿名性得点を算出しない。',
        '- 再集計: `python3 分析.py --aggregate-only`。計算の再開: `python3 分析.py --run`。進捗表示: `進捗を見る.command`。実行には記録したPython環境が必要。','',
        '## 結論','',conclusion,'',
        '| 基準差引後AIA（低い方が良い） | Zhang04-HCv2-3 | Zhang11-HCv2-1 |','|---|---:|---:|']
    for label,key in [('平均','mean'),('中央値','median'),('最小','minimum'),('最大','maximum'),('90パーセンタイル','p90'),('95パーセンタイル','p95'),('0の組数','zero_count')]:
        lines.append(f'| {label} | {a[key]:.6f} | {b[key]:.6f} |')
    lines += ['',f'- 平均差（11−04）: **{x["mean_difference_11_minus_04"]:.6f}**。95%信頼区間 **[{x["ci95_low"]:.6f}, {x["ci95_high"]:.6f}]**。p={x["p_two_sided"]:.6g}。',
        f'- ブートストラップ95%区間: [{x["bootstrap95_low"]:.6f}, {x["bootstrap95_high"]:.6f}]。',
        f'- 11案が低い {x["Zhang11_better"]}組、同点 {x["ties"]}組、04案が低い {x["Zhang04_better"]}組。',
        f'- 実測時間: {v["elapsed_seconds"]/60:.2f}分。7,200攻撃＋600基準の全採点を保存予測から独立再計算し、入力不変を確認。','',
        '## 詳細','',
        '- [全指標の要約](集計/候補別要約.csv) / [パネル別比較](集計/パネル別比較.csv) / [同じパネルでの差](集計/2案の対応差.csv)',
        '- [基準値](集計/各パネルの基準値.csv) / [手法別要約](集計/手法別要約.csv) / [全7200採点](集計/全7200採点.csv) / [高スコアの頻度](集計/高スコアの発生率.csv)',
        '- [事前固定した方針](結果を見る前の固定方針.json) / [検証結果](検証結果.json)',
        '- 統計式: [NIST・対応のある平均差の信頼区間](https://itl.nist.gov/div898/handbook/prc/section3/prc312.htm)。','']
    (HERE/'説明.md').write_text('\n'.join(lines))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--aggregate-only',action='store_true');args=ap.parse_args()
    with (HERE/'実行.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p=prepare();results=[]
            if args.aggregate_only:
                for c in p['cases']:
                    for q in p['panels']:results.append(load(HERE/'results'/c['id']/q['id']/'結果.json'))
            else:
                progress(COUNT,'既存パネルとの予測一致を確認');reference_check(p);done=COUNT
                ctx=multiprocessing.get_context('spawn')
                for c in p['cases']:
                    with ProcessPoolExecutor(max_workers=16,mp_context=ctx,initializer=worker_init,initargs=(p,)) as pool:
                        jobs=[pool.submit(score_job,c['id'],q['id']) for q in p['panels']]
                        n=0
                        for f in as_completed(jobs):
                            results.append(f.result());n+=1;done+=14 if c['id']=='C27' else 12
                            progress(done,f'{c["name"]} {n}/{COUNT}組')
            d=aggregate(p,results);progress(TOTAL-1,'全300組の集計完了・最終検証')
            v=verify(p,results,d);save(HERE/'検証結果.json',v);report(p,d,v)
            progress(TOTAL,'完了');print(json.dumps(d['paired_tests']['adjusted'],ensure_ascii=False,indent=2),flush=True)
        except BaseException as e:
            save(HERE/'エラー.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            raise

if __name__=='__main__':main()
