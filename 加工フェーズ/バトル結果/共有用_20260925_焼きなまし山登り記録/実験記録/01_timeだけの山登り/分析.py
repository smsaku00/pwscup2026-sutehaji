"""Fixed-panel strict hill climbing. --run executes; default verifies saved results."""
from pathlib import Path
import os
for env_name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[env_name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys,json,hashlib,importlib.util,subprocess,shutil,time,csv,traceback,argparse,fcntl
from collections import Counter
import numpy as np
import pandas as pd
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PREV=ROOT/'本戦/バトル/0923'
OLD=PREV/'神奈川387人固定_3案評価'
SOURCE=PREV/'AIA増分評価と改善'
PYTHON=ROOT/'スターターキット/初版/starter/.venv/bin/python'
BASES=['Zhang11','Zhang04','Seo03']
ATTACKS=['A02','A06','A07']
SHORT={'A02':'TT全体対応','A06':'Seo全体割当','A07':'Seo容量付き'}
COLS=['age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG','time','onset','death']
SETTINGS=dict(attribute='time',panel='old01',attacks=ATTACKS,weighting='138/12/9/4 over all 387 targets',
    objective='max weighted whole AIA of three attacks; no background subtraction',
    max_steps=500,patience=100,seed=2026092401,strict_improvement=True,tolerance=1e-12,
    utility_floor='each original saved C U',constant_attack_in_objective=False,
    pair_selection='unequal times, same onset/death, excluding (0,0), without replacement per current C',
    independent_validation=False,official_phi_applied=False)
LOG=None

def read(p):return pd.read_csv(p,float_precision='round_trip')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def out(p,df):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p,index=False,encoding='utf-8-sig',float_format='%.17g')
def say(message):
    line=f'[{time.strftime("%H:%M:%S")}] {message}'
    print(line,flush=True)
    if LOG:LOG.write(line+'\n');LOG.flush()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def prepare():
    pp=HERE/'計画.json'
    if pp.exists():
        p=load(pp);assert p['settings']==SETTINGS
        for f,h in p['hashes'].items():assert sha(f)==h,f
        return p
    old=load(SOURCE/'計画.json');panel=next(p for p in old['panels'] if p['id']=='old01')
    variants=[];attacks={};hashes={}
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        hashes[str(src)]=sha(src);hashes[str(dst)]=sha(dst);return str(dst)
    for v in old['variants'][:3]:
        assert v['label'] in BASES
        path=copy(Path(v['path']),HERE/'bases'/v['label']/'C.csv')
        variants.append(dict(v,path=path,source=v['path']))
    for aid in ATTACKS:
        a=next(a for a in old['attacks'] if a['id']==aid);src=PREV/a['path'];assert sha(src)==a['sha256']
        dest=HERE/'programs'/aid/src.name;copy(src,dest)
        for asset in a['assets']:
            assert sha(src.parent/asset['name'])==asset['sha256']
            copy(src.parent/asset['name'],dest.parent/asset['name'])
        attacks[aid]=dict(a,local_path=str(dest))
    for key,folder in [('challenge','public'),('truth','evaluation_private')]:
        panel[key]=copy(Path(panel[key]),HERE/folder/Path(panel[key]).name)
    bg=copy(Path(old['a_bg']),HERE/'public/A_bg.csv')
    attack_b=copy(OLD/'public/B_attacker.csv',HERE/'public/B_attacker.csv')
    for f in [Path(old['b']),Path(old['b']).with_name('B_self.csv'),Path(old['b']).with_name('utility_ref.json')]:hashes[str(f)]=sha(f)
    kit=Path(old['kit'])
    for f in list((kit/'starter/src/pwscup2026').rglob('*.py'))+[kit/'participant_data/scoring_config.yaml']:hashes[str(f)]=sha(f)
    p=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),settings=SETTINGS,variants=variants,
        attacks=attacks,panel=panel,a_bg=bg,attack_b=attack_b,b=old['b'],kit=old['kit'],hashes=hashes,
        attack_selection='A02/A06/A07 jointly attain each original base maximum over all 12 attacks on old01; fixed before hill climbing')
    save(pp,p);return p

class Scorer:
    def __init__(self,p):
        self.p=p;self.t=read(p['panel']['truth']);self.m=self.t.label.eq('member').to_numpy()
        self.w=self.t.weight_163.to_numpy();self.times=self.t.time.to_numpy();assert self.m.sum()==387 and (~self.m).sum()==387
        np.testing.assert_allclose([self.w[self.m].sum(),self.w[~self.m].sum()],[163,163],atol=1e-10)
        self.mods={aid:module('hill_'+aid,Path(a['local_path'])) for aid,a in p['attacks'].items()}
        self.tt=self.mods['A02'];self.core=self.mods['A06'];self.relaxed=self.mods['A07']
        self.q=self.core.read_table(Path(p['panel']['challenge']),'challenge')
        self.ttq=self.tt.read_table(Path(p['panel']['challenge']),'challenge')[0]
        self.bg=self.core.read_table(Path(p['a_bg']),'A');self.transform=self.core.BackgroundTransform(self.bg)
        self.ttbg=self.tt.read_table(Path(p['a_bg']),'background')[0];self.scale=self.tt.scale_from_background(self.ttbg)
        sys.path.insert(0,str(Path(p['kit'])/'starter/src'))
        from pwscup2026.kit.selfscore import self_score,_canonicalize_c
        from pwscup2026.scoring import utility
        self.full_score=self_score;self.canonicalize=_canonicalize_c;self.official=utility
        self.cfg=yaml.safe_load((Path(p['kit'])/'participant_data/scoring_config.yaml').read_text())
        self.b=read(p['b']);bs=read(Path(p['b']).with_name('B_self.csv'));r=load(Path(p['b']).with_name('utility_ref.json'))
        self.ref=utility.UtilityReference(mu=np.array(r['mu']),sigma=np.array(r['sigma']),band=r['band'],b_is_rare=bs.is_rare.to_numpy(bool),
            km_signal_D=r['km_signal_D'],km_floor=r['km_floor'],a_bg=self.bg)
        self.bp=utility._prepare(self.b)
    def full_u(self,c):return self.full_score(c,self.b,self.ref,self.cfg)
    def fast_u(self,c,base):
        cp=self.official._prepare(self.canonicalize(c));fac=base['facets'].copy()
        fac['U_spec']=self.official._score_u_spec(cp,self.bp)[0]
        fac['U_valid']=self.official._score_u_valid(cp,self.bp,self.ref,self.cfg)[0]
        assert np.isfinite(list(fac.values())).all()
        return dict(utility=min(fac.values()),facets=fac)
    def metrics(self,pred):
        hits=np.abs(pred-self.times)<=.5
        pm=float(np.average(hits[self.m],weights=self.w[self.m]));pc=float(np.average(hits[~self.m],weights=self.w[~self.m]))
        r=float(np.clip((pm-pc)/max(1-pc,1e-9),0,1))
        return dict(AIA=r,target_rate=pm,control_rate=pc,target_hits=int(hits[self.m].sum()),control_hits=int(hits[~self.m].sum())),hits
    def evaluate(self,c):
        # Match the original CLI's parsing, sorting and output precision.
        tc=c[COLS].copy()
        for col in COLS:
            if col!='prefecture':tc[col]=pd.to_numeric(tc[col].map(lambda x:format(x,'.17g'))).to_numpy(float)
        tc=tc.sort_values(COLS,kind='stable').reset_index(drop=True)
        pred={
            'A02':np.array([float(format(x,self.tt.FLOAT_FORMAT)) for x in self.tt.predict(tc,self.ttq,self.ttbg,self.scale)[0]]),
            'A06':self.core.aia_times(self.q,tc,self.transform)[0],
            'A07':self.relaxed.relaxed_times(self.q,tc,self.transform,2)[0]}
        scores={};hits={}
        for aid,x in pred.items():
            assert len(x)==774 and np.isfinite(x).all() and (x>=0).all() and (x<=10).all()
            scores[aid],hits[aid]=self.metrics(x)
        value=max(z['AIA'] for z in scores.values())
        return dict(value=value,scores=scores,pred=pred,hits=hits)

def compact(z):return dict(value=z['value'],scores=z['scores'])
def read_prediction(file,p):
    d=read(file);d=d[d.target_k.eq(p['panel']['target_id'])]
    q=read(p['panel']['challenge']);assert d.challenge_row_id.is_unique and set(d.challenge_row_id)==set(q.challenge_row_id)
    return d.set_index('challenge_row_id').loc[q.challenge_row_id,'time_hat'].to_numpy(float)
def cli(p,cp,dest,aid):
    a=p['attacks'][aid];src=Path(a['local_path']);dest.mkdir(parents=True,exist_ok=True)
    meta=dest/'実行記録.json';inputs=dict(C=sha(cp),challenge=sha(p['panel']['challenge']),background=sha(p['a_bg']),code=sha(src))
    if meta.exists():
        r=load(meta);assert r['inputs']==inputs and r['returncode']==0 and sha(r['prediction'])==r['prediction_sha256']
        return read_prediction(r['prediction'],p)
    shutil.copyfile(src,dest/src.name)
    for asset in a['assets']:shutil.copyfile(src.parent/asset['name'],dest/asset['name'])
    save(dest/'targets.json',[dict(target_k=p['panel']['target_id'],c_path=str(cp),aia_challenge_path=p['panel']['challenge'])])
    cmd=[str(PYTHON),'-B',str(dest/src.name),'--b',p['attack_b'],'--a-bg',p['a_bg'],'--targets',str(dest/'targets.json')]
    with (dest/'実行.log').open('w') as stream:r=subprocess.run(cmd,cwd=dest,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
    pred=dest/('F_aia_'+src.stem+'.csv');assert r.returncode==0 and pred.exists(),str(dest/'実行.log')
    save(meta,dict(inputs=inputs,returncode=r.returncode,prediction=str(pred),prediction_sha256=sha(pred),command=cmd))
    return read_prediction(pred,p)

def pairs(c,rng):
    result=[]
    for key,idx in c.groupby(['onset','death']).groups.items():
        if key==(0,0):continue
        ids=np.array(idx);i,j=np.triu_indices(len(ids),1)
        result.extend((int(ids[x]),int(ids[y])) for x,y in zip(i,j) if c.time.iloc[ids[x]]!=c.time.iloc[ids[y]])
    rng.shuffle(result);return result

def check_predictions(s,actual,expected):
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
    np.testing.assert_array_equal(np.abs(actual-s.times)<=.5,np.abs(expected-s.times)<=.5)

def verify_implementation(p,s):
    say('事前検算：元C・1交換後のCを、元の攻撃プログラムと照合します。')
    records=[]
    for v in p['variants']:
        name=v['label'];c=read(v['path']);assert not c[COLS[:10]].duplicated().any()
        u=s.full_u(c);np.testing.assert_allclose(u['utility'],v['utility']['U'],atol=1e-12,rtol=0)
        z=s.evaluate(c)
        for aid in ATTACKS:
            r=load(OLD/'work/panel_01'/name/aid/'実行記録.json');assert sha(r['prediction'])==r['prediction_sha256']
            check_predictions(s,z['pred'][aid],read_prediction(r['prediction'],p))
        i,j=pairs(c,np.random.default_rng(SETTINGS['seed']))[0];c.loc[[i,j],'time']=c.loc[[j,i],'time'].to_numpy()
        cp=HERE/'verification'/name/'swap.csv';out(cp,c);c=read(cp);z2=s.evaluate(c);uf=s.full_u(c);uq=s.fast_u(c,u)
        for key in uf['facets']:np.testing.assert_allclose(uf['facets'][key],uq['facets'][key],atol=1e-12,rtol=0)
        for aid in ATTACKS:check_predictions(s,z2['pred'][aid],cli(p,cp,HERE/'verification'/name/aid,aid))
        records.append(dict(base=name,prediction_comparisons=6,hit_masks_equal=True,full_and_fast_utility_equal=True,initial=compact(z)))
        say(f'検算OK {name} | 初期AIA={z["value"]:.6f} | U={u["utility"]:.6f}')
    save(HERE/'事前検証.json',dict(passed=True,engine_sha256=sha(__file__),records=records))

def save_predictions(p,path,z):
    rows=[];q=read(p['panel']['challenge'])
    for aid in ATTACKS:
        rows.extend(dict(attack=aid,challenge_row_id=int(rid),time_hat=float(x)) for rid,x in zip(q.challenge_row_id,z['pred'][aid]))
    out(path,pd.DataFrame(rows))

def run_base(p,s,v):
    name=v['label'];dst=HERE/'results'/name;result_path=dst/'結果.json'
    if result_path.exists():
        result=load(result_path);assert result['engine_sha256']==sha(__file__) and sha(result['C_path'])==result['C_sha256']
        say(f'{name}: 保存済みの完了結果を使用します。');return result
    dst.mkdir(parents=True,exist_ok=True);base=read(v['path']);cur=base.copy();ubase=s.full_u(base);floor=ubase['utility'];cur_u=ubase
    current=s.evaluate(cur);initial=current;save(dst/'開始時.json',dict(U=ubase,attack=compact(initial)))
    rng=np.random.default_rng(SETTINGS['seed']);remaining=pairs(cur,rng);initial_pairs=len(remaining)
    unchanged=0;accepted=0;rows=[];start=time.monotonic();constant=s.metrics(np.full(774,10.))[0]['AIA']
    say(f'開始 {name} | U下限={floor:.6f} | AIA={initial["value"]:.6f} | 定数10参考={constant:.6f} | 交換候補={initial_pairs}組')
    with (dst/'試行履歴.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=None
        for step in range(1,SETTINGS['max_steps']+1):
            if not remaining:break
            i,j=remaining.pop();cand=cur.copy();old_i=float(cur.time.iloc[i]);old_j=float(cur.time.iloc[j]);cand.loc[[i,j],'time']=[old_j,old_i]
            u=s.fast_u(cand,ubase);before=current['value'];z=None;adopt=False;changed_answers=changed_hits=None;diagnostic=''
            if u['utility']<floor-SETTINGS['tolerance']:
                reason='U不足';diagnostic=min(u['facets'],key=u['facets'].get)
            else:
                z=s.evaluate(cand);changed_answers=sum(int((z['pred'][a]!=current['pred'][a]).sum()) for a in ATTACKS)
                changed_hits=sum(int((z['hits'][a]!=current['hits'][a]).sum()) for a in ATTACKS)
                if z['value']<before-SETTINGS['tolerance']:
                    reason='採用';adopt=True
                elif z['value']>before+SETTINGS['tolerance']:reason='AIA悪化'
                else:
                    reason='AIA同点'
                    diagnostic='回答不変' if changed_answers==0 else '的中判定不変' if changed_hits==0 else '的中は変化・最大値不変'
            if adopt:
                cur=cand;current=z;cur_u=u;accepted+=1;unchanged=0;remaining=pairs(cur,rng)
                out(dst/'途中のC.csv',cur)
            else:unchanged+=1
            row=dict(ベース=name,試行=step,行1=i+1,行2=j+1,行1の旧time=old_i,行2の旧time=old_j,
                U=u['utility'],元U下限=floor,変更前AIA=before,提案AIA=None if z is None else z['value'],採否=reason,詳細=diagnostic,
                採用累計=accepted,連続非改善=unchanged,回答変更数_3攻撃延べ=changed_answers,的中変更数_3攻撃延べ=changed_hits,
                現在AIA=current['value'],経過秒=time.monotonic()-start)
            row.update({'U_'+k:val for k,val in u['facets'].items()})
            for aid in ATTACKS:
                row[aid+'_変更前']=initial['scores'][aid]['AIA'] if step==1 else rows[-1][aid+'_現在']
                for key in ['AIA','target_rate','control_rate','target_hits','control_hits']:
                    row[aid+'_'+key]=None if z is None else z['scores'][aid][key]
                row[aid+'_現在']=current['scores'][aid]['AIA']
            if writer is None:writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader()
            writer.writerow(row);stream.flush();rows.append(row)
            value='採点省略' if z is None else f'{z["value"]:.6f}'
            say(f'{name} {step:03d}/500 | 行{i+1}↔{j+1} | U={u["utility"]:.6f} | AIA {before:.6f}→{value} | {reason} {diagnostic} | 非改善{unchanged}/100')
            save(dst/'進捗.json',dict(base=name,step=step,accepted=accepted,no_improvement=unchanged,current_AIA=current['value'],seconds=time.monotonic()-start))
            if unchanged>=SETTINGS['patience']:break
    reason='100回連続改善なし' if unchanged>=SETTINGS['patience'] else '上限500回' if len(rows)>=SETTINGS['max_steps'] else '交換候補を一巡'
    pd.testing.assert_frame_equal(base.drop(columns='time'),cur.drop(columns='time'))
    for _,idx in base.groupby(['onset','death']).groups.items():np.testing.assert_array_equal(np.sort(base.loc[idx,'time']),np.sort(cur.loc[idx,'time']))
    cp=dst/'C.csv';out(cp,cur);saved=read(cp);full=s.full_u(saved)
    for key in full['facets']:np.testing.assert_allclose(full['facets'][key],cur_u['facets'][key],atol=1e-12,rtol=0)
    assert full['utility']>=floor-1e-12
    final=s.evaluate(saved);np.testing.assert_allclose(final['value'],current['value'],atol=1e-12,rtol=0)
    say(f'{name}: 保存Cの全有用性と元CLIの3攻撃を再確認しています。')
    for aid in ATTACKS:check_predictions(s,final['pred'][aid],cli(p,cp,dst/'final_cli'/aid,aid))
    save(dst/'有用性.json',full);save(dst/'最終AIA.json',compact(final));save_predictions(p,dst/'最終回答.csv',final)
    last=rows[-unchanged:] if unchanged else []
    result=dict(base=name,steps=len(rows),accepted=accepted,stop_reason=reason,no_improvement=unchanged,
        initial_AIA=initial['value'],final_AIA=final['value'],initial_U=floor,final_U=full['utility'],
        changed_rows=int((saved.time!=base.time).sum()),initial_pair_count=initial_pairs,remaining_pairs=len(remaining),
        constant10=constant,final_with_constant=max(constant,final['value']),
        decisions=dict(Counter(r['採否'] for r in rows)),terminal_decisions=dict(Counter(r['採否'] for r in last)),
        tie_diagnostics=dict(Counter(r['詳細'] for r in rows if r['採否']=='AIA同点')),
        terminal_tie_diagnostics=dict(Counter(r['詳細'] for r in last if r['採否']=='AIA同点')),
        initial_attacks=initial['scores'],final_attacks=final['scores'],seconds=time.monotonic()-start,
        C_path=str(cp),C_sha256=sha(cp),engine_sha256=sha(__file__),cli_verified=True,
        exhausted_neighborhood=not bool(remaining))
    save(result_path,result)
    say(f'停止 {name} | {reason} | 採用{accepted}回/{len(rows)}回 | AIA {initial["value"]:.6f}→{final["value"]:.6f} | U={full["utility"]:.6f}')
    return result

def report(results):
    rows=[]
    for r in results:
        rows.append(dict(ベース=r['base'],試行回数=r['steps'],採用回数=r['accepted'],元AIA=r['initial_AIA'],最終AIA=r['final_AIA'],
            元U=r['initial_U'],最終U=r['final_U'],変更行数=r['changed_rows'],U不足=r['decisions'].get('U不足',0),
            AIA同点=r['decisions'].get('AIA同点',0),AIA悪化=r['decisions'].get('AIA悪化',0),停止理由=r['stop_reason']))
    out(HERE/'3ベース比較.csv',pd.DataFrame(rows))
    lines=['# timeだけを動かす山登り法：3ベースの比較','',
        '- 【ローカル模擬実験】元のZhang11・Zhang04・Seo03から開始。前回の焼きなまし候補は使っていません。',
        '- 1回に2行のtimeだけを交換し、Uを元C以上に保ち、3攻撃の最大AIAが下がった場合だけ採用。同点・悪化は不採用です。',
        '- 各ベース最大500回、100回連続で改善がなければ停止します。停止は探索予算上の目安であり、局所最適や改善不能の証明ではありません。','',
        '| ベース | 試行／採用 | AIA：元→最後 | U：元→最後 | time変更行数 |',
        '| --- | ---: | --- | --- | ---: |']
    for r in results:lines.append(f'| {r["base"]} | {r["steps"]}／{r["accepted"]} | {r["initial_AIA"]:.6f} → {r["final_AIA"]:.6f} | {r["initial_U"]:.6f} → {r["final_U"]:.6f} | {r["changed_rows"]} |')
    lines+=['','## どこで詰まったか','','| ベース | U不足 | AIA同点 | AIA悪化 | 停止理由 |','| --- | ---: | ---: | ---: | --- |']
    for r in results:
        d=r['decisions'];lines.append(f'| {r["base"]} | {d.get("U不足",0)} | {d.get("AIA同点",0)} | {d.get("AIA悪化",0)} | {r["stop_reason"]} |')
    for r in results:
        lines+=['',f'**{r["base"]}**',f'- 最後の連続非改善{r["no_improvement"]}回：'+ '、'.join(f'{k}{v}回' for k,v in r['terminal_decisions'].items())+'。',
            '- そのうち同点の内訳：'+('、'.join(f'{k}{v}回' for k,v in r['terminal_tie_diagnostics'].items()) or 'なし')+'。',
            f'- 最後のCからの交換候補は、まだ{r["remaining_pairs"]:,}組未試行です。',
            f'- [候補C](results/{r["base"]}/C.csv)、[全試行の履歴](results/{r["base"]}/試行履歴.csv)、[停止時の詳細](results/{r["base"]}/結果.json)']
    lines+=['','## 採点した条件','',
        '- パネルは既存387人方式の1組目（old01）を固定。神奈川県の標的387人＋対照387人、計774行です。Zhangさん作成のパネルではありません。',
        '- 的中率は通常未発症・通常発症・希少未発症・希少発症の比率を138：12：9：4へ重み付けします。対照は対応する標的の重みを引き継ぎ、真の希少区分は不明です。',
        '- 3攻撃はTT-02全体対応推定、seo-03正則化生成比と全体割当、seo-04b容量付き全体割当AIA。元Cの保存済み12攻撃結果から、3ベースそれぞれの最大値を捉える共通の3攻撃として、探索前に固定しました。',
        '- 目的値は全体AIAの3攻撃中最大だけ。背景との差、群別増分、平均を足した前回の合成指標は使いません。',
        '- 前回の増分用3攻撃では、このパネルのZhang11・Zhang04が最初からAIA=0でした。攻撃を替えた理由は、今回のAIA最大値という目的に合わせるためです。',
        f'- 常に10年と答える参考値は{results[0]["constant10"]:.6f}。目的値には含めず、結果JSONに定数込み最大も保存しています。',
        '- 同じ発症・死亡区分内で時刻が異なる行を選び、非発症・非死亡行は固定。現在のCが変わらない間は同じ交換ペアを重複試行しません。採用したら交換候補を作り直します。',
        '- 乱数は3ベース共通で2026092401。U_gen・U_rareの不変性を検証し、探索中は変わる項目を再採点。保存Cでは全項目と元CLIの攻撃回答を再確認しました。','',
        '## この実験から言える範囲','',
        '- 「この固定パネル・3攻撃に対し、time交換でどこまで進み、何が採用を妨げたか」の実験です。本番の匿名性得点や採用候補の順位ではありません。',
        '- 1つの乱数・1パネルでの試行です。別の乱数や交換順なら結果が変わり得ます。',
        '- 他パネルへの一般化、他9攻撃、MIAは今回の実験では未確認です。公式φも適用していません。','',
        '## 実行とログ','',
        '- [実行.command](実行.command)をターミナルで実行すると、各試行のU・AIA・採否が表示されます。完了済みなら既存結果を検証して表示します。',
        '- 実行中の別ターミナルから `tail -f 実行.log` で追えます。',
        '- [全体ログ](実行.log)、[比較CSV](3ベース比較.csv)、[実験の説明](説明.md)、[検証記録](検証結果.json)。','']
    (HERE/'README.md').write_text('\n'.join(lines))

def audit(p):
    for file,h in p['hashes'].items():assert sha(file)==h,file
    records=[]
    for name in BASES:
        dst=HERE/'results'/name;r=load(dst/'結果.json');assert r['engine_sha256']==sha(__file__) and sha(r['C_path'])==r['C_sha256']
        base=read(next(v['path'] for v in p['variants'] if v['label']==name));final=read(r['C_path']);hist=read(dst/'試行履歴.csv')
        replay=base.copy();previous=r['initial_AIA'];streak=0;adoptions=0
        for row in hist.to_dict('records'):
            np.testing.assert_allclose(row['変更前AIA'],previous,atol=1e-12,rtol=0)
            if row['採否']=='採用':
                assert row['提案AIA']<previous-1e-12 and row['U']>=r['initial_U']-1e-12
                i=int(row['行1'])-1;j=int(row['行2'])-1
                assert replay.time.iloc[i]==row['行1の旧time'] and replay.time.iloc[j]==row['行2の旧time']
                replay.loc[[i,j],'time']=replay.loc[[j,i],'time'].to_numpy();previous=row['提案AIA'];streak=0;adoptions+=1
            else:streak+=1
            np.testing.assert_allclose(row['現在AIA'],previous,atol=1e-12,rtol=0);assert row['連続非改善']==streak
        pd.testing.assert_frame_equal(replay,final)
        assert adoptions==r['accepted'] and streak==r['no_improvement'] and len(hist)==r['steps']
        assert (len(hist)==500 or streak==100 or r['exhausted_neighborhood'])
        records.append(dict(base=name,replayed_swaps=adoptions,steps=len(hist),monotone=True,strict_acceptance=True,original_columns_unchanged=True))
    result=dict(passed=True,engine_sha256=sha(__file__),source_hashes_unchanged=True,records=records,
        original_C_and_swap_cli_checks=18,final_cli_checks=9,full_saved_C_utility_checked=True,official_Anon_calculated=False)
    save(HERE/'検証結果.json',result);say('最終検証OK：採用した交換の再適用、採否条件、元入力の不変性を確認しました。');return result

def main():
    global LOG
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    HERE.mkdir(parents=True,exist_ok=True)
    with (HERE/'実行.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('既に実行中です。実行.logをご覧ください。',flush=True);return
        with (HERE/'実行.log').open('a') as logfile:
            LOG=logfile;p=prepare()
            if args.run or args.verify_only:
                save(HERE/'状態.json',dict(status='running',pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),engine_sha256=sha(__file__)))
                if all((HERE/'results'/name/'結果.json').exists() for name in BASES) and not args.verify_only:
                    results=[load(HERE/'results'/name/'結果.json') for name in BASES];report(results);audit(p)
                else:
                    scorer=Scorer(p);verify_implementation(p,scorer)
                    if args.verify_only:say('事前検算のみ完了。探索はまだ実行していません。');return
                    results=[run_base(p,scorer,v) for v in p['variants']];report(results);audit(p)
                save(HERE/'状態.json',dict(status='complete',finished=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
                say('3ベースすべて完了。README.md と 3ベース比較.csv に結果を保存しました。')
            else:audit(p)
            LOG=None

if __name__=='__main__':
    try:main()
    except Exception as exc:
        save(HERE/'状態.json',dict(status='error',error=str(exc),traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True);raise SystemExit(1)
