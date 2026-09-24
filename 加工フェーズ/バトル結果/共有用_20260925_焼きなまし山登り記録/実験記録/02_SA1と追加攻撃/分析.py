"""SA1 names and held-out attack evaluation on the unchanged old01 panel.

Runs only the nine attacks not used for hill climbing. Existing validated
predictions supply the original baselines and three optimization attacks.
"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,hashlib,shutil,subprocess,sys,time,traceback,fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
HILL=HERE.parent/'山登り_time_3ベース'
PREV=ROOT/'本戦/バトル/0923'
OLD=PREV/'神奈川387人固定_3案評価'
PYTHON=ROOT/'スターターキット/初版/starter/.venv/bin/python'
BASES=['Zhang11','Zhang04','Seo03']
TRAIN={'A02','A06','A07'}
LOG=None

def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,float_precision='round_trip')
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def out(p,v):pd.DataFrame(v).to_csv(p,index=False,encoding='utf-8-sig',float_format='%.17g')
def say(s):
    line=f'[{time.strftime("%H:%M:%S")}] {s}';print(line,flush=True)
    if LOG:LOG.write(line+'\n');LOG.flush()
def copy(src,dst):
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert sha(src)==sha(dst),str(dst)
    else:shutil.copyfile(src,dst)
    return str(dst)

def prepare():
    if (HERE/'計画.json').exists():
        p=load(HERE/'計画.json')
        for f,h in p['hashes'].items():assert sha(f)==h,f
        assert p['engine_sha256']==sha(__file__)
        return p
    hp=load(HILL/'計画.json');op=load(OLD/'計画.json')
    assert load(HILL/'状態.json')['status']=='complete'
    assert load(HILL/'検証結果.json')['passed']
    for f,h in hp['hashes'].items():assert sha(f)==h,f
    oldpanel=next(p for p in op['panels'] if p['panel']==1)
    for k in ['challenge','truth']:assert sha(hp['panel'][k])==sha(oldpanel[k])
    for k,name in [('a_bg','A_bg.csv'),('attack_b','B_attacker.csv')]:assert sha(hp[k])==sha(OLD/'public'/name)
    files=[HILL/'計画.json',HILL/'検証結果.json',OLD/'計画.json']
    p=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),engine_sha256=sha(__file__),
        panel=hp['panel'],a_bg=hp['a_bg'],attack_b=hp['attack_b'],attacks=op['attacks'],variants=[],
        weighting='138:12:9:4; all 387 targets and 387 paired controls',
        additional_attacks=9,additional_cli_jobs=27,independent_panel=False,
        names_only_no_new_optimization=True,SA1_algorithm='strict hill climbing, not annealing')
    files.extend([Path(p['panel']['challenge']),Path(p['panel']['truth']),Path(p['a_bg']),Path(p['attack_b'])])
    for n in BASES:
        r=load(HILL/'results'/n/'結果.json');assert r['C_sha256']==sha(r['C_path'])
        v=next(v for v in hp['variants'] if v['label']==n)
        src=Path(r['C_path']);cp=copy(src,HERE/'候補C'/f'{n}-SA1.csv')
        u=copy(HILL/'results'/n/'有用性.json',HERE/'有用性'/f'{n}-SA1.json')
        p['variants'].append(dict(base=n,name=n+'-SA1',original=v['path'],candidate=cp,
            source=str(src),accepted=r['accepted'],changed_rows=r['changed_rows'],U=r['final_U'],
            original_U=r['initial_U'],C_sha256=sha(cp),utility_path=u))
        files.extend([Path(v['path']),src,Path(cp),Path(u),HILL/'results'/n/'結果.json'])
    for a in p['attacks']:
        src=PREV/a['path'];assert sha(src)==a['sha256'];files.append(src)
        for asset in a['assets']:
            f=src.parent/asset['name'];assert sha(f)==asset['sha256'];files.append(f)
    p['hashes']={str(f):sha(f) for f in files};save(HERE/'計画.json',p)
    save(HERE/'候補の命名.json',p['variants'])
    return p

def predictions(p,file):
    d=read(file);q=read(p['panel']['challenge'])
    assert set(d.columns)=={'target_k','challenge_row_id','time_hat'}
    assert len(d)==774 and d.target_k.eq(p['panel']['target_id']).all()
    assert d.challenge_row_id.is_unique and set(d.challenge_row_id)==set(q.challenge_row_id)
    x=d.set_index('challenge_row_id').loc[q.challenge_row_id,'time_hat'].to_numpy(float)
    assert np.isfinite(x).all() and (x>=0).all() and (x<=10).all()
    return x

def cached(p,v,a,kind):
    folder=OLD/'work/panel_01'/v['base']/a['id'] if kind=='original' else HILL/'results'/v['base']/'final_cli'/a['id']
    rec=load(folder/'実行記録.json');assert rec['returncode']==0 and sha(rec['prediction'])==rec['prediction_sha256']
    cmd=rec['command'];assert sha(cmd[2])==a['sha256']
    for asset in a['assets']:assert sha(Path(cmd[2]).parent/asset['name'])==asset['sha256']
    cfg=load(cmd[cmd.index('--targets')+1]);assert len(cfg)==1
    assert cfg[0]['target_k']==p['panel']['target_id']
    assert sha(cfg[0]['aia_challenge_path'])==sha(p['panel']['challenge'])
    assert sha(cfg[0]['c_path'])==sha(v['original' if kind=='original' else 'candidate'])
    assert sha(cmd[cmd.index('--a-bg')+1])==sha(p['a_bg'])
    assert sha(cmd[cmd.index('--b')+1])==sha(p['attack_b'])
    dst=HERE/'predictions'/kind/v['name']/a['id']/'F_aia.csv'
    copy(rec['prediction'],dst)
    return dict(path=str(dst),sha256=sha(dst),origin=rec['prediction'],origin_record=str(folder/'実行記録.json'),reused=True)

def job(p,v,a):
    work=HERE/'work'/v['name']/a['id'];work.mkdir(parents=True,exist_ok=True)
    meta=work/'実行記録.json'
    inputs=dict(C=sha(v['candidate']),challenge=sha(p['panel']['challenge']),background=sha(p['a_bg']),
        attack_B=sha(p['attack_b']),code=a['sha256'],assets={x['name']:x['sha256'] for x in a['assets']})
    if meta.exists():
        rec=load(meta);assert rec['inputs']==inputs and rec['returncode']==0 and sha(rec['path'])==rec['sha256'];return rec
    src=PREV/a['path'];script=Path(copy(src,work/src.name))
    for asset in a['assets']:copy(src.parent/asset['name'],work/asset['name'])
    save(work/'targets.json',[dict(target_k=p['panel']['target_id'],c_path=v['candidate'],aia_challenge_path=p['panel']['challenge'])])
    cmd=[str(PYTHON),'-B',str(script),'--b',p['attack_b'],'--a-bg',p['a_bg'],'--targets',str(work/'targets.json')]
    start=time.monotonic()
    with (work/'実行.log').open('w') as stream:
        result=subprocess.run(cmd,cwd=work,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    path=work/('F_aia_'+script.stem+'.csv')
    assert result.returncode==0 and path.exists(),str(work/'実行.log')
    predictions(p,path)
    rec=dict(path=str(path),sha256=sha(path),reused=False,returncode=result.returncode,
        command=cmd,inputs=inputs,seconds=time.monotonic()-start)
    save(meta,rec);return rec

def metrics(pred,truth):
    hit=np.abs(pred-truth.time.to_numpy(float))<=.5
    m=truth.label.eq('member').to_numpy();w=truth.weight_163.to_numpy(float)
    assert m.sum()==387 and (~m).sum()==387
    np.testing.assert_allclose([w[m].sum(),w[~m].sum()],[163,163],rtol=0,atol=1e-10)
    pm=float(np.average(hit[m],weights=w[m]));pc=float(np.average(hit[~m],weights=w[~m]))
    return dict(AIA=float(np.clip((pm-pc)/max(1-pc,1e-9),0,1)),
        target_rate=pm,control_rate=pc,target_hits=int(hit[m].sum()),control_hits=int(hit[~m].sum()))

def collect(p,records):
    truth=read(p['panel']['truth']);q=read(p['panel']['challenge'])
    np.testing.assert_array_equal(q.challenge_row_id,truth.challenge_row_id)
    rows=[];verification=[]
    for v in p['variants']:
        old=read(v['original']);new=read(v['candidate'])
        pd.testing.assert_frame_equal(old.drop(columns='time'),new.drop(columns='time'))
        assert int((old.time!=new.time).sum())==v['changed_rows']
        r=load(HILL/'results'/v['base']/'結果.json')
        assert sha(v['candidate'])==r['C_sha256'] and load(v['utility_path'])['utility']==v['U']
        for a in p['attacks']:
            vals={}
            for kind in ['original','candidate']:
                rec=records[(v['name'],a['id'],kind)];assert sha(rec['path'])==rec['sha256']
                vals[kind]=metrics(predictions(p,rec['path']),truth)
                if a['id'] in TRAIN:
                    expected=r['initial_attacks' if kind=='original' else 'final_attacks'][a['id']]
                    for k in expected:np.testing.assert_allclose(vals[kind][k],expected[k],atol=1e-12,rtol=0)
            row=dict(候補=v['name'],元案=v['base'],攻撃ID=a['id'],攻撃名=a['name'],
                評価区分='山登りに使用' if a['id'] in TRAIN else '追加評価',
                元AIA=vals['original']['AIA'],SA1_AIA=vals['candidate']['AIA'],
                差=vals['candidate']['AIA']-vals['original']['AIA'])
            row['変化']='改善' if row['差']<-1e-12 else '悪化' if row['差']>1e-12 else '同じ'
            for kind,label in [('original','元'),('candidate','SA1')]:
                for k,val in vals[kind].items():row[label+'_'+k]=val
            rows.append(row)
        verification.append(dict(name=v['name'],C_copy_identical=True,other_columns_unchanged=True,
            predictions_checked=24,three_optimization_scores_match=True,U=v['U']))
    assert len(rows)==36
    out(HERE/'全12攻撃_交換前後.csv',rows)
    constant=metrics(np.full(774,10.),truth)
    save(HERE/'定数10年.json',constant)
    df=pd.DataFrame(rows);summaries=[]
    for v in p['variants']:
        sub=df[df.候補.eq(v['name'])];other=sub[sub.評価区分.eq('追加評価')]
        summary=dict(name=v['name'],U=v['U'],additional_improved=int(other.変化.eq('改善').sum()),
            additional_worse=int(other.変化.eq('悪化').sum()),additional_same=int(other.変化.eq('同じ').sum()))
        for label,frame in [('other9',other),('all12',sub)]:
            for col,tag in [('元AIA','before'),('SA1_AIA','after')]:
                summary[label+'_'+tag]=dict(min=float(frame[col].min()),median=float(frame[col].median()),max=float(frame[col].max()),
                    max_attacks=frame.loc[np.isclose(frame[col],frame[col].max(),atol=1e-12,rtol=0),'攻撃名'].tolist())
        summary['with_constant_before']=max(float(sub.元AIA.max()),constant['AIA'])
        summary['with_constant_after']=max(float(sub.SA1_AIA.max()),constant['AIA'])
        summaries.append(summary)
    save(HERE/'集計.json',summaries)
    for f,h in p['hashes'].items():assert sha(f)==h,f
    save(HERE/'予測の来歴.json',[dict(candidate=n,attack=a,kind=k,**r) for (n,a,k),r in records.items()])
    save(HERE/'検証結果.json',dict(passed=True,engine_sha256=sha(__file__),input_hashes_unchanged=True,
        new_cli_jobs=27,reused_predictions=45,total_predictions=72,records=verification,
        same_development_panel=True,independent_panel_validation=False))
    report(p,df,summaries,constant)
    return summaries

def report(p,df,summaries,constant):
    lines=['# SA1の命名と、山登りに使わなかった9攻撃の結果','',
        '- 【ローカル模擬実験】交換を採用した最終Cを、Zhang11-SA1・Zhang04-SA1・Seo03-SA1と命名しました。',
        '- SA1は今回の版名です。作成方法は「改善した交換だけを採用する山登り法」です。',
        '- 標的387人＋対照387人の同じ1パネル（old01）・同じ本戦背景を使用。Zhangさん作成パネルではありません。',
        '- 追加9攻撃はSA1の交換選びに未使用ですが、このパネルは交換選びに使用済みです。別パネルへの一般化の検証ではありません。',
        '- 表は **交換前 → SA1**。AIAは小さいほど攻撃側の成績が低く、0は的中ゼロという意味ではありません。','',
        '| 候補名 | 採用した交換 | time変更行数 | 有用性U（維持） | 候補C |','| --- | ---: | ---: | ---: | --- |']
    for v in p['variants']:lines.append(f'| {v["name"]} | {v["accepted"]}回 | {v["changed_rows"]} | {v["U"]:.6f} | [CSV](候補C/{v["name"]}.csv) |')
    for group,title in [('追加評価','山登りに使わなかった9攻撃'),('山登りに使用','山登りに使った3攻撃（参考）')]:
        lines+=['',f'## {title}','','| 攻撃名 | Zhang11 → Zhang11-SA1 | Zhang04 → Zhang04-SA1 | Seo03 → Seo03-SA1 |','| --- | ---: | ---: | ---: |']
        for a in p['attacks']:
            f=df[df.攻撃ID.eq(a['id'])]
            if f.iloc[0]['評価区分']!=group:continue
            cells=[]
            for v in p['variants']:
                r=f[f.候補.eq(v['name'])].iloc[0];marker=' **悪化**' if r['変化']=='悪化' else ''
                cells.append(f'{r["元AIA"]:.6f} → {r["SA1_AIA"]:.6f}{marker}')
            lines.append('| '+a['name']+' | '+' | '.join(cells)+' |')
    lines+=['','## 追加9攻撃の要約','','| 候補 | 改善／同じ／悪化 | 最大：交換前→SA1 | SA1の中央値 | SA1の最小 |','| --- | --- | ---: | ---: | ---: |']
    for s in summaries:
        b,a=s['other9_before'],s['other9_after']
        lines.append(f'| {s["name"]} | {s["additional_improved"]}／{s["additional_same"]}／{s["additional_worse"]} | {b["max"]:.6f} → {a["max"]:.6f} | {a["median"]:.6f} | {a["min"]:.6f} |')
    lines+=['','## 全攻撃と定数の比較','','| 候補 | 12攻撃の最大：交換前→SA1 | SA1で最大の攻撃 | 定数10年も含む最大：交換前→SA1 |','| --- | ---: | --- | ---: |']
    for s in summaries:
        b,a=s['all12_before'],s['all12_after']
        lines.append(f'| {s["name"]} | {b["max"]:.6f} → {a["max"]:.6f} | {"、".join(a["max_attacks"])} | {s["with_constant_before"]:.6f} → {s["with_constant_after"]:.6f} |')
    lines+=['',f'- 常に10年と答えるAIAは全候補共通で **{constant["AIA"]:.6f}**。Cを参照しないので、今回の交換では変わりません。',
        '- 的中は真のtimeから±0.5年以内。標的・対照の的中率を138：12：9：4の構成へ重み付けし、AIA＝（標的的中率−対照的中率）÷（1−対照的中率）を0〜1に収めます。',
        '- 有用性は公式コードのローカル自己採点です。本番の匿名性得点・φによる補正値・MIAは今回の表に含みません。',
        '- 比較対象は0923で保存した12攻撃。元案36結果と山登り使用攻撃のSA1版9結果は、入力・プログラム・予測のハッシュを確認して再利用しました。追加9攻撃×3案＝27回を新たに実行しました。','',
        '## 保存したもの','',
        '- [全12攻撃の前後比較と標的・対照の的中率](全12攻撃_交換前後.csv)',
        '- [最大・中央値・最小の集計](集計.json)、[命名と元Cの対応](候補の命名.json)',
        '- [実行ログ](実行.log)、[検証記録](検証結果.json)、[分析の前提](説明.md)',
        '- `実行.command`から再実行できます。完了済みの攻撃結果はハッシュを確認して使います。','']
    (HERE/'README.md').write_text('\n'.join(lines))

def main():
    global LOG
    HERE.mkdir(parents=True,exist_ok=True)
    with (HERE/'実行.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:print('すでに実行中です。実行.logをご覧ください。');return
        with (HERE/'実行.log').open('a') as stream:
            LOG=stream;p=prepare();save(HERE/'状態.json',dict(status='running',pid=os.getpid()))
            records={}
            for v in p['variants']:
                say(f'命名 {v["name"]} | 交換{v["accepted"]}回を反映 | U={v["U"]:.6f}')
                for a in p['attacks']:
                    records[(v['name'],a['id'],'original')]=cached(p,v,a,'original')
                    if a['id'] in TRAIN:records[(v['name'],a['id'],'candidate')]=cached(p,v,a,'candidate')
            jobs=[(v,a) for v in p['variants'] for a in p['attacks'] if a['id'] not in TRAIN]
            with ThreadPoolExecutor(max_workers=2) as pool:
                fs={pool.submit(job,p,v,a):(v,a) for v,a in jobs}
                for i,f in enumerate(as_completed(fs),1):
                    v,a=fs[f];r=f.result();records[(v['name'],a['id'],'candidate')]=r
                    say(f'追加攻撃 {i}/27 完了 | {v["name"]} | {a["name"]}')
            summaries=collect(p,records)
            for s in summaries:say(f'{s["name"]} | 追加9攻撃 最大={s["other9_after"]["max"]:.6f} | 全12攻撃 最大={s["all12_after"]["max"]:.6f}')
            save(HERE/'状態.json',dict(status='complete',finished=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            say('完了：README.md に命名・全攻撃の交換前後比較を保存しました。')
            LOG=None

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(HERE/'状態.json',dict(status='error',error=str(e),traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True);raise SystemExit(1)
