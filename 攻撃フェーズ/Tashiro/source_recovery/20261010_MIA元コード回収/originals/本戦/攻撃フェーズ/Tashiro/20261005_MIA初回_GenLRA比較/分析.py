"""初回MIA作成・固定8構成比較。ルート.venv/bin/python -Bで実行。

既存入力・過去提出は読取専用。結果/があれば上書きを拒否する。
本戦には所属真値がない。予備戦公開正解は予測完了後の評価だけに読む。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_curve

from gen_lra import (VARIANTS, QI7, NUM7, NUM9, CATS, calculate, silverman)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / '結果'
FINAL_INPUT = HERE.parent / '20260928_配布データ基本分析'
KIT = next((ROOT/'本戦').glob('配布*/pwscup2026-kit-main_revised0912'))
sys.path.insert(0, str(KIT/'starter/src'))
CFG = {'scoring': {'low_fpr': .01}}
HASHES = {}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


official = module('pwscup2026.scoring.attack_score_20260928', HERE/'参照コード/attack_score_20260928.py')
validator = module('pwscup2026.kit.validate_20260928', HERE/'参照コード/validate_20260928.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protect(path):
    path = Path(path)
    HASHES[str(path.relative_to(ROOT))] = sha(path)
    return path


def read(path):
    return pd.read_csv(protect(path), float_precision='round_trip')


def jread(path):
    return json.loads(protect(path).read_text())


def input_data(phase):
    if phase == '本戦':
        manifest = jread(FINAL_INPUT/'入力記録.json')
        for f in manifest['files']:
            p = FINAL_INPUT/f['file']
            if sha(p) != f['sha256']:
                raise ValueError(f'固定入力が変更されています: {p}')
        inp = FINAL_INPUT/'入力'
        bpath = next((ROOT/'本戦').glob('配布*/PWSCup2026_final_team28_*/B_self.csv'))
        own = 28
        bg = read(inp/'A_bg.csv')
        assert sha(inp/'A_bg.csv') == sha(protect(bpath.with_name('A_bg.csv')))
    else:
        data = ROOT/'予備戦/配布データ'
        inp = data/'PWSCup2026_prelim_attack'
        bpath = next(data.glob('PWSCup2026_prelim_team05_*/B_self.csv'))
        own = 5
        bg = read(data/'PWSCup2026_prelim_common/A_bg.csv')
    b = read(bpath)
    # 真の希少ラベルは自己コホートで提供されたis_rareだけを使う。
    candidates = b.drop(columns='is_rare')
    if phase == '本戦':
        assert candidates.equals(read(bpath.with_name('B.csv')))
    q = read(inp/f'aia_challenge_{own}.csv')
    assert not b.duplicated(list(QI7)).any() and not q.duplicated(list(QI7)).any()
    matched = b.merge(q[list(QI7)], on=list(QI7), validate='one_to_one')
    private = b.record_id.isin(matched.record_id).to_numpy()
    rare = b.is_rare.to_numpy(bool)
    assert (len(b), len(bg), private.sum(), rare.sum(), (private&rare).sum()) == (1049,5143,163,24,13)
    declared = jread(inp/'mia_columns.json')
    teams = [str(t) for t in declared['target_columns'] if int(t) != own]
    assert len(teams) == (22 if phase == '本戦' else 20)
    assert len(set(teams)) == len(teams)
    assert candidates.record_id.is_unique and candidates.record_id.tolist() == list(range(1049))
    cs = {t: read(inp/f'C_{t}.csv') for t in teams}
    for c in cs.values():
        assert len(c)==1049 and not c.isna().any().any()
    return candidates, bg, cs, teams, private, rare, declared


def ordered_ids(values, ids, mask, top):
    ix = np.flatnonzero(mask)
    # IDは説明の表示順だけ。提出値の同点を破らない。
    ix = sorted(ix, key=lambda i: (-values[i], int(ids[i])))
    return [int(ids[i]) for i in ix[:top]]


def compare_rows(phase, team, answers, ids, rare, private):
    pairs = [('A_mixed7_9','B_ordinal7_9_fixedh'), ('B_ordinal7_9_fixedh','C_ordinal7_9_autoh'),
             ('C_ordinal7_9_autoh','D_ordinal10'), ('D_ordinal10','E_ordinal13_fixedh10'),
             ('E_ordinal13_fixedh10','F_ordinal13_autoh'), ('F_ordinal13_autoh','G_member_global_repaired'),
             ('G_member_global_repaired','H_member_original'), ('A_mixed7_9','G_member_global_repaired')]
    rows=[]
    for first, second in pairs:
        a, b = answers[first], answers[second]
        row={'phase':phase,'team':team,'before':first,'after':second}
        for group, mask, top in [('nonrare',~private&~rare,20),('rare',~private&rare,2)]:
            ao, bo = ordered_ids(a,ids,mask,top), ordered_ids(b,ids,mask,top)
            row[group+'_top_overlap']=len(set(ao)&set(bo))
            row[group+'_same_first']=ao[0]==bo[0]
            row[group+'_spearman']=float(spearmanr(a[mask],b[mask]).statistic)
        rows.append(row)
    return rows


def evaluate_prelim(predictions, ids, rare, private):
    """予測を全対象で確定した後にのみpool_idを読む。推定関数へ渡さない。"""
    truthdir=next((ROOT/'予備戦/配布データ/結果').glob('正解*'))
    own=read(truthdir/'pool_ids_5.csv').set_index('record_id').loc[ids]
    assert np.array_equal(own.is_rare.to_numpy(bool),rare)
    rows=[]
    max_roc_error=0.
    for team, answers in predictions.items():
        pool=read(truthdir/f'pool_ids_{team}.csv')
        truth=own.pool_id.isin(pool.pool_id).to_numpy(int)
        assert truth[private].sum()==0
        for variant, conf in answers.items():
            result=official.score_mia(conf,truth,CFG,rare_mask=rare)
            detail=result.breakdown
            for mask, name in [(np.ones(len(ids),dtype=bool),'tpr_at_low_fpr'),(rare,'tpr_rare')]:
                if truth[mask].sum()>0:
                    fp,tp,_=roc_curve(truth[mask],conf[mask],drop_intermediate=False)
                    err=abs(float(np.interp(.01,fp,tp))-detail[name])
                    max_roc_error=max(max_roc_error,err)
                    assert err<1e-12
            rows.append({'team':team,'variant':variant,'members':int(truth.sum()),
                         'rare_members':int(truth[rare].sum()),'mia_all':detail['tpr_at_low_fpr'],
                         'mia_rare':detail['tpr_rare'],'mia_score':result.score,'auc':detail['auc']})
    return pd.DataFrame(rows),max_roc_error


def save_csv(path, frame):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='') as out:
        frame.to_csv(out,index=False,float_format='%.17g')


def save_json(path, obj):
    with path.open('x',encoding='utf-8') as out:
        json.dump(obj,out,ensure_ascii=False,indent=2,allow_nan=False)
        out.write('\n')


def main():
    if OUT.exists():
        raise FileExistsError('結果/ は既に存在します。履歴保全のため再作成しません。検証.pyで確認してください。')
    started=perf_counter()
    started_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
    old21=jread(ROOT/'予備戦/攻撃フェーズ/分析/21_MIA全体と希少のGenLRA比較/比較結果.json')
    old22=jread(ROOT/'予備戦/攻撃フェーズ/分析/22_GenLRAの7属性9属性10属性比較/比較結果.json')
    for p in HERE.rglob('*.py'):
        protect(p)
    for p in [ROOT/'requirements.txt',ROOT/'Python環境.md',
              ROOT/'予備戦/攻撃フェーズ/分析/20_Team20希少のGenLRA検証/分析.py',
              ROOT/'予備戦/攻撃フェーズ/分析/21_MIA全体と希少のGenLRA比較/分析.py',
              ROOT/'予備戦/攻撃フェーズ/分析/22_GenLRAの7属性9属性10属性比較/分析.py']:
        protect(p)
    for p in (KIT/'starter/src/pwscup2026').rglob('*.py'):
        protect(p)
    comparisons=[];rare_rows=[];normal_rows=[];raw_rows=[];validations=[];old_checks=[]
    all_predictions={};all_data={};member_checks=[]
    expected_first2={'0':[.8698760724499524,.1854146806482364],
                     '1':[.7621544327931363,.1663489037178265]}
    for phase in ['本戦','予備戦']:
        candidates,bg,cs,teams,private,rare,declared=input_data(phase)
        ids=candidates.record_id.to_numpy()
        all_data[phase]=(ids,rare,private,declared)
        predictions={}
        for number,team in enumerate(teams,1):
            raw,answers=calculate(bg,candidates,cs[team],rare,private)
            predictions[team]=answers
            if phase=='予備戦':
                assert ids.tolist()==old21['record_ids']==old22['record_ids']
                for key,stored in [('mixed7',old21['raw_scores'][team]['qi7_gen']),
                                   ('mixed9',old22['raw_new_scores'][team]['attrs9'])]:
                    previous=np.array(stored)
                    error=float(np.max(np.abs(raw[key]-previous)))
                    # 数値一致と順位一致を分けて記録。小さい誤差を性能差と解釈しない。
                    same=bool(np.array_equal(rankdata(raw[key],method='min'),rankdata(previous,method='min')))
                    old_checks.append({'team':team,'feature':key,'max_abs_error':error,'all_ranks_equal':same,
                                       'top20_equal':ordered_ids(raw[key],ids,~private,20)==ordered_ids(previous,ids,~private,20)})
                    assert error<1e-10
            if phase=='本戦':
                if team in expected_first2:
                    actual=answers['H_member_original'][:2]
                    err=float(np.max(np.abs(actual-np.array(expected_first2[team]))))
                    member_checks.append({'team':team,'record_ids':[0,1],'max_percentile_error':err,'matches':err<1e-12})
                for variant,conf in answers.items():
                    frame=pd.DataFrame({'record_id':ids,team:conf})
                    for i in np.flatnonzero(rare):
                        mask=rare&~private
                        rank=None if private[i] else int(1+sum(conf[mask]>conf[i]))
                        rare_rows.append({'team':team,'record_id':int(ids[i]),'variant':variant,
                                          'private_nonmember':bool(private[i]),'rank_within_11':rank,
                                          'confidence':float(conf[i])})
                    top=ordered_ids(conf,ids,~rare&~private,20)
                    for position,rid in enumerate(top,1):
                        normal_rows.append({'team':team,'variant':variant,'display_position':position,
                                            'record_id':rid,'confidence':float(conf[np.flatnonzero(ids==rid)[0]])})
                for i,rid in enumerate(ids):
                    raw_rows.append({'team':team,'record_id':int(rid),**{name:float(v[i]) for name,v in raw.items()}})
            comparisons.extend(compare_rows(phase,team,answers,ids,rare,private))
            print(f'{phase} {number}/{len(teams)}対象完了、経過{perf_counter()-started:.1f}秒',flush=True)
        all_predictions[phase]=predictions
        for variant in VARIANTS:
            frame=pd.DataFrame({'record_id':ids,**{t:predictions[t][variant] for t in teams}})
            ref={'mia_columns':declared,'team_id':28 if phase=='本戦' else 5,'attacker_record_ids':set(map(int,ids))}
            check=validator.validate_mia_submission(frame,{},ref)
            assert check.ok,check.errors
            assert frame.columns.tolist()==['record_id',*teams]
            assert np.isfinite(frame.to_numpy(float)).all()
            if variant!='H_member_original':
                assert (frame.iloc[private,1:].to_numpy()==0).all()
            validations.append({'phase':phase,'variant':variant,'official_mia_validation':check.ok,
                                'rows':len(frame),'target_columns':len(teams)})
    ids,rare,private,_=all_data['予備戦']
    evaluation,roc_error=evaluate_prelim(all_predictions['予備戦'],ids,rare,private)
    aggregate=evaluation.groupby('variant',sort=False).agg(targets=('team','size'),
        mia_all=('mia_all','mean'),mia_rare_positive_targets=('mia_rare','mean'),
        mia_score=('mia_score','mean'),auc=('auc','mean')).reset_index()
    baseline=evaluation[evaluation.variant=='A_mixed7_9'].set_index('team')
    for i,row in aggregate.iterrows():
        other=evaluation[evaluation.variant==row.variant].set_index('team')
        diff=other.mia_score-baseline.mia_score
        aggregate.loc[i,'delta_baseline']=float(diff.mean())
        aggregate.loc[i,'improved_targets']=int((diff>1e-12).sum())
        aggregate.loc[i,'worse_targets']=int((diff < -1e-12).sum())
    assert all(sha(ROOT/p)==value for p,value in HASHES.items()),'入力変更を検出'
    OUT.mkdir()
    ids,rare,private,declared=all_data['本戦']
    for variant in VARIANTS:
        answers=all_predictions['本戦']
        frame=pd.DataFrame({'record_id':ids,**{t:ans[variant] for t,ans in answers.items()}})
        dest=OUT/'F_mia.csv' if variant=='A_mixed7_9' else OUT/'比較用_提出候補ではない'/f'{variant}.csv'
        save_csv(dest,frame)
        saved=pd.read_csv(dest,float_precision='round_trip')
        assert np.array_equal(frame.to_numpy(),saved.to_numpy()),dest
    save_csv(OUT/'本戦_生スコア.csv',pd.DataFrame(raw_rows))
    save_csv(OUT/'本戦_希少24人の順位.csv',pd.DataFrame(rare_rows))
    save_csv(OUT/'本戦_非希少上位20人.csv',pd.DataFrame(normal_rows))
    save_csv(OUT/'方式間の順位比較.csv',pd.DataFrame(comparisons))
    save_csv(OUT/'予備戦_対象別評価.csv',evaluation)
    save_csv(OUT/'予備戦_集約評価.csv',aggregate)
    save_csv(OUT/'予備戦_旧生スコア照合.csv',pd.DataFrame(old_checks))
    save_csv(OUT/'本戦_非会員163人.csv',pd.DataFrame({'record_id':ids[private],'is_rare':rare[private].astype(int)}))
    config={'chosen':'A_mixed7_9','status':'作成済み・未提出','self_team':28,'k':10,'rho':.98,
            'nonrare_features':list(QI7),'rare_features':list(NUM9+CATS),'rare_cap':.5,
            'private_nonmembers':163,'private_rare':13,'unknown_rare':11,
            'old48_negatives_reused':False,'old_team_exceptions_reused':False,
            'variant_descriptions':VARIANTS,
            'bandwidths':{str(d):silverman(5143,d) for d in [5,7,9,10,13]},
            'member_source':'https://github.com/smsaku00/pwscup2026-sutehaji/blob/a465f1051d8f002f3364d38f311d527608a1957e/攻撃フェーズ/Yamazaki/gen_lra/run_gen_lra.py',
            'member_port':'前ターンで読んだ固定コミットの計算部を移植。実行器・入力読取・出力処理は本分析側。',
            'input_csv_float_precision':'round_trip',
            'performance_limit':'本戦の所属真値は未公開。予備戦評価は事後検証で本戦の改善保証ではない。'}
    save_json(OUT/'計算条件.json',config)
    save_json(OUT/'入力SHA256.json',HASHES)
    save_json(OUT/'作成時検証.json',{'official':validations,'sklearn_roc_max_error':roc_error,
                                  'old_max_abs_error':max(r['max_abs_error'] for r in old_checks),
                                  'old_all_ranks_equal_cases':sum(r['all_ranks_equal'] for r in old_checks),
                                  'old_comparisons':len(old_checks),'member_saved_csv_sample_check':member_checks,
                                  'inputs_unchanged':True,'saved_csv_roundtrip_exact':True})
    save_json(OUT/'実行記録.json',{'started_jst':started_jst,
                                 'finished_jst':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                                 'elapsed_seconds':perf_counter()-started,'python':sys.executable,
                                 'version':platform.python_version(),'numpy':np.__version__,
                                 'pandas':pd.__version__,'scipy':scipy.__version__,
                                 'F_mia_sha256':sha(OUT/'F_mia.csv')})
    save_json(OUT/'出力SHA256.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
    print(aggregate.to_string(index=False),flush=True)
    print(f'作成完了: {OUT}/F_mia.csv。CodaBenchには未提出。',flush=True)


if __name__=='__main__':
    main()
