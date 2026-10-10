"""保存した第7回F_miaを検査する。ファイルは作成・更新しない。

python3 -B 検証.py （同梱Python、NumPy・pandas、予想10〜60秒）
採点の仮ラベルは実装検査専用で、攻撃性能の評価には用いない。
MIA形式検査は改訂版の純粋関数2本をASTから無改変で実行する。
これは同梱環境に不要なYAML依存を要求しないため。ZIP検査・外部提出は行わない。
"""
from __future__ import annotations

import ast
from collections import defaultdict
import csv
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
KIT = ROOT/'スターターキット/改訂版_20260824/starter/src'
DATA = ROOT/'予備戦/配布データ'
ANALYSIS = ROOT/'予備戦/攻撃フェーズ/分析'
sys.path.insert(0,str(KIT))
from pwscup2026.scoring.attack_score import _tpr_at_fpr
from pwscup2026.common.submission_schema import ValidationResult


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 1050 and all(len(row) == 21 for row in rows)
    ids = np.array([int(row[0]) for row in rows[1:]])
    assert len(set(ids)) == 1049 and all(str(x)==row[0] for x,row in zip(ids,rows[1:]))
    return rows[0],ids,np.array([[float(x) for x in row[1:]] for row in rows[1:]]),rows


def official_format(frame,ids):
    source = KIT/'pwscup2026/kit/validate.py'
    tree = ast.parse(source.read_text())
    names = {'_check_unit_interval','validate_mia_submission'}
    definitions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in definitions} == names
    namespace = dict(np=np,pd=pd,re=re,ValidationResult=ValidationResult)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(source),'exec'),namespace)
    result = namespace['validate_mia_submission'](frame,{},dict(attacker_record_ids=set(ids),
        mia_columns=read(DATA/'PWSCup2026_prelim_attack/mia_columns.json')))
    assert result.ok,result.errors
    return dict(ok=result.ok,errors=result.errors,method='official_pure_functions_unmodified_AST',source_sha256=sha(source))


def reference_tpr(scores,truth):
    positives = int(sum(truth))
    if not positives or positives==len(truth):
        return 0.
    blocks = defaultdict(lambda:[0,0])
    for value,y in zip(scores,truth):
        blocks[float(value)][int(y)] += 1
    target = .01*(len(truth)-positives)
    fp=tp=0
    for value,(n,p) in sorted(blocks.items(),reverse=True):
        if fp+n > target:
            return (tp+p*(target-fp)/n)/positives
        fp+=n
        tp+=p
    return 1.


def verify():
    start=perf_counter()
    before={p:sha(p) for p in HERE.iterdir() if p.is_file()}
    meta=read(HERE/'計算条件.json')
    hashes=read(HERE/'入力SHA256.json')
    assert all(not p.startswith('アーカイブ/') and sha(ROOT/p)==digest for p,digest in hashes.items())
    record_root=ROOT/'予備戦/攻撃フェーズ/提出記録'
    assert {str(p.relative_to(ROOT)) for p in record_root.rglob('*') if p.is_file()}==set(meta['protected_record_paths'])
    header,ids,scores,csv_rows=load_csv(HERE/'F_mia.csv')
    old_header,old_ids,old_scores,old_rows=load_csv(ROOT/meta['baseline'])
    assert header==old_header and np.array_equal(ids,old_ids)
    teams=[int(k) for k in header[1:]]
    assert header[0]=='record_id' and len(set(header))==21 and 5 not in teams
    assert teams==[t['team'] for t in meta['teams']]
    stored=read(HERE/'提出行列.json')
    assert stored['header']==header
    assert np.array_equal(np.array(stored['rows'])[:,1:],scores)
    source=read(ANALYSIS/'21_MIA全体と希少のGenLRA比較/比較結果.json')
    source9=read(ANALYSIS/'22_GenLRAの7属性9属性10属性比較/比較結果.json')
    proposal=source9['plans']['gen9_after48']
    assert meta['rare_plan']=='gen9_after48'
    b=pd.read_csv(DATA/'PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/B_self.csv',float_precision='round_trip')
    assert b.record_id.tolist()==ids.tolist()
    bg=pd.read_csv(DATA/'PWSCup2026_prelim_common/A_bg.csv',float_precision='round_trip')
    ch=pd.read_csv(DATA/'PWSCup2026_prelim_attack/aia_challenge_5.csv',float_precision='round_trip')
    nums=['age','SBP','TG','HDL','ALT'];cats=['sex','prefecture']
    nums9=['age','BMI','SBP','TG','HDL','ALT','FPG']
    assert meta['numeric']==nums and meta['categorical']==cats
    assert meta['rare_numeric']==nums9 and meta['rare_categorical']==cats
    private_ids=set(b.merge(ch[nums+cats],on=nums+cats,validate='one_to_one').record_id.astype(int))
    assert private_ids==set(meta['private_ids']) and len(private_ids)==163
    private=np.isin(ids,list(private_ids));rare=b.is_rare.to_numpy(bool);use=~private&~rare
    eligible=meta['eligible_ids'];pos={int(rid):i for i,rid in enumerate(ids)}
    assert (int(rare.sum()),int(use.sum()),len(eligible))==(24,875,11)
    known={(r['team'],r['record_id']) for r in meta['known_negative_pairs']}
    assert len(known)==48
    # 独立した過去集計との照合。xはチーム単位で、全対象共通の除外IDではない。
    old_status={(r['team'],r['record_id']) for r in source9['rare_rows'] if r['status']=='x'}
    assert known==old_status
    assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
    assert (scores[private]==0).all()
    assert int((scores[rare]==0).sum())==308 and int((scores[rare]>0).sum())==172
    assert int((scores!=old_scores).sum())==meta['total_changed_cells']
    spec=importlib.util.spec_from_file_location('independent_kernel',ANALYSIS/'20_Team20希少のGenLRA検証/検証.py')
    independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
    assert len(bg)==5143
    freq={col:{str(v):(int(n)+1)/(len(bg)+bg[col].nunique()) for v,n in bg[col].value_counts().items()} for col in cats}
    raw_errors={'gen7':0.,'gen9':0.};raw_checks={'gen7':0,'gen9':0}
    rare_checks=0;full_checks=0;literal_kept=0
    parsed=pd.read_csv(HERE/'F_mia.csv')
    parsed_old=pd.read_csv(ROOT/meta['baseline'])
    validation=official_format(parsed,ids)
    parsers=[('float_round_trip',scores,old_scores),('official_pandas_default',parsed.iloc[:,1:].to_numpy(float),parsed_old.iloc[:,1:].to_numpy(float))]
    rare_saved={(r['team'],r['record_id']):r for r in read(HERE/'希少24人のスコア.json')}
    assert len(rare_saved)==480
    for j,t in enumerate(meta['teams']):
        k=t['team'];raw=np.array(source['raw_scores'][str(k)]['qi7_gen'])
        raw9=np.array(source9['raw_new_scores'][str(k)]['attrs9'])
        ordered=sorted(np.flatnonzero(use),key=lambda i:(-raw[i],ids[i]))
        if k not in (1,2):
            rank=pd.Series(raw[use]).rank(method='min').to_numpy()
            expected=t['cap']+(1-t['cap'])*(.1+.8*rank/875)
            assert np.array_equal(scores[use,j],expected)
            assert t['nonrare_method']=='GenLRA_QI7'
        else:
            assert np.array_equal(scores[~rare,j],old_scores[~rare,j])
            for i in np.flatnonzero(~rare):
                assert csv_rows[i+1][j+1]==old_rows[i+1][j+1]
                literal_kept+=1
        remain=[rid for rid in eligible if (k,rid) not in known]
        pair=proposal[str(k)]
        expected_order=sorted(remain,key=lambda rid:(-raw9[pos[rid]],rid))
        assert t['rare_method']=='GenLRA_attrs9'
        assert t['rare_order']==expected_order and t['rare_top2']==pair==expected_order[:2]
        for rid in ids[rare]:
            rid=int(rid);row=rare_saved[k,rid]
            assert scores[pos[rid],j]==row['confidence']
            assert row['genlra7']==raw[pos[rid]] and row['genlra9']==raw9[pos[rid]]
            assert row['status']==('x' if rid in private_ids or (k,rid) in known else '?')
            assert row['submission_rank']==(expected_order.index(rid)+1 if rid in remain else None)
            for key,values in [('genlra7_rank11',raw),('genlra9_rank11',raw9)]:
                assert row[key]==(1+int(sum(values[pos[x]]>values[pos[rid]] for x in eligible)) if rid in eligible else None)
        for rank0,rid in enumerate(expected_order):
            assert scores[pos[rid],j]==t['cap']*(.9-.8*rank0/len(remain))
        for parser,new,old in parsers:
            assert np.isfinite(new).all() and (new[private]==0).all()
            assert all(new[pos[rid],j]==0 for kk,rid in known if kk==k)
            actual=sorted(remain,key=lambda rid:-new[pos[rid],j])
            assert actual==expected_order
            assert len(set(new[[pos[rid] for rid in remain],j]))==len(remain)
            if k not in (1,2):
                raw_sorted=raw[ordered];submitted=new[ordered,j]
                assert np.all((np.diff(raw_sorted)==0)==(np.diff(submitted)==0))
                assert np.all(np.diff(submitted)<=0)
            assert int(sum(new[~rare,j]>new[rare,j].max()))>=38
            control=old[:,j].copy();control[~rare]=new[~rare,j]
            assert int(sum(control[~rare]>max(control[rare].max(),new[rare,j].max())))>=38
            if k in (1,2):
                assert np.array_equal(new[~rare,j],old[~rare,j])
            for n in (1,2):
                for members in combinations(remain,n):
                    truth=np.isin(ids[rare],members).astype(int)
                    expected=(1. if n==1 or pair[1] in members else .5) if pair[0] in members else 0.
                    actual=_tpr_at_fpr(new[rare,j],truth,.01)[0]
                    assert abs(actual-expected)<1e-14
                    assert abs(actual-reference_tpr(new[rare,j],truth))<1e-14
                    rare_checks+=1
            for p in (25,26,27):
                for r in (0,1,2):
                    for mode in range(4):
                        rng=np.random.default_rng(100000*k+100*p+10*r+mode)
                        if mode==0:
                            nonrare_members=ordered[:p-r]
                        elif mode==1:
                            nonrare_members=ordered[-(p-r):]
                        else:
                            nonrare_members=rng.choice(np.flatnonzero(use),p-r,replace=False)
                        members=[*nonrare_members,*rng.choice([pos[rid] for rid in remain],r,replace=False)]
                        truth=np.zeros(1049,int);truth[members]=1
                        actual=_tpr_at_fpr(new[:,j],truth,.01)[0]
                        assert actual==_tpr_at_fpr(control,truth,.01)[0]
                        assert abs(actual-reference_tpr(new[:,j],truth))<1e-14
                        if k in (1,2):
                            assert actual==_tpr_at_fpr(old[:,j],truth,.01)[0]
                        full_checks+=1
        # 生スコアの独立再計算。希少24人＋非希少上位20人＋専用非希少5人。
        select=sorted(set(np.flatnonzero(rare))|set(ordered[:20])|set(np.flatnonzero(private&~rare)[:5]))
        c=pd.read_csv(DATA/f'PWSCup2026_prelim_attack/C_{k}.csv',float_precision='round_trip')
        # 9属性は希少24人全員を、近傍選択・密度・加算式から独立に照合。
        for name,features,check_ids,expected_raw in [
            ('gen7',nums,select,raw),('gen9',nums9,np.flatnonzero(rare),raw9)]:
            gx=independent.transform(bg,features)
            scales=np.subtract(*np.percentile(gx,[75,25],axis=0))/1.349
            bw=(4/(len(bg)*(len(features)+2)))**(1/(len(features)+4))
            cx=independent.transform(c,features)
            for i in check_ids:
                q=b.iloc[[i]]
                d2=(((cx-independent.transform(q,features))/scales)**2).sum(axis=1)
                for cat in cats:
                    d2+=c[cat].to_numpy()!=q[cat].iloc[0]
                near=np.argsort(d2,kind='stable')[:10]
                lr=independent.logk(c.iloc[near],bg,features,cats,scales,bw,freq)
                lx=independent.logk(c.iloc[near],q,features,cats,scales,bw,freq)
                actual=float((independent.logmean(np.concatenate([lr,lx],axis=1))-independent.logmean(lr)).sum())
                raw_errors[name]=max(raw_errors[name],abs(actual-expected_raw[i]))
                assert raw_errors[name]<1e-10
                raw_checks[name]+=1
    assert literal_kept==2050
    assert raw_checks==dict(gen7=980,gen9=480)
    assert all((38-p)/(1049-p)>.01 for p in (25,26,27))
    tie_checks=0
    for n in range(3,12):
        for r in (1,2):
            s=np.r_[np.ones(n),np.zeros(24-n)];y=np.r_[np.ones(r,int),np.zeros(24-r,int)]
            actual=_tpr_at_fpr(s,y,.01)[0]
            assert abs(actual-.01*(24-r)/(n-r))<1e-14
            assert abs(actual-reference_tpr(s,y))<1e-14
            tie_checks+=1
    assert all(sha(ROOT/p)==digest for p,digest in hashes.items())
    assert {str(p.relative_to(ROOT)) for p in record_root.rglob('*') if p.is_file()}==set(meta['protected_record_paths'])
    assert before=={p:sha(p) for p in HERE.iterdir() if p.is_file()}
    return dict(rows=1049,target_columns=20,official_mia_format=validation,
        two_parsers_verified=True,team1_2_nonrare_literal_unchanged=literal_kept,
        independent_genlra_values=raw_checks,max_independent_genlra_error=raw_errors,
        rare_label_cases=rare_checks,full_separation_cases=full_checks,tie_formula_cases=tie_checks,
        known_negative_cells=48,private_zero_cells=int(private.sum())*20,
        rare_zero_cells=int((scores[rare]==0).sum()),rare_positive_cells=int((scores[rare]>0).sum()),
        zero_cells=int((scores==0).sum()),positive_cells=int((scores>0).sum()),
        changed_nonrare_cells=meta['changed_nonrare_cells'],changed_rare_cells=meta['changed_rare_cells'],
        protected_record_files=len(meta['protected_record_paths']),protected_files_unchanged=True,
        csv_sha256=sha(HERE/'F_mia.csv'),seconds=perf_counter()-start,
        actual_score_improvement_unverified=True,zip_validation_not_performed=True)


if __name__=='__main__':
    print(json.dumps(verify(),ensure_ascii=False,indent=2))
