"""固定した非訓練方式の背景感度を検証する共通支援。

元main/元出力を実行・上書きしない。個人配列・回答CSVは保存しない。
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = ROOT/'本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索'
PREVIOUS = HERE.parent/'20261008_帯別横断検証'
sys.path.insert(0, str(SOURCE))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj

u = module('_background_robustness_original_common', SOURCE/'共通.py')
audit = module('_background_robustness_official', PREVIOUS/'official_audit.py')
CONDITIONS = [('full',1.,0)] + [(f'fraction{f:g}_seed{s}',f,s)
    for f in (.9,.8,.5) for s in (20261007,20261008)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def log(**values):
    print(json.dumps(values,ensure_ascii=False),flush=True)

def setup():
    """公開入力と攻撃者自身のBのみ。所属正解は読み込まない。"""
    return u.setup('prelim')

def conditions(data):
    """年齢群境界・Q・対象Cはfullのまま、Aの行選択のみ変更。"""
    for label,fraction,seed in CONDITIONS:
        changed = data.copy()
        changed['a'] = data['a'].copy() if fraction==1 else u.cal.stratified_subset(data['a'],fraction,seed)
        yield label,changed

def variant(value, data, team, calibration='raw'):
    return u.variants({'fixed':np.asarray(value)},data,team,calibration!='raw')['fixed__'+calibration]

def cross(by_team, data, coefficient):
    mat = np.column_stack([by_team[t] for t in data['teams']])
    assert mat.shape==(875,20) and np.isfinite(mat).all()
    if coefficient:
        med = np.column_stack([np.median(np.delete(mat,j,axis=1),axis=1) for j in range(len(data['teams']))])
        mat = mat-coefficient*med
    return mat

def expected_full(method):
    """既存公式再評価19対象と、元保存表/集計cacheのTeam12を照合用に読む。"""
    configs=pd.read_csv(PREVIOUS/'方式条件一覧.csv')
    rows=pd.read_csv(PREVIOUS/'全方式評価.csv')
    cfg=configs[(configs.method==method)&(configs.evaluation_class=='C')]
    cfg=cfg[cfg.condition.eq('full')] if cfg.condition.eq('full').any() else cfg
    ids=set(cfg.condition_id)
    part=rows[rows.condition_id.isin(ids)]
    result={int(r.team):float(r.tpr) for r in part.itertuples()}
    assert len(result)==19,(method,len(result))
    sources=pd.read_csv(PREVIOUS/'結果保存元.csv').set_index('source_id')
    aliases=set()
    for r in cfg.itertuples():aliases.update(str(r.source_method_aliases).split(';'))
    aliases.add(method)
    for r in cfg.itertuples():
        for source_id in str(r.source_ids).split(';'):
            path=ROOT/str(sources.loc[source_id,'path']).split('#',1)[0]
            if path.suffix!='.csv':continue
            source=pd.read_csv(path)
            if not {'method','team'}.issubset(source.columns):continue
            source=source[source.method.isin(aliases)&source.team.eq(12)]
            if 'fpr' in source:source=source[np.abs(source.fpr-.01)<1e-15]
            if 'condition' in source:source=source[source.condition.eq('full')]
            col='mia_all' if 'mia_all' in source else 'tpr' if 'tpr' in source else None
            if col and len(source):
                values=source[col].astype(float).to_numpy()
                if np.max(values)-np.min(values)<1e-12:
                    result[12]=float(values[0]);break
        if 12 in result:break
    if 12 not in result:
        for p in PREVIOUS.glob('historical*_cache.json'):
            cache=json.loads(p.read_text())
            records=cache.get('records',[])
            for r in records:
                if r.get('method')==method and r.get('team')==12:
                    result[12]=float(r['tpr']);break
            if 12 in result:break
    assert len(result)==20,(method,'Team12の元保存値が見つからない')
    return result

def score_and_save(data,predictions,csv_name,record_name,specs,started,score_kind='normal_rank',extra=None):
    """全背景・全対象の予測をメモリ確定した後に所属正解を初めて読む。"""
    destination=HERE/csv_name;recordpath=HERE/record_name
    if destination.exists() or recordpath.exists():raise FileExistsError(destination)
    names=sorted({key[1] for key in predictions})
    assert set(key[0] for key in predictions)=={x[0] for x in CONDITIONS}
    assert len(predictions)==len(CONDITIONS)*len(names)
    truth=audit.load_truth(ROOT);official=audit.load_official(ROOT)
    rows=[];full_errors={};boundary_errors=0.
    for (condition,name),mat in sorted(predictions.items()):
        assert mat.shape==((875 if score_kind=='normal_rank' else 1049),20)
        assert np.isfinite(mat).all()
        expected=expected_full(name) if condition=='full' else None
        for j,t in enumerate(data['teams']):
            scores=u.base.confidence(mat[:,j],data['normal'],data['rare'],data['private']) if score_kind=='normal_rank' else mat[:,j]
            members=int(truth[t].sum())
            for fpr in (.005,.01,.02):
                tpr,auc=official._tpr_at_fpr(scores,truth[t],fpr)
                if condition=='full' and fpr==.01:
                    err=abs(float(tpr)-expected[t]);full_errors[name]=max(full_errors.get(name,0.),err)
                    assert err<1e-12,(name,t,tpr,expected[t])
                if t!=12:rows.append(dict(condition=condition,method=name,team=t,fpr=fpr,tpr=float(tpr),members=members,
                    tp_equivalent=float(tpr)*members,above_010=bool(tpr>.10),evaluation_class='C',background_rows=len(next(d['a'] for label,d in conditions(data) if label==condition))))
    with destination.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for path,h in u.base.INPUT_HASHES.items():assert sha(path)==h,path
    record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,
        python=sys.executable,python_version=platform.python_version(),methods=names,conditions=CONDITIONS,
        members_used_only_after_all_predictions=True,training=False,evaluation_teams=data['teams'],report_teams=sorted(set(data['teams'])-{12}),
        fpr_success=.01,stability_fprs=[.005,.02],full20_max_error_by_method=full_errors,
        specs=specs,input_sha256=u.base.INPUT_HASHES,
        source_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/'共通.py',u.OLD/'分析.py',u.CAL/'分析.py',HERE/'background_common.py']},
        output_sha256={csv_name:sha(destination)},score_kind=score_kind,extra=extra or {},
        disclosure='予備戦の既存結果を見た方式固定の後、A抽出だけ変更する感度検証。方式選択の外部検証ではない。個人配列は保存しない。')
    with recordpath.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    log(stage='completed',output=csv_name,methods=len(names),rows=len(rows),seconds=record['seconds'],maximum_full_error=max(full_errors.values()))
    return pd.DataFrame(rows)
