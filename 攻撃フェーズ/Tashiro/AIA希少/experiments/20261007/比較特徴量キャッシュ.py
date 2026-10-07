"""学習比較の公開入力キャッシュ。既存の凍結実験を上書きしない。

既存キャッシュがある場合は当時のコードと入力のハッシュを確認する。
無い場合はこの分析フォルダの公開推定関数で作業用キャッシュを生成する。
本戦Bの真値、token、モデル、提出形式CSV/ZIPは保存しない。
"""
from __future__ import annotations
import 分析 as core
import hashlib, json
from pathlib import Path
import numpy as np
import 時間候補の特徴量 as features


def build(root):
    cache=core.HERE/'作業用/比較特徴量.npz';cache.parent.mkdir(exist_ok=True)
    result={};names=None
    for phase in ['prelim','main']:
        cases,bg=core.datasets(root,phase);rows=[]
        for v in cases:
            p,x=features.predict_methods(v['q'],v['c'],bg)
            if names is None:names=list(p)
            assert names==list(p)
            rows.append(dict(team=v['team'],x=x,p=np.column_stack([p[name] for name in names]),truth=v['truth']))
            print(dict(phase=phase,team=v['team'],cache_only=True),flush=True)
        result[phase+'_x']=np.vstack([v['x'] for v in rows]);result[phase+'_team']=np.concatenate([np.full(326,v['team']) for v in rows])
        result[phase+'_pred']=np.vstack([v['p'] for v in rows])
        if phase=='prelim':
            result['prelim_member']=np.concatenate([v['truth'].label.eq('member').to_numpy() for v in rows])
            result['prelim_time']=np.concatenate([v['truth'].time.to_numpy() for v in rows]);result['prelim_rare']=np.concatenate([v['truth'].is_rare.to_numpy(bool) for v in rows])
    result['names']=np.array(names);np.savez_compressed(cache,**result)
    core.record('比較特徴量再現',dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        feature_code_sha256=hashlib.sha256(Path(features.__file__).read_bytes()).hexdigest(),cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        private_B_used=False,main_truth_in_cache=False))
    return cache


def load(root):
    legacy=root/'本戦/攻撃フェーズ/Tashiro/20261007_AIA全体向上の分析'
    oldcache=legacy/'作業用/比較特徴量.npz';record=legacy/'実行記録.json'
    if oldcache.exists() and record.exists():
        r=json.loads(record.read_text())
        assert hashlib.sha256((legacy/'分析.py').read_bytes()).hexdigest()==r['code_sha256']
        for name,digest in r['input_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
        return np.load(oldcache),oldcache
    cache=core.HERE/'作業用/比較特徴量.npz';record=core.HERE/'比較特徴量再現_実行記録.json'
    if not cache.exists():cache=build(root)
    r=json.loads(record.read_text())
    assert hashlib.sha256(Path(features.__file__).read_bytes()).hexdigest()==r['feature_code_sha256']
    assert hashlib.sha256(cache.read_bytes()).hexdigest()==r['cache_sha256']
    for name,digest in r['input_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    return np.load(cache),cache


if __name__=='__main__':
    cache=build(core.ROOT)
    legacy=core.ROOT/'本戦/攻撃フェーズ/Tashiro/20261007_AIA全体向上の分析/作業用/比較特徴量.npz'
    if legacy.exists():
        a,b=np.load(cache),np.load(legacy);assert set(a.files)==set(b.files)
        for key in a.files:assert np.array_equal(a[key],b[key]),key
        core.save_json(core.HERE/'比較特徴量再現_一致確認.json',dict(arrays_equal=True,array_names=a.files,legacy_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
        print('all cached arrays exactly equal',flush=True)
