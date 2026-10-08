"""旧仮想大会の公開入力だけの固定攻撃profileを大会真値で再評価する。

仮想大会や加工物を生成せず、既存の純推定関数と固定重みだけを利用。
個人別score/ラベルを保存しない。oracleの加工再生成LOOは対象外。
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from lifelines import CoxPHFitter
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def _extract(path,names,environment,assignments=()):
    tree=ast.parse(path.read_text())
    selected=[]
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:
            selected.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignments for t in n.targets):
            selected.append(n)
        elif isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in assignments:
            selected.append(n)
    assert {n.name for n in selected if isinstance(n,(ast.FunctionDef,ast.ClassDef))}==set(names)
    mod=types.ModuleType('_history_virtual_'+path.stem)
    mod.__dict__.update(environment)
    sys.modules[mod.__name__]=mod
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec',flags=__import__('__future__').annotations.compiler_flag),mod.__dict__)
    return mod


def gather(root,use_cache=True):
    root=Path(root).resolve()
    cache=HERE/'historical_virtual_cache.json'
    if use_cache and cache.exists():
        saved=json.loads(cache.read_text())
        if all((root/p).exists() and _sha(root/p)==digest for p,digest in saved['source_hashes'].items()):
            return saved['records'],saved['notes']
    audit=_module('_history_virtual_official',HERE/'official_audit.py')
    official=audit.load_official(root)
    truths=audit.load_truth(root)
    b,_,_,_,dist=audit.load_inputs(root)
    bgpath=root/'予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv'
    bg=pd.read_csv(bgpath,float_precision='round_trip')
    teams=sorted(truths)
    cs={k:pd.read_csv(dist/f'C_{k}.csv',float_precision='round_trip') for k in teams}
    old=root/'アーカイブ/予備戦_攻撃フェーズ/work_20260901以前'
    ap=old/'attacks.py'; dp=old/'defense.py'; cp=old/'mia_redteam.py'; op=old/'outcome_mia_redteam.py'; wp=old/'whitebox_audit.py'; vp=old/'virtual_mia_engine.py'
    attacks=_module('_history_virtual_attacks',ap)
    env=dict(np=np,pd=pd,AT=attacks,math=math,rankdata=rankdata,dataclass=dataclass,CoxPHFitter=CoxPHFitter,
             warnings=warnings,NearestNeighbors=NearestNeighbors,ExtraTreesClassifier=ExtraTreesClassifier,
             HistGradientBoostingClassifier=HistGradientBoostingClassifier)
    defense=_extract(dp,['_design','fit_cox_raw','quantile_align'],dict(env,COX_COLS=['age','sex','BMI','SBP','log_TG','HDL','log_ALT','FPG','smoking'],AGE_RANGE=(40,74)))
    defense.HORIZON=10.0
    env['D']=defense
    clinical=_extract(cp,['_shadow_features','shadow_c2st','_scaled','multiview_tail_linkage'],
                      dict(env,CLINICAL_VIEWS=(tuple(attacks.CONT),('age','BMI','FPG'),('age','TG','HDL','ALT'),('BMI','SBP','FPG'),('TG','HDL','ALT','FPG'))))
    outcome=_extract(op,['_event_code','event_time_exact','event_time_near','_joint_features','joint_nn_gap','joint_knn','joint_shadow_c2st','_CauseModel','_exposure','_fit_cause','survival_scores'],env)
    whitebox=_extract(wp,['_map_A_to_C','_link_source_rows','_scaled','_knn_mean'],dict(env,CONT=attacks.CONT,CLIN=attacks.CLIN))
    virtual=_extract(vp,['AttackProfile','_rank01','_whitebox_recovered_components','compose_profile_score'],
                     dict(env,WB=whitebox),assignments=['DEFAULT_PROFILES'])
    def evaluate_target(k):
        c=cs[k]
        pieces={
            'cov_exact':attacks.m2_exact_match(c,bg,b),
            'cov_nn_gap':attacks.m3_nn_gap(c,bg,b),
            'cov_knn':attacks.m4_knn_density(c,bg,b),
            'domias':attacks.m1_domias(c,bg,b),
            'pref_prior':attacks.m5_pref_prior(c,bg,b),
            'multiview_tail':clinical.multiview_tail_linkage(c,bg,b),
            'event_time_exact':outcome.event_time_exact(c,bg,b),
            'event_time_near':outcome.event_time_near(c,bg,b),
            'joint_nn_gap':outcome.joint_nn_gap(c,bg,b),
            'joint_knn':outcome.joint_knn(c,bg,b),
            'shadow_c2st':clinical.shadow_c2st(c,bg,b,seed=20260812+101),
            'joint_shadow_c2st':outcome.joint_shadow_c2st(c,bg,b,seed=20260812+202),
        }
        pieces['survival_likelihood'],pieces['cox_residual']=outcome.survival_scores(c,bg,b)
        pieces.update(virtual._whitebox_recovered_components(c,bg,b,seed=20260812+303))
        # 同一重み/同一gateのclassic_generalistは旧mia_ensembleの明示的別名。
        outputs={'legacy_virtual_'+name:virtual.compose_profile_score(pieces,name)
                 for name in virtual.DEFAULT_PROFILES if name!='classic_generalist'}
        for name in ['wb_recovered_gap','wb_selected_negative','shadow_c2st','joint_shadow_c2st']:
            outputs['legacy_virtual_component_'+name]=pieces[name]
        return k,outputs
    results={}; started=time.monotonic()
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(evaluate_target,k) for k in teams]
        for future in as_completed(futures):
            k,scores=future.result(); results[k]=scores
            print(f'旧公開source回復/profileの公式再採点: {len(results)}/20対象 ({time.monotonic()-started:.0f}秒)',flush=True)
    records=[]
    for method in results[teams[0]]:
        profile=method.removeprefix('legacy_virtual_')
        parameters=dict(virtual.DEFAULT_PROFILES[profile].weights) if profile in virtual.DEFAULT_PROFILES else '元score_public_componentsの種20260812+101/+202/+303'
        for k in teams:
            scores=results[k][method]
            assert np.isfinite(scores).all()
            tpr,auc=official._tpr_at_fpr(scores,truths[k],.01)
            records.append(dict(team=k,method=method,condition='旧公開入力profileの既定重み・予備戦20対象',
                tpr=float(tpr),members=int(truths[k].sum()),attributes='profile成分の臨床/転帰属性。source回復は臨床6+sex+smoking、照会は連続7+sex+smoking',
                params=parameters,background='予備戦A_bg 5143行',evaluation_class='C',
                split='所属真値不使用。分類は公開C対A/公開source選択だけ。',transductive='対象CとAを参照',
                source=str(vp.relative_to(root))+'#'+profile,verification='既存純関数/固定profileを抽出して再計算・現公式採点',
                family='旧仮想大会の公開入力MIA',notes='仮想大会のproxy得点は使わない。加工再生成oracle LOOと区別。公式同点を保持し旧jitterは使わない。',
                selection='事前固定方式',auc=float(auc)))
    notes=[dict(scope='virtual classic_generalist',status='同一方式の明示的別名',
                reason='旧attacks.mia_ensembleと成分5種類・重み・exact gateが同じ。別方式へ加えない。',source=str(vp.relative_to(root))),
           dict(scope='whitebox targeted LOO',status='oracle入力不足・実行しない',
                reason='対象チームBの全個票と同クラスreplacementを用いた加工再生成が必要。公開Cだけから実行できない。公開source回復2成分は上で別途再評価。',source=str(wp.relative_to(root)))]
    paths=[Path(__file__),HERE/'official_audit.py',ap,dp,cp,op,wp,vp,bgpath]
    paths.extend(official._source_paths)
    paths.extend(dist/f'C_{k}.csv' for k in teams)
    own=next((root/'予備戦/配布データ').glob('PWSCup2026_prelim_team05_*/B_self.csv'));paths.append(own)
    truthdir=next((root/'予備戦/配布データ/結果').glob('正解*'));paths.extend(truthdir.glob('pool_ids_*.csv'))
    gather.last_source_hashes={str(p.relative_to(root)):_sha(p) for p in paths}
    return records,notes


if __name__=='__main__':
    root=Path(sys.argv[1]) if len(sys.argv)>1 else HERE.parents[4]
    records,notes=gather(root,use_cache=False)
    cache=HERE/'historical_virtual_cache.json'
    cache.write_text(json.dumps(dict(records=records,notes=notes,source_hashes=gather.last_source_hashes),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(methods=len({r['method'] for r in records}),records=len(records),over25=sorted({r['team'] for r in records if r['tpr']>.25})),ensure_ascii=False))
