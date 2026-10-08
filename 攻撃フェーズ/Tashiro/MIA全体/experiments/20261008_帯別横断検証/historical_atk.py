"""旧atk診断の既存公開入力scoreを大会正解で再評価。main/旧proxyは使わない。"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
CONT=['age','BMI','SBP','TG','HDL','ALT','FPG']

def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def gather(root,use_cache=True):
    root=Path(root).resolve();cache=HERE/'historical_atk_cache.json'
    if use_cache and cache.exists():
        saved=json.loads(cache.read_text())
        if all((root/p).exists() and _sha(root/p)==h for p,h in saved['source_hashes'].items()):
            return saved['records'],saved['notes']
    audit=_module('_historical_atk_official',HERE/'official_audit.py')
    official=audit.load_official(root);truths=audit.load_truth(root)
    b,_,_,_,dist=audit.load_inputs(root);teams=sorted(truths)
    bgpath=root/'予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv'
    a=pd.read_csv(bgpath,float_precision='round_trip')
    cs={k:pd.read_csv(dist/f'C_{k}.csv',float_precision='round_trip') for k in teams}
    old=root/'アーカイブ/予備戦_攻撃フェーズ/work_20260901以前'
    paths=[Path(__file__),HERE/'official_audit.py',old/'attacks.py',bgpath]
    paths.extend(old/p for p in ['atk19/mia.py','atk19222023/mia2.py','atk242629/p6_mia.py','atk242629b/mia.py','atk242629b/mia2.py'])
    at=_module('_historical_atk_library',old/'attacks.py')
    cols8=CONT+['time'];mu=a[cols8].mean().to_numpy();sd=a[cols8].std(ddof=0).to_numpy()
    def z(frame,cols):
        idx=[cols8.index(c) for c in cols]
        return (frame[cols].to_numpy()-mu[idx])/sd[idx]
    # scores/整合対照は元純関数をそのまま切り出す。
    tree=ast.parse((old/'atk242629b/mia.py').read_text())
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['zz','matched_controls','scores']]
    env=dict(np=np,pd=pd,NearestNeighbors=NearestNeighbors,AT=at,CONT=CONT,CONT8=cols8,mu=mu,sd=sd)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(old/'atk242629b/mia.py'),'exec'),env)
    za=z(a,CONT);zb=z(b,CONT);kde_a=gaussian_kde(za.T)
    matrices={};backgrounds={};details={};records=[];started=time.monotonic()
    with threadpool_limits(limits=1):
        for j,k in enumerate(teams):
            c=cs[k];zc=z(c,CONT)
            dc=NearestNeighbors(n_neighbors=5).fit(zc).kneighbors(zb)[0]
            da=NearestNeighbors(n_neighbors=5).fit(za).kneighbors(zb)[0]
            vals={
                'ratio_kA1_eps1e6':-(dc[:,0]/np.clip(da[:,0],1e-6,None)),
                'ratio_k5_eps1e6':-(dc[:,4]/np.clip(da[:,4],1e-6,None)),
                'logkde_floor1e300':np.log(np.clip(gaussian_kde(zc.T)(zb.T),1e-300,None))-np.log(np.clip(kde_a(zb.T),1e-300,None)),
                'raw_nn8':-NearestNeighbors(n_neighbors=1).fit(z(c,cols8)).kneighbors(z(b,cols8))[0][:,0],
            }
            # atk19のQI5 block。欠落blockの最大有限距離への置換も維持。
            qi=['age','SBP','TG','HDL','ALT'];dd=np.full(len(b),np.inf)
            for key,idx in c.groupby(['sex','prefecture']).groups.items():
                mask=((b.sex==key[0])&(b.prefecture==key[1])).to_numpy()
                if mask.any():
                    dd[mask]=NearestNeighbors(n_neighbors=1).fit(z(c.loc[idx],qi)).kneighbors(z(b.loc[mask],qi))[0][:,0]
            vals['block_qi5_maxfallback']=-np.where(np.isfinite(dd),dd,np.nanmax(dd[np.isfinite(dd)]))
            dac=NearestNeighbors(n_neighbors=1).fit(zc).kneighbors(za)[0][:,0]
            clean=a[dac>.05].reset_index(drop=True);backgrounds[k]=len(clean)
            for name,fn,params in [('nn_gap',at.m3_nn_gap,{}),('knn15',at.m4_knn_density,{'k':15}),('knn3',at.m4_knn_density,{'k':3}),('domias',at.m1_domias,{})]:
                vals['clean_rescaled_'+name]=fn(c,clean,b,**params)
            original=env['scores'](c,b,clean)
            for name in ['nn_gap','knn15','nnblock_neg']:
                vals['clean_originalscale_'+name]=original[name]
            for name,scores in vals.items():
                assert np.isfinite(scores).all(),(name,k)
                matrices.setdefault(name,np.empty((len(b),len(teams))))[:,j]=scores
            print(f'旧atk固定scoreの公式再採点: {j+1}/20対象 ({time.monotonic()-started:.0f}秒)',flush=True)
        # 種1の元rng・元対象順24/26/29を維持。対照抽出によるrng消費も維持。
        rng=np.random.default_rng(1)
        for k in [24,26,29]:
            c=cs[k];dac=NearestNeighbors(n_neighbors=1).fit(z(c,CONT)).kneighbors(za)[0][:,0]
            clean=a[dac>.05].reset_index(drop=True);perm=rng.permutation(len(clean));half=len(perm)//2
            ctl=clean.iloc[perm[:half]].reset_index(drop=True);ref=clean.iloc[perm[half:]].reset_index(drop=True)
            bm=b[b.prefecture!='東京都'].reset_index(drop=True)
            env['matched_controls'](bm,ctl,rng)
            out=env['scores'](c,b,ref)
            for name in ['nn_gap','knn15','domias']:
                details.setdefault('split_seed1_original_'+name,{})[k]=(out[name],len(ref))
    def add(name,k,scores,background):
        tpr,auc=official._tpr_at_fpr(scores,truths[k],.01)
        clean='clean_' in name or 'split_' in name
        source='atk19/mia.py' if 'block_qi5' in name else 'atk19222023/mia2.py' if name.startswith(('ratio','logkde','raw_nn8')) else 'atk242629/p6_mia.py' if 'rescaled' in name else 'atk242629b/mia2.py' if name.startswith('split') else 'atk242629b/mia.py'
        records.append(dict(team=k,method='legacy_atk_'+name,condition='既存score定義を予備戦B全1049へ適用',
            tpr=float(tpr),members=int(truths[k].sum()),attributes=cols8 if name=='raw_nn8' else ['age','SBP','TG','HDL','ALT','sex','prefecture'] if 'qi5' in name else CONT+['sex','prefecture'] if 'block' in name else CONT,
            params=name+';A_bg mean/SD ddof0;clean閾値.05;元seed1は24/26/29の順とrng消費を維持' if clean else name+';A_bg mean/SD ddof0;原コードのepsilon/fallbackを維持',
            background=background,evaluation_class='C',split='所属真値は採点だけ。旧B対対照のproxyラベルを使わない。',
            transductive='対象Cと公開A_bgを参照',source=str((old/source).relative_to(root))+'#'+name,
            verification='元純数値定義/score関数を再計算・現公式採点',family='旧atkの近傍・密度比条件',selection='事前固定方式',auc=float(auc),
            notes='原診断の保存AUC/TPRは使用しない。対照抽出5 seedはB-score不変なので別条件へ数えない。半分割rng順は元3対象のみ。'))
    for name,matrix in matrices.items():
        for j,k in enumerate(teams):
            add(name,k,matrix[:,j],f'予備戦A_bg5143→対象C近接除外{backgrounds[k]}' if name.startswith('clean') else '予備戦A_bg 5143行')
    for name,values in details.items():
        for k,(scores,nref) in values.items():add(name,k,scores,f'予備戦A_bg5143→対象C近接除外→元種1半分割A_ref{nref}')
    notes=[dict(scope='atk raw_nn7 / clean domias alias',status='明示的同一定義の別名',reason='raw7はlegacy_cross_nn1_raw_closeness、clean_originalscale domiasはlegacy_atk_clean_rescaled_domiasと同じ純関数/入力。別方式へ数えない。',source=str(old.relative_to(root))),
           dict(scope='atk242629b半分割の未定義対象',status='元3対象だけ再現',reason='元rng種1は24→26→29の順にpermutationと整合対照抽出で進む。残対象の新rng順を設けず、3対象のみ再評価。',source=str((old/'atk242629b/mia2.py').relative_to(root))),
           dict(scope='atk量子化grid診断',status='個別confidence未保存・未再現',reason='binmatch_B/ctlの集合一致件数だけが元診断出力。個別membership confidenceを定義する新攻撃へ作り替えない。',source=str((old/'atk19222023/mia2.py').relative_to(root)))]
    paths.extend(official._source_paths);paths.extend(dist/f'C_{k}.csv' for k in teams)
    paths.append(next((root/'予備戦/配布データ').glob('PWSCup2026_prelim_team05_*/B_self.csv')))
    truthdir=next((root/'予備戦/配布データ/結果').glob('正解*'));paths.extend(truthdir.glob('pool_ids_*.csv'))
    gather.last_source_hashes={str(p.relative_to(root)):_sha(p) for p in paths}
    return records,notes

if __name__=='__main__':
    root=Path(sys.argv[1]) if len(sys.argv)>1 else HERE.parents[4]
    records,notes=gather(root,use_cache=False)
    (HERE/'historical_atk_cache.json').write_text(json.dumps(dict(records=records,notes=notes,source_hashes=gather.last_source_hashes),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(methods=len({r['method'] for r in records}),records=len(records),over25=sorted({r['team'] for r in records if r['tpr']>.25})),ensure_ascii=False))
