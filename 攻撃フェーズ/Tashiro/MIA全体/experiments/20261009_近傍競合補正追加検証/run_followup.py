"""MIA全体・公開予備戦の近傍競合補正（探索研究）。
実所属を予測には一切使わず、全方式の予測をハッシュ固定してから採点する。
過去に予備戦正解へ繰返し接触しているため、独立検証とは主張しない。
本戦回答CSV・非公開の他チームB・採点サーバーは使用しない。
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PREVIOUS_SHA = '28b6a7c6106d89c9bcca6ca138c0aaae23c0b7c03e8dc299c34efd98c9bd4cb3'
BASELINE_TPR = (0.08079772079772081, 0.07895156695156695,
                0.07702279202279202, 0.07687464387464388)
VIEWS = ('full','sub80_seed20261009','sub80_seed20261010','sub80_seed20261011')
METHODS = ('P01','P01_plus_max0.1','nearest5_selfonly0.05',
           'nearest5_max0.05','nearest5_mean0.1','neighbor2_median0.1','cat025_gated0.1')
DROPS = {'all20':(), 'without1_2':(1,2), 'without1_2_12':(1,2,12)}
NUM5 = ('age','SBP','TG','HDL','ALT')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def emit(path, obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def import_baseline(root):
    path=root/'攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py'
    if digest(path)!=PREVIOUS_SHA: raise ValueError('過去実験コードのSHA256が変わっています')
    spec=importlib.util.spec_from_file_location('previous_mia_joint',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def public_neighbours(common,b,a,cs,teams):
    """公開Cと自B_selfだけで作る特徴。正解pool_idsは読まない。"""
    scale=common.scales(a,NUM5)
    B=common.numbers(b,NUM5)/scale
    ans={name:[] for name in ('num5_nearest','num5_two','num5_category025')}
    for team in teams:
        c=cs[team]
        C=common.numbers(c,NUM5)/scale
        d2=cdist(B,C,metric='sqeuclidean')
        near=np.partition(d2,1,axis=1)
        ans['num5_nearest'].append(near[:,0])
        ans['num5_two'].append(near[:,:2].mean(1))
        mismatch=((b.sex.to_numpy()[:,None]!=c.sex.to_numpy()[None,:]) |
                  (b.prefecture.to_numpy()[:,None]!=c.prefecture.to_numpy()[None,:]))
        # 元の探索コードと同じ：「どちらかが異なれば」一律0.25を加算。
        # sex差+prefecture差を別々に数える式ではない。
        ans['num5_category025'].append((d2+0.25*mismatch).min(1))
    return {name:np.column_stack(cols) for name,cols in ans.items()}

def score_methods(prev,cal,public,normal,indices):
    r=prev.ranks(cal[normal][:,indices]); p=r-0.1*prev.rivals(r,np.median)
    f=prev.ranks(-public['num5_nearest'][normal][:,indices])
    two=prev.ranks(-public['num5_two'][normal][:,indices])
    cat=prev.ranks(-public['num5_category025'][normal][:,indices])
    cat_other=prev.rivals(cat,np.max)
    methods={
        'P01':p,
        'P01_plus_max0.1':p-0.1*prev.rivals(r,np.max),
        'nearest5_selfonly0.05':p+0.05*f,
        'nearest5_max0.05':p+0.05*(f-prev.rivals(f,np.max)),
        'nearest5_mean0.1':p+0.1*(f-prev.rivals(f,np.mean)),
        'neighbor2_median0.1':p+0.1*(two-prev.rivals(two,np.median)),
        'cat025_gated0.1':p+0.1*(cat-cat_other)*(cat_other>0.9)
    }
    assert set(methods)==set(METHODS)
    assert all(z.shape==(normal.sum(),len(indices)) and np.isfinite(z).all() for z in methods.values())
    return methods

def overlap_top10(mat):
    a=np.concatenate([np.argsort(-mat[:,j],kind='stable')[:10] for j in range(mat.shape[1])])
    v=np.bincount(a,minlength=len(mat))
    return dict(duplicate_records=int((v>1).sum()),extra_top10_slots=int(np.maximum(v-1,0).sum()))

def metric_table(prev,values,normal,private,rare,y,indices,teams):
    conf=prev.confidence(values,normal,private,rare)
    assert (conf[private]==0).all() and (conf[rare & ~private]==0.4).all()
    return [prev.metrics(conf[:,j],y[:,ix]) for j,ix in enumerate(indices)]

def execute(root,out,score_cache=None,full_only=False):
    if out.exists(): raise FileExistsError(f'既存出力は上書きしません: {out}')
    out.mkdir(parents=True)
    prev=import_baseline(root)
    common,calib,base,paths,b,a,qs,cs,teams,private,rare,normal=prev.load_inputs(root)
    assert len(teams)==20 and int(normal.sum())==875
    public=public_neighbours(common,b,a,cs,teams)
    emit(out/'protocol.json',{'task':'Published preliminary MIA cross-target proximity exploratory follow-up',
        'source_previous_sha256':PREVIOUS_SHA,'methods':list(METHODS),'views':list(VIEWS),'reduced_inference_sets':DROPS,
        'scores':'Normal875 only; private163=0, non-private rare11=0.4 for ALL methods',
        'proximity':'age SBP TG HDL ALT, TG/ALT log1p; background IQR scales; squared Euclidean nearest or nearest-two mean',
        'comparison':'nearest feature column percentile ranks compared with foreign team max/mean/median; category mismatch is logical OR with 0.25 cost',
        'selection_warning':'Methods chosen after testing hundreds on exposed preliminary labels; backgrounds share individuals, not independent validation',
        'label_access':'No pool_ids before all predictions are fixed; labels used for evaluation only'})
    wanted=VIEWS[:1] if full_only else VIEWS
    bundles={};h=[];input_hashes={str(p.relative_to(root)):digest(p) for p in paths}
    # 事前に予測を全方式固定する。pool_idsを読まない。
    for view in wanted:
        if score_cache is not None:
            file=score_cache/f'_scores_{view}.npz'; data=np.load(file)
            raw,cal=data['raw'],data['cal']
            input_hashes['cache/'+file.name]=digest(file)
        else:
            if view=='full': aa=a
            else:
                rng=np.random.default_rng(int(view.rsplit('seed',1)[1]))
                chosen=np.sort(rng.choice(len(a),int(.8*len(a)),replace=False))
                aa=a.iloc[chosen]
            raw,cal=prev.raw_scores(common,calib,b,aa,qs,cs,teams)
        assert raw.shape==cal.shape==(1049,20)
        for setname,drop in DROPS.items():
            if view!='full' and setname!='all20':continue
            ix=[j for j,t in enumerate(teams) if t not in drop]
            ms=score_methods(prev,cal,public,normal,ix)
            bundles[view,setname]=(ix,ms)
            for method,v in ms.items():
                h.append({'view':view,'inference':setname,'method':method,
                    'sha256':hashlib.sha256(np.ascontiguousarray(v,dtype='<f8').tobytes()).hexdigest()})
    emit(out/'prediction_hashes.json',h)
    # 正解を初めて読む位置。上のpredictionsは以後書き換えない。
    y,audit,truth_paths=prev.audit_truth(base,b,private,rare,teams)
    emit(out/'constraint_audit.json',audit)
    emit(out/'truth_input_sha256.json',{str(p.relative_to(root)):digest(p) for p in truth_paths})
    summary=[]; per_target=[]; fpr=[]
    for (view,setname),(ix,ms) in bundles.items():
        ids=[teams[j] for j in ix]
        base_metrics=metric_table(prev,ms['P01'],normal,private,rare,y,ix,teams)
        baseline=np.array([z['tpr_0.01'] for z in base_metrics])
        for method,mat in ms.items():
            val=metric_table(prev,mat,normal,private,rare,y,ix,teams)
            current=np.array([z['tpr_0.01'] for z in val]);d=current-baseline
            summary.append({'background':view,'inference':setname,'method':method,'targets':len(ix),
                'mean_tpr':float(current.mean()),'mean_delta':float(d.mean()),
                'mean_delta_without1_2':float(np.mean([d[j] for j,t in enumerate(ids) if t not in (1,2)])),
                'wins':int((d>1e-12).sum()),'ties':int((abs(d)<=1e-12).sum()),'losses':int((d<-1e-12).sum()),
                'worst_delta':float(d.min()),**overlap_top10(mat)})
            if view=='full' and setname=='all20':
                per_target.extend({'team':t,'method':method,'tpr':float(current[j]),'delta':float(d[j])} for j,t in enumerate(ids))
            if setname=='all20':
                fpr.append({'background':view,'method':method,
                    'mean_tpr005':float(np.mean([z['tpr_0.005'] for z in val])),
                    'mean_tpr01':float(current.mean()),
                    'mean_tpr02':float(np.mean([z['tpr_0.02'] for z in val])),
                    'mean_auc':float(np.mean([z['auc'] for z in val])),
                    'expected_top10_hits':float(sum(z['top10_expected_hits'] for z in val))})
    for fname,rows in [('summary.csv',summary),('per_target_full.csv',per_target),('fpr_metrics.csv',fpr)]:
        pd.DataFrame(rows).to_csv(out/fname,index=False,float_format='%.17g')
    # Prior P01 regression check; stops on dataset/source mismatch
    for i,v in enumerate(wanted):
        obs=next(z for z in summary if z['background']==v and z['inference']=='all20' and z['method']=='P01')['mean_tpr']
        assert abs(obs-BASELINE_TPR[i])<1e-12,(v,obs,BASELINE_TPR[i])
    # Negative-control randomization is evaluated after predictions are frozen; post-selection comparator, not formal p.
    if 'full' in wanted:
        ix,ms=bundles['full','all20'];x=ms['P01']
        fp=prev.ranks(-public['num5_nearest'][normal]);rng=np.random.default_rng(20261009)
        original=next(z for z in summary if z['background']=='full' and z['inference']=='all20' and z['method']=='nearest5_mean0.1')['mean_delta']
        null=[]
        for _ in range(200):
            perm=np.argsort(rng.random(fp.shape),axis=1)
            shuffled=fp[np.arange(len(fp))[:,None],perm]
            sc=x+0.1*(shuffled-prev.rivals(shuffled,np.mean))
            vals=metric_table(prev,sc,normal,private,rare,y,ix,teams)
            now=np.mean([z['tpr_0.01'] for z in vals]);null.append(now-BASELINE_TPR[0])
        null=np.array(null)
        emit(out/'null_control.json',{'seed':20261009,'repetitions':200,'observed_delta':original,
            'null_mean':float(null.mean()),'null_p05':float(np.quantile(null,.05)),
            'null_p95':float(np.quantile(null,.95)),
            'fraction_ge_observed':float(np.mean(null>=original)),
            'meaning':'Shuffle 20 C-feature columns within each record. Descriptive negative control, NOT p-value due prior model selection.'})
    emit(out/'run_record.json',{'date_utc':datetime.now(timezone.utc).isoformat(),
        'python':platform.python_version(),'code_sha256':digest(__file__), 'previous_code_sha256':PREVIOUS_SHA,
        'input_sha256':input_hashes,'score_cache_used':score_cache is not None,
        'full_only':full_only,'no_main_truth':True,'submitted':False})
    print(pd.DataFrame(summary).query("background=='full' and inference=='all20'")[['method','mean_tpr','mean_delta','wins','losses','duplicate_records']].to_string(index=False))
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--score-cache',type=Path,help='前回の内部NPZ（ローカル専用）。省略時は公式データから再計算')
    p.add_argument('--full-only',action='store_true')
    args=p.parse_args()
    execute(args.root.resolve(),args.output_dir.resolve(),args.score_cache,args.full_only)
if __name__=='__main__':main()
