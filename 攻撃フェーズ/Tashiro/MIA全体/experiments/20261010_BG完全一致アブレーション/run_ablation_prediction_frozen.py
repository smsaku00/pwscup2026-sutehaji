"""Fixed public-background-overlap ablation; original input files are read-only.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_ablation.py --manifest ...
No membership labels are opened until every requested score matrix is frozen.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,unicodedata,platform
from datetime import datetime,timezone
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
import 分析 as common
import 非会員を使う順位較正の検討 as calib
import base,archive,bounded_core
bounded_core.install(common)
TARGETS=(18,19,29)
SEEDS=(20261010,20261011,20261012,20261013,20261014)
METHODS=('P01','q13_median_iqr_sex0_kde_ratio_h2__raw__cross0.1')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def msha(x):return hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
def dump(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return pd.read_csv(p,float_precision='round_trip')
def keys(df,cols):
    d=df[list(cols)].copy()
    for c in cols:
        if c=='prefecture':d[c]=d[c].map(lambda s:unicodedata.normalize('NFC',str(s)))
        else:d[c]=d[c].astype(float)
    return list(d.itertuples(index=False,name=None))

def load(m):
    meta=m['phases']['prelim']; attack=Path(meta['attack_dir']);b=read(meta['b_self'])
    cs={int(p.stem.split('_')[-1]):read(p) for p in attack.glob('C_*.csv')}
    qs={int(p.stem.split('_')[-1]):read(p) for p in attack.glob('aia_challenge_*.csv')}
    teams=meta['target_ids'];assert teams==sorted(set(cs)-{5})
    q=pd.concat([qs[t][list(common.QI7)] for t in sorted(qs)],ignore_index=True)
    qsources=np.concatenate([np.repeat(t,len(qs[t])) for t in sorted(qs)])
    private=pd.MultiIndex.from_frame(b[list(common.QI7)]).isin(pd.MultiIndex.from_frame(qs[5][list(common.QI7)]))
    rare=b.is_rare.to_numpy(bool);normal=~private & ~rare
    assert (len(b),private.sum(),rare.sum(),normal.sum())==(1049,163,24,875)
    frames=pd.concat([b[list(common.QI7)],q],ignore_index=True)
    return b,cs,qs,q,qsources,private,rare,normal,frames,teams

def overlap(m):
    checks=[dict(path=f['path'],expected=f['sha256'],actual=sha(f['path'])) for f in m['files']]
    assert len(checks)==142 and all(z['expected']==z['actual'] for z in checks)
    dump(HERE/'input_hash_verification.json',{'original_file_count':142,'checks':checks})
    rows=[]
    for cp in sorted(Path(m['phases']['prelim']['attack_dir']).glob('C_*.csv')):
        c=read(cp);team=int(cp.stem.split('_')[-1])
        for source,meta in m['backgrounds'].items():
            a=read(meta['path']);assert sha(meta['path'])==meta['sha256']
            row=dict(team=team,source=source,c_rows=len(c),bg_rows=len(a),c_path=str(cp),bg_path=meta['path'])
            for label,cols in [('full13',m['features13']),('qi7',m['qi7'])]:
                ck,ak=keys(c,cols),keys(a,cols);cs,aset=set(ck),set(ak)
                row.update({f'{label}_unique':len(cs&aset),f'{label}_c_rows':sum(k in aset for k in ck),f'{label}_bg_rows':sum(k in cs for k in ak)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(HERE/'prelim_all_teams_overlap.csv',index=False)

def predict(a,data):
    b,cs,qs,q,qsource,private,rare,normal,frames,teams=data
    nb=len(b);raw=[];cal=[];started=time.time()
    for t in teams:
        v=common.core(a,frames,cs[t],common.NUM7,(10,))[10]
        raw.append(v)
        valid=qsource!=t
        cal.append(calib.calibrate(v[:nb],b,v[nb:][valid],q.loc[valid],['prefecture','sex'],500))
    r=base.ranks(np.column_stack(cal)[normal]);p01=r-.1*base.rivals(r,np.median)
    kde,h=archive.scores_KDEA(common,b,a,cs,teams)
    kr=archive.ranks(kde,normal);kdea=kr-.1*base.rivals(kr,np.median)
    return {'P01':p01,METHODS[1]:kdea},np.column_stack(raw),{'h_kde13':h,'genlra_scale':common.scales(a,common.NUM7).tolist(),'seconds':time.time()-started}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=Path('/workspace/shared/mia_bg7_inputs/manifest.json'));ap.add_argument('--pool',default='starter+prelim');args=ap.parse_args()
    m=json.loads(args.manifest.read_text());overlap(m);data=load(m)
    b,cs,qs,q,qsource,private,rare,normal,frames,teams=data
    pool=args.pool;poolmeta=m['subsets'][pool];a=read(poolmeta['path']);assert sha(poolmeta['path'])==poolmeta['sha256']
    source=np.concatenate([np.repeat(s,m['backgrounds'][s]['rows']) for s in poolmeta['sources']]);assert len(source)==len(a)
    out=HERE/pool;out.mkdir(exist_ok=True)
    protocol={'pool':pool,'pool_rows':len(a),'targets':TARGETS,'methods':METHODS,'inference_targets':teams,
      'method_selection':'Fixed before new ablation outputs; historical method development used preliminary labels, so not independent validation.',
      'comparison':'For each target remove BG rows whose full13 tuple matches that target C; all20 model columns refit on that same modified BG.',
      'control':'Five predeclared seeds; source-stratified same-count removal from non-overlapping BG rows without replacement, retaining original row order.',
      'random_seeds':SEEDS,'matching':'No record_id joins; exact float64 round_trip numeric values, NFC prefecture, no rounding/tolerance.',
      'candidate_and_label_policy':'1049 fixed B candidates; original private=0, rare_nonprivate=.4, normal875=.6+.3*minrank/875. Scoring labels only after freeze.',
      'calibration':'Original non-target public Q prefecture/sex shrink500, all Q; no fold change or membership-training labels.',
      'preprocessing':'Refit all BG scales, bandwidth, category counts, KDE numerator alignment and denominator; fixed hyperparameters.',
      'KDE_variant':'Ungated standalone original q13 KDE ratio cross0.1, not gated mix1 or a newly chosen winner.',
      'manifest_sha256':sha(args.manifest),'background_sha256':poolmeta['sha256'],'code_hashes':{str(p.relative_to(HERE)):sha(p) for p in [Path(__file__)]+sorted((HERE/'source').glob('*.py'))}}
    p=out/'fixed_protocol.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(protocol))
    else:dump(p,protocol)
    aks=keys(a,m['features13']);conditions=[('original',0,'original',None,np.zeros(len(a),bool))]
    removalrows=[]
    for t in TARGETS:
        ck=set(keys(cs[t],m['features13']));exact=np.array([x in ck for x in aks],bool)
        counts={s:int(exact[source==s].sum()) for s in poolmeta['sources']}
        assert exact.sum()>0
        conditions.append((f'team{t}_exact',t,'exact',None,exact))
        for seed in SEEDS:
            rng=np.random.default_rng(seed);remove=np.zeros(len(a),bool)
            for s,n in counts.items():
                allowed=np.flatnonzero((source==s)&~exact)
                remove[rng.choice(allowed,n,replace=False)]=True
            assert remove.sum()==exact.sum() and not (remove&exact).any()
            conditions.append((f'team{t}_random{seed}',t,'random',seed,remove))
    # Run exact deletions before controls for progress, without changing the preset schedule.
    conditions.sort(key=lambda z:({'original':0,'exact':1,'random':2}[z[2]],z[1],z[3] or 0))
    hashes=[];bundles=[]
    for name,target,kind,seed,remove in conditions:
        t0=time.time();matrix=out/f'{name}.npz';side=out/f'{name}.json'
        cfg=dict(condition=name,target=target,kind=kind,seed=seed,removed=int(remove.sum()),remaining=int((~remove).sum()),
                 removed_by_source={s:int(remove[source==s].sum()) for s in poolmeta['sources']},
                 removal_mask_sha256=hashlib.sha256(remove.tobytes()).hexdigest(),protocol_sha256=sha(p))
        if matrix.exists() and side.exists():
            rec=json.loads(side.read_text());assert rec['config']==cfg and rec['matrix_sha256']==sha(matrix)
            with np.load(matrix) as z:scores={method:z[str(j)].copy() for j,method in enumerate(METHODS)}
        else:
            print('START',name,'removed',cfg['removed'],flush=True)
            scores,raw,detail=predict(a.loc[~remove].reset_index(drop=True),data)
            if kind=='original':
                cache=Path('/workspace/shared/mia_all_bg_available/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/available/results_prelim')/f'raw_cache_{pool}.npz'
                cached=np.load(cache)['raw'];assert np.array_equal(cached,raw)
                pred=cache.parent/f'predictions_{pool}_all20.npz';gold=np.load(pred)['P01'];assert np.array_equal(scores['P01'],gold)
                detail['original_raw_bitwise_regression']=True;detail['P01_bitwise_regression']=True
            np.savez_compressed(matrix,**{str(j):scores[x] for j,x in enumerate(METHODS)})
            rec={'config':cfg,'fit_details':detail,'matrix_sha256':sha(matrix),'prediction_hashes':{x:msha(v) for x,v in scores.items()},'frozen_utc':datetime.now(timezone.utc).isoformat()};dump(side,rec)
        hashes.append(rec);bundles.append((cfg,scores));print('FROZEN',name,round(time.time()-t0,2),'sec',flush=True)
    dump(out/'prediction_freeze.json',{'conditions':len(hashes),'frozen_utc':datetime.now(timezone.utc).isoformat(),'records':hashes})
    # Released preliminary truth is read only below this completed prediction freeze.
    folder=Path(m['phases']['prelim']['attack_dir']).parent
    y,audit,truthpaths=base.audit_truth(folder,b,private,rare,teams)
    dump(out/'truth_audit.json',{'audit':audit,'truth_hashes':{str(p):sha(p) for p in truthpaths}})
    original=bundles[0][1];rows=[]
    for cfg,scores in bundles:
        focus=TARGETS if cfg['kind']=='original' else (cfg['target'],)
        for method,score in scores.items():
            confidence=base.confidence(score,normal,private,rare)
            for t in focus:
                j=teams.index(t);r=score[:,j];ref=original[method][:,j]
                common_fields={**cfg,'team':t,'method':method,'spearman_original':float(spearmanr(r,ref).statistic),'top20_overlap_original':len(set(np.argsort(-r,kind='stable')[:20])&set(np.argsort(-ref,kind='stable')[:20])),'mean_abs_rank_shift':float(np.mean(np.abs(rankdata(r)-rankdata(ref))))}
                for pop,mask in [('all_fixed',np.ones(len(b),bool)),('normal',normal)]:
                    rows.append({**common_fields,'population':pop,'n':int(mask.sum()),'members':int(y[mask,j].sum()),**base.metrics(confidence[mask,j],y[mask,j])})
    df=pd.DataFrame(rows);df.to_csv(out/'by_condition.csv',index=False,float_format='%.17g')
    summaries=[]
    for (t,method,pop),g in df.groupby(['team','method','population'],sort=False):
        orig=g[g.kind=='original'].iloc[0];exact=g[g.kind=='exact'].iloc[0];rnd=g[g.kind=='random']
        for metric in ('tpr_0.01','auc','top20_expected_hits'):
            v=rnd[metric].to_numpy();delta=float(exact[metric]-orig[metric])
            summaries.append(dict(team=t,method=method,population=pop,metric=metric,original=float(orig[metric]),remove_exact=float(exact[metric]),delta_exact=delta,random_mean=float(v.mean()),random_min=float(v.min()),random_max=float(v.max()),random_sd=float(v.std(ddof=1)),excess_over_random_mean=float(exact[metric]-v.mean()),removed=int(exact.removed),random_repeats=len(v)))
    pd.DataFrame(summaries).to_csv(out/'ablation_summary.csv',index=False,float_format='%.17g')
    dump(out/'completion.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'condition_count':len(conditions),'rows':len(df),'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__})
    print(pd.DataFrame(summaries).query("population == 'all_fixed' and metric == 'tpr_0.01'").to_string(index=False),flush=True)
if __name__=='__main__':main()
