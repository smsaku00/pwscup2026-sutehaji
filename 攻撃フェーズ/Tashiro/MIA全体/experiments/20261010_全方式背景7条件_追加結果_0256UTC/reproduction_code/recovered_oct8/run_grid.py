"""Run every original fixed Oct8 setting, changing only selected background.

Imports intact original source. setup/output adaptation never reads labels
during inference. Target-independent caching and query-row blocking are
validated bit-exact in optimization_validation.json.
"""
from pathlib import Path
import argparse,csv,gc,hashlib,importlib,json,sys,time
from datetime import datetime,timezone
import numpy as np,pandas as pd
from scipy.stats import rankdata
from benchmark import data,u,HERE,SRC
import bg7_inputs
import optimized as opt

BG=('prelim','starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main')
MODULES={'histogram':'階層ヒストグラムの探索','geometry':'距離と分布の探索','alignment':'分布を揃えた距離の探索','mixture':'混合分布と局所密度の探索','local':'混合分布と局所密度の探索'}
EXPECTED={'histogram':1920,'geometry':666,'alignment':750,'mixture':960,'local':300}
COUNTS={}
class Frozen(Exception):pass
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def msha(x):return hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def seal(out,d,models,specs,started,extra=None):
    out=Path(out);out.mkdir(exist_ok=True,parents=True);names=sorted(models);mat=np.stack([models[n] for n in names])
    assert mat.shape==(EXPECTED[ARGS.family],875,len(d['teams'])) and np.isfinite(mat).all()
    np.savez_compressed(out/'scores.npz',names=np.array(names),scores=mat,teams=np.array(d['teams']))
    dump(out/'prediction_hashes.json',[{'method':n,'sha256':msha(models[n]),'shape':list(models[n].shape)} for n in names])
    pd.DataFrame(specs).to_csv(out/'method_specs.csv',index=False)
    caller=sys._getframe(1).f_locals
    if caller.get('fitrecords'):pd.DataFrame(caller['fitrecords']).to_csv(out/'fit_diagnostics.csv',index=False)
    dump(out/'prediction_record.json',{'phase':ARGS.phase,'background':CURRENT_BG,'family':ARGS.family,'methods':len(names),'bg_rows':len(d['a']),'targets':d['teams'],'seconds':time.perf_counter()-started,'frozen_at':datetime.now(timezone.utc).isoformat(),'truth_read':False,'scores_sha256':sha(out/'scores.npz'),'extra':extra or {}})
    print('FROZEN',ARGS.family,ARGS.phase,CURRENT_BG,len(names),round(time.perf_counter()-started,1),flush=True)
    raise Frozen()

def local_run(d,out):
    mix=importlib.import_module(MODULES['local']);geo=importlib.import_module(MODULES['geometry']);start=time.perf_counter();a,b,q=d['a'],d['b'],d['query'];banks={};specs=[]
    for label,nums,cats in [('q7',u.NUM7,u.CAT7),('q9',u.NUM9,u.CAT7),('q13',u.NUM13,u.CAT13)]:
        query=q if label=='q7' else b;calibration=label=='q7'
        for kind in ('robust','rank'):
            ar,x=geo.transform(a,[a,query],nums,kind);ka={k:min(len(a)-1,round(k*len(a)/1049)) for k in (5,15,50)}
            sa=opt.anchor_scales(ar,list(ka.values()),len(nums));den={(k,m):opt.adaptive(x,ar,query,a,a,cats,sa[ka[k]],m,len(nums)) for k in ka for m in (.5,1.,2.)}
            lof_a={}
            for k in (10,30,100):
                fit=mix.LocalOutlierFactor(n_neighbors=k,novelty=True,n_jobs=2);fit.fit(ar);lof_a[k]=fit.score_samples(x)
            for t,c in d['cs'].items():
                _,_,cr=geo.transform(a,[a,query,c],nums,kind);sc=opt.anchor_scales(cr,[5,15,50],len(nums));scores={}
                for k in (5,15,50):
                    for m in (.5,1.,2.):scores[f'{label}_{kind}_adaptivek{k}_h{m:g}']=opt.adaptive(x,cr,query,c,a,cats,sc[k],m,len(nums))-den[(k,m)]
                for k in (10,30,100):
                    fit=mix.LocalOutlierFactor(n_neighbors=k,novelty=True,n_jobs=2);fit.fit(cr);value=fit.score_samples(x)
                    scores[f'{label}_{kind}_lofC_k{k}']=value;scores[f'{label}_{kind}_lof_ratio_k{k}']=value-lof_a[k]
                for name,values in u.variants(scores,d,t,calibration).items():banks.setdefault(name,{})[t]=values
                if t==d['teams'][0]:specs.extend({'raw_method':name,'family':'local','public_null_calibration':calibration} for name in scores)
            print('LOCAL',CURRENT_BG,label,kind,round(time.perf_counter()-start,1),flush=True)
    seal(out,d,u.matrices(banks,d['teams']),specs,start,{'local_chunking':'row independent; exact validation at k5,15,50; all original15 signals per attr/transform retained'})

def infer(bg,out):
    global CURRENT_BG
    CURRENT_BG=bg;d=data(bg,phase=ARGS.phase);opt.clear()
    geo=importlib.import_module(MODULES['geometry']);geo.geometry=opt.geometry
    u.setup=lambda phase='prelim':d;u.save=seal
    mod=importlib.import_module(MODULES[ARGS.family])
    if ARGS.family=='alignment':mod.geometry=opt.geometry
    sys.argv=[str(SRC/(MODULES[ARGS.family]+'.py')),'--output-dir',str(out)]
    if ARGS.family=='mixture':sys.argv.extend(['--family','mixture'])
    try:
        if ARGS.family=='local':local_run(d,out)
        else:mod.main()
    except Frozen:pass
    del d;gc.collect()

def metrics(s,y):
    order=np.argsort(-s,kind='mergesort');s=s[order];y=y[order]
    end=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1];n=y.sum();nn=len(y)-n
    fp=np.r_[0,np.cumsum(1-y)[end]/nn];tp=np.r_[0,np.cumsum(y)[end]/n]
    return [float(np.interp(alpha,fp,tp)) for alpha in u.FPRS]+[float(np.trapezoid(tp,fp))]

def evaluate(out):
    # All requested background outputs are sealed and verified before truth.
    for bg in BG:
        d=out/bg;rec=json.loads((d/'prediction_record.json').read_text());assert sha(d/'scores.npz')==rec['scores_sha256'] and not rec['truth_read']
    phase=data('prelim');b=phase['b'];teams=phase['teams'];truthroot=Path('/workspace/shared/mia_bg7_inputs/root/公式データ/予備戦/結果/正解データ')
    pools={t:pd.read_csv(truthroot/f'pool_ids_{t}.csv') for t in [5]+teams};own=pools[5].set_index('record_id').loc[b.record_id].pool_id
    truth=np.column_stack([own.isin(pools[t].pool_id).to_numpy(int) for t in teams]);assert not truth[phase['private']].any()
    populations=[('all_fixed',np.ones(len(b),bool)),('nonrare',~phase['rare']),('normal',phase['normal'])]
    sys.path.insert(0,'/workspace/shared/mia_bg7_inputs/official_scoring/src');from pwscup2026.scoring.attack_score import _tpr_at_fpr
    summaries=[];checks=[]
    for bg in BG:
        z=np.load(out/bg/'scores.npz');names=z['names'];scores=z['scores'];summary_arrays={pop:np.empty((len(names),len(teams),5)) for pop,_ in populations}
        with (out/bg/'by_target.csv').open('w',newline='') as fp:
            writer=csv.writer(fp);writer.writerow(['background','family','method','team','population','n','members','tpr_0.005','tpr_0.01','tpr_0.02','tpr_0.05','auc'])
            for mi,name in enumerate(names):
                for j,t in enumerate(teams):
                    conf=u.base.confidence(scores[mi,:,j],phase['normal'],phase['rare'],phase['private'])
                    for pop,mask in populations:
                        y=truth[mask,j];v=metrics(conf[mask],y);summary_arrays[pop][mi,j]=v;writer.writerow([bg,ARGS.family,name,t,pop,int(mask.sum()),int(y.sum()),*v])
                        if mi==0 and t==teams[0]:
                            reference=_tpr_at_fpr(conf[mask],y,.01)[0];assert abs(reference-v[1])<1e-12
                if bg=='prelim' and name in ('q7_sex_bins40_s10_logratio__county500__cross0','q7_sex_bins40_s1_logratio__county500__cross0','q13_quantile_sex1_kde_ratio_h0.5__raw__cross0','q13_robust_adaptivek5_h0.5__raw__cross0'):
                    target=7 if name.startswith('q7_') else 26;value=summary_arrays['all_fixed'][mi,teams.index(target),1];checks.append({'method':str(name),'team':target,'expected':.16,'actual':float(value),'passed':bool(abs(value-.16)<1e-12)})
        for pop,vals in summary_arrays.items():
            for view,exclude in [('all',[]),('without12',[12]),('without1_2',[1,2]),('without1_2_12',[1,2,12])]:
                ii=[j for j,t in enumerate(teams) if t not in exclude]
                for mi,name in enumerate(names):summaries.append({'background':bg,'family':ARGS.family,'method':str(name),'population':pop,'evaluation':view,'targets':len(ii),'mean_tpr':float(vals[mi,ii,1].mean()),'mean_tpr005':float(vals[mi,ii,0].mean()),'mean_tpr020':float(vals[mi,ii,2].mean()),'mean_tpr050':float(vals[mi,ii,3].mean()),'mean_auc':float(vals[mi,ii,4].mean()),'TPR_ge010':int((vals[mi,ii,1]>=.1-1e-12).sum())})
        print('SCORED',ARGS.family,bg,len(names),'methods',flush=True)
    pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False,float_format='%.17g');dump(out/'historical_anchor_checks.json',checks);assert all(x['passed'] for x in checks),checks
    dump(out/'completion.json',{'complete':True,'phase':'prelim','backgrounds':7,'settings_per_background':EXPECTED[ARGS.family],'condition_panels':7*EXPECTED[ARGS.family],'targets':20,'main_deferred':True,'truth_only_after_all_background_prediction_seals':True,'anchor_checks':checks})

def main():
    global ARGS
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=list(MODULES),required=True);parser.add_argument('--phase',default='prelim');ARGS=parser.parse_args()
    out=HERE/ARGS.family/ARGS.phase;out.mkdir(parents=True,exist_ok=True)
    config={'family':ARGS.family,'phase':ARGS.phase,'backgrounds':BG,'settings_per_background':EXPECTED[ARGS.family],'source_sha256':{str(p.relative_to(HERE)):sha(p) for p in (HERE/'original').rglob('*.py')},'BG_inputs':{k:v['sha256'] for k,v in bg7_inputs.manifest()['subsets'].items()},'input_manifest_sha256':sha('/workspace/shared/mia_bg7_inputs/manifest.json'),'optimization_validation_sha256':sha(HERE/'optimization_validation.json'),'runner_sha256':sha(__file__),'optimizer_sha256':sha(HERE/'optimized.py'),'protocol':'All original settings; refit all BG-derived preprocessing; canonical all20; original raw/county500/age100 and cross0/0.1 retained; scores frozen before truth; no method selection or grid pruning; main deferred'}
    p=out/'fixed_protocol.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(config))
    else:dump(p,config)
    for bg in BG:
        target=out/bg
        if (target/'prediction_record.json').exists():
            rec=json.loads((target/'prediction_record.json').read_text());assert sha(target/'scores.npz')==rec['scores_sha256'];continue
        infer(bg,target)
    if ARGS.phase=='prelim':evaluate(out)

if __name__=='__main__':main()
