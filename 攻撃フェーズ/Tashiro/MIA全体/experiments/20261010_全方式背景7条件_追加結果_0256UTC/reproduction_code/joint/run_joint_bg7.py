"""Seven-source-subset replication of pinned Oct 9 MIA score families.

All original score equations and hyperparameters are imported unchanged.
Only BG selection, BG-dependent refitting, orchestration, and reporting differ.
Preliminary labels are opened only after prediction hashes have been frozen.
Candidate matrices remain in a separate local cache, not the report tree.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, platform, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau

HERE=Path(__file__).resolve().parent
BG=('starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main')
SOURCE={}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def msha(v): return hashlib.sha256(np.ascontiguousarray(v,dtype='<f8').tobytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def dump(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mod(name):
    p=HERE/'original'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('bg7_'+name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    SOURCE[name]=sha(p)
    return m

base,neighbor,p02,clinical,gates,p08,sparse=[mod(n) for n in ('joint','neighbor','p02','clinical','gates','p08','sparse')]
small=mod('smallgap')
sparse_small=mod('sparse')
sparse_small.MARGINS=small.MARGINS;sparse_small.PENALTIES=small.PENALTIES
sparse_small.METHODS=('P01',)+tuple(f'sparse_top{k}_margin{m:g}_penalty{p:g}' for k in small.TOP_K for m in small.MARGINS for p in small.PENALTIES)

def load(root,phase):
    if phase=='prelim':
        common,calib,folder,paths,b,a,qs,cs,teams,private,rare,normal=base.load_inputs(root)
        return common,calib,folder,paths,b,a,qs,cs,teams,private,rare,normal
    import importlib
    src=root/'攻撃フェーズ/Tashiro/MIA全体/experiments/20261007'
    for f,h in base.SOURCE_HASHES.items(): assert sha(src/f)==h
    sys.path.insert(0,str(src))
    common=importlib.import_module('分析');calib=importlib.import_module('非会員を使う順位較正の検討')
    folder=root/'公式データ';own=list(folder.glob('PWSCup2026_final_team28_*/B_self.csv'));assert len(own)==1
    attack=folder/'PWSCup2026_final_attack'
    paths=own+[own[0].parent/'A_bg.csv']+sorted(attack.glob('C_*.csv'))+sorted(attack.glob('aia_challenge_*.csv'))
    read=lambda p:pd.read_csv(p,float_precision='round_trip')
    b,a=read(paths[0]),read(paths[1])
    qs={int(p.stem.rsplit('_',1)[1]):read(p) for p in attack.glob('aia_challenge_*.csv')}
    cs={int(p.stem.rsplit('_',1)[1]):read(p) for p in attack.glob('C_*.csv') if p.stem!='C_28'};teams=sorted(cs)
    private=pd.MultiIndex.from_frame(b[list(common.QI7)]).isin(pd.MultiIndex.from_frame(qs[28][list(common.QI7)]))
    rare=b.is_rare.to_numpy(bool);normal=~private & ~rare
    assert (len(teams),len(b),private.sum(),rare.sum(),normal.sum())==(22,1049,163,24,875)
    return common,calib,folder,paths,b,a,qs,cs,teams,private,rare,normal

def cache_scores(cache_root,bg,manifest,phase,common,calib,b,qs,teams,paths):
    p=cache_root/f'raw_cache_{bg}.npz';mp=p.with_suffix('.json')
    meta=json.loads(mp.read_text())
    assert meta['background_sha256']==manifest['subsets'][bg]['sha256']
    assert meta['code']['分析.py']==base.SOURCE_HASHES['分析.py']
    for f in paths:
        if str(f) in meta['phase_inputs']: assert meta['phase_inputs'][str(f)]==sha(f)
    with np.load(p) as z: allraw=z['raw']
    q=pd.concat([qs[t][list(common.QI7)] for t in sorted(qs)],ignore_index=True)
    source=np.concatenate([np.repeat(t,len(qs[t])) for t in sorted(qs)])
    assert allraw.shape==(len(b)+len(q),len(teams)) and np.isfinite(allraw).all()
    cal=[]
    for j,t in enumerate(teams):
        mask=source!=t
        cal.append(calib.calibrate(allraw[:len(b),j],b,allraw[len(b):,j][mask],q.loc[mask],['prefecture','sex'],500))
    return allraw[:len(b)].copy(),np.column_stack(cal),{'raw_cache_sha256':sha(p),'raw_cache_metadata_sha256':sha(mp),'background_sha256':meta['background_sha256']}

def add(bundle,family,inference,teams,methods):
    for name,mat in methods.items():
        assert mat.shape==(875,len(teams)) and np.isfinite(mat).all(),(family,name,mat.shape)
        bundle.append({'family':family,'inference':inference,'teams':list(teams),'method':name,'values':mat})

def clinical_main(common,b,aa,qs,cs,teams,prelim_thresholds):
    # The original public-main projection transfers preliminary cutoffs.
    # Refit those cutoffs on the matching BG subset, not the old phase BG.
    old=clinical.MAIN_TARGETS
    clinical.MAIN_TARGETS=tuple(teams[:18])
    dist,groups,_=clinical.build_public_features(common,b,aa,qs,cs,teams)
    clinical.MAIN_TARGETS=old
    groups['aia_imprint_high']=(groups.AIA_imprint_log_BG_to_C_median_Q_NN>=prelim_thresholds['imprint_median']).astype(int)
    groups['c_home_above_median']=(groups.C_home_fraction>=prelim_thresholds['c_home_median']).astype(int)
    return dist,groups

def predict_one(root,manifest,phase,bg,cache_root,out,private_cache,prelim_out=None):
    t0=time.perf_counter()
    common,calib,folder,paths,b,a,qs,cs,teams,private,rare,normal=load(root,phase)
    aa=pd.read_csv(manifest['subsets'][bg]['path'],float_precision='round_trip')
    assert len(aa)==manifest['subsets'][bg]['rows'] and sha(manifest['subsets'][bg]['path'])==manifest['subsets'][bg]['sha256']
    raw,cal,cache_meta=cache_scores(cache_root,bg,manifest,phase,common,calib,b,qs,teams,paths)
    private_cache.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(private_cache/f'{phase}_{bg}_rawcal.npz',raw=raw,cal=cal,teams=np.asarray(teams))
    subset_out=out/bg;subset_out.mkdir(parents=True,exist_ok=True)
    bundle=[];ops=[];optim=[]
    scopes={'all20':teams,'without1_2':[t for t in teams if t not in (1,2)],'without1_2_12':[t for t in teams if t not in (1,2,12)]} if phase=='prelim' else {'all22':teams}
    if phase=='prelim':
        for inf,ts in scopes.items():
            ix=[teams.index(t) for t in ts]
            methods,diags=base.predict(raw,cal,normal,private,ix)
            assert len(methods)==12
            add(bundle,'joint',inf,ts,methods)
            optim.extend({'family':'joint','inference':inf,'method':m,**v} for m,v in diags.items())
        ms,diags,info=p02.make_predictions(base,raw,cal,b,qs,teams,normal,private,rare)
        add(bundle,'p02','all20',teams,ms);optim.extend({'family':'p02','inference':'all20',**v} for v in diags)
        dump(subset_out/'p02_public_priors.json',info)
    else:
        r=base.ranks(cal[normal]);g=base.ranks(raw[normal]);median=base.rivals(r,np.median);maximum=base.rivals(r,np.max)
        ms={'GenLRA7':g,'Q500':r,'P01':r-.1*median}
        for w in base.CONFIG['max_penalties']:ms[f'max_{w:g}']=r-w*maximum
        ms['P01_plus_max_0.1']=ms['P01']-.1*maximum
        add(bundle,'joint_noncapacity_projection','all22',teams,ms)
    public=neighbor.public_neighbours(common,b,aa,cs,teams)
    for inf,ts in scopes.items():
        ix=[teams.index(t) for t in ts]
        add(bundle,'neighbor',inf,ts,neighbor.score_methods(base,cal,public,normal,ix))
    if phase=='prelim':
        dist,groups,thresholds=clinical.build_public_features(common,b,aa,qs,cs,teams)
    else:
        thresholds=json.loads((prelim_out/bg/'clinical_thresholds.json').read_text())
        dist,groups=clinical_main(common,b,aa,qs,cs,teams,thresholds)
    groups.to_csv(subset_out/'clinical_groups.csv',index=False,float_format='%.17g')
    dump(subset_out/'clinical_thresholds.json',thresholds)
    smallscopes={k:v for k,v in scopes.items() if k!='without1_2_12'}
    for inf,ts in smallscopes.items():
        ix=[teams.index(t) for t in ts]
        methods=clinical.score_variants(base,cal,normal,ix,ts,dist,groups)
        add(bundle,'clinical' if phase=='prelim' else 'clinical_threshold_projection',inf,ts,methods)
        # Post-hoc gate choices are retained unchanged and labeled diagnostic.
        gts=[t for t in ts if t not in (1,2)] if phase=='prelim' else ts
        flags=groups.set_index('team');refs=[t for t in teams if t not in (1,2)] if phase=='prelim' else teams
        gate_methods={}
        selections=[]
        for m in gates.CHOICES:
            cols=[]
            for t in gts:
                peers=flags.loc[[q for q in refs if q!=t],'AIA_imprint_log_BG_to_C_median_Q_NN'].to_numpy()
                selected=gates.choose(m,flags.loc[t],peers)
                cols.append(methods[selected][:,ts.index(t)])
                selections.append({'inference':inf,'team':t,'method':m,'source_method':selected})
            gate_methods[m]=np.column_stack(cols)
        add(bundle,'posthoc_gates_diagnostic',inf,gts,gate_methods)
        pd.DataFrame(selections).to_csv(subset_out/f'gate_selections_{inf}.csv',index=False)
    # BG source controls every BG-derived operation, including exact-copy masks.
    variants,copy_records=p08.clean_public_C(cs,aa,teams)
    copy_records.to_csv(subset_out/'p08_public_copy_cohorts.csv',index=False)
    copy_cal={};touched={}
    for kind in variants:
        copy_cal[kind],touched[kind]=p08.recalibrate_changed(common,calib,b,aa,qs,cs,teams,cal,variants[kind])
    events={kind:np.column_stack([p08.time_conditional_signal(b,aa,cs[t],kind) for t in teams]) for kind in p08.TIME_CHANNELS}
    for inf,ts in smallscopes.items():
        add(bundle,'p08',inf,ts,p08.create_scores(base,cal,copy_cal,events,normal,ts,teams))
        ix=[teams.index(t) for t in ts]
        for family,module in [('sparse',sparse),('smallgap',sparse_small)]:
            ms,ds=module.panel(base,cal,normal,ix)
            add(bundle,family,inf,ts,ms)
            ops.extend({'family':family,'inference':inf,**v} for v in ds)
    hashes=[{k:v for k,v in item.items() if k!='values'}|{'sha256':msha(item['values']),'shape':list(item['values'].shape),'diagnostic_only':item['family']=='posthoc_gates_diagnostic' or item['method']=='hard_binary_diagnostic'} for item in bundle]
    dump(subset_out/'prediction_hashes.json',hashes)
    private_cache.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(private_cache/f'{phase}_{bg}.npz',**{str(i):item['values'] for i,item in enumerate(bundle)})
    dump(subset_out/'optimizer_diagnostics.json',optim)
    pd.DataFrame(ops).to_csv(subset_out/'collision_diagnostics.csv',index=False,float_format='%.17g')
    dump(subset_out/'prediction_record.json',{'phase':phase,'bg':bg,'bg_rows':len(aa),'frozen_utc':now(),'seconds':time.perf_counter()-t0,'conditions':len(bundle),'cache':cache_meta,'P08_changed':touched,'labels_read':False})
    print('FROZEN',phase,bg,len(bundle),'conditions',round(time.perf_counter()-t0,2),'seconds',flush=True)
    return bundle

def read_bundle(out,private_cache,phase,bg):
    records=json.loads((out/bg/'prediction_hashes.json').read_text())
    z=np.load(private_cache/f'{phase}_{bg}.npz')
    for i,r in enumerate(records):
        r['values']=z[str(i)];assert msha(r['values'])==r['sha256']
    return records

def evaluate(root,bundles,out):
    common,calib,folder,paths,b,a,qs,cs,teams,private,rare,normal=load(root,'prelim')
    y,audit,truth_paths=base.audit_truth(folder,b,private,rare,teams)
    dump(out/'truth_audit.json',audit);dump(out/'truth_sha256.json',{str(p):sha(p) for p in truth_paths})
    records=[]
    for bg,bundle in bundles.items():
        for item in bundle:
            conf=base.confidence(item['values'],normal,private,rare)
            assert (conf[private]==0).all() and (conf[rare&~private]==.4).all()
            for j,t in enumerate(item['teams']):
                for pop,mask in [('all_fixed',np.ones(len(normal),bool)),('normal',normal)]:
                    yy=y[mask,teams.index(t)]
                    records.append({'background':bg,'family':item['family'],'inference':item['inference'],'method':item['method'],'team':t,'population':pop,'n':int(mask.sum()),'members':int(yy.sum()),'diagnostic_only':item['family']=='posthoc_gates_diagnostic' or item['method']=='hard_binary_diagnostic',**base.metrics(conf[mask,j],yy)})
    by=pd.DataFrame(records);by.to_csv(out/'by_target.csv',index=False,float_format='%.17g')
    sums=[]
    for key,panel in by.groupby(['background','family','inference','population'],sort=False):
        bg,fam,inf,pop=key
        ref=panel[panel.method=='P01'].set_index('team')['tpr_0.01']
        for ev,drop in [('all',[]),('without1_2',[1,2]),('without12',[12]),('without1_2_12',[1,2,12])]:
            for method,g in panel[~panel.team.isin(drop)].groupby('method',sort=False):
                val=g.set_index('team')['tpr_0.01'];delta=val-ref.loc[val.index]
                sums.append({'background':bg,'family':fam,'inference':inf,'population':pop,'evaluation':ev,'method':method,'targets':len(g),'mean_tpr':float(val.mean()),'mean_delta_P01':float(delta.mean()),'wins':int((delta>1e-12).sum()),'losses':int((delta< -1e-12).sum()),'ties':int((abs(delta)<=1e-12).sum()),'TPR_ge010':int((val>=.1-1e-12).sum()),'mean_auc':float(g.auc.mean()),'diagnostic_only':bool(g.diagnostic_only.any())})
    summary=pd.DataFrame(sums);summary.to_csv(out/'summary.csv',index=False,float_format='%.17g')
    return by,summary

def compare_original(summary,snapshot,out):
    mapping={'joint':('全対象共同推定','results/summary.csv','mean_tpr'),'neighbor':('近傍競合補正追加検証','results/summary.csv','mean_tpr'),'p02':('P02_AIA二対象外共同推定','results/summary.csv','mean_tpr'),'clinical':('AIA臨床分布と加工特徴の群別補正','results/summary.csv','mean_tpr'),'p08':('P08_公開残差と時刻条件付','results/summary.csv','mean_tpr_001'),'sparse':('P08D_上位競合だけの疎な排他補正','results/summary.csv','mean_tpr'),'smallgap':('P08D_上位競合だけの疎な排他補正','small_gap/summary.csv','mean_tpr')}
    checks=[]
    for family,(name,rel,value) in mapping.items():
        p=snapshot/'攻撃フェーズ/Tashiro/MIA全体/experiments'/('20261009_'+name)/rel
        old=pd.read_csv(p,float_precision='round_trip')
        old=old[old.background=='full']
        for _,row in old.iterrows():
            inf=row.get('inference','all20');ev=row.get('evaluation','all')
            if ev=='all20':ev='all'
            found=summary[(summary.background=='prelim')&(summary.family==family)&(summary.population=='all_fixed')&(summary.inference==inf)&(summary.evaluation==ev)&(summary.method==row.method)]
            assert len(found)==1,(family,inf,ev,row.method,list(old.columns))
            error=abs(float(found.mean_tpr.iloc[0])-float(row[value]))
            checks.append({'family':family,'inference':inf,'evaluation':ev,'method':row.method,'expected':float(row[value]),'actual':float(found.mean_tpr.iloc[0]),'absolute_error':error,'pass':error<1e-12,'historical_source_sha256':sha(p)})
    dump(out/'historical_regression.json',checks)
    assert all(c['pass'] for c in checks),[c for c in checks if not c['pass']]
    print('HISTORICAL REGRESSION',len(checks),'comparisons, max error',max(c['absolute_error'] for c in checks),flush=True)

def main_rank(bundles,out):
    lookup={(x['family'],x['inference'],x['method']):x for x in bundles['main']}
    rows=[]
    for bg,bundle in bundles.items():
        for item in bundle:
            ref=lookup[(item['family'],item['inference'],item['method'])]
            assert item['teams']==ref['teams']
            for j,t in enumerate(item['teams']):
                s,r=item['values'][:,j],ref['values'][:,j]
                a=np.argsort(-s,kind='stable')[:20];b=np.argsort(-r,kind='stable')[:20]
                rows.append({'background':bg,'reference_background':'main','family':item['family'],'inference':item['inference'],'method':item['method'],'team':t,'spearman':float(spearmanr(s,r).statistic),'kendall_tau_b':float(kendalltau(s,r).statistic),'top20_overlap':len(set(a)&set(b)),'mean_absolute_rank_shift':float(abs(rankdata(s)-rankdata(r)).mean()),'diagnostic_only':True})
    by=pd.DataFrame(rows);by.to_csv(out/'rank_by_target.csv',index=False,float_format='%.17g')
    by.groupby(['background','family','inference','method'],sort=False).agg(targets=('team','size'),mean_spearman=('spearman','mean'),mean_kendall_tau_b=('kendall_tau_b','mean'),mean_top20_overlap=('top20_overlap','mean'),mean_absolute_rank_shift=('mean_absolute_rank_shift','mean')).reset_index().to_csv(out/'rank_summary.csv',index=False,float_format='%.17g')

def run(args):
    start=time.perf_counter();manifest=json.loads(args.manifest.read_text());root=Path(manifest['repository_root'])
    out=HERE/args.phase;out.mkdir(parents=True,exist_ok=True)
    pc=args.private_cache;cache=args.available/f'results_{args.phase}'
    dump(out/'protocol.json',{'source_code_sha256':SOURCE,'driver_sha256':sha(__file__),'manifest_sha256':sha(args.manifest),'phase':args.phase,'backgrounds':BG,'primary_population':'all_fixed1049; normal875 secondary','parameter_tuning':False,'all_BG_dependent_objects_refit':True,'P08_copy_mask_reference':'selected BG subset; original preliminary BG exactly reproduces old mask','clinical_reference':'each BG independently refits scaling, 1049 reference sample and public group thresholds','posthoc_gates':'diagnostic only; selected after historical truth exploration','preliminary_generalization':'not independent validation; repeatedly explored cohort','main_transport_and_p02':'not run; preliminary capacity and home assumptions not transferred','main_other_scores':'mechanical label-free all22 projection; no measured main TPR','no_submission':True,'candidate_scores_in_report_tree':False})
    order=('prelim',)+tuple(g for g in BG if g!='prelim') if args.phase=='prelim' else BG
    bundles={}
    for bg in order:
        existing=out/bg/'prediction_record.json'
        if existing.exists():bundles[bg]=read_bundle(out,pc,args.phase,bg)
        else:bundles[bg]=predict_one(root,manifest,args.phase,bg,cache,out,pc,HERE/'prelim')
        if args.phase=='prelim' and bg=='prelim':
            baseline=out/'baseline_verification';baseline.mkdir(exist_ok=True)
            _,ss=evaluate(root,{'prelim':bundles[bg]},baseline)
            compare_original(ss,args.snapshot,baseline)
            if args.baseline_only:return
    dump(out/'all_predictions_frozen.json',{'frozen_utc':now(),'conditions':sum(map(len,bundles.values())),'subset_count':len(bundles),'prediction_hashes_sha256':{bg:sha(out/bg/'prediction_hashes.json') for bg in BG}})
    if args.phase=='prelim':evaluate(root,bundles,out)
    else:main_rank(bundles,out)
    dump(out/'run_record.json',{'completed_utc':now(),'seconds':time.perf_counter()-start,'phase':args.phase,'subsets':len(bundles),'prediction_conditions':sum(map(len,bundles.values())),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'labels':'preliminary only after freeze' if args.phase=='prelim' else 'none','main_submission':False})
    print('COMPLETE',args.phase,round(time.perf_counter()-start,2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('/workspace/shared/mia_bg7_inputs/manifest.json'));p.add_argument('--snapshot',required=True,type=Path);p.add_argument('--available',required=True,type=Path);p.add_argument('--private-cache',type=Path,default=Path('/workspace/shared/mia_joint_cache'));p.add_argument('--phase',choices=('prelim','main'),required=True);p.add_argument('--baseline-only',action='store_true');run(p.parse_args())
