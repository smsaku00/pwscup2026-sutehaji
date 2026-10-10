"""Pinned Oct 9 KDE/P03 fusion equations over all seven BG source subsets.

Original hyperparameters and within-method reference sampling stay fixed.
All BG-dependent fits are recalculated for each selected BG. Preliminary
evaluation happens only after complete prediction freeze. No main labels.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, platform, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau

HERE = Path(__file__).resolve().parent
CACHE = Path('/workspace/shared/mia_bg7_inputs')
INFRA = Path('/workspace/shared/mia_bg7_stage/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_BG7横断比較')
AVAILABLE = Path('/workspace/shared/mia_all_bg_available/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/available')
JOINT = Path('/workspace/shared/mia_all_bg_joint/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/joint')
PRIVATE = Path('/workspace/shared/mia_fusion_cache')
ROOT = CACHE / 'root'
BG = ('starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def msha(x): return hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
def dump(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


inputs = module('fusion_inputs',INFRA/'bg7_inputs.py')
archive,fresh,p03,clinical = [module('fusion_'+k,HERE/'original'/f'{k}.py') for k in ('archive','fresh','p03','clinical')]
base = archive.import_base(ROOT)
sys.path.insert(0,str(ROOT/'攻撃フェーズ/Tashiro/MIA全体/experiments/20261007'))
import 分析 as common
import 非会員を使う順位較正の検討 as calib


def cached_cal(phase,bg,data,manifest):
    p=AVAILABLE/f'results_{phase}'/f'raw_cache_{bg}.npz'
    sidecar=json.loads(p.with_suffix('.json').read_text())
    assert sidecar['background_sha256']==manifest['subsets'][bg]['sha256']
    assert sidecar['code']['分析.py']==base.SOURCE_HASHES['分析.py']
    assert sidecar['code']['非会員を使う順位較正の検討.py']==base.SOURCE_HASHES['非会員を使う順位較正の検討.py']
    for path,expected in sidecar['phase_inputs'].items(): assert sha(path)==expected
    b,qs,teams=data['b'],data['q'],data['targets']
    raw=np.load(p)['raw']
    q=pd.concat([qs[t][list(common.QI7)] for t in sorted(qs)],ignore_index=True)
    source=np.concatenate([np.repeat(t,len(qs[t])) for t in sorted(qs)])
    assert raw.shape==(len(b)+len(q),len(teams)) and np.isfinite(raw).all()
    result=np.column_stack([calib.calibrate(raw[:len(b),j],b,raw[len(b):,j][source!=t],q.loc[source!=t],['prefecture','sex'],500) for j,t in enumerate(teams)])
    return result,{'cache_path':str(p),'cache_sha256':sha(p),'metadata_sha256':sha(p.with_suffix('.json'))}


def group_for(phase,bg,data,a,out):
    existing=JOINT/phase/bg/'clinical_groups.csv'
    thresholds=JOINT/phase/bg/'clinical_thresholds.json'
    if existing.exists() and thresholds.exists():
        groups=pd.read_csv(existing,float_precision='round_trip');cuts=json.loads(thresholds.read_text())
        provenance={'source':'matching joint BG-refit public features','source_path':str(existing),'sha256':sha(existing)}
    elif phase=='prelim':
        _,groups,cuts=clinical.build_public_features(common,data['b'],a,data['q'],data['c'],data['targets'])
        provenance={'source':'original public clinical feature computation on selected BG'}
    else:
        cuts=json.loads((HERE/'prelim'/bg/'clinical_thresholds.json').read_text())
        old=clinical.MAIN_TARGETS
        try:
            clinical.MAIN_TARGETS=tuple(data['targets'][:18])
            _,groups,_=clinical.build_public_features(common,data['b'],a,data['q'],data['c'],data['targets'])
        finally:clinical.MAIN_TARGETS=old
        groups['aia_imprint_high']=(groups.AIA_imprint_log_BG_to_C_median_Q_NN>=cuts['imprint_median']).astype(int)
        groups['c_home_above_median']=(groups.C_home_fraction>=cuts['c_home_median']).astype(int)
        provenance={'source':'original public-main transfer with same-BG preliminary cutoffs, no main labels'}
    assert set(groups.team)==set(data['targets'])
    groups.to_csv(out/'clinical_groups.csv',index=False,float_format='%.17g')
    dump(out/'clinical_thresholds.json',cuts)
    high=set(groups.loc[groups.aia_imprint_high==1,'team'].astype(int))
    if phase=='prelim': high-=set((1,2))
    return high,provenance


def build_variants(cal,kde,signals,one,two,normal,teams,ids,high):
    idx=[teams.index(t) for t in ids]
    r=base.ranks(cal[normal][:,idx]);p01=r-.1*base.rivals(r,np.median)
    gate=np.array([int(t in high) for t in ids],float)[None,:]
    k=archive.ranks(kde[:,idx],normal);ka=k-.1*base.rivals(k,np.median)
    archived={('P01' if w==0 else f'originalKDEA_mix{w:g}'):np.where(gate>0,(1-w)*p01+w*ka,p01) for w in archive.WEIGHTS}
    new={'P01':p01}
    for align in fresh.ALIGN:
        for h in fresh.BANDWIDTH:
            mat=np.column_stack([signals[(t,align,h)] for t in ids]);score=base.ranks(mat[normal])
            for w in fresh.WEIGHTS:new[f'kde_{align}_h{h:g}_mix{w:g}']=p01+w*score*gate
    n1=base.ranks(-one[normal][:,idx]);n2=base.ranks(-two[normal][:,idx]);other=base.rivals(n1,np.median);p01rank=base.ranks(p01)
    weights={'P01':p01}
    for w in p03.WEIGHTS:weights[f'P03_nn1_w{w:g}']=p01+gate*w*n1
    for name,w in p03.OTHER_VARIANTS:
        feature={'rel_med':n1-other,'two':n2,'top20':n1*(p01rank>=.80)}[name]
        weights[f'P03_{name}_w{w:g}']=p01+gate*w*feature
    return {'archived_kdeA_mix_full':archived,'new_kde_fusion':new,'p03_threshold010':weights}


def predict(phase,bg,manifest,out):
    start=time.perf_counter();out.mkdir(parents=True,exist_ok=True)
    d=inputs.load_phase(phase);a=inputs.load_background(bg);teams=d['targets'];b=d['b'];normal=d['normal']
    cal,rawmeta=cached_cal(phase,bg,d,manifest)
    high,groupmeta=group_for(phase,bg,d,a,out)
    kde,h=archive.scores_KDEA(common,b,a,d['c'],teams)
    signals,kde_params=fresh.public_kde(common,b,a,d['c'],teams)
    one,two=p03.nearest_features(common,b,a,d['c'],teams)
    scopes={'all20':teams,'legacy18':[t for t in teams if t not in (1,2)]} if phase=='prelim' else {'all22':teams}
    records=[];arrays={}
    for scope,ids in scopes.items():
        families=build_variants(cal,kde,signals,one,two,normal,teams,ids,high)
        # Equation-adapter regression to unmodified historical predictors.
        if phase=='prelim' and bg=='prelim' and scope=='legacy18':
            idx=[teams.index(t) for t in ids]
            direct=fresh.predictors(base,cal,normal,idx,ids,high,signals)
            direct_p03=p03.candidate_scores(base,cal,normal,ids,idx,one,two,np.array([t in high for t in teams]))
            for name,x in direct.items():assert np.array_equal(x,families['new_kde_fusion'][name])
            for name,x in direct_p03.items():assert np.array_equal(x,families['p03_threshold010'][name])
            for _,ts,methods in archive.scores_by_scope(base,cal,kde,normal,teams,high):
                own=build_variants(cal,kde,signals,one,two,normal,teams,ts,high)['archived_kdeA_mix_full']
                for name,x in methods.items():assert np.array_equal(x,own[name])
        for family,methods in families.items():
            for name,scores in methods.items():
                assert scores.shape==(875,len(ids)) and np.isfinite(scores).all()
                key=str(len(records));arrays[key]=scores
                records.append({'key':key,'phase':phase,'background':bg,'family':family,'inference':scope,'method':name,'teams':ids,'shape':list(scores.shape),'sha256':msha(scores)})
    PRIVATE.mkdir(exist_ok=True)
    artifact=PRIVATE/f'{phase}_{bg}.npz';np.savez_compressed(artifact,**arrays)
    dump(out/'prediction_hashes.json',records)
    dump(out/'prediction_record.json',{'phase':phase,'background':bg,'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'membership_truth_read':False,'condition_count':len(records),'bg_n':len(a),'archived_h':h,'new_kde_parameters':kde_params,'high_group':sorted(high),'raw_cache':rawmeta,'group_provenance':groupmeta,'artifact':str(artifact),'artifact_sha256':sha(artifact),'seconds':time.perf_counter()-start})
    print('FROZEN',phase,bg,len(records),'conditions',round(time.perf_counter()-start,2),flush=True)
    return records


def bundle(phase,bg):
    records=json.loads((HERE/phase/bg/'prediction_hashes.json').read_text());meta=json.loads((HERE/phase/bg/'prediction_record.json').read_text())
    assert sha(meta['artifact'])==meta['artifact_sha256'];arrays=np.load(meta['artifact'])
    for item in records:
        item['values']=arrays[item['key']];assert msha(item['values'])==item['sha256']
    return records


def baseline_hash_check():
    actual=bundle('prelim','prelim');checks=[]
    for family in ('archived_kdeA_mix_full','new_kde_fusion','p03_threshold010'):
        gold=json.loads((HERE/'historical'/family/'prediction_hashes.json').read_text())
        for item in gold:
            if item.get('background',item.get('view'))!='full':continue
            scope={'infer18':'legacy18','infer20':'all20'}.get(item.get('scope'),'legacy18')
            found=next(x for x in actual if x['family']==family and x['inference']==scope and x['method']==item['method'])
            checks.append({'family':family,'inference':scope,'method':item['method'],'expected_sha256':item['sha256'],'actual_sha256':found['sha256'],'exact_match':item['sha256']==found['sha256']})
    dump(HERE/'historical_prediction_regression.json',checks)
    assert all(c['exact_match'] for c in checks),[c for c in checks if not c['exact_match']]
    print('BASELINE',len(checks),'exact historical score hashes reproduced',flush=True)


def evaluate_prelim():
    data=inputs.load_phase('prelim');pred=[]
    for bg in BG:pred.extend(bundle('prelim',bg))
    # Assert the complete seven-subset manifest before opening membership truth.
    assert len(pred)==7*60
    artifact_paths=[PRIVATE/f'prelim_{bg}.npz' for bg in BG]
    seal=HERE/'prelim_prediction_freeze.json'
    inputs.freeze_prediction_files(artifact_paths,seal,HERE/'fixed_protocol.json')
    truth,teams=inputs.load_prelim_truth_for_scoring(seal)
    rows=[]
    for item in pred:
        confidence=base.confidence(item['values'],data['normal'],data['private'],data['rare'])
        for j,t in enumerate(item['teams']):
            for pop,mask in [('all_fixed',np.ones(len(data['b']),bool)),('nonrare',data['nonrare']),('normal',data['normal'])]:
                yy=truth[mask,teams.index(t)]
                rows.append({k:item[k] for k in ('phase','background','family','inference','method')}|{'team':t,'population':pop,'n':int(mask.sum()),'members':int(yy.sum()),**base.metrics(confidence[mask,j],yy)})
    df=pd.DataFrame(rows);df.to_csv(HERE/'prelim_by_target.csv',index=False,float_format='%.17g')
    summary=[]
    for key,g in df.groupby(['background','family','inference','population','method'],sort=False):
        bg,fam,inf,pop,method=key
        for view,exclude in [('all',[]),('without1_2',[1,2]),('without1_2_12',[1,2,12])]:
            sub=g[~g.team.isin(exclude)];ref=df[(df.background==bg)&(df.family==fam)&(df.inference==inf)&(df.population==pop)&(df.method=='P01')].set_index('team')['tpr_0.01']
            vals=sub.set_index('team')['tpr_0.01'];delta=vals-ref.loc[vals.index]
            summary.append({'background':bg,'family':fam,'inference':inf,'population':pop,'method':method,'evaluation':view,'targets':len(sub),'mean_tpr':float(vals.mean()),'delta_P01':float(delta.mean()),'wins':int((delta>1e-12).sum()),'losses':int((delta< -1e-12).sum()),'TPR_ge010':int((vals>=.1-1e-12).sum())})
    pd.DataFrame(summary).to_csv(HERE/'prelim_summary.csv',index=False,float_format='%.17g')
    checks=[]
    for family in ('archived_kdeA_mix_full','new_kde_fusion','p03_threshold010'):
        old=pd.read_csv(HERE/'historical'/family/'by_target.csv',float_precision='round_trip');old=old[old.background=='full']
        for row in old.to_dict('records'):
            inf={'infer18':'legacy18','infer20':'all20'}.get(row.get('inference'),'legacy18')
            found=df[(df.background=='prelim')&(df.family==family)&(df.inference==inf)&(df.population=='all_fixed')&(df.method==row['method'])&(df.team==row['team'])]
            assert len(found)==1
            err=abs(found['tpr_0.01'].iloc[0]-row['tpr_0.01']);checks.append({'family':family,'inference':inf,'method':row['method'],'team':row['team'],'absolute_error':err,'passed':bool(err<1e-12)})
    dump(HERE/'historical_metric_regression.json',checks);assert all(c['passed'] for c in checks)
    print('PRELIM METRICS',len(rows),'rows, historical checks',len(checks),flush=True)


def main_ranks():
    data=inputs.load_phase('main');reference={(x['family'],x['method']):x['values'] for x in bundle('main','main')};rows=[]
    for bg in BG:
        for item in bundle('main',bg):
            ref=reference[(item['family'],item['method'])]
            for j,t in enumerate(item['teams']):
                s,r=item['values'][:,j],ref[:,j]
                row={k:item[k] for k in ('background','family','inference','method')}|{'team':t,'population':'normal','n':875,'reference_background':'main','spearman':float(spearmanr(s,r).statistic),'kendall_tau_b':float(kendalltau(s,r).statistic),'mean_absolute_rank_shift':float(np.mean(np.abs(rankdata(s)-rankdata(r)))),'unique_scores':int(np.unique(s).size),'finite':bool(np.isfinite(s).all())}
                for k in (10,12,20,25):
                    ia=np.argsort(-s,kind='stable')[:k];ib=np.argsort(-r,kind='stable')[:k];cut=np.sort(s)[-k]
                    row[f'top{k}_overlap']=len(set(ia)&set(ib));row[f'top{k}_boundary_tie_size']=int((s==cut).sum())
                rows.append(row)
    df=pd.DataFrame(rows);df.to_csv(HERE/'main_rank_qa.csv',index=False,float_format='%.17g')
    df.groupby(['background','family','method'],sort=False).agg(targets=('team','size'),mean_spearman=('spearman','mean'),minimum_spearman=('spearman','min'),mean_top20_overlap=('top20_overlap','mean')).reset_index().to_csv(HERE/'main_rank_summary.csv',index=False,float_format='%.17g')
    print('MAIN RANK QA',len(rows),'rows; no labels or TPR used',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline-only',action='store_true');args=parser.parse_args()
    manifest=inputs.manifest()
    config={'sources':BG,'families':{'archived_kdeA_mix_full':5,'new_kde_fusion':13,'p03_threshold010':12},'primary_inference':{'prelim':'all20','main':'all22'},'legacy_regression_inference':'legacy18; archived family also all20',
            'BG_fit':'recompute all BG-derived scale/median/IQR/density/sample/gate/nearest-distance quantities from selected BG; preserve original hyperparameters',
            'historical_sampling':'new KDE uses original min(2048,n) deterministic sample seed20261009; public grouping uses original1049 sample. Pools themselves retain every source row.',
            'main_gates':'transfer same-BG public preliminary group thresholds via existing projection design; no main-trained membership model',
            'calibration':'exact original Q500 using all non-target public Q; original cached mixed GenLRA7 raw scores',
            'populations':{'all_fixed':1049,'nonrare':1025,'normal_unknown_nonrare':875},'source_hashes':json.loads((HERE/'original/source_paths.json').read_text()),'input_manifest_sha256':sha(CACHE/'manifest.json'),
            'common_protocol_sha256':sha(INFRA/'fixed_protocol.json'),'archive_weights':archive.WEIGHTS,'fresh_alignment':fresh.ALIGN,'fresh_h':fresh.BANDWIDTH,'fresh_weights':fresh.WEIGHTS,'P03_weights':p03.WEIGHTS,'P03_other':p03.OTHER_VARIANTS,
            'truth_policy':'exact-score baseline regression uses historical frozen hashes without current truth; all7 preliminary predictions frozen before current scoring; no main membership truth',
            'posthoc_warning':'method families and public group hypothesis historically selected after exploring preliminary labels; not independent validation','main_output':'ranking/score QA only; no attack submission',
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}}
    protocol=HERE/'fixed_protocol.json'
    if protocol.exists():assert json.loads(protocol.read_text())==json.loads(json.dumps(config))
    else:dump(protocol,config)
    predict('prelim','prelim',manifest,HERE/'prelim/prelim')
    baseline_hash_check()
    if args.baseline_only:return
    for bg in BG:
        if bg!='prelim':predict('prelim',bg,manifest,HERE/'prelim'/bg)
    for bg in BG:predict('main',bg,manifest,HERE/'main'/bg)
    evaluate_prelim();main_ranks()
    dump(HERE/'completion.json',{'complete':True,'preliminary_conditions':420,'main_conditions':210,'background_subsets':7,'all20_prelim':True,'all22_main':True,'main_truth_read':False,'source_inputs_modified':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
