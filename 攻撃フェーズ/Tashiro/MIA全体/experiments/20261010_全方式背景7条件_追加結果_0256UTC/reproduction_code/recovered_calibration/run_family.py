"""Execute pinned recovered kernels with canonical BG/data and private caches.

Score recipes, fit parameters and fold code are not rewritten here. The bridge
changes input paths, the external BG condition, and persistence/evaluation only.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time,shutil,platform,ast,resource
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve
import sklearn,scipy
import runtime as r
from execution_protocol import PROTOCOL

STAGE=Path('/workspace/shared/mia_all_bg_recovered_calibration/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_calibration')
WORK=Path('/workspace/shared/mia_recovered_calibration/work')
SCRIPTS={'direct':'直接密度比の探索.py','classifiers':'公開データ分類器の探索.py','attribute_seeds':'密度比の属性と乱数を確認.py','stable_centers':'中心配置を安定させた密度比.py','learned_base':'別系統特徴の学習.py','learned_split':'学習の分割感度.py','learned_ablation':'特徴を外して学習を確認.py','learned_lowfpr':'低FPRを重視する学習.py','smooth':'他対象との競合を滑らかに補正.py','local_ratio':'局所支持区間の密度比.py'}
FOLDERS={'direct':'直接密度比結果','classifiers':'公開データ分類器結果','attribute_seeds':'密度比_属性乱数結果','stable_centers':'中心配置の安定化結果','learned_base':'別系統特徴の学習結果','learned_split':'学習の分割感度結果','learned_ablation':'特徴除去の学習結果','learned_lowfpr':'低FPR学習結果','smooth':'滑らかな競合補正結果','local_ratio':'局所支持区間の密度比結果'}
SCRIPTS.update(learned_centers='学習でも密度比の乱数を確認.py',learned_yesterday='昨日の学習を18対象で確認.py')
FOLDERS.update(learned_centers='学習の密度比乱数結果',learned_yesterday='昨日の学習_18対象結果')
SCRIPTS.update(learned_bg_ablation='背景キャッシュで特徴除去を確認.py',learned_bg_transfer='背景変更で学習を確認.py',learned_multiview='複数背景と乱数で学習.py',learned_multiview_loo='複数ビューの対象保留確認.py',learned_main_transfer='本戦で順位を確認.py')
FOLDERS.update(learned_bg_ablation='特徴除去の背景結果',learned_bg_transfer='背景変更の学習結果',learned_multiview='複数ビューの学習結果',learned_multiview_loo='複数ビューの対象保留結果',learned_main_transfer='本戦の順位診断結果')

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def truth(data):
    p=r.INPUT/'root/公式データ/予備戦/結果/正解データ'
    own=pd.read_csv(p/'pool_ids_5.csv').set_index('record_id').loc[data['b'].record_id].pool_id
    return np.column_stack([own.isin(pd.read_csv(p/f'pool_ids_{t}.csv').pool_id).to_numpy(int) for t in data['teams']])

def evaluate(u,data,models):
    y=truth(data);rows=[]
    for name,mat in models.items():
        for j,t in enumerate(data['teams']):
            conf=u.base.confidence(mat[:,j],data['normal'],data['rare'],data['private'])
            fp,tp,_=roc_curve(y[:,j],conf,drop_intermediate=False)
            for point in u.FPRS:rows.append({'method':name,'team':t,'fpr':point,'mia_all':float(np.interp(point,fp,tp)),'members':int(y[:,j].sum())})
    return pd.DataFrame(rows)

def summary(table):
    rows=[]
    for (name,fpr),g in table.groupby(['method','fpr']):
        rows.append({'method':name,'fpr':fpr,'all20':float(g.mia_all.mean()),'other18':float(g[~g.team.isin([1,2])].mia_all.mean()),'tpr_ge010':int((g.mia_all>=.1-1e-12).sum())})
    return pd.DataFrame(rows)

def check_catalog(table):
    c=pd.read_csv(r.RECOVERY/'catalog/方式条件一覧.csv');checks=[]
    available=set(table.method)
    for _,row in c[c.condition=='full'].iterrows():
        aliases=set(str(row.source_method_aliases).split(';'))|{row.method}
        for method in available&aliases:
            ids=list(map(int,row.evaluated_teams.split(',')))
            q=table[(table.method==method)&(table.fpr==.01)].set_index('team')
            if not set(ids).issubset(q.index):continue
            actual=float(q.loc[ids].mia_all.mean());err=abs(actual-row.mean_tpr)
            checks.append({'condition_id':row.condition_id,'source_ids':row.source_ids,'method':method,'targets':ids,'historical_mean':row.mean_tpr,'measured_mean':actual,'absolute_error':err,'within_1e12':bool(err<1e-12)})
    return checks

def configure(u,args):
    if args.family.startswith('learned') and args.family!='learned_yesterday':
        from deterministic_features import ensure
        ensure(u,args.bg)
    work=WORK/args.bg
    if args.family.startswith('learned'):work=work/'learned_runtime'
    work.mkdir(parents=True,exist_ok=True)
    for p in r.SRC.iterdir():
        if p.is_file():
            dest=work/p.name
            if not dest.exists():dest.symlink_to(p)
    representative=Path('/workspace/shared/mia_recovered_calibration/cache/representative')/args.bg/'代表特徴.npz'
    if representative.exists():
        dest=work/'代表特徴の再現結果/代表特徴.npz';dest.parent.mkdir(exist_ok=True)
        if not dest.exists():dest.symlink_to(representative)
        if args.family in ('learned_base','learned_split'):
            defs=r.import_path('representative_signal_defs',r.SRC/'別系統特徴の学習.py')
            z=np.load(representative);groups={}
            for folder,name in defs.SIGNALS+defs.GEN:groups.setdefault(folder,[]).append(name)
            provenance={}
            for folder,names in groups.items():
                source=WORK/args.bg/folder/'内部順位キャッシュ.npz'
                if folder=='距離と分布結果':source=Path('/workspace/shared/mia_recovered_calibration/cache/source_features')/args.bg/'geometry.npz'
                if folder=='公開データ分類器結果':
                    seal=source.with_name('実行記録.json')
                    while not (source.exists() and seal.exists() and json.loads(seal.read_text()).get('forest_prediction_execution')==PROTOCOL):
                        print('WAIT_FOR_SERIAL_CLASSIFIER_CACHE',args.bg,flush=True);time.sleep(15)
                assert source.exists(),('Original cached-feature dependency missing',source)
                original=np.load(source);onames=list(original['names']);oteams=list(original['teams'])
                ix=[oteams.index(t) for t in z['teams']]
                assert np.array_equal(original['record_ids'],z['record_ids'])
                bank=original['scores'];models={name:bank[onames.index(name)][:,ix] for name in names}
                p=work/folder/'内部順位キャッシュ.npz';p.parent.mkdir(exist_ok=True)
                names=sorted(models)
                np.savez_compressed(p,names=names,scores=np.stack([models[n] for n in names]),teams=z['teams'],record_ids=z['record_ids'])
                provenance[folder]={'source_cache':str(source),'source_sha256':r.sha(source),'selected_names':names,'selected_matrix_hashes':{name:r.matsha(models[name]) for name in names}}
            dump(work/'exact_feature_sources.json',provenance)
    if args.family=='learned_centers':
        source=WORK/args.bg/'密度比_属性乱数結果/内部順位キャッシュ.npz'
        assert source.exists(),source
        target=work/'密度比_属性乱数結果/内部順位キャッシュ.npz';target.parent.mkdir(exist_ok=True)
        if not target.exists():target.symlink_to(source)
    u.HERE=work
    u.setup=lambda phase='prelim':r.data_for(u,args.bg,phase)
    def save(out,data,models,specs,started,extra=None):
        out=Path(out)
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
        dest=STAGE/args.family/args.bg;dest.mkdir(parents=True,exist_ok=True)
        hashes=[]
        for name,v in models.items():
            assert v.shape==(875,len(data['teams'])) and np.isfinite(v).all(),(name,v.shape)
            hashes.append({'method':name,'shape':list(v.shape),'sha256':r.matsha(v)})
        dump(dest/'prediction_hashes.json',hashes)
        names=sorted(models)
        np.savez_compressed(out/'内部順位キャッシュ.npz',names=np.asarray(names),scores=np.stack([models[n] for n in names]),teams=np.asarray(data['teams']),record_ids=data['b'].loc[data['normal'],'record_id'].to_numpy())
        pd.DataFrame(specs).to_csv(dest/'方式一覧.csv',index=False)
        table=evaluate(u,data,models);sums=summary(table)
        for folder in [out,dest]:
            table.to_csv(folder/'対象別TPR.csv',index=False,float_format='%.17g');sums.to_csv(folder/'集約比較.csv',index=False,float_format='%.17g')
        historical=check_catalog(table) if args.bg=='prelim' else []
        if historical:dump(dest/'catalog_regression.json',historical)
        learned=args.family.startswith('learned') or args.family=='smooth'
        record={'complete':True,'phase':data['phase'],'family':args.family,'background':args.bg,'models':len(models),'source':SCRIPTS[args.family],'source_sha256':r.sha(r.SRC/SCRIPTS[args.family]),'background_sha256':r.MANIFEST['subsets'][args.bg]['sha256'],'age_cuts':data['age_cuts'].tolist(),'team_order':data['teams'],'seconds':time.perf_counter()-started,'process_peak_rss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),'main_truth_used':False,'label_usage':'Published preliminary training labels with original person/target folds; not independent validation' if learned else 'Released preliminary labels read for scoring only after this panel is frozen','training_protocol_modified':False,'external_BG_substitution':True,'forest_prediction_execution':PROTOCOL,'forest_fit_n_jobs_original':True,'extra':extra or {},'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'recovery_time_sklearn':'1.9.0','original_experiment_versions_proven':False,'catalog_comparisons':len(historical),'catalog_max_error':max((x['absolute_error'] for x in historical),default=None),'candidate_cache_in_report_tree':False}
        for folder in [out,dest]:dump(folder/'実行記録.json',record)
        print('SAVED',args.family,args.bg,len(models),round(record['seconds'],2),'catalog max',record['catalog_max_error'],flush=True)
        return table,sums
    u.save=save
    return work

def run(args):
    u=r.initialize();work=configure(u,args);source=r.SRC/SCRIPTS[args.family]
    module=r.import_path('recovered_'+args.family,source)
    if args.family in ('learned_lowfpr','learned_centers','learned_ablation'):
        from checkpoints import install
        install(module,args,source,work)
    if args.family=='learned_yesterday':
        from reuse_legacy_core import install
        install(u,STAGE/args.family/args.bg/'legacy_core_cache_reuse.json',args.bg)
        # Retain every original feature, fold and fit. The historical-result-file
        # assertion is replaced by the separate catalog regression, because a
        # changed BG must not be asserted identical to the old BG's scores.
        tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        stop=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='old' for t in n.targets))
        fn.body=fn.body[:stop]
        if args.bg!='prelim':
            class HistoricalSignCheck(ast.NodeTransformer):
                def visit_Assert(self,node):
                    return ast.Pass() if ast.unparse(node.test)=='deviation < 1e-12' else node
            fn=HistoricalSignCheck().visit(fn);ast.fix_missing_locations(fn)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),module.__dict__)
    out=work/FOLDERS[args.family]
    historical_parallel=None
    if args.family in ('classifiers','learned_lowfpr') and out.exists():
        p=STAGE/args.family/args.bg/'実行記録.json'
        if p.exists() and json.loads(p.read_text()).get('forest_prediction_execution')!=PROTOCOL:
            historical_parallel=out.with_name(out.name+'_parallel_initial');assert not historical_parallel.exists();out.rename(historical_parallel)
            old_stage=p.parent.with_name(p.parent.name+'_parallel_initial');assert not old_stage.exists();p.parent.rename(old_stage)
    if args.family=='classifiers' and historical_parallel is not None:
        # Recompute every affected forest from the same fixed parameters. Other
        # classifiers are reused only from this same-BG, same-version frozen run.
        old=np.load(historical_parallel/'内部順位キャッシュ.npz');old_names=list(old['names']);old_scores=old['scores']
        previous=json.loads((historical_parallel/'実行記録.json').read_text())
        assert previous['background_sha256']==r.MANIFEST['subsets'][args.bg]['sha256'] and previous['numpy']==np.__version__ and previous['sklearn']==sklearn.__version__
        original_models=module.models
        module.models=lambda seed:[entry for entry in original_models(seed) if entry[0].startswith(('extra_','rf_'))]
        original_save=u.save
        def merged_save(out,data,models,specs,started,extra=None):
            assert len(models)==96
            merged={name:old_scores[i] for i,name in enumerate(old_names) if '_extra_' not in name and '_rf_' not in name}
            assert len(merged)==176;merged.update(models)
            original_specs=pd.read_csv(STAGE/args.family/(args.bg+'_parallel_initial')/'方式一覧.csv').to_dict('records')
            return original_save(out,data,merged,original_specs,started,{**(extra or {}),'deterministic_forest_recomputed_outputs':96,'unchanged_nonforest_cached_outputs':176,'cache_sha256':r.sha(historical_parallel/'内部順位キャッシュ.npz')})
        u.save=merged_save
    if out.exists() and args.family!='learned_lowfpr':raise FileExistsError(out)
    sys.argv=[str(source),'--output-dir',str(out)]
    module.main()
    # Copy non-candidate audit tables created by the original runner after save.
    target=STAGE/args.family/args.bg;target.mkdir(parents=True,exist_ok=True)
    for p in out.rglob('*'):
        if p.is_file() and p.suffix in ('.csv','.json'):
            q=target/p.relative_to(out);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    dump(target/'completion.json',{'source_main_returned':True,'family':args.family,'background':args.bg})
    if args.family in ('learned_split','learned_centers') and all((STAGE/f/args.bg/'completion.json').exists() for f in ('learned_split','learned_centers')):
        from prepare_smooth_dependencies import run as prepare_smooth
        prepare_smooth(args.bg)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=sorted(SCRIPTS),required=True);p.add_argument('--bg',choices=r.BG,required=True);run(p.parse_args())
