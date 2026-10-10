from pathlib import Path
import argparse,json,time,sys,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import roc_curve
import runtime as r

OUT=Path('/workspace/shared/mia_all_bg_recovered_calibration/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_calibration/broad')
CACHE=Path('/workspace/shared/mia_recovered_calibration/cache/broad')

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def scores(u,data,bg,phase,verify=False):
    raw=r.raw_cache(data,bg);original=u.cal.common.core
    byid={id(c):(j,t) for j,(t,c) in enumerate(data['cs'].items())}
    def core(a,query,c,columns,ks=(10,),**kw):
        j,t=byid[id(c)];ii=np.r_[np.arange(len(data['b'])),data['controls'][t]]
        assert list(columns)==list(u.NUM7) and tuple(ks)==(10,)
        assert query[list(u.base.QI7)].reset_index(drop=True).equals(data['query'].iloc[ii][list(u.base.QI7)].reset_index(drop=True))
        return {10:raw[ii,j]}
    controls={t:data['query'].iloc[ii].reset_index(drop=True) for t,ii in data['controls'].items()}
    u.cal.common.core=core
    try:
        models,counts=u.cal.predict(data['b'],data['a'],controls,data['cs'],data['normal'],u.cal.specs(),bg)
        mix=r.import_path('broad_mix',r.SRC.parent/'20261008_MIA全体_較正の広範比較/追加混合/分析.py')
        mix.sweep=u.cal;mix.common=u.cal.common
        first,configs=mix.configurations()
        mixed=mix.predict(data['b'],data['a'],controls,data['cs'],data['normal'],first,configs,bg)
    finally:u.cal.common.core=original
    overlap={name:bool(np.array_equal(v,models[name])) for name,v in mixed.items() if name in models}
    assert len(overlap)==7 and all(overlap.values())
    result={**{'broad65::'+n:v for n,v in models.items()},**{'mixture31::'+n:v for n,v in mixed.items()}}
    checks={}
    if verify:
        direct,_=u.cal.predict(data['b'],data['a'],controls,data['cs'],data['normal'],u.cal.specs(),'original_direct')
        checks={name:{'exact_match':bool(np.array_equal(v,direct[name])),'max_error':float(abs(v-direct[name]).max())} for name,v in models.items()}
        assert all(v['exact_match'] for v in checks.values())
    return result,counts,checks

def truth(data):
    p=r.INPUT/'root/公式データ/予備戦/結果/正解データ'
    b=data['b'];own=pd.read_csv(p/'pool_ids_5.csv').set_index('record_id').loc[b.record_id].pool_id
    return np.column_stack([own.isin(pd.read_csv(p/f'pool_ids_{t}.csv').pool_id).to_numpy(int) for t in data['teams']])

def evaluate(u,data,models,bg):
    y=truth(data);rows=[]
    for name,values in models.items():
        family,method=name.split('::')
        for j,t in enumerate(data['teams']):
            conf=u.base.confidence(values[:,j],data['normal'],data['rare'],data['private'])
            fp,tp,_=roc_curve(y[:,j],conf,drop_intermediate=False)
            rows.append({'background':bg,'family':family,'method':method,'inference':'all20','team':t,'members':int(y[:,j].sum()),**{f'tpr_{a:g}':float(np.interp(a,fp,tp)) for a in u.FPRS}})
    return pd.DataFrame(rows)

def catalog_check(table):
    catalog=pd.read_csv(r.RECOVERY/'catalog/方式条件一覧.csv');checks=[]
    for _,row in catalog[(catalog.condition=='full')&catalog.source_ids.str.contains('S0335|S0336',regex=True)].iterrows():
        names=set(str(row.source_method_aliases).split(';'))|{row.method}
        candidates=table[table.method.isin(names)]
        for (family,method),g in candidates.groupby(['family','method']):
            ids=list(map(int,row.evaluated_teams.split(',')));g=g.set_index('team').loc[ids]
            measured=float(g['tpr_0.01'].mean());err=abs(measured-row.mean_tpr)
            checks.append({'condition_id':row.condition_id,'family':family,'method':method,'evaluation_targets':ids,'historical':row.mean_tpr,'current':measured,'absolute_error':err,'pass':err<1e-12})
    return checks

def run(args):
    u=r.initialize();start=time.perf_counter();rows=[];allpred={}
    bgs=['prelim'] if args.baseline_only else ['prelim']+[x for x in r.BG if x!='prelim']
    for bg in bgs:
        data=r.data_for(u,bg,args.phase);dest=OUT/args.phase/bg;cache=CACHE/f'{args.phase}_{bg}.npz'
        if cache.exists():
            z=np.load(cache);models={n:z[n] for n in z.files}
        else:
            before=time.perf_counter();models,counts,checks=scores(u,data,bg,args.phase,args.baseline_only and args.phase=='prelim')
            cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**models)
            dump(dest/'prediction_hashes.json',[{'family':n.split('::')[0],'method':n.split('::')[1],'shape':list(v.shape),'sha256':r.matsha(v)} for n,v in models.items()])
            dump(dest/'record.json',{'phase':args.phase,'bg':bg,'methods':len(models),'unique_methods':89,'age_cuts':data['age_cuts'].tolist(),'bg_sha256':r.MANIFEST['subsets'][bg]['sha256'],'seconds':time.perf_counter()-before,'labels_used_for_prediction':False})
            if checks:dump(dest/'raw_cache_original_regression.json',checks)
            pd.DataFrame(counts).to_csv(dest/'support_counts.csv',index=False)
            print('FROZEN',args.phase,bg,'96 registrations,89 unique',round(time.perf_counter()-before,3),flush=True)
        allpred[bg]=models
    if args.phase=='prelim':
        data=r.data_for(u,'prelim','prelim')
        for bg,models in allpred.items():rows.append(evaluate(u,data,models,bg))
        table=pd.concat(rows,ignore_index=True);table.to_csv(OUT/'prelim/by_target.csv',index=False,float_format='%.17g')
        checks=catalog_check(table[table.background=='prelim']);dump(OUT/'prelim/catalog_regression.json',checks)
        assert checks and all(v['pass'] for v in checks),[v for v in checks if not v['pass']][:10]
        sums=[]
        for evaluation,excluded in [('all20',[]),('without1_2',[1,2])]:
            q=table[~table.team.isin(excluded)]
            for key,g in q.groupby(['background','family','method']):sums.append(dict(zip(['background','family','method'],key))|{'inference':'all20','evaluation':evaluation,'targets':len(g),'mean_tpr':float(g['tpr_0.01'].mean()),'tpr_ge010':int((g['tpr_0.01']>=.1-1e-12).sum())})
        pd.DataFrame(sums).to_csv(OUT/'prelim/summary.csv',index=False,float_format='%.17g')
    else:
        data=r.data_for(u,'main','main');rows=[]
        for bg,models in allpred.items():
            for name,v in models.items():
                ref=allpred['main'][name]
                for j,t in enumerate(data['teams']):rows.append({'background':bg,'family':name.split('::')[0],'method':name.split('::')[1],'inference':'all22','team':t,'spearman':float(spearmanr(v[:,j],ref[:,j]).statistic),'top20_overlap':len(set(np.argsort(-v[:,j],kind='stable')[:20])&set(np.argsort(-ref[:,j],kind='stable')[:20]))})
        pd.DataFrame(rows).to_csv(OUT/'main/rank_by_target.csv',index=False,float_format='%.17g')
    dump(OUT/args.phase/'run_record.json',{'complete':True,'backgrounds':bgs,'registrations_per_BG':96,'unique_methods_per_BG':89,'seconds':time.perf_counter()-start,'catalog_checks':len(checks) if args.phase=='prelim' else None,'no_main_truth':True})
    print('COMPLETE',args.phase,time.perf_counter()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prelim','main'],default='prelim');p.add_argument('--baseline-only',action='store_true');run(p.parse_args())
