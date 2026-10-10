"""C-only attack selection and rank ensembles, target-held-out evaluation.
No base attack is rerun. Only pinned, public-input score banks are consumed.
All writes remain under this experiment; candidate predictions are private.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,hashlib,gc,time,argparse,datetime
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import normalized_mutual_info_score
HERE=Path(__file__).resolve().parent
ROOT=Path('/workspace/shared/mia_bg7_inputs/root/公式データ/予備戦')
OCT=Path('/workspace/shared/mia_all_bg_fusion/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_oct8')
CAL=Path('/workspace/shared/mia_all_bg_recovered_calibration/攻撃フェーズ/Tashiro/MIA全体/experiments/20261010_全方式背景7条件/recovered_calibration')
WORK=Path('/workspace/shared/mia_recovered_calibration/work')
TEAMS=[1,2,3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
FAMILIES={'histogram':1920,'geometry':666,'alignment':750,'mixture':960,'local':300,'direct':1008,'attribute_seeds':320,'stable_centers':240}
FOLDERS={'direct':'直接密度比結果','attribute_seeds':'密度比_属性乱数結果','stable_centers':'中心配置の安定化結果'}
BGS=['starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main']
QI=['age','sex','prefecture','SBP','TG','HDL','ALT']
NUM=['age','BMI','SBP','TG','HDL','ALT','FPG','time']
CAT=['sex','prefecture','smoking','onset','death']
OPTIONS=[(1,0.)]+[(k,a) for a in (.25,.5,1.) for k in (1,3,5,9)]
read=lambda p,**kw:pd.read_csv(p,float_precision='round_trip',**kw)
def dump(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for s in iter(lambda:f.read(4*1024**2),b''):h.update(s)
    return h.hexdigest()
def msha(x):return hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()
def scalar(x):return float(np.nan_to_num(x,nan=0,posinf=0,neginf=0))
def features(c):
    # Receives one C only; no target identifier, B, Q, BG, or labels.
    n=len(c);x=c[NUM].apply(pd.to_numeric,errors='coerce')
    f={'missing_fraction':c.isna().to_numpy().mean(),'duplicate_full':c.duplicated().mean(),'duplicate_qi7':c[QI].duplicated().mean()}
    uniq=np.array([x[v].nunique()/n for v in NUM]);f.update(numeric_unique_mean=uniq.mean(),numeric_unique_min=uniq.min())
    for unit in (.1,1.,5.,10.):
        f[f'grid_multiple_{unit:g}']=np.mean([np.mean(np.isclose(v/unit,np.round(v/unit),atol=1e-8,rtol=0)) for v in [x[s].dropna().to_numpy() for s in NUM]])
    ent=[];top=[];card=[]
    for v in CAT:
        p=c[v].astype(str).value_counts(normalize=True).to_numpy();ent.append(-(p*np.log(p)).sum()/np.log(max(len(p),2)));top.append(p.max());card.append(len(p)/n)
    f.update(category_entropy_mean=np.mean(ent),category_topmass_mean=np.mean(top),category_cardinality_mean=np.mean(card),category_cardinality_max=max(card))
    corr=x.corr(method='spearman').to_numpy();v=np.abs(corr[np.triu_indices(len(NUM),1)]);v=np.nan_to_num(v)
    f.update(spearman_abs_mean=v.mean(),spearman_abs_max=v.max())
    mi=[normalized_mutual_info_score(c[a].astype(str),c[b].astype(str)) for i,a in enumerate(CAT) for b in CAT[i+1:]]
    f['category_association_mean']=np.mean(mi)
    skew=[];tail=[];disp=[]
    for col in NUM:
        p05,p25,p50,p75,p95=x[col].quantile([.05,.25,.5,.75,.95]);iqr=p75-p25
        skew.append((p75+p25-2*p50)/(iqr+1e-8));tail.append((p95-p05)/(iqr+1e-8));disp.append(iqr/(abs(p50)+iqr+1e-8))
    f.update(bowley_skew_mean=np.mean(skew),tail_iqr_ratio_mean=np.mean(np.clip(tail,0,100)),relative_dispersion_mean=np.mean(disp))
    return {k:scalar(v) for k,v in f.items()}
def extract_features():
    rows=[];inputs=[]
    for t in TEAMS:
        p=ROOT/f'PWSCup2026_prelim_attack/C_{t}.csv';c=read(p);assert len(c)==1049
        rows.append({'team':t,**features(c)});inputs.append({'team':t,'file':str(p),'sha256':sha(p),'rows':len(c)})
    f=pd.DataFrame(rows);f.to_csv(HERE/'C_only_features.csv',index=False,float_format='%.17g')
    dump(HERE/'C_only_feature_input_audit.json',{'only_C_files_read_for_feature_construction':True,'inputs':inputs,'feature_columns':list(f.columns[1:]),'feature_count':len(f.columns)-1,'target_id_excluded_from_feature_matrix':True})
    return f.drop(columns='team').to_numpy()
def catalog():
    rows=[];banks=[];ms=[]
    for fam,count in FAMILIES.items():
        reference=None
        for bg in BGS:
            if fam in FOLDERS:
                d=CAL/fam/bg;p=WORK/bg/FOLDERS[fam]/'内部順位キャッシュ.npz';record=d/'実行記録.json';seal=d/'completion.json'
                r=json.loads(record.read_text());assert r['complete'] and r['models']==count and seal.exists()
                table=read(d/'対象別TPR.csv',usecols=['method','team','fpr','mia_all']);table=table[np.isclose(table.fpr,.01)].rename(columns={'mia_all':'tpr'})
            else:
                d=OCT/fam/'prelim'/bg;p=d/'scores.npz';record=d/'prediction_record.json';r=json.loads(record.read_text());assert r['truth_read'] is False and r['methods']==count
                table=read(d/'by_target.csv',usecols=['method','team','population','tpr_0.01']);table=table[table.population=='all_fixed'].rename(columns={'tpr_0.01':'tpr'})
            with np.load(p) as z:names=z['names'].tolist();teams=z['teams'].tolist()
            assert len(names)==count and set(teams)==set(TEAMS)
            assert set(table.method)==set(names) and len(table)==count*20
            if reference is None:reference=names
            assert names==reference,('grid mismatch',fam,bg)
            hashes=json.loads((d/'prediction_hashes.json').read_text());hd={v['method']:v['sha256'] for v in hashes};assert set(hd)==set(names)
            matrix=table.pivot(index='method',columns='team',values='tpr').loc[names,TEAMS].to_numpy();assert np.isfinite(matrix).all();ms.append(matrix)
            offset=len(rows)
            rows.extend({'candidate_index':offset+i,'family':fam,'background':bg,'method':n,'source_index':i} for i,n in enumerate(names))
            banks.append({'family':fam,'background':bg,'path':str(p),'record':str(record),'names':names,'teams':teams,'offset':offset,'count':count,'hashes':hd,'record_sha256':sha(record),'metrics_sha256':sha(d/('対象別TPR.csv' if fam in FOLDERS else 'by_target.csv'))})
            print('CATALOG',fam,bg,count,flush=True)
    meta=pd.DataFrame(rows);m=np.vstack(ms);assert m.shape==(43148,20)
    meta.to_csv(HERE/'candidate_catalog.csv',index=False)
    np.save(HERE/'private_cache'/'candidate_metrics.npy',m)
    return meta,banks,m

def neighbors(x,train,test):
    tr=x[train];sd=tr.std(axis=0);usable=sd>1e-10
    # Standardization is fitted on outer/inner training targets, never all C's.
    d=np.mean(((tr[:,usable]-x[test,usable])/sd[usable])**2,axis=1)
    return np.asarray(train)[np.argsort(d,kind='stable')]
def choose_all(m,x,train,test):
    avg=m[:,train].mean(axis=1);nn=neighbors(x,train,test)
    partial=np.cumsum(m[:,nn[:min(9,len(nn))]],axis=1)
    idx=[]
    for k,a in OPTIONS:
        k=min(k,len(nn));pred=avg if a==0 else (1-a)*avg+a*partial[:,k-1]/k
        idx.append(int(np.argmax(pred)))
    return idx,nn

def select(m,x,meta):
    all_plans=[];foldrows=[];nestedrows=[];leak=[]
    for scope,active in [('all20',list(range(20))),('train_and_test_without1_2',list(range(2,20)))]:
        for t in active:
            train=[j for j in active if j!=t];opts,nn=choose_all(m,x,train,t)
            inner=[]
            for valid in train:
                it=[j for j in train if j!=valid];ii,_=choose_all(m,x,it,valid)
                inner.append(m[ii,valid])
            objective=np.mean(inner,axis=0);oi=int(np.argmax(objective));fixed=OPTIONS.index((3,.5))
            selected={'global_train_best':opts[0],'C_gate_fixed_k3_weight05':opts[fixed],'C_gate_nested':opts[oi],'oracle_infeasible':int(np.argmax(m[:,t]))}
            champions=[];avg=m[:,train].mean(axis=1)
            for fam in FAMILIES:
                indices=meta.index[meta.family==fam].to_numpy();champions.append(int(indices[np.argmax(avg[indices])]))
            order=np.argsort(-avg[champions],kind='stable');cw=np.exp((avg[champions]-avg[champions].max())/.05);cw/=cw.sum()
            all_plans.append({'scope':scope,'target_column':t,'train_columns':train,'selected':selected,'champions':champions,'champion_order':order.tolist(),'champion_weights':cw.tolist()})
            for strategy,i in selected.items():
                foldrows.append({'scope':scope,'team':TEAMS[t],'strategy':strategy,'candidate_index':i,**meta.iloc[i][['family','background','method']].to_dict(),'training_mean_tpr':avg[i],'test_tpr':m[i,t], 'k':OPTIONS[oi][0] if strategy=='C_gate_nested' else 3 if 'fixed_k3' in strategy else None,'local_weight':OPTIONS[oi][1] if strategy=='C_gate_nested' else .5 if 'fixed_k3' in strategy else None,'nearest_training_teams':','.join(str(TEAMS[j]) for j in nn[:3])})
            for o,value in zip(OPTIONS,objective):nestedrows.append({'scope':scope,'team':TEAMS[t],'k':o[0],'local_weight':o[1],'inner_loto_mean_tpr':value,'chosen':o==OPTIONS[oi]})
            # Strong exclusion audit: perturb all evaluation-target metrics.
            altered=m.copy();altered[:,t]=np.linspace(100,200,len(m))
            check,_=choose_all(altered,x,train,t);assert check==opts
            ach=[int(meta.index[meta.family==fam].to_numpy()[np.argmax(altered[meta.family==fam][:,train].mean(axis=1))]) for fam in FAMILIES]
            assert ach==champions
            leak.append({'scope':scope,'heldout_team':TEAMS[t],'heldout_metric_perturbation_changes_fixed_gate_or_global_or_champions':False,'heldout_target_not_in_any_inner_training_or_validation_set':t not in train})
            print('SELECT',scope,TEAMS[t],OPTIONS[oi],flush=True)
    pd.DataFrame(foldrows).to_csv(HERE/'selected_methods_by_target.csv',index=False,float_format='%.17g')
    pd.DataFrame(nestedrows).to_csv(HERE/'nested_gate_inner_validation.csv',index=False,float_format='%.17g')
    dump(HERE/'heldout_exclusion_audit.json',leak)
    dump(HERE/'private_cache'/'selection_plan.json',all_plans)
    return all_plans

def load_eval():
    b=read(next(ROOT.glob('PWSCup2026_prelim_team05_*/B_self.csv')));q=read(ROOT/'PWSCup2026_prelim_attack/aia_challenge_5.csv')
    private=pd.MultiIndex.from_frame(b[QI]).isin(pd.MultiIndex.from_frame(q[QI]));rare=b.is_rare.to_numpy(bool);normal=~private&~rare
    assert (private.sum(),rare.sum(),(private&rare).sum(),normal.sum())==(163,24,13,875)
    folder=ROOT/'結果/正解データ';own=read(folder/'pool_ids_5.csv').set_index('record_id').loc[b.record_id].pool_id
    y=np.column_stack([own.isin(read(folder/f'pool_ids_{t}.csv').pool_id).to_numpy(int) for t in TEAMS])
    return b,private,rare,normal,y

def confidence(s,private,rare,normal):
    result=np.zeros(1049);result[rare&~private]=.4;result[normal]=.6+.3*rankdata(s,method='min')/normal.sum();return result

def roc(s,y):
    ix=np.argsort(-s,kind='stable');ss=s[ix];yy=y[ix];ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(s)-1];cs=np.cumsum(yy)
    tp=np.r_[0,cs[ends]]/y.sum();fp=np.r_[0,ends+1-cs[ends]]/(len(y)-y.sum())
    return float(np.interp(.01,fp,tp))

def stream(banks,meta,m,plans):
    b,private,rare,normal,y=load_eval()
    needed=set()
    for p in plans:
        t=p['target_column'];needed.update((i,t) for i in list(p['selected'].values())+p['champions'])
    collected={};group_ranks=[];group_info=[];uniform=np.zeros((875,20));total=0;audit=[];maxerr=0.
    for bank in banks:
        p=Path(bank['path']);print('STREAM',bank['family'],bank['background'],flush=True)
        with np.load(p) as z:
            scores=z['scores'];teams=z['teams'].tolist();order=[teams.index(t) for t in TEAMS]
            if 'record_ids' in z.files:assert np.array_equal(z['record_ids'],b.loc[normal,'record_id'])
            assert scores.shape==(bank['count'],875,20) and np.isfinite(scores).all()
            for i,name in enumerate(bank['names']):assert msha(scores[i])==bank['hashes'][name]
            sums=np.zeros((875,20));verified=0
            for i in range(bank['count']):
                s=scores[i][:,order];r=(rankdata(s,axis=0,method='average')-.5)/875
                sums+=r
                gi=bank['offset']+i
                for t in range(20):
                    if (gi,t) in needed:collected[(gi,t)]=r[:,t].copy()
                if i in (0,bank['count']-1):
                    for t in range(20):
                        v=roc(confidence(s[:,t],private,rare,normal),y[:,t]);err=abs(v-m[gi,t]);maxerr=max(err,maxerr);assert err<1e-12,(bank['family'],bank['background'],name,t,v,m[gi,t]);verified+=1
            uniform+=sums;total+=bank['count'];group_ranks.append(sums/bank['count']);group_info.append((bank['family'],bank['background']))
            audit.append({'family':bank['family'],'background':bank['background'],'candidate_count':bank['count'],'all_matrix_hashes_passed':True,'sample_metric_checks':verified,'score_file_sha256':sha(p),'record_sha256':bank['record_sha256'],'metrics_sha256':bank['metrics_sha256'],'source_team_order':teams,'canonical_team_order':TEAMS})
        del scores;gc.collect()
    assert len(collected)==len(needed)
    groups=np.stack(group_ranks);family_ranks=np.stack([np.mean(groups[[i for i,v in enumerate(group_info) if v[0]==fam]],axis=0) for fam in FAMILIES])
    frozen={'fixed_uniform_all_candidates':uniform/total,'fixed_equal_family_background_mean':family_ranks.mean(axis=0),'fixed_family_consensus_median':np.median(family_ranks,axis=0),'fixed_family_consensus_max':family_ranks.max(axis=0)}
    dump(HERE/'source_score_audit.json',{'banks':audit,'matrices_hashed':total,'max_sample_metric_error':maxerr,'n_eval':1049,'n_normal':875,'n_private':163,'n_rare':24,'primary_metric':'official-style linear interpolated TPR at FPR .01','private_predictions_published':False})
    np.savez_compressed(HERE/'private_cache'/'fixed_ensembles.npz',**frozen,teams=TEAMS)
    res=[];weights=[];outs={}
    for plan in plans:
        t=plan['target_column'];scores={k:collected[(i,t)] for k,i in plan['selected'].items()}
        champions=np.stack([collected[(i,t)] for i in plan['champions']]);order=plan['champion_order'];cw=np.asarray(plan['champion_weights'])
        scores.update(ensemble_train_champion_mean=champions.mean(axis=0),ensemble_train_champion_median=np.median(champions,axis=0),ensemble_train_champion_max=champions.max(axis=0),ensemble_train_top3_families_mean=champions[order[:3]].mean(axis=0),ensemble_train_top5_families_mean=champions[order[:5]].mean(axis=0),ensemble_train_champion_weighted=cw@champions)
        scores.update({k:s[:,t] for k,s in frozen.items()})
        for k,s in scores.items():
            v=roc(confidence(s,private,rare,normal),y[:,t]);res.append({'scope':plan['scope'],'team':TEAMS[t],'strategy':k,'tpr_0.01':v,'members':int(y[:,t].sum()),'population':'all_fixed','n':1049})
            outs[f"{plan['scope']}::{TEAMS[t]}::{k}"]=s
            if k in plan['selected']:assert abs(v-m[plan['selected'][k],t])<1e-12
        for fam,i,w in zip(FAMILIES,plan['champions'],cw):weights.append({'scope':plan['scope'],'team':TEAMS[t],'family':fam,'candidate_index':i,'background':meta.loc[i,'background'],'method':meta.loc[i,'method'],'weight_temperature005':w,'training_mean_tpr':m[i,plan['train_columns']].mean()})
    np.savez_compressed(HERE/'private_cache'/'heldout_prediction_scores.npz',**outs)
    pd.DataFrame(weights).to_csv(HERE/'ensemble_training_weights.csv',index=False,float_format='%.17g')
    results=pd.DataFrame(res);results.to_csv(HERE/'heldout_by_target.csv',index=False,float_format='%.17g')
    return results

def summarize(results):
    rows=[];deltas=[];rng=np.random.default_rng(20261010)
    for view,g in [('all20',results[results.scope=='all20']),('without1_2_summary_of_all20',results[(results.scope=='all20')&~results.team.isin([1,2])]),('train_and_test_without1_2',results[results.scope=='train_and_test_without1_2'])]:
        base=g[g.strategy=='global_train_best'].set_index('team')['tpr_0.01'];draw=rng.integers(0,len(base),(10000,len(base)))
        for name,h in g.groupby('strategy',sort=False):
            s=h.set_index('team')['tpr_0.01'].loc[base.index];delta=s-base;bs=delta.to_numpy()[draw].mean(axis=1)
            rows.append({'view':view,'strategy':name,'targets':len(h),'mean_tpr':s.mean(),'median_tpr':s.median(),'TPR_ge010':int((s>=.1-1e-12).sum()),'TPR_gt010':int((s>.1+1e-12).sum()),'mean_delta_train_global':delta.mean(),'wins':int((delta>1e-12).sum()),'ties':int((abs(delta)<=1e-12).sum()),'losses':int((delta< -1e-12).sum()),'delta_bootstrap_p025':np.quantile(bs,.025),'delta_bootstrap_p975':np.quantile(bs,.975),'bootstrap_note':'Descriptive target bootstrap; overlapping LOTO folds and adaptive historical grid prevent confirmatory inference'})
            for t in s.index:deltas.append({'view':view,'strategy':name,'team':t,'tpr':s[t],'baseline_tpr':base[t],'delta':delta[t]})
    sums=pd.DataFrame(rows);sums.to_csv(HERE/'summary.csv',index=False,float_format='%.17g');pd.DataFrame(deltas).to_csv(HERE/'paired_target_deltas.csv',index=False,float_format='%.17g')
    print(sums[['view','strategy','mean_tpr','TPR_ge010','TPR_gt010','mean_delta_train_global']].to_string(index=False),flush=True)
    return sums

def test_roc():
    sys.path.insert(0,'/workspace/shared/mia_bg7_inputs/official_scoring/src')
    from pwscup2026.scoring.attack_score import _tpr_at_fpr
    rng=np.random.default_rng(20261010);err=0;cases=0
    for n in (875,1049):
        for members in (1,7,25,100):
            y=np.r_[np.ones(members),np.zeros(n-members)];rng.shuffle(y)
            for s in (rng.random(n),rng.integers(0,8,n),np.ones(n)):
                err=max(err,abs(roc(s,y)-_tpr_at_fpr(s,y,.01)[0]));cases+=1
    assert err<1e-12;dump(HERE/'official_scoring_test.json',{'cases':cases,'max_error':err,'passed':True})

def main():
    (HERE/'private_cache').mkdir(exist_ok=True);start=time.time();x=extract_features();test_roc()
    meta,banks,m=catalog();plans=select(m,x,meta);results=stream(banks,meta,m,plans);summarize(results)
    dump(HERE/'completion.json',{'complete':True,'elapsed_seconds':time.time()-start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_sha256':sha(__file__),'protocol_sha256':sha(HERE/'protocol.json'),'candidate_count':len(meta),'features':x.shape[1],'threads':1,'base_models_rerun':0})
if __name__=='__main__':main()
