"""Fixed, cross-fitted high-confidence AIA pair audit. No submission or network I/O.

Run from any working directory:
  python analyze.py --repo /path/to/pwscup2026-sutehaji
Only this experiment's output directory is written. Published preliminary truth
is used for cross-fitted rare classification and subsequent scoring; membership,
onset and time never enter the deployable pair selector. Oracle is labelled.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, hashlib, importlib, importlib.util, json, platform, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import scipy, sklearn
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

HERE=Path(__file__).resolve().parent
METHODS=('P01','all4','top2')
QI=['age','SBP','TG','HDL','ALT','sex','prefecture']
METRICS=['r_all','r_rare','r_max','rare_extra','p_m','p_c','hit_m','hit_c','hit_m_rare','hit_c_rare']
HASHES:dict[str,str]={}

def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path:Path,x:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def read_csv(path:Path,root:Path)->pd.DataFrame:
    HASHES[str(path.relative_to(root))]=sha(path)
    return pd.read_csv(path,float_precision='round_trip')

def module(path:Path,name:str):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj

def originals(root:Path):
    old=root/'攻撃フェーズ/Tashiro/AIA全体/experiments/20261007'
    sys.path.insert(0,str(old))
    stable=importlib.import_module('同点に依存しない対応')
    colpath=root/'攻撃フェーズ/Tashiro/AIA全体/experiments/20261008/属性値集合による選別/分析.py'
    column=module(colpath,'_pair_original_columns')
    rarepath=root/'攻撃フェーズ/Tashiro/AIA希少/experiments/20261007/希少候補の学習.py'
    raremod=module(rarepath,'_pair_original_rare')
    # Its core.raw/NUM/SEED references resolve to the identical numeric transform
    # in the already imported core. No original script's run() is invoked.
    assert stable.core.NUM==['age','SBP','TG','HDL','ALT']
    assert stable.core.SEED==20261007
    for p in [old/'分析.py',old/'同点に依存しない対応.py',old/'対応競合の分析.py',
              old/'対応競合の頑健性.py',colpath,colpath.with_name('固定条件.json'),rarepath]:
        HASHES[str(p.relative_to(root))]=sha(p)
    return stable,column,raremod

def load_cases(root:Path):
    base=root/'公式データ/予備戦';attack=base/'PWSCup2026_prelim_attack'
    truth=base/'結果/正解データ'
    bg=read_csv(base/'PWSCup2026_prelim_common/A_bg.csv',root)
    cases=[]
    for p in sorted(attack.glob('aia_challenge_*.csv'),key=lambda p:int(p.stem.rsplit('_',1)[1])):
        k=int(p.stem.rsplit('_',1)[1]);q=read_csv(p,root).sort_values('challenge_row_id').reset_index(drop=True)
        c=read_csv(attack/f'C_{k}.csv',root)
        t=read_csv(truth/f'aia_truth_{k}.csv',root).sort_values('challenge_row_id').reset_index(drop=True)
        assert len(q)==len(t)==326 and len(c)==1049 and q.challenge_row_id.is_unique
        assert np.array_equal(q.challenge_row_id,t.challenge_row_id)
        assert set(q.columns)==set(QI+['challenge_row_id'])
        assert t.label.value_counts().to_dict()=={'member':163,'control':163} or t.label.value_counts().to_dict()=={'control':163,'member':163}
        assert t.groupby('label').is_rare.sum().eq(13).all()
        assert t.time.between(0,10).all()
        cases.append(dict(team=k,q=q,c=c,truth=t))
    assert len(cases)==21 and len(bg)==5143
    # Check official per-file digest list, including files not used for prediction.
    check=attack/'SHA256SUMS.txt';hashcheck=[]
    for line in check.read_text().splitlines():
        if not line.strip():continue
        digest,rel=line.split(maxsplit=1);p=attack/rel.lstrip('*').strip()
        assert p.is_file() and sha(p)==digest, f'Official checksum mismatch: {p}'
        hashcheck.append(rel)
    return cases,bg,hashcheck

def raw_score_counts(m:int,c:int,n:int=163,z:float=0)->float:
    pm,pc=m/n,c/n
    se=np.sqrt((pm*(1-pm)+pc*(1-pc))/n)
    return float(np.clip((pm-pc-z*se)/max(1-pc,1e-9),0,1))

def evaluate(pred:np.ndarray,truth:pd.DataFrame)->dict:
    assert pred.shape==(len(truth),) and np.isfinite(pred).all() and (pred>=0).all()
    mem=truth.label.eq('member').to_numpy();rare=truth.is_rare.to_numpy(bool)
    hit=np.abs(pred-truth.time.to_numpy())<=.5
    m,c=int(hit[mem].sum()),int(hit[~mem].sum())
    mr,cr=int(hit[mem&rare].sum()),int(hit[~mem&rare].sum())
    a=raw_score_counts(m,c);r=raw_score_counts(mr,cr,13,1.64)
    return dict(r_all=a,r_rare=r,r_max=max(a,r),rare_extra=max(a,r)-a,
                p_m=m/163,p_c=c/163,hit_m=m,hit_c=c,hit_m_rare=mr,hit_c_rare=cr)

def scorer_checks(src:Path,out:Path)->dict:
    sys.path.insert(0,str(src))
    from pwscup2026.scoring.attack_score import R_from_rate,R_from_rate_deadzone
    from pwscup2026.kit.validate import validate_aia_submission
    err=0.
    for n,z in [(13,1.64),(163,0.)]:
        for m in range(n+1):
            for c in range(n+1):
                a=np.r_[np.ones(m),np.zeros(n-m)];b=np.r_[np.ones(c),np.zeros(n-c)]
                ref=R_from_rate_deadzone(a,b,z) if z else R_from_rate(a,b)
                err=max(err,abs(raw_score_counts(m,c,n,z)-ref))
    assert err<1e-12
    fixture=pd.DataFrame({'target_k':[1],'challenge_row_id':[0],'time_hat':[11.]})
    vr=validate_aia_submission(fixture,{},dist={'mia_columns':{'target_columns':['1']},'challenge':{1:{0}}})
    assert vr.ok,vr.errors
    x={'all_count_cases':164**2,'rare_count_cases':14**2,'maximum_error':err,
       'local_validator_11_fixture_ok':bool(vr.ok),'live_CodaBench_test':False,
       'scorer_sha256':sha(src/'pwscup2026/scoring/attack_score.py'),
       'validator_sha256':sha(src/'pwscup2026/kit/validate.py')}
    write_json(out/'scorer_checks.json',x);return x

def fit_rare(cases,raremod,protocol,out):
    X=np.concatenate([raremod.features(v['q']) for v in cases])
    y=np.concatenate([v['truth'].is_rare.to_numpy(bool) for v in cases])
    teams=np.concatenate([np.full(326,v['team']) for v in cases])
    ids=np.array([v['team'] for v in cases]);idx=np.arange(len(cases))
    primary=list(GroupKFold(n_splits=7).split(idx,groups=ids))
    splits=[primary]
    for seed in protocol['rare_crossfit']['secondary_shuffle_seeds']:
        order=np.random.default_rng(seed).permutation(idx)
        splits.append([(np.setdiff1d(idx,te),te) for te in np.array_split(order,7)])
    results=[];fold_records=[];detection=[]
    for rep,folds in enumerate(splits):
        p=np.full(len(y),np.nan)
        for fold,(train,test) in enumerate(folds):
            tr=np.isin(teams,ids[train]);te=np.isin(teams,ids[test])
            assert not set(ids[train])&set(ids[test])
            clf=raremod.make_model().fit(X[tr],y[tr]);p[te]=clf.predict_proba(X[te])[:,1]
            fold_records.append(dict(repeat=rep,fold=fold,train=ids[train].tolist(),test=ids[test].tolist()))
        assert np.isfinite(p).all()
        split={}
        for i,v in enumerate(cases):
            pr=p[i*326:(i+1)*326];yr=y[i*326:(i+1)*326];split[v['team']]=pr
            rec=dict(repeat=rep,team=v['team'],auc=roc_auc_score(yr,pr),average_precision=average_precision_score(yr,pr))
            for n in [26,52]:
                mask=raremod.top(pr,n);rec[f'hit_top{n}']=int(yr[mask].sum());rec[f'recall_top{n}']=float(yr[mask].sum()/26)
            detection.append(rec)
        results.append(split)
        print(f'rare fit repeat={rep} completed',flush=True)
    write_json(out/'folds.json',fold_records)
    pd.DataFrame(detection).to_csv(out/'rare_detection.csv',index=False)
    return results

def percentiles(score,sex):
    r=np.empty(len(score),float)
    for s in np.unique(sex):
        a=sex==s;r[a]=(rankdata(score[a],method='average')-1)/max(int(a.sum())-1,1)
    return r

def make_baselines(cases,bg,stable,column,out):
    cache={};reproduction=[]
    for v in cases:
        q,c=v['q'],v['c'];pair,dc=stable.matrices(q,c,bg)
        masks={'P01':stable.select(pair,dc,q.sex.to_numpy(),c.sex.to_numpy(),'sex_quota')}
        ev=column.evidence(q,c,bg)
        masks['all4']=column.independent_select(q,ev,'column_all4_mean')
        masks['top2']=column.independent_select(q,ev,'column_top2_mean')
        scores={'P01':pair.max(axis=1),'all4':ev['all4'],'top2':ev['top2']}
        cache[v['team']]={k:dict(pred=np.where(masks[k],9.5,0.),score=scores[k],rank=percentiles(scores[k],q.sex.to_numpy())) for k in METHODS}
        for k in METHODS:reproduction.append(dict(team=v['team'],method=k,**evaluate(cache[v['team']][k]['pred'],v['truth'])))
        print(f'baseline team={v["team"]} completed',flush=True)
    pd.DataFrame(reproduction).to_csv(out/'baseline_reproduction.csv',index=False)
    return cache

def candidate_pairs(q,prob,score,ranks,policy,protocol,oracle_mask=None):
    if policy.startswith('oracle'):
        assert oracle_mask is not None
        pool=np.flatnonzero(oracle_mask);gate=protocol['gates']['moderate'];minprob=0.
    elif policy.startswith('visible'):
        pool=np.arange(len(q));gate=protocol['gates']['moderate'];minprob=0.
    else:
        pool=np.lexsort((q.challenge_row_id.to_numpy(),-prob))[:52]
        gate=protocol['gates'][policy.rsplit('_',1)[1]];minprob=gate['min_rare_score']
    groups={}
    for i in pool:
        x=q.iloc[i];key=(x.prefecture,int(x.sex),int(x.age)//5)
        groups.setdefault(key,[]).append(int(i))
    pairs=[]
    for key,group in sorted(groups.items()):
        if len(group)!=2:continue
        a,b=group
        if score[a]==score[b]:continue
        hi,lo=(a,b) if score[a]>score[b] else (b,a)
        gap=ranks[hi]-ranks[lo]
        if min(prob[a],prob[b])<minprob or gap<gate['min_gap'] or ranks[hi]<gate['min_high_rank']:continue
        pairs.append(dict(high=hi,low=lo,prefecture=key[0],sex=key[1],age_band_start=key[2]*5,
            rare_min=float(min(prob[a],prob[b])),high_rank=float(ranks[hi]),low_rank=float(ranks[lo]),gap=float(gap)))
    return pairs,pool

def pair_and_group_audit(cases,out):
    records=[]
    for v in cases:
        q=v['q'];t=v['truth'];f=q.assign(label=t.label.to_numpy(),is_rare=t.is_rare.to_numpy(),onset=t.onset.to_numpy())
        for width in [1,5,10]:
            for scope in ['all','rare_oracle']:
                d=f if scope=='all' else f[f.is_rare.eq(1)]
                for onset in [False,True]:
                    d=d.assign(band=d.age//width)
                    keys=['prefecture','sex','band']+(['onset'] if onset else [])
                    gr=d.groupby(keys).label.value_counts().unstack(fill_value=0).reindex(columns=['member','control'],fill_value=0)
                    n=gr.sum(axis=1)
                    records.append(dict(team=v['team'],width=width,scope=scope,onset_oracle=onset,
                      groups=len(gr),imbalanced_groups=int((gr.member!=gr.control).sum()),odd_groups=int((n%2!=0).sum()),
                      minimum_cross_band_pairs=int((gr.member-gr.control).abs().sum()/2),
                      two_row_groups=int(n.eq(2).sum()),one_each_two_row_groups=int((gr.member.eq(1)&gr.control.eq(1)).sum())))
    pd.DataFrame(records).to_csv(out/'age_group_audit.csv',index=False)

def compare(cases,base,probabilities,protocol,out):
    rows=[];pair_audit=[];changes=[]
    for rep,pr in enumerate(probabilities):
        for v in cases:
            team=v['team']
            if team==5:continue
            q,t=v['q'],v['truth'];mem=t.label.eq('member').to_numpy();rare=t.is_rare.to_numpy(bool)
            for method in METHODS:
                b=base[team][method];pred0=b['pred'];s=b['score'];r=b['rank'];prob=pr[team]
                bm=evaluate(pred0,t);oldhit=np.abs(pred0-t.time.to_numpy())<=.5
                rows.append(dict(repeat=rep,team=team,method=method,policy='unchanged',low_value=-1,attempts=0,changed=0,
                    changed_rare=0,lost_member_hits=0,removed_control_hits=0,new_member_hits=0,new_control_hits=0,**bm))
                selected={}
                for policy in protocol['policies']:
                    if policy=='budget_moderate':
                        primary=selected['rare52_moderate'];n=len(primary)
                        pool=np.lexsort((q.challenge_row_id.to_numpy(),-prob))[:52]
                        lows=pool[np.lexsort((q.challenge_row_id.to_numpy()[pool],s[pool]))[:n]]
                        pairs=[]
                    else:
                        pairs,pool=candidate_pairs(q,prob,s,r,policy,protocol,rare if policy.startswith('oracle') else None)
                        lows=np.array([p['low'] for p in pairs],dtype=int);selected[policy]=pairs
                        for p in pairs:
                            a,z=p['high'],p['low']
                            pair_audit.append(dict(repeat=rep,team=team,method=method,policy=policy,
                                high_id=int(q.challenge_row_id.iloc[a]),low_id=int(q.challenge_row_id.iloc[z]),**p,
                                high_is_member=bool(mem[a]),low_is_member=bool(mem[z]),
                                both_true_rare=bool(rare[a]&rare[z]),one_member_one_control=bool(mem[a]!=mem[z]),
                                same_onset=bool(t.onset.iloc[a]==t.onset.iloc[z]),
                                structurally_consistent_rare_pair=bool(rare[a]&rare[z]&(mem[a]!=mem[z])&(t.onset.iloc[a]==t.onset.iloc[z])),
                                correct_orientation=bool(mem[a]&~mem[z])))
                    for low in protocol['low_row_values']:
                        pred=pred0.copy();pred[lows]=low
                        outside=np.ones(len(q),bool);outside[lows]=False
                        assert np.array_equal(pred[outside],pred0[outside])
                        hit=np.abs(pred-t.time.to_numpy())<=.5;changed=pred!=pred0
                        metrics=evaluate(pred,t)
                        rows.append(dict(repeat=rep,team=team,method=method,policy=policy,low_value=low,
                            attempts=len(lows),changed=int(changed.sum()),changed_rare=int((changed&rare).sum()),
                            lost_member_hits=int((oldhit&~hit&mem).sum()),removed_control_hits=int((oldhit&~hit&~mem).sum()),
                            new_member_hits=int((~oldhit&hit&mem).sum()),new_control_hits=int((~oldhit&hit&~mem).sum()),**metrics))
                        for j in np.flatnonzero(changed):
                            changes.append(dict(repeat=rep,team=team,method=method,policy=policy,low_value=low,
                              challenge_row_id=int(q.challenge_row_id.iloc[j]),old_time=float(pred0[j]),new_time=float(pred[j]),
                              is_member=bool(mem[j]),is_rare=bool(rare[j]),old_hit=bool(oldhit[j]),new_hit=bool(hit[j])))
    frame=pd.DataFrame(rows);pairs=pd.DataFrame(pair_audit);ch=pd.DataFrame(changes)
    frame.to_csv(out/'per_target.csv',index=False,float_format='%.17g')
    pairs.to_csv(out/'candidate_pairs.csv',index=False,float_format='%.17g')
    ch.to_csv(out/'changed_rows.csv',index=False,float_format='%.17g')
    return frame,pairs

def summarize(frame,pairs,protocol,out):
    refs=frame[frame.policy.eq('unchanged')][['repeat','team','method']+METRICS].rename(columns={x:'base_'+x for x in METRICS})
    d=frame.merge(refs,on=['repeat','team','method'],validate='many_to_one')
    for m in METRICS:d['delta_'+m]=d[m]-d['base_'+m]
    scopes={'all20':sorted(d.team.unique()),'processed18':sorted(set(d.team)-{1,2})}
    scopes.update(protocol['cohort_groups'])
    result=[];boot=protocol['bootstrap'];rng=np.random.default_rng(boot['seed'])
    for scope_index,(scope,ids) in enumerate(scopes.items()):
        sub=d[d.team.isin(ids)]
        ix=np.random.default_rng(boot['seed']+scope_index).integers(0,len(ids),size=(boot['draws'],len(ids)))
        for (rep,method,policy,low),g in sub.groupby(['repeat','method','policy','low_value'],sort=True):
            rec=dict(scope=scope,repeat=int(rep),method=method,policy=policy,low_value=float(low),targets=len(g))
            rec.update({k:float(g[k].mean()) for k in METRICS+['delta_'+m for m in METRICS]})
            for k in ['attempts','changed','changed_rare','lost_member_hits','removed_control_hits','new_member_hits','new_control_hits']:rec[k]=int(g[k].sum())
            delta=g.sort_values('team').delta_r_max.to_numpy()
            rec.update(improved=int((delta>1e-12).sum()),worsened=int((delta< -1e-12).sum()),equal=int((abs(delta)<=1e-12).sum()))
            # Cohort resampling is a sensitivity interval, not independent confirmation.
            if rep==0 and scope in ['all20','processed18']:
                vals=delta[ix].mean(axis=1)
                rec.update(bootstrap_low=float(np.quantile(vals,.025)),bootstrap_high=float(np.quantile(vals,.975)),
                  remove_largest_gain=float(np.sort(delta)[:-1].mean()))
            result.append(rec)
    summary=pd.DataFrame(result);summary.to_csv(out/'summary.csv',index=False,float_format='%.17g')
    # A compact, primary-fold table suitable for repository review.
    main=summary[(summary.scope.isin(['all20','processed18']))&summary.repeat.eq(0)]
    cols=['scope','method','policy','low_value','targets','r_all','r_rare','r_max','delta_r_max','improved','worsened','equal','attempts','changed','lost_member_hits','removed_control_hits','bootstrap_low','bootstrap_high']
    main[cols].to_csv(out/'primary_summary.csv',index=False,float_format='%.12g')
    if len(pairs):
        diag=pairs.groupby(['repeat','method','policy']).agg(candidate_pairs=('team','size'),
          both_true_rare=('both_true_rare','sum'),one_member_one_control=('one_member_one_control','sum'),
          structurally_consistent_rare_pairs=('structurally_consistent_rare_pair','sum'),correct_orientation=('correct_orientation','sum'))
        diag.reset_index().to_csv(out/'pair_quality.csv',index=False)
    return main

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=HERE.parents[4]);ap.add_argument('--out',type=Path,default=HERE/'results')
    ap.add_argument('--official-src',type=Path)
    args=ap.parse_args();root=args.repo.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((HERE/'protocol.json').read_text());start=time.time()
    stable,col,rare=originals(root);cases,bg,hashcheck=load_cases(root)
    if args.official_src:src=args.official_src.resolve()
    else:src=root/'加工フェーズ/加工案/zhang_04_05_06_final28_AIA_study/zhang_04_05_06/official_scoring/src'
    if not src.is_dir():raise FileNotFoundError(f'Official scorer missing: {src}; supply --official-src')
    checks=scorer_checks(src,out)
    probabilities=fit_rare(cases,rare,cfg,out)
    baselines=make_baselines(cases,bg,stable,col,out)
    pair_and_group_audit(cases,out)
    frame,pairs=compare(cases,baselines,probabilities,cfg,out)
    primary=summarize(frame,pairs,cfg,out)
    meta=dict(input_commit=cfg['input_commit'],protocol_sha256=sha(HERE/'protocol.json'),script_sha256=sha(Path(__file__)),
       input_sha256=HASHES,official_checksums_verified=len(hashcheck),
       python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
       seconds=time.time()-start,cohorts_total=21,evaluation_targets=20,primary_fold=0,repeats=len(probabilities),
       estimated_pair_is_not_proven_generation_pair=True,phi_imputed=False,live_CodaBench_submitted=False,
       previous_exploratory_results_not_independent=True,official_scorer_checks=checks,
       rare_model_parameters=rare.make_model().get_params(),B_self_used=False)
    write_json(out/'run.json',meta)
    manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256.json'}
    write_json(out/'SHA256.json',manifest)
    selected=primary[(primary.scope=='all20')&primary.policy.isin(['unchanged','rare52_moderate','rare52_strict'])]
    print(selected[['method','policy','low_value','r_all','r_rare','r_max','delta_r_max','changed','improved','worsened']].to_string(index=False),flush=True)
    print('Completed; seconds=',round(time.time()-start,2),flush=True)

if __name__=='__main__':main()
