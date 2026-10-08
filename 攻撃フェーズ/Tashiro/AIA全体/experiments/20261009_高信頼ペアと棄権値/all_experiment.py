#!/usr/bin/env python3
"""AIA overall: fixed visible-pair correction and 0/11 miss-value audit.
No rare classifier, no hidden labels in inference, no submission/network write.
Shared evidence and scorer are loaded from the immutable rare snapshot.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, importlib.util, json, math, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REL_CORE=Path('攻撃フェーズ/Tashiro/AIA希少/experiments/20261009_高信頼ペアと棄権値/experiment.py')
CORE_SHA='0f490e82715fc192d545d62c7dcc42d63021a434b351db045cdeec6d035e2958'
POLICIES=('baseline','visible_pair','force_half','sex_half','visible_ungated')


def load_core(path):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CORE_SHA:
        raise ValueError('Shared core changed; review before re-running')
    spec=importlib.util.spec_from_file_location('pair_core',path)
    core=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=core
    spec.loader.exec_module(core)
    return core


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def infer(core,q,scores,width):
    # Membership/onset/time/is_rare are not accepted here.
    if set(q.columns)!=set(['challenge_row_id']+core.QI):
        raise ValueError('Inference requires exactly public query columns')
    ids=q.challenge_row_id.to_numpy(int)
    ranks=core.within_sex_ranks(q,scores)
    masks={}; audit=[]
    groups=core.group_indices(q,np.arange(len(q)),width)
    for method in core.METHODS:
        s=scores[method]
        base=core.baseline_mask(q,s)
        for policy in POLICIES:
            masks[method,policy]=base.copy()
        half=np.zeros(len(q),bool)
        for sex in sorted(q.sex.unique()):
            ind=np.flatnonzero(q.sex.to_numpy()==sex)
            half[ind[core.top_indices(s[ind],ids[ind],math.ceil(len(ind)/2))]]=True
        masks[method,'sex_half']=half
        force=np.zeros(len(q),bool); extras=[]
        for ind in groups.values():
            order=ind[core.top_indices(s[ind],ids[ind],len(ind))]
            force[order[:len(ind)//2]]=True
            extras.extend(order[len(ind)//2:])
        need=len(q)//2-int(force.sum()); extras=np.asarray(extras,int)
        if need:
            force[extras[core.top_indices(s[extras],ids[extras],need)]]=True
        masks[method,'force_half']=force
        for key,ind in groups.items():
            if len(ind)!=2:
                continue
            a,b=map(int,ind)
            diff=float(ranks[method][a]-ranks[method][b]); gap=abs(diff)
            direction=np.sign(diff)
            agreement=bool(direction!=0 and
                np.sign(ranks['near4'][a]-ranks['near4'][b])==direction and
                np.sign(ranks['col4_simple'][a]-ranks['col4_simple'][b])==direction)
            passed=bool(gap>=.25 and agreement)
            high,low=(a,b) if diff>=0 else (b,a)
            audit.append(dict(method=method,width=width,key='|'.join(map(str,key)),
                a=int(ids[a]),b=int(ids[b]),high=int(ids[high]),low=int(ids[low]),
                rank_gap=gap,agree=agreement,passed=passed))
            # No-score-tie rule for the ungated diagnostic.
            if direction!=0:
                masks[method,'visible_ungated'][high]=True
                masks[method,'visible_ungated'][low]=False
            if passed:
                masks[method,'visible_pair'][high]=True
                masks[method,'visible_pair'][low]=False
    return masks,audit


def run(root,out,core_path):
    start=time.time(); out.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():
        raise FileExistsError('Completed output exists; use a fresh directory')
    core=load_core(core_path)
    plan=dict(version='20261009-overall-v1',primary_metric='r_all',primary_width=5,
        widths=[5,10],policies=list(POLICIES),methods=list(core.METHODS),miss_values=[0,11],
        positive_time=9.5,baseline_sex_fraction=.75,confidence_rank_gap=.25,
        agreement=['near4','col4_simple','current_method'],
        rare_model_used=False,rare_or_onset_filter=False,
        primary_excluded_teams=[5],sensitivity_excluded_teams=[1,2,5],
        bootstrap_seed=20261009,bootstrap_repetitions=10000,
        caveat='Visible-pair/baseline/force are exact fixed extensions of prior study. '
        'sex_half and ungated are explanatory ablations frozen before this run. '
        'Previously explored benchmark; not independent confirmatory evidence.')
    dump(out/'frozen_plan.json',plan)
    data=root/'公式データ/予備戦'; hashes={}
    def read(p):
        hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return pd.read_csv(p,float_precision='round_trip')
    bg=read(data/'PWSCup2026_prelim_common/A_bg.csv')
    truth_paths=sorted((data/'結果/正解データ').glob('aia_truth_*.csv'),
                       key=lambda p:int(p.stem.rsplit('_',1)[1]))
    if len(truth_paths)!=21:
        raise ValueError('Expected all 21 official preliminary cohorts')
    records=[]; audits=[]; changes=[]; reproducible={}; tests={'cases':0,'truth_guard':0,'order_checks':0}
    for tp in truth_paths:
        team=int(tp.stem.rsplit('_',1)[1])
        q=read(data/f'PWSCup2026_prelim_attack/aia_challenge_{team}.csv').sort_values('challenge_row_id').reset_index(drop=True)
        c=read(data/f'PWSCup2026_prelim_attack/C_{team}.csv')
        truth=read(tp).sort_values('challenge_row_id').reset_index(drop=True)
        if len(q)!=326 or len(c)!=1049 or not q.challenge_row_id.is_unique:
            raise ValueError('Incorrect input shapes')
        if not q.challenge_row_id.equals(truth.challenge_row_id):
            raise ValueError('Misaligned Q and truth')
        scores=core.evidence(q,c,bg)
        try:
            infer(core,q.assign(label=truth.label),scores,5)
            raise AssertionError('Hidden label accepted')
        except ValueError:
            tests['truth_guard']+=1
        for width in plan['widths']:
            masks,aa=infer(core,q,scores,width)
            # Prove the overall policies do not depend on the rare model.
            prior,_=core.predict_masks(q,scores,np.zeros(len(q)),width)
            for method in core.METHODS:
                for policy in ('baseline','visible_pair','force_half'):
                    assert np.array_equal(masks[method,policy],prior[method,policy])
            perm=np.arange(len(q))[::-1]
            flipped,_=infer(core,q.iloc[perm].reset_index(drop=True),
                           {m:s[perm] for m,s in scores.items()},width)
            for k,v in masks.items():
                assert np.array_equal(v,flipped[k][perm]),k
            tests['order_checks']+=1
            lookup=truth.set_index('challenge_row_id')
            for a in aa:
                a.update(team=team,one_member_one_control=bool(lookup.loc[a['a'],'label']!=lookup.loc[a['b'],'label']),
                    high_is_member=bool(lookup.loc[a['high'],'label']=='member'),
                    onset_equal=bool(lookup.loc[a['a'],'onset']==lookup.loc[a['b'],'onset']),
                    is_rare_equal=bool(lookup.loc[a['a'],'is_rare']==lookup.loc[a['b'],'is_rare']),
                    original_pair_id_known=False)
                audits.append(a)
            member=truth.label.eq('member').to_numpy()
            for (method,policy),mask in masks.items():
                for miss in plan['miss_values']:
                    pred=np.where(mask,9.5,miss)
                    base=np.where(masks[method,'baseline'],9.5,miss)
                    met=core.score_prediction(pred,truth); ref=core.score_prediction(base,truth)
                    bh=np.abs(base-truth.time.to_numpy())<=.5
                    nh=np.abs(pred-truth.time.to_numpy())<=.5
                    row=dict(team=team,width=width,method=method,policy=policy,miss_value=miss,
                        changed_rows=int((pred!=base).sum()),selected_rows=int(mask.sum()),
                        member_hit_gains=int((member&~bh&nh).sum()),member_hit_losses=int((member&bh&~nh).sum()),
                        control_hit_gains=int((~member&~bh&nh).sum()),control_hit_losses=int((~member&bh&~nh).sum()),
                        delta_r_all=met['r_all']-ref['r_all'],**met)
                    records.append(row)
                    if policy in ('visible_pair','visible_ungated'):
                        for i in np.flatnonzero(pred!=base):
                            changes.append(dict(team=team,width=width,method=method,policy=policy,miss_value=miss,
                                challenge_row_id=int(q.challenge_row_id.iloc[i]),baseline=float(base[i]),modified=float(pred[i]),
                                label=str(truth.label.iloc[i]),baseline_hit=bool(bh[i]),modified_hit=bool(nh[i])))
        tests['cases']+=1
    table=pd.DataFrame(records); pairs=pd.DataFrame(audits)
    def save(df,name):
        df.to_csv(out/name,index=False,float_format='%.17g')
        reproducible[name]=hashlib.sha256((out/name).read_bytes()).hexdigest()
    save(table,'case_metrics.csv'); save(pairs,'pair_audit.csv'); save(pd.DataFrame(changes),'changed_rows.csv')
    keys=['width','method','policy','miss_value']
    metrics=['r_all','r_rare','r_max','rare_extra','p_m','p_c','delta_r_all']
    counts=['changed_rows','member_hit_gains','member_hit_losses','control_hit_gains','control_hit_losses']
    aggregates=[]; boot=[]; coverage=[]; contrasts=[]
    for name,exclude in [('official20',[5]),('official18',[1,2,5])]:
        subset=table[~table.team.isin(exclude)]
        for key,group in subset.groupby(keys,sort=True):
            d=group.delta_r_all.to_numpy()
            aggregates.append(dict(evaluation_set=name,**dict(zip(keys,key)),cases=len(group),
                **group[metrics].mean().to_dict(),**{k:int(group[k].sum()) for k in counts},
                improved=int((d>1e-12).sum()),worsened=int((d< -1e-12).sum()),tied=int((np.abs(d)<=1e-12).sum())))
            rng=np.random.default_rng(20261009)
            samples=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
            boot.append(dict(evaluation_set=name,**dict(zip(keys,key)),mean_delta=float(d.mean()),
                lower=float(np.quantile(samples,.025)),upper=float(np.quantile(samples,.975))))
        for key,group in pairs[~pairs.team.isin(exclude)].groupby(['width','method']):
            passed=group[group.passed]; valid=passed[passed.one_member_one_control]
            coverage.append(dict(evaluation_set=name,width=key[0],method=key[1],
                candidates=len(group),one_member_one_control=int(group.one_member_one_control.sum()),
                passed=len(passed),passed_valid=len(valid),correct_direction=int(valid.high_is_member.sum())))
        zero=subset[subset.miss_value.eq(0)].set_index(['team','width','method','policy'])
        eleven=subset[subset.miss_value.eq(11)].set_index(['team','width','method','policy'])
        contrast=eleven[['r_all','hit_m','hit_c']]-zero[['r_all','hit_m','hit_c']]
        contrast.columns=['delta_r_all_11_minus_0','delta_hit_m','delta_hit_c']
        contrast=contrast.reset_index();contrast.insert(0,'evaluation_set',name);contrasts.append(contrast)
    save(pd.DataFrame(aggregates),'summary.csv');save(pd.DataFrame(boot),'bootstrap.csv')
    save(pd.DataFrame(coverage),'pair_coverage.csv');save(pd.concat(contrasts,ignore_index=True),'miss_value_contrast.csv')
    dump(out/'tests.json',tests)
    dump(out/'run_manifest.json',dict(version=plan['version'],elapsed_seconds=time.time()-start,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),shared_core_sha256=CORE_SHA,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=core.scipy.__version__,
        sklearn=core.sklearn.__version__,input_sha256=hashes,result_sha256=reproducible,
        trained_rare_model=False,inference_uses_truth=False,official_phi_computed=False,
        contest_submitted=False,primary_metric='r_all'))
    print(pd.DataFrame(aggregates).query('evaluation_set=="official20" and width==5')[
        ['method','policy','miss_value','r_all','delta_r_all','changed_rows','improved','worsened']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--core',type=Path)
    args=parser.parse_args()
    run(args.root.resolve(),args.out.resolve(),args.core or args.root/REL_CORE)
