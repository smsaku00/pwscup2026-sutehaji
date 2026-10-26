#!/usr/bin/env python3
"""PWS Cup 2026 published final-ground-truth audit & experiments.

Inputs: the two 2026-10-26 official ZIP archives + the XLSX value extract
produced by extract_official.py.  Does not use privileged unpublished B data.
Only published final submissions are considered; these are in-sample/post-hoc.
"""
from __future__ import annotations
import csv, io, json, math, zipfile, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
TABLE = json.loads((HERE/'official_table_data.json').read_text(encoding='utf8'))
ID=sorted(int(r['コホート番号']) for r in TABLE['総括'])
ROOT=HERE/'正解データ'
ZIP=HERE/'攻撃提出物'
FLOORS={int(r['team_id']):float(r['aia_floor']) for r in csv.DictReader((ROOT/'aia_floor.csv').open())}


def readcsv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as fp: return list(csv.DictReader(fp))

POOL={i: readcsv(ROOT/f'pool_ids_{i}.csv') for i in ID}
POOLS={i:set(int(r['pool_id']) for r in POOL[i]) for i in ID}
TRUTHS={i: readcsv(ROOT/f'aia_truth_{i}.csv') for i in ID}
O_MIA={(int(r['攻撃者']),int(r['標的'])):r for r in TABLE['攻撃_標的別_MIA']}
O_AIA={(int(r['攻撃者']),int(r['標的'])):r for r in TABLE['攻撃_標的別_AIA']}


def interpolated_roc(x: np.ndarray, y: np.ndarray, fpr=0.01):
    """Exact current published scorer: tie-block ROC then np.interp."""
    npos=int(y.sum());nneg=len(y)-npos
    if npos==0 or nneg==0: return 0.0,0.5
    order=np.argsort(-x,kind='mergesort');xs=x[order]; ys=y[order]
    idx=np.r_[np.flatnonzero(np.diff(xs)!=0),len(xs)-1]
    tpr=np.r_[0,np.cumsum(ys)[idx]/npos]
    fp=np.r_[0,np.cumsum(1-ys)[idx]/nneg]
    return float(np.interp(fpr,fp,tpr)),float(np.trapezoid(tpr,fp))


def r_rate(s1, s0, rare=False):
    a,b=np.asarray(s1,dtype=float),np.asarray(s0,dtype=float)
    if not len(a) or not len(b):return float('nan')
    pm,pc=float(a.mean()),float(b.mean())
    se=math.sqrt(pm*(1-pm)/len(a)+pc*(1-pc)/len(b)) if rare else 0
    return float(np.clip((pm-pc-1.64*se)/max(1-pc,1e-9),0,1))


def aia_eval(k:int, pred: dict[int,float]):
    tr=TRUTHS[k]
    selected=[r for r in tr if int(r['challenge_row_id']) in pred]
    pm=np.array([r['label']=='member' for r in selected],dtype=bool)
    pr=np.array([int(r['is_rare'])==1 for r in selected],dtype=bool)
    success=np.array([abs(pred[int(r['challenge_row_id'])]-float(r['time']))<=.5 for r in selected],dtype=float)
    r_all=r_rate(success[pm],success[~pm]);r_rare=r_rate(success[pm&pr],success[(~pm)&pr],rare=True)
    r_rare_raw=r_rate(success[pm&pr],success[(~pm)&pr],rare=False)
    r=max(r_all,r_rare) if np.isfinite(r_rare) else r_all
    return {'aia_all':r_all,'aia_rare':r_rare,'aia_rare_raw':r_rare_raw,'aia_floored':max(0,r-FLOORS[k]),
            'p_m':float(success[pm].mean()),'p_c':float(success[~pm].mean()),
            'p_m_rare':float(success[pm&pr].mean()) if (pm&pr).sum()>0 else float('nan'),
            'p_c_rare':float(success[(~pm)&pr].mean()) if ((~pm)&pr).sum()>0 else float('nan'),
            'rows':len(selected)}


def mia_eval(a:int,k:int,scores:np.ndarray,fpr=0.01):
    pools=np.array([int(r['pool_id']) for r in POOL[a]])
    y=np.isin(pools,np.array(list(POOLS[k]))).astype(int)
    rare=np.array([int(r['is_rare']) for r in POOL[a]],dtype=bool)
    t,auc=interpolated_roc(scores,y,fpr)
    rt=float('nan')
    if y[rare].sum()>0:rt,_=interpolated_roc(scores[rare],y[rare],fpr)
    return {'tpr_all':t,'tpr_rare':rt,'tpr_score':max(t,rt) if np.isfinite(rt) else t,'auc':auc,'n_members':int(y.sum()),
            'n_rare_members':int(y[rare].sum()),'n_rare_candidates':int(rare.sum())}


def read_submission(a):
    with zipfile.ZipFile(ZIP/f'cohort{a:02d}_F.zip') as z:
        mia=list(csv.DictReader(io.StringIO(z.read('F_mia.csv').decode('utf-8-sig'))))
        aia=list(csv.DictReader(io.StringIO(z.read('F_aia.csv').decode('utf-8-sig'))))
    assert len(mia)==len(POOL[a])==1049
    mid={int(r['record_id']):r for r in mia}
    assert len(mid)==1049 and set(mid)==set(range(1049))
    preds=defaultdict(dict)
    for r in aia:
        k=int(r['target_k']);j=int(r['challenge_row_id']);preds[k][j]=float(r['time_hat'])
    return mid,preds


def cleanfloat(v):return round(float(v),12) if np.isfinite(v) else None


def dump_csv(fname,rows):
    if not rows:return
    with (HERE/fname).open('w',newline='',encoding='utf8') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run():
    mrs,ars,atts,preds,by_att={}, {},{}, {}, {}
    diffs=defaultdict(list)
    for a in ID:
        mi,ai=read_submission(a); preds[a]=ai; av=[];mv=[]
        for k in ID:
            if k==a:continue
            scores=np.array([float(mi[j][str(k)]) for j in range(1049)],dtype=float)
            m=mia_eval(a,k,scores);A=aia_eval(k,ai[k]); mrs[a,k]=m;ars[a,k]=A
            expectedM=O_MIA[a,k]; expectedA=O_AIA[a,k]
            expected_pairs=[('n_members','標的のメンバー数',m['n_members']),('n_rare_members','希少なメンバー数',m['n_rare_members']),
                            ('tpr_all','TPR（全体）',m['tpr_all']),('tpr_rare','TPR（希少な候補だけ）',m['tpr_rare']),('tpr_score','t_j（悪いほう）',m['tpr_score']),('auc','AUC',m['auc'])]
            for key,col,v in expected_pairs:
                x=expectedM[col]
                if x is not None and np.isfinite(v):diffs['MIA_'+key].append(abs(float(x)-v))
            for key,col in [('aia_all','R（全体）'),('aia_rare','R（希少な患者だけ）'),('aia_rare_raw','R（希少・誤差の範囲を差し引く前）'),('p_m','p_m'),('p_c','p_c'),('p_m_rare','p_m_rare'),('p_c_rare','p_c_rare')]:
                v=A[key];x=expectedA[col]
                if x is not None and np.isfinite(v):diffs['AIA_'+key].append(abs(float(x)-v))
            mv.append(m['tpr_score']);av.append(A['aia_floored'])
        by_att[a]={'mia_power':float(np.mean(mv)), 'aia_power':float(np.mean(av)), 'attack_power':float(np.mean(mv)+np.mean(av))}
        official=next(x for x in TABLE['総括'] if int(x['コホート番号'])==a)
        diffs['ATTACK_POWER'].append(abs(by_att[a]['attack_power']-float(official['攻撃力'])))
    assert all(max(vals or [0])<1e-10 for vals in diffs.values()),str({k:max(v) for k,v in diffs.items()})

    rows=[]
    for a in ID:
        row=next(r for r in TABLE['最終得点'] if int(r['本戦コホート'])==a)
        rows.append({'cohort':a,'team':row['チーム名'],'attack_rank':int(row['本戦のみの攻撃力順位']),
                    'mia_power':by_att[a]['mia_power'],'aia_power':by_att[a]['aia_power'],
                    'attack_power':by_att[a]['attack_power'], 'official_power':row['本戦_攻撃力']})
    rows.sort(key=lambda r:-r['attack_power']); dump_csv('all_teams_recomputed.csv',rows)
    per=[]
    for k in ID:
        if k==28:continue
        m=mrs[28,k];r=ars[28,k]
        per.append({'target':k,'team':next(t['チーム名'] for t in TABLE['総括'] if int(t['コホート番号'])==k),
                    'n_members':m['n_members'],'n_rare_members':m['n_rare_members'],'mia_all':m['tpr_all'],
                    'mia_rare':cleanfloat(m['tpr_rare']),'mia_used':m['tpr_score'],'mia_auc':m['auc'],
                    'aia_all':r['aia_all'],'aia_rare':r['aia_rare'],'aia_used':r['aia_floored'],
                    'aia_p_m':r['p_m'],'aia_p_c':r['p_c'],'phi':FLOORS[k]})
    dump_csv('team28_per_target.csv',per)

    # Ground truth: enumerate rare memberships of team28, check published score's perfect separation.
    r28=np.array([int(r['is_rare'])==1 for r in POOL[28]],dtype=bool)
    own28=np.array([int(r['pool_id']) for r in POOL[28]])
    rare_detail=[]; mi28,_=read_submission(28)
    for j in np.flatnonzero(r28):
        positive_targets=[k for k in ID if k!=28 and own28[j] in POOLS[k]]
        rare_detail.append({'record_id':int(j),'pool_id':int(own28[j]),'member_targets':positive_targets,
                            'score1_targets':[k for k in ID if k!=28 and float(mi28[int(j)][str(k)])>=1.0]})
    rare_labels=sum(len(v['member_targets']) for v in rare_detail)
    rare_targets=sorted(set(k for v in rare_detail for k in v['member_targets']))
    rare_mat_errors=0
    for v in rare_detail:
        for k in ID:
            if k==28:continue
            guess=float(mi28[v['record_id']][str(k)])
            truth=k in v['member_targets']
            rare_mat_errors += int((guess>0.5)!=truth)

    # Experiment 1: hold out target and choose attacker by mean AIA score on other target cohorts,
    # exclude each attacker's own target from training to avoid inflated self.
    lto=[]
    for k in ID:
        train={a:np.mean([ars[a,j]['aia_floored'] for j in ID if j not in (a,k)]) for a in ID if a!=k}
        best=max(train,key=lambda a:train[a]);actual=ars[best,k]['aia_floored']
        self28=ars[28,k]['aia_floored'] if k!=28 else None
        lto.append({'target':k,'selected_attacker':best,'train_avg_aia':float(train[best]),
                    'held_out_aia':actual,'team28_aia':self28})
    dump_csv('aia_leave_target_out.csv',lto)

    # Experiment 2: diagnostic ensemble (median, mean) of published submitted time predictions.
    # Ensemble is not a valid historical attack, evaluated ex post with ground truth.
    en=[]
    for k in ID:
        teams=[a for a in ID if a!=k and k in preds[a]]
        ids=sorted(int(r['challenge_row_id']) for r in TRUTHS[k])
        for how in ('median','mean'):
            p={}
            for j in ids:
                x=[preds[a][k][j] for a in teams if j in preds[a][k]]
                if x:p[j]=float(np.median(x) if how=='median' else np.mean(x))
            val=aia_eval(k,p)
            en.append({'target':k,'method':how,'n_attackers':len(teams), 'aia_power':val['aia_floored'],
                       'aia_all':val['aia_all'],'aia_rare':cleanfloat(val['aia_rare'])})
    dump_csv('aia_ensembles.csv',en)

    # Experiment 3: MIA FPR sweep for attacker 28, with official low_fpr shown separately.
    sweep=[]
    for fpr in (.005,.01,.02,.05):
        ms=[];rs=[]
        for k in ID:
            if k==28:continue
            scores=np.array([float(mi28[j][str(k)]) for j in range(1049)])
            obj=mia_eval(28,k,scores,fpr)
            ms.append(obj['tpr_all'])
            if np.isfinite(obj['tpr_rare']):rs.append(obj['tpr_rare'])
        sweep.append({'fpr':fpr,'mia_all_mean':float(np.mean(ms)), 'mia_rare_mean':float(np.mean(rs)),
                      'mia_rare_targets':len(rs)})
    dump_csv('team28_fpr_sweep.csv',sweep)

    # Compare AI attack vs top other teams on the same target, count ranks for m and a separately.
    rel=[]
    for k in ID:
        if k==28:continue
        rank_m=1+sum(mrs[a,k]['tpr_score']>mrs[28,k]['tpr_score']+1e-12 for a in ID if a not in (k,28))
        rank_a=1+sum(ars[a,k]['aia_floored']>ars[28,k]['aia_floored']+1e-12 for a in ID if a not in (k,28))
        rel.append({'target':k,'rank_mia':rank_m,'rank_aia':rank_a,'best_mia':max(mrs[a,k]['tpr_score'] for a in ID if a!=k),
                    'best_aia':max(ars[a,k]['aia_floored'] for a in ID if a!=k),
                    'team28_mia':mrs[28,k]['tpr_score'],'team28_aia':ars[28,k]['aia_floored']})
    dump_csv('team28_target_ranks.csv',rel)

    agg={
      'publication_date':'2026-10-26','cohorts':len(ID),'verified_mia_cells':len(mrs), 'verified_aia_cells':len(ars),
      'scorer_max_absolute_error':{k:max(v or [0]) for k,v in diffs.items()},
      'team28':{**by_att[28],
        'mean_tpr_all':float(np.mean([mrs[28,k]['tpr_all'] for k in ID if k!=28])),
        'mean_tpr_rare':float(np.mean([mrs[28,k]['tpr_rare'] for k in ID if k!=28 and np.isfinite(mrs[28,k]['tpr_rare'])])),
        'rare_candidates':int(r28.sum()),'rare_target_memberships':int(rare_labels),'rare_target_count':len(rare_targets),
        'rare_targets':rare_targets, 'rare_record_count_with_positive':sum(bool(v['member_targets']) for v in rare_detail),
        'rare_matrix_classification_errors':rare_mat_errors,
        'nonrare_mia_mean':float(np.mean([mrs[28,k]['tpr_score'] for k in ID if k!=28 and mrs[28,k]['n_rare_members']==0])),
        'rare_target_mia_mean':float(np.mean([mrs[28,k]['tpr_score'] for k in rare_targets])),
        'aia_member_accuracy_mean':float(np.mean([ars[28,k]['p_m'] for k in ID if k!=28])),
        'aia_control_accuracy_mean':float(np.mean([ars[28,k]['p_c'] for k in ID if k!=28])),
        'aia_floor_mean':float(np.mean([FLOORS[k] for k in ID if k!=28])),
        'mia_first_place_target_count':sum(x['rank_mia']==1 for x in rel),
        'aia_first_place_target_count':sum(x['rank_aia']==1 for x in rel),
        'positive_aia_targets':sum(ars[28,k]['aia_floored']>1e-12 for k in ID if k!=28),
      },
      'leaderboard_top5':[{'rank':i+1,'cohort':r['cohort'],'power':r['attack_power']} for i,r in enumerate(rows[:5])],
      'fpr_sweep':sweep,
      'aia_ensemble_by_method':{method:{'mean':float(np.mean([r['aia_power'] for r in en if r['method']==method and r['target']!=28])),
                   'median':float(np.median([r['aia_power'] for r in en if r['method']==method and r['target']!=28]))}
                  for method in ['median','mean']},
      'aia_leave_target_out':{'mean':float(np.mean([r['held_out_aia'] for r in lto if r['target']!=28])),
                  'selection_frequency':{str(a):sum(r['selected_attacker']==a for r in lto) for a in ID if any(r['selected_attacker']==a for r in lto)}},
      'team28_vs_competitors':{
        'margin_to_second':rows[0]['attack_power']-rows[1]['attack_power'],
        'top_other_mia':sorted([(a,by_att[a]['mia_power']) for a in ID if a!=28],key=lambda p:-p[1])[:3],
        'top_other_aia':sorted([(a,by_att[a]['aia_power']) for a in ID if a!=28],key=lambda p:-p[1])[:3],
      }
    }
    # Score-channel ablation: replace Team28 max(MIA_all, MIA_rare) with MIA_all only.
    # This is an arithmetic counterfactual for Team28, not a resubmitted attack or
    # an official competition reranking (other teams' scoring remains unchanged).
    mia_all28 = agg['team28']['mean_tpr_all']
    aia28 = by_att[28]['aia_power']
    all_only_attack = mia_all28 + aia28
    rarity_gain = by_att[28]['mia_power'] - mia_all28
    simulated_rank = 1+sum(r['attack_power'] > all_only_attack for r in rows if r['cohort'] != 28)
    agg['team28'].update({
        'mia_rare_increment_over_all_only': rarity_gain,
        'fraction_of_total_attack_power_from_mia_rare_increment': rarity_gain / by_att[28]['attack_power'],
        'all_only_channel_hypothetical_attack_power': all_only_attack,
        'all_only_channel_hypothetical_rank_others_unchanged': simulated_rank,
        'aia_power_rank': 1+sum(by_att[a]['aia_power'] > aia28 for a in ID if a != 28),
        'mia_power_rank': 1+sum(by_att[a]['mia_power'] > by_att[28]['mia_power'] for a in ID if a != 28),
    })
    (HERE/'experiment_results.json').write_text(json.dumps(agg,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    (HERE/'team28_rare_truth_audit.json').write_text(json.dumps(rare_detail,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(agg,ensure_ascii=False,indent=2))

if __name__=='__main__':run()