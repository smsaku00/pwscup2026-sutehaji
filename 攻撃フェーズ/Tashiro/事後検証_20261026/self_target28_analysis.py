#!/usr/bin/env python3
"""Cohort 28 (our defended cohort) AIA constant-time & observed MIA audit.

Use only 2026-10-26 official releases. No unpublished ground truth needed.
Execute run_all.py first to populate official_table_data.json, 正解データ/,
攻撃提出物/. AIA and MIA scores are reproduced using posthoc_analysis.py.
All comparisons of optimally selected constants use truth after publication.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np
from posthoc_analysis import (
    HERE, ID, TRUTHS, POOL, POOLS, FLOORS, TABLE,
    aia_eval, mia_eval, read_submission, interpolated_roc,
)

TARGET=28
OUT=HERE/'self_target28_results'
OUT.mkdir(exist_ok=True)
TRUTH=TRUTHS[TARGET]
FLOOR=FLOORS[TARGET]
TRUTH_IDS=[int(r['challenge_row_id']) for r in TRUTH]
TIMES=np.asarray([float(r['time']) for r in TRUTH])
MEMBER=np.asarray([r['label']=='member' for r in TRUTH])
RARE=np.asarray([r['is_rare']=='1' for r in TRUTH])


def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def make_const(t):
    p={i:float(t) for i in TRUTH_IDS}
    return aia_eval(TARGET,p)


def score_grid():
    # Piecewise-constant success indicators change only at time_i +/- 0.5.
    # Check boundary and interval interiors, including endpoints [0,10].
    cuts=sorted(set([0.,10.]+[max(0.,min(10.,x+d)) for x in TIMES for d in (-.5,.5)]))
    grid=sorted(set(cuts+[(x+y)/2 for x,y in zip(cuts,cuts[1:])]))
    vals=[]
    for t in grid:
        r=make_const(t)
        vals.append({'time_hat':t,'aia_raw':r['aia_all'],'aia_rare':r['aia_rare'],
                     'aia_score':r['aia_floored'],'member_success':r['p_m'],
                     'control_success':r['p_c']})
    vals.sort(key=lambda r:(-r['aia_score'],abs(r['time_hat']-9.5)))
    write_csv('aia28_constant_grid.csv',vals)
    return vals


def roc_first_fp(scores, truth):
    order=np.argsort(-scores,kind='mergesort')
    y=truth[order]
    first_pos=np.flatnonzero(y)
    fps_before_first=int(first_pos[0]) if len(first_pos) else None
    return {'top10_members':int(y[:10].sum()), 'top20_members':int(y[:20].sum()),
            'fp_before_first_member':fps_before_first}


def exact_hypergeom_right_tail(n,n_success,draws,observed):
    """One-sided permutation diagnostic (exchangeability of member/control)."""
    total=math.comb(n,draws)
    return sum(math.comb(n_success,k)*math.comb(n-n_success,draws-k)
               for k in range(observed,min(n_success,draws)+1)
               if 0<=draws-k<=n-n_success)/total


def main():
    assert len(TRUTH)==326 and len(set(TRUTH_IDS))==326
    assert int(MEMBER.sum())==int((~MEMBER).sum())==163
    # Observe only member/control labels and true time on our own AIA challenge.
    constants=[]
    for t in [0,2.5,5,7.5,9,9.5,9.6,9.7,10]:
        v=make_const(t)
        constants.append({'time_hat':t,'aia_all_before_floor':v['aia_all'],
                          'aia_rare_before_floor':v['aia_rare'],
                          'phi':FLOOR,'aia_after_floor':v['aia_floored'],
                          'p_member':v['p_m'],'p_control':v['p_c'],
                          'n_member_correct':round(v['p_m']*int(MEMBER.sum())),
                          'n_control_correct':round(v['p_c']*int((~MEMBER).sum()))})
    write_csv('aia28_fixed_times.csv',constants)
    grid=score_grid()
    best_value=grid[0]['aia_score']
    best_pts=[v['time_hat'] for v in grid if abs(v['aia_score']-best_value)<1e-12]

    table_a=[];table_m=[];sweep=[];aia_shifts=[]
    for attacker in ID:
        if attacker==TARGET:continue
        mi,ai=read_submission(attacker)
        a=aia_eval(TARGET,ai[TARGET])
        pred_vals=np.asarray(list(ai[TARGET].values()))
        shifted={j:(9.5 if v==10.0 else v) for j,v in ai[TARGET].items()}
        shifted_ge={j:(9.5 if v>=9.95 else v) for j,v in ai[TARGET].items()}
        ash=aia_eval(TARGET,shifted)
        ash_ge=aia_eval(TARGET,shifted_ge)
        aia_shifts.append({'attacker':attacker,'actual_aia':a['aia_floored'],
                           'replace_exact_10_with_9p5':ash['aia_floored'],
                           'delta_exact':ash['aia_floored']-a['aia_floored'],
                           'replace_ge9p95_with_9p5':ash_ge['aia_floored'],
                           'delta_ge':ash_ge['aia_floored']-a['aia_floored'],
                           'n_exact_10':sum(v==10.0 for v in ai[TARGET].values())})
        table_a.append({'attacker':attacker, 'score_phi_subtracted':a['aia_floored'],
                        'R_all':a['aia_all'], 'R_rare':a['aia_rare'],
                        'member_success':a['p_m'], 'control_success':a['p_c'],
                        'n_member_correct':round(a['p_m']*163),
                        'n_control_correct':round(a['p_c']*163),
                        'n_distinct_prediction_values':len(np.unique(pred_vals)),
                        'n_prediction_9p5':int(np.isclose(pred_vals,9.5,atol=1e-9).sum()),
                        'n_prediction_10':int(np.isclose(pred_vals,10,atol=1e-9).sum())})
        scores=np.asarray([float(mi[j][str(TARGET)]) for j in range(1049)])
        one=mia_eval(attacker,TARGET,scores)
        truths=np.isin(np.asarray([int(r['pool_id']) for r in POOL[attacker]]),
                       np.asarray(list(POOLS[TARGET]))).astype(int)
        top=roc_first_fp(scores,truths)
        table_m.append({'attacker':attacker, 'tpr_all_fpr1pct':one['tpr_all'],
                        'tpr_rare_fpr1pct':one['tpr_rare'] if math.isfinite(one['tpr_rare']) else '',
                        'official_mia_used':one['tpr_score'], 'auc':one['auc'],
                        'n_members':one['n_members'], 'n_rare_members':one['n_rare_members'],
                        'n_distinct_confidences':len(np.unique(scores)),**top})
        for fpr in [0.005,0.01,0.02,0.05,0.1]:
            r=mia_eval(attacker,TARGET,scores,fpr)
            sweep.append({'attacker':attacker,'fpr':fpr,'tpr_all':r['tpr_all'],
                          'tpr_rare':r['tpr_rare'] if math.isfinite(r['tpr_rare']) else '',
                          'tpr_used':r['tpr_score']})
        # Crosscheck against published score for the defended cohort.
        x=next(z for z in TABLE['攻撃_標的別_AIA'] if z['攻撃者']==attacker and z['標的']==TARGET)
        y=next(z for z in TABLE['攻撃_標的別_MIA'] if z['攻撃者']==attacker and z['標的']==TARGET)
        assert abs(float(x['R（全体）'])-a['aia_all'])<1e-10
        assert abs(float(y['TPR（全体）'])-one['tpr_all'])<1e-10
    table_a.sort(key=lambda r:(-r['score_phi_subtracted'],r['attacker']))
    table_m.sort(key=lambda r:(-r['tpr_all_fpr1pct'],r['attacker']))
    write_csv('aia28_official_attackers.csv',table_a)
    aia_shifts.sort(key=lambda r:-r['replace_exact_10_with_9p5'])
    write_csv('aia28_high_time_shift.csv',aia_shifts)
    write_csv('mia28_official_attackers.csv',table_m)
    write_csv('mia28_fpr_sweep.csv',sweep)

    const95=make_const(9.5)
    count_success=(np.abs(TIMES-9.5)<=.5)
    successes_member=int((count_success & MEMBER).sum())
    successes_control=int((count_success & ~MEMBER).sum())
    pnull=exact_hypergeom_right_tail(len(TRUTH),int(count_success.sum()),163,successes_member)
    rawM={r['attacker']:r['official_mia_used'] for r in table_m}
    official_def=next(r for r in TABLE['総括'] if int(r['コホート番号'])==TARGET)
    mia_risk=float(official_def['MIA 成分']);aia_risk=float(official_def['AIA 成分'])
    report={
     'cohort':TARGET,'challenge_rows':len(TRUTH),'member_rows':int(MEMBER.sum()),
     'control_rows':int((~MEMBER).sum()),'rare_member_rows':int((MEMBER&RARE).sum()),
     'rare_control_rows':int(((~MEMBER)&RARE).sum()),
     'phi':FLOOR,'constant_9p5':const95,
     'constant_9p5_exact_member_hits':successes_member,
     'constant_9p5_exact_control_hits':successes_control,
     'constant_9p5_one_sided_label_permutation_pvalue':pnull,
     'constant_10':make_const(10),
     'truth_selected_optimal_constant_score':best_value,
     'truth_selected_best_grid_time_points':best_pts,
     'observed_best_aia_attacker':table_a[0],
     'observed_median_aia_score':float(np.median([r['score_phi_subtracted'] for r in table_a])),
     'observed_mean_aia_score':float(np.mean([r['score_phi_subtracted'] for r in table_a])),
     'aia_attacker_count_beating_fixed95':sum(r['score_phi_subtracted']>const95['aia_floored']+1e-12 for r in table_a),
     'aia_attacker_count_above_zero':sum(r['score_phi_subtracted']>1e-12 for r in table_a),
     'aia_any_rare_channel_dominates':any(r['R_rare']>r['R_all']+1e-12 for r in table_a),
     'aia_exact_10_to_9p5_best':aia_shifts[0],
     'aia_exact_10_to_9p5_num_improved':sum(r['delta_exact']>1e-12 for r in aia_shifts),
     'aia_exact_10_to_9p5_num_worse':sum(r['delta_exact']<-1e-12 for r in aia_shifts),
     'mia_best_all_attacker':table_m[0],
     'mia_nonzero_all_attacker_count':sum(r['tpr_all_fpr1pct']>0 for r in table_m),
     'mia_nonzero_rare_attacker_count':sum(float(r['tpr_rare_fpr1pct'])>0 for r in table_m if r['tpr_rare_fpr1pct']!=''),
     'mia_mean_official_raw_tpr_max':float(np.mean(list(rawM.values()))),
     'mia_mean_all_only':float(np.mean([r['tpr_all_fpr1pct'] for r in table_m])),
     'mia_rare_only_increment_in_mean':float(np.mean([r['official_mia_used']-r['tpr_all_fpr1pct'] for r in table_m])),
     'mia_rare_two_attacker_contribution_to_raw_mean':float((rawM[6]+rawM[16])/22),
     'defense_mia_component':mia_risk, 'defense_aia_component':aia_risk,
     'official_anonymity':float(official_def['Anon']),
     'check_anonymity_equals_one_minus_max':abs(1-max(mia_risk,aia_risk)-float(official_def['Anon']))<1e-10,
     'mia_auc_best_attacker':max(table_m,key=lambda r:r['auc']),
     'mia_all_tpr_at_fpr':{str(f):{'max':float(max(r['tpr_all'] for r in sweep if r['fpr']==f)),
                                       'mean':float(np.mean([r['tpr_all'] for r in sweep if r['fpr']==f]))}
                                     for f in [0.005,0.01,0.02,0.05,0.1]},
    }
    (OUT/'self_target28_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()