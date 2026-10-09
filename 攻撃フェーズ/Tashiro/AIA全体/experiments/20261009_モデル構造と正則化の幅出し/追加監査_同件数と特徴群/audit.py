#!/usr/bin/env python3
"""PWS CUP 2026: post-hoc audit of frozen limited-change model decision.

Read saved results from the 2026-10-09 model-width study. Never learn from or
change target truth; all truth here is used only to score an already selected
prediction. No CodaBench submission or modification of original artifacts.

One new counterfactual is a *diagnostic* matched-action-count HGB subset: in
cohort k, keep only the top K reference-model number of edits from the HGB's
already saved, positive-gain candidates. This is not an independent predictive
model and does not estimate what HGB would have selected under re-calibration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

GROUPS = {
    'strong_row_preservation_1_2':[1,2],
    'column_preserved_row_disrupted_19_24':[19,24],
    'column_and_row_bg_matched_18_29':[18,29],
    'heavy_discretization_23':[23],
    'other13_not_assumed_homogeneous':[3,4,6,7,8,9,10,12,13,17,20,22,26],
}
SCOPE={
    'nonself20':sorted(set(sum(GROUPS.values(),[]))),
    'processed18':sorted(set(sum(GROUPS.values(),[]))-{1,2}),
}

def score(m:int,c:int)->float:
    assert 0 <= m <= 163 and 0 <= c <= 163
    return max(0.,min(1.,(m-c)/max(163-c,1)))

def wins(x:pd.Series,tol:float=1e-12)->dict:
    return dict(wins=int((x>tol).sum()),losses=int((x< -tol).sum()),ties=int((abs(x)<=tol).sum()))

def paired_boot(x:np.ndarray,seed:int=20261009,nboot:int=30000)->tuple[float,float,float]:
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(x),size=(nboot,len(x)))
    y=x[idx].mean(axis=1)
    return float(x.mean()),float(np.quantile(y,.025)),float(np.quantile(y,.975))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=Path(__file__).resolve().parent.parent/'model_results');ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    D=pd.read_csv(a.input/'limited_model_per_team.csv'); E=pd.read_csv(a.input/'limited_model_edited_rows.csv'); P=pd.read_csv(a.input/'per_team.csv')
    assert set(D.decision_model)>= {'reference','HGB_shallow'}
    assert D.team.nunique()==21 and D.groupby('team').decision_model.nunique().eq(6).all()
    assert D.groupby('team').baseline_r_all.nunique().eq(1).all()
    assert D.groupby(['team','decision_model']).size().eq(1).all()
    assert set(sum(GROUPS.values(),[]))==set(D.team)-{5}
    exact=[];grouprows=[];budget=[]
    D['improve_baseline']=D.r_all-D.baseline_r_all
    REF=D[D.decision_model.eq('reference')].set_index('team')
    H=D[D.decision_model.eq('HGB_shallow')].set_index('team')
    for k in D.team.unique():
        ref=REF.loc[k]; h=H.loc[k]
        e=E[(E.team==k)&(E.decision_model=='HGB_shallow')].sort_values(['expected_gain','row_id'],ascending=[False,True]);
        er=E[(E.team==k)&(E.decision_model=='reference')]
        assert len(e)==int(h.changed) and len(er)==int(ref.changed)
        m0=int(ref.member_hits+ref.member_lost);c0=int(ref.control_hits+ref.control_hit_removed)
        assert np.isclose(score(m0,c0),ref.baseline_r_all,atol=1e-12)
        assert np.isclose(score(m0-int(h.member_lost),c0-int(h.control_hit_removed)),h.r_all,atol=1e-12)
        assert e.post_hit.eq(False).all() and er.post_hit.eq(False).all()
        check_m=int((e.member_bool_for_posthoc_only.astype(int).eq(1) & e.pre_hit.astype(bool)).sum());check_c=int((e.member_bool_for_posthoc_only.astype(int).eq(0) & e.pre_hit.astype(bool)).sum());
        assert check_m==h.member_lost and check_c==h.control_hit_removed
        eq=e.head(int(ref.changed))
        lm=int((eq.member_bool_for_posthoc_only.astype(int).eq(1)&eq.pre_hit.astype(bool)).sum());lc=int((eq.member_bool_for_posthoc_only.astype(int).eq(0)&eq.pre_hit.astype(bool)).sum())
        budget.append(dict(team=int(k),ref_edited=int(ref.changed),hgb_edited_full=int(h.changed),hgb_edited_matched=len(eq),reference_r=float(ref.r_all),hgb_full_r=float(h.r_all),hgb_equal_action_r=score(m0-lm,c0-lc),baseline_r=float(ref.baseline_r_all),reference_member_lost=int(ref.member_lost),reference_control_hit_removed=int(ref.control_hit_removed),hgb_equal_member_lost=lm,hgb_equal_control_hit_removed=lc))
    B=pd.DataFrame(budget).sort_values('team')
    B['full_hgb_minus_reference']=B.hgb_full_r-B.reference_r
    B['equal_hgb_minus_reference']=B.hgb_equal_action_r-B.reference_r
    B.to_csv(a.out/'equal_action_by_team.csv',index=False,float_format='%.15g')
    details=[]
    for scope,ts in SCOPE.items():
        d=B[B.team.isin(ts)];assert len(d)==len(ts)
        for label,column in [('baseline','baseline_r'),('reference','reference_r'),('HGB_full','hgb_full_r'),('HGB_equal_actions','hgb_equal_action_r')]:
            diff=d[column]-d.reference_r
            v,w,z=paired_boot(diff.to_numpy(dtype=float),seed=20261009)
            base=d[column]-d.baseline_r
            row=dict(scope=scope,method=label,teams=len(d),r_all_mean=float(d[column].mean()),delta_vs_reference_mean=float(v),delta_vs_baseline_mean=float(base.mean()),ci95_diff_vs_reference_lo=w,ci95_diff_vs_reference_hi=z,**wins(diff))
            if label=='HGB_full':row.update(changes=int(d.hgb_edited_full.sum()),member_lost=int(H.loc[ts].member_lost.sum()),control_hit_removed=int(H.loc[ts].control_hit_removed.sum()))
            if label=='HGB_equal_actions':row.update(changes=int(d.hgb_edited_matched.sum()),member_lost=int(d.hgb_equal_member_lost.sum()),control_hit_removed=int(d.hgb_equal_control_hit_removed.sum()))
            if label=='reference':row.update(changes=int(d.ref_edited.sum()),member_lost=int(d.reference_member_lost.sum()),control_hit_removed=int(d.reference_control_hit_removed.sum()))
            details.append(row)
    pd.DataFrame(details).to_csv(a.out/'matched_action_summary.csv',index=False,float_format='%.15g')
    for name,ts in GROUPS.items():
        group=B[B.team.isin(ts)]; diff=group.full_hgb_minus_reference
        grouprows.append(dict(group=name,group_definition_prior=True,n_teams=len(ts),teams=';'.join(map(str,ts)),logistic_r_all=float(group.reference_r.mean()),hgb_r_all=float(group.hgb_full_r.mean()),delta_hgb_minus_logistic=float(diff.mean()),**wins(diff),logistic_changes=int(group.ref_edited.sum()),hgb_changes=int(group.hgb_edited_full.sum()),hgb_member_lost=int(group.team.map(H.member_lost).sum()),hgb_control_removed=int(group.team.map(H.control_hit_removed).sum()),hgb_equal_actions_r_all=float(group.hgb_equal_action_r.mean())))
    pd.DataFrame(grouprows).to_csv(a.out/'public_group_comparison.csv',index=False,float_format='%.15g')
    for k in SCOPE['nonself20']:
        tmp=D[D.team.eq(k)].set_index('decision_model')
        exact.append(dict(team=k,reference_r=float(tmp.loc['reference','r_all']),HGB_shallow_r=float(tmp.loc['HGB_shallow','r_all']),HGB_vs_reference=float(tmp.loc['HGB_shallow','r_all']-tmp.loc['reference','r_all']),reference_changes=int(tmp.loc['reference','changed']),HGB_changes=int(tmp.loc['HGB_shallow','changed'])))
    pd.DataFrame(exact).to_csv(a.out/'comparison_by_team.csv',index=False,float_format='%.15g')
    # Aggregate selection overlap (memberships and outcomes are test diagnostics only).
    oo=[]
    for k in SCOPE['nonself20']:
        rf=set(E[(E.team==k)&(E.decision_model=='reference')].row_id)
        hg=set(E[(E.team==k)&(E.decision_model=='HGB_shallow')].row_id)
        oo.append(dict(team=k,reference_count=len(rf),hgb_count=len(hg),overlap=len(rf&hg)))
    pd.DataFrame(oo).to_csv(a.out/'candidate_overlap.csv',index=False)
    # Hold both feature packs fixed. The original limited-change decision swap
    # compared c_bg_local LR with c_bg HGB; that was not purely a model-only
    # comparison. This separate diagnostic isolates estimator families under
    # identical public inputs and a common top-75 selection rule.
    comparable=P[P.features.isin(['c_bg','c_bg_local']) & P.model.isin(['lr_c004','hgb_shallow'])]
    assert len(comparable)==21*2*2
    rows=[]; same_group_rows=[]
    for feature in ['c_bg','c_bg_local']:
        F=comparable[comparable.features==feature]
        lr=F[F.model=='lr_c004'].set_index('team')
        hb=F[F.model=='hgb_shallow'].set_index('team')
        assert set(lr.index)==set(hb.index)==set(D.team)
        for scope,ts in SCOPE.items():
            delta=(hb.loc[ts].r_all-lr.loc[ts].r_all).to_numpy(float)
            auc=(hb.loc[ts].auc-lr.loc[ts].auc).to_numpy(float)
            mu,lo,hi=paired_boot(delta,seed=20261009)
            amu,alo,ahi=paired_boot(auc,seed=20261010)
            rows.append(dict(scope=scope,features=feature,n_teams=len(ts),logistic_r=float(lr.loc[ts].r_all.mean()),hgb_r=float(hb.loc[ts].r_all.mean()),hgb_minus_logistic=float(mu),r_diff_ci_lo=lo,r_diff_ci_hi=hi,**wins(pd.Series(delta)),logistic_auc=float(lr.loc[ts].auc.mean()),hgb_auc=float(hb.loc[ts].auc.mean()),auc_diff=float(amu),auc_diff_ci_lo=alo,auc_diff_ci_hi=ahi))
        for name,ts in GROUPS.items():
            delta=hb.loc[ts].r_all-lr.loc[ts].r_all
            same_group_rows.append(dict(features=feature,group=name,teams=';'.join(map(str,ts)),n_teams=len(ts),logistic_r=float(lr.loc[ts].r_all.mean()),hgb_r=float(hb.loc[ts].r_all.mean()),hgb_minus_logistic=float(delta.mean()),**wins(delta)))
    pd.DataFrame(rows).to_csv(a.out/'same_features_global_ranking.csv',index=False,float_format='%.15g')
    pd.DataFrame(same_group_rows).to_csv(a.out/'same_features_by_public_group.csv',index=False,float_format='%.15g')
    metrics={
      'input_tables':['limited_model_per_team.csv','limited_model_edited_rows.csv','per_team.csv'],
      'analysis':'fixed-protocol diagnostics; not fresh held-out validation',
      'group_source':'2026-10-08 publicly observable C feature groupings, defined after earlier outcome inspections',
      'actions':'HGB first K chosen edits where K is reference edits for same team, ranked by frozen expected_gain',
      'hidden_truth_in_prediction':False,
      'group_inference_warning':'1-2 cohorts/groups insufficient for new routing',
      'tests':'reconstructed baseline, full HGB R, and per-edit hit counts agree exactly with saved results',
      'HGB_avg_actions_per_nonself':float(B[B.team.ne(5)].hgb_edited_full.mean()),
      'Logistic_avg_actions_per_nonself':float(B[B.team.ne(5)].ref_edited.mean()),
      'zero_reference_action_teams_nonself':int((B[B.team.ne(5)].ref_edited==0).sum()),
      'important_feature_confound':'reference decision model=c_bg_local LR; HGB_shallow decision model=c_bg HGB',
      'same_feature_diagnostic':'global top75 selection separately reproduced by saved per_team.csv; compare c_bg and c_bg_local separately',
    }
    (a.out/'verification.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('verification: PASS, 21 cohort reconstruction, edits and hit counts')
    print('MATCHED ACTION SUMMARY:')
    print(pd.DataFrame(details)[['scope','method','r_all_mean','delta_vs_reference_mean','ci95_diff_vs_reference_lo','ci95_diff_vs_reference_hi','wins','losses','ties','changes']].fillna('-').to_string(index=False))
    print('PUBLIC GROUP SUMMARY:')
    print(pd.DataFrame(grouprows).to_string(index=False))
    print('SAME FEATURES, GLOBAL RANKING:')
    print(pd.DataFrame(rows).to_string(index=False))
    print('ZERO REFERENCE ACTION TEAMS:',metrics['zero_reference_action_teams_nonself'])
if __name__=='__main__':main()
