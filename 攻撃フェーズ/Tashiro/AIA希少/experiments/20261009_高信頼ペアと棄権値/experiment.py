#!/usr/bin/env python3
"""High-confidence candidate-pair audit. No contest submission or network writes.

Official mode reads the existing repository. Mock mode uses the archived five
main-B / preliminary-background panels; these are NOT official prelim cases.
Predictions never receive membership truth or pair_id. The official rare model
uses is_rare only on other folds, with the existing HGB settings.
"""
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import scipy
import sklearn

VERSION = '20261009-pair-audit-v1'
NUM = ['age', 'SBP', 'TG', 'HDL', 'ALT']
FOUR = ['SBP', 'TG', 'HDL', 'ALT']
QI = NUM + ['sex', 'prefecture']
SEED = 20261007  # existing rare classifier seed
BOOT_SEED = 20261009
HIGH_GAP = 0.25  # rank-unit gap, not a calibrated probability
HIT_TIME = 9.5
POSITIVE_FRACTION = 0.75
METHODS = ('near4', 'col4_simple', 'col2_simple')
POLICIES = ('baseline', 'force_half', 'visible_pair', 'stable_rare_pair', 'combined_pair')
HASHES: dict[str, str] = {}


def read_csv(path: Path) -> pd.DataFrame:
    HASHES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path, float_precision='round_trip')


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def transformed(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    a = frame[cols].to_numpy(float).copy()
    if not np.isfinite(a).all():
        raise ValueError('Non-finite features')
    for name in ['TG', 'ALT']:
        if name in cols:
            j = cols.index(name)
            if (a[:, j] <= -1).any():
                raise ValueError('log1p domain')
            a[:, j] = np.log1p(a[:, j])
    return a


def rare_features(frame: pd.DataFrame) -> np.ndarray:
    # Identical features/model parameters to 希少候補の学習.py at e0bf1b74.
    return np.column_stack([transformed(frame, NUM), frame.sex.to_numpy(float)])


def rare_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=140, max_leaf_nodes=15,
        min_samples_leaf=35, l2_regularization=5, random_state=SEED)


def top_indices(score: np.ndarray, ids: np.ndarray, n: int) -> np.ndarray:
    if not np.isfinite(score).all():
        raise ValueError('Non-finite score')
    return np.lexsort((ids, -score))[:n]


def nearest_1d(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if not len(ref):
        raise ValueError('Empty same-sex reference')
    a = np.sort(ref)
    pos = np.searchsorted(a, q)
    left = np.abs(q - a[np.maximum(pos - 1, 0)])
    right = np.abs(q - a[np.minimum(pos, len(a) - 1)])
    return np.minimum(left, right)


def evidence(q: pd.DataFrame, c: pd.DataFrame, bg: pd.DataFrame) -> dict[str, np.ndarray]:
    """New fixed baselines, not exact replications of P01/all4/top2."""
    x, y, b = [transformed(f, FOUR) for f in (q, c, bg)]
    scale = np.maximum(np.quantile(b, .75, axis=0)-np.quantile(b, .25, axis=0), 1e-5)
    x, y, b = x/scale, y/scale, b/scale
    scores = {m: np.empty(len(q)) for m in METHODS}
    for sex in sorted(q.sex.unique()):
        qi = np.flatnonzero(q.sex.to_numpy() == sex)
        cy = y[c.sex.to_numpy() == sex]
        by = b[bg.sex.to_numpy() == sex]
        if len(cy) == 0 or len(by) == 0:
            raise ValueError('Same-sex reference missing; no implicit fallback')
        scores['near4'][qi] = -np.sqrt(cdist(x[qi], cy, metric='sqeuclidean').min(axis=1))
        logs = np.column_stack([np.log((nearest_1d(x[qi, j], by[:, j])+.001) /
            (nearest_1d(x[qi, j], cy[:, j])+.001)) for j in range(4)])
        scores['col4_simple'][qi] = logs.mean(axis=1)
        scores['col2_simple'][qi] = np.sort(logs, axis=1)[:, -2:].mean(axis=1)
    return scores


def within_sex_ranks(q: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ranks = {m: np.empty(len(q)) for m in scores}
    for sex in sorted(q.sex.unique()):
        ind = np.flatnonzero(q.sex.to_numpy() == sex)
        for m, s in scores.items():
            ranks[m][ind] = (rankdata(s[ind], method='average')-.5)/len(ind)
    return ranks


def baseline_mask(q: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    chosen = np.zeros(len(q), bool)
    ids = q.challenge_row_id.to_numpy(int)
    for sex in sorted(q.sex.unique()):
        ind = np.flatnonzero(q.sex.to_numpy() == sex)
        keep = top_indices(score[ind], ids[ind], math.ceil(POSITIVE_FRACTION*len(ind)))
        chosen[ind[keep]] = True
    return chosen


def group_indices(q: pd.DataFrame, selected: np.ndarray, width: int) -> dict[tuple, np.ndarray]:
    if width <= 0:
        raise ValueError('Nonpositive age-band width')
    groups: dict[tuple, list[int]] = {}
    for i in selected:
        r = q.iloc[int(i)]
        key = (str(r.prefecture), int(r.sex), int(r.age)//width)
        groups.setdefault(key, []).append(int(i))
    return {k: np.asarray(v, int) for k, v in groups.items()}


def candidates(q: pd.DataFrame, prob: np.ndarray, width: int) -> list[dict]:
    ids = q.challenge_row_id.to_numpy(int)
    gfull = group_indices(q, np.arange(len(q)), width)
    g26 = group_indices(q, top_indices(prob, ids, 26), width)
    g52 = group_indices(q, top_indices(prob, ids, 52), width)
    out = []
    for key, ind in gfull.items():
        if len(ind) == 2:
            out.append(dict(source='visible', key=key, a=int(ind[0]), b=int(ind[1])))
    for key, ind in g26.items():
        # Both endpoints must remain the ONLY two rare candidates in their
        # block when the candidate cutoff expands from 26 to 52.
        if len(ind) == 2 and len(g52.get(key, [])) == 2 and set(ind) == set(g52[key]):
            out.append(dict(source='rare_stable', key=key, a=int(ind[0]), b=int(ind[1])))
    return out


def predict_masks(q: pd.DataFrame, scores: dict[str, np.ndarray], prob: np.ndarray,
                  width: int) -> tuple[dict[tuple[str, str], np.ndarray], list[dict]]:
    """No truth, pair_id, onset or is_rare argument is accepted."""
    if any(c not in QI+['challenge_row_id'] for c in q.columns):
        raise ValueError('Inference Q must contain public columns only')
    rank = within_sex_ranks(q, scores)
    masks = {}
    audits = []
    for method in METHODS:
        base = baseline_mask(q, scores[method])
        for policy in POLICIES:
            masks[method, policy] = base.copy()
        # Deliberately aggressive comparator, not the recommended policy.
        force = np.zeros(len(q), bool)
        extra = []
        ids = q.challenge_row_id.to_numpy(int)
        for ind in group_indices(q, np.arange(len(q)), width).values():
            order = ind[top_indices(scores[method][ind], ids[ind], len(ind))]
            force[order[:len(ind)//2]] = True
            extra.extend(order[len(ind)//2:])
        need = len(q)//2-int(force.sum())
        extra = np.asarray(extra, int)
        if need:
            force[extra[top_indices(scores[method][extra], ids[extra], need)]] = True
        masks[method, 'force_half'] = force
        proposals: dict[str, dict[int, bool]] = {p: {} for p in POLICIES[2:]}
        conflicts: dict[str, set[int]] = {p: set() for p in POLICIES[2:]}
        for pair in candidates(q, prob, width):
            a, b = pair['a'], pair['b']
            diff = rank[method][a]-rank[method][b]
            gap = abs(float(diff))
            direction = np.sign(diff)
            agree = bool(direction != 0 and
                np.sign(rank['near4'][a]-rank['near4'][b]) == direction and
                np.sign(rank['col4_simple'][a]-rank['col4_simple'][b]) == direction)
            passed = bool(gap >= HIGH_GAP and agree)
            hi, lo = (a, b) if diff >= 0 else (b, a)
            audits.append(dict(method=method, width=width, source=pair['source'],
                key='|'.join(map(str, pair['key'])), a=int(ids[a]), b=int(ids[b]),
                high=int(ids[hi]), low=int(ids[lo]), rank_gap=gap,
                agree=agree, passed=passed, rare_prob_a=float(prob[a]), rare_prob_b=float(prob[b])))
            if not passed:
                continue
            targets = ['combined_pair', 'visible_pair' if pair['source']=='visible' else 'stable_rare_pair']
            for policy in targets:
                for i, value in [(hi, True), (lo, False)]:
                    if i in proposals[policy] and proposals[policy][i] != value:
                        conflicts[policy].add(i)
                    proposals[policy][i] = value
        for policy, changes in proposals.items():
            for i, value in changes.items():
                if i not in conflicts[policy]:
                    masks[method, policy][i] = value
    return masks, audits


def score_prediction(pred: np.ndarray, truth: pd.DataFrame) -> dict:
    pred = np.asarray(pred, float)
    if pred.shape != (len(truth),) or not np.isfinite(pred).all() or (pred<0).any():
        raise ValueError('Invalid prediction')
    m = truth.label.eq('member').to_numpy()
    c = truth.label.eq('control').to_numpy()
    if not (m|c).all() or m.sum()!=163 or c.sum()!=163:
        raise ValueError('Expected 163 member + 163 control')
    t = truth.time.to_numpy(float)
    if not np.isfinite(t).all() or (t<0).any() or (t>10).any():
        raise ValueError('Out-of-range truth time')
    hit = np.abs(pred-t) <= .5
    hm, hc = int(hit[m].sum()), int(hit[c].sum())
    pm, pc = hm/163, hc/163
    r_all = float(np.clip((pm-pc)/max(1-pc, 1e-9), 0, 1))
    out = dict(hit_m=hm, hit_c=hc, p_m=pm, p_c=pc, r_all=r_all,
        hit_m_rare=None, hit_c_rare=None, r_rare=None, r_max=None, rare_extra=None)
    if 'is_rare' in truth and truth.is_rare.isin([0,1]).all():
        rare = truth.is_rare.eq(1).to_numpy()
        if (m&rare).sum()!=13 or (c&rare).sum()!=13:
            raise ValueError('Official rare sizes must be 13+13')
        hmr, hcr = int(hit[m&rare].sum()), int(hit[c&rare].sum())
        pmr, pcr = hmr/13, hcr/13
        se = math.sqrt((pmr*(1-pmr)+pcr*(1-pcr))/13)
        rr = float(np.clip((pmr-pcr-1.64*se)/max(1-pcr,1e-9),0,1))
        out.update(hit_m_rare=hmr, hit_c_rare=hcr, r_rare=rr,
                   r_max=max(r_all,rr), rare_extra=max(r_all,rr)-r_all)
    elif 'member_true_rare_control_unknown' in truth:
        known_mr = m & truth.member_true_rare_control_unknown.eq(1).to_numpy()
        out['hit_m_rare'] = int(hit[known_mr].sum())
    return out


@dataclass
class Case:
    name: str
    team: int
    panel: int
    q: pd.DataFrame
    c: pd.DataFrame
    truth: pd.DataFrame
    pairmap: pd.DataFrame | None = None
    phi_mock: float | None = None
    control_case: bool = False


def check_case(case: Case) -> None:
    if len(case.q)!=326 or len(case.c)!=1049 or not case.q.challenge_row_id.is_unique:
        raise ValueError('Shape/id mismatch')
    if not np.array_equal(case.q.challenge_row_id.to_numpy(), case.truth.challenge_row_id.to_numpy()):
        raise ValueError('Q/truth id misalignment')


def official_inputs(repo: Path) -> tuple[list[Case], pd.DataFrame, dict]:
    root = repo/'公式データ/予備戦'
    bg = read_csv(root/'PWSCup2026_prelim_common/A_bg.csv')
    case_list=[]
    for p in sorted((root/'結果/正解データ').glob('aia_truth_*.csv'),key=lambda p:int(p.stem.split('_')[-1])):
        k=int(p.stem.split('_')[-1])
        q=read_csv(root/f'PWSCup2026_prelim_attack/aia_challenge_{k}.csv').sort_values('challenge_row_id').reset_index(drop=True)
        c=read_csv(root/f'PWSCup2026_prelim_attack/C_{k}.csv')
        t=read_csv(p).sort_values('challenge_row_id').reset_index(drop=True)
        case=Case(f'prelim_{k}',k,0,q,c,t)
        check_case(case)
        case_list.append(case)
    if len(case_list)!=21:
        raise ValueError(f'Official mode expects all 21 cases, found {len(case_list)}')
    X=np.concatenate([rare_features(x.q) for x in case_list])
    yr=np.concatenate([x.truth.is_rare.to_numpy(bool) for x in case_list])
    groups=np.concatenate([np.full(len(x.q),x.team) for x in case_list])
    probabilities=np.empty(len(X));folds=[]
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=7).split(X,yr,groups)):
        tr_groups=sorted(map(int,np.unique(groups[tr])))
        te_groups=sorted(map(int,np.unique(groups[te])))
        assert not set(tr_groups)&set(te_groups)
        model=rare_model().fit(X[tr],yr[tr])
        probabilities[te]=model.predict_proba(X[te])[:,1]
        folds.append(dict(fold=fold,train=tr_groups,test=te_groups))
    prob_by_name={c.name: probabilities[i*326:(i+1)*326] for i,c in enumerate(case_list)}
    return case_list,bg,dict(probabilities=prob_by_name,folds=folds,
        rare_model_training='Official is_rare on other GroupKFold(7) cohorts only',
        training_parameters=rare_model().get_params())


def mock_inputs(work: Path) -> tuple[list[Case],pd.DataFrame,dict]:
    bg=read_csv(work/'A_bg.csv');own=read_csv(work/'B_self.csv')
    # Avoid training on any Kanagawa member of any evaluation panel. This is
    # a MOCK auxiliary-data condition, not the official prelim OOF experiment.
    train=own[own.prefecture.ne('神奈川県')].copy()
    if len(np.unique(train.is_rare))!=2:
        raise ValueError('Insufficient rare training classes')
    model=rare_model().fit(rare_features(train),train.is_rare.to_numpy(bool))
    phi=read_csv(work/'mock/phi_results.csv')
    cframes={p.stem:read_csv(p) for p in sorted((work/'Cs').glob('*.csv'))}
    rng=np.random.default_rng(BOOT_SEED)
    cframes['CONTROL_identity']=own.drop(columns=['record_id','is_rare'])
    cframes['CONTROL_background_only']=bg.iloc[rng.choice(len(bg),1049,replace=False)].drop(columns=['record_id']).reset_index(drop=True)
    cases=[];probs={}
    for panel in range(1,6):
        key=f'prelim/panel_{panel:02d}'
        q=read_csv(work/f'mock/public/{key}/aia_challenge_{9500+panel}.csv').sort_values('challenge_row_id').reset_index(drop=True)
        t=read_csv(work/f'mock/private/{key}/truth.csv').sort_values('challenge_row_id').reset_index(drop=True)
        mapping=read_csv(work/f'mock/private/{key}/row_mapping.csv').sort_values('challenge_row_id').reset_index(drop=True)
        if not q.prefecture.eq('神奈川県').all(): raise ValueError('Mock-region mismatch')
        probs_panel=model.predict_proba(rare_features(q))[:,1]
        pval=float(phi.loc[phi.panel.eq(key),'phi_mock'].iloc[0])
        for name,c in cframes.items():
            case=Case(name,9500+panel,panel,q,c,t,mapping,pval,name.startswith('CONTROL_'))
            check_case(case);cases.append(case)
            probs[(case.name,case.panel)]=probs_panel.copy()
    return cases,bg,dict(probabilities=probs,folds=[],
        rare_model_training='MOCK ONLY: main B non-Kanagawa rows; evaluation query members excluded by public region',
        auxiliary_n=len(train),auxiliary_rare_n=int(train.is_rare.sum()),
        training_parameters=rare_model().get_params())


def run(mode: str, root: Path, out: Path) -> None:
    start=time.time()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run_manifest.json').exists():
        raise FileExistsError('Refuse overwriting a completed run; select a fresh --out')
    # Freeze the plan BEFORE model fitting or evaluation, not after results.
    plan=dict(version=VERSION,mode=mode,methods=list(METHODS),policies=list(POLICIES),
        widths=[5,10],primary_width=5 if mode=='official' else 10,
        rank_gap=HIGH_GAP,baseline_positive_fraction=POSITIVE_FRACTION,hit_time=HIT_TIME,
        miss_values=[0.,11.],rare_cutoffs=[26,52],model=rare_model().get_params(),
        note='All candidates fixed for this run. Historical benchmark was already inspected; not a pristine holdout.')
    dump(out/'frozen_plan.json',plan)
    cases,bg,meta=official_inputs(root) if mode=='official' else mock_inputs(root)
    rows=[];audits=[];pred_rows=[];detections=[]
    for case in cases:
        prob=meta['probabilities'][case.name if mode=='official' else (case.name,case.panel)]
        q=case.q[ ['challenge_row_id']+QI ].copy()
        scores=evidence(q,case.c,bg)
        ids=q.challenge_row_id.to_numpy(int)
        if mode=='official':
            rare=case.truth.is_rare.to_numpy(bool)
            n26=top_indices(prob,ids,26)
            detections.append(dict(team=case.team,auc=float(roc_auc_score(rare,prob)),
                ap=float(average_precision_score(rare,prob)),hits26=int(rare[n26].sum()),recall26=float(rare[n26].sum()/26)))
        for width in (5,10):
            masks,aa=predict_masks(q,scores,prob,width)
            lookup=case.truth.set_index('challenge_row_id')
            pairlookup=case.pairmap.set_index('challenge_row_id') if case.pairmap is not None else None
            for a in aa:
                ar=dict(dataset=case.name,team=case.team,panel=case.panel,**a)
                ar['one_member_one_control']=bool(lookup.loc[a['a'],'label'] != lookup.loc[a['b'],'label'])
                ar['high_is_member']=bool(lookup.loc[a['high'],'label']=='member')
                ar['true_pair_match']=None if pairlookup is None else bool(pairlookup.loc[a['a'],'pair_id']==pairlookup.loc[a['b'],'pair_id'])
                audits.append(ar)
            for (method,policy),mask in masks.items():
                basemask=masks[method,'baseline']
                for miss in (0.,11.):
                    pred=np.where(mask,HIT_TIME,miss)
                    basepred=np.where(basemask,HIT_TIME,miss)
                    metrics=score_prediction(pred,case.truth)
                    base_metrics=score_prediction(basepred,case.truth)
                    main_metric='r_max' if mode=='official' else 'r_all'
                    changed=pred!=basepred
                    m=case.truth.label.eq('member').to_numpy()
                    oldhit=np.abs(basepred-case.truth.time.to_numpy())<=.5
                    newhit=np.abs(pred-case.truth.time.to_numpy())<=.5
                    row=dict(mode=mode,dataset=case.name,team=case.team,panel=case.panel,
                        control_case=case.control_case,width=width,method=method,policy=policy,
                        miss_value=miss,changed_rows=int(changed.sum()),
                        member_hit_gains=int((m&~oldhit&newhit).sum()),member_hit_losses=int((m&oldhit&~newhit).sum()),
                        control_hit_gains=int((~m&~oldhit&newhit).sum()),control_hit_losses=int((~m&oldhit&~newhit).sum()),
                        delta_r_all=metrics['r_all']-base_metrics['r_all'],
                        delta_primary=metrics[main_metric]-base_metrics[main_metric],**metrics)
                    row['mock_phi_corrected_all']=None if case.phi_mock is None else max(0.,metrics['r_all']-case.phi_mock)
                    rows.append(row)
                    # Diagnostic long table, NOT a competition-ready F_aia.csv.
                    for i in np.flatnonzero(changed):
                        pred_rows.append(dict(dataset=case.name,team=case.team,panel=case.panel,width=width,
                            method=method,policy=policy,miss_value=miss,challenge_row_id=int(ids[i]),
                            baseline=float(basepred[i]),modified=float(pred[i]),
                            label=case.truth.label.iloc[i],true_time=float(case.truth.time.iloc[i]),
                            baseline_hit=bool(oldhit[i]),modified_hit=bool(newhit[i])))
    table=pd.DataFrame(rows)
    table.to_csv(out/'case_metrics.csv',index=False,float_format='%.17g')
    pd.DataFrame(audits).to_csv(out/'pair_audit.csv',index=False,float_format='%.17g')
    pd.DataFrame(pred_rows).to_csv(out/'changed_rows_diagnostics.csv',index=False,float_format='%.17g')
    if detections: pd.DataFrame(detections).to_csv(out/'rare_detector_oof.csv',index=False,float_format='%.17g')
    group_cols=['width','method','policy','miss_value']
    metrics=['r_all','r_rare','r_max','rare_extra','delta_primary','changed_rows',
        'member_hit_gains','member_hit_losses','control_hit_gains','control_hit_losses','mock_phi_corrected_all']
    subsets=[('official20',table[table.team.ne(5)]),('official18',table[~table.team.isin([1,2,5])])] if mode=='official' else [
        ('mock3C_5panels',table[~table.control_case]),
        ('CONTROL_identity',table[table.dataset.eq('CONTROL_identity')]),
        ('CONTROL_background_only',table[table.dataset.eq('CONTROL_background_only')])]
    parts=[]
    for name,subset in subsets:
        agg=subset.groupby(group_cols)[metrics].mean().reset_index()
        agg.insert(0,'evaluation_set',name)
        agg['cases']=subset.groupby(group_cols).size().to_numpy()
        agg['improved_cases']=subset.groupby(group_cols).delta_primary.apply(lambda x:int((x>1e-12).sum())).to_numpy()
        agg['worsened_cases']=subset.groupby(group_cols).delta_primary.apply(lambda x:int((x< -1e-12).sum())).to_numpy()
        parts.append(agg)
    pd.concat(parts,ignore_index=True).to_csv(out/'summary.csv',index=False,float_format='%.17g')
    # Same rule/mask/time estimates; change only the intentional-miss value.
    keys=['dataset','team','panel','control_case','width','method','policy']
    zero=table[table.miss_value.eq(0)].set_index(keys)
    eleven=table[table.miss_value.eq(11)].set_index(keys)
    contrast=(eleven[['r_all','r_rare','r_max','hit_m','hit_c']]-zero[['r_all','r_rare','r_max','hit_m','hit_c']])
    contrast.columns=['eleven_minus_zero_'+x for x in contrast.columns]
    contrast.reset_index().to_csv(out/'miss_value_contrast.csv',index=False,float_format='%.17g')
    manifest=dict(version=VERSION,mode=mode,elapsed_seconds=time.time()-start,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,
        inference_receives_truth=False,official_phi_computed=False,
        official_rare_computable=(mode=='official'),
        main_contest_submitted=False,github_written=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_metadata={k:v for k,v in meta.items() if k!='probabilities'},input_sha256=HASHES,
        caveats=['Mock panels share main-B members and the same Cs; not independent teams.',
            'A two-row block and score gap do not prove true pairing or membership.',
            'Mock control is_rare is unknown; no official Rrare or Rmax reported for mocks.',
            '0 and 11 comparisons include lost accidental member hits; 11 need not improve R.',
            'Baseline score families are newly implemented; not the exact historic P01/all4/top2.'])
    dump(out/'run_manifest.json',manifest)
    print(table.groupby(['method','policy','miss_value']).r_all.mean().to_string())
    print('Completed:',out,'cases',len(cases),'settings',len(table))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['official','mock'],required=True)
    p.add_argument('--root',type=Path,required=True,help='Repository root (official) or prepared mock work root')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.mode,a.root.resolve(),a.out.resolve())

if __name__=='__main__':main()
