"""公開背景と同じC行を除く、会員ラベルを使わない固定規則の診断。

全対象に Cそのまま / 公開13項目完全一致を除外 / QI7完全一致を除外 を適用。
固定Gen-LRA7 k10、QI7 NNratio k3カテゴリ罰則4、混合KDE比 h倍率1だけを比較。
固定県×性別CDF300と他対象中央値0.1を使う。方式・強度を選び直さない。
公開非会員の所有者を丸ごと保留したCRPS/PITはCDF段階を診断する。
通常875行の内部順位だけ保存し、所属正解・MIA採点・提出形式を使わない。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


def prohibit_label_and_archive_access(event, args):
    if event != 'open' or not args or isinstance(args[0], int):
        return
    path = unicodedata.normalize('NFC', str(args[0]))
    forbidden = ('pool_ids', '/正解データ/', '/アーカイブ/', 'F_mia.csv',
                 '対象別TPR.csv', '集約比較.csv', 'result.txt', 'attack_score.py')
    if any(term in path for term in forbidden):
        raise PermissionError(f'このラベル無し診断では参照禁止: {path}')


sys.addaudithook(prohibit_label_and_archive_access)
import 共通 as u

public = u.module('copy_probe_fixed_public_null', u.HERE/'公開非会員だけで較正を選ぶ.py')
distance = u.module('copy_probe_fixed_distance', u.HERE/'距離と分布の探索.py')
CONDITIONS = ('unchanged', 'remove_exact_public13', 'remove_exact_qi7')
METHODS = ('gen7_k10', 'qi7_nnratio_k3_cat4', 'qi7_mixed_kde_ratio_h1')
ALL13 = u.NUM13 + u.CAT13
STRENGTH = 300
CROSS = .1
BLOCK = 512
TOP = 12


def write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def pinned_read_npz(path):
    u.base.INPUT_HASHES[str(path.resolve())] = u.sha(path)
    return np.load(path, allow_pickle=False)


def matched(frame, reference, columns):
    return pd.MultiIndex.from_frame(frame[list(columns)]).isin(
        pd.MultiIndex.from_frame(reference[list(columns)]))


def references(data):
    a, query = data['a'], data['query']
    ar, x = distance.transform(a, [a, query], u.NUM7, 'robust')
    ref_near = np.empty((len(query), 150))
    background_log = np.empty(len(query))
    h = (4/(len(u.NUM7)+2)/len(a))**(1/(len(u.NUM7)+4))
    for start in range(0, len(query), BLOCK):
        stop = min(start+BLOCK, len(query))
        qq = query.iloc[start:stop]
        da = cdist(x[start:stop], ar, 'sqeuclidean')
        background_log[start:stop] = distance.mixed_logs(da, qq, a, u.CAT7, a, h)
        for col in u.CAT7:
            da += 4*(qq[col].to_numpy()[:, None] != a[col].to_numpy()[None, :])
        ref_near[start:stop] = distance.nearest(da, 150)
    return dict(ar=ar, x=x, ref_near=ref_near, background_log=background_log, h=h)


def raw_scores(data, c, ref):
    """既存core/transform/nearest/mixed_logsをそのまま使用する固定3式。"""
    a, query = data['a'], data['query']
    output = {'gen7_k10': u.base.core(a, query, c, u.NUM7, (10,))[10]}
    _, _, cr = distance.transform(a, [a, query.iloc[:0], c], u.NUM7, 'robust')
    nn = np.empty(len(query)); kde = np.empty(len(query))
    for start in range(0, len(query), BLOCK):
        stop = min(start+BLOCK, len(query)); qq = query.iloc[start:stop]
        dc = cdist(ref['x'][start:stop], cr, 'sqeuclidean')
        kde[start:stop] = distance.mixed_logs(dc, qq, c, u.CAT7, a, ref['h']) - ref['background_log'][start:stop]
        for col in u.CAT7:
            dc += 4*(qq[col].to_numpy()[:, None] != c[col].to_numpy()[None, :])
        near = distance.nearest(dc, 30)
        nn[start:stop] = np.log(ref['ref_near'][start:stop, :3].mean(axis=1)+1e-12)-np.log(near[:, :3].mean(axis=1)+1e-12)
    output['qi7_nnratio_k3_cat4'] = nn
    output['qi7_mixed_kde_ratio_h1'] = kde
    assert tuple(output) == METHODS
    assert all(np.isfinite(vector).all() for vector in output.values())
    return output


def verify_geometry_source(data, c, ref, raw):
    """非会員課題と通常候補を含む固定位置で元のgeometry関数と照合。"""
    ids = np.unique(np.r_[0, 1048, np.flatnonzero(data['normal'])[:7],
                            np.linspace(len(data['b']), len(data['query'])-1, 15, dtype=int)])
    q = data['query'].iloc[ids]
    expected_nn = distance.geometry(data['a'], q, c, u.NUM7, u.CAT7,
                                    'robust', 4, ref['ref_near'][ids])['nnratio_k3']
    expected_kde = distance.geometry(data['a'], q, c, u.NUM7, u.CAT7,
                                     'robust', 1, ref['ref_near'][ids])['kde_ratio_h1']
    errors = {'nnratio_max_error': float(np.max(np.abs(expected_nn-raw[METHODS[1]][ids]))),
              'kde_max_error': float(np.max(np.abs(expected_kde-raw[METHODS[2]][ids]))),
              'checked_rows': len(ids)}
    assert errors['nnratio_max_error'] == 0, errors
    assert errors['kde_max_error'] == 0, errors
    return errors


def owner_holdout(raw, data, target, owner):
    """対象自身のQを除き、他のQ所有者を丸ごと保留。尺度は訓練行のみ。"""
    ii = data['controls'][target]
    values = raw[ii]; q = data['query'].iloc[ii].reset_index(drop=True)
    qowner = owner[ii]
    rows = []
    for heldowner in np.unique(qowner):
        train = qowner != heldowner; held = ~train
        training = values[train]
        scale = float(np.quantile(training, .75)-np.quantile(training, .25))
        fallback = 'training_iqr'
        if scale <= np.finfo(float).eps:
            scale = max(float(np.std(training)), 1.)
            fallback = 'training_std_or_one'
        scaled = values/scale; global_null = scaled[train]
        held_indices = np.flatnonzero(held); heldq = q.iloc[held_indices]
        losses = np.full(len(held_indices), np.nan); pits = losses.copy()
        counts = np.zeros(len(held_indices), int)
        for (prefecture, sex), positions in heldq.groupby(['prefecture', 'sex'], sort=True).indices.items():
            group = (q.prefecture.to_numpy()==prefecture)&(q.sex.to_numpy()==sex)
            local = scaled[train&group]; observed = scaled[held_indices[positions]]
            a, b, c = public.crps_coefficients(global_null, local, observed)
            w = len(local)/(len(local)+STRENGTH)
            losses[positions] = a+2*w*b+w*w*c
            g = u.cal.empirical_cdf(observed, global_null)
            l = u.cal.empirical_cdf(observed, local) if len(local) else g
            pits[positions] = g+w*(l-g); counts[positions] = len(local)
        assert np.isfinite(losses).all() and np.isfinite(pits).all()
        assert (losses>=-1e-10).all()
        rows.append(dict(phase=data['phase'], target=target, heldout_null_owner=int(heldowner),
                         heldout_rows=len(pits), training_rows=int(train.sum()), training_scale=scale,
                         scale_rule=fallback, crps=float(losses.mean()), unscaled_crps=float(losses.mean()*scale),
                         pit_mean=float(pits.mean()), pit_above_99=float((pits>.99).mean()),
                         pit_above_95=float((pits>.95).mean()), pit_below_01=float((pits<.01).mean()),
                         local_n_min=int(counts.min()), local_n_median=float(np.median(counts)),
                         zero_local_support_rows=int((counts==0).sum()),
                         membership_labels_used=False, diagnostic_stage='CDF300_before_cross_median'))
    return rows


def top_set(values):
    return set(np.flatnonzero(rankdata(-values, method='min')<=TOP))


def run_phase(phase, out, qa):
    started = time.perf_counter(); data = u.setup(phase)
    owner, _, duplicate_q = public.public_folds(data)
    kit_a = u.base.read(u.ROOT/'スターターキット/初版/participant_data/A_bg.csv')
    source_union = pd.concat([data['a'], kit_a], ignore_index=True)
    profile = u.base.read(u.HERE/'公開加工特徴の比較結果/対象別の公開特徴.csv')
    profile = profile[profile.phase==phase].set_index('team')
    matchrows = []; removedrows = []; masks = {}; direct_affected = set()
    for target, c in data['cs'].items():
        phase13 = matched(c, data['a'], ALL13); kit13 = matched(c, kit_a, ALL13)
        phase7 = matched(c, data['a'], u.base.QI7); kit7 = matched(c, kit_a, u.base.QI7)
        assert int(phase13.sum()) == int(profile.loc[target, 'c_to_phase_a_exact_13'])
        assert int(kit13.sum()) == int(profile.loc[target, 'c_to_kit_a_exact_13'])
        assert int(phase7.sum()) == int(profile.loc[target, 'c_to_phase_a_exact_qi7'])
        assert int(kit7.sum()) == int(profile.loc[target, 'c_to_kit_a_exact_qi7'])
        public13 = matched(c, source_union, ALL13); qi7 = matched(c, source_union, u.base.QI7)
        assert np.array_equal(public13, phase13|kit13)
        assert np.array_equal(qi7, phase7|kit7) and np.all(~public13|qi7)
        masks[target] = dict(unchanged=np.zeros(len(c), bool), remove_exact_public13=public13,
                             remove_exact_qi7=qi7)
        if qi7.any(): direct_affected.add(target)
        for condition in CONDITIONS:
            remove = masks[target][condition]
            assert int((~remove).sum())>=30
            matchrows.append(dict(phase=phase, target=target, condition=condition, c_rows=len(c),
                                  removed_rows=int(remove.sum()), remaining_rows=int((~remove).sum()),
                                  phase_A13=int(phase13.sum()), kit_A13=int(kit13.sum()),
                                  sources_overlap13=int((phase13&kit13).sum()),
                                  phase_A_qi7=int(phase7.sum()), kit_A_qi7=int(kit7.sum()),
                                  sources_overlap_qi7=int((phase7&kit7).sum()),
                                  copy_processing_algorithm_unconfirmed=True))
            for position in np.flatnonzero(remove):
                removedrows.append(dict(phase=phase, target=target, condition=condition, c_row_position=int(position),
                                        matches_phase_A13=bool(phase13[position]), matches_kit_A13=bool(kit13[position]),
                                        matches_phase_A_qi7=bool(phase7[position]), matches_kit_A_qi7=bool(kit7[position])))
    ref = references(data); raws = {}; rows = []; banks = {}; controls = []
    old_raw_path = u.HERE/'非会員課題の丸ごと保留結果'/phase/'候補と公開課題の生スコア.npz'
    with pinned_read_npz(old_raw_path) as old:
        old_gen = old['raw'].copy(); old_teams = old['teams'].copy()
        assert np.array_equal(old['query_owner'], owner)
        assert int(old['b_rows']) == len(data['b'])
    old_ranks_path = u.HERE/'公開非会員の較正結果'/phase/'内部順位だけ.npz'
    with pinned_read_npz(old_ranks_path) as old:
        old500 = old['scores'][list(old['names']).index('fixed500__cross0.1')].copy()
        old300 = old['scores'][list(old['names']).index('owner_one_se__cross0.1')].copy()
        assert np.array_equal(old['teams'], data['teams'])
        assert np.array_equal(old['record_ids'], data['b'].loc[data['normal'], 'record_id'])
    control500 = {}
    for index, (target, c) in enumerate(data['cs'].items(), 1):
        baseline = raw_scores(data, c, ref)
        gen_error = float(np.max(np.abs(baseline[METHODS[0]]-old_gen[:, list(old_teams).index(target)])))
        assert gen_error == 0, (phase, target, gen_error)
        geom_qa = verify_geometry_source(data, c, ref, baseline)
        controls.append(dict(phase=phase, target=target, gen_saved_raw_max_error=gen_error, **geom_qa))
        control500[target] = public.calibrated(baseline[METHODS[0]], data, target, 500)
        for condition in CONDITIONS:
            remove = masks[target][condition]
            if condition=='unchanged' or not remove.any():
                scores = baseline
            elif np.array_equal(remove, masks[target]['remove_exact_public13']) and condition=='remove_exact_qi7':
                scores = raws[(target, 'remove_exact_public13')]
            else:
                scores = raw_scores(data, c.iloc[np.flatnonzero(~remove)].reset_index(drop=True), ref)
            raws[(target, condition)] = scores
            for method in METHODS:
                bankname = condition+'__'+method
                banks.setdefault(bankname, {})[target] = public.calibrated(scores[method], data, target, STRENGTH)
                rows.extend(dict(condition=condition, method=method, **row)
                            for row in owner_holdout(scores[method], data, target, owner))
        u.log(stage='public_row_removal_target', phase=phase, completed_targets=index,
              targets=len(data['teams']), direct_removed13=int(masks[target]['remove_exact_public13'].sum()),
              direct_removed7=int(masks[target]['remove_exact_qi7'].sum()), seconds=time.perf_counter()-started)
    matrices = u.matrices(banks, data['teams'], cross=(CROSS,))
    reproduced500 = u.matrices({'control500': control500}, data['teams'], cross=(CROSS,))['control500__cross0.1']
    reproduced300 = matrices['unchanged__gen7_k10__cross0.1']
    rank500_error = float(np.max(np.abs(reproduced500-old500)))
    rank300_error = float(np.max(np.abs(reproduced300-old300)))
    assert rank500_error == 0 and rank300_error == 0, (rank500_error, rank300_error)
    drifts = []; top_rows = []; support = []
    record_ids = data['b'].loc[data['normal'], 'record_id'].to_numpy()
    for method in METHODS:
        baseline = matrices['unchanged__'+method+'__cross0.1']
        for condition in CONDITIONS:
            current = matrices[condition+'__'+method+'__cross0.1']
            for j, target in enumerate(data['teams']):
                old, new = top_set(baseline[:, j]), top_set(current[:, j])
                before, after = rankdata(baseline[:, j]), rankdata(current[:, j])
                removed = int(masks[target][condition].sum())
                drifts.append(dict(phase=phase, target=target, condition=condition, method=method, top=TOP,
                                   original_top_count=len(old), filtered_top_count=len(new), overlap=len(old&new),
                                   added=len(new-old), removed=len(old-new), c_rows_removed=removed,
                                   directly_affected=removed>0, top_changed=old!=new,
                                   spearman=float(np.corrcoef(before, after)[0, 1]),
                                   mean_absolute_rank_shift=float(np.abs(after-before).mean()),
                                   maximum_absolute_rank_shift=float(np.abs(after-before).max()),
                                   membership_accuracy_unknown=True, cross_median=CROSS))
                for position in sorted(old|new):
                    top_rows.append(dict(phase=phase, target=target, condition=condition, method=method,
                                         record_id=record_ids[position], in_original_top12=position in old,
                                         in_filtered_top12=position in new, original_descending_rank=float(876-before[position]),
                                         filtered_descending_rank=float(876-after[position]), membership_accuracy_unknown=True))
                ii = data['controls'][target]; q = data['query'].iloc[ii]
                filtered = data['cs'][target].iloc[np.flatnonzero(~masks[target][condition])]
                for key, positions in data['b'].loc[data['normal']].groupby(['prefecture', 'sex'], sort=True).indices.items():
                    prefecture, sex = key
                    null = (q.prefecture.to_numpy()==prefecture)&(q.sex.to_numpy()==sex)
                    cgroup = (filtered.prefecture==prefecture)&(filtered.sex==sex)
                    support.append(dict(phase=phase, target=target, condition=condition, method=method,
                                        prefecture=prefecture, sex=int(sex), candidate_rows=len(positions),
                                        null_rows=int(null.sum()), null_owners=int(len(np.unique(owner[ii][null]))),
                                        local_cdf_weight=float(null.sum()/(null.sum()+STRENGTH)),
                                        remaining_c_same_group_rows=int(cgroup.sum()), membership_accuracy_unknown=True))
    table = pd.DataFrame(rows)
    metric_columns = ['crps', 'unscaled_crps', 'pit_mean', 'pit_above_99', 'pit_above_95', 'pit_below_01']
    target_summary = table.groupby(['phase', 'target', 'condition', 'method'], as_index=False)[metric_columns].mean()
    summary = []
    for subset, subset_table in [('all_targets', table), ('predeclared_other18' if phase=='prelim' else 'all_targets_duplicate',
                                                        table[~table.target.isin([1,2])] if phase=='prelim' else table)]:
        if subset=='all_targets_duplicate': continue
        for (condition, method), part in subset_table.groupby(['condition', 'method'], sort=True):
            summary.append(dict(phase=phase, subset=subset, condition=condition, method=method,
                                targets=int(part.target.nunique()), owner_target_cases=len(part),
                                **{column: float(part[column].mean()) for column in metric_columns},
                                membership_labels_used=False, used_for_method_selection=False))
    delta_rows = []
    baseline_summary = target_summary[target_summary.condition=='unchanged'].set_index(['target', 'method'])
    for row in target_summary.to_dict('records'):
        reference = baseline_summary.loc[(row['target'], row['method'])]
        delta_rows.append(dict(**row, **{'delta_'+column: float(row[column]-reference[column]) for column in metric_columns},
                               directly_affected=bool(masks[row['target']][row['condition']].any()),
                               membership_labels_used=False))
    dest = out/phase; dest.mkdir()
    pd.DataFrame(matchrows).to_csv(dest/'公開背景一致と除外数.csv', index=False, float_format='%.17g')
    pd.DataFrame(removedrows).to_csv(dest/'除外行の公開根拠.csv', index=False)
    table.to_csv(dest/'課題所有者を保留したCRPSとPIT.csv', index=False, float_format='%.17g')
    pd.DataFrame(summary).to_csv(dest/'全対象共通の固定条件診断.csv', index=False, float_format='%.17g')
    pd.DataFrame(delta_rows).to_csv(dest/'対象別の非会員診断差.csv', index=False, float_format='%.17g')
    pd.DataFrame(drifts).to_csv(dest/'通常875行の上位12変化.csv', index=False, float_format='%.17g')
    pd.DataFrame(top_rows).to_csv(dest/'上位12の内部行診断.csv', index=False, float_format='%.17g')
    pd.DataFrame(support).to_csv(dest/'通常候補の較正とCの支持範囲.csv', index=False, float_format='%.17g')
    pd.DataFrame(controls).to_csv(dest/'固定元方式の再現対照.csv', index=False, float_format='%.17g')
    names = sorted(matrices)
    cross_scores = np.stack([matrices[name] for name in names])
    final_ranks = np.stack([np.column_stack([rankdata(mat[:, j], method='average')/875
                                           for j in range(len(data['teams']))]) for mat in cross_scores])
    np.savez_compressed(dest/'通常875行の内部順位だけ.npz', names=np.array(names), teams=np.array(data['teams']),
                        ranks=final_ranks, cross_scores=cross_scores, record_ids=record_ids)
    for path, hashed in u.base.INPUT_HASHES.items(): assert u.sha(path)==hashed, path
    assert not any('pool_ids' in path or '正解データ' in path for path in u.base.INPUT_HASHES)
    record = dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                  seconds=time.perf_counter()-started, python=sys.executable, python_version=platform.python_version(),
                  numpy_version=np.__version__, pandas_version=pd.__version__, code_path=str(Path(__file__).resolve()),
                  code_sha256=u.sha(__file__), support_code_sha256=qa['support_code_sha256'],
                  input_sha256=dict(u.base.INPUT_HASHES), output_sha256={p.name: u.sha(p) for p in dest.iterdir()},
                  numerical_crps_qa=qa['crps_integral'], baseline_gen_raw_max_error=float(max(r['gen_saved_raw_max_error'] for r in controls)),
                  baseline_gen_cdf500_cross_max_error=rank500_error, baseline_gen_cdf300_cross_max_error=rank300_error,
                  baseline_geometry_max_errors={k: float(max(r[k] for r in controls)) for k in ('nnratio_max_error', 'kde_max_error')},
                  direct_qi7_affected_targets=sorted(direct_affected), duplicated_public_qi7_rows=duplicate_q,
                  label_guard_installed=True, membership_labels_used=False, mia_scores_used=False,
                  heldout_scale='For each held owner and condition, IQR from training null rows only; train std/one fallback.',
                  diagnostic_stage='LOQ-owner-out CRPS/PIT assesses CDF300 before cross correction. Candidate ranks use cross median0.1.',
                  rule_selection='None: 3 rules and 3 methods are common to all targets and remain fixed.',
                  interpretation='Exact public matches are observed public patterns; not proof of a copy algorithm or a membership-performance effect.',
                  limitations=['Unknown-membership rows have no MIA accuracy labels.',
                               'Conditions rescale predictive CRPS using their own training-only IQR; unscaled CRPS is also retained.',
                               'Removing public-matched C rows changes sample size and may discard coincidental QI7 matches.',
                               'Targets without removed rows may still change through the other-target median.'])
    write_json(dest/'実行記録.json', record)
    u.log(stage='public_row_removal_complete', phase=phase, seconds=record['seconds'],
          output=str(dest), baseline_gen_cdf300_cross_max_error=rank300_error,
          direct_qi7_affected_targets=sorted(direct_affected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=u.HERE/'公開背景行の除外診断結果')
    args = parser.parse_args()
    assert not args.output_dir.exists(), args.output_dir
    assert sys.dont_write_bytecode, 'ルート .venv/bin/python -B を使うこと'
    assert Path(sys.executable).absolute() == u.ROOT/'.venv/bin/python', sys.executable
    supports = [u.HERE/'共通.py', u.HERE/'公開非会員だけで較正を選ぶ.py', u.HERE/'距離と分布の探索.py',
                u.OLD/'分析.py', u.CAL/'分析.py']
    qa = dict(crps_integral=public.verify_integral(), support_code_sha256={str(p): u.sha(p) for p in supports})
    args.output_dir.mkdir()
    plan = dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(), code_sha256=u.sha(__file__),
                phases=['prelim', 'main'], conditions=list(CONDITIONS), methods=list(METHODS),
                public_source_union='The phase-specific A_bg plus the original kit participant_data/A_bg; exact typed CSV values.',
                public13_columns=list(ALL13), qi7_columns=list(u.base.QI7),
                strength=STRENGTH, cross_median=CROSS, top=TOP, top_ties='Include every row with minimum descending rank <=12.',
                gen7_k10='Existing base.core NUM7, k10; unchanged.',
                qi7_nnratio_k3_cat4='Existing robust transform and nearest mean distances; category mismatch penalty4; k3.',
                qi7_mixed_kde_ratio_h1='Existing mixed_logs, robust transform, h=(4/(d+2)/nA)^(1/(d+4))*1.',
                calibration='Existing county×sex CDF300, .999 CDF + .001 raw B rank, normal875 rank; other-target median correction0.1.',
                nonmember_diagnostic='Leave one Q owner out, never use own-target Q; IQR only from training null scores in each condition.',
                aggregation='All-target summaries; also predeclared prelim other18 excluding1/2. No condition/method selection.',
                negative_controls='Unchanged Gen raw and county CDF500/300 ranks reproduce saved label-free controls; NNratio/KDE source geometry checks.',
                input_label_use=False, no_mia_score_or_submission=True, no_per_target_selection=True,
                final_prose_or_github_created=False, numerical_qa=qa,
                hypothesis='Removing C rows matching public data is a public-pattern probe, not evidence of a specific copy algorithm or membership gain.')
    write_json(args.output_dir/'事前固定条件.json', plan)
    for phase in ('prelim', 'main'):
        run_phase(phase, args.output_dir, qa)
    for path, hashed in qa['support_code_sha256'].items(): assert u.sha(path)==hashed, path
    write_json(args.output_dir/'完了記録.json', dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                                                  phases_completed=['prelim', 'main'], membership_labels_used=False,
                                                  mia_scoring_performed=False, final_prose_or_github_created=False,
                                                  code_sha256=u.sha(__file__)))


if __name__=='__main__': main()
