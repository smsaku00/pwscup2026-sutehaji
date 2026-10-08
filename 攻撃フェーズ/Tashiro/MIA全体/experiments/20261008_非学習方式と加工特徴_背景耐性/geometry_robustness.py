"""既存の距離・KDE八設定を固定し、公開Aの七抽出条件だけを変更する。

原mainは呼ばず、原純関数と必要な距離式だけを計算する。
全背景・全対象の予測をメモリ上で確定してから所属正解を採点に使う。
個人配列、所属正解、提出回答は保存しない。
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np

import background_common as bc

GEOMETRY_SOURCE = bc.SOURCE / '距離と分布の探索.py'
ALIGN_SOURCE = bc.SOURCE / '分布を揃えた距離の探索.py'
OLD_POINT_SOURCE = bc.u.OLD / '追加検討.py'
geo = bc.module('_fixed_background_geometry', GEOMETRY_SOURCE)
alignment = bc.module('_fixed_background_alignment', ALIGN_SOURCE)

SPECS = [
    dict(method='global_nn_k1', attributes='q7', statistic='nearest_distance_ratio',
         coordinate='original_20261007_iqr_without_centering', alignment=None,
         calibration='raw', cross=0., k=1, source=str(OLD_POINT_SOURCE.relative_to(bc.ROOT))),
    dict(method='q13_median_iqr_sex0_kde_ratio_h2__raw__cross0.1', attributes='q13',
         statistic='kde_ratio', coordinate='robust', alignment='median_iqr',
         alignment_by_sex=False, multiplier=2., calibration='raw', cross=.1,
         source=str(ALIGN_SOURCE.relative_to(bc.ROOT))),
    dict(method='q13_mean_sd_sex0_kde_witness_h2__raw__cross0', attributes='q13',
         statistic='kde_witness', coordinate='robust', alignment='mean_sd',
         alignment_by_sex=False, multiplier=2., calibration='raw', cross=0.,
         source=str(ALIGN_SOURCE.relative_to(bc.ROOT))),
    dict(method='q7_quantile_sex1_kde_ratio_h0.5__county500__cross0.1', attributes='q7',
         statistic='kde_ratio', coordinate='robust', alignment='quantile',
         alignment_by_sex=True, multiplier=.5, calibration='county500', cross=.1,
         source=str(ALIGN_SOURCE.relative_to(bc.ROOT))),
    dict(method='q13_quantile_sex1_kde_ratio_h0.5__raw__cross0', attributes='q13',
         statistic='kde_ratio', coordinate='robust', alignment='quantile',
         alignment_by_sex=True, multiplier=.5, calibration='raw', cross=0.,
         source=str(ALIGN_SOURCE.relative_to(bc.ROOT))),
    dict(method='q7_median_iqr_sex1_dcr_k1__raw__cross0', attributes='q7',
         statistic='distance_to_closest_release', coordinate='robust',
         alignment='median_iqr', alignment_by_sex=True, k=1,
         calibration='raw', cross=0., source=str(ALIGN_SOURCE.relative_to(bc.ROOT))),
    dict(method='q7_rank_kde_witness_h0.5__age100__cross0', attributes='q7',
         statistic='kde_witness', coordinate='rank', alignment=None,
         multiplier=.5, calibration='age100', cross=0.,
         source=str(GEOMETRY_SOURCE.relative_to(bc.ROOT))),
    dict(method='q7_robust_kde_ratio_h1__raw__cross0.1', attributes='q7',
         statistic='kde_ratio', coordinate='robust', alignment=None,
         multiplier=1., calibration='raw', cross=.1,
         source=str(GEOMETRY_SOURCE.relative_to(bc.ROOT))),
]


def prepare(data, attributes, coordinate, multipliers):
    """原transform/mixed_logsでA側密度を作り、対象間で共用する。"""
    nums = bc.u.NUM7 if attributes == 'q7' else bc.u.NUM13
    cats = bc.u.CAT7 if attributes == 'q7' else bc.u.CAT13
    query = data['query'] if attributes == 'q7' else data['b']
    a = data['a']
    ar, x = geo.transform(a, [a, query], nums, coordinate)
    h0 = (4 / (len(nums) + 2) / len(a)) ** (1 / (len(nums) + 4))
    density = {}
    if multipliers:
        distance = geo.cdist(x, ar, 'sqeuclidean')
        for multiplier in multipliers:
            density[multiplier] = geo.mixed_logs(distance, query, a, cats, a, h0 * multiplier)
        del distance
    return dict(nums=nums, cats=cats, query=query, x=x, h0=h0,
                coordinate=coordinate, density=density)


def fixed_score(data, c, prepared, spec):
    """原geometryの指定された一式だけを実行する。"""
    a, query, nums, cats = data['a'], prepared['query'], prepared['nums'], prepared['cats']
    if spec.get('alignment'):
        c = alignment.align(a, c, nums, spec['alignment'], spec['alignment_by_sex'])
    cr = geo.transform(a, [c], nums, prepared['coordinate'])[0]
    distance = geo.cdist(prepared['x'], cr, 'sqeuclidean')
    if spec['statistic'] == 'distance_to_closest_release':
        for col in cats:
            distance += (query[col].to_numpy()[:, None] != c[col].to_numpy()[None, :])
        # 原geometryのpartition/sort後のk=1平均と同じ要素を取る。
        return -np.log(distance.min(axis=1) + 1e-12)
    multiplier = spec['multiplier']
    lc = geo.mixed_logs(distance, query, c, cats, a, prepared['h0'] * multiplier)
    la = prepared['density'][multiplier]
    return lc - la if spec['statistic'] == 'kde_ratio' else np.exp(lc) - np.exp(la)


def original_nn_reference(data):
    """10/7 point_scoresの中央値を引かない数値式でA側k=1距離。"""
    a, b = data['a'], data['b']
    scale = bc.u.base.scales(a, bc.u.NUM7)
    ar, br = [bc.u.base.numbers(f, bc.u.NUM7) / scale for f in (a, b)]
    distance = geo.cdist(br, ar, 'sqeuclidean')
    for col in ('sex', 'prefecture'):
        distance += (b[col].to_numpy()[:, None] != a[col].to_numpy()[None, :])
    return br, scale, distance.min(axis=1)


def original_nn_score(data, c, reference):
    br, scale, closest_a = reference
    cr = bc.u.base.numbers(c, bc.u.NUM7) / scale
    distance = geo.cdist(br, cr, 'sqeuclidean')
    for col in ('sex', 'prefecture'):
        distance += (data['b'][col].to_numpy()[:, None] != c[col].to_numpy()[None, :])
    return np.log(closest_a + 1e-12) - np.log(distance.min(axis=1) + 1e-12)


def verify_full_ranks(data, predictions):
    """10/8の七設定は保存内部順位とも照合。予測/正解の保存はしない。"""
    errors, hashes = {}, {}
    for folder, specs in [('分布を揃えた距離結果', SPECS[1:6]),
                          ('距離と分布結果', SPECS[6:])]:
        path = bc.SOURCE / folder / '内部順位キャッシュ.npz'
        hashes[str(path.relative_to(bc.ROOT))] = bc.sha(path)
        with np.load(path, allow_pickle=False) as cache:
            names, teams = list(cache['names']), list(cache['teams'])
            assert np.array_equal(cache['record_ids'], data['b'].loc[data['normal'], 'record_id'].to_numpy())
            indexes = [teams.index(t) for t in data['teams']]
            scores = cache['scores']
            for spec in specs:
                name = spec['method']
                saved = scores[names.index(name)][:, indexes]
                error = float(np.max(np.abs(saved - predictions[('full', name)])))
                errors[name] = error
                assert error < 1e-12, (name, '元内部順位との不一致', error)
        bc.log(stage='full_saved_rank_verified', source=folder,
               methods=len(specs), maximum_error=max(errors[s['method']] for s in specs))
    return errors, hashes


def main():
    started = time.perf_counter()
    for filename in ('geometry_results.csv', 'geometry_run.json'):
        if (bc.HERE / filename).exists():
            raise FileExistsError(bc.HERE / filename)
    tracked = [Path(__file__).resolve(), GEOMETRY_SOURCE, ALIGN_SOURCE, OLD_POINT_SOURCE]
    hashes = {str(p.relative_to(bc.ROOT)): bc.sha(p) for p in tracked}
    for p in tracked:
        bc.u.base.INPUT_HASHES[str(p)] = bc.sha(p)
    data = bc.setup()
    assert len(data['teams']) == 20 and 12 in data['teams'] and 5 not in data['teams']
    bc.log(stage='public_inputs_ready', targets=len(data['teams']), background_rows=len(data['a']),
           query_rows=len(data['query']), fixed_settings=len(SPECS))
    predictions = {}
    # 原alignは整数列へのfloat代入警告を出すが、その元の数値挙動を保持する。
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        for label, changed in bc.conditions(data):
            prepared = {
                ('q7', 'robust'): prepare(changed, 'q7', 'robust', (.5, 1.)),
                ('q7', 'rank'): prepare(changed, 'q7', 'rank', (.5,)),
                ('q13', 'robust'): prepare(changed, 'q13', 'robust', (.5, 2.)),
            }
            nn_reference = original_nn_reference(changed)
            banks = {s['method']: {} for s in SPECS}
            for index, t in enumerate(changed['teams'], 1):
                c = changed['cs'][t]
                banks[SPECS[0]['method']][t] = bc.variant(
                    original_nn_score(changed, c, nn_reference), changed, t, 'raw')
                for spec in SPECS[1:]:
                    raw = fixed_score(changed, c, prepared[(spec['attributes'], spec['coordinate'])], spec)
                    banks[spec['method']][t] = bc.variant(raw, changed, t, spec['calibration'])
                if index % 5 == 0:
                    bc.log(stage='fixed_targets', condition=label, targets=index,
                           seconds=time.perf_counter() - started)
            for spec in SPECS:
                predictions[(label, spec['method'])] = bc.cross(banks[spec['method']], changed, spec['cross'])
            del prepared, banks, nn_reference
            bc.log(stage='background_predictions_fixed', condition=label,
                   background_rows=len(changed['a']), seconds=time.perf_counter() - started)
    bc.log(stage='all_predictions_fixed_before_labels', settings=len(SPECS), backgrounds=len(bc.CONDITIONS),
           matrices=len(predictions), seconds=time.perf_counter() - started)
    rank_errors, cache_hashes = verify_full_ranks(data, predictions)
    for path, expected in hashes.items():
        assert bc.sha(bc.ROOT / path) == expected
    bc.score_and_save(data, predictions, 'geometry_results.csv', 'geometry_run.json', SPECS, started,
        extra=dict(source_code_sha256=hashes, full_normal_rank_max_errors=rank_errors,
                   source_cache_sha256=cache_hashes, parameter_search=False,
                   all_predictions_fixed_before_truth=True, cross_reference_teams=data['teams'],
                   cross_self_excluded=True, public_query_and_age_boundaries_fixed=True,
                   source_function_calls=['transform', 'mixed_logs', 'align', 'numbers', 'scales',
                                          'variants', 'stratified_subset'],
                   equivalent_selected_formula='原geometry/point_scoresから指定式だけを取り出し、k=1のpartition/sortは同じ最小値計算で置換。'))


if __name__ == '__main__':
    main()
