"""第5回 MIA の指定5方式を全22本戦対象で再現する。大会へは提出しない。

原コードの指定 pure 関数だけを AST で読込む。評価関数・main・所属正解は読込まない。
実行例: .venv/bin/python -B generate_selected.py --data-root REPO --source-root REPO
--starter-bg /path/to/starter/participant_data/A_bg.csv --output NEW_DIRECTORY
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import itertools
import json
import os
import platform
import sys
import time
import types
import unicodedata
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, ndtri
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from threadpoolctl import threadpool_info, threadpool_limits

# 原alignの整数列→小数代入はpandas2.3.3では同じ値を生成するが、大量の将来警告を出す。
# 数学関数を変更せず表示だけ抑制。n_jobs=2の物理CPU照会もsandbox内で不要。
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '2')

HERE = Path(__file__).resolve().parent
METHODS = (
    'q7_hist5_s100_pairwise__county500__cross0.1',
    'q7_pref_sex_bins20_s1_zscore__county500__cross0.1',
    'q7_robust_lofC_k100__age100__cross0.1',
    'q13_robust_sex1_diag_reg0.01_gmm4__raw__cross0.1',
    'q7_median_iqr_sex1_kde_ratio_h1__age100__cross0.1',
)
SELECTED = {5: METHODS[3], 7: METHODS[2], 13: METHODS[1], 14: METHODS[4], 15: METHODS[0]}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def log(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


def normalized_path(root, relative):
    """GitHub NFC と macOS NFD の同一パスを一意に解決。"""
    p = root
    for name in Path(relative).parts:
        candidates = [x for x in p.iterdir() if unicodedata.normalize('NFC', x.name) == unicodedata.normalize('NFC', name)]
        if len(candidates) != 1:
            raise FileNotFoundError(str(root / relative))
        p = candidates[0]
    return p


def input_path(root, role, entry, starter):
    if role == 'background_starter':
        return starter
    try:
        return normalized_path(root, entry['path'])
    except FileNotFoundError:
        if role in ('main_B', 'background_main'):
            parent = normalized_path(root, '本戦/配布データ')
            candidates = list(parent.glob('PWSCup2026_final_team28_*/' + Path(entry['path']).name))
            assert len(candidates) == 1, candidates
            return candidates[0]
        if role == 'background_prelim':
            return normalized_path(root, '予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv')
        return normalized_path(root, '本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/' + Path(entry['path']).name)


def pure_source(path, functions, constants=(), injected=None):
    """関数本文を変更せず抽出。imports/top-level I/O/evaluate/main は実行しない。"""
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants:
            nodes.append(node)
    assert found == set(functions), (path, found, functions)
    namespace = dict(np=np, pd=pd, rankdata=rankdata, itertools=itertools, cdist=cdist,
                     logsumexp=logsumexp, ndtri=ndtri, solve_triangular=solve_triangular,
                     LedoitWolf=LedoitWolf, GaussianMixture=GaussianMixture,
                     LocalOutlierFactor=LocalOutlierFactor, ConvergenceWarning=ConvergenceWarning,
                     warnings=warnings)
    namespace.update(injected or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def modules(source_paths):
    base = pure_source(source_paths['base_oct7.py'], ['confidence'], ['NUM7', 'NUM9', 'QI7'])
    cal = pure_source(source_paths['calibration.py'], ['empirical_cdf', 'group_stats', 'blend'])
    u = pure_source(source_paths['common.py'], ['rank', 'variants', 'matrices'],
                    ['NUM7', 'NUM9', 'NUM13', 'CAT7', 'CAT13'], {'base': base, 'cal': cal})
    geo = pure_source(source_paths['distance_distribution.py'],
                     ['numeric', 'transform', 'distances', 'nearest', 'mixed_logs', 'geometry', 'categorical_logratio', 'binned'],
                     ['KS'], {'u': u})
    hist = pure_source(source_paths['hierarchical_hist.py'], ['group_codes', 'histograms'], ['STRENGTHS'], {'u': u})
    mix = pure_source(source_paths['gmm_local.py'], ['mixture'], injected={'u': u})
    align = pure_source(source_paths['aligned.py'], ['align'], injected={'u': u})
    return u, geo, hist, mix, align


def chunked_density(x, anchor, query, reference, cats, background, bandwidth, geo, block):
    """元mixed_logsのrowごとの計算は変更せず、query rowsだけ分割。"""
    result = np.empty(len(query))
    for start in range(0, len(query), block):
        stop = min(start + block, len(query))
        result[start:stop] = geo.mixed_logs(cdist(x[start:stop], anchor, 'sqeuclidean'),
                                           query.iloc[start:stop], reference, cats, background, bandwidth)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--starter-bg', type=Path, required=True)
    parser.add_argument('--input-spec', type=Path, default=HERE / 'input_spec.json')
    parser.add_argument('--output', type=Path, default=HERE / 'generated')
    parser.add_argument('--block-rows', type=int, default=256)
    args = parser.parse_args()
    assert args.block_rows > 0
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    spec = json.loads(args.input_spec.read_text())
    source_paths = {}
    source_records = []
    for origin in spec['source_origins']:
        try:
            path = normalized_path(args.source_root, origin['repository_path'])
        except FileNotFoundError:
            path = HERE / origin['local_name']
        content = path.read_bytes()
        assert git_blob(content) == origin['git_blob_sha'], path
        source_paths[origin['local_name']] = path
        source_records.append({**origin, 'verified_sha256': sha_bytes(content), 'pure_functions_only': True})
    u, geo, hist, mix, align = modules(source_paths)
    inputs = {}
    input_records = []
    background_bytes = []
    for role, entry in spec['inputs'].items():
        path = input_path(args.data_root, role, entry, args.starter_bg)
        content = path.read_bytes()
        assert git_blob(content) == entry['sha'], (role, path, git_blob(content), entry['sha'])
        # 所属正解のファイル名・ディレクトリは受け付けない。
        assert 'pool_ids' not in path.name and '正解データ' not in str(path)
        inputs[role] = pd.read_csv(io.BytesIO(content), float_precision='round_trip')
        input_records.append(dict(role=role, repository=entry.get('repository', spec['repository']),
                                  repository_path=entry['path'], git_blob_sha=entry['sha'],
                                  sha256=sha_bytes(content), bytes=len(content), rows=len(inputs[role])))
    for role in ('background_starter', 'background_prelim', 'background_main'):
        path = input_path(args.data_root, role, spec['inputs'][role], args.starter_bg)
        background_bytes.append(path.read_bytes())
    header = background_bytes[0].splitlines(keepends=True)[0]
    assert all(x.splitlines(keepends=True)[0] == header for x in background_bytes)
    concatenated = header + b''.join(b''.join(x.splitlines(keepends=True)[1:]) for x in background_bytes)
    assert sha_bytes(concatenated) == spec['background_concat_sha256']
    a = pd.read_csv(io.BytesIO(concatenated), float_precision='round_trip')
    assert len(a) == 15429
    b = inputs['main_B'].copy()
    assert len(b) == 1049 and b.record_id.is_unique and not b.duplicated(list(u.base.QI7)).any()
    cs = {int(role[2:]): frame for role, frame in inputs.items() if role.startswith('C_') and role != 'C_28'}
    qs = {int(role.rsplit('_', 1)[1]): frame for role, frame in inputs.items() if role.startswith('aia_challenge_')}
    teams = sorted(cs)
    assert len(teams) == 22 and len(qs) == 23 and set(qs) == set(teams) | {28}
    private = pd.MultiIndex.from_frame(b[list(u.base.QI7)]).isin(pd.MultiIndex.from_frame(qs[28][list(u.base.QI7)]))
    rare = b.is_rare.to_numpy(bool)
    normal = ~private & ~rare
    assert (int(private.sum()), int(rare.sum()), int((private & rare).sum()), int(normal.sum())) == (163, 24, 13, 875)
    query = pd.concat([b[list(u.base.QI7)]] + [qs[t][list(u.base.QI7)] for t in sorted(qs)], ignore_index=True)
    owner = np.r_[np.full(len(b), -1), *[np.full(len(qs[t]), t) for t in sorted(qs)]]
    cuts = np.unique(np.quantile(a.age, (.2, .4, .6, .8)))
    query['age_bin'] = np.searchsorted(cuts, query.age, side='right')
    b['age_bin'] = query.age_bin.iloc[:len(b)].to_numpy()
    a['age_bin'] = np.searchsorted(cuts, a.age, side='right')
    controls = {t: np.flatnonzero((owner != -1) & (owner != t)) for t in teams}
    for indices in controls.values():
        assert not pd.MultiIndex.from_frame(b.loc[normal, list(u.base.QI7)]).isin(pd.MultiIndex.from_frame(query.iloc[indices][list(u.base.QI7)])).any()
    data = dict(b=b, a=a, private=private, rare=rare, normal=normal, cs=cs, teams=teams,
                query=query, controls=controls, age_cuts=cuts, qs=qs, self_id=28)
    # mixed_logsのカテゴリlevel数がquery row分割で変わらない条件を明示検証。
    for column in u.CAT7:
        assert set(query[column]).issubset(set(a[column])), column
    log(stage='inputs_verified',background='starter+prelim+main',background_rows=len(a),targets=teams,
        query_rows=len(query),normal_rows=int(normal.sum()),input_count=len(input_records),truth_read=False)
    stems = [m.rsplit('__', 2)[0] for m in METHODS]
    calibrations = [m.rsplit('__', 2)[1] for m in METHODS]
    banks = {stem + '__' + label: {} for stem, label in zip(stems, calibrations)}
    raw_b = np.empty((5, len(b), len(teams)))
    raw_q = np.empty((4, len(query), len(teams)))
    fit_records = []
    kernel_checks = []
    with threadpool_limits(limits=1):
        inference_threadpools = threadpool_info()
        ar7, x7 = geo.transform(a, [a, query], u.NUM7, 'robust')
        ar13, x13 = geo.transform(a, [a, b], u.NUM13, 'robust')
        background_gmm, fits = mix.mixture(ar13, x13, 4, 'diag', .01, 20261008,
                                            a.sex.to_numpy(), b.sex.to_numpy())
        fit_records.extend(dict(source='A', team=None, seed=20261008, **r) for r in fits)
        h0 = (4 / (len(u.NUM7) + 2) / len(a)) ** (1 / (len(u.NUM7) + 4))
        background_kde = chunked_density(x7, ar7, query, a, u.CAT7, a, h0, geo, args.block_rows)
        for j, team in enumerate(teams):
            c = cs[team]
            pairhist = geo.binned(a, query, c, u.NUM7, u.CAT7, 5, 100)['pairwise']
            hierarchical = hist.histograms(a, query, c, u.NUM7, u.CAT7, 20, ['prefecture', 'sex'])['s1_zscore']
            cr7 = geo.transform(a, [c], u.NUM7, 'robust')[0]
            lof = LocalOutlierFactor(n_neighbors=100, novelty=True, n_jobs=2)
            lof.fit(cr7)
            lof_score = lof.score_samples(x7)
            cr13 = geo.transform(a, [c], u.NUM13, 'robust')[0]
            gmm_c, fits = mix.mixture(cr13, x13, 4, 'diag', .01, 20261008 + team,
                                      c.sex.to_numpy(), b.sex.to_numpy())
            fit_records.extend(dict(source='C', team=team, seed=20261008 + team, **r) for r in fits)
            gmm_score = gmm_c - background_gmm + geo.categorical_logratio(a, c, b, u.CAT13, 20)
            aligned = align.align(a, c, u.NUM7, 'median_iqr', True)
            aligned_coords = geo.transform(a, [aligned], u.NUM7, 'robust')[0]
            kde = chunked_density(x7, aligned_coords, query, aligned, u.CAT7, a, h0, geo, args.block_rows) - background_kde
            # 同じ元geometryから抜いた12行で、KDE分割計算が完全一致することを確認。
            check_rows = np.linspace(0, len(query) - 1, 12, dtype=int)
            direct = geo.geometry(a, query.iloc[check_rows], aligned, u.NUM7, u.CAT7, 'robust', 1,
                                  np.zeros((len(check_rows), 150)))['kde_ratio_h1']
            max_error = float(np.max(np.abs(kde[check_rows] - direct)))
            assert np.array_equal(kde[check_rows], direct), (team, max_error)
            kernel_checks.append(dict(team=team, rows=len(check_rows), direct_geometry_max_error=max_error))
            values = [pairhist, hierarchical, lof_score, gmm_score, kde]
            for i, (stem, label, value) in enumerate(zip(stems, calibrations, values)):
                assert np.isfinite(value).all()
                key = stem + '__' + label
                variants = u.variants({stem: value}, data, team, calibration=label != 'raw')
                banks[key][team] = variants[key]
                raw_b[i, :, j] = value[:len(b)]
            for i, value in enumerate([pairhist, hierarchical, lof_score, kde]):
                raw_q[i, :, j] = value
            log(stage='target_scores_complete',target=team,completed=j+1,total=len(teams),seconds=time.perf_counter()-started)
    variants = np.stack([np.column_stack([banks[stem + '__' + label][t] for t in teams]) for stem, label in zip(stems, calibrations)])
    models = u.matrices(banks, teams, cross=(.1,))
    cross_scores = np.stack([models[m] for m in METHODS])
    confidence = np.stack([np.column_stack([u.base.confidence(models[m][:, j], normal, rare, private)
                                          for j in range(len(teams))]) for m in METHODS])
    assert cross_scores.shape == (5, 875, 22) and confidence.shape == (5, 1049, 22)
    assert np.isfinite(confidence).all() and (confidence[:, normal] > .6).all() and (confidence <= .9).all()
    assert (confidence[:, private] == 0).all()
    args.output.mkdir(parents=True)
    chosen = pd.DataFrame({'record_id': b.loc[normal, 'record_id'].to_numpy()})
    for team, method in SELECTED.items():
        chosen[str(team)] = confidence[METHODS.index(method), normal, teams.index(team)]
    chosen.to_csv(args.output / 'selected_confidence.csv', index=False, float_format='%.17g')
    partitions = np.where(rare, 'rare', np.where(private, 'nonrare_known_negative', 'ordinary'))
    pd.DataFrame(dict(record_id=b.record_id, partition=partitions)).to_csv(args.output / 'row_partitions.csv', index=False)
    np.savez_compressed(args.output / 'all_method_arrays.npz', methods=np.array(METHODS), teams=np.array(teams),
                        record_ids=b.record_id.to_numpy(), normal_record_ids=b.loc[normal, 'record_id'].to_numpy(),
                        variant_ranks=variants, cross_scores=cross_scores, all_confidence=confidence,
                        raw_b_scores=raw_b, raw_qi7_scores=raw_q, query_owner=owner)
    pd.DataFrame(fit_records).to_csv(args.output / 'gmm_fit_diagnostics.csv', index=False, float_format='%.17g')
    manifest = dict(status='generated_not_submitted',created_utc=datetime.now(timezone.utc).isoformat(),
        base_ref=spec['base_ref'],repository=spec['repository'],source_tree_sha=spec['tree_sha'],
        methods=list(METHODS),selected_by_main_target=SELECTED,targets=teams,normal_rows=875,
        partitions={'ordinary':875,'nonrare_known_negative':150,'rare':24,'known_negative_including_rare':163,'rare_overlap_known_negative':13},
        background={'name':'starter+prelim+main','order':['starter','prelim','main'],'rows':15429,'source_rows_each':5143,
                    'bare_record_id_duplicates_retained':True,'concatenated_csv_sha256':sha_bytes(concatenated)},
        csv_float_parser="pandas.read_csv(float_precision='round_trip')",age_bin_cuts=cuts.tolist(),kde_bandwidth=h0,
        calibration={'county500':{'columns':['prefecture','sex'],'strength':500},'age100':{'columns':['sex','age_bin'],'strength':100},
                     'controls':'all public main AIA Q including self28, excluding the scored target',
                     'formula':'.999*(local midpoint CDF * n/(n+s) + global midpoint CDF * s/(n+s)) + .001*all1049 B average_rank/1049'},
        rank_and_cross={'first':'average_rank on ordinary875 /875','cross':'each target minus .1*row median of the other21 targets in the same method',
                        'confidence':'.6+.3*min_rank(cross_score on ordinary875)/875','no_cross_after_column_selection':True},
        seeds={'GMM_A':20261008,'GMM_C':'20261008+main_target_id','LOF_histogram_alignment':'deterministic; no random seed'},
        fit_labels_used=False,truth_files_read=False,source_loading='AST whitelist of named pure functions; no source module top-level execution/evaluate/main',
        source_files=source_records,input_files=input_records,
        runtime={'python_executable':sys.executable,'python_version':platform.python_version(),'platform':platform.platform(),
                 'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
                 'threadpool_limit':1,'inference_threadpools':inference_threadpools,'kde_query_block_rows':args.block_rows,
                 'LOKY_MAX_CPU_COUNT':os.environ['LOKY_MAX_CPU_COUNT'],
                 'warnings_note':'FutureWarning表示のみ抑制。原alignのint列からfloatへの代入結果をpandas2.3.3で保持。'},
        validation={'KDE_original_geometry_checks':kernel_checks,'KDE_query_category_levels_all_present_in_background':True,
                    'all_GMM_fits_converged':all(r['converged'] for r in fit_records),
                    'GMM_fit_groups':len(fit_records),'selected_rows':len(chosen),'finite_confidence':True,
                    'ordinary_confidence_min':float(confidence[:,normal].min()),'ordinary_confidence_max':float(confidence[:,normal].max()),
                    'known_negative_confidence_all_zero':True},
        output_sha256={p.name:sha_bytes(p.read_bytes()) for p in args.output.iterdir()},
        driver_sha256=sha_bytes(Path(__file__).read_bytes()),input_spec_sha256=sha_bytes(args.input_spec.read_bytes()),
        seconds=time.perf_counter()-started,
        limitation='選定は既存公開予備戦結果に基づく。本戦TPR・優位性・真の会員識別は未確認。希少質問の更新と旧提出保存列の統合は別スクリプトで行う。')
    dump(args.output / 'generation_manifest.json', manifest)
    log(stage='generated',output=str(args.output),selected_rows=len(chosen),truth_read=False,seconds=manifest['seconds'])


if __name__ == '__main__':
    main()
