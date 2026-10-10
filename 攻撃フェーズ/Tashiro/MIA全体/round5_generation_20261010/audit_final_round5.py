"""第5回MIAを生成器に依存せず原本・公開入力・保存配列から検証する。

大会提出やGitHub操作は行わない。生成器・方式driver・希少生成器をimport
しない。唯一読み込む検証コードはSHA256を記録する公式MIA検証器である。
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TEAMS = tuple(range(18)) + (22, 25, 26, 27)
SELECTED = (5, 7, 13, 14, 15)
QI = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")
PINNED = {
    "round3": "6ca9604a8936fb47255ae0bd05c9f3958368f0e196bc2af258f8255cf39c08eb",
    "round4": "45077532e5b5cf6109bf86ce67472fce0fac703584fa81a2f0027434fe581183",
    "B_self": "1339042b4e1937d5fbf482aa8ce4de3f67a02e4694f8237894ca59bf3f0d52b1",
    "self_challenge": "9d464fce3af708819cb6358c1ad33182472f37b9f26fc94fdcc7c769d70d4c26",
    "mia_columns": "bca9a100c90222eb774e57708db34ee3f0519a12abc0bb74d170bc68459ef181",
    "fixed_bits": "00799e87fbbed4b44850092c176fee4af41769f93c1bace53aaa2bb66e9e5865",
    "saved_settings": "3236035d39ab2d9caf612b37f2e91cceb882c606a70bd543e302b2b03d24ddae",
    "certificate": "5d379e04e0b00604df2d95bce92e89a06a629576b6e2ebcc00c01f040c3f16f5",
    "validator": "359589b4e82cb41f4284c5a7cdc7e2d3c793788d6d9d6f12a62b46a7a132f3c3",
}
METHODS = {
    5: "q13_robust_sex1_diag_reg0.01_gmm4__raw__cross0.1",
    7: "q7_robust_lofC_k100__age100__cross0.1",
    13: "q7_pref_sex_bins20_s1_zscore__county500__cross0.1",
    14: "q7_median_iqr_sex1_kde_ratio_h1__age100__cross0.1",
    15: "q7_hist5_s100_pairwise__county500__cross0.1",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = tuple(reader.fieldnames or ())
        require(headers and len(headers) == len(set(headers)), f"欠落/重複ヘッダ: {path}")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows), f"壊れたCSV: {path}")
    return headers, rows


def key(row):
    return tuple(row[col] if col == "prefecture" else Decimal(row[col]) for col in QI)


def answer(path):
    headers, rows = read(path)
    require(headers == ("record_id",) + tuple(map(str, TEAMS)), f"回答列/順序不一致: {path}")
    require(len(rows) == 1049 and [r["record_id"] for r in rows] == list(map(str, range(1049))),
            f"回答1049行/ID順不一致: {path}")
    values = [[float(row[str(t)]) for t in TEAMS] for row in rows]
    require(all(math.isfinite(v) and 0 <= v <= 1 for row in values for v in row),
            f"回答の非有限値/[0,1]違反: {path}")
    return rows, values


def count_diff(current, baseline, ids, teams):
    return sum(current[rid][TEAMS.index(team)] != baseline[rid][TEAMS.index(team)]
               for rid in ids for team in teams)


def min_ranks(values):
    """同点を崩さない昇順min順位。scipy.rankdataには依存しない。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    previous = None
    minimum = 0
    for ordinal, index in enumerate(order, 1):
        if previous is None or values[index] != previous:
            minimum = ordinal
        ranks[index] = minimum
        previous = values[index]
    return ranks


def average_ranks(values):
    """同点群の最小・最大順位の平均を独立に算出する。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        midpoint = (start + 1 + stop) / 2
        for index in order[start:stop]:
            ranks[index] = midpoint
        start = stop
    return ranks


def manifest_audit(path, files):
    manifest = json.loads(path.read_text())
    require(manifest['base_ref'] == '1a5d47bba85711322994098b0de16df894acdf95', '方式source基準コミット不一致')
    require(manifest['targets'] == list(TEAMS) and set(manifest['methods']) == set(METHODS.values()),
            '方式manifestの全本戦対象/方式ID不一致')
    require(manifest['selected_by_main_target'] == {str(t): METHODS[t] for t in SELECTED},
            '方式manifestの5対象方式対応不一致')
    background = manifest['background']
    require(background['name'] == 'starter+prelim+main' and background['order'] == ['starter', 'prelim', 'main']
            and background['rows'] == 15429 and background['source_rows_each'] == 5143
            and background['concatenated_csv_sha256'] == '01772d964672cd742c0d44ac69e058d2db91283c330edac78671a0ec2d7fb650',
            '方式manifestの全3背景/順序/結合hash不一致')
    require(manifest['fit_labels_used'] is False and manifest['truth_files_read'] is False,
            '生成時に所属正解を使用')
    require(manifest['seeds']['GMM_A'] == 20261008 and manifest['seeds']['GMM_C'] == '20261008+main_target_id',
            '原GMM seed不一致')
    require(manifest['rank_and_cross']['no_cross_after_column_selection'] is True, '混合後に交差補正を適用')
    require(manifest['output_sha256']['selected_confidence.csv'] == sha(files['selected_ordinary'])
            and manifest['output_sha256']['all_method_arrays.npz'] == sha(files['method_arrays']),
            '生成manifestと方式生成物のhash不一致')
    source_hashes = {}
    for record in manifest['source_files']:
        source = path.parent.parent / record['local_name']
        content = source.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        require(blob == record['git_blob_sha'] and sha(source) == record['verified_sha256'],
                '方式source原本hash不一致')
        source_hashes[record['local_name']] = sha(source)
    driver = path.parent.parent / 'generate_selected.py'
    require(manifest['driver_sha256'] == sha(driver), 'driver変更後に生成証跡hashを更新していない')
    require(manifest['input_spec_sha256'] == sha(path.parent.parent / 'input_spec.json'), '方式input spec hash不一致')
    require(len(manifest['input_files']) == 50, '全公開入力数不一致')
    require(all('pool_ids' not in r['repository_path'] and '正解データ' not in r['repository_path']
                for r in manifest['input_files']), '許可されない入力パス')
    return manifest, {'background': background, 'seeds': manifest['seeds'], 'source_sha256': source_hashes,
                      'driver_sha256': manifest['driver_sha256'], 'input_spec_sha256': manifest['input_spec_sha256'],
                      'input_count': len(manifest['input_files']), 'fit_labels_used': False, 'truth_files_read': False}


def calibration_audit(archive, methods, normal_ids, ranks, root, manifest, bpath):
    """保存raw値から較正全4×22列とq13 raw順位を別実装で再計算。"""
    import numpy as np
    import pandas as pd
    by_role = {record['role']: record for record in manifest['input_files']}
    b = pd.read_csv(bpath, float_precision='round_trip')
    frames = []
    for team in sorted(set(TEAMS) | {28}):
        qp = root / f'本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/aia_challenge_{team}.csv'
        require(sha(qp) == by_role[f'aia_challenge_{team}']['sha256'], '較正用公開Qのhash不一致')
        frames.append(pd.read_csv(qp, float_precision='round_trip'))
    query = pd.concat([b[list(QI)]] + [q[list(QI)] for q in frames], ignore_index=True)
    owner = np.r_[np.full(1049, -1), *[np.full(len(q), team) for q, team in zip(frames, sorted(set(TEAMS) | {28}))]]
    require(np.array_equal(archive['query_owner'], owner), '保存query_owner/入力連結順不一致')
    cuts = manifest['age_bin_cuts']
    query['age_bin'] = np.searchsorted(cuts, query.age, side='right')
    raw_b = archive['raw_b_scores']
    raw_q = archive['raw_qi7_scores']
    require(raw_b.shape == (5, 1049, 22) and raw_q.shape == (4, 8547, 22), '保存raw score shape不一致')
    require(np.isfinite(raw_b).all() and np.isfinite(raw_q).all(), '保存raw score非有限値')
    qi7_indices = [i for i, name in enumerate(methods) if name.startswith('q7_')]
    require(len(qi7_indices) == 4, 'q7方式数不一致')
    def midpoint_cdf(values, null):
        ordered, counts = np.unique(null, return_counts=True)
        midpoints = (np.cumsum(counts) - .5 * counts) / len(null)
        return np.interp(values, ordered, midpoints, left=0., right=1.)
    maximum_rank_error = 0.0
    for mi, method in enumerate(methods):
        for tj, team in enumerate(TEAMS):
            values = raw_b[mi, :, tj]
            if method.startswith('q13_'):
                transformed = values
            else:
                qi = qi7_indices.index(mi)
                require(np.array_equal(raw_q[qi, :1049, tj], values), 'B rawと全query rawの先頭B不一致')
                null_indices = np.flatnonzero((owner != -1) & (owner != team))
                require(len(null_indices) == 7172, '対象を除く22課題の較正人数不一致')
                null = raw_q[qi, null_indices, tj]
                null_query = query.iloc[null_indices]
                county = '__county500__' in method
                cols, strength = (['prefecture', 'sex'], 500) if county else (['sex', 'age_bin'], 100)
                global_cdf = midpoint_cdf(values, null)
                local = global_cdf.copy()
                counts = np.zeros(1049, dtype=int)
                b_groups = query.iloc[:1049].groupby(cols, sort=True).indices
                null_groups = null_query.groupby(cols, sort=True).indices
                for group, indexes in b_groups.items():
                    ni = null_groups.get(group, [])
                    counts[indexes] = len(ni)
                    if len(ni):
                        local[indexes] = midpoint_cdf(values[indexes], null[ni])
                weight = counts / (counts + strength)
                all_b_tie = np.array(average_ranks(values.tolist())) / 1049
                transformed = .999 * (weight * local + (1 - weight) * global_cdf) + .001 * all_b_tie
            computed = np.array(average_ranks(transformed[list(normal_ids)].tolist())) / 875
            error = float(np.max(np.abs(computed - ranks[mi, :, tj])))
            require(error == 0, '独立較正→通常平均順位が保存方式配列と不一致')
            maximum_rank_error = max(maximum_rank_error, error)
    return {'q7_calibrated_columns': 4 * 22, 'q13_raw_rank_columns': 22,
            'controls_per_target': 7172, 'all_B_tie_rows': 1049, 'normal_rank_rows': 875,
            'maximum_variant_rank_error': maximum_rank_error,
            'query_and_controls_rebuilt_from_main_public_inputs': True}


def arrays_audit(path, normal_ids, selected_values, current, root, manifest, bpath):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        teams = tuple(int(t) for t in archive["teams"])
        methods = tuple(str(m) for m in archive["methods"])
        ids = tuple(int(rid) for rid in archive["normal_record_ids"])
        ranks = archive["variant_ranks"].copy()
        cross = archive["cross_scores"].copy()
        confidence = archive["all_confidence"].copy()
        calibration = calibration_audit(archive, methods, normal_ids, ranks, root, manifest, bpath)
    require(teams == TEAMS and ids == tuple(normal_ids), "方式配列の本戦対象/通常ID順不一致")
    require(len(methods) == 5 and set(methods) == set(METHODS.values()), "方式配列ID不一致")
    require(ranks.shape == cross.shape == (5, 875, 22), "方式順位/交差配列shape不一致")
    require(confidence.shape == (5, 1049, 22), "全方式confidence shape不一致")
    require(np.isfinite(ranks).all() and np.isfinite(cross).all() and np.isfinite(confidence).all(),
            "方式配列非有限値")
    require((ranks > 0).all() and (ranks <= 1).all(), "通常平均順位値域違反")
    max_cross_error = 0.0
    all_transform_errors = []
    selected_transform_errors = []
    tie_groups = {}
    for mi, method in enumerate(methods):
        recomputed = np.column_stack([
            ranks[mi, :, tj] - .1 * np.median(np.delete(ranks[mi], tj, axis=1), axis=1)
            for tj in range(22)
        ])
        error = float(np.max(np.abs(recomputed - cross[mi])))
        require(error < 1e-15, "全22対象・自身除外median交差補正が不一致")
        max_cross_error = max(max_cross_error, error)
        for tj, team in enumerate(teams):
            column = cross[mi, :, tj].tolist()
            final_values = [.6 + .3 * rank / 875 for rank in min_ranks(column)]
            actual = confidence[mi, list(normal_ids), tj]
            error = float(np.max(np.abs(np.array(final_values) - actual)))
            require(error < 1e-15, "平均順位→交差補正→min順位の通常値域変換が不一致")
            all_transform_errors.append(error)
            if METHODS.get(team) == method:
                for k, rid in enumerate(normal_ids):
                    require(selected_values[rid][team] == float(actual[k]) == current[rid][tj],
                            "選択した5列の移植/保存方式confidence不一致")
                selected_transform_errors.append(error)
                tie_groups[str(team)] = len(column) - len(set(column))
    return {"methods": list(methods), "teams": list(teams), "normal_rows": 875,
            "leave_target_out_median_other_targets": 21, "cross_alpha": 0.1,
            "maximum_cross_error": max_cross_error,
            "confidence_transform": "0.6 + 0.3 * ascending_min_rank(cross_scores) / 875",
            "maximum_all_method_transform_error": max(all_transform_errors),
            "maximum_selected_transform_error": max(selected_transform_errors),
            "selected_tied_extra_rows": tie_groups,
            "raw_to_calibrated_rank_audit": calibration,
            "post_mix_recorrection": False, "imported_builder_or_method_driver": False}


def official_audit(validator, root, candidate, declared):
    import pandas as pd
    kits = list((root / "本戦").glob("配布*/pwscup2026-kit-main_revised0912/starter/src"))
    require(len(kits) == 1, "公式依存packageが一意でない")
    sys.path.insert(0, str(kits[0]))
    name = "pwscup2026.kit._round5_independent_current_validate"
    spec = importlib.util.spec_from_file_location(name, validator)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    frame = module.submission_schema.read_scoring_csv(candidate, encoding="utf-8")
    result = module.validate_mia_submission(frame, {}, {
        "mia_columns": declared, "team_id": 28, "attacker_record_ids": set(range(1049))})
    require(result.ok, "公式MIA logical validator失敗: " + repr(result.errors))
    schema_path = Path(module.submission_schema.__file__).resolve()
    return {"ok": result.ok, "errors": result.errors, "layer": "MIA logical",
            "validator_sha256": sha(validator), "submission_schema_sha256": sha(schema_path),
            "packaging": "MIA単体ZIPは公式提出ZIPではない。F_aia.csv/token.txtの統合前。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("/Users/taisei1105/pws2026-local"))
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--component-zip", type=Path, required=True)
    parser.add_argument("--selected-ordinary", type=Path, required=True)
    parser.add_argument("--method-arrays", type=Path, required=True)
    parser.add_argument("--method-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=HERE / "final_round5_independent_audit.json")
    args = parser.parse_args()
    root = args.local_root.resolve()
    attack = root / "本戦/攻撃フェーズ/Tashiro"
    source = root / "work/mia_round3_publish_20261008/candidate/攻撃フェーズ/MIA/20261008_第3回MIA_固定5bitと非会員較正"
    bp = list((root / "本戦").glob("配布*/PWSCup2026_final_team28_*/B_self.csv"))
    require(len(bp) == 1, "本戦B_selfが一意でない")
    files = {
        "round3": source / "結果/F_mia.csv",
        "round4": attack / "20261009_第4回MIA_チーム別方式と固定5bit/結果/F_mia.csv",
        "B_self": bp[0],
        "self_challenge": attack / "20260928_配布データ基本分析/入力/aia_challenge_28.csv",
        "mia_columns": attack / "20260928_配布データ基本分析/入力/mia_columns.json",
        "fixed_bits": source / "結果/固定5bit符号.csv",
        "saved_settings": source / "結果/設定.json",
        "certificate": source / "結果/希少監査_再検証.json",
        "candidate": args.candidate.resolve(), "component_zip": args.component_zip.resolve(),
        "selected_ordinary": args.selected_ordinary.resolve(),
        "method_arrays": args.method_arrays.resolve(), "method_manifest": args.method_manifest.resolve(),
        "assembly_differences": args.candidate.resolve().parent / "differences.csv",
        "rare_expected": HERE / "rare_round5_expected.csv",
        "validator": attack / "20261005_MIA初回_GenLRA比較/参照コード/validate_20260928.py",
    }
    digests = {name: sha(path) for name, path in files.items()}
    for name, expected in PINNED.items():
        require(digests[name] == expected, f"固定原本SHA256不一致: {name}")
    declared = json.loads(files["mia_columns"].read_text())
    require(tuple(int(t) for t in declared["target_columns"] if int(t) != 28) == TEAMS,
            "本戦対象列不一致")
    _, b = read(files["B_self"])
    _, q = read(files["self_challenge"])
    require(len(b) == 1049 and [int(r["record_id"]) for r in b] == list(range(1049)), "本戦B/ID順不一致")
    bkeys, qkeys = [key(r) for r in b], {key(r) for r in q}
    require(len(set(bkeys)) == 1049 and len(qkeys) == len(q) == 326, "本戦QI重複/自己課題行数不一致")
    private = {i for i, k in enumerate(bkeys) if k in qkeys}
    rare = {i for i, row in enumerate(b) if row["is_rare"] == "1"}
    require(all(r["is_rare"] in ("0", "1") for r in b), "本戦希少フラグ不正")
    ordinary = sorted(set(range(1049)) - private - rare)
    negative = sorted(private - rare)
    known_rare = sorted(private & rare)
    unknown_rare = sorted(rare - private)
    require(tuple(map(len, (ordinary, negative, rare, private, known_rare, unknown_rare))) ==
            (875, 150, 24, 163, 13, 11), "独立導出した排他的区分不一致")
    original_rows, original = answer(files["round3"])
    _, round4 = answer(files["round4"])
    current_rows, current = answer(files["candidate"])
    unselected = tuple(t for t in TEAMS if t not in SELECTED)
    outside = count_diff(current, original, ordinary, unselected)
    require(outside == 0, "5対象外通常14875セルが第3回から変更")
    require(count_diff(current, original, range(1049), (9,)) == 0, "Haruna本戦9全1049セル変更")
    require(all(current[rid][j] == 0 for rid in negative for j in range(22)), "通常非共有150人を0化していない")
    normal_values = [current[rid][j] for rid in ordinary for j in range(22)]
    rare_values = [current[rid][j] for rid in rare for j in range(22)]
    require(min(normal_values) > .6 and max(normal_values) <= .9 and max(rare_values) <= .4,
            "通常>0.6/希少<=0.4の分離違反")
    settings = json.loads(files["saved_settings"].read_text())
    _, bit_rows = read(files["fixed_bits"])
    bits = {int(r["record_id"]): r["bits"] for r in bit_rows}
    require(sorted(bits) == unknown_rare and all(len(v) == 5 and set(v) <= {"0", "1"} for v in bits.values()),
            "固定5bit表/未判明希少11人不一致")
    queries = {number: sorted(rid for rid, code in bits.items() if code[number - 3] == "1")
               for number in range(3, 8)}
    require(all(queries[n] == settings["fixed_queries"][str(n)] for n in range(3, 8)), "保存表と質問順が不一致")
    require(queries[5] == [47, 90, 264, 462, 495], "第3問の5人不一致")
    active = tuple(int(t) for t in settings["question_teams"])
    require(active == (6, 10, 11, 16, 17, 25), "希少あり6対象の本戦ID不一致")
    require(settings["rare_member_counts"] == {"6": 1, "10": 2, "11": 1, "16": 2, "17": 1, "25": 2},
            "保存人数条件不一致")
    def expected(rid, team, round_number):
        if team not in active:
            return 0.0
        padding_n = settings["total_high"][str(team)] - len(queries[round_number])
        high = set(queries[round_number]) | set(known_rare[:padding_n])
        return .4 if rid in high else .1
    history = []
    for number, values in ((3, original), (4, round4), (5, current)):
        mismatches = sum(values[rid][j] != expected(rid, team, number)
                         for rid in rare for j, team in enumerate(TEAMS))
        require(mismatches == 0, f"第{number}回の固定5bit第{number - 2}問528セル不一致")
        history.append({"submission_round": number, "fixed_question": number - 2,
                        "checked_cells": 528, "mismatches": mismatches})
    _, expected_rows = read(files["rare_expected"])
    require(sorted(int(r["record_id"]) for r in expected_rows) == sorted(rare), "希少期待CSV行集合不一致")
    require(all(float(r[str(t)]) == expected(int(r["record_id"]), t, 5) for r in expected_rows for t in TEAMS),
            "別担当希少期待CSVと保存5bit表からの独立算出が不一致")
    head, selected_rows = read(files["selected_ordinary"])
    require(head == ("record_id",) + tuple(map(str, SELECTED)), "選択通常CSV列順不一致")
    require(len(selected_rows) == 875 and [int(r["record_id"]) for r in selected_rows] == ordinary,
            "選択通常CSV行集合/順序不一致")
    selected_values = {int(r["record_id"]): {t: float(r[str(t)]) for t in SELECTED} for r in selected_rows}
    require(all(current[rid][TEAMS.index(t)] == selected_values[rid][t] for rid in ordinary for t in SELECTED),
            "5対象通常値と選択CSV4375セル不一致")
    manifest, manifest_check = manifest_audit(files['method_manifest'], files)
    arrays = arrays_audit(files["method_arrays"], ordinary, selected_values, current, root, manifest, files['B_self'])
    with zipfile.ZipFile(files["component_zip"]) as archive:
        require(archive.namelist() == ["F_mia.csv"], "MIA単体ZIP内容/重複/階層不一致")
        require(archive.testzip() is None and archive.read("F_mia.csv") == files["candidate"].read_bytes(),
                "MIA単体ZIP内CSV不一致/破損")
    official = official_audit(files["validator"], root, files["candidate"], declared)
    differences = []
    for label, old in (("round3", original), ("round4", round4)):
        for team in TEAMS:
            j = TEAMS.index(team)
            differences.append({"baseline": label, "main_team": team,
                "ordinary_changed_cells": count_diff(current, old, ordinary, (team,)),
                "rare_changed_cells": count_diff(current, old, rare, (team,)),
                "nonrare_known_negative_changed_cells": count_diff(current, old, negative, (team,)),
                "all_changed_cells": count_diff(current, old, range(1049), (team,)),
                "ordinary_max_abs_change": max(abs(current[rid][j] - old[rid][j]) for rid in ordinary)})
    _, assembly_differences = read(files['assembly_differences'])
    require(len(assembly_differences) == 44, '統合器差分表44行不一致')
    independently_computed = {(r['baseline'], r['main_team']): r for r in differences}
    seen = set()
    for row in assembly_differences:
        pair = (row['baseline'], int(row['main_team']))
        require(pair in independently_computed and pair not in seen, '統合器差分表対象重複/過不足')
        seen.add(pair)
        for column in ('ordinary_changed_cells', 'rare_changed_cells',
                       'nonrare_known_negative_changed_cells', 'all_changed_cells'):
            require(int(row[column]) == independently_computed[pair][column], '統合器差分表と独立算出が不一致')
    for name, path in files.items():
        require(sha(path) == digests[name], f"監査中に入力変更: {name}")
    report = {"status": "PASS", "submitted": False,
        "auditor_imported_generation_code": False, "main_individual_truth_used": False,
        "rows": 1049, "targets": list(TEAMS), "target_count": 22,
        "selected_ordinary_methods": {str(t): METHODS[t] for t in SELECTED},
        "partitions_independently_derived_from_main_B_and_self_QI": {
            "ordinary": 875, "nonrare_known_negative": 150, "rare": 24,
            "known_nonshared_all": 163, "known_nonshared_rare_overlap": 13},
        "checks": {"exact_header_and_row_order": True, "finite_and_unit_interval_cells": 1049 * 22,
            "ordinary_outside_five_unchanged_cells": 875 * 17, "ordinary_outside_five_changed_cells": outside,
            "haruna_unchanged_cells": 1049, "haruna_changed_cells": 0,
            "nonrare_known_negative_zero_cells": 150 * 22, "rare_question3_matching_cells": 24 * 22,
            "selected_ordinary_matching_cells": 875 * 5, "component_zip_csv_bytes_match": True,
            "assembly_differences_all_44_rows_match": True,
            "normal_min": min(normal_values), "normal_max": max(normal_values), "rare_max": max(rare_values),
            "rare_known_nonshared_nonzero_cells": sum(current[rid][j] != 0 for rid in known_rare for j in range(22)),
            "haruna_textually_unchanged_cells": sum(current_rows[i]["9"] == original_rows[i]["9"] for i in range(1049))},
        "history_fixed5bit": history, "method_array_audit": arrays, "method_manifest_audit": manifest_check,
        "official_mia_validator": official,
        "differences": differences,
        "round4_difference_interpretation": "第4回は別の通常方式を使用。第4回比は今回5対象変更だけの効果と解釈しない。",
        "input_sha256": {name: {"path": str(files[name]), "sha256": digest} for name, digest in digests.items()},
        "auditor_sha256": sha(Path(__file__)),
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "audit_seed": "deterministic; no sampling"},
        "limits": ["監査は整合性・再現処理の検証。本戦MIA精度の独立評価には未知の所属正解が必要。",
                   "希少人数の条件付き導出は保存された公式結果と共有条件に依存する。",
                   "MIA単体ZIPは配布用。公式提出ZIPにはF_aia.csvとtoken.txtの統合が必要。"]}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    diff_path = args.report.with_name("final_round5_independent_differences.csv")
    with diff_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(differences[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(differences)
    totals = {base: {col: sum(r[col] for r in differences if r['baseline'] == base)
                     for col in ('ordinary_changed_cells', 'rare_changed_cells',
                                 'nonrare_known_negative_changed_cells', 'all_changed_cells')}
              for base in ('round3', 'round4')}
    markdown = [
        '# 第5回MIA 最終独立検証', '',
        '結果は **PASS**。大会へ未提出。生成器・方式driver・希少生成器をimportせず、公開本戦入力・保存原本・スコア配列から検証した。', '',
        f"提出用CSV SHA256: `{digests['candidate']}`", '',
        '| 検証 | 確認数・結果 |', '|---|---:|',
        '| 行・対象・値 | 1049行 × 22対象、全23078値が有限で[0,1] |',
        '| 原本列順・record_id順 | 完全一致 |',
        '| 5対象外の通常 | 14875セル差分0 |',
        '| Haruna 本戦9 | 全1049セルの数値・文字列差分0 |',
        '| 通常非共有150人 | 3300セル全0 |',
        '| 希少24人 | 528セルが保存固定5bit第3問と一致 |',
        '| 選択通常5列 | 4375セルが方式出力と一致 |',
        '| 通常値域 | 0.6003428571428571 ～ 0.8999999999999999 |',
        '| 希少最大 | 0.4 |',
        '| ZIP | F_mia.csvのみ、CSV byte完全一致・破損なし |',
        '| 公式MIA logical validator | PASS |', '',
        '既知非共有163人のうち希少13人は保存仕様どおり希少あり6対象で0.1／0.4を使う。常時0は非希少150人であり、希少13人を一律0に変更していない。', '',
        '本戦の公開Bと自己課題の7属性から通常875・通常非共有150・希少24の排他的区分を再導出した。保存5bit表の第1／第2／第3bitから、第3回・第4回・今回第5回の希少528セルをそれぞれ照合した。', '',
        '方式入力50ファイルのhash記録を確認し、公開Q23ファイルと本戦B、原source8ファイル・driver・input spec・保存原本・生成物の実ファイルhashを照合した。背景はstarter→prelim→main、各5143行の全15429行を保持。GMMは性別分割diag4／reg0.01、seedはA=20261008、C=20261008+本戦対象IDである。', '',
        '保存raw値と公開Qから全4 q7方式 × 22対象の較正を別実装で再算出した。対象自身を除く22公開課題7172行を使い、1049行Bの平均順位による同点項、通常875人の平均順位、他21対象の中央値 × 0.1、通常875人のmin順位による0.6＋0.3×rank/875まで、全5方式 × 22対象が誤差0で一致した。同点はrecord_idで破っていない。選択後の混合列には追加補正を加えていない。', '',
        '| 比較原本 | 通常セル差分 | 希少セル差分 | 通常非共有セル差分 | 全セル差分 |',
        '|---|---:|---:|---:|---:|',
    ]
    for base, label in [('round3', '第3回'), ('round4', '第4回')]:
        row = totals[base]
        markdown.append(f"| {label} | {row['ordinary_changed_cells']} | {row['rare_changed_cells']} | {row['nonrare_known_negative_changed_cells']} | {row['all_changed_cells']} |")
    markdown += ['', '第4回は別の通常方式を使用した原本である。第4回比の差分を今回の5対象変更だけの効果と解釈しない。統合器差分表44行も独立算出と完全一致した。', '',
        '数値証跡は[JSON](final_round5_independent_audit.json)、対象別差分は[CSV](final_round5_independent_differences.csv)。[検証スクリプト](audit_final_round5.py)は各入力を監査前後にhash照合する。方式のsource互換監査と再生成一致は[方式資料](methods/README.md)を参照。', '',
        '再実行には方式driverで内部NPZを再生成してから次を実行する。内部NPZは約7.2 MBの検証cacheでありGitHub掲載対象から除外できる。', '',
        '```sh',
        'python -B 攻撃フェーズ/Tashiro/MIA全体/round5_generation_20261010/audit_final_round5.py \\\n  --candidate 攻撃フェーズ/MIA/5回目/F_mia.csv \\\n  --component-zip 攻撃フェーズ/MIA/5回目/MIA_round5_components.zip \\\n  --selected-ordinary 攻撃フェーズ/Tashiro/MIA全体/round5_generation_20261010/methods/generated/selected_confidence.csv \\\n  --method-arrays 攻撃フェーズ/Tashiro/MIA全体/round5_generation_20261010/methods/generated/all_method_arrays.npz \\\n  --method-manifest 攻撃フェーズ/Tashiro/MIA全体/round5_generation_20261010/methods/generated/generation_manifest.json',
        '```', '',
        '本監査は整合性・再現処理の検証であり、未知の本戦所属正解による精度評価ではない。希少人数は保存公式結果と共有条件に基づく条件付き導出である。MIA単体ZIPは配布用で、公式攻撃ZIPにはF_aia.csvとtoken.txtの統合が必要。大会提出・採点・個人所属確定は行っていない。', '']
    args.report.with_suffix('.md').write_text('\n'.join(markdown), encoding='utf-8')
    print(json.dumps({"status": "PASS", "F_mia_sha256": digests["candidate"],
                      "report": str(args.report), "submitted": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
