"""第3回MIA通常者部分の独立監査。builderや既存計算器はimportしない。

入力CSVから混合Gen-LRA、非会員較正、他対象補正、提出順位を再計算する。
書込先は --output の監査記録のみ。既存ファイルは上書きしない。
"""
from __future__ import annotations

import argparse
import hashlib
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

NUMERIC = ("age", "SBP", "TG", "HDL", "ALT")
CATEGORICAL = ("sex", "prefecture")
QI = NUMERIC + CATEGORICAL
TARGETS = tuple(range(18)) + (22, 25, 26, 27)
SELF = 28
TOLERANCE = 1e-10


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_child(parent, name):
    matches = [p for p in Path(parent).iterdir()
               if unicodedata.normalize("NFC", p.name) == unicodedata.normalize("NFC", name)]
    require(len(matches) == 1, f"入力を一意に選べません: {parent}/{name}")
    return matches[0]


class Reader:
    def __init__(self):
        self.hashes = {}

    def csv(self, path):
        path = Path(path).resolve()
        self.hashes[str(path)] = sha(path)
        return pd.read_csv(path, float_precision="round_trip")

    def json(self, path):
        path = Path(path).resolve()
        self.hashes[str(path)] = sha(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def unchanged(self):
        require(all(sha(Path(p)) == h for p, h in self.hashes.items()), "読み取った入力が変化しました")


def input_data(root, reader):
    distributed = canonical_child(root / "本戦", "配布データ")
    own_dirs = list(distributed.glob("PWSCup2026_final_team28_*"))
    require(len(own_dirs) == 1, "自己データフォルダが一意ではありません")
    own = own_dirs[0]
    source = root / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
    b, a = reader.csv(own / "B_self.csv"), reader.csv(own / "A_bg.csv")
    require(len(b) == 1049 and len(a) == 5143, "Bまたは背景の人数が違います")
    require(b.record_id.is_unique and set(b.record_id) == set(range(1049)), "自己IDが不正です")
    require(not b.duplicated(list(QI)).any(), "自己QI7に重複があります")
    require(set(b.is_rare.unique()) <= {0, 1, False, True}, "is_rareが0/1ではありません")
    qs = {t: reader.csv(source / f"aia_challenge_{t}.csv") for t in TARGETS + (SELF,)}
    for t, q in qs.items():
        require(len(q) == 326, f"Team{t} AIA課題の人数が違います")
        require(q.challenge_row_id.is_unique and not q.duplicated(list(QI)).any(),
                f"Team{t} AIA課題に重複があります")
    own_qi = set(qs[SELF][list(QI)].itertuples(index=False, name=None))
    private = np.array([key in own_qi for key in b[list(QI)].itertuples(index=False, name=None)])
    rare = b.is_rare.to_numpy(bool)
    ordinary = ~private & ~rare
    require(int(private.sum()) == 163 and int(rare.sum()) == 24, "163人/希少24人の前提が違います")
    require(int((private & rare).sum()) == 13 and int(ordinary.sum()) == 875,
            "既知希少13人/未知通常875人の前提が違います")
    require(int((private & ~rare).sum()) == 150, "既知通常非会員150人が違います")
    actual_c_ids = {int(p.stem.removeprefix("C_")) for p in source.glob("C_*.csv")}
    require(actual_c_ids == set(TARGETS) | {SELF}, "公開Cのチーム集合が違います")
    cs = {t: reader.csv(source / f"C_{t}.csv") for t in TARGETS}
    require(SELF not in cs and tuple(cs) == TARGETS, "自己Cが攻撃対象へ混入しました")
    return a, b, qs, cs, private, rare, ordinary


def numeric(frame):
    values = frame[list(NUMERIC)].to_numpy(dtype=np.float64, copy=True)
    for index in (2, 4):
        require(np.all(values[:, index] > -1), "log1pの定義域外です")
        values[:, index] = np.log1p(values[:, index])
    require(np.isfinite(values).all(), "数値属性が有限ではありません")
    return values


def squared_distance(left, right):
    # 属性ごとの積み上げで距離を作り、内積展開の桁落ちを避ける。
    result = np.zeros((len(left), len(right)), dtype=np.float64)
    for index in range(left.shape[1]):
        result += np.square(left[:, index, None] - right[None, :, index])
    return result


def rank(values, mode="average"):
    """同値をひとまとめにする独立順位器。IDを同点解消に使わない。"""
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all(), "順位対象が不正です")
    require(mode in ("average", "min"), "順位modeが不正です")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    answer = np.empty(len(values), dtype=float)
    for start, end in zip(starts, ends):
        answer[order[start:end]] = start + 1 if mode == "min" else (start + 1 + end) / 2
    return answer


def mixed_lra(a, query, c, chunk):
    """参照密度と局所追加効果を独立に構成。queryにはBと非会員課題が入る。"""
    ref = numeric(a)
    quartiles = np.percentile(ref, [25, 75], axis=0)
    scale = (quartiles[1] - quartiles[0]) / 1.349
    require(np.all(scale > 0), "IQR尺度が0以下です")
    ref = ref / scale
    syn, candidates = numeric(c) / scale, numeric(query) / scale
    h = (4.0 / 7.0 / len(a)) ** (1.0 / 9.0)
    kernel_factor = -0.5 / (h * h)
    frequencies = {}
    for col in CATEGORICAL:
        categories = set(a[col]) | set(query[col]) | set(c[col])
        counts = a[col].value_counts().to_dict()
        frequencies[col] = {x: (counts.get(x, 0) + 1) / (len(a) + len(categories))
                            for x in categories}
    base = np.empty(len(c), dtype=float)
    # C側を分割するので参照5143人×全queryの大行列を保持しない。
    for start in range(0, len(c), chunk):
        stop = min(start + chunk, len(c))
        log_kernel = squared_distance(syn[start:stop], ref) * kernel_factor
        for col in CATEGORICAL:
            equal = c[col].to_numpy()[start:stop, None] == a[col].to_numpy()[None, :]
            smooth = np.array([frequencies[col][x] for x in c[col].to_numpy()[start:stop]])[:, None]
            log_kernel += np.log(0.98 * equal + 0.02 * smooth)
        maximum = np.max(log_kernel, axis=1)
        base[start:stop] = maximum + np.log(np.exp(log_kernel - maximum[:, None]).sum(axis=1))
    answer = np.empty(len(query), dtype=float)
    for start in range(0, len(query), chunk):
        stop = min(start + chunk, len(query))
        numerical_distance = squared_distance(candidates[start:stop], syn)
        distance = numerical_distance.copy()
        log_kernel = numerical_distance * kernel_factor
        for col in CATEGORICAL:
            equal = query[col].to_numpy()[start:stop, None] == c[col].to_numpy()[None, :]
            smooth = np.array([frequencies[col][x] for x in c[col].to_numpy()])[None, :]
            log_kernel += np.log(0.98 * equal + 0.02 * smooth)
            distance += ~equal
        nearby = np.argsort(distance, axis=1, kind="stable")[:, :10]
        log_ratio = np.take_along_axis(log_kernel - base[None, :], nearby, axis=1)
        local_delta = np.logaddexp(0.0, log_ratio) - np.log1p(1.0 / len(a))
        answer[start:stop] = local_delta.sum(axis=1)
    require(np.isfinite(answer).all(), "Gen-LRAの結果が有限ではありません")
    return answer


def cdf(values, controls):
    """同値群の中点確率を補間。既存のcalibrate関数は呼ばない。"""
    ordered = np.sort(np.asarray(controls), kind="mergesort")
    require(len(ordered) > 0 and np.isfinite(ordered).all(), "非会員スコアが不正です")
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(ordered)]
    midpoints = (starts + ends) / (2.0 * len(ordered))
    return np.interp(values, ordered[starts], midpoints, left=0.0, right=1.0)


def calibration(raw, b, null_scores, controls, shrink=500.0):
    global_distribution = cdf(raw, null_scores)
    corrected = global_distribution.copy()
    bkeys = list(zip(b.prefecture, b.sex))
    qkeys = list(zip(controls.prefecture, controls.sex))
    for group in set(bkeys):
        candidate_mask = np.array([key == group for key in bkeys])
        control_mask = np.array([key == group for key in qkeys])
        count = int(control_mask.sum())
        if count:
            local = cdf(raw[candidate_mask], null_scores[control_mask])
            corrected[candidate_mask] = (count * local + shrink * global_distribution[candidate_mask]) / (count + shrink)
    return 0.999 * corrected + 0.001 * rank(raw) / len(raw)


def contrast(ordinary_rank, alpha=0.1, include_self=False):
    out = ordinary_rank.copy()
    for column in range(ordinary_rank.shape[1]):
        other_columns = list(range(ordinary_rank.shape[1]))
        if not include_self:
            other_columns.remove(column)
        out[:, column] -= alpha * np.median(ordinary_rank[:, other_columns], axis=1)
    return out


def submitted(cross):
    return np.column_stack([0.6 + 0.3 * rank(cross[:, column], "min") / 875
                            for column in range(cross.shape[1])])


def compare_candidate(frame, b, private, rare, ordinary, expected):
    require(list(frame.columns) == ["record_id"] + [str(t) for t in TARGETS], "候補CSVの列集合・順序が違います")
    require(len(frame) == 1049 and np.array_equal(frame.record_id.to_numpy(), b.record_id.to_numpy()),
            "候補CSVのID・順序が違います")
    values = frame[[str(t) for t in TARGETS]].to_numpy(dtype=float)
    require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)), "候補CSVに不正確信度があります")
    require(np.all(values[private & ~rare] == 0), "既知通常非会員150人に0以外があります")
    require(np.all(values[ordinary] > 0.6) and np.all(values[ordinary] <= 0.9), "通常875人の値域が違います")
    actual = values[ordinary]
    difference = float(np.max(np.abs(actual - expected)))
    require(difference < TOLERANCE, f"通常者提出値が独立計算と不一致: 最大差{difference}")
    reconstructed_rank = np.column_stack([rank(actual[:, col], "min") for col in range(len(TARGETS))])
    expected_rank = np.column_stack([rank(expected[:, col], "min") for col in range(len(TARGETS))])
    require(np.array_equal(reconstructed_rank, expected_rank), "通常者の最終順位が違います")
    return {"ordinary_cells": int(actual.size), "private_ordinary_zero_cells": int((private & ~rare).sum() * len(TARGETS)),
            "max_submit_error": difference, "submit_values_exact": bool(np.array_equal(actual, expected)),
            "final_ranks_exact": True}


def compare_trace(path, reader, b, ordinary, arrays):
    if path is None:
        return {"provided": False}
    trace = reader.csv(path)
    required = {"team", "record_id", "raw7", "calibrated", "ordinary_rank", "cross_corrected", "submit_rank"}
    require(required <= set(trace.columns), "trace列が不足しています")
    require(len(trace) == 875 * 22 and not trace.duplicated(["team", "record_id"]).any(), "traceのセル集合が違います")
    ordinary_ids = b.loc[ordinary, "record_id"]
    maximum_errors = {}
    for column, team in enumerate(TARGETS):
        rows = trace[trace.team == team].set_index("record_id")
        require(set(rows.index) == set(ordinary_ids), f"Team{team}のtrace IDが違います")
        rows = rows.loc[ordinary_ids]
        for name in ("raw7", "calibrated"):
            expected = arrays[name][ordinary, column]
            actual = rows[name].to_numpy(float)
            require(np.isfinite(actual).all(), f"trace {name}に非有限値があります")
            error = float(np.max(np.abs(actual - expected)))
            maximum_errors[name] = max(maximum_errors.get(name, 0.0), error)
            require(error < TOLERANCE, f"trace {name}が不一致: {error}")
        for name in ("ordinary_rank", "cross_corrected", "submit_rank"):
            actual = rows[name].to_numpy(float)
            expected = arrays[name][:, column]
            if name == "cross_corrected":
                error = float(np.max(np.abs(actual - expected)))
                maximum_errors[name] = max(maximum_errors.get(name, 0.0), error)
                require(error < TOLERANCE, f"trace {name}が不一致: {error}")
            else:
                require(np.array_equal(actual, expected), f"trace {name}の順位が不一致です")
    return {"provided": True, "cells": len(trace), "max_errors": maximum_errors,
            "ordinary_rank_exact": True, "submit_rank_exact": True}


def compare_npz(path, reader, b, ordinary, arrays):
    path = Path(path).resolve()
    reader.hashes[str(path)] = sha(path)
    require(path.exists(), "計算値npzがありません")
    errors = {}
    with np.load(path, allow_pickle=False) as stored:
        require(np.array_equal(stored["teams"], TARGETS), "npzチーム順が違います")
        require(np.array_equal(stored["record_ids"], b.record_id.to_numpy()), "npz自己ID順が違います")
        require(np.array_equal(stored["normal_ids"], b.loc[ordinary, "record_id"].to_numpy()), "npz通常ID順が違います")
        for name, expected in arrays.items():
            actual = stored[name]
            require(actual.shape == expected.shape and np.isfinite(actual).all(), f"npz {name}の形・値が違います")
            error = float(np.max(np.abs(actual - expected)))
            errors[name] = error
            if name in ("ordinary_rank", "submit_rank"):
                require(np.array_equal(actual, expected), f"npz {name}が完全一致しません")
            else:
                require(error < TOLERANCE, f"npz {name}が不一致: {error}")
    return {"max_errors": errors, "ordinary_rank_exact": True, "submit_rank_exact": True,
            "other_median_recomputed_with_target_exclusion": True}


def verify_pins(root, candidate, reader, independent_input_paths):
    pins_path = Path(__file__).resolve().parent / "input_pins.json"
    manifest_path = Path(candidate).parent / "inputs_manifest.json"
    pins = reader.json(pins_path)["input_sha256"]
    manifest = reader.json(manifest_path)["input_sha256"]
    require(pins == manifest, "入力pinsと候補manifestが一致しません")
    require(isinstance(pins, dict) and pins, "入力pinsが空です")
    canonical_pinned = set()
    for relative, expected in pins.items():
        require(not Path(relative).is_absolute(), "pinはroot相対パスです")
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "pinが作業rootの外を参照しています")
        actual = sha(path)
        require(actual == expected, f"固定入力のhashが違います: {relative}")
        reader.hashes[str(path)] = actual
        canonical_pinned.add(unicodedata.normalize("NFC", str(path)))
    require(all(unicodedata.normalize("NFC", str(p)) in canonical_pinned for p in independent_input_paths),
            "独立に選んだ配布入力の一部がpinsにありません")
    return {"pinned_files": len(pins), "pins_match_manifest": True, "all_pins_match_actual_sha256": True,
            "independent_source_inputs_covered": True}


def rejection_test(name, candidate, b, private, rare, ordinary, expected):
    try:
        compare_candidate(candidate, b, private, rare, ordinary, expected)
    except ValueError as exc:
        return {"name": name, "rejected": True, "reason": str(exc)}
    raise ValueError(f"意図的誤りを見逃しました: {name}")


def audit(args):
    started = time.perf_counter()
    reader = Reader()
    a, b, qs, cs, private, rare, ordinary = input_data(args.root, reader)
    independently_selected_input_paths = list(reader.hashes)
    pins_report = verify_pins(args.root, args.candidate, reader, independently_selected_input_paths)
    source_path = Path(__file__).resolve()
    reader.hashes[str(source_path)] = sha(source_path)
    baseline_path = args.root / "本戦/攻撃フェーズ/Tashiro/20261005_MIA初回_GenLRA比較/結果/本戦_生スコア.csv"
    baseline = reader.csv(baseline_path)
    require(len(baseline) == 1049 * 22 and not baseline.duplicated(["team", "record_id"]).any(), "初回生スコアのセル集合が違います")
    raw_columns, calibrated_columns, weak_columns, controls_report = [], [], [], []
    baseline_error = 0.0
    eligible_keys = set(b.loc[ordinary, list(QI)].itertuples(index=False, name=None))
    for team in TARGETS:
        included = [t for t in TARGETS + (SELF,) if t != team]
        require(team not in included and SELF in included and len(included) == 22, "非会員課題の対象除外が不正です")
        controls = pd.concat([qs[t][list(QI)] for t in included], ignore_index=True)
        require(len(controls) == 7172 and not controls.duplicated(list(QI)).any(), "7172人の非会員に人数違い・重複があります")
        require(eligible_keys.isdisjoint(set(controls.itertuples(index=False, name=None))), "通常候補が非会員標本へ混入しました")
        query = pd.concat([b[list(QI)], controls], ignore_index=True)
        raw = mixed_lra(a, query, cs[team], args.chunk)
        candidate_raw, control_raw = raw[:1049], raw[1049:]
        prior = baseline[baseline.team == team].set_index("record_id").loc[b.record_id, "mixed7"].to_numpy(float)
        error = float(np.max(np.abs(candidate_raw - prior)))
        require(error < TOLERANCE, f"Team{team}初回Gen-LRA生スコアと不一致: {error}")
        require(np.array_equal(rank(candidate_raw[ordinary]), rank(prior[ordinary])), f"Team{team}初回生スコアの順位が違います")
        baseline_error = max(baseline_error, error)
        corrected = calibration(candidate_raw, b, control_raw, controls, shrink=500.0)
        weakened = calibration(candidate_raw, b, control_raw, controls, shrink=100.0)
        raw_columns.append(candidate_raw)
        calibrated_columns.append(corrected)
        weak_columns.append(weakened)
        controls_report.append({"team": team, "included_challenges": included, "rows": 7172,
                                "target_excluded": True, "self_included": True})
        print(json.dumps({"audited_team": team, "completed": len(raw_columns), "targets": 22,
                          "baseline_max_error": error}, ensure_ascii=False), flush=True)
    raw_matrix = np.column_stack(raw_columns)
    calibrated = np.column_stack(calibrated_columns)
    rank_matrix = np.column_stack([rank(calibrated[ordinary, j]) / 875 for j in range(22)])
    cross = contrast(rank_matrix)
    expected = submitted(cross)
    candidate = reader.csv(args.candidate)
    candidate_report = compare_candidate(candidate, b, private, rare, ordinary, expected)
    arrays = {"raw7": raw_matrix, "calibrated": calibrated, "ordinary_rank": rank_matrix,
              "cross_corrected": cross,
              "submit_rank": np.column_stack([rank(cross[:, j], "min") for j in range(22)])}
    arrays["other_median"] = (rank_matrix - cross) / 0.1
    trace_report = compare_trace(args.trace, reader, b, ordinary, arrays)
    npz_report = compare_npz(Path(args.candidate).parent / "計算値.npz", reader, b, ordinary, arrays)
    injected = []
    weak_matrix = np.column_stack(weak_columns)
    weak_rank = np.column_stack([rank(weak_matrix[ordinary, j]) / 875 for j in range(22)])
    alternatives = [("cross_median_includes_own_target", submitted(contrast(rank_matrix, include_self=True))),
                    ("alpha_0.25_instead_of_0.1", submitted(contrast(rank_matrix, alpha=0.25))),
                    ("shrink100_instead_of500", submitted(contrast(weak_rank)))]
    for name, incorrect in alternatives:
        require(np.any(incorrect != expected), f"今回のデータでは誤り注入が順位を変えません: {name}")
        corrupted = candidate.copy(deep=True)
        corrupted.loc[ordinary, [str(t) for t in TARGETS]] = incorrect
        injected.append(rejection_test(name, corrupted, b, private, rare, ordinary, expected))
    corrupted = candidate.copy(deep=True)
    corrupted.loc[np.flatnonzero(ordinary)[0], str(TARGETS[0])] = 0.0
    injected.append(rejection_test("ordinary_record_set_to_zero", corrupted, b, private, rare, ordinary, expected))
    corrupted = candidate.copy(deep=True)
    corrupted.loc[np.flatnonzero(private & ~rare)[0], str(TARGETS[0])] = 0.1
    injected.append(rejection_test("known_ordinary_nonmember_nonzero", corrupted, b, private, rare, ordinary, expected))
    corrupted = candidate.copy(deep=True)
    corrupted = corrupted[["record_id"] + [str(t) for t in reversed(TARGETS)]]
    injected.append(rejection_test("target_columns_reversed", corrupted, b, private, rare, ordinary, expected))
    # 同値をIDで解かず、average/min順位を正しく出す小さな検査。
    tied = np.array([2.0, 1.0, 2.0, 3.0])
    require(np.array_equal(rank(tied), [2.5, 1.0, 2.5, 4.0]), "average順位の同点検査が失敗")
    require(np.array_equal(rank(tied, "min"), [2.0, 1.0, 2.0, 4.0]), "min順位の同点検査が失敗")
    reader.unchanged()
    return {"status": "passed", "scope": "ordinary_MIA_only; rare audit is separate",
            "independent_implementation": True, "builder_imported": False,
            "legacy_core_or_calibrate_imported": False,
            "parameters": {"numeric": list(NUMERIC), "categorical": list(CATEGORICAL),
                           "log1p": ["TG", "ALT"], "scale": "background IQR/1.349", "rho": 0.98,
                           "numeric_dimension": 5, "neighbours": 10, "shrink": 500.0, "alpha": 0.1,
                           "cdf_tail_tiebreak": "0.999*CDF+0.001*raw_average_rank/1049",
                           "ordinary_rank": "average_rank/875", "submit": "0.6+0.3*minrank/875"},
            "counts": {"B": 1049, "background": 5143, "ordinary_unknown": 875,
                       "ordinary_known_nonmember": 150, "rare": 24, "targets": 22},
            "baseline_raw_max_error": baseline_error, "baseline_ordinary_ranks_exact": True,
            "candidate": candidate_report, "trace": trace_report, "npz": npz_report,
            "pins": pins_report,
            "controls": controls_report, "intentional_errors": injected,
            "ties_preserved": True, "inputs_unchanged": True,
            "input_sha256": reader.hashes,
            "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__, "seconds": time.perf_counter() - started}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=128)
    args = parser.parse_args()
    args.root = args.root.resolve()
    require(args.chunk > 0, "chunkは正数です")
    require(not args.output.exists(), "既存の監査記録は上書きしません")
    require(sys.version_info[:2] == (3, 12), "Python3.12を使用してください")
    require(Path(sys.executable).absolute() == args.root / ".venv/bin/python", "ルート.venv/bin/pythonを使用してください")
    started_jst = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    try:
        report = audit(args)
    except Exception as exc:
        report = {"status": "failed", "error_type": type(exc).__name__, "reason": str(exc)}
    report["started_jst"] = started_jst
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "reason": report.get("reason")}, ensure_ascii=False), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
