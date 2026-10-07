"""既知の非会員を用い、性別・県ごとのGen-LRAスコアの偏りを較正する探索。

背景Aを採点するときは、その本人を参照の密度和から除く。尺度・帯域・カテゴリ
頻度は全背景で固定する。公開正解は全予測確定後だけに読む。回答は作成しない。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import rankdata

import 分析 as common


def background_loo_scores(a, c):
    """各背景者本人のカーネルを密度和から除いた追加効果。近傍は固定。

    参照パラメータは全Aで固定した近似LOOであり、5143回の全面再学習ではない。
    """
    scale = common.scales(a, common.NUM7)
    ar, cr = [common.numbers(x, common.NUM7) / scale for x in (a, c)]
    distance = cdist(cr, ar, "sqeuclidean")
    h = (4 / (len(common.NUM7) + 2) / len(a)) ** (1 / (len(common.NUM7) + 4))
    log_kernel = -.5 * distance / h**2
    nearby = distance.T.copy()
    for col in ("sex", "prefecture"):
        cats = sorted(set(a[col]) | set(c[col]))
        freq = (a[col].value_counts().reindex(cats, fill_value=0) + 1) / (len(a) + len(cats))
        cf = c[col].map(freq).to_numpy()[:, None]
        eq = c[col].to_numpy()[:, None] == a[col].to_numpy()[None, :]
        log_kernel += np.log(common.RHO * eq + (1 - common.RHO) * cf)
        nearby += ~eq.T
    base = logsumexp(log_kernel, axis=1)
    ratio = np.minimum(log_kernel - base[:, None], 0.)
    good = ratio < 0
    log_without = np.empty_like(log_kernel)
    log_without[good] = np.broadcast_to(base[:, None], ratio.shape)[good] + np.log(-np.expm1(ratio[good]))
    # 密度和が一人の寄与に丸められた極端な点も、他の参照者から正確に再計算する。
    fallback_rows, fallback_cols = np.nonzero(~good)
    for row, col in zip(fallback_rows, fallback_cols):
        keep = np.arange(len(a)) != col
        log_without[row, col] = logsumexp(log_kernel[row, keep])
    delta = (np.logaddexp(0, log_kernel - log_without) - np.log1p(1 / (len(a) - 1))).T
    nearest = np.argsort(nearby, axis=1, kind="stable")[:, :10]
    result = np.take_along_axis(delta, nearest, axis=1).sum(axis=1)
    assert np.isfinite(result).all()
    return result, len(fallback_rows)


def empirical_cdf(values, null_values):
    ordered, counts = np.unique(null_values, return_counts=True)
    probabilities = (np.cumsum(counts) - .5 * counts) / len(null_values)
    return np.interp(values, ordered, probabilities, left=0., right=1.)


def calibrate(values, b, null_values, controls, groups, shrink):
    global_cdf = empirical_cdf(values, null_values)
    answer = global_cdf.copy()
    if groups:
        b_groups = b.groupby(groups, sort=True).indices
        null_groups = controls.groupby(groups, sort=True).indices
        for key, indices in b_groups.items():
            control_indices = null_groups.get(key, [])
            if len(control_indices):
                local_cdf = empirical_cdf(values[indices], null_values[control_indices])
                answer[indices] = (len(control_indices) * local_cdf + shrink * global_cdf[indices]) / (len(control_indices) + shrink)
    # 全体較正の単調性を保ち、経験CDFの外端や同値に元の順位で小さな差を付ける。
    return .999 * answer + .001 * rankdata(values, method="average") / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError("既存の結果を上書きしません。")
    started, started_jst = time.perf_counter(), datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    b, a, _, private, rare, cs, target_dir, self_id = common.phase_inputs(args.root, "prelim")
    normal = ~private & ~rare
    qs = {t: common.read(target_dir / f"aia_challenge_{t}.csv") for t in sorted(set(cs) | {self_id})}
    scores, records = {}, []
    for team, c in cs.items():
        negative_q = pd.concat([q[list(common.QI7)] for t, q in qs.items() if t != team], ignore_index=True)
        query = pd.concat([b[list(common.QI7)], negative_q], ignore_index=True)
        combined = common.core(a, query, c, common.NUM7, (10,))[10]
        values, q_scores = combined[:len(b)], combined[len(b):]
        expected = common.core(a, b, c, common.NUM7, (10,))[10]
        assert np.max(np.abs(values - expected)) < 1e-12
        a_scores, fallback_count = background_loo_scores(a, c)
        banks = {"Q": (negative_q, q_scores), "A_loo": (a, a_scores)}
        values_by_method = {"global7": values}
        for bank, (controls, null_values) in banks.items():
            values_by_method[bank + "_global_control"] = calibrate(values, b, null_values, controls, [], 0)
            for grouping in (("sex",), ("prefecture", "sex")):
                for shrink in (100., 500.):
                    name = bank + "_" + "_".join(grouping) + f"_shrink{int(shrink)}"
                    values_by_method[name] = calibrate(values, b, null_values, controls, list(grouping), shrink)
        scores[team] = values_by_method
        records.append({"team": team, "background_rows": len(a), "Q_nonmember_rows": len(negative_q),
                        "LOO_numerical_fallbacks": fallback_count})
        print(json.dumps({"predictions_completed": len(scores), "targets": len(cs)}, ensure_ascii=False), flush=True)
    matrices = {name: np.column_stack([rankdata(scores[t][name][normal], method="average") / normal.sum() for t in cs])
                for name in next(iter(scores.values()))}
    for bank in ("Q", "A_loo"):
        assert np.max(np.abs(matrices[bank + "_global_control"] - matrices["global7"])) < 1e-12
    for name, matrix in list(matrices.items()):
        if name.endswith("global_control"):
            continue
        median = np.column_stack([np.median(np.delete(matrix, j, axis=1), axis=1) for j in range(len(cs))])
        matrices[name + "_cross01"] = matrix - .1 * median
    rows, aggregate = common.evaluate_prelim(args.root, b, private, rare, matrices, list(cs))
    record = {"started_jst": started_jst, "seconds": time.perf_counter() - started,
              "code_sha256": common.sha(Path(__file__)), "common_code_sha256": common.sha(Path(common.__file__)),
              "input_sha256": common.INPUT_HASHES, "writes_submission_files": False,
              "note": "LOOは尺度・帯域・カテゴリ頻度を固定し、本人の密度寄与だけを除く近似。公開正解を再利用した探索。"}
    for path, expected in common.INPUT_HASHES.items():
        assert common.sha(Path(path)) == expected
    result = {"prelim_aggregate": aggregate, "prelim_by_target": rows, "null_counts": records, "record": record}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "非会員較正.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        pd.DataFrame(rows).to_csv(args.output_dir / "非会員較正_対象別比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(aggregate).to_csv(args.output_dir / "非会員較正_集約比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(records).to_csv(args.output_dir / "較正に使った人数.csv", index=False)
    print(json.dumps({"prelim_aggregate": aggregate, "seconds": record["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
