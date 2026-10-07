"""固定したGen-LRA7属性で、背景の層別ブートストラップを比較する分析。

個人別スコア・提出回答は保存しない。予備戦の公開正解は予測確定後だけに読む。
性別×県の人数、数値尺度、帯域を固定し、背景内の行の重みだけを変える。
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


def bootstrap_weights(a, count, seed):
    rng = np.random.default_rng(seed)
    weights = np.zeros((len(a), count))
    for indices in a.groupby(["sex", "prefecture"], sort=True).indices.values():
        indices = np.asarray(indices)
        for j in range(count):
            weights[indices, j] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)))
    assert (weights.sum(axis=0) == len(a)).all()
    return weights


def score_draws(a, b, c, weights):
    scale = common.scales(a, common.NUM7)
    ar, br, cr = [common.numbers(x, common.NUM7) / scale for x in (a, b, c)]
    h = (4 / (len(common.NUM7) + 2) / len(a)) ** (1 / (len(common.NUM7) + 4))
    dra, drb = cdist(cr, ar, "sqeuclidean"), cdist(cr, br, "sqeuclidean")
    distance = drb.T.copy()
    loga, logb = -.5 * dra / h**2, -.5 * drb / h**2
    for col in ("sex", "prefecture"):
        cats = sorted(set(a[col]) | set(b[col]) | set(c[col]))
        freq = (a[col].value_counts().reindex(cats, fill_value=0) + 1) / (len(a) + len(cats))
        cf = c[col].map(freq).to_numpy()[:, None]
        eqa = c[col].to_numpy()[:, None] == a[col].to_numpy()[None, :]
        eqb = c[col].to_numpy()[:, None] == b[col].to_numpy()[None, :]
        loga += np.log(common.RHO * eqa + (1 - common.RHO) * cf)
        logb += np.log(common.RHO * eqb + (1 - common.RHO) * cf)
        distance += ~eqb.T
    near = np.argsort(distance, axis=1, kind="stable")[:, :10]
    numerator = np.take_along_axis(logb.T, near, axis=1)
    maximum = loga.max(axis=1)
    kernel = np.exp(loga - maximum[:, None])
    allweights = np.column_stack([np.ones(len(a)), weights])
    totals = kernel @ allweights
    bases = np.empty_like(totals)
    for j in range(allweights.shape[1]):
        good = totals[:, j] > 0
        bases[good, j] = maximum[good] + np.log(totals[good, j])
        if (~good).any():
            positive = allweights[:, j] > 0
            bases[~good, j] = logsumexp(loga[~good][:, positive] + np.log(allweights[positive, j])[None, :], axis=1)
    result = np.column_stack([(np.logaddexp(0, numerator - bases[:, j][near]) - np.log1p(1 / len(a))).sum(axis=1)
                              for j in range(allweights.shape[1])])
    assert np.isfinite(result).all()
    return result[:, 0], result[:, 1:]


def reductions(baseline, draws, normal):
    x = draws[normal]
    rank = np.column_stack([rankdata(col, method="average") / normal.sum() for col in x.T])
    return {"baseline": baseline[normal], "raw_mean": x.mean(axis=1), "raw_median": np.median(x, axis=1),
            "raw_q20": np.quantile(x, .2, axis=1), "raw_mean_minus_sd": x.mean(axis=1) - x.std(axis=1),
            "rank_mean": rank.mean(axis=1), "rank_median": np.median(rank, axis=1),
            "rank_q20": np.quantile(rank, .2, axis=1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--include-main", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    assert args.draws >= 2
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError("既存の結果を上書きしません。")
    started = time.perf_counter()
    started_jst = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    b, a, _, private, rare, cs, _, _ = common.phase_inputs(args.root, "prelim")
    normal = ~private & ~rare
    matrices = {}
    for name, aa in (("full", a), ("half0", a.iloc[::2].reset_index(drop=True)), ("half1", a.iloc[1::2].reset_index(drop=True))):
        weights = bootstrap_weights(aa, args.draws, 20261007)
        computed = {}
        for team, c in cs.items():
            base, draws = score_draws(aa, b, c, weights)
            expected = common.core(aa, b, c, common.NUM7, (10,))[10]
            assert np.max(np.abs(base - expected)) < 1e-9
            computed[team] = reductions(base, draws, normal)
        for method in next(iter(computed.values())):
            mat = np.column_stack([rankdata(computed[t][method], method="average") / normal.sum() for t in cs])
            matrices[name + "__" + method] = mat
            if method in ("baseline", "rank_mean", "rank_q20"):
                median = np.column_stack([np.median(np.delete(mat, j, axis=1), axis=1) for j in range(len(cs))])
                matrices[name + "__" + method + "_cross01"] = mat - .1 * median
        print(json.dumps({"reference_completed": name, "background_rows": len(aa)}, ensure_ascii=False), flush=True)
    matrices["global7"] = matrices["full__baseline"]
    rows, aggregate = common.evaluate_prelim(args.root, b, private, rare, matrices, list(cs))
    table = pd.DataFrame(rows)
    for item in aggregate:
        if item["method"] == "global7":
            continue
        ref = item["method"].split("__")[0]
        baseline = table[table.method == ref + "__baseline"].set_index("team").mia_all
        current = table[table.method == item["method"]].set_index("team").mia_all
        item["same_reference_delta"] = float((current - baseline).mean())
        item["same_reference_wins"] = int(((current - baseline) > 1e-12).sum())
        item["same_reference_losses"] = int(((current - baseline) < -1e-12).sum())
    main_stability = []
    if args.include_main:
        mb, ma, _, mp, mr, mcs, _, _ = common.phase_inputs(args.root, "main")
        mn = ~mp & ~mr
        weights = bootstrap_weights(ma, args.draws, 20261007)
        for team, c in mcs.items():
            base, draws = score_draws(ma, mb, c, weights)
            base_top = set(np.argsort(-base[mn], kind="stable")[:12])
            common_counts = [len(base_top & set(np.argsort(-col[mn], kind="stable")[:12])) for col in draws.T]
            reduced = reductions(base, draws, mn)
            main_stability.append({"team": team, "baseline_top12_in_bootstrap_mean": float(np.mean(common_counts)),
                                   "baseline_top12_in_bootstrap_min": int(min(common_counts)),
                                   "baseline_top12_in_rank_mean": len(base_top & set(np.argsort(-reduced["rank_mean"], kind="stable")[:12])),
                                   "mia_all": None, "note": "本戦の所属正解は未使用。順位の安定性だけを計算。"})
    record = {"started_jst": started_jst, "seconds": time.perf_counter() - started, "draws": args.draws,
              "seed": 20261007, "code_sha256": common.sha(Path(__file__)),
              "common_code_sha256": common.sha(Path(common.__file__)), "input_sha256": common.INPUT_HASHES,
              "writes_submission_files": False}
    for path, expected in common.INPUT_HASHES.items():
        assert common.sha(Path(path)) == expected
    result = {"prelim_aggregate": aggregate, "prelim_by_target": rows, "main_stability": main_stability, "record": record}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "背景再標本化.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        pd.DataFrame(rows).to_csv(args.output_dir / "背景再標本化_対象別比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(aggregate).to_csv(args.output_dir / "背景再標本化_集約比較.csv", index=False, float_format="%.17g")
        if main_stability:
            pd.DataFrame(main_stability).to_csv(args.output_dir / "本戦の順位安定性.csv", index=False)
    print(json.dumps({"prelim_aggregate": aggregate, "seconds": record["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
