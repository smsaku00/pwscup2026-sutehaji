"""予備戦の公開正解で、点密度比・近傍比・弱い補正の感度を比較する分析。

攻撃回答・個人別スコアは保存しない。既定は標準出力のみ。
公開正解は全方式の予測確定後に採点だけで読む。探索的な再利用データである。
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


def point_scores(a, b, c, by_sex=False):
    """Gen-LRAと同じ5数値・2カテゴリで候補点の密度比と近傍比を計算する。"""
    result = {f"nn_k{k}": np.empty(len(b)) for k in (1, 3, 10)}
    result.update({f"point_h{h}": np.empty(len(b)) for h in (.5, 1., 2.)})
    groups = sorted(b.sex.unique()) if by_sex else [None]
    for sex in groups:
        mask = np.ones(len(b), dtype=bool) if sex is None else b.sex.to_numpy() == sex
        aa = a if sex is None else a[a.sex == sex]
        cc = c if sex is None else c[c.sex == sex]
        bb = b.loc[mask]
        scale = common.scales(aa, common.NUM7)
        ar, br, cr = [common.numbers(x, common.NUM7) / scale for x in (aa, bb, cc)]
        da, dc = cdist(br, ar, "sqeuclidean"), cdist(br, cr, "sqeuclidean")
        near_a, near_c = da.copy(), dc.copy()
        cat_a, cat_c = np.zeros_like(da), np.zeros_like(dc)
        for col in ("sex", "prefecture"):
            categories = sorted(set(aa[col]) | set(bb[col]) | set(cc[col]))
            freq = (aa[col].value_counts().reindex(categories, fill_value=0) + 1) / (len(aa) + len(categories))
            eq_a = bb[col].to_numpy()[:, None] == aa[col].to_numpy()[None, :]
            eq_c = bb[col].to_numpy()[:, None] == cc[col].to_numpy()[None, :]
            bf = bb[col].map(freq).to_numpy()[:, None]
            cat_a += np.log(common.RHO * eq_a + (1 - common.RHO) * bf)
            cat_c += np.log(common.RHO * eq_c + (1 - common.RHO) * bf)
            near_a += ~eq_a
            near_c += ~eq_c
        nearest_a = np.sort(np.partition(near_a, 9, axis=1)[:, :10], axis=1)
        nearest_c = np.sort(np.partition(near_c, 9, axis=1)[:, :10], axis=1)
        for k in (1, 3, 10):
            result[f"nn_k{k}"][mask] = np.log(nearest_a[:, :k].mean(axis=1) + 1e-12) - np.log(nearest_c[:, :k].mean(axis=1) + 1e-12)
        base_h = (4 / (len(common.NUM7) + 2) / len(aa)) ** (1 / (len(common.NUM7) + 4))
        for multiplier in (.5, 1., 2.):
            h = base_h * multiplier
            loga = logsumexp(-.5 * da / h**2 + cat_a, axis=1) - np.log(len(aa))
            logc = logsumexp(-.5 * dc / h**2 + cat_c, axis=1) - np.log(len(cc))
            result[f"point_h{multiplier}"][mask] = logc - loga
    assert all(np.isfinite(value).all() for value in result.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError("既存の結果を上書きしません。")
    started = time.perf_counter()
    started_jst = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    b, a, q, private, rare, cs, _, _ = common.phase_inputs(args.root, "prelim")
    normal = ~private & ~rare
    calculated = {}
    for team, c in cs.items():
        values = common.geometry_scores(a, b, c)
        values.update({"global_" + name: value for name, value in point_scores(a, b, c).items()})
        values.update({"sex_" + name: value for name, value in point_scores(a, b, c, True).items()})
        calculated[team] = values
    matrices = {name: np.column_stack([rankdata(calculated[t][name][normal], method="average") / normal.sum() for t in cs])
                for name in next(iter(calculated.values()))}
    for name in ("global_nn_k1", "global_nn_k3", "global_point_h1.0", "sex_nn_k1", "sex_point_h1.0"):
        mat = matrices[name]
        matrices[name + "_cross025"] = mat - .25 * np.median(mat, axis=1)[:, None]
    for alpha in (.05, .1, .15, .2, .25, .3, .4, .5, .75, 1.):
        mat = matrices["global7"]
        matrices[f"cross_alpha_{alpha}"] = mat - alpha * np.median(mat, axis=1)[:, None]
    mat = matrices["global7"]
    other_medians = np.column_stack([np.median(np.delete(mat, j, axis=1), axis=1) for j in range(mat.shape[1])])
    matrices["global7_leave_target_out_cross025"] = mat - .25 * other_medians
    for weight in (.25, .5):
        combined = (1 - weight) * mat + weight * matrices["sex7"]
        matrices[f"global_sex_mix_{weight}"] = combined
        matrices[f"global_sex_mix_{weight}_cross025"] = combined - .25 * np.median(combined, axis=1)[:, None]
    rows, aggregate = common.evaluate_prelim(args.root, b, private, rare, matrices, list(cs))
    record = {"started_jst": started_jst, "seconds": time.perf_counter() - started,
              "code_sha256": common.sha(Path(__file__)), "common_code_sha256": common.sha(Path(common.__file__)),
              "input_sha256": common.INPUT_HASHES, "writes_submission_files": False,
              "method_note": "公開正解を再利用する探索的比較。方式選択後の独立検証ではない。"}
    for path, expected in common.INPUT_HASHES.items():
        assert common.sha(Path(path)) == expected
    result = {"prelim_aggregate": aggregate, "prelim_by_target": rows, "record": record}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "追加再集計.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        pd.DataFrame(rows).to_csv(args.output_dir / "予備戦_追加対象別比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(aggregate).to_csv(args.output_dir / "予備戦_追加集約比較.csv", index=False, float_format="%.17g")
        print(json.dumps({"output_dir": str(args.output_dir), "prelim_aggregate": aggregate, "seconds": record["seconds"]}, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
