"""Gen-LRAの数値属性を一つずつ減らし、部分集合と固定した組合せを比較する。

公式の本戦所属正解は使用しない。個人別スコア・回答CSVは保存しない。
予備戦公開正解を再利用する探索であり、最良値だけから方式を採用しない。
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
from scipy.stats import rankdata

import 分析 as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError("既存の結果を上書きしません。")
    started = time.perf_counter()
    started_jst = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    b, a, _, private, rare, cs, _, _ = common.phase_inputs(args.root, "prelim")
    normal = ~private & ~rare
    subsets = {"global7": common.NUM7}
    subsets.update({"drop_" + col: tuple(c for c in common.NUM7 if c != col) for col in common.NUM7})
    subsets.update({"age_TG_HDL": ("age", "TG", "HDL"), "SBP_TG_HDL": ("SBP", "TG", "HDL"),
                    "age_SBP_ALT": ("age", "SBP", "ALT")})
    computed = {team: {name: common.core(a, b, c, cols, (10,))[10] for name, cols in subsets.items()}
                for team, c in cs.items()}
    matrices = {name: np.column_stack([rankdata(computed[t][name][normal], method="average") / normal.sum() for t in cs])
                for name in subsets}
    drops = np.stack([matrices["drop_" + col] for col in common.NUM7], axis=2)
    matrices["drop_mean"] = drops.mean(axis=2)
    matrices["drop_median"] = np.median(drops, axis=2)
    matrices["drop_max"] = drops.max(axis=2)
    matrices["baseline075_dropmean025"] = .75 * matrices["global7"] + .25 * matrices["drop_mean"]
    for name, mat in list(matrices.items()):
        median = np.column_stack([np.median(np.delete(mat, j, axis=1), axis=1) for j in range(len(cs))])
        matrices[name + "_cross01"] = mat - .1 * median
    rows, aggregate = common.evaluate_prelim(args.root, b, private, rare, matrices, list(cs))
    record = {"started_jst": started_jst, "seconds": time.perf_counter() - started,
              "code_sha256": common.sha(Path(__file__)), "common_code_sha256": common.sha(Path(common.__file__)),
              "input_sha256": common.INPUT_HASHES, "numeric_subsets": subsets,
              "categorical_columns": ["sex", "prefecture"], "writes_submission_files": False}
    for path, expected in common.INPUT_HASHES.items():
        assert common.sha(Path(path)) == expected
    result = {"prelim_aggregate": aggregate, "prelim_by_target": rows, "record": record}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "部分集合再集計.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        pd.DataFrame(rows).to_csv(args.output_dir / "部分集合_対象別比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(aggregate).to_csv(args.output_dir / "部分集合_集約比較.csv", index=False, float_format="%.17g")
    print(json.dumps({"prelim_aggregate": aggregate, "seconds": record["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
