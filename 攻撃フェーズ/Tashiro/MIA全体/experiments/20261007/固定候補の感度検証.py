"""固定した補正係数と参照追加を、背景の人数・抽出標本を変えて比較する。

性別×県の層内で非復元抽出する。独立した正解での検証ではなく感度分析。
公開正解は全予測の確定後に読む。提出回答と個人別スコアは保存しない。
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
from 参照データの検討 import weighted_core


def stratified_subset(a, fraction, seed):
    rng = np.random.default_rng(seed)
    indices = []
    for group in a.groupby(["sex", "prefecture"], sort=True).indices.values():
        size = max(1, int(round(len(group) * fraction)))
        indices.extend(rng.choice(group, size=size, replace=False).tolist())
    return a.iloc[sorted(indices)].reset_index(drop=True)


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
    negative_q = {team: pd.concat([q[list(common.QI7)] for t, q in qs.items() if t != team], ignore_index=True)
                  for team in cs}
    matrices, counts = {}, []
    references = [("full", 1., 0, a)]
    references += [(f"fraction{fraction}_seed{seed}", fraction, seed, stratified_subset(a, fraction, seed))
                   for fraction in (.9, .8, .5) for seed in (20261007, 20261008)]
    for label, fraction, seed, aa in references:
        scores = {"baseline": [], "augmented": []}
        for team, c in cs.items():
            rr = pd.concat([aa[list(common.QI7)], negative_q[team]], ignore_index=True)
            assert not rr.duplicated(list(common.QI7)).any()
            scores["baseline"].append(common.core(aa, b, c, common.NUM7, (10,))[10])
            scores["augmented"].append(weighted_core(rr, b, c, np.ones(len(rr)), aa))
        for name, values in scores.items():
            mat = np.column_stack([rankdata(value[normal], method="average") / normal.sum() for value in values])
            matrices[label + "__" + name] = mat
            median = np.column_stack([np.median(np.delete(mat, j, axis=1), axis=1) for j in range(len(cs))])
            for alpha in ((.1, .25) if name == "baseline" else (.1,)):
                matrices[label + "__" + name + f"_cross{alpha}"] = mat - alpha * median
        counts.append({"reference": label, "fraction": fraction, "seed": seed, "background_rows": len(aa),
                       "additional_Q_rows": len(next(iter(negative_q.values())))})
        print(json.dumps({"reference_completed": label, "background_rows": len(aa)}, ensure_ascii=False), flush=True)
    matrices["global7"] = matrices["full__baseline"]
    rows, aggregate = common.evaluate_prelim(args.root, b, private, rare, matrices, list(cs))
    table = pd.DataFrame(rows)
    for item in aggregate:
        if item["method"] == "global7":
            continue
        reference, method = item["method"].split("__")
        baseline = table[table.method == reference + "__baseline"].set_index("team").mia_all
        current = table[table.method == item["method"]].set_index("team").mia_all
        delta = current - baseline
        item.update({"reference": reference, "local_method": method, "same_reference_delta": float(delta.mean()),
                     "same_reference_wins": int((delta > 1e-12).sum()),
                     "same_reference_losses": int((delta < -1e-12).sum())})
    record = {"started_jst": started_jst, "seconds": time.perf_counter() - started,
              "code_sha256": common.sha(Path(__file__)), "common_code_sha256": common.sha(Path(common.__file__)),
              "reference_code_sha256": common.sha(Path(__file__).with_name("参照データの検討.py")),
              "input_sha256": common.INPUT_HASHES, "writes_submission_files": False,
              "note": "固定した5方式について人数と層別抽出標本を変更する探索。独立した本戦検証ではない。"}
    for path, expected in common.INPUT_HASHES.items():
        assert common.sha(Path(path)) == expected
    result = {"prelim_aggregate": aggregate, "prelim_by_target": rows, "reference_counts": counts, "record": record}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "感度検証.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        table.to_csv(args.output_dir / "感度検証_対象別比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(aggregate).to_csv(args.output_dir / "感度検証_集約比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(counts).to_csv(args.output_dir / "背景の抽出条件.csv", index=False)
    print(json.dumps({"prelim_aggregate": aggregate, "seconds": record["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
