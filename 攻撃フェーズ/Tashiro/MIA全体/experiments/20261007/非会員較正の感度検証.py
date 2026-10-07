"""公開課題による非会員較正の、背景抽出・低FPR・本戦順位変化の感度分析。

較正の縮小強度100/500、補正0.1を固定する。公開正解は全予測確定後に読む。
本戦は順位変化だけを評価し、所属正解・個人別回答は使用も保存もしない。
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
from sklearn.metrics import roc_curve

import 分析 as common
from 固定候補の感度検証 import stratified_subset
from 非会員を使う順位較正の検討 import calibrate
from 低FPRと対象偏りの検討 import top_diagnostics, official_roc


def predict(b, aa, negative_q, cs, normal):
    scores = {"baseline": [], "Q_pref_sex100": [], "Q_pref_sex500": []}
    for team, c in cs.items():
        controls = negative_q[team]
        queries = pd.concat([b[list(common.QI7)], controls], ignore_index=True)
        raw = common.core(aa, queries, c, common.NUM7, (10,))[10]
        candidates, null = raw[:len(b)], raw[len(b):]
        scores["baseline"].append(candidates)
        for shrink in (100., 500.):
            scores[f"Q_pref_sex{int(shrink)}"].append(calibrate(candidates, b, null, controls, ["prefecture", "sex"], shrink))
    matrices = {name: np.column_stack([rankdata(value[normal], method="average") / normal.sum() for value in values])
                for name, values in scores.items()}
    for name, matrix in list(matrices.items()):
        median = np.column_stack([np.median(np.delete(matrix, j, axis=1), axis=1) for j in range(len(cs))])
        matrices[name + "_cross01"] = matrix - .1 * median
    return matrices


def banks(qs, teams):
    return {team: pd.concat([q[list(common.QI7)] for t, q in qs.items() if t != team], ignore_index=True)
            for team in teams}


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
    negative_q = banks(qs, cs)
    references = [("full", 1., 0, a)]
    references += [(f"fraction{fraction}_seed{seed}", fraction, seed, stratified_subset(a, fraction, seed))
                   for fraction in (.9, .8, .5) for seed in (20261007, 20261008)]
    matrices, counts = {}, []
    for label, fraction, seed, aa in references:
        computed = predict(b, aa, negative_q, cs, normal)
        matrices.update({label + "__" + name: value for name, value in computed.items()})
        counts.append({"reference": label, "fraction": fraction, "seed": seed, "background_rows": len(aa)})
        print(json.dumps({"reference_completed": label}, ensure_ascii=False), flush=True)
    mb, ma, _, mp, mr, mcs, mt, ms = common.phase_inputs(args.root, "main")
    mqs = {t: common.read(mt / f"aia_challenge_{t}.csv") for t in sorted(set(mcs) | {ms})}
    main_matrices = predict(mb, ma, banks(mqs, mcs), mcs, ~mp & ~mr)
    # 予備戦・本戦の全方式の予測確定後に公開正解を読む。
    matrices["global7"] = matrices["full__baseline"]
    rows, aggregate = common.evaluate_prelim(args.root, b, private, rare, matrices, list(cs))
    table = pd.DataFrame(rows)
    for item in aggregate:
        if item["method"] == "global7":
            continue
        reference, method = item["method"].split("__")
        base = table[table.method == reference + "__baseline"].set_index("team").mia_all
        current = table[table.method == item["method"]].set_index("team").mia_all
        delta = current - base
        item.update({"reference": reference, "local_method": method, "same_reference_delta": float(delta.mean()),
                     "same_reference_wins": int((delta > 1e-12).sum()), "same_reference_losses": int((delta < -1e-12).sum())})
    truthdir = common.child(common.child(args.root / "予備戦", "配布データ") / "結果", "正解データ")
    own = common.read(truthdir / "pool_ids_5.csv").set_index("record_id").loc[b.record_id]
    official, low_rows, max_error = official_roc(args.root), [], 0.
    for j, team in enumerate(cs):
        truth = own.pool_id.isin(common.read(truthdir / f"pool_ids_{team}.csv").pool_id).to_numpy(int)
        for method, mat in matrices.items():
            if not method.startswith("full__"):
                continue
            confidence = common.confidence(mat[:, j], normal, rare, private)
            fpr, tpr, _ = roc_curve(truth, confidence, drop_intermediate=False)
            for point in (.005, .01, .02, .05):
                value = float(np.interp(point, fpr, tpr))
                expected, _ = official(confidence, truth, point)
                max_error = max(max_error, abs(value - expected))
                assert abs(value - expected) < 1e-12
                low_rows.append({"team": team, "method": method, "fpr": point, "mia_all": value})
    low_table = pd.DataFrame(low_rows)
    low_aggregate = low_table.groupby(["method", "fpr"], as_index=False).mia_all.mean()
    main_top = top_diagnostics(main_matrices, list(mcs))
    record = {"started_jst": started_jst, "seconds": time.perf_counter() - started,
              "code_sha256": common.sha(Path(__file__)), "common_code_sha256": common.sha(Path(common.__file__)),
              "calibration_code_sha256": common.sha(Path(__file__).with_name("非会員を使う順位較正の検討.py")),
              "sampling_code_sha256": common.sha(Path(__file__).with_name("固定候補の感度検証.py")),
              "tail_code_sha256": common.sha(Path(__file__).with_name("低FPRと対象偏りの検討.py")),
              "input_sha256": common.INPUT_HASHES, "official_tpr_max_error": max_error,
              "writes_submission_files": False, "main_membership_truth_used": False,
              "note": "人数と抽出を変えた感度分析。較正方式の選択に予備戦を再利用しており、独立した本戦評価ではない。"}
    for path, expected in common.INPUT_HASHES.items():
        assert common.sha(Path(path)) == expected
    result = {"prelim_aggregate": aggregate, "prelim_by_target": rows, "reference_counts": counts,
              "low_fpr_aggregate": low_aggregate.to_dict("records"), "low_fpr_by_target": low_rows,
              "main_top": main_top, "record": record}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "較正感度検証.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        table.to_csv(args.output_dir / "較正感度_対象別比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(aggregate).to_csv(args.output_dir / "較正感度_集約比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(counts).to_csv(args.output_dir / "背景の抽出条件.csv", index=False)
        low_table.to_csv(args.output_dir / "較正低FPR_対象別比較.csv", index=False, float_format="%.17g")
        low_aggregate.to_csv(args.output_dir / "較正低FPR_集約比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(main_top).to_csv(args.output_dir / "本戦の較正順位診断.csv", index=False)
    print(json.dumps({"prelim_aggregate": aggregate, "seconds": record["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
