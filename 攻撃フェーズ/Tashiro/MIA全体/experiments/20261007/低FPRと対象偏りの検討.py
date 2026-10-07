"""固定した5候補について低FPR、対象への偏り、本戦の順位変化を診断する。

本戦の所属正解・個人別回答は使わず、個人別スコアも保存しない。
予備戦の公開正解は、予備戦と本戦の全予測が確定した後の採点にだけ使う。
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

import 分析 as common
from 参照データの検討 import weighted_core


def predictions(root, phase):
    b, a, _, private, rare, cs, target_dir, self_id = common.phase_inputs(root, phase)
    normal = ~private & ~rare
    qs = {t: common.read(target_dir / f"aia_challenge_{t}.csv") for t in sorted(set(cs) | {self_id})}
    raw = {"baseline": [], "augmented": []}
    reference_counts = []
    for team, c in cs.items():
        negative_q = pd.concat([q[list(common.QI7)] for t, q in qs.items() if t != team], ignore_index=True)
        assert not negative_q.duplicated(list(common.QI7)).any()
        eligible_keys = pd.MultiIndex.from_frame(b.loc[normal, list(common.QI7)])
        assert not eligible_keys.isin(pd.MultiIndex.from_frame(negative_q)).any()
        reference = pd.concat([a[list(common.QI7)], negative_q], ignore_index=True)
        assert not reference.duplicated(list(common.QI7)).any()
        raw["baseline"].append(common.core(a, b, c, common.NUM7, (10,))[10])
        raw["augmented"].append(weighted_core(reference, b, c, np.ones(len(reference)), a))
        reference_counts.append({"phase": phase, "team": team, "background_rows": len(a),
                                 "additional_Q_rows": len(negative_q), "reference_rows": len(reference)})
    methods = {name: np.column_stack([rankdata(value[normal], method="average") / normal.sum()
                                     for value in values]) for name, values in raw.items()}
    for name, alphas in (("baseline", (.1, .25)), ("augmented", (.1,))):
        mat = methods[name]
        median = np.column_stack([np.median(np.delete(mat, j, axis=1), axis=1) for j in range(mat.shape[1])])
        for alpha in alphas:
            methods[name + f"_cross{alpha}"] = mat - alpha * median
    return b, private, rare, list(cs), methods, reference_counts


def top_diagnostics(methods, teams, truth=None):
    records = []
    for count in (12, 20):
        baseline = np.column_stack([np.argsort(-v, kind="stable")[:count] for v in methods["baseline"].T])
        for method, matrix in methods.items():
            top = np.column_stack([np.argsort(-v, kind="stable")[:count] for v in matrix.T])
            appearances = np.bincount(top.ravel(), minlength=len(matrix))
            for j, team in enumerate(teams):
                previous, current = set(baseline[:, j]), set(top[:, j])
                added, removed = sorted(current - previous), sorted(previous - current)
                row = {"team": team, "method": method, "top_count": count,
                       "common_with_baseline": len(previous & current), "changed_candidates": len(added)}
                if truth is not None:
                    row.update({"members_in_top": int(truth[top[:, j], j].sum()),
                                "members_added": int(truth[added, j].sum()),
                                "members_removed": int(truth[removed, j].sum())})
                records.append(row)
            row = {"team": "all", "method": method, "top_count": count,
                   "unique_candidates": int((appearances > 0).sum()),
                   "candidates_in_at_least_3_targets": int((appearances >= 3).sum()),
                   "largest_target_count": int(appearances.max())}
            if truth is not None:
                positive_hits = truth[top, np.arange(len(teams))[None, :]]
                assert (truth.sum(axis=1) <= 1).all()
                row["member_appearances"] = int(positive_hits.sum())
                row["nonmember_appearances"] = int(top.size - positive_hits.sum())
            records.append(row)
    return records


def official_roc(root):
    kit = common.child(common.child(root / "本戦", "配布データ"), "pwscup2026-kit-main_revised0912")
    sys.path.insert(0, str(kit / "starter/src"))
    module = importlib.import_module("pwscup2026.scoring.attack_score")
    common.INPUT_HASHES[str(Path(module.__file__).resolve())] = common.sha(Path(module.__file__))
    return module._tpr_at_fpr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError("既存の結果を上書きしません。")
    started, started_jst = time.perf_counter(), datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    prelim = predictions(args.root, "prelim")
    main_data = predictions(args.root, "main")
    # 全方式・両フェーズの予測が確定した後で、予備戦の公開正解を読む。
    b, private, rare, teams, methods, reference_counts = prelim
    normal = ~private & ~rare
    truthdir = common.child(common.child(args.root / "予備戦", "配布データ") / "結果", "正解データ")
    own = common.read(truthdir / "pool_ids_5.csv").set_index("record_id").loc[b.record_id]
    truth = np.column_stack([own.pool_id.isin(common.read(truthdir / f"pool_ids_{t}.csv").pool_id).to_numpy(int)
                             for t in teams])
    official = official_roc(args.root)
    official_error = 0.
    rows = []
    for method, matrix in methods.items():
        for j, team in enumerate(teams):
            confidence = common.confidence(matrix[:, j], normal, rare, private)
            fpr, tpr, _ = roc_curve(truth[:, j], confidence, drop_intermediate=False)
            for low_fpr in (.005, .01, .02, .05):
                value = float(np.interp(low_fpr, fpr, tpr))
                expected, _ = official(confidence, truth[:, j], low_fpr)
                official_error = max(official_error, abs(value - expected))
                assert abs(value - expected) < 1e-12
                members = int(truth[:, j].sum())
                rows.append({"team": team, "method": method, "fpr": low_fpr, "mia_all": value,
                             "members": members, "effective_true_positives": value * members,
                             "allowed_false_positives": low_fpr * (len(b) - members)})
    table = pd.DataFrame(rows)
    summary = []
    for (method, low_fpr), frame in table.groupby(["method", "fpr"], sort=True):
        baseline = table[(table.method == "baseline") & (table.fpr == low_fpr)].set_index("team")
        delta = frame.set_index("team").mia_all - baseline.mia_all
        effective_gain = frame.set_index("team").effective_true_positives - baseline.effective_true_positives
        summary.append({"method": method, "fpr": low_fpr, "targets": len(frame),
                        "mia_all": float(frame.mia_all.mean()),
                        "other18_mean": float(frame[~frame.team.isin([1, 2])].mia_all.mean()),
                        "paired_mean_delta": float(delta.mean()), "wins": int((delta > 1e-12).sum()),
                        "losses": int((delta < -1e-12).sum()),
                        "effective_additional_true_positives": float(effective_gain.sum()),
                        "smallest_delta_when_one_target_removed": float((delta.sum() - delta.max()) / (len(delta) - 1)),
                        "largest_delta_when_one_target_removed": float((delta.sum() - delta.min()) / (len(delta) - 1))})
    _, _, _, main_teams, main_methods, main_reference_counts = main_data
    main_top = top_diagnostics(main_methods, main_teams)
    prelim_top = top_diagnostics(methods, teams, truth[normal])
    record = {"started_jst": started_jst, "seconds": time.perf_counter() - started,
              "code_sha256": common.sha(Path(__file__)), "common_code_sha256": common.sha(Path(common.__file__)),
              "reference_code_sha256": common.sha(Path(__file__).with_name("参照データの検討.py")),
              "input_sha256": common.INPUT_HASHES, "official_tpr_max_error": official_error,
              "writes_submission_files": False, "main_membership_truth_used": False,
              "note": "全て予備戦を再利用した探索。本戦は順位変化と人数のみで、MIA全体の達成を評価していない。"}
    for path, expected in common.INPUT_HASHES.items():
        assert common.sha(Path(path)) == expected
    result = {"low_fpr_aggregate": summary, "low_fpr_by_target": rows,
              "prelim_top": prelim_top, "main_top": main_top,
              "reference_counts": reference_counts + main_reference_counts, "record": record}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "低FPR再集計.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        table.to_csv(args.output_dir / "低FPR_対象別比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(summary).to_csv(args.output_dir / "低FPR_集約比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(prelim_top).to_csv(args.output_dir / "予備戦の上位候補診断.csv", index=False)
        pd.DataFrame(main_top).to_csv(args.output_dir / "本戦の上位候補診断.csv", index=False)
        pd.DataFrame(reference_counts + main_reference_counts).to_csv(args.output_dir / "参照人数.csv", index=False)
    print(json.dumps({"low_fpr_aggregate": summary, "seconds": record["seconds"],
                      "official_tpr_max_error": official_error}, ensure_ascii=False))


if __name__ == "__main__":
    main()
