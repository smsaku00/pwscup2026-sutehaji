"""既存純関数でP01と最近傍距離比k=1だけを再現。人物スコア/正解は保存しない。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from official_audit import band, child, load_official, load_truth, sha


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def roc_neighbors(confidence, truth):
    order = np.argsort(-confidence, kind="mergesort")
    c, y = confidence[order], truth[order]
    ends = np.r_[np.flatnonzero(np.diff(c) != 0), len(c) - 1]
    positives, negatives = int(y.sum()), len(y) - int(y.sum())
    tp = np.r_[0, np.cumsum(y)[ends]]
    fp = np.r_[0, np.cumsum(1 - y)[ends]]
    fpr = fp / negatives
    right = int(np.searchsorted(fpr, .01, side="right"))
    left = right - 1
    weight = (.01 - fpr[left]) / (fpr[right] - fpr[left])
    return {"low_fp_count": int(fp[left]), "high_fp_count": int(fp[right]),
            "low_tp_count": int(tp[left]), "high_tp_count": int(tp[right]),
            "interpolation_weight": float(weight), "false_positive_equivalent": .01 * negatives,
            "true_positive_equivalent": float(tp[left] + weight * (tp[right] - tp[left]))}


def reproduce(root: Path):
    root = root.resolve()
    started = time.perf_counter()
    directory = root / "本戦/攻撃フェーズ/Tashiro/20261007_MIA全体改善の検討"
    paths = {"common": directory / "分析.py", "P01": directory / "非会員を使う順位較正の検討.py",
             "nearest": directory / "追加検討.py"}
    old_analysis = sys.modules.get("分析")
    common = module("分析", paths["common"])
    cal = module("representative_calibration", paths["P01"])
    nearest = module("representative_nearest", paths["nearest"])
    official = load_official(root)
    saved_json = {"P01": directory / "非会員較正結果/非会員較正.json",
                  "nearest": directory / "追加結果_最終版/追加再集計.json"}
    history = {}
    for key, path in saved_json.items():
        old = json.loads(path.read_text(encoding="utf-8"))["record"]
        code_key = "P01" if key == "P01" else "nearest"
        history[key] = {"source": str(path.relative_to(root)), "source_sha256": sha(path),
                        "saved_code_sha256": old["code_sha256"], "current_code_sha256": sha(paths[code_key]),
                        "code_hash_matches": old["code_sha256"] == sha(paths[code_key]),
                        "common_hash_matches": old["common_code_sha256"] == sha(paths["common"]),
                        "saved_python": old.get("python"), "saved_python_version": old.get("python_version"),
                        "stored_input_hashes": old.get("input_sha256", {})}
        assert history[key]["code_hash_matches"] and history[key]["common_hash_matches"], key
    # 明示固定Pythonの有無。旧mainや子プロセスは起動しない。
    fixed_paths = {}
    for key, path in paths.items():
        text = path.read_text(encoding="utf-8")
        fixed_paths[key] = sorted(set(re.findall(r"(?:/Users|/opt|/usr)[^\s\"']*(?:python[0-9.]*)", text)))
    b, a, _, private, rare, cs, target_dir, self_id = common.phase_inputs(root, "prelim")
    normal = ~private & ~rare
    teams = list(cs)
    qs = {t: common.read(target_dir / f"aia_challenge_{t}.csv") for t in sorted(set(cs) | {self_id})}
    p01_columns, nn_columns = [], []
    for team in teams:
        q = pd.concat([q[list(common.QI7)] for t, q in qs.items() if t != team], ignore_index=True)
        query = pd.concat([b[list(common.QI7)], q[list(common.QI7)]], ignore_index=True)
        raw = common.core(a, query, cs[team], common.NUM7, (10,))[10]
        values, null = raw[:len(b)], raw[len(b):]
        calibrated = cal.calibrate(values, b, null, q, ["prefecture", "sex"], 500.)
        p01_columns.append(rankdata(calibrated[normal], method="average") / normal.sum())
        nn_values = nearest.point_scores(a, b, cs[team], by_sex=False)["nn_k1"]
        nn_columns.append(rankdata(nn_values[normal], method="average") / normal.sum())
        print(json.dumps({"stage": "predicted", "completed_targets": len(p01_columns), "targets": len(teams)}, ensure_ascii=False), flush=True)
    p01 = np.column_stack(p01_columns)
    median = np.column_stack([np.median(np.delete(p01, j, axis=1), axis=1) for j in range(len(teams))])
    models = {"P01": p01 - .1 * median, "global_nn_k1": np.column_stack(nn_columns)}
    # 全20対象の予測を確定した後だけに公開所属正解を読む。
    truth = load_truth(root)
    saved_paths = {"P01": directory / "非会員較正結果/非会員較正_対象別比較.csv",
                   "global_nn_k1": directory / "追加結果_最終版/予備戦_追加対象別比較.csv"}
    saved_names = {"P01": "Q_prefecture_sex_shrink500_cross01", "global_nn_k1": "global_nn_k1"}
    rows = []
    max_error = 0.
    for name, mat in models.items():
        saved = pd.read_csv(saved_paths[name], float_precision="round_trip")
        expected = saved[saved.method == saved_names[name]].set_index("team").mia_all
        assert len(expected) == 20
        for col, team in enumerate(teams):
            confidence = common.confidence(mat[:, col], normal, rare, private)
            tpr, auc = official._tpr_at_fpr(confidence, truth[team], .01)
            old_tpr = common.roc(confidence, truth[team])[0]
            error = abs(tpr - float(expected.loc[team]))
            max_error = max(max_error, error)
            assert error < 1e-12 and abs(old_tpr - tpr) < 1e-12, (name, team, error)
            row = {"method": name, "team": team, "mia_all": tpr, "band": band(tpr),
                   "members": int(truth[team].sum()), "auc": auc, "saved_mia_all": float(expected.loc[team]),
                   "absolute_error": error, "detected_equivalent": tpr * int(truth[team].sum()),
                   "tpr_fpr_005": official._tpr_at_fpr(confidence, truth[team], .005)[0],
                   "tpr_fpr_02": official._tpr_at_fpr(confidence, truth[team], .02)[0],
                   "roc_fpr_01_neighbors": roc_neighbors(confidence, truth[team]),
                   "final_report_included": team != 12}
            assert abs(row["detected_equivalent"] - row["roc_fpr_01_neighbors"]["true_positive_equivalent"]) < 1e-10
            rows.append(row)
    code_hashes = {str(p.relative_to(root)): sha(p) for p in paths.values()}
    for p, digest in common.INPUT_HASHES.items():
        assert sha(Path(p)) == digest, p
    comparison = []
    for key, item in history.items():
        unchanged, missing = 0, []
        for p, digest in item.pop("stored_input_hashes").items():
            source = Path(p)
            if source.exists():
                assert sha(source) == digest, source
                unchanged += 1
            else:
                missing.append(p)
        item["matched_saved_input_hashes"] = unchanged
        item["missing_original_paths"] = missing
        comparison.append(item)
    truthdir = child(child(root / "予備戦", "配布データ") / "結果", "正解データ")
    input_hashes = {str(Path(p).relative_to(root)): h for p, h in common.INPUT_HASHES.items()}
    for team in [5] + teams:
        p = truthdir / f"pool_ids_{team}.csv"
        input_hashes[str(p.relative_to(root))] = sha(p)
    if old_analysis is None:
        sys.modules.pop("分析", None)
    else:
        sys.modules["分析"] = old_analysis
    summary = {}
    for name in models:
        final = [r for r in rows if r["method"] == name and r["final_report_included"]]
        summary[name] = {"final_targets": len(final), "mean_tpr": float(np.mean([r["mia_all"] for r in final])),
                         "band2_plus_teams": sorted(r["team"] for r in final if r["band"] >= 2),
                         "min_neighbor_tpr_team1": min(r["tpr_fpr_005"] for r in final if r["team"] == 1),
                         "min_neighbor_tpr_team2": min(r["tpr_fpr_005"] for r in final if r["team"] == 2)}
    return {"passed": True, "compared_target_records": len(rows), "max_abs_error": max_error,
            "python": sys.executable, "python_version": platform.python_version(),
            "old_python_fixed_paths_in_source": fixed_paths,
            "old_python_record_limitation": "元の20261007記録にはpython/python_version欄がない。コード固定パスは検査したが当時の実Pythonは不明。今回はルート.venvのみを使い旧main/子プロセスを実行しない。",
            "code_sha256": code_hashes, "runner_sha256": sha(Path(__file__)),
            "official_sha256": official._source_sha256, "input_sha256": input_hashes,
            "original_record_checks": comparison,
            "saved_result_sha256": {str(p.relative_to(root)): sha(p) for p in saved_paths.values()},
            "method_design": {"P01": "Gen-LRA7 k10;Qによる県×性別 shrink500;通常875候補平均順位;対象自身を除く他19対象中央値のcross0.1;全20対象を同時利用。",
                              "global_nn_k1": "5数値(age,SBP,TG,HDL,ALT)、TG/ALT log1p、背景IQR/1.349尺度、sex/prefecture不一致各+1、背景最近傍/公開C最近傍の対数距離比。crossなし。"},
            "labels": "予測確定後に予備戦公開所属正解で採点。人物別配列/正解/予測はメモリのみ。",
            "summary": summary, "target_metrics": rows,
            "completed_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(), "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    path = args.output_dir / "代表方式再現監査.json"
    if path.exists():
        raise FileExistsError(path)
    result = reproduce(args.root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "compared_target_records": result["compared_target_records"],
                      "max_abs_error": result["max_abs_error"], "seconds": result["seconds"], "summary": result["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
