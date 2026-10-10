"""予備戦公開正解を学習に使う探索。人物と対象チームの両方を分けて評価する。

本戦の正解や個人別回答は使用・保存しない。モデルへの入力にID・患者の生属性・
対象チーム番号を含めない。公開正解の再利用なので独立した本戦検証ではない。
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
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import 分析 as common

FEATURES = ["logit_rank7", "logit_other_median7", "logit_other_max7", "logit_sex_rank7", "logit_rank9"]


def features(computed, teams, normal):
    ranked = {name: np.column_stack([rankdata(computed[t][name][normal], method="average") / normal.sum() for t in teams])
              for name in ("global7", "global9", "sex7")}
    base = ranked["global7"]
    med = np.column_stack([np.median(np.delete(base, j, axis=1), axis=1) for j in range(len(teams))])
    maximum = np.column_stack([np.max(np.delete(base, j, axis=1), axis=1) for j in range(len(teams))])
    lower = .5 / normal.sum()
    array = np.stack([logit(np.clip(x, lower, 1 - lower)) for x in
                      (base, med, maximum, ranked["sex7"], ranked["global9"])], axis=2)
    return array, base


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
    teams = list(cs)
    contexts = {}
    matrices = {}
    for context, aa in (("full", a), ("half0", a.iloc[::2]), ("half1", a.iloc[1::2])):
        computed = {t: common.geometry_scores(aa, b, c) for t, c in cs.items()}
        contexts[context], base = features(computed, teams, normal)
        matrices[context + "__baseline"] = base
        med = np.column_stack([np.median(np.delete(base, j, axis=1), axis=1) for j in range(len(teams))])
        matrices[context + "__fixed_cross01"] = base - .1 * med
        print(json.dumps({"features_completed": context}, ensure_ascii=False), flush=True)
    # ここから過去の公開正解を学習に使う。分割は正解と独立に固定する。
    truthdir = common.child(common.child(args.root / "予備戦", "配布データ") / "結果", "正解データ")
    own = common.read(truthdir / "pool_ids_5.csv").set_index("record_id").loc[b.record_id]
    truth = np.column_stack([own.pool_id.isin(common.read(truthdir / f"pool_ids_{t}.csv").pool_id).to_numpy(int)[normal]
                             for t in teams])
    assert (truth.sum(axis=0) == 25).all()
    rng = np.random.default_rng(20261007)
    person_fold = np.empty(normal.sum(), dtype=int)
    person_fold[rng.permutation(normal.sum())] = np.arange(normal.sum()) % 5
    target_fold = np.array([sorted(teams).index(t) % 4 for t in teams])
    models = {"rank1": [0], "nuisance2": [0, 1], "signals5": [0, 1, 2, 3, 4]}
    coefficients, splits = [], []
    for training in ("full", "all_contexts"):
        training_contexts = ["full"] if training == "full" else list(contexts)
        for model, columns in models.items():
            oof = {name: np.full(truth.shape, np.nan) for name in contexts}
            for pf in range(5):
                for tf in range(4):
                    train_people, test_people = np.flatnonzero(person_fold != pf), np.flatnonzero(person_fold == pf)
                    train_targets, test_targets = np.flatnonzero(target_fold != tf), np.flatnonzero(target_fold == tf)
                    assert not np.intersect1d(train_people, test_people).size
                    assert not np.intersect1d(train_targets, test_targets).size
                    xx = np.concatenate([contexts[name][np.ix_(train_people, train_targets, columns)].reshape(-1, len(columns))
                                         for name in training_contexts])
                    yy = np.tile(truth[np.ix_(train_people, train_targets)].ravel(), len(training_contexts))
                    sample_weight = np.full(len(yy), 1 / len(training_contexts))
                    fit = make_pipeline(StandardScaler(), LogisticRegression(C=.5, class_weight="balanced", max_iter=1000))
                    fit.fit(xx, yy, logisticregression__sample_weight=sample_weight)
                    for name, array in contexts.items():
                        testx = array[np.ix_(test_people, test_targets, columns)].reshape(-1, len(columns))
                        prediction = fit.predict_proba(testx)[:, 1].reshape(len(test_people), len(test_targets))
                        assert np.isnan(oof[name][np.ix_(test_people, test_targets)]).all()
                        oof[name][np.ix_(test_people, test_targets)] = prediction
                    classifier = fit.named_steps["logisticregression"]
                    coefficients.append({"training": training, "model": model, "person_fold": pf, "target_fold": tf,
                                         **{FEATURES[col]: float(coef) for col, coef in zip(columns, classifier.coef_[0])}})
                    if training == "full" and model == "rank1":
                        splits.append({"person_fold": pf, "target_fold": tf, "train_people": len(train_people),
                                       "test_people": len(test_people), "train_targets": len(train_targets),
                                       "test_targets": len(test_targets), "train_positive_pairs": int(yy.sum()),
                                       "train_pairs": len(yy)})
            for context, value in oof.items():
                assert np.isfinite(value).all()
                matrices[f"{context}__learn_{training}_{model}"] = value
            print(json.dumps({"cross_validation_completed": training + "_" + model}, ensure_ascii=False), flush=True)
    matrices["global7"] = matrices["full__baseline"]
    rows, aggregate = common.evaluate_prelim(args.root, b, private, rare, matrices, teams)
    table = pd.DataFrame(rows)
    for item in aggregate:
        if item["method"] == "global7":
            continue
        context = item["method"].split("__")[0]
        base = table[table.method == context + "__baseline"].set_index("team").mia_all
        score = table[table.method == item["method"]].set_index("team").mia_all
        delta = score - base
        item.update({"same_reference_delta": float(delta.mean()), "same_reference_wins": int((delta > 1e-12).sum()),
                     "same_reference_losses": int((delta < -1e-12).sum())})
    record = {"started_jst": started_jst, "seconds": time.perf_counter() - started, "seed": 20261007,
              "code_sha256": common.sha(Path(__file__)), "common_code_sha256": common.sha(Path(common.__file__)),
              "input_sha256": common.INPUT_HASHES, "writes_submission_files": False,
              "input_features": FEATURES, "patient_folds": 5, "target_folds": 4,
              "note": "二方向の分割で学習ラベルを分離した探索的評価。本戦への転移と独立した方式選択の検証は未実施。"}
    for path, expected in common.INPUT_HASHES.items():
        assert common.sha(Path(path)) == expected
    result = {"prelim_aggregate": aggregate, "prelim_by_target": rows, "splits": splits, "coefficients": coefficients, "record": record}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "学習再集計.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        pd.DataFrame(rows).to_csv(args.output_dir / "学習_対象別比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(aggregate).to_csv(args.output_dir / "学習_集約比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(splits).to_csv(args.output_dir / "交差検証の分割.csv", index=False)
        pd.DataFrame(coefficients).to_csv(args.output_dir / "標準化後の係数.csv", index=False, float_format="%.17g")
    print(json.dumps({"prelim_aggregate": aggregate, "seconds": record["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
