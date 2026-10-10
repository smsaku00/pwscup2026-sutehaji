"""参照分布・参照人数・公開AIA課題の追加を比較する分析。提出回答は作成しない。

標的i以外のAIA課題は、非共有P_jと全コホート外の対照からなるためB_iの非会員。
地理・性別・年齢や希少の比率は背景と異なるので、追加の効果を無条件には想定しない。
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


def keys(frame, columns):
    out = frame[list(columns)].copy()
    if "age_band10" in columns:
        raise ValueError("age_band10は別途付与する")
    return pd.Index(out.iloc[:, 0]) if len(columns) == 1 else pd.MultiIndex.from_frame(out)


def weighted_core(reference, b, c, weights, scale_reference, h_count=None):
    weights = np.asarray(weights, dtype=float)
    assert len(weights) == len(reference) and np.isfinite(weights).all() and (weights > 0).all()
    weights = weights / weights.mean()
    scale = common.scales(scale_reference, common.NUM7)
    r, x, s = [common.numbers(f, common.NUM7) / scale for f in (reference, b, c)]
    dcr, dcb = cdist(s, r, "sqeuclidean"), cdist(s, x, "sqeuclidean")
    near_distance = dcb.T.copy()
    n = len(reference)
    h_count = n if h_count is None else h_count
    h = (4 / (len(common.NUM7) + 2) / h_count) ** (1 / (len(common.NUM7) + 4))
    logcr, logcb = -.5 * dcr / h**2, -.5 * dcb / h**2
    for col in ("sex", "prefecture"):
        cats = sorted(set(reference[col]) | set(b[col]) | set(c[col]))
        mass = pd.Series(weights).groupby(reference[col].reset_index(drop=True)).sum().reindex(cats, fill_value=0)
        freq = (mass + 1) / (n + len(cats))
        sf = c[col].map(freq).to_numpy()[:, None]
        eqr = c[col].to_numpy()[:, None] == reference[col].to_numpy()[None, :]
        eqb = c[col].to_numpy()[:, None] == b[col].to_numpy()[None, :]
        logcr += np.log(common.RHO * eqr + (1 - common.RHO) * sf)
        logcb += np.log(common.RHO * eqb + (1 - common.RHO) * sf)
        near_distance += ~eqb.T
    base = logsumexp(logcr + np.log(weights)[None, :], axis=1)
    delta = (np.logaddexp(0, logcb - base[:, None]) - np.log1p(1 / n)).T
    near = np.argsort(near_distance, axis=1, kind="stable")[:, :10]
    score = np.take_along_axis(delta, near, axis=1).sum(axis=1)
    assert np.isfinite(score).all()
    return score


def mixture_weights(a, q, columns, private_fraction=313 / 1049):
    counts = a.groupby(columns).size()
    bg = counts / len(a)
    qcounts = q.groupby(columns).size().reindex(bg.index, fill_value=0)
    private = (qcounts + 5 * bg) / (len(q) + 5)
    per_group = (1 - private_fraction) + private_fraction * private / bg
    return per_group.reindex(keys(a, columns)).to_numpy(float)


def calibrated_weights(reference, a, age=False):
    rr, aa = reference.copy(), a.copy()
    columns = ["prefecture", "sex"]
    if age:
        rr["age_band10"] = (rr.age // 10).astype(int)
        aa["age_band10"] = (aa.age // 10).astype(int)
        columns += ["age_band10"]
    observed = rr.groupby(columns).size()
    expected = aa.groupby(columns).size().reindex(observed.index, fill_value=0).astype(float) + .5
    expected /= expected.sum()
    group_weights = len(reference) * expected / observed
    return group_weights.reindex(pd.MultiIndex.from_frame(rr[columns])).to_numpy(float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        raise FileExistsError("既存の結果を上書きしません。")
    started = time.perf_counter()
    started_jst = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    b, a, _, private, rare, cs, targets, self_id = common.phase_inputs(args.root, "prelim")
    normal = ~private & ~rare
    qs = {t: common.read(targets / f"aia_challenge_{t}.csv") for t in sorted(set(cs) | {self_id})}
    raw, ref_records = {}, []
    for team, c in cs.items():
        negative_q = pd.concat([q[list(common.QI7)] for t, q in qs.items() if t != team], ignore_index=True)
        q_total = len(negative_q)
        negative_q = negative_q.drop_duplicates(list(common.QI7)).reset_index(drop=True)
        eligible = pd.MultiIndex.from_frame(b.loc[normal, list(common.QI7)])
        assert not eligible.isin(pd.MultiIndex.from_frame(negative_q[list(common.QI7)])).any()
        values = {"global7": common.core(a, b, c, common.NUM7, (10,))[10]}
        for cols, label in ((["prefecture"], "pref"), (["prefecture", "sex"], "pref_sex")):
            weights = mixture_weights(a, qs[team], cols)
            values["target_mixture_" + label] = weighted_core(a, b, c, weights, a)
        for half in (0, 1):
            aa = a.iloc[half::2].copy().reset_index(drop=True)
            values[f"half{half}_bg"] = common.core(aa, b, c, common.NUM7, (10,))[10]
            reference = pd.concat([aa[list(common.QI7)], negative_q], ignore_index=True)
            assert not reference.duplicated(list(common.QI7)).any()
            values[f"half{half}_aug_plain"] = weighted_core(reference, b, c, np.ones(len(reference)), aa)
            for age, label in ((False, "pref_sex"), (True, "pref_sex_age10")):
                weights = calibrated_weights(reference, aa, age)
                effective_n = float(weights.sum()**2 / np.square(weights).sum())
                values[f"half{half}_aug_{label}"] = weighted_core(reference, b, c, weights, aa, effective_n)
                ref_records.append({"team": team, "background_half": half, "calibration": label,
                                    "A_rows": len(aa), "Q_rows_before_deduplication": q_total,
                                    "Q_rows": len(negative_q), "reference_rows": len(reference), "effective_n": effective_n})
        raw[team] = values
        print(json.dumps({"predictions_completed": len(raw), "targets": len(cs)}, ensure_ascii=False), flush=True)
    matrices = {name: np.column_stack([rankdata(raw[t][name][normal], method="average") / normal.sum() for t in cs])
                for name in next(iter(raw.values()))}
    for name, mat in list(matrices.items()):
        med = np.column_stack([np.median(np.delete(mat, j, axis=1), axis=1) for j in range(len(cs))])
        for alpha in (.1, .25):
            matrices[name + f"_cross{alpha}"] = mat - alpha * med
    rows, aggregate = common.evaluate_prelim(args.root, b, private, rare, matrices, list(cs))
    record = {"started_jst": started_jst, "seconds": time.perf_counter() - started,
              "code_sha256": common.sha(Path(__file__)), "common_code_sha256": common.sha(Path(common.__file__)),
              "input_sha256": common.INPUT_HASHES, "writes_submission_files": False,
              "note": "背景人数・参照分布の探索的感度分析。AIA課題のlabelやtimeを推定入力に使わない。"}
    for path, expected in common.INPUT_HASHES.items():
        assert common.sha(Path(path)) == expected
    result = {"prelim_aggregate": aggregate, "prelim_by_target": rows, "references": ref_records, "record": record}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "参照再集計.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        pd.DataFrame(rows).to_csv(args.output_dir / "参照_対象別比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(aggregate).to_csv(args.output_dir / "参照_集約比較.csv", index=False, float_format="%.17g")
        pd.DataFrame(ref_records).to_csv(args.output_dir / "参照の人数.csv", index=False, float_format="%.17g")
    print(json.dumps({"prelim_aggregate": aggregate, "seconds": record["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
