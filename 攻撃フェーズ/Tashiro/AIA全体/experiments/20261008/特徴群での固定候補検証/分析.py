"""列値保持・行部分保持群の固定他C補正を、既存32背景条件で再検証する。

群は公開Cの既存特徴だけで決め、係数・候補・seedを探索しない。
全予測を保存後に公開予備戦正解を読み込む。32参照は感度条件である。
"""
from __future__ import annotations

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").exists())
OUT = HERE / "集計"
WORK = HERE / "作業用"
METHODS = ["baseline", "other_q75_lambda025"]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


b = module("group_fixed_background", PARENT / "基準方式の背景感度/分析.py")
o = module("group_fixed_other", PARENT / "他Cとの差による選別/分析.py")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups():
    columns = b.read(PARENT / "加工特徴と方式選択/集計/BG検査値の値集合一致_個人対応を示さない.csv")
    rows = b.read(PARENT / "加工特徴と方式選択/集計/BG公開QI一致の数値許容差監査.csv")
    assert columns.groupby(["phase", "team"]).size().eq(4).all()
    table = columns.groupby(["phase", "team"]).union_BG_value_share.min().rename("clinical4_min_BG_union_share").reset_index()
    table = table.merge(rows, on=["phase", "team"], validate="one_to_one")
    table["group"] = "other_or_unconfirmed"
    complete = table.clinical4_min_BG_union_share.ge(1 - 1e-12)
    table.loc[complete & table.approx_union_share.ge(.98), "group"] = "column_and_row_BG_preserved"
    table.loc[complete & table.approx_union_share.lt(.98), "group"] = "column_BG_preserved_row_partial"
    table.to_csv(OUT / "公開特徴による群_44C.csv", index=False)
    return table


def select_pair(dc, full, sorted_ref, ref_n, qs, cs, penalty):
    pair = b.smooth_score(dc, sorted_ref, ref_n) - b.stable.EPSILON * np.minimum(full, 1000.)
    return o.select(pair - penalty[:, None], dc, qs, cs, "sex_quota")


def other_penalty(qx, qs, collections, references, scale, sorted_ref, ref_n):
    nearest = []
    for team in references:
        c = collections[team]
        dc, _ = b.distances(qx, b.core.raw(c) / scale, qs, c.sex.to_numpy())
        minimum = dc.min(axis=1)
        scores = np.empty(len(qs))
        for i, (one, ref) in enumerate(zip(minimum, sorted_ref)):
            finite = ref[np.isfinite(ref)]
            count = np.interp(one, np.r_[0., finite], np.r_[0., np.arange(1, len(finite) + 1)], right=len(finite))
            scores[i] = -np.log((count + .25) / (ref_n + 1)) if np.isfinite(one) else -1e4
        nearest.append(scores)
    assert nearest
    return .25 * np.quantile(np.column_stack(nearest), .75, axis=1)


def run():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)
    features = groups()
    targets = sorted(features.loc[features.phase.eq("prelim") & features.group.eq("column_BG_preserved_row_partial"), "team"].tolist())
    assert targets == [19, 24]
    plan = dict(group_definition="each of clinical4 has BG union value share 1 within absolute tolerance 1e-8; approximate QI7 union row share < .98",
                feature_grouping_after_prior_results=True, group_thresholds_not_score_optimized=True,
                targets=targets, methods=METHODS, correction_quantile=.75, correction_lambda=.25,
                seeds=b.SEEDS, null_repeats=b.REPEATS, reference_protocol="existing 32 splits and sex-matched null draws",
                reference_exclusion="frozen original otherC reference list; no exclusion repair in this comparison",
                predictions_before_truth=True, new_independent_teams=0,
                background_conditions_are_not_independent_validation_units=True)
    dump(HERE / "固定条件.json", plan)
    folder = ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack"
    bg = b.read(ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
    assert len(bg) == 5143
    raw_bg, bgsex, setups = b.reference_setup(bg)
    refs = b.read(PARENT / "他Cとの差による選別/集計/他C参照の採否.csv")
    refs = refs[refs.phase.eq("prelim")]
    collections = {int(cp.stem.rsplit("_", 1)[1]): b.read(cp) for cp in sorted(folder.glob("C_*.csv"))}
    pending = []
    prior = b.read(PARENT / "他Cとの差による選別/集計/予備戦_対象別.csv")
    matching = 0
    for team in targets:
        q = b.read(folder / f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        c = collections[team]
        references = sorted(refs.loc[refs.team.eq(team) & refs.reason.eq("included"), "other_team"].tolist())
        assert team not in references
        qs, cs = q.sex.to_numpy(), c.sex.to_numpy()
        qr, cr = b.core.raw(q), b.core.raw(c)
        fullscale = o.scale_for(bg)
        db, full_bg = b.distances(qr / fullscale, raw_bg / fullscale, qs, bgsex)
        dc, full_c = b.distances(qr / fullscale, cr / fullscale, qs, cs)
        sorted_ref = np.sort(db, axis=1)
        penalty = other_penalty(qr / fullscale, qs, collections, references, fullscale, sorted_ref, len(bg))
        full = np.stack([select_pair(dc, full_c, sorted_ref, len(bg), qs, cs, np.zeros(len(q))),
                         select_pair(dc, full_c, sorted_ref, len(bg), qs, cs, penalty)])
        corr, _ = o.corrections(q, collections, references, db, fullscale)
        pair, original_dc = o.matrices(q, c, bg, fullscale, db)
        expected = o.predictions(pair, original_dc, q, c, corr)
        for j, method in enumerate(METHODS):
            assert np.array_equal(full[j], expected[method] == 9.5)
            matching += 1
        real = np.empty((len(setups), len(METHODS), len(q)), bool)
        null = np.empty((len(setups), b.REPEATS, len(METHODS), len(q)), bool)
        n0, n1 = int((cs == 0).sum()), int((cs == 1).sum())
        for i, setup in enumerate(setups):
            scale = setup["scale"]
            x = qr / scale
            dbg, fbg = b.distances(x, setup["norm_BG"], qs, bgsex)
            sorted_ref = np.sort(dbg[:, setup["ref"]], axis=1)
            ref_n = len(setup["ref"])
            penalty = other_penalty(x, qs, collections, references, scale, sorted_ref, ref_n)
            dc, full_c = b.distances(x, cr / scale, qs, cs)
            real[i, 0], _ = b.choose(dc, full_c, sorted_ref, ref_n, qs, cs)
            assert np.array_equal(real[i, 0], select_pair(dc, full_c, sorted_ref, ref_n, qs, cs, np.zeros(len(q))))
            real[i, 1] = select_pair(dc, full_c, sorted_ref, ref_n, qs, cs, penalty)
            for repeat, orders in enumerate(setup["draw_orders"]):
                sample = np.sort(np.r_[orders[0][:n0], orders[1][:n1]])
                assert len(sample) == len(c) == 1049 and len(np.unique(sample)) == 1049
                assert set(sample).isdisjoint(setup["ref"])
                null[i, repeat, 0], _ = b.choose(dbg[:, sample], fbg[:, sample], sorted_ref, ref_n, qs, bgsex[sample])
                null[i, repeat, 1] = select_pair(dbg[:, sample], fbg[:, sample], sorted_ref, ref_n, qs, bgsex[sample], penalty)
            if (i + 1) % 4 == 0:
                print(json.dumps(dict(stage="prediction", team=team, references=len(references), backgrounds_completed=i + 1, seconds=round(time.time() - start, 1))), flush=True)
        pending.append(dict(team=team, ids=q.challenge_row_id.to_numpy(), sex=qs, references=references, full=full, real=real, null=null))
    predpath = WORK / "固定2方式_採点前予測.npz"
    np.savez_compressed(predpath, teams=np.array(targets), methods=np.array(METHODS), seeds=np.array(b.SEEDS),
                        ids=np.stack([v["ids"] for v in pending]), sex=np.stack([v["sex"] for v in pending]),
                        full=np.stack([v["full"] for v in pending]), real=np.stack([v["real"] for v in pending]), null=np.stack([v["null"] for v in pending]))
    prediction_hash = sha(predpath)
    print(json.dumps(dict(stage="predictions_saved_before_truth", sha256=prediction_hash, seconds=round(time.time() - start, 1))), flush=True)
    truth_folder = next((ROOT / "予備戦/配布データ/結果").glob("正解*"))
    scores = []
    for item in pending:
        truth = b.read(truth_folder / f"aia_truth_{item['team']}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        assert truth.challenge_row_id.tolist() == item["ids"].tolist()
        for j, method in enumerate(METHODS):
            score = b.score_chosen(item["full"][j], truth)
            old = prior[prior.team.eq(item["team"]) & prior.method.eq(method)].iloc[0]
            assert abs(score["r_all"] - old.r_all) < 1e-12
            assert score["hit_m"] == old.hit_m and score["hit_c"] == old.hit_c
            scores.append(dict(team=item["team"], method=method, source="full_reference_real_C", seed=-1, repeat=-1, **score))
            for i, seed in enumerate(b.SEEDS):
                scores.append(dict(team=item["team"], method=method, source="split_real_C", seed=seed, repeat=-1,
                                   **b.score_chosen(item["real"][i, j], truth)))
                for repeat in range(b.REPEATS):
                    scores.append(dict(team=item["team"], method=method, source="split_BG_C", seed=seed, repeat=repeat,
                                       **b.score_chosen(item["null"][i, repeat, j], truth)))
    table = pd.DataFrame(scores)
    table.to_csv(OUT / "32背景_実Cと対照_全採点.csv", index=False)
    baseline_old = b.read(PARENT / "基準方式の背景感度/集計/32参照_実C対象別.csv")
    actual = table[table.source.eq("split_real_C")]
    base_new = actual[actual.method.eq("baseline")].merge(baseline_old, on=["team", "seed"], suffixes=("_new", "_old"), validate="one_to_one")
    assert len(base_new) == 64
    assert np.max(np.abs(base_new.r_all_new - base_new.r_all_old)) < 1e-12
    assert (base_new.hit_m_new == base_new.hit_m_old).all() and (base_new.hit_c_new == base_new.hit_c_old).all()
    controls = table[table.source.eq("split_BG_C")].groupby(["team", "seed", "method"])[["r_all", "signed_hit_difference"]].mean().add_prefix("BG_").reset_index()
    paired = actual.merge(controls, on=["team", "seed", "method"], validate="one_to_one")
    paired["excess_R"] = paired.r_all - paired.BG_r_all
    paired["excess_hit_gap"] = paired.signed_hit_difference - paired.BG_signed_hit_difference
    paired.to_csv(OUT / "32背景_実Cと対照差.csv", index=False)
    means = paired.groupby(["team", "method"])[["r_all", "signed_hit_difference", "BG_r_all", "excess_R", "excess_hit_gap"]].mean().reset_index()
    means.to_csv(OUT / "対象別平均.csv", index=False)
    deltas = paired.pivot(index=["team", "seed"], columns="method", values=["r_all", "signed_hit_difference", "excess_R", "excess_hit_gap"])
    changes = pd.DataFrame({name + "_delta": deltas[name][METHODS[1]] - deltas[name][METHODS[0]] for name in ["r_all", "signed_hit_difference", "excess_R", "excess_hit_gap"]}).reset_index()
    changes.to_csv(OUT / "基準方式との差_32背景.csv", index=False)
    counts = changes.groupby("team").r_all_delta.agg(mean="mean", min="min", max="max", wins=lambda v: int((v > 1e-12).sum()), losses=lambda v: int((v < -1e-12).sum()), ties=lambda v: int((v.abs() <= 1e-12).sum())).reset_index()
    counts.to_csv(OUT / "対象別_改善条件数.csv", index=False)
    dump(HERE / "実行記録.json", dict(plan=plan, seconds=time.time() - start, python=platform.python_version(), executable=sys.executable,
         code_sha256=sha(Path(__file__)), imported_code_sha256={str(Path(m.__file__).relative_to(ROOT)): sha(Path(m.__file__)) for m in [b, o, b.stable, b.core]},
         input_sha256=b.HASHES, prediction_sha256=prediction_hash, full_original_masks_matched=matching,
         baseline_existing_32_scores_matched=64, scored_rows=len(table),
         reference_teams={str(v["team"]): v["references"] for v in pending}, main_truth_read=False, B_self_read=False,
         official_phi=False, submission_created=False, github_publication=False))
    print(means.to_string(index=False), flush=True)
    print(counts.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
