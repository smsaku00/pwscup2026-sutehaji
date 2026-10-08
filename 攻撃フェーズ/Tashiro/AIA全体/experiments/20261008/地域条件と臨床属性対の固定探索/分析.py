"""地域条件付きの列値集合と、年齢を主尺度から外した臨床6対を固定比較する。

4追加則・P01/k1対照3則。4未使用背景seedと2種類の条件付きbootstrap。
予測の保存・hash固定後に公開予備正解を読む。候補・群・seedを得点で変更しない。
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
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
P = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").exists())
OUT = HERE / "集計"
WORK = HERE / "作業用"
SEEDS = list(range(20261040, 20261044))
REPEATS = 5
METHODS = ["P01", "column_all4", "column_top2", "pref_column_all4", "pref_column_top2", "clinical_pair_top1", "clinical_pair_top3"]
CLINICAL = ["SBP", "TG", "HDL", "ALT"]
PAIRS = list(combinations(range(1, 5), 2))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


b = module("regional_fixed_P01", P / "基準方式の背景感度/分析.py")
v = module("regional_fixed_columns", P / "属性値集合による選別/分析.py")
a = module("regional_fixed_pairs", P / "属性対ごとの証拠統合/分析.py")
core = b.core


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(q, ref):
    raw = core.raw(ref)
    scale = np.maximum(np.quantile(raw, .75, axis=0) - np.quantile(raw, .25, axis=0), 1e-5)
    x, y = core.raw(q) / scale, raw / scale
    sq = (x[:, None] - y[None]) ** 2
    mismatch = q.sex.to_numpy()[:, None] != ref.sex.to_numpy()[None]
    adaptive = np.sort(sq, axis=2)[:, :, :2].mean(axis=2)
    adaptive[mismatch] = np.inf
    pairs = []
    for i, j in PAIRS:
        distance = (sq[:, :, i] + sq[:, :, j]) / 2
        distance[mismatch] = np.inf
        pairs.append(np.sort(distance, axis=1))
    return dict(scale=scale, x=x, adaptive=np.sort(adaptive, axis=1), pair_refs=pairs, n=len(ref), ref=ref)


def column_pref(q, c, ref):
    dc, db = np.empty((len(q), 4)), np.empty((len(q), 4))
    fallback = 0
    for (sex, prefecture), index in q.groupby(["sex", "prefecture"], sort=True).groups.items():
        index = np.asarray(index)
        cm = c.sex.eq(sex) & c.prefecture.eq(prefecture)
        bm = ref.sex.eq(sex) & ref.prefecture.eq(prefecture)
        # 両側を同じ条件にする。どちらかの層が空なら両側とも性別条件へ戻す。
        if not cm.any() or not bm.any():
            cm, bm = c.sex.eq(sex), ref.sex.eq(sex)
            fallback += len(index)
        for j, col in enumerate(CLINICAL):
            dc[index, j] = v.nearest1d(q.loc[index, col], c.loc[cm, col])
            db[index, j] = v.nearest1d(q.loc[index, col], ref.loc[bm, col])
    ratio = np.log((db + v.EPS) / (dc + v.EPS))
    assert np.isfinite(ratio).all()
    return dict(all4=ratio.mean(axis=1), top2=np.sort(ratio, axis=1)[:, -2:].mean(axis=1)), fallback


def predict(q, c, info):
    """予測側には正解・member label・チーム番号を渡さない。"""
    qs, cs = q.sex.to_numpy(), c.sex.to_numpy()
    y = core.raw(c) / info["scale"]
    sq = (info["x"][:, None] - y[None]) ** 2
    mismatch = qs[:, None] != cs[None]
    dc = np.sort(sq, axis=2)[:, :, :2].mean(axis=2)
    dc[mismatch] = np.inf
    full = cdist(info["x"], y, "sqeuclidean") / 5
    baseline, bad = b.choose(dc, full, info["adaptive"], info["n"], qs, cs)
    ev = v.evidence(q, c, info["ref"])
    pref, fallback = column_pref(q, c, info["ref"])
    chosen = [baseline,
              v.independent_select(q, ev, "column_all4_mean"), v.independent_select(q, ev, "column_top2_mean"),
              v.independent_select(q, pref, "column_all4_mean"), v.independent_select(q, pref, "column_top2_mean")]
    scores = []
    for (i, j), sorted_ref in zip(PAIRS, info["pair_refs"]):
        d = (sq[:, :, i] + sq[:, :, j]) / 2
        d[mismatch] = np.inf
        scores.append(a.continuous_score(d, sorted_ref, info["n"]))
    scores = np.sort(np.stack(scores), axis=0)
    for combined in [scores[-1], scores[-3:].mean(axis=0)]:
        eligible = np.where(mismatch, -np.inf, combined)
        limit = np.partition(eligible, -12, axis=1)[:, -12]
        valid = ~mismatch & (eligible >= limit[:, None])
        mask, invalid = a.select(combined - b.stable.EPSILON * np.minimum(full, 1000), valid, qs, cs)
        bad += invalid
        chosen.append(mask)
    result = np.stack(chosen)
    for sex in np.unique(qs):
        assert (result[:, qs == sex].sum(axis=1) == int(np.ceil((qs == sex).sum() * .75))).all()
    return result, dict(pref_fallback_rows=fallback, invalid_assignments=bad)


def bootstrap(pool, c, seed, repeat, mode):
    # 2条件とも復元抽出し、重複の扱いを統一する。人数は各実Cの構成に固定。
    keys = ["sex"] if mode == "sex" else ["sex", "prefecture"]
    rng = np.random.default_rng(seed * 100 + repeat)
    indexes = []
    for key, rows in c.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        mask = np.ones(len(pool), bool)
        for col, value in zip(keys, key):
            mask &= pool[col].to_numpy() == value
        available = np.flatnonzero(mask)
        assert len(available) > 0, (keys, key)
        indexes.extend(rng.choice(available, len(rows), replace=True).tolist())
    indexes = np.sort(np.asarray(indexes))
    sample = pool.iloc[indexes].reset_index(drop=True)
    assert len(sample) == len(c) == 1049
    assert sample.groupby(keys).size().equals(c.groupby(keys).size())
    return sample, dict(unique_source_rows=len(np.unique(indexes)), null_stratification=mode)


def run():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)
    plan = dict(methods=METHODS, new_rules=4, clinical_pairs=[[core.NUM[i], core.NUM[j]] for i, j in PAIRS],
         pref_empty_stratum="both C and BG fallback to sex-only", pair_tie_break="original full5 tiny1e-10; age excluded from primary6-pair score",
         select_fraction=.75, selected_answer=9.5, other_answer=0, seeds=SEEDS, null_repeats=REPEATS,
         reference="phase BG only; same scale and reference for actual and null", null_pool="phase pool half plus all old-kit public BG, phase reference source rows excluded",
         null_models=["sex", "sex_prefecture"], both_nulls_use_replacement=True, nulls_are_diagnostics_not_official_phi=True,
         regional_group="maximum prefecture share in public C >=.25; diagnostic only, no method switching",
         existing_groups="frozen previous5 groups; other13 not assumed homogeneous", candidate_or_group_tuning=False,
         design_after_old_results=True, independent_new_teams=0, predictions_before_truth=True)
    dump(HERE / "固定条件.json", plan)
    features = b.read(P / "加工特徴と方式選択/集計/公開Cの加工特徴_44組.csv")
    oldgroups = json.loads((P / "特徴群での固定候補検証/再集計記録.json").read_text())["groups"]
    frozen_columns = b.read(P / "属性値集合による選別/集計/予備戦_対象別と20反復null.csv")
    frozen_base = b.read(P / "基準方式の背景感度/集計/全BG基準_予備戦21対象.csv")
    kit = b.read(ROOT / "スターターキット/初版/participant_data/A_bg.csv")
    phases = [("prelim", ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack", ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
              ("main", ROOT / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力", ROOT / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]
    pending, diagnostics = [], []
    for phase, folder, bgpath in phases:
        bg = b.read(bgpath)
        assert len(bg) == 5143 and len(kit) == 5143
        setups = []
        for seed in SEEDS:
            order = np.random.default_rng(seed).permutation(len(bg))
            pool_ix, ref_ix = order[:len(bg)//2], order[len(bg)//2:]
            assert set(pool_ix).isdisjoint(ref_ix)
            pool = pd.concat([bg.iloc[pool_ix], kit], ignore_index=True)
            setups.append((seed, pool, bg.iloc[ref_ix].reset_index(drop=True)))
        for cp in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            team = int(cp.stem.rsplit("_", 1)[1])
            q = b.read(folder / f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c = b.read(cp)
            full, diag = predict(q, c, prepare(q, bg))
            original_pair, original_dc = b.stable.matrices(q, c, bg)
            assert np.array_equal(full[0], b.stable.select(original_pair, original_dc, q.sex.to_numpy(), c.sex.to_numpy(), "sex_quota"))
            diagnostics.append(dict(phase=phase, team=team, seed=-1, source="actual", repeat=-1, **diag))
            real = np.empty((len(SEEDS), len(METHODS), len(q)), bool)
            null = np.empty((len(SEEDS), 2, REPEATS, len(METHODS), len(q)), bool) if phase == "prelim" and team != 5 else None
            for i, (seed, pool, ref) in enumerate(setups):
                info = prepare(q, ref)
                real[i], diag = predict(q, c, info)
                diagnostics.append(dict(phase=phase, team=team, seed=seed, source="actual", repeat=-1, **diag))
                if null is not None:
                    for n, mode in enumerate(["sex", "sex_prefecture"]):
                        for repeat in range(REPEATS):
                            sample, extra = bootstrap(pool, c, seed, repeat, mode)
                            null[i, n, repeat], diag = predict(q, sample, info)
                            diagnostics.append(dict(phase=phase, team=team, seed=seed, source=mode, repeat=repeat, **diag, **extra))
            pending.append(dict(phase=phase, team=team, ids=q.challenge_row_id.to_numpy(), sex=q.sex.to_numpy(), full=full, real=real, null=null))
            print(json.dumps(dict(stage="predictions", phase=phase, team=team, seconds=round(time.time() - start, 1))), flush=True)
    assert len(pending) == 44
    arrays = {"methods":np.array(METHODS), "seeds":np.array(SEEDS)}
    for phase in ["prelim", "main"]:
        items = [v for v in pending if v["phase"] == phase]
        arrays[phase + "_teams"] = np.array([v["team"] for v in items])
        for name in ["ids", "sex", "full", "real"]:
            arrays[phase + "_" + name] = np.stack([v[name] for v in items])
    items = [v for v in pending if v["null"] is not None]
    arrays["null_teams"] = np.array([v["team"] for v in items])
    arrays["null"] = np.stack([v["null"] for v in items])
    path = WORK / "全予測_公開正解読込み前.npz"
    np.savez_compressed(path, **arrays)
    prediction_hash = sha(path)
    print(json.dumps(dict(stage="predictions_saved_before_truth", sha256=prediction_hash, seconds=round(time.time() - start, 1))), flush=True)
    td = next((ROOT / "予備戦/配布データ/結果").glob("正解*"))
    scores = []
    baseline_matched = 0
    for item in pending:
        if item["phase"] != "prelim":
            continue
        team = item["team"]
        truth = b.read(td / f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        assert np.array_equal(item["ids"], truth.challenge_row_id.to_numpy())
        for j, method in enumerate(METHODS):
            one = b.score_chosen(item["full"][j], truth)
            scores.append(dict(team=team, method=method, source="full_actual", seed=-1, repeat=-1, **one))
            if method == "P01":
                old = frozen_base[frozen_base.team.eq(team)].iloc[0]
                assert one["hit_m"] == old.hit_m and one["hit_c"] == old.hit_c
                baseline_matched += 1
            elif method in ["column_all4", "column_top2"]:
                oldname = "column_all4_mean" if method == "column_all4" else "column_top2_mean"
                old = frozen_columns[frozen_columns.team.eq(team) & frozen_columns.method.eq(oldname) & frozen_columns.context.eq("full_reference_overlap") & frozen_columns.source.eq("real_C")].iloc[0]
                assert one["hit_m"] == old.hit_m and one["hit_c"] == old.hit_c
                baseline_matched += 1
            for i, seed in enumerate(SEEDS):
                scores.append(dict(team=team, method=method, source="split_actual", seed=seed, repeat=-1, **b.score_chosen(item["real"][i, j], truth)))
                if item["null"] is not None:
                    for n, mode in enumerate(["sex", "sex_prefecture"]):
                        for repeat in range(REPEATS):
                            scores.append(dict(team=team, method=method, source="BG_bootstrap_" + mode, seed=seed, repeat=repeat,
                                               **b.score_chosen(item["null"][i, n, repeat, j], truth)))
    table = pd.DataFrame(scores)
    table.to_csv(OUT / "予備戦_全採点.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "公開対象と対照の動作.csv", index=False)
    actual = table[table.source.eq("split_actual") & table.team.ne(5)]
    bgmeans = table[table.source.str.startswith("BG_bootstrap_")].groupby(["team", "seed", "method", "source"])[["r_all", "signed_hit_difference"]].mean().reset_index()
    paired = actual.merge(bgmeans, on=["team", "seed", "method"], suffixes=("_actual", "_null"), validate="one_to_many")
    paired["excess_R"] = paired.r_all_actual - paired.r_all_null
    paired["excess_hit_gap"] = paired.signed_hit_difference_actual - paired.signed_hit_difference_null
    paired.to_csv(OUT / "背景条件ごとの実Cと対照差.csv", index=False)
    means = paired.groupby(["team", "method", "source_null"])[["r_all_actual", "r_all_null", "excess_R", "excess_hit_gap"]].mean().reset_index()
    means.to_csv(OUT / "対象別_背景対照2条件.csv", index=False)
    scope = dict(oldgroups)
    concentrated = features[features.phase.eq("prelim") & features.team.ne(5) & features.prefecture_mode_share.ge(.25)].team.tolist()
    scope["regional_concentration_ge025"] = concentrated
    scope["regional_concentration_lt025"] = [v["team"] for v in pending if v["phase"] == "prelim" and v["team"] != 5 and v["team"] not in concentrated]
    records = []
    for name, teams in scope.items():
        for context in ["full_actual", "split_actual"]:
            sub = table[table.team.isin(teams) & table.source.eq(context)]
            base = sub[sub.method.eq("P01")].set_index(["team", "seed"]).r_all
            for method, rows in sub.groupby("method", sort=False):
                delta = rows.set_index(["team", "seed"]).r_all - base
                by_team = delta.groupby(level="team").mean()
                records.append(dict(group=name, teams=" ".join(map(str, teams)), context=context, method=method, targets=len(teams),
                     r_all=float(rows.r_all.mean()), delta_R=float(delta.mean()), wins=int(by_team.gt(1e-12).sum()),
                     losses=int(by_team.lt(-1e-12).sum()), ties=int(by_team.abs().le(1e-12).sum())))
    pd.DataFrame(records).to_csv(OUT / "特徴群ごとの固定7方式.csv", index=False)
    dump(HERE / "実行記録.json", dict(plan=plan, seconds=time.time() - start, python=platform.python_version(), executable=sys.executable,
         code_sha256=sha(Path(__file__)), input_sha256=b.HASHES, prediction_sha256=prediction_hash,
         imported_code_sha256={str(Path(m.__file__).relative_to(ROOT)):sha(Path(m.__file__)) for m in [b, v, a, b.stable, core]},
         full_control_scores_matched=baseline_matched, scored_rows=len(table), predictions_after_truth=0,
         B_self_read=False, main_truth_read=False, official_phi=False, submission_created=False))
    print(pd.DataFrame(records).query("context == 'split_actual'").to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
