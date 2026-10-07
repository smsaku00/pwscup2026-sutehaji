"""固定候補による丸め耐性のある条件付き時間窓の分析。

提出物は作成しない。予測には課題、C、公開背景のみを渡し、予備戦正解は
全予測が完了してから採点する。本戦自チーム正解は診断のみ。
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
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").exists())
OLD = HERE.parent.parent / "20261007_AIA全体向上の分析"
spec = importlib.util.spec_from_file_location("aia_conditioned_core", OLD / "分析.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

SEED = 20261008
STEPS = {"age": [1, 2, 5, 10], "SBP": [.5, 1, 2, 5, 10, 20],
         "TG": [.5, 1, 2, 5, 10, 20, 50], "HDL": [.5, 1, 2, 5, 10],
         "ALT": [.5, 1, 2, 5, 10, 20]}
METHODS = ["C/raw_k8_window", "C/box_k8_window", "C/box_k32_window",
           "C/box_k8_contrast", "C/box_k8_window_gate75", "C/box_k8_contrast_gate75"]
# BG/rawは厳密なC-free対照。BG/boxはCの格子だけを転用する形状対照。
CONTROLS = ["BG/raw_k40_window", "BG/box_k40_window", "constant9.5"]
PLAN = dict(methods=METHODS, controls=CONTROLS, k_C=[8, 32], k_BG=40,
            rounding_grid_candidates=STEPS, grid_coverage=.98, time_tolerance=.5,
            grid_inferred_from_C_only=True, sex_exact=True, prefecture_used=False,
            gating_fraction=.75, candidate_selection_before_truth=True,
            truth_used_for_fitting=False, official_phi_estimated=False,
            submission_created=False, codabench_submitted=False)


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rounding_widths(c):
    """候補刻みがCの98%以上を説明する場合のみ最大幅を採用する。

    grid原点は最頻剰余で求める。年代等の自然な整数表記も含むため、加工手法の
    確定ではない。この幅を使う/使わない両方を固定比較する。
    """
    widths, records = [], []
    for col in core.NUM:
        values = c[col].to_numpy(float)
        width = 0.
        coverage = 0.
        origin = 0.
        for step in STEPS[col]:
            remainder = np.mod(values, step)
            rounded = np.round(remainder / step, 6)
            rounded[rounded == 1] = 0.
            modes, counts = np.unique(rounded, return_counts=True)
            offset = modes[np.argmax(counts)] * step
            residual = np.abs(values - (np.rint((values-offset)/step)*step+offset))
            share = float((residual <= max(1e-6, step*2e-6)).mean())
            if share >= .98:
                width, coverage, origin = step, share, float(offset)
        widths.append(width)
        records.append(dict(column=col, grid_step=width, grid_coverage=coverage, grid_origin=origin))
    return np.array(widths), records


def distance(q, ref, bg, width):
    b = core.raw(bg)
    scale = np.maximum(np.quantile(b, .75, axis=0)-np.quantile(b, .25, axis=0), 1e-5)
    x = core.raw(q)
    values = ref[core.NUM].to_numpy(float)
    lower, upper = values-width/2, values+width/2
    lower[:, [2, 4]] = np.log1p(np.maximum(lower[:, [2, 4]], 0))
    upper[:, [2, 4]] = np.log1p(np.maximum(upper[:, [2, 4]], 0))
    delta = np.maximum(np.maximum(lower[None]-x[:, None], x[:, None]-upper[None]), 0)
    d = ((delta / scale)**2).mean(2)
    d[q.sex.to_numpy()[:, None] != ref.sex.to_numpy()[None]] = np.inf
    return d


def local_distribution(d, times, k):
    """k番目と同じ距離の行を含め、C行順に依存しない局所分布にする。"""
    arrays = []
    for row in d:
        finite = np.flatnonzero(np.isfinite(row))
        assert len(finite)
        radius = np.partition(row[finite], min(k, len(finite))-1)[min(k, len(finite))-1]
        chosen = finite[row[finite] <= radius+1e-14]
        h = max(.35*radius, 1e-8)
        weights = np.exp(-np.minimum(row[chosen]/h, 700))
        weights /= weights.sum()
        # 同じtimeをまとめる。計算順を固定して並べ替えの丸め誤差も減らす。
        order = np.lexsort((row[chosen], times[chosen]))
        tt, ww = times[chosen][order], weights[order]
        unique, first = np.unique(tt, return_index=True)
        arrays.append((unique, np.add.reduceat(ww, first), len(chosen)))
    return arrays


def best_window(ct, cw, bt=None, bw=None):
    """各区間の端と区間間の中点を比較し、連続timeの最大質量窓を選ぶ。

    差分の場合はC条件付き質量からBG条件付き質量を引く。端の浮動小数点
    境界も候補として評価。9.5が同率最大なら9.5、その他はC中央値に近い候補。
    """
    t = ct if bt is None else np.r_[ct, bt]
    endpoints = np.unique(np.clip(np.r_[t-.5, t+.5, 0., 10.], 0., 10.))
    median = ct[np.searchsorted(np.cumsum(cw), .5, side="left")]
    candidate = np.unique(np.r_[endpoints, (endpoints[1:]+endpoints[:-1])/2,
                                np.nextafter(endpoints, 0), np.nextafter(endpoints, 10), 9.5, median])
    mass = ((np.abs(candidate[:, None]-ct[None]) <= .5) * cw).sum(1)
    if bt is not None:
        mass -= ((np.abs(candidate[:, None]-bt[None]) <= .5) * bw).sum(1)
    best = float(mass.max())
    valid = np.flatnonzero(mass >= best-1e-12)
    at95 = np.flatnonzero(candidate[valid] == 9.5)
    if len(at95):
        pos = valid[at95[0]]
    else:
        dist = np.abs(candidate[valid]-median)
        tie = valid[dist <= dist.min()+1e-12]
        pos = tie[-1]
    return float(candidate[pos]), best


def density_score(dc, db):
    """同じ丸め区間・尺度でCと背景の最近傍差を計算。"""
    return np.log((db.min(1)+1e-8)/(dc.min(1)+1e-8))


def sex_keep(score, q):
    out = np.zeros(len(q), bool)
    for sex in np.unique(q.sex):
        m = q.sex.eq(sex).to_numpy()
        # 同点で別の行を恣意的に切らない。選択数が75%を超える場合は記録する。
        count = int(np.ceil(m.sum()*.75))
        cutoff = np.sort(score[m])[len(score[m])-count]
        out[m] = score[m] >= cutoff-1e-12
    return out


def predict(q, c, bg):
    width, grid = rounding_widths(c)
    dc_raw, db_raw = (distance(q, v, bg, np.zeros(5)) for v in (c, bg))
    dc, db = (distance(q, v, bg, width) for v in (c, bg))
    c8 = local_distribution(dc, c.time.to_numpy(), 8)
    c32 = local_distribution(dc, c.time.to_numpy(), 32)
    cr8 = local_distribution(dc_raw, c.time.to_numpy(), 8)
    b40 = local_distribution(db, bg.time.to_numpy(), 40)
    br40 = local_distribution(db_raw, bg.time.to_numpy(), 40)
    pred = {name: np.empty(len(q)) for name in METHODS+CONTROLS}
    mass, contrast = [], []
    for i in range(len(q)):
        pred["C/raw_k8_window"][i], _ = best_window(*cr8[i][:2])
        pred["C/box_k8_window"][i], cm = best_window(*c8[i][:2])
        pred["C/box_k32_window"][i], _ = best_window(*c32[i][:2])
        pred["C/box_k8_contrast"][i], diff = best_window(*c8[i][:2], *b40[i][:2])
        pred["BG/raw_k40_window"][i], _ = best_window(*br40[i][:2])
        pred["BG/box_k40_window"][i], _ = best_window(*b40[i][:2])
        mass.append(cm); contrast.append(diff)
    score = density_score(dc, db)
    keep = sex_keep(score, q)
    pred["C/box_k8_window_gate75"] = np.where(keep, pred["C/box_k8_window"], 0)
    pred["C/box_k8_contrast_gate75"] = np.where(keep, pred["C/box_k8_contrast"], 0)
    pred["constant9.5"][:] = 9.5
    assert all(np.isfinite(v).all() and ((0 <= v) & (v <= 10)).all() for v in pred.values())
    diagnostics = dict(grid=grid, selected=keep, score=score, mass=np.array(mass), contrast=np.array(contrast),
                       near8=np.array([v[2] for v in c8]), near32=np.array([v[2] for v in c32]))
    return pred, diagnostics


def holdout(table, targets, label):
    records = []
    for test in targets:
        train = table[table.team.isin([v for v in targets if v != test]) & table.method.isin(METHODS)]
        average = train.groupby("method").r_all.mean().reindex(METHODS)
        method = average.idxmax()
        one = table[table.team.eq(test) & table.method.eq(method)].iloc[0]
        records.append(dict(group=label, held_out_team=test, selected_method=method,
                            train_R=float(average.loc[method]), **{k: float(one[k]) for k in ["r_all", "r_rare", "r_max", "p_m", "p_c"]}))
    return records


def run():
    started = time.time()
    HERE.mkdir(parents=True, exist_ok=True)
    out = HERE / "集計"; out.mkdir(exist_ok=True)
    planfile = HERE / "事前固定候補.json"
    if planfile.exists():
        assert json.loads(planfile.read_text()) == PLAN, "固定候補の上書きを防ぐ"
    else:
        dump(planfile, PLAN)
    prelim = ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack"
    main = ROOT / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
    sets = [("prelim", prelim, ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
            ("main", main, main / "A_bg.csv")]
    items, diagnostics, rounding, permutations = [], [], [], []
    # 予備戦・本戦を全部予測してから正解のファイルを開く。
    for phase, folder, bgpath in sets:
        bg = core.read(bgpath)
        for cp in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            team = int(cp.stem.rsplit("_", 1)[1])
            q = core.read(folder / f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c = core.read(cp)
            pred, extra = predict(q, c, bg)
            items.append(dict(phase=phase, team=team, q=q, pred=pred, extra=extra))
            rounding += [dict(phase=phase, team=team, **v) for v in extra["grid"]]
            for name, prediction in pred.items():
                diagnostics.append(dict(phase=phase, team=team, method=name, zero=int((prediction == 0).sum()),
                    long=int((prediction >= 9).sum()), short=int(((prediction > 0) & (prediction < 9)).sum()),
                    median=float(np.median(prediction)), distinct=int(len(np.unique(prediction))),
                    selection_total=int(extra["selected"].sum()), max_near8=int(extra["near8"].max()),
                    mean_window_mass=float(extra["mass"].mean()), mean_window_contrast=float(extra["contrast"].mean())))
            # 1回のC・背景の行順入れ替えを全44Cで検証。少数固定候補の実装確認。
            rng = np.random.default_rng(SEED + team)
            moved, _ = predict(q, c.iloc[rng.permutation(len(c))].reset_index(drop=True),
                                bg.iloc[rng.permutation(len(bg))].reset_index(drop=True))
            for name in pred:
                permutations.append(dict(phase=phase, team=team, method=name,
                    changed_predictions=int((np.abs(moved[name]-pred[name]) > 1e-10).sum()),
                    max_difference=float(np.max(np.abs(moved[name]-pred[name])))))
            print(json.dumps(dict(phase=phase, team=team, seconds=round(time.time()-started, 1))), flush=True)
    td = next((ROOT / "予備戦/配布データ/結果").glob("正解*"))
    truth_map, rows, differences, members = {}, [], [], []
    for item in items:
        if item["phase"] != "prelim":
            continue
        team, q = item["team"], item["q"]
        truth = core.read(td / f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        assert q.challenge_row_id.tolist() == truth.challenge_row_id.tolist()
        truth_map[team] = truth
        for name, prediction in item["pred"].items():
            result = core.scores(prediction, truth)
            result["r_unclipped"] = (result["p_m"]-result["p_c"])/(1-result["p_c"]) if result["p_c"] < 1 else 0.
            rows.append(dict(team=team, method=name, **result))
            for sex in np.unique(q.sex):
                mask = q.sex.eq(sex).to_numpy()
                for label in ["member", "control"]:
                    labelmask = truth.label.eq(label).to_numpy() & mask
                    hit = np.abs(prediction-truth.time.to_numpy()) <= .5
                    members.append(dict(team=team, method=name, sex=int(sex), label=label, n=int(labelmask.sum()), hits=int(hit[labelmask].sum())))
        for name in METHODS:
            baseline = "BG/raw_k40_window" if name == "C/raw_k8_window" else "BG/box_k40_window"
            a, b = core.scores(item["pred"][name], truth), core.scores(item["pred"][baseline], truth)
            differences.append(dict(team=team, method=name, background_method=baseline,
                delta_R=a["r_all"]-b["r_all"], delta_pm=a["p_m"]-b["p_m"], delta_pc=a["p_c"]-b["p_c"],
                delta_member_control_gap=(a["p_m"]-a["p_c"])-(b["p_m"]-b["p_c"])))
    table = pd.DataFrame(rows)
    table.to_csv(out / "予備戦_対象別.csv", index=False)
    pd.DataFrame(differences).to_csv(out / "背景対照との差.csv", index=False)
    pd.DataFrame(members).to_csv(out / "性別と会員別の的中数.csv", index=False)
    groups = [("20対象_自5除外", sorted(truth_map.keys()-{5})),
              ("18対象_1_2_5除外", sorted(truth_map.keys()-{1, 2, 5}))]
    summaries, holdouts = [], []
    metrics = ["r_all", "r_rare", "r_max", "p_m", "p_c", "r_unclipped"]
    for label, targets in groups:
        t = table[table.team.isin(targets)]
        s = t.groupby("method")[metrics].mean().reset_index()
        s.insert(0, "group", label); s["targets"] = len(targets)
        summaries.append(s)
        holdouts += holdout(table, targets, label)
    pd.concat(summaries).to_csv(out / "予備戦_20対象と18対象平均.csv", index=False)
    h = pd.DataFrame(holdouts)
    h.to_csv(out / "チーム留保選択_対象別.csv", index=False)
    h.groupby("group")[metrics[:-1]].mean().to_csv(out / "チーム留保選択_平均.csv")
    pd.DataFrame(diagnostics).to_csv(out / "予測動作_診断.csv", index=False)
    pd.DataFrame(rounding).to_csv(out / "Cから検出した格子.csv", index=False)
    p = pd.DataFrame(permutations)
    p.to_csv(out / "Cと背景の行順変更.csv", index=False)
    # 自チームは評価や方法の選択が完了した後に診断する。
    own = core.read(next((ROOT / "本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    item = next(v for v in items if v["phase"] == "main" and v["team"] == 28)
    joined = item["q"].merge(own[core.QI+["time", "is_rare"]], on=core.QI, how="left", validate="one_to_one")
    m = joined.time.notna().to_numpy(); r = joined.is_rare.fillna(False).to_numpy(bool)
    assert m.sum() == 163 and r.sum() == 13
    self_rows = []
    for name, prediction in item["pred"].items():
        hit = np.abs(prediction-joined.time.to_numpy(float)) <= .5
        self_rows.append(dict(method=name, n_member=163, hit_member=int(hit[m].sum()), p_m=float(hit[m].mean()),
            hit_member_rare=int(hit[r].sum()), selected_member=int((item["extra"]["selected"] & m).sum()),
            selected_control=int((item["extra"]["selected"] & ~m).sum()),
            selection_auc=float(roc_auc_score(m, item["extra"]["score"])), control_time_known=False,
            R_unknown=True, diagnosis_only=True))
    pd.DataFrame(self_rows).to_csv(out / "本戦自28_診断のみ.csv", index=False)
    checked = core.validate_scores(truth_map)
    dump(HERE / "実行記録.json", dict(plan=PLAN, seed=SEED, python=platform.python_version(), executable=sys.executable,
        seconds=time.time()-started, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_code_sha256=hashlib.sha256((OLD / "分析.py").read_bytes()).hexdigest(), input_sha256=core.HASHES,
        official_score_check=checked, main_truth_used_for_method_selection=False,
        permutation_comparisons=len(p), changed_predictions=int(p.changed_predictions.sum()),
        R_is_before_unknown_phi=True, strict_C_free_background="BG/raw_k40_window",
        background_box_uses_C_grid_only=True, submission_created=False, codabench_submitted=False))
    print(pd.concat(summaries).to_string(index=False), flush=True)
    print(h.groupby("group")[metrics[:-1]].mean().to_string(), flush=True)
    print(p.groupby("method").changed_predictions.sum().to_string(), flush=True)


if __name__ == "__main__":
    run()
