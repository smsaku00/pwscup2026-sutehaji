"""対応競合候補の同点・行順・背景差し替えを確認する。

最高設定を探す分析から区別するため、部分2属性・性別一致に固定する。
背景差し替えは独自の対照条件であり、公式phiを再現しない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import 分析 as core
from 対応競合の分析 import partial_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPEATS = 30
PERMUTATIONS = 10


def distance(q, c, scale):
    x, y = core.raw(q) / scale, core.raw(c) / scale
    d = np.sort((x[:, None] - y[None]) ** 2, axis=2)[:, :, :2].mean(2)
    d[q.sex.to_numpy()[:, None] != c.sex.to_numpy()[None]] = np.inf
    return d


def pair_score(dc, db):
    sorted_bg = np.sort(db, axis=1)
    raw, smooth = np.empty_like(dc), np.empty_like(dc)
    for i, (row, ref) in enumerate(zip(dc, sorted_bg)):
        finite = ref[np.isfinite(ref)]
        count = np.searchsorted(finite, row, side="right")
        raw[i] = -np.log((count + .5) / (len(db[i]) + 1))
        # n=2の二乗距離では局所の体積が距離に比例する。CDFを区間補間する。
        xp = np.r_[0., finite]
        fp = np.r_[0., np.arange(1, len(finite) + 1)]
        interpolated = np.interp(row, xp, fp, right=len(finite))
        smooth[i] = -np.log((interpolated + .25) / (len(db[i]) + 1))
    raw[~np.isfinite(dc)] = -1e4
    smooth[~np.isfinite(dc)] = -1e4
    return {"discrete": raw, "continuous": smooth}


def predict(pair, dc, qsex, csex, mode, fraction):
    nearest = dc.argmin(1)
    if mode == "independent":
        return core.keep(pair[np.arange(len(pair)), nearest], fraction)
    # 元候補と同じ12近傍の制限。
    order = np.argsort(dc, axis=1, kind="stable")[:, :12]
    valid = np.zeros_like(pair, bool)
    np.put_along_axis(valid, order, True, axis=1)
    limited = np.where(valid, pair, -1e4)
    if mode == "one_to_one":
        return partial_assignment(limited, int(np.ceil(len(qsex)*fraction)))[0]
    result = np.zeros(len(qsex), bool)
    for sex in np.unique(qsex):
        a, b = qsex == sex, csex == sex
        result[a] = partial_assignment(limited[a][:, b], int(np.ceil(a.sum()*fraction)))[0]
    return result


def evaluate_all(pair, dc, qsex, csex, truth):
    result = []
    for cdf, scores in pair.items():
        for mode in ["independent", "one_to_one", "sex_quota"]:
            for fraction in [.5, .75]:
                selected = predict(scores, dc, qsex, csex, mode, fraction)
                result.append(dict(method=f"robust/{cdf}/{mode}/gate{fraction}", **core.scores(np.where(selected, 9.5, 0), truth)))
    return result


def run():
    start = time.time()
    folder = ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack"
    bg = core.read(ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
    raw = core.raw(bg)
    scale = np.maximum(np.quantile(raw, .75, axis=0)-np.quantile(raw, .25, axis=0), 1e-5)
    order = np.random.default_rng(core.SEED).permutation(len(bg))
    pool, ref = (bg.iloc[idx].reset_index(drop=True) for idx in [order[:len(bg)//2], order[len(bg)//2:]])
    samples = [np.sort(np.random.default_rng(core.SEED+400+r).choice(len(pool), 1049, False)) for r in range(REPEATS)]
    td = next((ROOT / "予備戦/配布データ/結果").glob("正解*"))
    rows, perm_rows = [], []
    for cp in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        k = int(cp.stem.rsplit("_", 1)[1])
        if k in [1, 2, 5]:
            continue
        q = core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        c = core.read(cp)
        truth = core.read(td/f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        dc, db = distance(q, c, scale), distance(q, bg, scale)
        full_pair = pair_score(dc, db)
        qs, cs = q.sex.to_numpy(), c.sex.to_numpy()
        rows += [dict(team=k, source="real_C_full_reference", repeat=-1, **v) for v in evaluate_all(full_pair, dc, qs, cs, truth)]
        for cdf, pair in full_pair.items():
            for mode in ["independent", "one_to_one"]:
                baseline = predict(pair, dc, qs, cs, mode, .75)
                for r in range(PERMUTATIONS):
                    rng = np.random.default_rng(core.SEED + k*100 + r)
                    a, b = rng.permutation(len(q)), rng.permutation(len(c))
                    perm = predict(pair[a][:, b], dc[a][:, b], qs[a], cs[b], mode, .75)
                    selected = perm[np.argsort(a)]
                    perm_rows.append(dict(team=k, method=f"robust/{cdf}/{mode}/gate0.75", repeat=r,
                        selected_changed=int((selected != baseline).sum()), **core.scores(np.where(selected, 9.5, 0), truth)))
        db_ref, dpool = distance(q, ref, scale), distance(q, pool, scale)
        split_pair = pair_score(dc, db_ref)
        pool_pair = pair_score(dpool, db_ref)
        rows += [dict(team=k, source="real_C_split_reference", repeat=-1, **v) for v in evaluate_all(split_pair, dc, qs, cs, truth)]
        for r, idx in enumerate(samples):
            pair = {name: values[:, idx] for name, values in pool_pair.items()}
            rows += [dict(team=k, source="background_C", repeat=r, **v) for v in evaluate_all(pair, dpool[:, idx], qs, pool.sex.to_numpy()[idx], truth)]
        print(json.dumps(dict(team=k, seconds=round(time.time()-start, 1))), flush=True)
    out = HERE / "集計"
    table = pd.DataFrame(rows)
    table.to_csv(out / "対応競合頑健性_反復.csv", index=False)
    per = pd.DataFrame(perm_rows)
    per.to_csv(out / "対応競合頑健性_行順変更.csv", index=False)
    table.groupby(["source", "method"])[["r_all", "p_m", "p_c"]].mean().to_csv(out / "対応競合頑健性_平均.csv")
    comparisons = []
    for (team, method), g in table.groupby(["team", "method"]):
        null = g[g.source.eq("background_C")].r_all.to_numpy()
        actual = g[g.source.eq("real_C_split_reference")].iloc[0].r_all
        comparisons.append(dict(team=team, method=method, real_R=actual, background_mean=null.mean(),
            excess_vs_background_mean=actual-null.mean(), background_q95=np.quantile(null, .95)))
    pd.DataFrame(comparisons).to_csv(out / "対応競合頑健性_背景との比較.csv", index=False)
    per.groupby("method").agg(changed_mean=("selected_changed", "mean"), changed_max=("selected_changed", "max"),
        mean_R=("r_all", "mean")).to_csv(out / "対応競合頑健性_行順変更平均.csv")
    (HERE / "対応競合頑健性_実行記録.json").write_text(json.dumps(dict(seed=core.SEED, repeats=REPEATS, permutations=PERMUTATIONS,
        selected_teams="all except 1,2,5", part_dimension=2, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=core.HASHES, seconds=time.time()-start, is_official_phi=False, submission_created=False), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(table.groupby(["source", "method"])[["r_all", "p_m", "p_c"]].mean().to_string(), flush=True)
    print(per.groupby("method").selected_changed.agg(["mean", "max"]).to_string(), flush=True)


if __name__ == "__main__":
    run()
