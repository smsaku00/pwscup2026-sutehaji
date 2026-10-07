"""C行から課題と背景への近さを比較し、一対一対応に加える固定比較。

出所スコアを真の会員確率・校正済み密度比とは呼ばない。
背景対照は公式phiではない。全予測後に予備戦の公開正解で評価する。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = ROOT / "本戦/攻撃フェーズ/Tashiro/20261007_AIA全体向上の分析"
sys.path.insert(0, str(OLD))
import 分析 as core
from 同点に依存しない対応 import select
from 対応競合の頑健性 import pair_score

SEED = 2026100810
REPEATS = 20
METHODS = (("baseline", 0, 0.), ("row_partial2_l025", 2, .25),
           ("row_partial2_l1", 2, 1.), ("row_full5_l025", 5, .25), ("row_full5_l1", 5, 1.))


def distances(x, y, xs, ys, dimension):
    if dimension == 5:
        d = cdist(x, y, "sqeuclidean") / 5
    else:
        d = np.sort((x[:, None] - y[None]) ** 2, axis=2)[:, :, :2].mean(2)
    d[xs[:, None] != ys[None]] = np.inf
    return d


def reverse_bg_min(y, z, ys, zs):
    result = {d: np.empty(len(y)) for d in (2, 5)}
    for start in range(0, len(y), 128):
        end = min(len(y), start + 128)
        for dimension in (2, 5):
            result[dimension][start:end] = distances(y[start:end], z, ys[start:end], zs, dimension).min(1)
    return result


def make_matrices(q, c, ref):
    z = core.raw(ref)
    scale = np.maximum(np.quantile(z, .75, axis=0) - np.quantile(z, .25, axis=0), 1e-5)
    x, y, z = (core.raw(f) / scale for f in (q, c, ref))
    qs, cs, bs = (f.sex.to_numpy() for f in (q, c, ref))
    dc = distances(x, y, qs, cs, 2)
    db = distances(x, z, qs, bs, 2)
    pair = pair_score(dc, db)["continuous"]
    full = distances(x, y, qs, cs, 5)
    pair -= 1e-10 * np.minimum(np.where(np.isfinite(full), full, 1000.), 1000.)
    minima = reverse_bg_min(y, z, cs, bs)
    reverse = {}
    for dimension, query_distance in ((2, dc), (5, full)):
        count_correction = np.zeros(len(c))
        for sex in np.unique(cs):
            count_correction[cs == sex] = np.log((bs == sex).sum() / (qs == sex).sum())
        reverse[dimension] = count_correction + dimension / 2 * np.log(
            (minima[dimension] + 1e-10) / (query_distance.min(0) + 1e-10))
        assert np.isfinite(reverse[dimension]).all()
    return pair, dc, reverse


def predict(pair, dc, reverse, qs, cs):
    result = {}
    for method, dimension, strength in METHODS:
        adjusted = pair if dimension == 0 else pair + strength * reverse[dimension][None, :]
        chosen = select(adjusted, dc, qs, cs, "sex_quota")
        result[method] = np.where(chosen, 9.5, 0.)
    return result


def evaluate(predictions, truth):
    return [dict(method=m, signed_hit_gap=core.scores(p, truth)["hit_m"] - core.scores(p, truth)["hit_c"],
        **core.scores(p, truth)) for m, p in predictions.items()]


def run():
    started = time.time()
    out = HERE / "結果"
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(seed=SEED, methods=METHODS, null_repeats=REPEATS,
        selection="partial2 candidates12 + sex75 + one-to-one + 9.5/0",
        reverse_score="log(N_bg_sex/N_q_sex)+(dimension/2)*log((D_C_bg+1e-10)/(D_C_q+1e-10))",
        interpretation="uncalibrated source contrast; adaptive partial2 is not Euclidean dimension2",
        frozen_before_truth=True, is_official_phi=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "固定条件.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    saved, permutations, diagnostics = [], [], []
    for phase in ("prelim", "main"):
        folder = ROOT / ("予備戦/配布データ/PWSCup2026_prelim_attack" if phase == "prelim"
                         else "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力")
        bgpath = ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv" if phase == "prelim" else folder / "A_bg.csv"
        bg = core.read(bgpath)
        order = np.random.default_rng(SEED).permutation(len(bg))
        pool = bg.iloc[order[:len(bg)//2]].reset_index(drop=True)
        ref = bg.iloc[order[len(bg)//2:]].reset_index(drop=True)
        for cp in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            team = int(cp.stem.rsplit("_", 1)[1])
            q = core.read(folder / f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c = core.read(cp)
            pair, dc, rev = make_matrices(q, c, bg)
            pred = predict(pair, dc, rev, q.sex.to_numpy(), c.sex.to_numpy())
            saved.append((phase, team, "real_C_full", -1, pred))
            for dimension in (2, 5):
                diagnostics.append(dict(phase=phase, team=team, dimension=dimension,
                    row_score_q10=np.quantile(rev[dimension], .1), row_score_median=np.median(rev[dimension]),
                    row_score_q90=np.quantile(rev[dimension], .9)))
            for repeat in range(3):
                rng = np.random.default_rng(SEED + team*100 + repeat)
                a, b = rng.permutation(len(q)), rng.permutation(len(c))
                perm = predict(pair[a][:, b], dc[a][:, b], {d: r[b] for d, r in rev.items()},
                    q.sex.to_numpy()[a], c.sex.to_numpy()[b])
                for method in pred:
                    permutations.append(dict(phase=phase, team=team, method=method, repeat=repeat,
                        changed=int((pred[method] != perm[method][np.argsort(a)]).sum())))
            if phase == "prelim" and team not in (1, 2, 5):
                p, d, r = make_matrices(q, c, ref)
                saved.append((phase, team, "real_C_split", -1, predict(p, d, r, q.sex.to_numpy(), c.sex.to_numpy())))
                p, d, r = make_matrices(q, pool, ref)
                for repeat in range(REPEATS):
                    rng = np.random.default_rng(SEED + team*1000 + repeat)
                    indices = []
                    for sex, count in c.sex.value_counts().items():
                        available = np.flatnonzero(pool.sex.eq(sex).to_numpy())
                        indices.extend(rng.choice(available, int(count), replace=False))
                    indices = np.sort(indices)
                    fake = pool.iloc[indices]
                    predictions = predict(p[:, indices], d[:, indices], {dim: v[indices] for dim, v in r.items()},
                        q.sex.to_numpy(), fake.sex.to_numpy())
                    saved.append((phase, team, "background_C", repeat, predictions))
            print(json.dumps(dict(phase=phase, team=team, seconds=round(time.time()-started, 1))), flush=True)
    prediction_seconds = time.time() - started
    truthdir = next((ROOT / "予備戦/配布データ/結果").glob("正解*"))
    truths = {}
    records = []
    for phase, team, source, repeat, pred in saved:
        if phase != "prelim":
            continue
        if team not in truths:
            truths[team] = core.read(truthdir / f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        records.extend(dict(team=team, source=source, repeat=repeat, **r) for r in evaluate(pred, truths[team]))
    table = pd.DataFrame(records)
    table.to_csv(out / "対象別と背景反復.csv", index=False)
    real = table[table.source.eq("real_C_full")]
    previous = pd.read_csv(OLD / "集計/同点修正_対象別.csv")
    previous = previous[previous.method.eq("stable_matching/sex_quota/long75")].set_index("team")
    current = real[real.method.eq("baseline")].set_index("team")
    assert np.array_equal(current.loc[previous.index, ["hit_m", "hit_c"]], previous[["hit_m", "hit_c"]])
    summaries, loto = [], []
    for scope, excluded in (("20targets", [5]), ("18processed", [1, 2, 5])):
        subset = real[~real.team.isin(excluded)]
        base = subset[subset.method.eq("baseline")].set_index("team").r_all
        for method, group in subset.groupby("method"):
            delta = group.set_index("team").r_all - base
            summaries.append(dict(scope=scope, method=method, r_all=group.r_all.mean(),
                delta=delta.mean(), wins=int((delta > 1e-12).sum()), losses=int((delta < -1e-12).sum()),
                signed_hit_gap=group.signed_hit_gap.mean()))
        pivot = subset.pivot(index="team", columns="method", values="r_all")
        for team in pivot.index:
            means = pivot.drop(team).mean()
            best = means.idxmax()
            loto.append(dict(scope=scope, team=team, selected_method=best, r_all=pivot.loc[team, best]))
    pd.DataFrame(summaries).to_csv(out / "主比較.csv", index=False)
    pd.DataFrame(loto).to_csv(out / "対象留保選択.csv", index=False)
    table[~table.team.isin([1, 2, 5])].groupby(["source", "method"])[["r_all", "p_m", "p_c", "signed_hit_gap"]].mean().to_csv(out / "独立背景比較.csv")
    pd.DataFrame(permutations).to_csv(out / "行順変更.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "C行の対比量分布.csv", index=False)
    record = dict(protocol, prediction_seconds=prediction_seconds, total_seconds=time.time()-started,
        baseline_reproduced=True, row_permutation_cases=len(permutations),
        row_changed=sum(r["changed"] for r in permutations), input_sha256=core.HASHES,
        submission_created=False, main_truth_read=False)
    (out / "実行記録.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(pd.DataFrame(summaries).query("scope == '18processed'").to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
