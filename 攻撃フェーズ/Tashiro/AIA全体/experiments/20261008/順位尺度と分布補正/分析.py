"""Cの順位尺度・周辺分布補正だけを変え、固定75%選別を比較する。

推定関数に正解を渡さない。全フェーズの予測が終わってから公開正解で採点する。
提出F/ZIPは作成しない。保存予測は作業用キャッシュで、共有対象ではない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtri
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = ROOT / "本戦/攻撃フェーズ/Tashiro/20261007_AIA全体向上の分析"
sys.path.insert(0, str(OLD))
import 分析 as core
from 対応競合の頑健性 import pair_score
from 同点に依存しない対応 import select
SEED = 20261008
METHODS = (
    ("baseline_partial2", "raw", 2, 0.),
    ("rank_global_partial2", "rank_global", 2, 0.),
    ("rank_sex_partial2", "rank_sex", 2, 0.),
    ("rank_global_full5", "rank_global", 5, 0.),
    ("rank_sex_full5", "rank_sex", 5, 0.),
    ("map_sex_half", "map_sex", 2, .5),
    ("map_sex_full", "map_sex", 2, 1.),
    ("map_sex_pref_half", "map_sex_pref", 2, .5),
    ("map_sex_pref_full", "map_sex_pref", 2, 1.),
)


def empirical_cdf(reference, values):
    u, counts = np.unique(reference, return_counts=True)
    # 同じ値のCDFは中点。外側は有限の確率に固定し、無限の正規順位を避ける。
    p = (np.cumsum(counts) - .5 * counts + .25) / (len(reference) + .5)
    return np.interp(values, u, p, left=.25 / (len(reference) + .5),
                     right=1 - .25 / (len(reference) + .5))


def gaussian_rank(frames, bg, by_sex):
    result = []
    br = core.raw(bg)
    for frame in frames:
        raw = core.raw(frame)
        out = np.empty_like(raw)
        groups = (0, 1) if by_sex else (None,)
        for sex in groups:
            rows = np.ones(len(frame), bool) if sex is None else frame.sex.eq(sex).to_numpy()
            ref = br if sex is None else br[bg.sex.eq(sex).to_numpy()]
            for j in range(raw.shape[1]):
                out[rows, j] = ndtri(empirical_cdf(ref[:, j], raw[rows, j]))
        result.append(out)
    return result


def marginal_map(c, bg, by_pref):
    original = core.raw(c)
    target = core.raw(bg)
    mapped = original.copy()
    columns = ["sex", "prefecture"] if by_pref else ["sex"]
    fallback_groups = 0
    for key, indices in c.groupby(columns, sort=True).indices.items():
        key = key if isinstance(key, tuple) else (key,)
        src = np.ones(len(c), bool)
        ref = np.ones(len(bg), bool)
        for col, value in zip(columns, key):
            src &= c[col].eq(value).to_numpy()
            ref &= bg[col].eq(value).to_numpy()
        if src.sum() < 20 or ref.sum() < 40:
            # 小群の経験分布を無理に推定しない。性別だけへ戻す。
            src = c.sex.eq(key[0]).to_numpy()
            ref = bg.sex.eq(key[0]).to_numpy()
            fallback_groups += 1
        for j in range(original.shape[1]):
            probability = empirical_cdf(original[src, j], original[indices, j])
            mapped[indices, j] = np.quantile(target[ref, j], probability)
    return mapped, fallback_groups


def arrays(q, c, bg, kind, blend):
    if kind.startswith("rank"):
        out = gaussian_rank([q, c, bg], bg, kind == "rank_sex")
        fallback = 0
    else:
        x, y, z = (core.raw(f) for f in (q, c, bg))
        fallback = 0
        if kind.startswith("map"):
            moved, fallback = marginal_map(c, bg, kind == "map_sex_pref")
            y = (1 - blend) * y + blend * moved
        out = [x, y, z]
    scale = np.maximum(np.quantile(out[2], .75, axis=0) - np.quantile(out[2], .25, axis=0), 1e-5)
    return [v / scale for v in out], fallback


def distance(x, y, xs, ys, dimension):
    d = np.sort((x[:, None] - y[None]) ** 2, axis=2)[:, :, :dimension].mean(axis=2)
    d[xs[:, None] != ys[None]] = np.inf
    return d


def predictions(q, c, bg, methods=METHODS):
    qs, cs, bs = (f.sex.to_numpy() for f in (q, c, bg))
    result, diagnostics = {}, []
    cache = {}
    for name, kind, dimension, blend in methods:
        key = (kind, blend)
        if key not in cache:
            cache[key] = arrays(q, c, bg, kind, blend)
        (x, y, z), fallback = cache[key]
        dc = distance(x, y, qs, cs, dimension)
        db = distance(x, z, qs, bs, dimension)
        pair = pair_score(dc, db)["continuous"]
        full = cdist(x, y, "sqeuclidean") / 5
        pair -= 1e-10 * np.minimum(full, 1000.)
        chosen = select(pair, dc, qs, cs, "sex_quota")
        result[name] = np.where(chosen, 9.5, 0.)
        diagnostics.append(dict(method=name, selected=int(chosen.sum()), fallback_groups=fallback,
            max_abs_coordinate=float(max(np.abs(x).max(), np.abs(y).max()))))
    return result, diagnostics


def setup(phase):
    if phase == "prelim":
        folder = ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack"
        bgpath = ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"
    else:
        folder = ROOT / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
        bgpath = folder / "A_bg.csv"
    return folder, bgpath


def run():
    start = time.time()
    (HERE / "結果").mkdir(parents=True, exist_ok=True)
    (HERE / "作業用").mkdir(parents=True, exist_ok=True)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = dict(seed=SEED, methods=METHODS, selection="fixed sex75 and long9.5", frozen_before_truth=True,
        gaussian_cdf="midrank piecewise linear", min_C_group=20, min_BG_group=40, code_sha256=code_hash)
    (HERE / "結果/固定比較条件.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    cached, diagnostics = [], []
    for phase in ("prelim", "main"):
        folder, bgpath = setup(phase)
        bg = core.read(bgpath)
        for f in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            k = int(f.stem.rsplit("_", 1)[1])
            q = core.read(folder / f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c = core.read(f)
            pred, diag = predictions(q, c, bg)
            cached.append((phase, k, q, pred))
            diagnostics += [dict(phase=phase, team=k, **r) for r in diag]
            np.savez_compressed(HERE / "作業用" / f"{phase}_{k}.npz", **pred)
            print(json.dumps(dict(phase=phase, team=k, elapsed=round(time.time()-start, 1))), flush=True)
    prediction_finished = time.time()
    # 正解を読む段階を予測完了の後へ分ける。
    truthdir = next((ROOT / "予備戦/配布データ/結果").glob("正解*"))
    ownpath = next((ROOT / "本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv"))
    own = core.read(ownpath)
    rows, ownrows = [], []
    for phase, k, q, pred in cached:
        if phase == "prelim":
            truth = core.read(truthdir / f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            assert q.challenge_row_id.tolist() == truth.challenge_row_id.tolist()
            rows += [dict(team=k, method=m, **core.scores(v, truth)) for m, v in pred.items()]
        elif k == 28:
            joined = q.merge(own[core.QI + ["time", "is_rare"]], on=core.QI, how="left", validate="one_to_one")
            member = joined.time.notna().to_numpy()
            assert member.sum() == 163
            for m, v in pred.items():
                hit = np.abs(v - joined.time.to_numpy(float)) <= .5
                ownrows.append(dict(method=m, selected_member=int(((v > 0) & member).sum()),
                    selected_control=int(((v > 0) & ~member).sum()), hit_member=int(hit[member].sum()),
                    p_m=float(hit[member].mean()), p_c_unknown=True))
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "結果/対象別.csv", index=False)
    old = pd.read_csv(OLD / "集計/同点修正_対象別.csv")
    base = table[table.method.eq("baseline_partial2")].set_index("team")
    prior = old[old.method.eq("stable_matching/sex_quota/long75")].set_index("team")
    assert base.index.sort_values().equals(prior.index.sort_values())
    assert np.allclose(base.loc[prior.index, "r_all"], prior.r_all, atol=1e-12, rtol=0)
    summary = []
    for label, excluded in (("20targets", [5]), ("18processed", [1, 2, 5])):
        s = table[~table.team.isin(excluded)].copy()
        b = s[s.method.eq("baseline_partial2")].set_index("team").r_all
        for method, g in s.groupby("method"):
            delta = g.set_index("team").r_all - b
            summary.append(dict(scope=label, method=method, targets=len(g), r_all=g.r_all.mean(),
                p_m=g.p_m.mean(), p_c=g.p_c.mean(), delta_vs_baseline=delta.mean(),
                wins=int((delta > 1e-12).sum()), losses=int((delta < -1e-12).sum())))
    pd.DataFrame(summary).to_csv(HERE / "結果/平均比較.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(HERE / "結果/動作診断.csv", index=False)
    pd.DataFrame(ownrows).to_csv(HERE / "結果/自チーム診断.csv", index=False)
    record = dict(protocol, python=platform.python_version(), executable=sys.executable,
        input_sha256=core.HASHES, prediction_finished_seconds=prediction_finished-start,
        total_seconds=time.time()-start, old_baseline_reproduced=True, submission_created=False,
        codabench_submitted=False, own_truth_used_to_choose_method=False)
    (HERE / "結果/実行記録.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(pd.DataFrame(summary).query("scope == '18processed'").to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
