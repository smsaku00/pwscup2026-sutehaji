"""固定9方式の行順不変性と独立背景対照を追加確認する。

新しい方式・係数・選別人数を追加しない。背景対照は公式phiではない。
予測を固定後に公開正解を読む。個票予測は共有用ファイルに保存しない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rank_methods_1008", HERE / "分析.py")
rank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rank)
core = rank.core
REPEATS = 20
PERMUTATIONS = 3
SEED = 2026100809


def score_saved(predictions, truth):
    records = []
    for method, prediction in predictions.items():
        metric = core.scores(prediction, truth)
        metric["signed_hit_gap"] = metric["hit_m"] - metric["hit_c"]
        records.append(dict(method=method, **metric))
    return records


def same_sex_sample(pool, c, rng):
    selected = []
    for sex, count in c.sex.value_counts().items():
        indices = np.flatnonzero(pool.sex.eq(sex).to_numpy())
        assert len(indices) >= count
        selected.extend(rng.choice(indices, int(count), replace=False))
    return pool.iloc[np.sort(selected)].reset_index(drop=True)


def run():
    started = time.time()
    out = HERE / "結果"
    protocol = dict(methods=rank.METHODS, seed=SEED, null_repeats=REPEATS,
        row_permutations=PERMUTATIONS, sex_counts_equal_actual_C=True,
        split="2571 sample pool / 2572 independent reference", is_official_phi=False,
        methods_unchanged_after_first_results=True,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method_code_sha256=hashlib.sha256((HERE / "分析.py").read_bytes()).hexdigest())
    (out / "追加監査_固定条件.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    permutation_rows, detail_rows = [], []
    pending = []
    # 44公開Cで予測を作る。正解はこのループ内で読まない。
    for phase in ("prelim", "main"):
        folder, bgpath = rank.setup(phase)
        bg = core.read(bgpath)
        order = np.random.default_rng(SEED).permutation(len(bg))
        pool = bg.iloc[order[:len(bg)//2]].reset_index(drop=True)
        reference = bg.iloc[order[len(bg)//2:]].reset_index(drop=True)
        for cp in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            team = int(cp.stem.rsplit("_", 1)[1])
            q = core.read(folder / f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c = core.read(cp)
            qs, cs, bs = (f.sex.to_numpy() for f in (q, c, bg))
            cached = np.load(HERE / "作業用" / f"{phase}_{team}.npz")
            for method, kind, dimension, blend in rank.METHODS:
                (x, y, z), _ = rank.arrays(q, c, bg, kind, blend)
                dc = rank.distance(x, y, qs, cs, dimension)
                db = rank.distance(x, z, qs, bs, dimension)
                pair = rank.pair_score(dc, db)["continuous"]
                pair -= 1e-10 * np.minimum(rank.cdist(x, y, "sqeuclidean") / 5, 1000.)
                selected = rank.select(pair, dc, qs, cs, "sex_quota")
                assert np.array_equal(np.where(selected, 9.5, 0.), cached[method])
                for repeat in range(PERMUTATIONS):
                    rng = np.random.default_rng(SEED + team * 100 + repeat)
                    a, b = rng.permutation(len(q)), rng.permutation(len(c))
                    permuted = rank.select(pair[a][:, b], dc[a][:, b], qs[a], cs[b], "sex_quota")[np.argsort(a)]
                    permutation_rows.append(dict(phase=phase, team=team, method=method, repeat=repeat,
                        selected_changed=int((selected != permuted).sum())))
                # 200組は5次元の差を一つずつ計算し、同じC行の距離であることを確認。
                rng = np.random.default_rng(SEED + team)
                for i, j in zip(rng.integers(len(q), size=200), rng.integers(len(c), size=200)):
                    manual = float(np.mean(sorted((x[i] - y[j]) ** 2)[:dimension])) if qs[i] == cs[j] else np.inf
                    assert (np.isinf(manual) and np.isinf(dc[i, j])) or abs(manual - dc[i, j]) < 1e-12
            if phase == "prelim" and team not in (1, 2, 5):
                real, _ = rank.predictions(q, c, reference)
                nulls = []
                for repeat in range(REPEATS):
                    rng = np.random.default_rng(SEED + team * 1000 + repeat)
                    fake = same_sex_sample(pool, c, rng)
                    null, _ = rank.predictions(q, fake, reference)
                    nulls.append(null)
                pending.append((team, real, nulls))
            print(json.dumps(dict(phase=phase, team=team, seconds=round(time.time()-started, 1))), flush=True)
    prediction_finished = time.time()
    truthdir = next((rank.ROOT / "予備戦/配布データ/結果").glob("正解*"))
    for team, real, nulls in pending:
        truth = core.read(truthdir / f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        detail_rows.extend(dict(team=team, source="real_C_split", repeat=-1, **r) for r in score_saved(real, truth))
        for repeat, null in enumerate(nulls):
            detail_rows.extend(dict(team=team, source="background_C", repeat=repeat, **r) for r in score_saved(null, truth))
    perm = pd.DataFrame(permutation_rows)
    perm.to_csv(out / "行順変更.csv", index=False)
    data = pd.DataFrame(detail_rows)
    data.to_csv(out / "独立背景対照_全反復.csv", index=False)
    aggregate = []
    for method, group in data.groupby("method"):
        real = group[group.source.eq("real_C_split")]
        null = group[group.source.eq("background_C")]
        aggregate.append(dict(method=method, teams=len(real), real_R=real.r_all.mean(),
            null_R=null.r_all.mean(), excess_R=real.r_all.mean()-null.r_all.mean(),
            real_signed_hit_gap=real.signed_hit_gap.mean(), null_signed_hit_gap=null.signed_hit_gap.mean(),
            real_p_m=real.p_m.mean(), real_p_c=real.p_c.mean()))
    pd.DataFrame(aggregate).to_csv(out / "独立背景対照_18対象平均.csv", index=False)
    record = dict(protocol, prediction_seconds=prediction_finished-started, total_seconds=time.time()-started,
        row_permutation_cases=len(perm), changed=int(perm.selected_changed.sum()),
        distance_cases=44 * len(rank.METHODS) * 200,
        input_sha256=core.HASHES, executable=sys.executable,
        submission_created=False, main_truth_read=False)
    (out / "追加監査_実行記録.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(pd.DataFrame(aggregate).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
