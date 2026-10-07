"""距離・選別を固定し、条件付きtimeの追加的な寄与を切り分ける。

原始候補の改良探索ではない。Cのtimeを性別内で置換する20反復は独自の対照で
公式phiではない。予測に正解は渡さず、全予測後に予備戦truthを読む。
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import 分析 as core

HERE = Path(__file__).resolve().parent
ROOT = core.ROOT
REPEATS = 20
METHODS = ["all_window", "gate75_window", "gate75_long95"]


def estimate(d, times):
    return np.array([core.best_window(*v[:2])[0] for v in core.local_distribution(d, times, 8)])


def run():
    start = time.time()
    plan = dict(repeats=REPEATS, C_distance="box_full5_sex", k=8, gate=.75,
                null="C.time shuffled within each sex; QI and marginal time unchanged",
                methods=METHODS, is_official_phi=False, configuration_search=False)
    core.dump(HERE / "寄与検証_固定手順.json", plan)
    folder = ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack"
    bg = core.core.read(ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
    items = []
    for cp in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        team = int(cp.stem.rsplit("_", 1)[1])
        if team == 5:
            continue
        c = core.core.read(cp)
        q = core.core.read(folder / f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        width, _ = core.rounding_widths(c)
        dc, db = (core.distance(q, v, bg, width) for v in (c, bg))
        score = core.density_score(dc, db)
        selected = core.sex_keep(score, q)
        actual = estimate(dc, c.time.to_numpy())
        pred = [dict(source="real_C", repeat=-1, method="all_window", prediction=actual),
                dict(source="real_C", repeat=-1, method="gate75_window", prediction=np.where(selected, actual, 0)),
                dict(source="fixed95", repeat=-1, method="gate75_long95", prediction=np.where(selected, 9.5, 0))]
        for repeat in range(REPEATS):
            rng = np.random.default_rng(core.SEED + 5000 + team*100 + repeat)
            times = c.time.to_numpy().copy()
            for sex in np.unique(c.sex):
                m = c.sex.eq(sex).to_numpy()
                times[m] = times[m][rng.permutation(m.sum())]
            moved = estimate(dc, times)
            pred += [dict(source="time_shuffle", repeat=repeat, method="all_window", prediction=moved),
                     dict(source="time_shuffle", repeat=repeat, method="gate75_window", prediction=np.where(selected, moved, 0))]
        items.append(dict(team=team, q=q, predictions=pred))
        print(json.dumps(dict(team=team, seconds=round(time.time()-start, 1))), flush=True)
    td = next((ROOT / "予備戦/配布データ/結果").glob("正解*"))
    rows, comparisons = [], []
    for item in items:
        team = item["team"]
        truth = core.core.read(td / f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        assert item["q"].challenge_row_id.tolist() == truth.challenge_row_id.tolist()
        for pred in item["predictions"]:
            rows.append(dict(team=team, source=pred["source"], repeat=pred["repeat"], method=pred["method"],
                             **core.core.scores(pred["prediction"], truth)))
    out = HERE / "集計"
    table = pd.DataFrame(rows)
    table.to_csv(out / "時間関係の寄与_全反復.csv", index=False)
    summaries = []
    for label, excluded in [("20対象_自5除外", [5]), ("18対象_1_2_5除外", [1, 2, 5])]:
        d = table[~table.team.isin(excluded)]
        s = d.groupby(["source", "method"])[["r_all", "r_rare", "r_max", "p_m", "p_c"]].mean().reset_index()
        s.insert(0, "group", label)
        summaries.append(s)
    pd.concat(summaries).to_csv(out / "時間関係の寄与_平均.csv", index=False)
    for (team, method), d in table[table.method.isin(["all_window", "gate75_window"])].groupby(["team", "method"]):
        actual = d[d.source.eq("real_C")].iloc[0]
        null = d[d.source.eq("time_shuffle")]
        comparisons.append(dict(team=team, method=method, actual_R=actual.r_all,
            shuffled_mean_R=null.r_all.mean(), delta_R=actual.r_all-null.r_all.mean(),
            shuffled_R_q05=null.r_all.quantile(.05), shuffled_R_q95=null.r_all.quantile(.95),
            delta_pm=actual.p_m-null.p_m.mean(), delta_pc=actual.p_c-null.p_c.mean()))
    pd.DataFrame(comparisons).to_csv(out / "時間関係の寄与_対象別.csv", index=False)
    # 6候補で集計済みの予測とここで再計算した予測は同じになるはず。
    prior = pd.read_csv(out / "予備戦_対象別.csv")
    reference = prior[prior.method.eq("C/box_k8_window_gate75") & prior.team.ne(5)].sort_values("team")
    calculated = table[table.source.eq("real_C") & table.method.eq("gate75_window")].sort_values("team")
    assert reference.team.tolist() == calculated.team.tolist()
    assert np.max(np.abs(reference.r_all.to_numpy()-calculated.r_all.to_numpy())) < 1e-12
    core.dump(HERE / "寄与検証_実行記録.json", dict(plan=plan, seconds=time.time()-start,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prediction_code_sha256=hashlib.sha256((HERE / "分析.py").read_bytes()).hexdigest(),
        input_sha256=core.core.HASHES, preserved_original_predictions=True,
        phi_known=False, main_truth_used=False, submission_created=False, codabench_submitted=False))
    print(pd.concat(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
