"""保存済みの固定9予測を、公開正解で選別の変更に分解する。

新しい方式や閾値を作らない。個人のID・正解を共有集計に残さない。
"""
from __future__ import annotations
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rank_methods_1008_decomposition", HERE / "分析.py")
rank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rank)
core = rank.core


def run():
    folder, _ = rank.setup("prelim")
    truthdir = next((rank.ROOT / "予備戦/配布データ/結果").glob("正解*"))
    rows, checks, hashes = [], [], {}
    saved_scores = pd.read_csv(HERE / "結果/対象別.csv").set_index(["team", "method"])
    for cp in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        team = int(cp.stem.rsplit("_", 1)[1])
        q = core.read(folder / f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        truth = core.read(truthdir / f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        assert q.challenge_row_id.tolist() == truth.challenge_row_id.tolist()
        file = HERE / "作業用" / f"prelim_{team}.npz"
        hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        predictions = np.load(file)
        baseline = predictions["baseline_partial2"] > 0
        actual_time = truth.time.to_numpy()
        bins = {"long_9_to_10": actual_time >= 9., "early_0_to_half": actual_time <= .5,
                "middle_half_to_9": (actual_time > .5) & (actual_time < 9.)}
        for method, *_ in rank.METHODS:
            selected = predictions[method] > 0
            member = truth.label.eq("member").to_numpy()
            long = bins["long_9_to_10"]
            early = bins["early_0_to_half"]
            hit_by_branches = (selected & long) | (~selected & early)
            hit_direct = np.abs(predictions[method] - actual_time) <= .5
            assert np.array_equal(hit_by_branches, hit_direct)
            known = saved_scores.loc[(team, method)]
            assert hit_direct[member].sum() == known.hit_m
            assert hit_direct[~member].sum() == known.hit_c
            checks.append(dict(team=team, method=method, formula_and_saved_scores_equal=True))
            for label in ("member", "control"):
                for sex in (0, 1):
                    for time_type, time_mask in bins.items():
                        mask = truth.label.eq(label).to_numpy() & q.sex.eq(sex).to_numpy() & time_mask
                        base_hit = (baseline & long) | (~baseline & early)
                        rows.append(dict(team=team, method=method, label=label, sex=sex, time_type=time_type,
                            n=int(mask.sum()), base_selected=int((mask & baseline).sum()),
                            new_selected=int((mask & selected).sum()),
                            newly_selected=int((mask & ~baseline & selected).sum()),
                            newly_unselected=int((mask & baseline & ~selected).sum()),
                            delta_hits=int(hit_direct[mask].sum()-base_hit[mask].sum())))
    data = pd.DataFrame(rows)
    data.to_csv(HERE / "結果/選別変更_対象性別時間帯.csv", index=False)
    for scope, excluded in (("20対象", [5]), ("加工18対象", [1, 2, 5])):
        d = data[~data.team.isin(excluded)]
        n_teams = d.team.nunique()
        totals = d.groupby(["method", "label", "time_type"])[[
            "n", "base_selected", "new_selected", "newly_selected", "newly_unselected", "delta_hits"]].sum() / n_teams
        totals.to_csv(HERE / f"結果/選別変更_{scope}平均.csv")
    record = dict(predictions_changed=False, parameters_added=False, cases=len(checks),
        formula="hit = selected & time>=9 or unselected & time<=0.5",
        individual_truth_exported=False, checks=checks, input_sha256=core.HASHES,
        prediction_cache_sha256=hashes, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / "結果/選別変更_照合記録.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(cases=len(checks), passed=True)), flush=True)


if __name__ == "__main__":
    run()
