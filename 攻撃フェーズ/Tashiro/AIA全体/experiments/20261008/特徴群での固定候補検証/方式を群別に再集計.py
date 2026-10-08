"""既存の固定54候補を共通全BGの特徴群ごとに再集計する。再学習しない。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P = HERE.parent
HASHES = {}


def read(relative):
    path = P / relative
    HASHES[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def run():
    pool = read("候補群の留保監査/共通全BG_全候補対象別.csv")
    keep = ["team", "candidate_id", "r_all", "hit_m", "hit_c"]
    pool = pool[pool.eligible_primary.eq(True) & pool.team.ne(5)][keep]
    assert pool.candidate_id.nunique() == 39
    pieces = [pool]
    prior = read("加工特徴と方式選択/集計/固定方式の予備戦得点.csv")
    prior = prior[prior.team.ne(5) & prior.family.ne("partial2_matching")].copy()
    prior["candidate_id"] = "previous:" + prior.family
    assert prior.candidate_id.nunique() == 7
    pieces.append(prior[keep])
    files = [("臨床値集合の複数近傍/集計/予備戦_対象別と20反復null.csv", "multiple:", "column_k"),
             ("臨床周辺分布の平滑化/集計/予備戦_固定5候補と20null対象別.csv", "kde:", "kde_"),
             ("臨床近傍の背景質量較正/集計/予備戦_固定5候補と20null対象別.csv", "mass:", "mass_")]
    base = pool[pool.candidate_id.eq("anchor:P01")].set_index("team").sort_index()
    for file, prefix, selected in files:
        table = read(file)
        actual = table[table.context.eq("full_reference_overlap") & table.source.eq("real_C") & table.team.ne(5)]
        old = actual[actual.method.eq("baseline_partial2_matching")].set_index("team").sort_index()
        assert old.index.tolist() == base.index.tolist()
        assert np.max(np.abs(old.r_all - base.r_all)) < 1e-12
        table = actual[actual.method.str.startswith(selected)].copy()
        table["candidate_id"] = prefix + table.method
        pieces.append(table[keep])
    combined = pd.concat(pieces, ignore_index=True)
    assert combined.candidate_id.nunique() == 54
    assert len(combined) == 54 * 20 and not combined.duplicated(["team", "candidate_id"]).any()
    combined = combined.merge(base[["r_all", "hit_m", "hit_c"]].add_prefix("P01_"), left_on="team", right_index=True, validate="many_to_one")
    combined["R_delta"] = combined.r_all - combined.P01_r_all
    combined["hit_gap_delta"] = (combined.hit_m - combined.hit_c) - (combined.P01_hit_m - combined.P01_hit_c)
    definitions = {"prior_strong_record_1_2": [1, 2], "column_and_row_BG_preserved": [18, 29],
                   "column_BG_preserved_row_partial": [19, 24], "prior_low_unique": [23]}
    definitions["other_processed"] = [int(v) for v in base.index if v not in sum(definitions.values(), [])]
    means = []
    for group, teams in definitions.items():
        for candidate, table in combined[combined.team.isin(teams)].groupby("candidate_id", sort=False):
            means.append(dict(group=group, teams=" ".join(map(str, teams)), target_count=len(teams), candidate_id=candidate,
                 R_mean=float(table.r_all.mean()), P01_R_mean=float(table.P01_r_all.mean()), R_delta_mean=float(table.R_delta.mean()),
                 hit_gap_delta_mean=float(table.hit_gap_delta.mean()), wins=int(table.R_delta.gt(1e-12).sum()),
                 losses=int(table.R_delta.lt(-1e-12).sum()), ties=int(table.R_delta.abs().le(1e-12).sum())))
    pd.DataFrame(means).to_csv(HERE / "集計/既存54候補_共通全BG群別.csv", index=False)
    (HERE / "再集計記録.json").write_text(json.dumps(dict(source_sha256=HASHES, nominal_candidates=54, source_context="same full phase BG",
        source_baselines_match=True, scored_target_count=20, groups=definitions, group_review_is_posthoc=True,
        candidates_not_selected_for_submission=True, independent_new_targets=0), ensure_ascii=False, indent=2) + "\n")
    print(dict(candidates=54, cohorts=len(definitions), summaries=len(means), baseline_matches=True))


if __name__ == "__main__":
    run()
