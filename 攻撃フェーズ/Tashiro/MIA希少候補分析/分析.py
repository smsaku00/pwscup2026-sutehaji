"""本戦の希少候補11行×22チームを、固定した3距離で診断する。

自チームAIA標的の希少13行を既知非会員として併記する。
会員確率・所属の断定・提出ファイルは生成しない。
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import platform
import sys
from importlib.metadata import version

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
from 共通設定 import configure, checked_paths
OPTIONS = None
OUT = None
NUM = ["age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG"]
QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
CCOLS = ["age", "sex", "prefecture", "BMI", "SBP", "TG", "HDL", "ALT", "smoking", "FPG", "time", "onset", "death"]
METHODS = {
    "numeric7_L2": {"metric": "sqeuclidean", "categories": [], "denominator": 7},
    "mixed9_L2": {"metric": "sqeuclidean", "categories": ["sex", "smoking"], "denominator": 9},
    "mixed10_L1": {"metric": "cityblock", "categories": ["sex", "smoking", "prefecture"], "denominator": 10},
}
PRIMARY = "mixed9_L2"
EPS = 1e-12


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values(df):
    x = df[NUM].to_numpy(float).copy()
    for col in ["TG", "ALT"]:
        x[:, NUM.index(col)] = np.log1p(x[:, NUM.index(col)])
    return x


def distances(a, b, method, scale):
    cfg = METHODS[method]
    d = cdist(values(a)/scale, values(b)/scale, cfg["metric"])
    for col in cfg["categories"]:
        d += a[col].to_numpy()[:, None] != b[col].to_numpy()[None, :]
    d /= cfg["denominator"]
    return np.sqrt(d) if cfg["metric"] == "sqeuclidean" else d


def key(df, cols):
    return list(df[cols].itertuples(index=False, name=None))


def rank(series, ascending):
    # 同値の優劣をCSV行順で決めない。
    return series.rank(method="min", ascending=ascending).astype(int)


def join_ids(series):
    return ",".join(str(int(v)) for v in series)


def load_inputs():
    paths, manifest = checked_paths(OPTIONS)
    b = read(paths["B_self.csv"])
    assert len(b) == 1049 and set(b.columns) == set(CCOLS + ["record_id", "is_rare"])
    assert b.record_id.is_unique and not b.duplicated(QI).any()
    q = read(paths["aia_challenge_28.csv"])
    assert not q.duplicated(QI).any()
    match = b.merge(q, on=QI, how="inner", validate="one_to_one")
    assert len(match) == 163 and int(match.onset.sum()) == 16
    r = b[b.is_rare == 1].copy()
    private_ids = set(match.loc[match.is_rare == 1, "record_id"])
    assert len(r) == 24 and len(private_ids) == 13
    r["区分"] = np.where(r.record_id.isin(private_ids), "確定非会員_AIA標的", "共有候補_未確定")
    map_challenge = match.set_index("record_id").challenge_row_id
    r["自課題ID"] = r.record_id.map(map_challenge).astype("Int64")
    candidates = r.loc[~r.record_id.isin(private_ids), "record_id"].tolist()
    assert candidates == [47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675]
    teams = sorted(int(t) for t in json.loads(paths["mia_columns.json"].read_text())["target_columns"] if int(t) != 28)
    assert len(teams) == 22
    bg = read(paths["A_bg.csv"])
    cs = {t: read(paths[f"C_{t}.csv"]) for t in teams}
    hashes = {name: sha(p) for name, p in paths.items()}
    return b, r, bg, cs, teams, candidates, private_ids, hashes, manifest["source_commit"]



def main():
    global OPTIONS, OUT
    OPTIONS = configure("希少候補の距離・固有性を再計算します")
    OUT = OPTIONS.output_dir / "集計"
    b, rare, bg, cs, teams, candidates, private_ids, hashes, commit = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    rawbg = values(bg)
    scale = (np.quantile(rawbg, .75, axis=0)-np.quantile(rawbg, .25, axis=0))/1.349
    assert np.all(scale > 0)
    rare_pos = b.index[b.is_rare == 1].to_numpy()
    b_ids = b.record_id.to_numpy()
    rare_ids = rare.record_id.to_numpy()
    rows, counts = [], []
    # 背景との距離は同じ候補について全チーム共通で、一度だけ計算。
    qbg = {method: distances(rare, bg, method, scale).min(axis=1) for method in METHODS}
    for team in teams:
        c = cs[team]
        exact = {}
        for label, cols in [("13", CCOLS), ("num7", NUM), ("mixed9", NUM+["sex", "smoking"])]:
            lookup = Counter(key(c, cols))
            exact[label] = [lookup[k] for k in key(rare, cols)]
        for method, cfg in METHODS.items():
            d = distances(b, c, method, scale)
            dr = d[rare_pos]
            nearest_indices = np.argsort(dr, axis=1, kind="stable")[:, :5]
            nearest_values = np.take_along_axis(dr, nearest_indices, axis=1)
            nearest_all_b = d.min(axis=1)
            idx = nearest_indices[:, 0]
            chosen = c.iloc[idx].reset_index(drop=True)
            c_bg = distances(chosen, bg, method, scale)
            for row_number, (pos, record_id) in enumerate(zip(rare_pos, rare_ids)):
                p = b.iloc[pos]
                cpos = int(idx[row_number])
                cc = c.iloc[cpos]
                d1, d2 = nearest_values[row_number, :2]
                inverse = d[:, cpos].copy()
                inverse[pos] = np.inf
                other_pos = int(np.argmin(inverse))
                other_d = float(inverse[other_pos])
                spec = float(np.log((other_d+EPS)/(d1+EPS)))
                numeric_difference = (values(b.iloc[[pos]])[0]-values(c.iloc[[cpos]])[0])/scale
                contribution = np.abs(numeric_difference) if cfg["metric"] == "cityblock" else numeric_difference**2
                labels = NUM + cfg["categories"]
                contribution = np.r_[contribution, [float(p[col] != cc[col]) for col in cfg["categories"]]]
                total = contribution.sum()
                order = np.argsort(-contribution, kind="stable")
                blocks = (c.sex == p.sex) & (c.prefecture == p.prefecture)
                strict_d = float(dr[row_number, blocks].min()) if blocks.any() else np.nan
                rows.append({
                    "team": team, "record_id": int(record_id), "role": "known_nonmember" if int(record_id) in private_ids else "candidate",
                    "method": method, "d1": d1, "d2": d2, "d5": nearest_values[row_number, 4],
                    "relative_gap_d1_d2": (d2-d1)/(d2+EPS),
                    "nearest_C_row0": cpos, "nearest_C_csv_line": cpos+2,
                    "nearest_tie_count": int(np.isclose(dr[row_number], d1, atol=EPS, rtol=0).sum()),
                    "C_to_B_rank_min": int((d[:, cpos] < d1-EPS).sum()+1),
                    "C_to_B_distance_ties": int(np.isclose(d[:, cpos], d1, atol=EPS, rtol=0).sum()),
                    "nearest_competitor_B_id": int(b_ids[other_pos]), "nearest_competitor_distance": other_d,
                    "specificity": spec,
                    "B_rows_selecting_this_C_including_ties": int((d[:, cpos] <= nearest_all_b+EPS).sum()),
                    "A_bg_rows_closer_to_this_C": int((c_bg[row_number] < d1-EPS).sum()),
                    "query_to_A_bg_distance": qbg[method][row_number],
                    "log_background_distance_ratio": float(np.log((qbg[method][row_number]+EPS)/(d1+EPS))),
                    "C_same_sex_pref_count": int(blocks.sum()), "same_sex_pref_d1": strict_d,
                    "sex_match": bool(p.sex == cc.sex), "prefecture_match": bool(p.prefecture == cc.prefecture),
                    "smoking_match": bool(p.smoking == cc.smoking),
                    "C_nearest_prefecture": cc.prefecture, "C_nearest_sex": int(cc.sex),
                    "C_nearest_time": cc.time, "abs_time_difference": abs(p.time-cc.time),
                    "onset_match": bool(p.onset == cc.onset), "death_match": bool(p.death == cc.death),
                    "largest_distance_component": labels[order[0]], "largest_component_share": contribution[order[0]]/total if total > 0 else 0,
                    "second_distance_component": labels[order[1]],
                    "exact13_C_rows": exact["13"][row_number], "exact_num7_C_rows": exact["num7"][row_number],
                    "exact_mixed9_C_rows": exact["mixed9"][row_number],
                })
        print(f"team {team}: 24 records x 3 settings", flush=True)
    detail = pd.DataFrame(rows)
    for (team, method), ix in detail.groupby(["team", "method"]).groups.items():
        group = detail.loc[ix]
        neg = group[group.role == "known_nonmember"]
        cand = group[group.role == "candidate"]
        # 13既知非会員をチューニングには使わず、同じ定義の診断値を付ける。
        for i, row in group.iterrows():
            detail.loc[i, "known13_distance_le_count"] = int((neg.d1 <= row.d1+EPS).sum())
            detail.loc[i, "known13_specificity_ge_count"] = int((neg.specificity >= row.specificity-EPS).sum())
            detail.loc[i, "known13_both_at_least_as_strong_count"] = int(((neg.d1 <= row.d1+EPS)&(neg.specificity >= row.specificity-EPS)).sum())
        detail.loc[cand.index, "candidate_distance_rank"] = rank(cand.d1, True)
        detail.loc[cand.index, "candidate_specificity_rank"] = rank(cand.specificity, False)
        counts.append({"team": team, "method": method, "known_nonmember_n": 13,
                       "known_nonmember_mutual_nearest": int((neg.C_to_B_rank_min == 1).sum()),
                       "candidate_mutual_nearest": int((cand.C_to_B_rank_min == 1).sum()),
                       "known_nonmember_d1_median": neg.d1.median(), "candidate_d1_median": cand.d1.median(),
                       "known_nonmember_specificity_max": neg.specificity.max(),
                       "candidate_specificity_max": cand.specificity.max(),
                       "known_nonmember_closer_than_best_candidate": int((neg.d1 <= cand.d1.min()+EPS).sum()),
                       "known_nonmember_more_specific_than_best_candidate": int((neg.specificity >= cand.specificity.max()-EPS).sum())})
    for (_, method), ix in detail[detail.role == "candidate"].groupby(["record_id", "method"]).groups.items():
        group = detail.loc[ix]
        detail.loc[ix, "team_distance_rank_for_record"] = rank(group.d1, True)
        detail.loc[ix, "team_specificity_rank_for_record"] = rank(group.specificity, False)
    pairs = []
    for (record_id, team), group in detail[detail.role == "candidate"].groupby(["record_id", "team"], sort=True):
        first = group[group.method == PRIMARY].iloc[0]
        item = {"record_id": int(record_id), "team": int(team), "membership": "未確定",
                "primary_d1": first.d1, "primary_specificity": first.specificity,
                "primary_C_to_B_rank_min": int(first.C_to_B_rank_min),
                "primary_C_row0": int(first.nearest_C_row0),
                "primary_known13_distance_le": int(first.known13_distance_le_count),
                "primary_known13_specificity_ge": int(first.known13_specificity_ge_count),
                "primary_known13_both_ge": int(first.known13_both_at_least_as_strong_count),
                "primary_same_sex_pref_count": int(first.C_same_sex_pref_count),
                "primary_same_sex_pref_d1": first.same_sex_pref_d1,
                "primary_prefecture_match": bool(first.prefecture_match),
                "exact13_C_rows": int(first.exact13_C_rows),
                "exact_num7_C_rows": int(first.exact_num7_C_rows),
                "exact_mixed9_C_rows": int(first.exact_mixed9_C_rows),
                "settings_mutual_nearest": int((group.C_to_B_rank_min == 1).sum()),
                "settings_top3_team_for_record": int((group.team_distance_rank_for_record <= 3).sum()),
                "team_distance_rank_min": int(group.team_distance_rank_for_record.min()),
                "team_distance_rank_max": int(group.team_distance_rank_for_record.max()),
                "candidate_distance_rank_min": int(group.candidate_distance_rank.min()),
                "candidate_distance_rank_max": int(group.candidate_distance_rank.max())}
        for g in group.itertuples():
            item[g.method+"_team_distance_rank"] = int(g.team_distance_rank_for_record)
            item[g.method+"_candidate_distance_rank"] = int(g.candidate_distance_rank)
            item[g.method+"_specificity"] = g.specificity
        pairs.append(item)
    pairs = pd.DataFrame(pairs)
    # 主設定は計算前に指定。別設定の回数は独立試行・確率と解釈しない。
    record_rows, team_rows = [], []
    candidates_only = detail[detail.role == "candidate"]
    for record_id in candidates:
        g = candidates_only[candidates_only.record_id == record_id]
        primary = g[g.method == PRIMARY]
        closest = primary.sort_values(["d1", "team"]).iloc[0]
        bestspec = primary.sort_values(["specificity", "team"], ascending=[False, True]).iloc[0]
        top_sets = [set(v.loc[v.team_distance_rank_for_record <= 3, "team"].astype(int)) for _, v in g.groupby("method")]
        r = {"record_id": record_id, "primary_nearest_team": int(closest.team), "primary_d1": closest.d1,
             "primary_specificity_at_nearest": closest.specificity, "primary_nearest_known13_both_ge": int(closest.known13_both_at_least_as_strong_count),
             "primary_most_specific_team": int(bestspec.team), "primary_max_specificity": bestspec.specificity,
             "top3_teams_common_to_all_settings": ",".join(map(str, sorted(set.intersection(*top_sets)))),
             "primary_top3_closest_teams": join_ids(primary.sort_values(["d1", "team"]).head(3).team)}
        for method, v in g.groupby("method"):
            r[method+"_nearest_teams"] = join_ids(v[v.team_distance_rank_for_record == 1].team)
        record_rows.append(r)
    for team in teams:
        g = candidates_only[candidates_only.team == team]
        primary = g[g.method == PRIMARY]
        closest = primary.sort_values(["d1", "record_id"]).iloc[0]
        bestspec = primary.sort_values(["specificity", "record_id"], ascending=[False, True]).iloc[0]
        r = {"team": team, "closest_candidate_id": int(closest.record_id), "closest_d1": closest.d1,
             "closest_specificity": closest.specificity, "closest_known13_distance_le": int(closest.known13_distance_le_count),
             "most_specific_candidate_id": int(bestspec.record_id), "max_specificity": bestspec.specificity,
             "most_specific_known13_specificity_ge": int(bestspec.known13_specificity_ge_count),
             "primary_mutual_nearest_candidates": join_ids(primary[primary.C_to_B_rank_min == 1].record_id)}
        for method, v in g.groupby("method"):
            r[method+"_closest_candidate_ids"] = join_ids(v[v.candidate_distance_rank == 1].record_id)
        team_rows.append(r)
    frames = {"希少24行の区分.csv": rare[["record_id", "is_rare", "区分", "自課題ID"]], "全設定診断.csv": detail,
              "候補242組.csv": pairs, "候補別まとめ.csv": pd.DataFrame(record_rows),
              "チーム別まとめ.csv": pd.DataFrame(team_rows), "既知非会員比較.csv": pd.DataFrame(counts)}
    assert len(detail) == 24*22*3 and len(pairs) == 242
    assert not pairs.duplicated(["record_id", "team"]).any()
    assert not detail[["d1", "specificity"]].isna().any().any()
    frames["自チーム28検証.csv"] = self_check(b, rare, bg, scale, candidates)
    from レポート import candidate_summary
    frames["チーム別候補.csv"] = candidate_summary(detail, candidates)
    for name, frame in frames.items():
        frame.to_csv(OUT/name, index=False, encoding="utf-8-sig", float_format="%.17g")
    checked_paths(OPTIONS)
    run = {"source_commit": commit, "self_team": 28, "target_teams": teams,
           "out_of_scope_cohorts": sorted(set(range(30))-set(teams)-{28}),
           "candidate_record_ids": candidates, "known_nonmember_record_ids": sorted(private_ids),
           "methods": METHODS, "primary_method": PRIMARY, "numeric_columns": NUM,
           "log1p_columns": ["TG", "ALT"], "background_IQR_div_1_349": scale.tolist(),
           "input_sha256": hashes, "analysis_sha256": sha(Path(__file__)), "report_code_sha256": sha(HERE/"レポート.py"),
           "support_sha256": {name: sha(HERE/name) for name in ["共通設定.py", "候補選定.json", "入力ハッシュ.json", "検証.py", "説明テンプレート.md"]},
           "python": sys.version, "executable": Path(sys.executable).name, "pandas": pd.__version__,
           "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": version("matplotlib"), "platform": platform.platform(),
           "new_confirmed_members": 0, "new_confirmed_nonmembers_among_candidates": 0,
           "remaining_unknown_pairs": 242, "output_rows": {k: len(v) for k, v in frames.items()}}
    (OPTIONS.output_dir/"実行記録.json").write_text(json.dumps(run, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    from レポート import write_reports
    write_reports(frames, run, OPTIONS.output_dir)
    print(json.dumps({"pairs":len(pairs), "diagnostic_rows":len(detail)}, ensure_ascii=False))


def self_check(b, rare, bg, scale, candidates):
    """自チームは24行すべて既知会員。非会員比較には混ぜない。"""
    paths, _ = checked_paths(OPTIONS)
    c = read(paths["C_28.csv"])
    rows = []
    for method in METHODS:
        d = distances(b, c, method, scale)
        for pos in b.index[b.is_rare == 1]:
            cp = int(d[pos].argmin())
            other = d[:, cp].copy()
            other[pos] = np.inf
            record_id = int(b.loc[pos, "record_id"])
            rows.append({
                "team": 28, "record_id": record_id, "membership": "既知会員",
                "other_team_candidate": record_id in candidates, "method": method,
                "d1": d[pos, cp], "nearest_C_row0": cp, "nearest_C_csv_line": cp + 2,
                "nearest_competitor_distance": float(other.min()),
                "specificity": float(np.log((other.min()+EPS)/(d[pos, cp]+EPS))),
                "C_to_B_rank_min": int((d[:, cp] < d[pos, cp]-EPS).sum()+1),
                "sex_match": bool(b.loc[pos, "sex"] == c.loc[cp, "sex"]),
                "prefecture_match": bool(b.loc[pos, "prefecture"] == c.loc[cp, "prefecture"]),
                "smoking_match": bool(b.loc[pos, "smoking"] == c.loc[cp, "smoking"]),
                "onset_match": bool(b.loc[pos, "onset"] == c.loc[cp, "onset"]),
                "abs_time_difference": abs(b.loc[pos, "time"]-c.loc[cp, "time"]),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
