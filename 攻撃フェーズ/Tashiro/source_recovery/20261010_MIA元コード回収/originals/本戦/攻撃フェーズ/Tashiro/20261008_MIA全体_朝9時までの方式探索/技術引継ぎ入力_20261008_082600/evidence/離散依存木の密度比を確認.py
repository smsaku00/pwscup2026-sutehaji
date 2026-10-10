"""事前固定三形のsex条件付き離散PMF密度比。Gen/GT/採点/方式選択なし。

--write-planでtoy QA・全支援source/data inputのSHAを固定し、引数なしで実行。
現行A二分割の共通bin上の独自categorical adaptationであり連続density比ではない。
出力は集団単位CSV/JSONだけ。B個人行・ID・row位置・NPZは保存しない。
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import unicodedata
from collections import Counter, deque
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def no_gt_guard(event, args):
    if event != "open" or not args or isinstance(args[0], int): return
    path = unicodedata.normalize("NFC", str(args[0]))
    forbidden = ("pool_ids", "/正解データ/", "/アーカイブ/", "F_mia.csv", "対象別TPR.csv",
                 "集約比較.csv", "result.txt", "attack_score.py", ".npz")
    if any(term in path for term in forbidden): raise PermissionError(f"本診断は参照禁止: {path}")


sys.addaudithook(no_gt_guard)
import 共通 as u
split = u.module("fixed_split_A_for_discrete_tree", u.HERE / "公開背景の分割較正.py")
FORMS = ("independent_pmf", "Aref_shared_MI_tree", "source_specific_MI_tree")
ALPHA, RHO, ROOT_INDEX = .5, .98, 0
QUANTILES = tuple(j / 10 for j in range(1, 10))
CAL_INDEX = next(k for k, c in enumerate(split.CONFIGS)
                 if c["source"] == "Q_A_half" and c["shape"] == "county300_sex_age100")
PRIMARY = {"title": "Forest Density Estimation", "authors": "Liu et al.", "year": 2011,
           "url": "https://www.jmlr.org/papers/v12/liu11a.html",
           "pdf": "https://www.jmlr.org/papers/volume12/liu11a/liu11a.pdf",
           "relevant_framework": "Forest factorization and maximum spanning tree weighted by pair mutual information; JMLR pp908,910,912-913",
           "adaptation": "Independent fixed categorical decile-bin PMF adaptation; not the paper continuous KDE/heldout forest pruning procedure or full reproduction"}
METHOD = {
    "version": 1, "forms": list(FORMS), "numeric_columns": list(u.NUM7),
    "numeric_transform": "TG/ALT log1p before binning; no other scaling or Jacobian",
    "sex_conditioning": "Separate models/cuts for sex0/sex1; no sex prior multiplier",
    "bins": {"reference": "Current phase fold Aref within same sex", "quantiles": list(QUANTILES),
             "dedup_cuts": True, "outer_edges": ["-infinity", "infinity"], "cut_side": "right",
             "same_C_A_query_cuts": True},
    "MI": "Unsmooth empirical pair categorical MI in natural-log units; zero-count terms omitted; negative numerical residue clamped at zero",
    "tree": "Four-edge maximum spanning tree, Kruskal sort(-MI,index_i,index_j); canonical lexicographic edge output; root index0(age), sorted-neighbor BFS parents",
    "forms_tree_rule": {FORMS[0]: "No edges; each attribute marginal",
                        FORMS[1]: "Sex Aref empirical MI tree shared by both C/A fits",
                        FORMS[2]: "Each sex source own empirical MI tree; common Aref bin state spaces"},
    "smoothing": {"alpha_per_state": ALPHA,
                  "root_or_independent": "(count_state+alpha)/(n_source+alpha*K_node)",
                  "child_given_parent": "(count_parent_child+alpha)/(count_parent+alpha*K_child)",
                  "unseen_parent": "uniform child conditional over all common Aref bin states",
                  "separate_joint_marginal_Laplace_division": False},
    "pref": "Independent source-average rho*empirical_source_pref + (1-rho)*same-sex Aref counts+1 frequency over fold Aref category universe",
    "rho": RHO, "score": "log conditional discrete PMF(C bins,pref)-log conditional discrete PMF(Aref bins,pref)",
    "continuous_density_or_Jacobian_ratio": False, "primary_source": PRIMARY,
    "A_split": "Current A 5143; seed20261008 half2571/2572 swapped; each A row heldout once",
    "calibration": split.CONFIGS[CAL_INDEX],
    "owner_holdout": "Q target owner and heldowner excluded; outer-training Q-only IQR per form/fold, std-or-one fallback shared pipeline",
    "candidate": "Two-fold mean CDF -> fixed normal875 average rank -> cross median0.1 jointly across all20/22targets; no raw tie term",
    "candidate_control": FORMS[0], "top": 12, "head_ties": "descending minimum rank<=12; head size may exceed12",
    "aggregation": "Eligible targets mean within Q owner then owner equal weight; preliminary primary targets exclude1/2",
    "exports": "Aggregate tables, public learned cuts/MI/tree structure, preexecution plan/QA/execution records; no individual ID/position/query row/NPZ",
    "limits": ["Different forms change the raw score variable; normalized CRPS does not rank membership discrimination or overall model performance.",
               "Averaging fold CDFs is not generally uniform PIT; mean-fold threshold rates and averaged-score thresholds are separate.",
               "A/Q known-nonmember status does not guarantee representation of mixed normal875 candidate nonmembers.",
               "Owner-heldout CRPS/CDF diagnostics are before candidate rank/cross; neither establishes official FPR/accuracy.",
               "Empirical trees/bins are fixed descriptors and models, not processing mechanism proofs or causal identification."],
    "membership_labels_used": False, "mia_scoring": False, "form_selection": False,
    "bin_search": False, "alpha_search": False, "only_current_phase_A": True,
    "final_prose_or_github_created": False,
}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")


def stamp(): return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def loaded_sources():
    expected = {u.OLD / "分析.py", u.OLD / "非会員を使う順位較正の検討.py", u.CAL / "分析.py",
                u.HERE / "共通.py", u.HERE / "公開背景の分割較正.py",
                u.HERE / "公開非会員だけで較正を選ぶ.py", u.HERE / "非会員較正の広い集団を保留検証.py"}
    actual = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
              if getattr(m, "__file__", None) and Path(m.__file__).suffix == ".py"
              and Path(m.__file__).resolve().is_relative_to(u.ROOT / "本戦")
              and Path(m.__file__).resolve() != Path(__file__).resolve()}
    assert actual == {p.resolve() for p in expected}, actual
    return {str(p): u.sha(p) for p in sorted(actual)}


def input_paths():
    result = []
    for phase in ("prelim", "main"):
        if phase == "prelim":
            folder = u.ROOT / "予備戦/配布データ"
            own = sorted(folder.glob("PWSCup2026_prelim_team05_*")); assert len(own) == 1
            background = folder / "PWSCup2026_prelim_common/A_bg.csv"
            targets = folder / "PWSCup2026_prelim_attack"; self_id = 5
        else:
            folder = u.base.child(u.ROOT / "本戦", "配布データ")
            own = sorted(folder.glob("PWSCup2026_final_team28_*")); assert len(own) == 1
            background = own[0] / "A_bg.csv"
            targets = u.HERE.parent / "20260928_配布データ基本分析/入力"; self_id = 28
        cp = sorted(targets.glob("C_*.csv")); teams = {int(p.stem.split("_")[-1]) for p in cp}
        qp = [targets / f"aia_challenge_{t}.csv" for t in sorted(teams | {self_id})]
        result.extend([background, own[0] / "B_self.csv"] + cp + qp)
    assert len(result) == len(set(result)) == 92
    return {str(p.resolve()): u.sha(p) for p in result}


def bin_values(values, cuts):
    values = np.asarray(values, float)
    result = np.column_stack([np.searchsorted(cut, values[:, j], side="right") for j, cut in enumerate(cuts)])
    assert all((result[:, j] >= 0).all() and (result[:, j] <= len(cut)).all() for j, cut in enumerate(cuts))
    return result.astype(np.int64)


def empirical_mi(values, cards):
    n, d = values.shape; result = np.zeros((d, d))
    for i, j in combinations(range(d), 2):
        joint = np.bincount(values[:, i] * cards[j] + values[:, j], minlength=cards[i] * cards[j]).reshape(cards[i], cards[j])
        ni, nj = joint.sum(axis=1), joint.sum(axis=0)
        contributions = [float(count) / n * math.log(float(n * count) / float(ni[a] * nj[b]))
                         for a, b in zip(*np.nonzero(joint)) for count in [joint[a, b]]]
        result[i, j] = result[j, i] = max(math.fsum(contributions), 0.)
    return result


def mst(mi):
    d = len(mi); ancestors = list(range(d)); edges = []
    def find(i):
        while ancestors[i] != i:
            ancestors[i] = ancestors[ancestors[i]]; i = ancestors[i]
        return i
    ordered = sorted(combinations(range(d), 2), key=lambda pair: (-mi[pair], pair[0], pair[1]))
    for i, j in ordered:
        a, b = find(i), find(j)
        if a != b:
            ancestors[b] = a; edges.append((i, j))
        if len(edges) == d - 1: break
    assert len(edges) == d - 1
    return tuple(sorted(edges))


def parents_from_edges(edges, d):
    adjacent = [[] for _ in range(d)]
    for i, j in edges: adjacent[i].append(j); adjacent[j].append(i)
    parents = [-2] * d; parents[ROOT_INDEX] = -1; queue = deque([ROOT_INDEX])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacent[node]):
            if parents[neighbor] == -2: parents[neighbor] = node; queue.append(neighbor)
    assert -2 not in parents
    return tuple(parents)


def fit_pmf(values, cards, edges):
    n, d = values.shape; parents = tuple([-1] * d) if edges is None else parents_from_edges(edges, d)
    tables = []; row_error = 0.
    for j, parent in enumerate(parents):
        if parent < 0:
            count = np.bincount(values[:, j], minlength=cards[j])
            table = (count + ALPHA) / (n + ALPHA * cards[j])
            error = abs(float(table.sum()) - 1.)
        else:
            count = np.bincount(values[:, parent] * cards[j] + values[:, j],
                                minlength=cards[parent] * cards[j]).reshape(cards[parent], cards[j])
            table = (count + ALPHA) / (count.sum(axis=1, keepdims=True) + ALPHA * cards[j])
            error = float(np.max(np.abs(table.sum(axis=1) - 1.)))
        assert np.isfinite(table).all() and (table > 0).all() and error < 1e-14
        tables.append(table); row_error = max(row_error, error)
    return {"parents": parents, "tables": tables, "cards": tuple(cards), "normalization_row_error": row_error}


def log_pmf(model, query):
    query = np.asarray(query, np.int64); result = np.zeros(len(query))
    for j, parent in enumerate(model["parents"]):
        table = model["tables"][j]
        probability = table[query[:, j]] if parent < 0 else table[query[:, parent], query[:, j]]
        result += np.log(probability)
    assert np.isfinite(result).all()
    return result


def numerical_qa():
    a = np.array([[x, x, y, y, 0] for x, y in product(range(2), repeat=2)])
    c = np.array([[0, 0, 0, 1, 0], [0, 1, 1, 0, 1], [1, 0, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
    mi = empirical_mi(a, (2,) * 5); expected = np.zeros((5, 5))
    expected[0, 1] = expected[1, 0] = expected[2, 3] = expected[3, 2] = math.log(2.)
    mi_error = float(np.max(np.abs(mi - expected))); assert mi_error < 1e-15
    tree = mst(mi); assert tree == ((0, 1), (0, 2), (0, 4), (2, 3))
    assert parents_from_edges(tree, 5) == (-1, 0, 0, 2, 0)
    assert mst(np.zeros((5, 5))) == ((0, 1), (0, 2), (0, 3), (0, 4))
    maximum, spanning = -math.inf, 0
    for edges in combinations(combinations(range(5), 2), 4):
        reach = {0}
        while True:
            added = reach | {j for i, j in edges if i in reach} | {i for i, j in edges if j in reach}
            if added == reach: break
            reach = added
        if len(reach) == 5:
            spanning += 1; maximum = max(maximum, sum(mi[i, j] for i, j in edges))
    assert spanning == 125 and abs(sum(mi[i, j] for i, j in tree) - maximum) < 1e-15
    full_error = row_error = single_error = 0.; checks = 0
    for cards in [(2,) * 5, (2, 3, 2, 2, 2)]:
        shared = mst(empirical_mi(a, cards)); grid = np.array(list(product(*(range(k) for k in cards))))
        for source in (a, c):
            own = mst(empirical_mi(source, cards))
            for edges in (None, shared, own):
                model = fit_pmf(source, cards, edges); value = log_pmf(model, grid)
                full_error = max(full_error, abs(float(np.exp(value).sum()) - 1.))
                row_error = max(row_error, model["normalization_row_error"])
                single = np.array([log_pmf(model, row[None, :])[0] for row in grid])
                direct = np.array([math.prod(model["tables"][j][row[j]] if parent < 0
                                    else model["tables"][j][row[parent], row[j]]
                                    for j, parent in enumerate(model["parents"])) for row in grid])
                single_error = max(single_error, float(np.max(np.abs(single - value))),
                                   float(np.max(np.abs(direct - np.exp(value)))))
                checks += 1
        forced = fit_pmf(a, cards, ((0, 1), (1, 2), (2, 3), (3, 4)))
        if cards[1] == 3: assert np.array_equal(forced["tables"][2][2], np.array([.5, .5]))
        full_error = max(full_error, abs(float(np.exp(log_pmf(forced, grid)).sum()) - 1.)); checks += 1
    hand = fit_pmf(a, (2,) * 5, tree)
    assert np.array_equal(hand["tables"][0], np.array([.5, .5]))
    assert np.allclose(hand["tables"][4], np.array([[5/6, 1/6], [5/6, 1/6]]), rtol=0, atol=1e-15)
    cuts = [np.unique(np.quantile(np.array([0., 0., 1., 1.]), QUANTILES)) for _ in range(5)]
    binned = bin_values(np.array([[-1e9] * 5, [1e9] * 5]), cuts)
    assert np.array_equal(binned[0], np.zeros(5, int)) and np.array_equal(binned[1], [len(k) for k in cuts])
    ref_freq = np.array([.2, .3, .5]); source_freq = np.array([0., .75, .25])
    pref_error = abs(float((RHO * source_freq + (1-RHO) * ref_freq).sum()) - 1.)
    assert full_error < 1e-14 and single_error < 1e-14 and pref_error < 1e-14
    return {"finite_grid_pmf_checks": checks, "full_grid_sum_one_max_error": full_error,
            "conditional_row_sum_one_max_error": row_error, "batch_single_and_direct_product_max_error": single_error,
            "hand_MI_max_error": mi_error, "enumerated_spanning_trees": spanning, "MST_weight_error": 0.,
            "zero_MI_lexical_tie_check": True, "BFS_age_root_check": True,
            "unseen_parent_uniform_check": True, "dedup_cut_and_infinite_outer_bin_check": True,
            "pref_factor_sum_one_error": pref_error, "shared_calibration_CRPS_QA": split.numerical_qa()}


def contexts(reference):
    levels = sorted(reference.prefecture.unique()); result = {}
    for sex, frame in reference.groupby("sex", sort=True):
        values = u.base.numbers(frame, u.NUM7)
        cuts = tuple(np.unique(np.quantile(values[:, j], QUANTILES)) for j in range(5))
        cards = tuple(len(cut) + 1 for cut in cuts); bins = bin_values(values, cuts)
        mi = empirical_mi(bins, cards); edges = mst(mi)
        freq = (frame.prefecture.value_counts().reindex(levels, fill_value=0) + 1) / (len(frame) + len(levels))
        result[int(sex)] = {"cuts": cuts, "cards": cards, "MI": mi, "edges": edges, "freq": freq, "rows": len(frame)}
    assert set(result) == {0, 1}
    return result


def density(source, query, context):
    result = np.full((3, len(query)), np.nan); structures = []; pairs = []
    for sex, ctx in sorted(context.items()):
        part = source.loc[source.sex.eq(sex)]; assert len(part) > 0
        positions = np.flatnonzero(query.sex.to_numpy() == sex); q = query.iloc[positions]
        bins = bin_values(u.base.numbers(part, u.NUM7), ctx["cuts"])
        qb = bin_values(u.base.numbers(q, u.NUM7), ctx["cuts"])
        mi = empirical_mi(bins, ctx["cards"]); own_edges = mst(mi)
        assert set(part.prefecture) <= set(ctx["freq"].index) and set(q.prefecture) <= set(ctx["freq"].index)
        source_freq = part.prefecture.value_counts().reindex(ctx["freq"].index, fill_value=0) / len(part)
        pref = RHO * source_freq + (1-RHO) * ctx["freq"]
        pref_error = abs(float(pref.sum()) - 1.); assert pref_error < 1e-14
        pref_log = np.log(q.prefecture.map(pref).to_numpy(float))
        for i, j in combinations(range(5), 2):
            pairs.append({"sex": sex, "attribute1": u.NUM7[i], "attribute2": u.NUM7[j],
                          "source_rows": len(part), "empirical_MI": float(mi[i, j]),
                          "in_source_MST": (i, j) in own_edges, "in_Aref_MST": (i, j) in ctx["edges"]})
        for k, edges in enumerate((None, ctx["edges"], own_edges)):
            model = fit_pmf(bins, ctx["cards"], edges)
            result[k, positions] = log_pmf(model, qb) + pref_log
            structures.append({"sex": sex, "form": FORMS[k], "source_rows": len(part),
                "bin_states": json.dumps(ctx["cards"]), "undirected_edges": json.dumps(edges or []),
                "directed_parents": json.dumps(model["parents"]), "edges": len(edges or []),
                "root_index": ROOT_INDEX, "conditional_row_sum_one_error": model["normalization_row_error"],
                "pref_factor_sum_one_error": pref_error})
    assert np.isfinite(result).all()
    return result, structures, pairs


def phase_run(phase, out, qa, plan):
    started = time.perf_counter(); data = u.setup(phase)
    owner, _, duplicates = split.wide.pub.public_folds(data)
    assignment, parts = split.split_a(data); a_start = len(data["query"])
    queries = [pd.concat([data["query"][list(u.base.QI7) + ["age_bin"]],
                          parts[1-f][list(u.base.QI7) + ["age_bin"]]], ignore_index=True) for f in range(2)]
    contexts_by_fold = [contexts(part) for part in parts]
    background = []; structures = []; pairs = []; cut_rows = []
    for fold in range(2):
        values, ss, pp = density(parts[fold], queries[fold], contexts_by_fold[fold]); background.append(values)
        assert np.array_equal(values[1], values[2])
        structures.extend(dict(phase=phase, fold=fold, target=-1, source="Aref", **row) for row in ss)
        pairs.extend(dict(phase=phase, fold=fold, target=-1, source="Aref", **row) for row in pp)
        for sex, ctx in contexts_by_fold[fold].items():
            for j, cut in enumerate(ctx["cuts"]):
                cut_rows.append({"phase": phase, "fold": fold, "sex": sex, "attribute": u.NUM7[j],
                                 "reference_rows": ctx["rows"], "bin_states": len(cut)+1,
                                 "finite_cuts": cut.tolist(), "outer_edges": ["-infinity", "infinity"]})
    eligible_targets = [t for t in data["teams"] if phase != "prelim" or t not in (1, 2)]
    banks = {form: {} for form in FORMS}; outer = []; fold_rows = []; raw_checks = []
    for complete, (target, c) in enumerate(data["cs"].items(), 1):
        raws = []; predictions = []
        for fold in range(2):
            values, ss, pp = density(c, queries[fold], contexts_by_fold[fold]); raw = values-background[fold]; raws.append(raw)
            structures.extend(dict(phase=phase, fold=fold, target=target, source="C", **row) for row in ss)
            pairs.extend(dict(phase=phase, fold=fold, target=target, source="C", **row) for row in pp)
            indices = np.unique(np.r_[0, 1048, a_start, len(queries[fold])-1])
            single_c = np.column_stack([density(c, queries[fold].iloc[[j]], contexts_by_fold[fold])[0][:, 0] for j in indices])
            single_a = np.column_stack([density(parts[fold], queries[fold].iloc[[j]], contexts_by_fold[fold])[0][:, 0] for j in indices])
            error = float(np.max(np.abs(single_c-single_a-raw[:, indices]))); assert error == 0
            raw_checks.append({"phase": phase, "target": target, "fold": fold,
                               "batch_single_raw_max_error": error, "raw_min": float(raw.min()), "raw_max": float(raw.max()),
                               "all_log_PMFS_finite": True})
            predictions.append(np.stack([split.predict_cdf(raw[k], queries[fold], data, target, owner, a_start)[0][CAL_INDEX]
                                         for k in range(3)]))
        mean = (predictions[0]+predictions[1])/2
        for k, form in enumerate(FORMS): banks[form][target] = u.rank(mean[k], data["normal"])
        if target in eligible_targets:
            for heldowner in sorted(data["qs"]):
                if heldowner == target: continue
                for k, form in enumerate(FORMS):
                    cases = [split.evaluate_fold(raws[f][k], queries[f], data, target, owner, heldowner, a_start) for f in range(2)]
                    losses = (cases[0]["losses"][:, CAL_INDEX]+cases[1]["losses"][:, CAL_INDEX])/2
                    score = (cases[0]["pits"][:, CAL_INDEX]+cases[1]["pits"][:, CAL_INDEX])/2
                    for f, case in enumerate(cases):
                        pit = case["pits"][:, CAL_INDEX]
                        fold_rows.append({"phase": phase, "target": target, "heldout_null_owner": heldowner, "fold": f, "form": form,
                            "crps": float(case["losses"][:, CAL_INDEX].mean()), "cdf_score_mean": float(pit.mean()),
                            "fold_cdf_above_99": float((pit>.99).mean()), "fold_cdf_above_95": float((pit>.95).mean()),
                            "training_Q_rows": case["q_training_rows"], "training_Q_iqr": case["scale"], "scale_rule": case["scale_rule"],
                            "A_null_rows": case["a_null_rows"], "heldout_rows": 326, "membership_labels_used": False})
                    outer.append({"phase": phase, "target": target, "heldout_null_owner": heldowner, "form": form,
                        "crps": float(losses.mean()),
                        "mean_fold_cdf_above_99": float(np.mean([(case["pits"][:, CAL_INDEX]>.99).mean() for case in cases])),
                        "mean_fold_cdf_above_95": float(np.mean([(case["pits"][:, CAL_INDEX]>.95).mean() for case in cases])),
                        "averaged_cdf_score_above_99": float((score>.99).mean()), "averaged_cdf_score_above_95": float((score>.95).mean()),
                        "averaged_cdf_score_mean": float(score.mean()), "heldout_rows": 326,
                        "used_for_selection": False, "membership_labels_used": False})
        u.log(stage="discrete_tree_target", phase=phase, targets_complete=complete, targets=len(data["teams"]), seconds=time.perf_counter()-started)
    matrices = u.matrices(banks, data["teams"], cross=(.1,)); table = pd.DataFrame(outer)
    metrics = ["crps", "mean_fold_cdf_above_99", "mean_fold_cdf_above_95", "averaged_cdf_score_above_99",
               "averaged_cdf_score_above_95", "averaged_cdf_score_mean"]
    panel = table.groupby(["phase", "heldout_null_owner", "form"], as_index=False)[metrics].mean()
    summary = panel.groupby(["phase", "form"], as_index=False)[metrics].mean()
    summary["targets"] = len(eligible_targets); summary["owners"] = len(data["qs"]); summary["used_for_selection"] = False
    target_table = table.groupby(["phase", "target", "form"], as_index=False)[metrics].mean()
    control = matrices[FORMS[0]+"__cross0.1"]; drift = []
    for form, mat in matrices.items():
        for j, target in enumerate(data["teams"]):
            old = set(np.flatnonzero(rankdata(-control[:, j], method="min")<=12))
            new = set(np.flatnonzero(rankdata(-mat[:, j], method="min")<=12))
            a, b = rankdata(control[:, j]), rankdata(mat[:, j])
            corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a)>0 and np.std(b)>0 else None
            drift.append({"phase": phase, "target": target, "form": form, "independent_head_rows": len(old), "form_head_rows": len(new),
                          "added": len(new-old), "removed": len(old-new), "spearman": corr, "membership_accuracy_unknown": True})
    dest = out/phase; dest.mkdir()
    write_json(dest/"Arefの固定bin境界.json", cut_rows)
    pd.DataFrame(structures).to_csv(dest/"木構造とPMF正規化.csv", index=False, float_format="%.17g")
    pd.DataFrame(pairs).to_csv(dest/"経験pairMIと木の辺.csv", index=False, float_format="%.17g")
    pd.DataFrame(raw_checks).to_csv(dest/"batchとsingleの生スコア検証.csv", index=False, float_format="%.17g")
    pd.DataFrame(fold_rows).to_csv(dest/"fold別の課題所有者保留.csv", index=False, float_format="%.17g")
    table.to_csv(dest/"二fold集約の課題所有者保留.csv", index=False, float_format="%.17g")
    panel.to_csv(dest/"Q所有者等重みの診断パネル.csv", index=False, float_format="%.17g")
    summary.to_csv(dest/"固定三形の非会員診断.csv", index=False, float_format="%.17g")
    target_table.to_csv(dest/"対象別の固定三形診断.csv", index=False, float_format="%.17g")
    pd.DataFrame(drift).to_csv(dest/"独立形との上位12変化.csv", index=False, float_format="%.17g")
    for path, hashed in u.base.INPUT_HASHES.items():
        assert path in plan["input_sha256_before_execution"] and u.sha(path) == hashed == plan["input_sha256_before_execution"][path]
    for path, hashed in plan["support_code_sha256"].items(): assert u.sha(path) == hashed
    write_json(dest/"実行記録.json", {"completed_jst": stamp(), "seconds": time.perf_counter()-started,
        "code_sha256": u.sha(__file__), "support_code_sha256": plan["support_code_sha256"],
        "input_sha256": dict(u.base.INPUT_HASHES), "output_sha256": {p.name: u.sha(p) for p in sorted(dest.iterdir())},
        "python": sys.executable, "python_version": platform.python_version(), "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "forms": list(FORMS), "candidate_targets": data["teams"], "diagnostic_targets": eligible_targets,
        "A_reference_rows": [len(p) for p in parts], "A_split_seed": 20261008, "public_Q_duplicates": duplicates,
        "owner_target_cases": len(table), "fold_owner_target_cases": len(fold_rows), "numerical_qa": qa,
        "batch_single_raw_max_error": max(r["batch_single_raw_max_error"] for r in raw_checks),
        "conditional_row_sum_one_max_error": max(r["conditional_row_sum_one_error"] for r in structures),
        "pref_factor_sum_one_max_error": max(r["pref_factor_sum_one_error"] for r in structures),
        "candidate_protocol": METHOD["candidate"], "calibration": METHOD["calibration"], "limitations": METHOD["limits"],
        "no_GT_guard_installed": True, "membership_labels_used": False, "mia_scoring": False,
        "form_selection": False, "bin_search": False, "alpha_search": False, "individual_rows_or_NPZ_exported": False,
        "final_prose_or_github_created": False})
    u.log(stage="discrete_tree_phase_complete", phase=phase, owner_target_cases=len(table), seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=u.HERE/"離散依存木の密度比結果")
    args = parser.parse_args(); out = args.output_dir.resolve()
    assert sys.dont_write_bytecode and Path(sys.executable).absolute() == u.ROOT/".venv/bin/python"
    if args.write_plan:
        assert not out.exists(); supports = loaded_sources(); qa = numerical_qa(); inputs = input_paths()
        out.mkdir(); write_json(out/"事前固定条件.json", {"created_jst": stamp(), "method": METHOD,
            "code_sha256": u.sha(__file__), "support_code_sha256": supports, "input_sha256_before_execution": inputs, "numerical_qa": qa})
        u.log(stage="discrete_tree_plan_fixed", code_sha256=u.sha(__file__), support_sources=len(supports), input_files=len(inputs), numerical_qa=qa)
        return
    assert out.is_dir() and {p.name for p in out.iterdir()} == {"事前固定条件.json"}
    plan = json.loads((out/"事前固定条件.json").read_text())
    assert plan["method"] == METHOD and plan["code_sha256"] == u.sha(__file__)
    assert plan["support_code_sha256"] == loaded_sources() and plan["input_sha256_before_execution"] == input_paths()
    qa = numerical_qa(); assert qa == plan["numerical_qa"]
    for phase in ("prelim", "main"): phase_run(phase, out, qa, plan)
    assert dict(u.base.INPUT_HASHES) == plan["input_sha256_before_execution"]
    for path, hashed in plan["input_sha256_before_execution"].items(): assert u.sha(path) == hashed
    for path, hashed in plan["support_code_sha256"].items(): assert u.sha(path) == hashed
    write_json(out/"完了記録.json", {"completed_jst": stamp(), "code_sha256": u.sha(__file__),
        "phases_completed": ["prelim", "main"], "support_sources": 7, "input_files": 92,
        "support_and_input_hash_guards_passed": True,
        "output_sha256": {str(p.relative_to(out)): u.sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "membership_labels_used": False, "mia_scoring": False, "method_selection": False,
        "individual_rows_or_NPZ_exported": False, "final_prose_or_github_created": False})


if __name__ == "__main__": main()
