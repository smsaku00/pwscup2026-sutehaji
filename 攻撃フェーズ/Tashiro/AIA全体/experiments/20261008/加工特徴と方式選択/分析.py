"""Cの公開特徴と固定AIA方式の関連。提出個票・ZIP・tokenは作成しない。

ルータは予備戦だけで nested leave-team-out 検証する。本戦正解・スコアは
入力に含めない。Cの特徴抽出関数にチーム番号・得点・正解は渡さない。
既存20261007分析を読み取り参照し、新規集計だけをこのフォルダに保存する。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, io, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, wasserstein_distance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = HERE.parents[1] / "20261007_AIA全体向上の分析"
OUT = HERE / "集計"
SEED = 20261008
QI = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
NUM = QI[:5]
LAB = NUM[1:]
ROUTE_FEATURES = ["lab_round2_share", "lab_unique_share", "corr_gap_bg_sex",
                  "q_C_full_nn_median", "time_long_share", "BG_QI7_copy_share"]
PENALTIES = [0., .025, .05, .1]
MIN_LEAF = 4
# 少数の固定代表方式。対象別の探索最高方式を代表として採用しない。
CANDIDATES = [
    ("partial2_matching", "同点修正_対象別.csv", "stable_matching/sex_quota/long75"),
    ("S4_nn", "予備戦_全設定と対象別.csv", "S4_log_iqr_soft/nn_all"),
    ("full_sex_pref_nn", "予備戦_全設定と対象別.csv", "log_iqr_both/nn_all"),
    ("partial4_nn", "予備戦_全設定と対象別.csv", "partial4_log_iqr_both/nn_all"),
    ("partial2_density", "多尺度密度_対象別.csv", "density/partial2_sex_k3/long_gate0.75"),
    ("GenLRA_nn", "GenLRA_AIA_対象別.csv", "GenLRA/ordinal7h7/NN/gate0.75"),
    ("gaussian_source", "分布識別_対象別.csv", "sex/linear_gaussian/pooled/long0.75"),
    ("constant95", "予備戦_全設定と対象別.csv", "constant_9.5"),
]
HASHES = {}


def read(path):
    data = Path(path).read_bytes()
    HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return pd.read_csv(io.BytesIO(data), float_precision="round_trip")


def raw(df):
    x = df[NUM].to_numpy(float).copy()
    x[:, [2, 4]] = np.log1p(x[:, [2, 4]])
    return x


def copy_share(df, reference):
    keys = set(reference[QI].itertuples(index=False, name=None))
    return sum(k in keys for k in df[QI].itertuples(index=False, name=None))/len(df)


def distance(q, ref, bg):
    scale = np.maximum(np.quantile(raw(bg), .75, axis=0)-np.quantile(raw(bg), .25, axis=0), 1e-5)
    d = cdist(raw(q)/scale, raw(ref)/scale, "sqeuclidean")/5
    d[q.sex.to_numpy()[:, None] != ref.sex.to_numpy()[None, :]] = np.inf
    assert np.isfinite(d).any(1).all()
    return np.sqrt(d)


def features(c, q, bg, kitbg):
    """公開C・課題のQI・背景だけ。正解・得点・番号は引数にない。"""
    row = {}
    for col in NUM+["BMI", "FPG", "time"]:
        row[f"unique_{col}_share"] = c[col].nunique()/len(c)
        row[f"round2_{col}_share"] = float(np.isclose(c[col], np.round(c[col], 2), atol=1e-8, rtol=0).mean())
    row["lab_round2_share"] = float(np.mean([row[f"round2_{col}_share"] for col in LAB]))
    row["lab_unique_share"] = float(np.mean([row[f"unique_{col}_share"] for col in LAB]))
    row["QI7_duplicate_share"] = c.duplicated(QI).mean()
    row["male_share"] = c.sex.mean()
    row["prefecture_unique"] = c.prefecture.nunique()
    row["prefecture_mode_share"] = c.prefecture.value_counts(normalize=True).iloc[0]
    row["time_long_share"] = c.time.ge(9.5).mean()
    row["time10_share"] = c.time.eq(10).mean()
    row["onset_share"] = c.onset.mean()
    row["onset1_long_share"] = c.loc[c.onset.eq(1), "time"].ge(9.5).mean() if c.onset.eq(1).any() else 0.
    noevent = c.onset.eq(0) & c.death.eq(0)
    row["no_event_short_share"] = c.loc[noevent, "time"].lt(9.5).mean() if noevent.any() else 0.
    row["BG_phase_QI7_copy_share"] = copy_share(c, bg)
    row["BG_kit_QI7_copy_share"] = copy_share(c, kitbg)
    row["BG_QI7_copy_share"] = max(row["BG_phase_QI7_copy_share"], row["BG_kit_QI7_copy_share"])
    cb, bb = raw(c), raw(bg)
    scales = np.maximum(np.quantile(bb, .75, axis=0)-np.quantile(bb, .25, axis=0), 1e-5)
    row["distribution_gap_bg"] = float(np.mean([wasserstein_distance(cb[:, j], bb[:, j])/scales[j] for j in range(5)]))
    gaps = []
    off = np.triu_indices(5, 1)
    for sex in [0, 1]:
        xc, xb = cb[c.sex.eq(sex)], bb[bg.sex.eq(sex)]
        if min(len(xc), len(xb)) >= 10:
            rc, rb = np.corrcoef(xc, rowvar=False), np.corrcoef(xb, rowvar=False)
            gaps.extend(np.abs(rc[off]-rb[off]).tolist())
    row["corr_gap_bg_sex"] = float(np.mean(gaps))
    row["BG_full_nn_median"] = float(np.median(distance(c, bg, bg).min(1)))
    dc = distance(q, c, bg)
    row["q_C_full_nn_median"] = float(np.median(dc.min(1)))
    row["q_C_full_nn_q25"] = float(np.quantile(dc.min(1), .25))
    rng = np.random.default_rng(SEED)
    ref = bg.iloc[np.sort(rng.choice(len(bg), len(c), replace=False))]
    db = distance(q, ref, bg)
    row["q_C_BG_nn_ratio_median"] = float(np.median((dc.min(1)+1e-9)/(db.min(1)+1e-9)))
    near = np.argsort(dc, axis=1, kind="stable")[:, :5]
    spans = np.ptp(c.time.to_numpy()[near], axis=1)
    row["q_C_near5_time_span_median"] = float(np.median(spans))
    row["q_C_near5_time_same_share"] = float(np.mean(spans <= .5))
    return row


def predict_policy(policy, x):
    if policy["feature"] is None:
        return np.full(len(x), policy["left_method"], int)
    return np.where(x[:, policy["feature"]] <= policy["threshold"], policy["left_method"], policy["right_method"])


def fit_policy(x, y, penalty):
    """固定1分岐。外側評価の特徴/得点はこの関数に入れない。"""
    adjusted = y.copy()
    adjusted[:, 1:] -= penalty
    mean = adjusted.mean(0)
    constant = int(np.argmax(mean))
    best_value = float(mean[constant])
    best = dict(feature=None, threshold=None, left_method=constant, right_method=constant, penalty=penalty)
    n = len(x)
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="stable")
        values = x[order, j]
        sums = np.cumsum(adjusted[order], axis=0)
        cuts = np.arange(MIN_LEAF, n-MIN_LEAF+1)
        cuts = cuts[values[cuts-1] < values[cuts]]
        if not len(cuts):
            continue
        left = sums[cuts-1]/cuts[:, None]
        right = (sums[-1]-sums[cuts-1])/(n-cuts)[:, None]
        lm, rm = left.argmax(1), right.argmax(1)
        candidate = (left[np.arange(len(cuts)), lm]*cuts+right[np.arange(len(cuts)), rm]*(n-cuts))/n
        z = int(candidate.argmax())
        if candidate[z] > best_value + 1e-12:
            cut = cuts[z]
            best_value = float(candidate[z])
            best = dict(feature=j, threshold=float((values[cut-1]+values[cut])/2), left_method=int(lm[z]), right_method=int(rm[z]), penalty=penalty)
    return best


def choose_penalty(x, y):
    """inner leave-team-outで非基準選択ペナルティだけ選択。"""
    means = []
    for penalty in PENALTIES:
        values = []
        for i in range(len(x)):
            train = np.arange(len(x)) != i
            policy = fit_policy(x[train], y[train], penalty)
            method = predict_policy(policy, x[[i]])[0]
            values.append(y[i, method])
        means.append(float(np.mean(values)))
    maximum = max(means)
    # 同点なら大きいペナルティへ寄せる。
    chosen = max(p for p, score in zip(PENALTIES, means) if score >= maximum-1e-12)
    return chosen, dict(zip(map(str, PENALTIES), means))


def nested(x, y, teams, groups=None):
    rows = []
    units = teams if groups is None else np.unique(groups)
    for unit in units:
        test = teams == unit if groups is None else groups == unit
        train = ~test
        penalty, inner = choose_penalty(x[train], y[train])
        policy = fit_policy(x[train], y[train], penalty)
        chosen = predict_policy(policy, x[test])
        for k, method, rowid in zip(teams[test], chosen, np.flatnonzero(test)):
            rows.append(dict(team=int(k), chosen_method=CANDIDATES[method][0], r_all=float(y[rowid, method]),
                             baseline_r_all=float(y[rowid, 0]), delta=float(y[rowid, method]-y[rowid, 0]),
                             heldout_group=str(unit) if groups is not None else "single_team",
                             feature=None if policy["feature"] is None else ROUTE_FEATURES[policy["feature"]],
                             threshold=policy["threshold"], left_method=CANDIDATES[policy["left_method"]][0],
                             right_method=CANDIDATES[policy["right_method"]][0], penalty=penalty,
                             inner_means=json.dumps(inner, sort_keys=True)))
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def processing_group(f):
    # 閾値はスコアから選んでいない。加工形式の記述区分に固定。
    return np.where(f.BG_QI7_copy_share.gt(.5), "BG_copy", np.where(f.lab_round2_share.ge(.95) | f.lab_unique_share.lt(.6), "coarse", "continuous"))


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = dict(seed=SEED, route_features=ROUTE_FEATURES, fixed_candidates=CANDIDATES,
                penalties=PENALTIES, min_leaf=MIN_LEAF, main_scores_used=False,
                family_parameters_chosen_per_team=False)
    (HERE/"事前固定仕様.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2)+"\n")
    table = []
    for name, file, method in CANDIDATES:
        src = read(OLD/"集計"/file)
        one = src[src.method.eq(method)].copy()
        assert len(one) == 21, (name, method, len(one))
        one["family"] = name
        table.append(one)
    scores = pd.concat(table, ignore_index=True)
    scores.to_csv(OUT/"固定方式の予備戦得点.csv", index=False)
    teams = sorted(map(int, scores.team.unique()))
    kit = read(ROOT/"スターターキット/初版/participant_data/A_bg.csv")
    records = []
    sets = [("prelim", ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack", ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv", teams),
            ("main", ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力", ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv", None)]
    for phase, folder, bgpath, phase_teams in sets:
        bg = read(bgpath)
        if phase_teams is None:
            phase_teams = sorted(int(p.stem.rsplit("_", 1)[1]) for p in folder.glob("C_*.csv"))
        for k in phase_teams:
            c = read(folder/f"C_{k}.csv")
            q = read(folder/f"aia_challenge_{k}.csv")
            records.append(dict(phase=phase, team=k, **features(c, q, bg, kit)))
            print(json.dumps(dict(phase=phase, team=k, seconds=round(time.time()-start, 1))), flush=True)
    f = pd.DataFrame(records)
    assert len(f)==44 and np.isfinite(f[ROUTE_FEATURES].to_numpy()).all()
    f["processing_group"] = processing_group(f)
    f.to_csv(OUT/"公開Cの加工特徴_44組.csv", index=False)
    pd.DataFrame([dict(feature=n, definition=d) for n, d in {
        "lab_round2_share":"SBP/TG/HDL/ALTの小数2桁格子一致率の平均、絶対許容差1e-8",
        "lab_unique_share":"SBP/TG/HDL/ALTのユニーク値数/C行数の平均",
        "corr_gap_bg_sex":"性別ごとlogQI5相関の非対角差絶対値平均、TG/ALTはlog1p",
        "q_C_full_nn_median":"性別一致5属性/IQR最近傍距離の課題326行中央値",
        "time_long_share":"公開Cのtime>=9.5比率",
        "BG_QI7_copy_share":"同フェーズBGまたは旧キットBGとの公開QI7完全一致率の大きい方",
    }.items()]).to_csv(OUT/"ルータ6特徴の定義.csv", index=False)
    pf = f[f.phase.eq("prelim")].set_index("team")
    wide = scores.pivot(index="team", columns="family", values="r_all")[[c[0] for c in CANDIDATES]]
    summaries, outputs, policies = [], {}, []
    for scope, excluded in [("all20", [5]), ("processed18", [1, 2, 5])]:
        use = wide.index[~wide.index.isin(excluded)].to_numpy()
        x, y = pf.loc[use, ROUTE_FEATURES].to_numpy(float), wide.loc[use].to_numpy(float)
        result = nested(x, y, use)
        result["scope"] = scope
        result.to_csv(OUT/f"nested_LTO_{scope}.csv", index=False)
        group = nested(x, y, use, pf.loc[use, "processing_group"].to_numpy())
        group["scope"] = scope
        group.to_csv(OUT/f"加工群を外した検証_{scope}.csv", index=False)
        for kind, result_ in [("nested_LTO", result), ("processing_group_out", group)]:
            summaries.append(dict(scope=scope, evaluation=kind, n=len(use), baseline_r_all=float(y[:, 0].mean()),
                                  router_r_all=float(result_.r_all.mean()), delta=float(result_.delta.mean()),
                                  changed=int(result_.chosen_method.ne(CANDIDATES[0][0]).sum()),
                                  improved=int(result_.delta.gt(1e-12).sum()), worsened=int(result_.delta.lt(-1e-12).sum())))
        for j, (name, _, _) in enumerate(CANDIDATES):
            summaries.append(dict(scope=scope, evaluation="fixed_"+name, n=len(use), baseline_r_all=float(y[:, 0].mean()),
                                  router_r_all=float(y[:, j].mean()), delta=float((y[:, j]-y[:, 0]).mean())))
        summaries.append(dict(scope=scope, evaluation="per_target_oracle_not_validated", n=len(use), baseline_r_all=float(y[:, 0].mean()),
                              router_r_all=float(y.max(1).mean()), delta=float((y.max(1)-y[:, 0]).mean())))
        penalty, inner = choose_penalty(x, y)
        policy = fit_policy(x, y, penalty)
        policies.append(dict(scope=scope, **policy, feature_name=None if policy["feature"] is None else ROUTE_FEATURES[policy["feature"]], inner_scores=inner))
        # チームを入れ替えた偽の特徴にもnested選択を丸ごと適用し、探索の偶然差を診断。
        if scope == "processed18":
            rng = np.random.default_rng(SEED)
            null = []
            for repeat in range(50):
                perm = rng.permutation(len(x))
                shuffled = nested(x[perm], y, use)
                null.append(dict(repeat=repeat, delta=float(shuffled.delta.mean()), router_r_all=float(shuffled.r_all.mean())))
            pd.DataFrame(null).to_csv(OUT/"加工特徴の入れ替え対照_50回.csv", index=False)
        outputs[scope] = result
    pd.DataFrame(summaries).to_csv(OUT/"方式選択の比較.csv", index=False)
    (HERE/"全予備戦で学習したルータ_本戦予測ではない.json").write_text(json.dumps(policies, ensure_ascii=False, indent=2)+"\n")
    # 成功関連の記述はここで初めて得点と結合。特徴定義・ルータには使わない。
    joined = scores.merge(f[f.phase.eq("prelim")], on="team", validate="many_to_one")
    joined.to_csv(OUT/"加工特徴と固定方式_予備戦結合.csv", index=False)
    corr, group_scores = [], []
    for scope, excluded in [("all20", [5]), ("processed18", [1, 2, 5])]:
        subset = joined[~joined.team.isin(excluded)]
        baseline = wide[CANDIDATES[0][0]]
        for family, one in subset.groupby("family"):
            deltas = one.r_all.to_numpy()-one.team.map(baseline).to_numpy()
            for feature in ROUTE_FEATURES:
                corr.append(dict(scope=scope, family=family, feature=feature,
                                 spearman_delta=float(spearmanr(one[feature], deltas).statistic) if np.std(one[feature])>0 and np.std(deltas)>0 else np.nan,
                                 n=len(one), exploratory_only=True))
        for (group, family), one in subset.groupby(["processing_group", "family"]):
            group_scores.append(dict(scope=scope, processing_group=group, family=family, n=len(one), r_all=float(one.r_all.mean()),
                                     hit_difference_mean=float((one.hit_m-one.hit_c).mean()), p_m=float(one.p_m.mean()), p_c=float(one.p_c.mean())))
    pd.DataFrame(corr).to_csv(OUT/"特徴と利得の順位相関_記述用.csv", index=False)
    pd.DataFrame(group_scores).to_csv(OUT/"加工群別の固定方式.csv", index=False)
    # 本戦分布外診断。予備戦範囲内でも移行性能を証明しない。
    train = pf.loc[~pf.index.isin([1, 2, 5]), ROUTE_FEATURES]
    mainf = f[f.phase.eq("main")].set_index("team")
    scale = np.maximum(train.quantile(.75)-train.quantile(.25), .05)
    diag = []
    for team, row in mainf.iterrows():
        outside = [name for name in ROUTE_FEATURES if row[name]<train[name].min() or row[name]>train[name].max()]
        d = np.sqrt(np.mean(((train.to_numpy()-row[ROUTE_FEATURES].to_numpy(float))/scale.to_numpy())**2, axis=1))
        near = int(np.argmin(d))
        diag.append(dict(team=team, processing_group=row.processing_group, outside_count=len(outside), outside_features=";".join(outside),
                         nearest_prelim_team=int(train.index[near]), scaled_nearest_distance=float(d[near]), main_score_used=False))
    pd.DataFrame(diag).to_csv(OUT/"本戦の分布外診断.csv", index=False)
    record = dict(spec, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_sha256=HASHES,
                  input_data_used_for_prediction="C, challenge QI, phase BG, old-kit BG only",
                  truth_loaded_by_feature_extractor=False, official_main_scores_used=False,
                  main_predictions_written=False, submission_created=False, python=sys.executable,
                  python_version=platform.python_version(), seconds=time.time()-start)
    (HERE/"実行記録.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
