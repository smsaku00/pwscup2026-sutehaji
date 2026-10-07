"""主分析の凍結後に、数値一致と公式P01結果の記述を追加する。

公式本戦結果は方式ルータへ渡さない。ルータの設定を変更しない。
本戦公開結果は補正後値であり、raw R や対象ごとのphiは不明。
"""
from __future__ import annotations
import hashlib, importlib.util, io, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OUT = HERE/"集計"
BASE = HERE/"分析.py"
spec = importlib.util.spec_from_file_location("c_structure", BASE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
HASHES = {}


def read(path):
    data = Path(path).read_bytes()
    HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return pd.read_csv(io.BytesIO(data), float_precision="round_trip")


def approximate_copies(c, bg, tolerance=1e-8):
    # 最近傍を候補として、その5数値の絶対差を直接確認する。
    x, y = c[core.NUM].to_numpy(float), bg[core.NUM].to_numpy(float)
    scale = np.maximum(y.std(0, ddof=1), 1e-5)
    d = cdist(x/scale, y/scale)
    same = (c.sex.to_numpy()[:, None]==bg.sex.to_numpy()[None, :]) & (c.prefecture.to_numpy()[:, None]==bg.prefecture.to_numpy()[None, :])
    d[~same] = np.inf
    j = d.argmin(1)
    return np.isfinite(d[np.arange(len(c)), j]) & (np.abs(x-y[j])<=tolerance).all(1)


def marginal_share(values, reference, tolerance=1e-8):
    ref = np.sort(np.asarray(reference, float))
    x = np.asarray(values, float)
    index = np.searchsorted(ref, x)
    lo, hi = np.clip(index-1, 0, len(ref)-1), np.clip(index, 0, len(ref)-1)
    delta = np.minimum(np.abs(x-ref[lo]), np.abs(x-ref[hi]))
    return float((delta <= tolerance).mean())


def stable_time_span(q, c, bg):
    d = core.distance(q, c, bg)
    fifth = np.partition(d, 4, axis=1)[:, 4]
    near = d <= fifth[:, None]
    span = np.where(near, c.time.to_numpy()[None, :], -np.inf).max(1)-np.where(near, c.time.to_numpy()[None, :], np.inf).min(1)
    return float(np.median(span))


def run():
    start = time.time()
    f = read(OUT/"公開Cの加工特徴_44組.csv")
    methods = read(OUT/"固定方式の予備戦得点.csv")
    sets = [("prelim", ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack", ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
            ("main", ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力", ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]
    kit = read(ROOT/"スターターキット/初版/participant_data/A_bg.csv")
    precision, feature_perms, provenance = [], [], []
    for phase, folder, bgpath in sets:
        bg = read(bgpath)
        for team in f[f.phase.eq(phase)].team:
            c, q = read(folder/f"C_{team}.csv"), read(folder/f"aia_challenge_{team}.csv")
            a, b = approximate_copies(c, bg), approximate_copies(c, kit)
            precision.append(dict(phase=phase, team=team, exact_max_share=float(f[(f.phase==phase)&(f.team==team)].BG_QI7_copy_share.iloc[0]),
                                  approx_phase_share=float(a.mean()), approx_kit_share=float(b.mean()), approx_union_share=float((a|b).mean()),
                                  absolute_numeric_tolerance=1e-8))
            for col in core.LAB:
                provenance.append(dict(phase=phase, team=team, column=col,
                                       phase_BG_value_share=marginal_share(c[col], bg[col]),
                                       kit_BG_value_share=marginal_share(c[col], kit[col]),
                                       union_BG_value_share=marginal_share(c[col], np.r_[bg[col], kit[col]]),
                                       absolute_numeric_tolerance=1e-8,
                                       same_source_or_individual_link_not_proven=True))
            before = core.features(c, q, bg, kit)
            rng = np.random.default_rng(core.SEED+int(team))
            after = core.features(c.iloc[rng.permutation(len(c))], q.iloc[rng.permutation(len(q))], bg, kit)
            errors = [abs(before[name]-after[name]) for name in core.ROUTE_FEATURES]
            stable_before, stable_after = stable_time_span(q, c, bg), stable_time_span(q.iloc[::-1], c.iloc[::-1], bg)
            feature_perms.append(dict(phase=phase, team=team, max_router_feature_difference=float(max(errors)), feature_count=len(errors),
                                      first5_time_span_permutation_delta=float(after["q_C_near5_time_span_median"]-before["q_C_near5_time_span_median"]),
                                      boundary_included_span=stable_before, boundary_included_span_permutation_delta=stable_after-stable_before))
            assert max(errors)<1e-12
            assert stable_before==stable_after
    pd.DataFrame(precision).to_csv(OUT/"BG公開QI一致の数値許容差監査.csv", index=False)
    pd.DataFrame(feature_perms).to_csv(OUT/"ルータ特徴の行順変更監査.csv", index=False)
    pd.DataFrame(provenance).to_csv(OUT/"BG検査値の値集合一致_個人対応を示さない.csv", index=False)
    # 選択された方式の符号付き差も保存。clipped Rの平均だけで解釈しない。
    signature = ["chosen_method", "feature", "threshold", "left_method", "right_method", "penalty"]
    pf = f[f.phase.eq("prelim")].set_index("team")
    names = [candidate[0] for candidate in core.CANDIDATES]
    wide = methods.pivot(index="team", columns="family", values="r_all")[names]
    leakage = []
    score_rows = []
    for scope in ["all20", "processed18"]:
        selected = read(OUT/f"nested_LTO_{scope}.csv")
        use = selected.team.to_numpy()
        x, y = pf.loc[use, core.ROUTE_FEATURES].to_numpy(float), wide.loc[use].to_numpy(float)
        for i, team in enumerate(use):
            changed_y = y.copy()
            changed_y[i] = np.random.default_rng(core.SEED+int(team)).uniform(0, 1, len(names))
            rerun = core.nested(x, changed_y, use)
            original = selected[selected.team.eq(team)].iloc[0]
            changed = rerun[rerun.team.eq(team)].iloc[0]
            same = all((pd.isna(original[name]) and pd.isna(changed[name])) or original[name]==changed[name] for name in signature)
            assert same, (scope, team)
            leakage.append(dict(scope=scope, team=team, heldout_scores_replaced=True, policy_unchanged=same))
            one = methods[(methods.team==team)&(methods.family==original.chosen_method)].iloc[0]
            baseline = methods[(methods.team==team)&(methods.family==names[0])].iloc[0]
            score_rows.append(dict(scope=scope, team=int(team), chosen_method=original.chosen_method,
                                   r_all=float(one.r_all), baseline_r_all=float(baseline.r_all),
                                   hit_m=int(one.hit_m), hit_c=int(one.hit_c), signed_hit_difference=int(one.hit_m-one.hit_c),
                                   baseline_signed_hit_difference=int(baseline.hit_m-baseline.hit_c),
                                   r_unclipped=float((one.hit_m-one.hit_c)/max(163-one.hit_c, 1))))
    pd.DataFrame(leakage).to_csv(OUT/"外側対象得点の遮断監査.csv", index=False)
    pd.DataFrame(score_rows).to_csv(OUT/"nested選択の符号付き的中差.csv", index=False)
    # 公式本戦はここで初めて読んで記述結合。core.nestedは呼ばない。
    official_path = HERE.parent/"現状監査/第3回対象別方式.json"
    official_data = official_path.read_bytes()
    HASHES[str(official_path)] = hashlib.sha256(official_data).hexdigest()
    official = pd.DataFrame(json.loads(official_data))
    main_features = f[f.phase.eq("main")].merge(pd.DataFrame(feature_perms)[pd.DataFrame(feature_perms).phase.eq("main")][["team", "boundary_included_span"]], on="team")
    joined = official.merge(main_features, on="team", validate="one_to_one")
    joined["P01"] = joined.round3_method.str.startswith("P01")
    joined["final_positive"] = joined.round3_final.gt(0)
    joined.to_csv(OUT/"本戦第3回の方式と公開加工特徴_記述用.csv", index=False)
    p01 = joined[joined.P01]
    assert len(p01)==15
    descriptions = []
    feature_names = core.ROUTE_FEATURES+["distribution_gap_bg", "q_C_BG_nn_ratio_median", "q_C_near5_time_span_median", "boundary_included_span"]
    for positive, one in p01.groupby("final_positive"):
        for name in feature_names:
            descriptions.append(dict(final_positive=bool(positive), n=len(one), feature=name,
                                     median=float(one[name].median()), minimum=float(one[name].min()), maximum=float(one[name].max()),
                                     mean=float(one[name].mean()), main_router_tuning=False, score_channel="max(R_all,R_rare)-phi floored at zero"))
    pd.DataFrame(descriptions).to_csv(OUT/"P01補正後正群とゼロ群の特徴_記述用.csv", index=False)
    record = dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  analysis_code_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(), input_sha256=HASHES,
                  python=sys.executable, seconds=time.time()-start, outer_score_isolation_cases=len(leakage),
                  row_permutation_cases=len(feature_perms), main_score_only_descriptive=True,
                  original_router_redefined=False, submission_created=False)
    (HERE/"追加監査_実行記録.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({key:value for key,value in record.items() if key!="input_sha256"}, ensure_ascii=False), flush=True)


if __name__=="__main__":
    run()
