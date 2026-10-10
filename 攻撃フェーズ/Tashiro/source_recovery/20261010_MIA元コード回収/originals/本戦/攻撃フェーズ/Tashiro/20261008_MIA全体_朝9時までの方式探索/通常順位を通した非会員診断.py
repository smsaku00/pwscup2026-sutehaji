"""公開Qを固定較正・候補通常順位・共同crossに通す集約診断。

ルート .venv/bin/python -B で --write-plan を先に実行し、引数なしで計算する。
公開非会員の通常分岐の上位閾値超過であり、公式FPR/所属accuracyではない。
個人行・ID・公式回答・NPZは出力しない。B/Qは既知normal875と固定較正にだけ使う。
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import 共通 as u

HERE, ROOT = u.HERE, u.ROOT
CONFIGS = [dict(name="flat500", local=500, parent=None, parent_strength=None),
           dict(name="flat300", local=300, parent=None, parent_strength=None),
           dict(name="county300_sex_age100", local=300, parent="sex_age", parent_strength=100)]
HEADS = (10, 12)
PANELS = ("all_public_Q_normal_branch", "excluding_self_private_QI7_normal_branch")
METHOD = {
    "version": 1, "configs": CONFIGS, "cross": .1, "heads": list(HEADS), "panels": list(PANELS),
    "calibration": "全Q所有者を1人ずつ丸ごと保留。各targetのcontrolsからそのownerとtarget自身とB行を除外し、同じCDFでB/heldQを変換。",
    "full_contrast": "fullと毎owner除外、予備戦のみ1/2同時除外も既存順位cacheと照合。full/同時除外はB対照のみ。",
    "rank_extension": "R_ref(x)=(#ref<x + .5*#ref=x + .5)/n。ref観測点でrankdata平均/nに一致。",
    "raw_tie": ".001*R_raw_B1049(x)。Qを追加してB1049の参照を再fitしない。",
    "cal_value": ".999*calCDF(x)+.001*R_raw_B1049(x)。",
    "normal_rank": "calBの既知normal875を固定参照にB/QのRを計算。Qを参照に追加しない。",
    "rank_beyond_one": "最大1+.5/nの定義上の挿入順位をそのまま保存し、確率と呼ばない。",
    "cross_scope": "各owner保留・methodについて全20/22targetsを共同に計算。Qowner=targetのcomponentもcrossに必要。",
    "threshold": "Bnormal cross scoreのkth最大。Q>=thresholdおよびQ>thresholdを両方数える。Bのties込みhead行数も保存。",
    "self_private_panel": "selfownerのみ、Bの既知private QI7との一致163行を除く。その他ownerは同じQ集合。",
    "private_zero": "既知privateを固定0分岐にするなら正のhead閾値を超えない。主パネルにはその分岐を適用せず通常分岐を明示。",
    "rare_policy": "Qにis_rareタグがないため他のrare policyは適用せず、通常分岐限定。",
    "eligible_target": "集計ではowner=targetを除外。予備戦primary集約はtarget1/2を除外、本戦22targets。cross自体は常に全targets。",
    "aggregation": "primary targetsでowner内target率平均、次にowner等重み。対象別もowner等重み。分母row×target exposureは独立患者数ではない。",
    "independent_checks": "Counter+bisectのRとnumpy searchsorted Rを全点照合。ref点はscipy rankdata平均/nとも照合。Bscoreは既存除外stress cache全scenario/methodで差0。",
    "inputs": "凍結raw cache、凍結B順位感度cache、u.setupによるB/A/public C/Q。所属GTと公式採点結果は読まない。",
    "limits": [
        "通常分岐・B上位閾値に対する公開非会員超過率であり、official B真非会員FPRや会員判別accuracyではない。",
        "Q分布とB正常非会員分布は異なる。Bnormalのrank参照自体は所属未知のmixed candidates。",
        "開発全体の外部testではなく、保証・confidence interval・手法選択には使わない。",
    ],
    "membership_labels_used": False, "official_scores_used": False, "selection": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def stamp():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def source_paths():
    result = []
    for phase in ("prelim", "main"):
        distributed = u.base.child(ROOT / ("予備戦" if phase == "prelim" else "本戦"), "配布データ")
        own = next(distributed.glob("PWSCup2026_prelim_team05_*" if phase == "prelim" else "PWSCup2026_final_team28_*"))
        targets = distributed / "PWSCup2026_prelim_attack" if phase == "prelim" else HERE.parent / "20260928_配布データ基本分析/入力"
        background = distributed / "PWSCup2026_prelim_common/A_bg.csv" if phase == "prelim" else own / "A_bg.csv"
        cs = sorted(targets.glob("C_*.csv"))
        result.extend([own / "B_self.csv", background] + cs + [targets / f"aia_challenge_{p.stem.split('_')[1]}.csv" for p in cs])
        result.extend([HERE / "非会員課題の丸ごと保留結果" / phase / "候補と公開課題の生スコア.npz",
                       HERE / "公開課題の除外に対する順位感度結果" / phase / "内部順位だけ.npz"])
    return sorted(set(p.resolve() for p in result))


def frozen_hashes():
    files = [HERE / "共通.py", u.OLD / "分析.py", u.CAL / "分析.py", HERE / "公開課題の除外に対する順位感度.py"]
    files.extend(p for p in (HERE / "公開課題の除外に対する順位感度結果").rglob("*") if p.is_file())
    return {str(p.resolve()): sha(p) for p in sorted(files)}


def rank_extend(values, reference):
    ordered = np.sort(np.asarray(reference, float))
    values = np.asarray(values, float)
    left = np.searchsorted(ordered, values, side="left")
    right = np.searchsorted(ordered, values, side="right")
    return (left + .5 * (right - left) + .5) / len(ordered)


def counter_rank_extend(values, reference):
    counts = Counter(np.asarray(reference, float).tolist())
    keys = sorted(counts)
    prefix = [0]
    for value in keys:
        prefix.append(prefix[-1] + counts[value])
    n = len(reference)
    return np.array([(prefix[bisect.bisect_left(keys, float(value))] + .5 * counts.get(float(value), 0) + .5) / n for value in values])


def cross(values):
    return np.column_stack([values[:, j] - .1 * np.median(np.delete(values, j, axis=1), axis=1)
                            for j in range(values.shape[1])])


def calibrate(values, test_frame, null, controls, config):
    global_cdf = u.cal.empirical_cdf(values, null)
    local, n = u.cal.group_stats(values, test_frame, null, controls, ["prefecture", "sex"])
    wl = n / (n + config["local"])
    fallback = global_cdf
    if config["parent"]:
        parent, m = u.cal.group_stats(values, test_frame, null, controls, ["sex", "age_bin"])
        wp = m / (m + config["parent_strength"])
        fallback = wp * parent + (1 - wp) * global_cdf
    return wl * local + (1 - wl) * fallback


def phase_run(phase, checks):
    data = u.setup(phase)
    raw_path = HERE / "非会員課題の丸ごと保留結果" / phase / "候補と公開課題の生スコア.npz"
    baseline_path = HERE / "公開課題の除外に対する順位感度結果" / phase / "内部順位だけ.npz"
    raw_cache, baseline = np.load(raw_path), np.load(baseline_path)
    teams = data["teams"]
    assert list(raw_cache["teams"]) == list(baseline["teams"]) == teams
    owner = np.r_[np.full(len(data["b"]), -1), *[np.full(len(q), t) for t, q in data["qs"].items()]]
    assert np.array_equal(owner, raw_cache["query_owner"])
    assert int(raw_cache["b_rows"]) == len(data["b"]) == 1049
    assert int(data["normal"].sum()) == 875
    raw = raw_cache["raw"]
    assert raw.shape == (len(data["query"]), len(teams)) and np.isfinite(raw).all()
    raw_ranks = np.empty_like(raw)
    for j, target in enumerate(teams):
        reference = raw[:1049, j]
        raw_ranks[:, j] = rank_extend(raw[:, j], reference)
        independent = counter_rank_extend(raw[:, j], reference)
        error = float(np.max(abs(raw_ranks[:, j] - independent)))
        scipy_error = float(np.max(abs(raw_ranks[:1049, j] - rankdata(reference, method="average") / 1049)))
        assert error == scipy_error == 0
        checks["raw_counter_queries"] += len(raw)
        checks["raw_ref_scipy_queries"] += 1049
    scenarios = [("full", ())] + [(f"without_owner_{t}", (t,)) for t in sorted(data["qs"])]
    if phase == "prelim":
        scenarios.append(("without_owners_1_2", (1, 2)))
    expected = {f"{label}::{config['name']}__cross0.1" for label, _ in scenarios for config in CONFIGS}
    assert set(baseline["names"]) == expected
    baseline_index = {name: i for i, name in enumerate(baseline["names"])}
    private_keys = pd.MultiIndex.from_frame(data["b"].loc[data["private"], list(u.base.QI7)])
    detail, contrasts = [], []
    for label, removed in scenarios:
        held_owner = removed[0] if len(removed) == 1 else None
        held_indices = np.flatnonzero(owner == held_owner) if held_owner is not None else np.array([], int)
        q = data["query"].iloc[held_indices]
        test_frame = pd.concat([data["b"], q], ignore_index=True)
        controls = {t: ii[~np.isin(owner[ii], removed)] for t, ii in data["controls"].items()}
        for t, ii in controls.items():
            assert not np.isin(owner[ii], list(removed) + [t, -1]).any()
        private = np.zeros(len(q), bool)
        if held_owner == data["self_id"]:
            private = pd.MultiIndex.from_frame(q[list(u.base.QI7)]).isin(private_keys)
            assert int(private.sum()) == 163 and len(q) == 326
        for config in CONFIGS:
            method = config["name"] + "__cross0.1"
            b_rank, q_rank = [], []
            for j, target in enumerate(teams):
                indices = controls[target]
                values = np.r_[raw[:1049, j], raw[held_indices, j]]
                cal = calibrate(values, test_frame, raw[indices, j], data["query"].iloc[indices], config)
                cal = .999 * cal + .001 * np.r_[raw_ranks[:1049, j], raw_ranks[held_indices, j]]
                reference = cal[:1049][data["normal"]]
                extended = rank_extend(cal, reference)
                independent = counter_rank_extend(cal, reference)
                assert float(np.max(abs(extended - independent))) == 0
                normal_values = extended[:1049][data["normal"]]
                scipy_values = rankdata(reference, method="average") / 875
                assert float(np.max(abs(normal_values - scipy_values))) == 0
                assert (extended >= .5 / 875).all() and (extended <= 1 + .5 / 875).all()
                b_rank.append(normal_values); q_rank.append(extended[1049:])
                checks["cal_counter_queries"] += len(cal)
                checks["cal_ref_scipy_queries"] += 875
            b_rank, q_rank = np.column_stack(b_rank), np.column_stack(q_rank)
            b_score, q_score = cross(b_rank), cross(q_rank)
            old = baseline["scores"][baseline_index[label + "::" + method]]
            error = float(np.max(abs(b_score - old)))
            assert error == 0, (phase, label, method, error)
            contrasts.append(dict(phase=phase, scenario=label, method=method, targets=len(teams), normal_rows=875,
                                  b_score_max_absolute_error=error, arbitrary_Q_does_not_refit_B_rank=True))
            if held_owner is None:
                continue
            for j, target in enumerate(teams):
                if target == held_owner:
                    continue
                primary = phase == "main" or target not in (1, 2)
                for k in HEADS:
                    threshold = float(np.sort(b_score[:, j])[-k])
                    assert threshold > 0
                    b_ge, b_gt = int((b_score[:, j] >= threshold).sum()), int((b_score[:, j] > threshold).sum())
                    assert b_ge >= k and b_gt < k
                    for panel in PANELS:
                        mask = np.ones(len(q), bool) if panel == PANELS[0] else ~private
                        values = q_score[mask, j]
                        assert len(values) > 0
                        ge, gt = int((values >= threshold).sum()), int((values > threshold).sum())
                        detail.append(dict(phase=phase, null_owner=held_owner, target=target, method=method,
                            panel=panel, k=k, included_primary_aggregate=primary, q_rows=len(values),
                            q_known_private_rows_in_panel=int((private & mask).sum()),
                            b_threshold=threshold, b_head_rows_ge=b_ge, b_head_rows_gt=b_gt,
                            q_exceed_rows_ge=ge, q_exceed_rows_gt=gt,
                            q_exceed_fraction_ge=ge / len(values), q_exceed_fraction_gt=gt / len(values),
                            q_private_normal_exceed_rows_ge=int((private & mask & (q_score[:, j] >= threshold)).sum()),
                            q_private_normal_exceed_rows_gt=int((private & mask & (q_score[:, j] > threshold)).sum()),
                            known_private_fixed_zero_exceed_rows=0,
                            q_raw_rank_above_one_rows=int((raw_ranks[held_indices[mask], j] > 1).sum()),
                            q_cal_rank_above_one_rows=int((q_rank[mask, j] > 1).sum()),
                            q_raw_rank_max=float(raw_ranks[held_indices[mask], j].max()),
                            q_cal_rank_max=float(q_rank[mask, j].max()),
                            normal_branch_only=True, official_FPR=False, membership_accuracy_unknown=True))
        u.log(stage="held_public_Q_rank_extension", phase=phase, scenario=label)
    return pd.DataFrame(detail), pd.DataFrame(contrasts)


def aggregate(detail):
    eligible = detail[detail.included_primary_aggregate].copy()
    common = ["phase", "method", "panel", "k"]
    owners = eligible.groupby(common + ["null_owner"], sort=True).agg(
        eligible_targets=("target", "nunique"), q_rows_per_target=("q_rows", "first"),
        mean_normal_exceed_fraction_ge=("q_exceed_fraction_ge", "mean"),
        mean_normal_exceed_fraction_gt=("q_exceed_fraction_gt", "mean"),
        min_target_exceed_fraction_ge=("q_exceed_fraction_ge", "min"),
        max_target_exceed_fraction_ge=("q_exceed_fraction_ge", "max"),
        q_row_target_exposures=("q_rows", "sum"), q_exceed_exposures_ge=("q_exceed_rows_ge", "sum"),
        q_exceed_exposures_gt=("q_exceed_rows_gt", "sum")).reset_index()
    targets = eligible.groupby(common + ["target"], sort=True).agg(
        owners=("null_owner", "nunique"), mean_owner_exceed_fraction_ge=("q_exceed_fraction_ge", "mean"),
        mean_owner_exceed_fraction_gt=("q_exceed_fraction_gt", "mean"),
        min_owner_exceed_fraction_ge=("q_exceed_fraction_ge", "min"),
        max_owner_exceed_fraction_ge=("q_exceed_fraction_ge", "max")).reset_index()
    phases = owners.groupby(common, sort=True).agg(
        owners=("null_owner", "nunique"), owner_equal_mean_exceed_fraction_ge=("mean_normal_exceed_fraction_ge", "mean"),
        owner_equal_mean_exceed_fraction_gt=("mean_normal_exceed_fraction_gt", "mean"),
        min_owner_exceed_fraction_ge=("mean_normal_exceed_fraction_ge", "min"),
        max_owner_exceed_fraction_ge=("mean_normal_exceed_fraction_ge", "max"),
        q_row_target_exposures=("q_row_target_exposures", "sum"),
        q_exceed_exposures_ge=("q_exceed_exposures_ge", "sum"), q_exceed_exposures_gt=("q_exceed_exposures_gt", "sum")).reset_index()
    for table in (owners, targets, phases):
        table["normal_branch_only"] = True
        table["official_FPR"] = False
        table["membership_accuracy_unknown"] = True
    return owners, targets, phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=HERE / "通常順位を通した非会員診断結果")
    args = parser.parse_args()
    out, code = args.output_dir.resolve(), Path(__file__).resolve()
    if args.write_plan:
        assert not out.exists()
        sources = {str(p): sha(p) for p in source_paths()}
        assert not any(word in p for p in sources for word in ("pool_ids", "正解データ", "result.txt"))
        out.mkdir()
        (out / "方法.json").write_text(dump(dict(created_jst=stamp(), method=METHOD,
            code_sha256_before_execution=sha(code), source_sha256_before_execution=sources,
            frozen_artifact_sha256=frozen_hashes())), encoding="utf-8")
        print(dump(dict(stage="plan_fixed_before_execution", path=str(out / "方法.json"))), end="")
        return
    assert Path(sys.executable).parent == ROOT / ".venv/bin"
    assert out.is_dir() and {p.name for p in out.iterdir()} == {"方法.json"}
    plan = json.loads((out / "方法.json").read_text())
    assert plan["method"] == METHOD and plan["code_sha256_before_execution"] == sha(code)
    assert plan["frozen_artifact_sha256"] == frozen_hashes()
    for path, expected in plan["source_sha256_before_execution"].items():
        assert sha(path) == expected
    started = time.perf_counter()
    checks = dict(raw_counter_queries=0, raw_ref_scipy_queries=0, cal_counter_queries=0, cal_ref_scipy_queries=0,
                  maximum_rank_extension_error=0.0, maximum_B_contrast_error=0.0)
    parts, contrasts = [], []
    for phase in ("prelim", "main"):
        detail, contrast = phase_run(phase, checks)
        parts.append(detail); contrasts.append(contrast)
    detail, contrast = pd.concat(parts, ignore_index=True), pd.concat(contrasts, ignore_index=True)
    assert len(detail) == 10608 and len(contrast) == 141
    assert detail.q_exceed_rows_gt.le(detail.q_exceed_rows_ge).all()
    assert detail.q_exceed_rows_ge.le(detail.q_rows).all()
    assert not (detail.null_owner == detail.target).any()
    owner, target, phase = aggregate(detail)
    outputs = {"対象所有者別の通常閾値超過.csv": detail, "所有者別の通常閾値超過.csv": owner,
               "対象別の通常閾値超過.csv": target, "フェーズ別の通常閾値超過.csv": phase,
               "B順位対照.csv": contrast}
    for name, table in outputs.items():
        assert not any(column in table for column in ("record_id", "challenge_row_id", "pool_id"))
        table.to_csv(out / name, index=False, float_format="%.17g")
    checks.update(b_contrasts=len(contrast), target_owner_rows=len(detail), owner_rows=len(owner),
                  target_rows=len(target), phase_rows=len(phase), no_GT=True,
                  raw_extension_upper_bound=1 + .5 / 1049, cal_extension_upper_bound=1 + .5 / 875)
    (out / "独立順位検証.json").write_text(dump(checks), encoding="utf-8")
    assert not any(word in p for p in u.base.INPUT_HASHES for word in ("pool_ids", "正解データ", "result.txt"))
    assert set(u.base.INPUT_HASHES) <= set(plan["source_sha256_before_execution"])
    for path, expected in plan["source_sha256_before_execution"].items():
        assert sha(path) == expected
    assert plan["frozen_artifact_sha256"] == frozen_hashes()
    assert plan["code_sha256_before_execution"] == sha(code)
    record = dict(completed_jst=stamp(), seconds=time.perf_counter() - started,
        python=sys.executable, python_version=platform.python_version(), code_sha256=sha(code),
        input_sha256=plan["source_sha256_before_execution"],
        output_sha256={p.name: sha(p) for p in sorted(out.iterdir())}, output_hash_scope="実行記録.json自身を除く。",
        plan_fixed_before_execution=True, frozen_artifacts_unchanged=True, membership_labels_used=False,
        official_scores_used=False, no_GT=True, normal_branch_only=True, official_FPR=False,
        model_or_threshold_selection=False, individual_rows_or_IDs_exported=False, NPZ_exported=False,
        meaning="通常分岐・B上位閾値に対する公開非会員超過率。Q分布・mixed B順位fit・開発内評価の限界がある。")
    (out / "実行記録.json").write_text(dump(record), encoding="utf-8")
    print(dump(dict(stage="normal_rank_public_Q_diagnosis_complete", rows=len(detail), b_contrasts=len(contrast),
                    seconds=record["seconds"])), end="")


if __name__ == "__main__":
    main()
