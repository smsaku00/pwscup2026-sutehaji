"""採点の意味、候補比較、公平な公開回答比較を再現する分析集約。

提出物は作らない。既存の集計と公開結果を読み、この分析内だけへ保存する。
"""
from __future__ import annotations
import hashlib, html, io, json, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import 分析 as core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / "集計"
EXCLUDE = [1, 2, 5]


def published_score(pred, truth):
    # 当時の公開Fには11年の回答がある。値を変更せず採点式で再評価する。
    pred = np.asarray(pred, float)
    assert np.isfinite(pred).all() and len(pred) == len(truth)
    m = truth.label.eq("member").to_numpy()
    r = truth.is_rare.to_numpy(bool)
    h = np.abs(pred - truth.time.to_numpy(float)) <= .5
    pm, pc = h[m].mean(), h[~m].mean()
    pmr, pcr = h[m & r].mean(), h[~m & r].mean()
    se = np.sqrt(pmr*(1-pmr)/13 + pcr*(1-pcr)/13)
    ra = np.clip((pm-pc)/max(1-pc, 1e-9), 0, 1)
    rr = np.clip((pmr-pcr-1.64*se)/max(1-pcr, 1e-9), 0, 1)
    return dict(r_all=ra, r_rare=rr, r_max=max(ra, rr), p_m=pm, p_c=pc,
                hit_m=int(h[m].sum()), hit_c=int(h[~m].sum()))


def source_table():
    paths = [ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv",
             ROOT/"スターターキット/初版/participant_data/A_bg.csv",
             ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv"]
    rows = []
    for p in paths:
        b = core.read(p)
        rows.append(dict(path=str(p.relative_to(ROOT)), rows=len(b), columns=len(b.columns),
                         sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    pd.DataFrame(rows).to_csv(OUT/"入力背景の実測.csv", index=False)


def official():
    rows, targets = [], []
    for n in [1, 2]:
        p = HERE/"参照"/f"第{n}回_detailed_results.html"
        content = p.read_text(encoding="utf-8")
        payload = json.loads(html.unescape(re.search(r"<pre>(.*?)</pre>", content, re.S).group(1)))
        totals = list(payload["aia_per_target"].values())
        assert len(totals) == 22
        rows.append(dict(round=n, aia_all_raw=payload["aia"]["r_time_all"],
                         aia_rare_raw=payload["aia"]["r_time_rare"], aia_power_final=float(np.mean(totals)),
                         p_m=payload["aia"]["p_m"], p_c=payload["aia"]["p_c"],
                         nonzero_targets=int(np.count_nonzero(totals))))
        targets += [dict(round=n, team=int(k), aia_power_final=v) for k,v in payload["aia_per_target"].items()]
    pd.DataFrame(rows).to_csv(OUT/"本戦公式結果の意味.csv", index=False)
    pd.DataFrame(targets).to_csv(OUT/"本戦公式AIAの対象別.csv", index=False)
    second = rows[1]
    best = next(v["aia_power_final"] for v in targets if v["round"] == 2 and v["team"] == 9)
    result = dict(round2_team9_fraction_of_AIA_power=(best/22)/second["aia_power_final"],
                  team9_alone_max_remaining_AIA_power_gain=(1-best)/22,
                  published_highest_raw_gap=.1432-second["aia_all_raw"],
                  aia_per_target_is_floored_max=True, raw_mean_is_mean_of_ratios=True)
    (HERE/"公式結果の解釈.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def competitors():
    td = next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    folder = ROOT/"予備戦/配布データ/結果/攻撃提出物"
    rows = []
    for attacker in [5, 9, 19, 23, 29]:
        p = next(folder.glob(f"*{attacker:02d}*F.zip"))
        core.HASHES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        with zipfile.ZipFile(p) as z:
            f = pd.read_csv(io.BytesIO(z.read("F_aia.csv")), float_precision="round_trip")
        for k,g in f.groupby("target_k"):
            t = core.read(td/f"aia_truth_{k}.csv").sort_values("challenge_row_id")
            q = g.sort_values("challenge_row_id")
            assert np.array_equal(q.challenge_row_id.to_numpy(), t.challenge_row_id.to_numpy())
            v = q.time_hat.to_numpy(float)
            rows.append(dict(attacker=attacker, team=k, **published_score(v,t),
                             long_answer=int((v>=9).sum()), early_answer=int((v<=.5).sum()),
                             zero_answer=int((v==0).sum()), answers_outside_main_0_10=int(((v<0)|(v>10)).sum())))
    table = pd.DataFrame(rows)
    table.to_csv(OUT/"予備戦公開上位回答_対象別.csv", index=False)
    common = sorted(set(table.team)-{1,2,5,9,19,23,29})
    records = []
    for scope, teams in [("original_targets", None), ("exclude_1_2", [k for k in set(table.team) if k not in [1,2]]), ("common_14", common)]:
        a = table if teams is None else table[table.team.isin(teams)]
        for attacker, g in a.groupby("attacker"):
            records.append(dict(scope=scope, attacker=attacker, targets=len(g),
                                **g[["r_all", "r_rare", "r_max", "p_m", "p_c"]].mean().to_dict()))
    pd.DataFrame(records).to_csv(OUT/"予備戦公開上位回答_公平比較.csv", index=False)
    return common


def candidate_tables():
    files = {
        "nearest_family": "予備戦_全設定と対象別.csv", "public_truth_learning": "公開正解学習_チームを分離した結果.csv",
        "adaptive_partial_density": "多尺度密度_対象別.csv", "membership_learning": "多尺度選別学習_対象別.csv",
        "density_time_window": "密度差時間窓_対象別.csv", "source_classifier": "分布識別_対象別.csv",
        "reverse_time": "時間逆探索_対象別.csv", "GenLRA": "GenLRA_AIA_対象別.csv",
        "one_to_one": "対応競合_対象別.csv", "no_C": "課題単独_対象別.csv"}
    families, inputs, unsupervised = [], [], []
    for family, name in files.items():
        table = core.read(OUT/name)
        table = table[~table.team.isin(EXCLUDE)]
        by = table.groupby("method").r_all.mean()
        best = by.idxmax()
        families.append(dict(family=family, settings=len(by), best_method=best, best_R=float(by.max()),
                             selection_uses_same_public_benchmark=True, targets=18))
        if family not in ["public_truth_learning", "membership_learning", "no_C"]:
            unsupervised.append(table[["team", "method", "r_all"]])
        inputs.append(table.assign(family=family))
    robust = core.read(OUT/"対応競合頑健性_反復.csv")
    robust = robust[robust.source.eq("real_C_full_reference")]
    by = robust.groupby("method").r_all.mean()
    families.append(dict(family="one_to_one_robust", settings=len(by), best_method=by.idxmax(), best_R=by.max(),
                         selection_uses_same_public_benchmark=True, targets=18))
    unsupervised.append(robust[["team", "method", "r_all"]])
    pd.DataFrame(families).sort_values("best_R", ascending=False).to_csv(OUT/"方法別の探索最高値.csv", index=False)
    # ラベル学習なしの候補だけで、外したチームの正解を使わず設定を選ぶ。
    all_u = pd.concat(unsupervised, ignore_index=True)
    assert not all_u.duplicated(["team", "method"]).any()
    p = all_u.pivot(index="team", columns="method", values="r_all")
    loto = []
    for team in p.index:
        train = p.drop(team).mean()
        selected = train.idxmax()
        loto.append(dict(team=team, selected_method=selected, r_all=p.loc[team, selected], train_mean=train[selected], settings=len(p.columns)))
    pd.DataFrame(loto).to_csv(OUT/"学習なし全候補_相手を外した設定選択.csv", index=False)
    return pd.DataFrame(families), pd.DataFrame(loto)


def uncertainty(common):
    base = core.read(OUT/"予備戦_全設定と対象別.csv")
    bg = core.read(OUT/"背景参照感度_対象別.csv")
    compare = {
        "S4_style75": base[base.method.eq("S4_log_iqr_soft/nn_gate0.75_0")].set_index("team").r_all,
        "continuous_one_to_one75": bg[bg.background.eq("prelim_5143") & bg.method.eq("fixed/continuous/one_to_one/gate0.75")].set_index("team").r_all,
        "continuous_sex_quota75": bg[bg.background.eq("prelim_5143") & bg.method.eq("fixed/continuous/sex_quota/gate0.75")].set_index("team").r_all}
    summaries = []
    for scope, teams in [("all20_exclude_self5", [k for k in compare["S4_style75"].index if k != 5]),
                         ("processed18_exclude1_2_self5", [k for k in compare["S4_style75"].index if k not in EXCLUDE]),
                         ("common14_with_published_attackers", common)]:
        for name, values in compare.items():
            v = values.loc[teams].to_numpy()
            diff = v-compare["S4_style75"].loc[teams].to_numpy()
            idx = np.random.default_rng(core.SEED).integers(0,len(v),(10000,len(v)))
            boot = diff[idx].mean(1)
            summaries.append(dict(scope=scope, method=name, targets=len(v), mean_R=v.mean(),
                                  diff_vs_S4=diff.mean(), exploratory_diff_ci_low=np.quantile(boot,.025),
                                  exploratory_diff_ci_high=np.quantile(boot,.975), improved=int((diff>0).sum()), worsened=int((diff<0).sum()),
                                  accounts_for_parameter_selection=False))
    pd.DataFrame(summaries).to_csv(OUT/"有力候補の同一対象比較.csv", index=False)


def run():
    source_table(); official(); common=competitors(); families,loto=candidate_tables(); uncertainty(common)
    (HERE/"集約監査_実行記録.json").write_text(json.dumps(dict(seed=core.SEED,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_sha256=core.HASHES,
        candidate_families=len(families), unsupervised_loto_mean_R=float(loto.r_all.mean()),
        historical_benchmark_independent=False, submission_created=False), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(families[["family", "settings", "best_R"]].to_string(index=False))
    print("all-unsupervised LOTO", float(loto.r_all.mean()))
    print(pd.read_csv(OUT/"有力候補の同一対象比較.csv").to_string(index=False))


if __name__ == "__main__":
    run()
