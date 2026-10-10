"""Join published aggregates only; never read participant-level data.

Run: python build_score_summary.py
Inputs and outputs are relative to this file. These are frozen-preliminary
counterfactuals, not actual main-round results or statistical intervals.
"""
from pathlib import Path
import csv
import json
import math

HERE = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run():
    utility = {int(r["team"]): r for r in read(HERE / "utility/all22_utility_bounds.csv")}
    anonymity = {int(r["team"]): r for r in read(HERE / "anonymity/main_rule_anonymity_bounds_all22.csv")}
    paired = {r["metric"]: r for r in read(HERE / "utility/self5_paired_scores.csv")}
    assert len(utility) == len(anonymity) == 22
    assert set(utility) == set(anonymity)
    assert max(abs(float(r["prelim_reproduction_difference"])) for r in paired.values()) < 1e-9
    rows = []
    for team, u in utility.items():
        a = anonymity[team]
        assert u["team_name"] == a["team_name"]
        r = {"prelim_cohort": team, "team_name": u["team_name"],
             "status": "missing_preliminary_C_and_official_scores" if team == 21 else "logical_bounds_missing_B_and_phi",
             "prelim_U": u["prelim_U"], "main_rule_U_lower": "", "main_rule_U_upper": "",
             "main_rule_U_C_only_upper": "", "main_rule_total_C_only_upper": "",
             "ledger_bounds_assumption": "Transferred old facets unchanged apart from stated ledger rounding; finite-precision parser effects excluded. C-only cap does not require this assumption.",
             "prelim_Anon": a["official_prelim_Anon"], "main_rule_Anon_lower": a["main_formula_Anon_lower"],
             "main_rule_Anon_upper": a["main_formula_Anon_upper"],
             "Anon_independent_of_phi": a["Anon_independent_of_phi"],
             "prelim_total": a["official_prelim_total"], "main_rule_total_lower": "", "main_rule_total_upper": "",
             "guaranteed_U_drop_at_least": "", "guaranteed_total_drop_at_least": "",
             "scope": "preliminary C, B, utility reference, background and historical attacks fixed; main scoring formula only"}
        if team != 21:
            assert math.isclose(float(u["prelim_U"]), float(a["official_prelim_U"]), abs_tol=1e-12)
            lo, hi = float(u["main_U_lower"]), float(u["main_U_upper"])
            c_cap = float(u["main_U_C_only_upper"])
            if team == 5:
                lo = hi = float(paired["U"]["main_recomputed"])
                r["status"] = "local_paired_U_recomputed_Anon_logical_bounds"
            alo, ahi = float(a["main_formula_Anon_lower"]), float(a["main_formula_Anon_upper"])
            tlo, thi = min(lo, alo), min(hi, ahi)
            assert 0 <= lo <= hi <= 1 and 0 <= alo <= ahi <= 1 and 0 <= tlo <= thi <= 1
            r.update(main_rule_U_lower=lo, main_rule_U_upper=hi,
                     main_rule_U_C_only_upper=c_cap, main_rule_total_C_only_upper=min(c_cap, ahi),
                     main_rule_total_lower=tlo, main_rule_total_upper=thi,
                     guaranteed_U_drop_at_least=max(0, float(u["prelim_U"])-1e-9-hi),
                     guaranteed_total_drop_at_least=max(0, float(a["official_prelim_total"])-1e-9-thi))
            if team in (1, 2):
                assert tlo == thi == 0
                r["status"] = "total_exact_zero_under_frozen_attacks_U_bounded"
        rows.append(r)
    with (HERE / "named_team_score_comparison_all22.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    def fmt(x):
        return "—" if x == "" else f"{float(x):.6f}"
    def interval(lo, hi):
        return "—" if lo == "" else fmt(lo) if float(lo) == float(hi) else f"{fmt(lo)}〜{fmt(hi)}"
    text = ["# チーム名付き得点一覧", "", "本戦ルールへの仮想変更。U・Anonとも大きいほど良い。範囲は論理的な包絡であり、予測・信頼区間・順位ではない。Uの下限0は欠損入力による広い範囲で、0点になるという予想ではない。", "", "| 予備戦ID | チーム | 予備戦U | 本戦式U（条件付き） | C単独U上限 | 予備戦Anon | 本戦式Anon | 予備戦総合 | 本戦式総合（条件付き） |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['prelim_cohort']} | {r['team_name']} | {fmt(r['prelim_U'])} | {interval(r['main_rule_U_lower'],r['main_rule_U_upper'])} | {fmt(r['main_rule_U_C_only_upper'])} | {fmt(r['prelim_Anon'])} | {interval(r['main_rule_Anon_lower'],r['main_rule_Anon_upper'])} | {fmt(r['prelim_total'])} | {interval(r['main_rule_total_lower'],r['main_rule_total_upper'])} |")
    text += ["", "- 自チーム5のUのみ、保有する元B・C・utility referenceを用いたローカル再計算値。公式再採点ではなく、実行環境の留保はutilityの説明を参照。", "- 他チームのUはBと採点参照値が不足するため範囲。台帳付きの範囲は、同じ実行式の旧成分が読込精度差で変わらない条件付き。予備戦defaultと本戦round_tripの有限精度差は、この範囲には含めていない。", "- C単独U上限は上の台帳転用条件に依存しない。総合のC単独上限もCSVに併記した。上限順を得点順位とは扱わない。", "- Anonは予備戦の全攻撃出力・真値・参加者を固定し、未取得の予備戦世界固有のAIA補正φ∈[0,1]を動かした範囲。本戦の別コホートのφは代入していない。", "- 総合=min(U,Anon)。予備戦0.1・本戦0.9による大会最終合算ではない。", "- 1・2は総合0に確定。17「あれ」は総合0.858909から上限0.750000へ下がる。19・29のU上限の低下だけでは総合点低下を確定できない。", "- bot035は未提出・得点欠損のまま。0点に置換しない。", "- 表は小数6桁の表示。CSVは計算精度を保持し、公表Uの丸めには±1e-9の保守的幅を入れた。", ""]
    (HERE / "チーム名付き得点一覧.md").write_text("\n".join(text), encoding="utf-8")
    result = {"rows": len(rows), "submitted": 21, "missing": [21], "locally_recomputed_U": [5],
              "exact_total_zero": [1, 2], "guaranteed_U_drop": [r["prelim_cohort"] for r in rows if r["guaranteed_U_drop_at_least"] != "" and float(r["guaranteed_U_drop_at_least"]) > 1e-8],
              "guaranteed_total_drop": [r["prelim_cohort"] for r in rows if r["guaranteed_total_drop_at_least"] != "" and float(r["guaranteed_total_drop_at_least"]) > 1e-8],
              "formula": "total bounds = [min(U_lower,Anon_lower), min(U_upper,Anon_upper)]",
              "unobserved_data_imputed": False}
    assert result["guaranteed_total_drop"] == [17]
    (HERE / "score_summary_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    run()
