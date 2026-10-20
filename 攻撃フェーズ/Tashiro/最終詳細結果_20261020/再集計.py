#!/usr/bin/env python3
"""紙6ページの転記・画像・第8回公式結果を検証し、寄与を再集計する。

実行: python 再集計.py
出力: このフォルダの検証と集計.json。標準ライブラリのみ。ネットワーク不要。
表示値の演算は丸め区間で検査し、公式JSONの未丸め値と区別する。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP, getcontext
from fractions import Fraction
from pathlib import Path

getcontext().prec = 40
D = Decimal
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ZERO, ONE = D(0), D(1)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_float=D)


def git_blob_sha1(content: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()


def s(value):
    """Decimalの全桁を文字列にし、検証出力を再現可能にする。"""
    if isinstance(value, D):
        return str(value)
    raise TypeError(type(value).__name__)


def mean(values):
    values = list(values)
    return sum(values, ZERO) / len(values)


def box(value):
    """印刷値の四捨五入区間。端点を含める保守的な検査。"""
    value = D(value)
    half_unit = D(10) ** value.as_tuple().exponent / 2
    return value - half_unit, value + half_unit


def add_box(left, right):
    return left[0] + right[0], left[1] + right[1]


def sub_box(left, right):
    return left[0] - right[1], left[1] - right[0]


def mean_box(values):
    pairs = [box(v) for v in values]
    return mean(x[0] for x in pairs), mean(x[1] for x in pairs)


def overlaps(left, right):
    return max(left[0], right[0]) <= min(left[1], right[1])


def clip(value, low=ZERO, high=ONE):
    return max(low, min(high, value))


def rank_number(value):
    return int(re.match(r"\d+", value).group())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "検証と集計.json")
    args = parser.parse_args()
    data = read_json(HERE / "読み取りデータ.json")
    manifest = read_json(HERE / "画像対応と出典.json")
    pages = data["pages"]
    official_path = REPO / "攻撃フェーズ/MIA/8回目/採点結果/解析結果.json"
    official = read_json(official_path)
    scores = read_json(official_path.with_name("scores.json"))
    checks = []

    def check(name, valid, **details):
        checks.append({"name": name, "status": "PASS" if valid else "FAIL", **details})

    expected_counts = {"main_attack": 22, "preliminary_attack": 20,
                       "main_anonymity": 22, "preliminary_defense": 20, "utility": 24}
    for page, count in expected_counts.items():
        check(f"row_count:{page}", len(pages[page]["rows"]) == count,
              expected=count, observed=len(pages[page]["rows"]))
    check("six_pages", sorted(p["page_number"] for p in pages.values()) == list(range(1, 7)))
    for name, page in pages.items():
        check(f"no_unresolved_cells:{name}", not page.get("uncertain_cells", page.get("uncertainties", [])))

    image_results = []
    for item in manifest["images"]:
        raw = (HERE / item["renamed_path"]).read_bytes()
        observed = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                    "git_blob_sha1": git_blob_sha1(raw)}
        check(f"image_integrity:{item['renamed_path']}", all(observed[k] == item[k] for k in observed), **observed)
        check(f"image_page_binding:{item['section']}", pages[item["section"]]["source"] == item["renamed_path"]
              and pages[item["section"]]["page_number"] == item["page"])
        image_results.append({"path": item["renamed_path"], **observed})
    check("six_unique_images", len(image_results) == 6 and len({x["sha256"] for x in image_results}) == 6)
    for source in data["metadata"]["sources"]:
        observed = git_blob_sha1((REPO / source["path"]).read_bytes())
        check(f"reference_blob:{source['path']}", observed == source["git_blob_sha1"],
              expected=source["git_blob_sha1"], observed=observed)

    with (REPO / "攻撃フェーズ/参加チーム対応表.csv").open(encoding="utf-8-sig", newline="") as f:
        mapping = {int(r["登録順"]): r for r in csv.DictReader(f)}
    self_no = data["metadata"]["presentation_no"]
    check("own_identity", int(mapping[self_no]["本戦コホート番号"]) == data["metadata"]["main_cohort_id"]
          and int(mapping[self_no]["予備戦コホート番号"]) == data["metadata"]["preliminary_cohort_id"])
    main_nos = set(mapping) - {self_no}
    # 紙の予備戦表には本戦から参加したNo.18・No.23が載らない。
    prelim_nos = main_nos - {18, 23}
    for page in ("main_attack", "main_anonymity", "preliminary_attack", "preliminary_defense"):
        rows = pages[page]["rows"]
        expected_nos = main_nos if page.startswith("main") else prelim_nos
        nos = [int(r.get("no", r.get("presentation_no"))) for r in rows]
        check(f"identities_unique_complete:{page}", len(nos) == len(set(nos)) and set(nos) == expected_nos)
        for row in rows:
            identity = row["identity"]
            no = int(row.get("no", row.get("presentation_no")))
            record = mapping[no]
            prelim_id = int(record["予備戦コホート番号"]) if record["予備戦コホート番号"] else None
            check(f"identity:{page}:{no}", identity["presentation_no"] == no
                  and identity["main_cohort_id"] == int(record["本戦コホート番号"])
                  and identity["preliminary_cohort_id"] == prelim_id
                  and identity["official_team_name"] == record["公式チーム名"])
            rare_n = int(row.get("rare_members", row.get("rare_member_count")))
            count = int(row.get("candidate_members", row.get("member_count")))
            check(f"membership_count:{page}:{no}", 0 <= rare_n <= count)
            t = D(row.get("mia_t", row.get("mia_t_j")))
            check(f"mia_missing_rare:{page}:{no}", (row["mia_rare"] == "—") == (rare_n == 0))
            expected_t = D(row["mia_all"]) if rare_n == 0 else max(D(row["mia_all"]), D(row["mia_rare"]))
            check(f"mia_max:{page}:{no}", t == expected_t)

    official_aia = official["official_breakdown"]["aia_per_target"]
    aia_matches = []
    for row in pages["main_attack"]["rows"]:
        cid = str(row["identity"]["main_cohort_id"])
        value = official_aia[cid]
        rounded = format(value.quantize(D("0.001"), rounding=ROUND_HALF_UP), ".3f")
        match = rounded == row["aia_r_minus_phi"]
        check(f"main_aia_round8:{cid}", match, official_full_precision=value,
              official_rounded=rounded, paper=row["aia_r_minus_phi"])
        aia_matches.append({"cohort_id": int(cid), "official_full_precision": value,
                            "paper": row["aia_r_minus_phi"], "match": match})
        check(f"aia_subtraction_rounding:{cid}", overlaps(sub_box(box(row["aia_r"]), box(row["target_phi"])),
                                                       box(row["aia_r_minus_phi"])))
        check(f"main_target_total_rounding:{cid}", overlaps(add_box(box(row["mia_t"]), box(row["aia_r_minus_phi"])),
                                                          box(row["total"])))
        check(f"main_rank_range:{cid}", 1 <= int(row["target_rank"]) <= int(row["target_rank_denominator"]) == 22)
    check("main_aia_all_targets_covered", {str(r["identity"]["main_cohort_id"]) for r in pages["main_attack"]["rows"]}
          == set(official_aia))
    for row in pages["preliminary_attack"]["rows"]:
        no = row["presentation_no"]
        check(f"prelim_target_total_rounding:{no}", overlaps(add_box(box(row["mia_t"]), box(row["aia_R"])), box(row["total"])))
        check(f"prelim_rank_range:{no}", 1 <= rank_number(row["target_rank"]) <= 20 and row["target_rank"].endswith("/ 20"))
    for page, all_key, rare_key, r_key in (("main_anonymity", "aia_R_all", "aia_R_rare", "aia_R"),
                                         ("preliminary_defense", "aia_r_all", "aia_r_rare", "aia_r")):
        for row in pages[page]["rows"]:
            no = row.get("no", row.get("presentation_no"))
            check(f"defense_aia_max:{page}:{no}", D(row[r_key]) == max(D(row[all_key]), D(row[rare_key])))

    summary = pages["summary"]
    main_def = pages["main_anonymity"]["printed_summary"]
    pre_def = pages["preliminary_defense"]["anonymity"]
    for key in ("attacker_count", "mia_component", "beta_raw", "beta_chance", "beta", "aia_component", "anon"):
        check(f"summary_main_defense_duplicate:{key}", summary["main_anonymity"][key] == main_def[key])
    for skey, pkey in (("max_aia_r", "maximum_R"), ("target_phi", "phi"), ("max_aia_attacker", "maximum_R_attacker")):
        check(f"summary_main_defense_duplicate:{skey}", summary["main_anonymity"][skey] == main_def[pkey])
    utility = {r["metric"]: r for r in pages["utility"]["rows"]}
    for row in summary["round_scores"]:
        mode = "main" if row["round"] == "本戦" else "preliminary"
        if row["metric"] == "有用性 U":
            for k in ("score", "median", "rank"):
                check(f"summary_utility_duplicate:{mode}:{k}", row[k] == utility["U"][f"{mode}_{k}"])
        elif row["metric"] == "匿名性 Anon":
            check(f"summary_anonymity_duplicate:{mode}", row["score"] == (main_def if mode == "main" else pre_def)["anon"])
        else:
            check(f"overall_min:{mode}", D(row["score"]) == min(D(utility["U"][f"{mode}_score"]),
                  D((main_def if mode == "main" else pre_def)["anon"])))
    for row in summary["final_score_rows"]:
        calc = D("0.1") * D(row["preliminary"]) + D("0.9") * D(row["main"])
        source_box = (D("0.1") * box(row["preliminary"])[0] + D("0.9") * box(row["main"])[0],
                      D("0.1") * box(row["preliminary"])[1] + D("0.9") * box(row["main"])[1])
        check(f"final_weight_rounding:{row['metric']}", overlaps(source_box, box(row["final"])),
              arithmetic_using_printed_values=calc, paper=row["final"])
        own_key = "attack_final_score" if row["metric"] == "攻撃力" else "overall_final_score"
        check(f"final_box_duplicate:{row['metric']}", row["final"] == summary[own_key])
        if row["metric"] == "攻撃力":
            check("summary_attack_duplicate:main", row["main"] == pages["main_attack"]["printed_means"]["total"])
            check("summary_attack_duplicate:preliminary", row["preliminary"] == pages["preliminary_attack"]["printed_summary"]["attack_score"])
        else:
            check("summary_overall_duplicate:main", row["main"] == main_def["anon"])
            check("summary_overall_duplicate:preliminary", row["preliminary"] == pre_def["anon"])

    paper_aggregates = {}
    for page, fields in (("main_attack", ("mia_t", "aia_r_minus_phi", "total")),
                         ("preliminary_attack", ("mia_t", "aia_R", "total"))):
        rows = pages[page]["rows"]
        printed = pages[page].get("printed_means")
        if printed is None:
            ps = pages[page]["printed_summary"]
            printed = {"mia_t": ps["mia_mean"], "aia_R": ps["aia_R_mean"], "total": ps["attack_score"]}
        paper_aggregates[page] = {}
        for field in fields:
            avg = mean(D(r[field]) for r in rows)
            check(f"table_mean_rounding:{page}:{field}", overlaps(mean_box(r[field] for r in rows), box(printed[field])))
            paper_aggregates[page][field] = {"mean_of_printed_cells": avg, "printed_mean": printed[field],
                                            "difference": avg - D(printed[field])}
        check(f"attack_sum_rounding:{page}", overlaps(add_box(box(printed[fields[0]]), box(printed[fields[1]])), box(printed[fields[2]])))
    for page, printed in (("main_anonymity", main_def), ("preliminary_defense", pre_def)):
        rows = pages[page]["rows"]
        r_key = "aia_R" if page == "main_anonymity" else "aia_r"
        name_key = "attacker" if page == "main_anonymity" else "team_name"
        maximum_key = "maximum_R" if page == "main_anonymity" else "max_aia_r"
        maximum_name_key = "maximum_R_attacker" if page == "main_anonymity" else "max_aia_attacker"
        maximum = max(D(r[r_key]) for r in rows)
        check(f"defense_max_aia:{page}", maximum == D(printed[maximum_key]))
        check(f"defense_max_attacker:{page}", printed[maximum_name_key] in
              [r[name_key] for r in rows if D(r[r_key]) == maximum])
        expected_aia = maximum - D(printed["phi"]) if page == "main_anonymity" else maximum
        check(f"defense_aia_component:{page}", expected_aia == D(printed["aia_component"]))
        check(f"defense_attacker_count:{page}", int(printed["attacker_count"]) == len(rows))
        avg = mean(D(r["mia_t_j"]) for r in rows)
        check(f"defense_beta_raw_mean_rounding:{page}", overlaps(mean_box(r["mia_t_j"] for r in rows), box(printed["beta_raw"])))
        beta_range0 = sub_box(mean_box(r["mia_t_j"] for r in rows), box(printed["beta_chance"]))
        beta_range = (max(beta_range0[0] + D("0.01"), D("0.01")), max(beta_range0[1] + D("0.01"), D("0.01")))
        check(f"defense_beta_rounding:{page}", overlaps(beta_range, box(printed["beta"])))
        component_range = tuple(clip((v / D("0.01") - 1) / 11) for v in beta_range)
        check(f"defense_mia_component_rounding:{page}", overlaps(component_range, box(printed["mia_component"])))
        check(f"defense_anon:{page}", D(printed["anon"]) == 1 - max(D(printed["mia_component"]), D(printed["aia_component"])))
        paper_aggregates[page] = {"mean_of_printed_mia_t_j": avg, "printed_beta_raw": printed["beta_raw"],
                                  "mia_component_from_printed_beta": clip((D(printed["beta"]) / D("0.01") - 1) / 11),
                                  "paper_mia_component": printed["mia_component"],
                                  "warning": "表示丸め値を代入した再計算値は公式未丸め計算を置換しない。"}

    for mode in ("preliminary", "main"):
        key = f"{mode}_score"
        families = {"U": ("U_gen", "U_spec", "U_rare", "U_valid"),
                    "U_gen": ("marginal", "pMSE", "time_dz") if mode == "preliminary" else ("marginal", "corr", "pMSE", "time_dz", "tail"),
                    "U_spec": ("IO", "tstr_cal"), "U_rare": ("mass_star", "prof_cal", "onset_cal", "c2st_cal")}
        if mode == "main":
            families.update({"corr": ("corr_mean", "corr_tail"), "tail": ("U_shift", "U_spike")})
        for parent, children in families.items():
            check(f"utility_min:{mode}:{parent}", D(utility[parent][key]) == min(D(utility[c][key]) for c in children))
        children = ("cens", "no_fab") if mode == "preliminary" else ("cens", "no_fab", "no_dup", "novel")
        check(f"utility_valid_mean:{mode}", overlaps(mean_box(utility[c][key] for c in children), box(utility["U_valid"][key])))
        marginals = pages["utility"]["marginal_columns"][f"{mode}_values"]
        check(f"utility_marginal_10_columns:{mode}", len(marginals) == 10)
        check(f"utility_marginal_mean_rounding:{mode}", overlaps(mean_box(marginals.values()), box(utility["marginal"][key])))

    full = official["official_scores"]
    check("official_json_score_agreement", full == scores)
    aia = mean(official_aia.values())
    mia = official["mia_power_from_exact_total_minus_aia"]
    total = full["score_1"]
    check("official_aia_mean", abs(aia - official["aia_power_np_mean"]) < D("1e-15"))
    check("official_total_components", abs(mia + aia - total) < D("1e-15"))
    for value, paper in ((mia, pages["main_attack"]["printed_means"]["mia_t"]),
                         (aia, pages["main_attack"]["printed_means"]["aia_r_minus_phi"]),
                         (total, pages["main_attack"]["printed_means"]["total"])):
        check(f"official_main_aggregate_rounding:{paper}", box(paper)[0] <= value <= box(paper)[1])
    check("official_whole_mia_vs_paper_rounding", mean_box(r["mia_all"] for r in pages["main_attack"]["rows"])[0]
          <= full["score_2"] <= mean_box(r["mia_all"] for r in pages["main_attack"]["rows"])[1])

    rare = [r for r in pages["main_attack"]["rows"] if int(r["rare_members"]) > 0]
    rare_ids = sorted(r["identity"]["main_cohort_id"] for r in rare)
    check("rare_six_targets_perfect", len(rare) == 6 and all(r["mia_rare"] == "1.000" for r in rare)
          and full["score_3"] == 1 and rare_ids == [6, 10, 11, 16, 17, 25])
    check("rare_nine_members", sum(int(r["rare_members"]) for r in rare) == 9)
    rare_term = D(len(rare)) / len(pages["main_attack"]["rows"])
    mia_fraction = Fraction(official["mia_power_prediction_exact_fraction"])
    rare_fraction = Fraction(len(rare), len(pages["main_attack"]["rows"]))
    check("mia_recorded_exact_fraction", abs(D(mia_fraction.numerator) / mia_fraction.denominator - mia) < D("1e-15"))
    uplift = mia - full["score_2"]
    def identify(row):
        return {"presentation_no": row["identity"]["presentation_no"],
                "main_cohort_id": row["identity"]["main_cohort_id"],
                "name": row.get("team_name", row.get("target", row.get("attacker"))),
                "rank": row.get("target_rank")}
    stars = [r for r in pages["main_attack"]["rows"] if r["marker"] == "★"]
    firsts = [r for r in pages["main_attack"]["rows"] if int(r["target_rank"]) == 1]
    check("main_aia_four_decisive_stars", sorted(r["identity"]["main_cohort_id"] for r in stars) == [3, 4, 15, 26])
    check("main_four_target_first_places", sorted(r["identity"]["main_cohort_id"] for r in firsts) == [6, 10, 16, 25])
    haruna = next(r for r in pages["main_attack"]["rows"] if r["identity"]["main_cohort_id"] == 9)
    check("haruna_high_value_but_rank14", haruna["mia_all"] == "0.720" and haruna["target_rank"] == "14")
    defense_sorted = sorted(pages["main_anonymity"]["rows"], key=lambda r: D(r["aia_R"]), reverse=True)
    phi = D(main_def["phi"])
    mia_damage = D(main_def["mia_component"])
    next_aia = max(ZERO, D(defense_sorted[1]["aia_R"]) - phi)
    single_attack_lowered = 1 - max(mia_damage, next_aia)
    all_aia_lowered = 1 - mia_damage
    utility_profile_only = min(D(utility[k]["main_score"]) for k in ("U_gen", "U_spec", "U_valid"))
    check("conditional_scenarios", single_attack_lowered == D("0.811") and all_aia_lowered == D("0.826")
          and utility_profile_only == D("0.937"))
    prelim_firsts = [r for r in pages["preliminary_attack"]["rows"] if rank_number(r["target_rank"]) == 1]
    failed = [c for c in checks if c["status"] == "FAIL"]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "check_count": len(checks), "failure_count": len(failed),
        "basis_commit": data["metadata"]["reference_commit"],
        "precision_policy": {
            "paper": "3桁表示値はそのまま保持。加算・減算・平均は各入力±0.0005の区間の重なりで検査。端点は保守的に含む。",
            "official": "第8回JSONの未丸め数値をDecimalで読み、浮動小数の集計差は1e-15未満で照合。全標的AIAは0.001へ丸めて一致を確認。",
            "counterfactuals": "紙の表示値を使い、指定した指標以外を固定した算術例。新加工による達成値・順位予測ではない。",
            "not_inferred": "ROC補間があるためTPR×人数から確定ヒット人数を復元しない。第1〜7回の対象別最良回も算出しない。"
        },
        "row_counts": {k: len(pages[k]["rows"]) for k in expected_counts},
        "images": image_results,
        "round8_aia_target_matches": aia_matches,
        "aggregates_from_printed_cells": paper_aggregates,
        "contributions_from_official_round8": {
            "attack_total": total, "mia_power": mia, "aia_power": official["aia_power_np_mean"] ,
            "mia_prior_record_exact_fraction": str(mia_fraction),
            "other_16_mia_term_exact_fraction": str(mia_fraction - rare_fraction),
            "mia_whole_channel_mean": full["score_2"], "mia_rare_channel_mean": full["score_3"],
            "mia_fraction_of_attack_percent": mia / total * 100,
            "aia_fraction_of_attack_percent": official["aia_power_np_mean"] / total * 100,
            "rare_mia_term_fraction": "6/22", "rare_mia_term": rare_term,
            "rare_mia_term_percent_of_attack": rare_term / total * 100,
            "rare_mia_term_percent_of_mia": rare_term / mia * 100,
            "other_16_mia_term": mia - rare_term,
            "rare_max_uplift_over_whole_channel": uplift,
            "rare_max_uplift_percent_of_attack": uplift / total * 100,
            "whole_channel_plus_aia_arithmetic": full["score_2"] + official["aia_power_np_mean"],
            "uplift_interpretation": "同じ第8回回答のmax(全体,希少)集計と全体のみの集計との差。希少攻撃未実施の実験・因果効果ではない。",
            "rare_targets": [identify(r) | {"rare_members": int(r["rare_members"]), "paper_rare_tpr": r["mia_rare"]} for r in rare]
        },
        "target_results": {
            "main_aia_decisive_stars": [identify(r) for r in stars],
            "main_total_first_places": [identify(r) for r in firsts],
            "preliminary_total_first_places_including_ties": [identify(r) for r in prelim_firsts],
            "haruna": identify(haruna) | {k: haruna[k] for k in ("mia_all", "aia_r_minus_phi", "total", "target_rank_denominator")},
            "main_mia_whole_printed_zero_count": sum(r["mia_all"] == "0.000" for r in pages["main_attack"]["rows"])
        },
        "conditional_scenarios_from_paper": {
            "current_anon": main_def["anon"],
            "lower_only_max_aia_to_at_most_second": {"remaining_max_R": defense_sorted[1]["aia_R"],
                "remaining_max_attacker": defense_sorted[1]["attacker"], "remaining_aia_component": next_aia,
                "anon": single_attack_lowered, "condition": "最大攻撃者のRだけを次点以下に下げ、他攻撃者・phi・MIA成分・Uを固定。"},
            "lower_all_aia_to_at_most_mia": {"anon_ceiling": all_aia_lowered,
                "final_score_arithmetic_gain": D("0.9") * (all_aia_lowered - D(main_def["anon"])),
                "condition": "MIA成分0.174とU=0.922を固定し、AIA最大成分だけを0.174以下に下げる。"},
            "improve_only_profile_to_one": {"U_ceiling": utility_profile_only,
                "overall_score_if_anon_unchanged": min(utility_profile_only, D(main_def["anon"])),
                "condition": "prof_calだけを1へ上げ、他の有用性項目・匿名性を固定。U_gen=0.937が次の制約。"}
        },
        "checks": checks,
        "failures": failed,
        "limitations": [
            "数値検証は転記の整合性検査であり、画像の自動OCR再認識ではない。画像は別途目視で照合済み。",
            "紙の攻撃力順位1位は予備戦10%・本戦90%の最終順位。本戦単独の攻撃力順位は紙にない。",
            "他チームの総合点・攻撃力の全順位表は含まれず、2位との差や反実仮想の順位は求められない。",
            "画像のハッシュは出典台帳との一致を検証。元の添付写真はリポジトリ外なので再実行時には不要。"
        ]
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=s) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": len(checks), "failures": failed, "output": str(args.output)}, ensure_ascii=False, default=s))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
