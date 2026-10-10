#!/usr/bin/env python3
"""Recompute a score-only reference from five pinned official result HTMLs.

No prediction CSV, private truth, token, or generation code is read.
Usage: python -B reference_combination_r1to5.py --html R1.html R2.html
       R3.html R4.html R5.html [--output-dir DIRECTORY]
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, localcontext
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import re
import sys

import numpy as np


TARGETS = tuple(range(18)) + (22, 25, 26, 27)
FIXED_REF = "8188510261744ebf622fad583dc226e7ae6da840"
REPO = "smsaku00/pwscup2026-sutehaji"
NAME = "reference_combination_r1to5"
SCORING_MAPPING_SOURCE = {
    "repository": "pwscup/pwscup2026-kit", "ref": "main-attack-20260928",
    "path": "starter/src/pwscup2026/kit/codabench/scoring_io.py",
    "git_blob_sha": "6cf93250763b7ba51fc6e29050a275a0222d8e42",
    "sha256": "32df3a36762ca876449f6c635f4ed8d5fc3ecb5f322697ff167d6c17949d5a9b",
    "url": "https://github.com/pwscup/pwscup2026-kit/blob/main-attack-20260928/starter/src/pwscup2026/kit/codabench/scoring_io.py#L400-L408",
    "evidence": "L400-L408 explicitly label LB breakdown1-4; L357 floors per_target; L366 averages corrected values; L368 averages unfloored R_all",
    "validation": "Root independently fetched official pinned-tag file and matched local SHA256/Git blob; not fetched by this offline script",
}
EXPECTED = (
    ("1d06e508ae0e7fd8119bb6da34b205df8e1cd4b0e4aeaa621dba6cd47bcf6687", "cb811aa86a6013bd7028a2a2540c2b1c804f354c", "攻撃フェーズ/AIA/一回目/detailed_results.html"),
    ("470a3971dcfc138269cb336d47d54c02202855c1ebe56a0989527b76d822267b", "f7df47e26351e60708109729f0f8f61c95ba46c1", "攻撃フェーズ/AIA/第2回/detailed_results.html"),
    ("34e4f6897de0e400c35a5dbcfd3c9f6b1a0aaff48f100530527b09c5619cf86f", "1f167977f2869df83ed92ab9463d9d0e754ad10a", "攻撃フェーズ/AIA/第3回/detailed_results.html"),
    ("5d1b8fd5b2eb1e3ab0ed492ebced4b58d139e49a2e2de682c5d359fa89faa4af", "d3b533195422cac83cd95ba375c2bb1365b75f53", "攻撃フェーズ/AIA/第4回/detailed_results.html"),
    ("2921cdf8a7af4958b78ac64bae04e6297d46eabf1e7b664d48e48f07a8a980dd", "3adac7628bc6697caa07c4f0b7ddc2118814740b", "攻撃フェーズ/MIA/5回目/採点結果/detailed_results.html"),
)
# Exact totals inherited from the separately verified history_metrics.json ledger.
# They are not reconstructed from the four-decimal HTML display.
EXACT_TOTAL_LEDGER = {
    2: {"value": 0.06724437975727421, "path": "攻撃フェーズ/MIA/20261006_MIA第2回_seo02と希少同点/採点結果/第2回_scores.json", "sha256": "df641ac1d1865ab8bd9278034426b557b3ad59cf134a9789324de34da8335e2c", "git_blob_sha": "563e4705b1c09f84708d20c8e7923c8482e55d6f"},
    4: {"value": 0.09566698484923238, "path": "攻撃フェーズ/AIA/第4回/scores.json", "sha256": "47e79e6bcf6330464610f501494608efdb94c515b9b7f9b34b88b8e13238ced9", "git_blob_sha": "3f066f3a204bcc143c72f07abb7af95c59d77b66"},
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def read_official(path: Path, round_number: int) -> dict:
    data = path.read_bytes()
    expected_sha, expected_blob, repository_path = EXPECTED[round_number - 1]
    assert sha256(data) == expected_sha, (round_number, "SHA256 mismatch")
    assert git_blob(data) == expected_blob, (round_number, "Git blob mismatch")
    source = data.decode("utf-8")
    text = html.unescape(re.sub(r"<[^>]+>", " ", source))
    assert re.search(r"チームID\s+28\b", text), (round_number, "team28 missing")
    assert "検証OK・採点完了" in text, (round_number, "incomplete result")
    blocks = re.findall(r"<pre(?:\s[^>]*)?>(.*?)</pre>", source, re.I | re.S)
    assert len(blocks) == 1, (round_number, "expected exactly one result block")
    decoded = html.unescape(blocks[0])
    result = json.loads(decoded)
    decimal_result = json.loads(decoded, parse_float=Decimal, parse_int=Decimal)
    values = result["aia_per_target"]
    assert sorted(map(int, values)) == list(TARGETS)
    assert len(values) == len(TARGETS)
    assert all(math.isfinite(values[str(t)]) and 0 <= values[str(t)] <= 1 for t in TARGETS)
    total = re.search(r"score_1\s+攻撃力合計（MIA力＋AIA力）\s+(\d+\.\d{4})", text)
    assert total, (round_number, "four-decimal total missing")
    ref = FIXED_REF if round_number < 5 else None
    return {
        "round": round_number,
        "result": result,
        "decimal_values": decimal_result["aia_per_target"],
        "total_display": total.group(1),
        "source": {
            "round": round_number, "local_read_path": str(path.resolve()),
            "bytes": len(data), "sha256": expected_sha, "git_blob_sha": expected_blob,
            "repository": REPO, "repository_path": repository_path, "fixed_ref": ref,
            "source_url": f"https://github.com/{REPO}/blob/{ref}/{repository_path}" if ref else None,
            "origin": "Pinned saved official HTML; re-fetched with authorized GitHub fetch_file" if ref else "User-provided Library official HTML, materialized by root",
            "sha256_and_git_blob_match": True, "team_id": 28,
            "validation_and_scoring_complete": True,
        },
    }


def aggregate(values: list[float], decimal_values: list[Decimal]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    assert a.shape == (22,) and np.isfinite(a).all()
    with localcontext() as ctx:
        ctx.prec = 60
        decimal_sum = sum(decimal_values, Decimal(0))
        decimal_mean = decimal_sum / Decimal(22)
    return {
        "target_count": 22, "primary_mean_np_mean": float(np.mean(a)),
        "official_equivalent_np_nanmean": float(np.nanmean(a)),
        "sum_np_sum": float(np.sum(a)), "independent_sum_math_fsum": math.fsum(values),
        "independent_mean_math_fsum_div22": math.fsum(values) / 22,
        "decimal_sum_of_original_JSON_numbers": str(decimal_sum),
        "decimal_mean_of_original_JSON_numbers": str(decimal_mean),
        "np_vs_fsum_sum_abs_error": abs(float(np.sum(a)) - math.fsum(values)),
        "np_vs_fsum_mean_abs_error": abs(float(np.mean(a)) - math.fsum(values) / 22),
        "all_values_finite_no_NaN_exclusion": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", nargs=5, required=True, type=Path, metavar="HTML", help="Official HTML originals in R1,R2,R3,R4,R5 order")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    records = [read_official(path, i) for i, path in enumerate(args.html, 1)]
    matrix = np.array([[r["result"]["aia_per_target"][str(t)] for t in TARGETS] for r in records])
    assert matrix.shape == (5, 22) and np.isfinite(matrix).all()
    assert all(r["result"]["constants"] == records[0]["result"]["constants"] for r in records)
    rows = []
    representatives = {str(r): [] for r in range(1, 6)}
    for column, team in enumerate(TARGETS):
        values = matrix[:, column].tolist()
        maximum, previous = max(values), max(values[:4])
        ties = [r + 1 for r, value in enumerate(values) if value == maximum]
        prior_ties = [r + 1 for r, value in enumerate(values[:4]) if value == previous]
        representative = max(ties)
        representatives[str(representative)].append(team)
        decimal_values = [r["decimal_values"][str(team)] for r in records]
        assert ties == [r + 1 for r, value in enumerate(decimal_values) if value == max(decimal_values)]
        rows.append({
            "main_team": team, **{f"r{r}_aia_corrected": values[r - 1] for r in range(1, 6)},
            "reference_max_r1to5": maximum, "all_max_rounds_r1to5": ties,
            "reference_representative_round": representative,
            "representative_rule": "R5 kept for all-zero tie; convenience only" if team == 8 else "latest round among all exact maxima; convenience only",
            "reference_max_r1to4": previous, "all_max_rounds_r1to4": prior_ties,
            "reference_representative_round_r1to4": max(prior_ties),
            "reference_max_r3to5": max(values[2:]),
            "reference_minus_r5": maximum - values[4],
            "reference_minus_r1to4_reference": maximum - previous,
        })
    assert [r["main_team"] for r in rows if len(r["all_max_rounds_r1to5"]) > 1] == [8]
    assert rows[8]["all_max_rounds_r1to5"] == [1, 2, 3, 4, 5]
    assert rows[8]["reference_representative_round"] == 5
    summaries = {}
    for key, first, last in [("r1to5_reference", 0, 5), ("r1to4_reference", 0, 4), ("r3to5_reference", 2, 5), ("r5_single_submission", 4, 5)]:
        vals = [max(matrix[first:last, column]) for column in range(22)]
        dec = [max(r["decimal_values"][str(team)] for r in records[first:last]) for team in TARGETS]
        summaries[key] = aggregate(vals, dec)
    reference = summaries["r1to5_reference"]["primary_mean_np_mean"]
    r5 = summaries["r5_single_submission"]["primary_mean_np_mean"]
    prior = summaries["r1to4_reference"]["primary_mean_np_mean"]
    comparisons = {"r1to5_reference_minus_r5": reference - r5, "r1to5_reference_minus_r1to4_reference": reference - prior, "r1to5_reference_minus_r3to5_reference": reference - summaries["r3to5_reference"]["primary_mean_np_mean"]}
    assert abs(reference - 0.15528042928257568) < 1e-15
    assert abs(r5 - 0.1027627850065516) < 1e-15
    mia_context = []
    for record in records:
        rn = record["round"]
        aia_mean = float(np.mean(matrix[rn - 1]))
        displayed_total = Decimal(record["total_display"])
        lower = displayed_total - Decimal("0.00005") - Decimal(str(aia_mean))
        upper = displayed_total + Decimal("0.00005") - Decimal(str(aia_mean))
        exact_total = EXACT_TOTAL_LEDGER.get(rn)
        power = exact_total["value"] - aia_mean if exact_total else None
        mia_context.append({
            "round": rn, "mia_all_whole_configuration": record["result"]["mia"]["tpr_at_low_fpr"],
            "total_four_decimal_display": record["total_display"],
            "aia_corrected_power_mean": aia_mean,
            "MIA_power_precision": "derived from independently verified exact scores.json total" if exact_total else "closed conservative interval from four-decimal total; exact value unavailable",
            "MIA_power_exact_derived": power,
            "MIA_power_conservative_interval": {"lower_decimal": str(lower), "upper_decimal": str(upper)},
            "exact_total_inherited_ledger": {**exact_total, "fixed_ref": FIXED_REF, "ledger": "history_metrics.json"} if exact_total else None,
        })
    r3_interval = mia_context[2]["MIA_power_conservative_interval"]
    r3_lower, r3_upper = Decimal(r3_interval["lower_decimal"]), Decimal(r3_interval["upper_decimal"])
    assert max(r["mia_all_whole_configuration"] for r in mia_context) == mia_context[2]["mia_all_whole_configuration"]
    for record in mia_context:
        if record["round"] != 3:
            upper = Decimal(str(record["MIA_power_exact_derived"])) if record["MIA_power_exact_derived"] is not None else Decimal(record["MIA_power_conservative_interval"]["upper_decimal"])
            assert r3_lower > upper
    with localcontext() as ctx:
        ctx.prec = 60
        conditional_r3 = [str(Decimal(str(reference)) + x) for x in (r3_lower, r3_upper)]
    conditional_r4 = reference + mia_context[3]["MIA_power_exact_derived"]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    csv_path, json_path, md_path = (out / f"{NAME}.{suffix}" for suffix in ("csv", "json", "md"))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ";".join(map(str, value)) if isinstance(value, list) else repr(value) if isinstance(value, float) else value for key, value in row.items()})
    with csv_path.open(newline="", encoding="utf-8") as stream:
        restored = list(csv.DictReader(stream))
    assert len(restored) == 22 and [int(row["main_team"]) for row in restored] == list(TARGETS)
    for expected, actual in zip(rows, restored):
        for key, value in expected.items():
            if isinstance(value, float):
                assert float(actual[key]) == value
            elif isinstance(value, list):
                assert list(map(int, actual[key].split(";"))) == value
    lines = [
        "# 第1〜5回AIAの対象別最高・参考組合せ", "",
        f"第1〜5回の公式補正後AIA実値から各対象の最大値を取った参考平均は **{reference!r}**。第5回単独 **{r5!r}** に対して **+{comparisons['r1to5_reference_minus_r5']!r}**、第1〜4回参考 **{prior!r}** に対して **+{comparisons['r1to5_reference_minus_r1to4_reference']!r}**。22対象の合計は `math.fsum` で **{summaries['r1to5_reference']['independent_sum_math_fsum']!r}**。", "",
        "これは過去の採点結果から事後的に最大値を選んだ参考統計です。公式採点済みの単一提出ではありません。新しい組合せ回答CSV・提出ZIPは作成しておらず、次回得点を保証しません。", "",
        "## 指標名とLB内訳3の違い", "",
        "**0.15528042928257568は、総得点に入る補正後AIA力の参考平均です。LB内訳3の値ではありません。** 内訳3は `score_4` のAIA全体 `r_time_all`（床控除前の対象別Rの平均）で、第5回実績は **0.14775730680466903**、4桁表示は **0.1478** です。同じ第5回の補正後AIA力は **0.1027627850065516**。総得点 `score_1` はMIA力と補正後AIA力の和です。", "",
        "公式採点コードはLB内訳1＝MIA全体、内訳2＝MIA希少、内訳3＝AIA全体、内訳4＝AIA希少と明記しています。`score_3` 自体はMIA希少であり、UIの『内訳3』とは番号が異なります。[公式の内訳対応](" + SCORING_MAPPING_SOURCE["url"] + ")／[score_1〜6の写像](https://github.com/pwscup/pwscup2026-kit/blob/main-attack-20260928/starter/src/pwscup2026/kit/codabench/run.py#L56-L60)。", "",
        "補正後の各対象値は `clip(max(R_all,R_rare_SE補正後)−aia_floor[k],0,1)`。最大化・床控除・clipにより元の全体Rを一意に戻せず、原本HTMLには対象別の床控除前Rがありません。したがって、過去ベストを混ぜた参考組合せの**内訳3は、これらのHTMLだけから厳密算出できません**。選んだ回・対象ごとの床控除前R、または当該回答の公式再採点が必要です。参考組合せの内訳3や希少内訳を推定表示していません。", "",
        "## 対象ごとの全値・同点・参考用の代表回", "",
        "番号は本戦対象番号。最大値の同点は丸め前の公式実値で厳密判定し、同点の全回を記録しました。唯一の同点は対象8で、全5回とも0です。対象8の代表回を第5回とするのは既存第5回回答を維持する便宜で、実回答が同じ・その回が優れるという判定ではありません。他の同点があれば全回を表示し、代表は最新回とする規則です。", "",
        "| 本戦対象 | R1 | R2 | R3 | R4 | R5 | R1〜5最大 | 最大の全回 | 参考用代表回 |", "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join([str(row["main_team"])] + [repr(row[f"r{i}_aia_corrected"]) for i in range(1, 6)] + [repr(row["reference_max_r1to5"]), ", ".join(map(str, row["all_max_rounds_r1to5"])), str(row["reference_representative_round"])]) + " |")
    lines += ["", "代表回別の対象一覧：", ""]
    for round_number, teams in representatives.items():
        lines.append(f"- 第{round_number}回：{len(teams)}対象 — {', '.join(map(str, teams)) or 'なし'}")
    lines += ["", "## 集計と丸め前の確認", "", "| 範囲 | 対象max合計（fsum） | 平均（np.mean） |", "|---|---:|---:|"]
    for key, label in [("r1to4_reference", "第1〜4回参考"), ("r3to5_reference", "第3〜5回参考（1・2回を除外）"), ("r1to5_reference", "第1〜5回参考（1・2回を含む）"), ("r5_single_submission", "第5回単独・採点済み")]:
        item = summaries[key]
        lines.append(f"| {label} | {item['independent_sum_math_fsum']!r} | {item['primary_mean_np_mean']!r} |")
    lines += ["", f"第1〜5回参考の `np.sum` は{summaries['r1to5_reference']['sum_np_sum']!r}、`fsum`との差は{summaries['r1to5_reference']['np_vs_fsum_sum_abs_error']!r}です。22値すべて有限なので `np.mean` と公式の `np.nanmean` は同値。原JSON数値をDecimalで読む独立対照もJSONに保存しました。CSVの全110公式値と最大値・差分を読み戻し、float値が完全一致しました。", "",
        "第1・2回を含めることで、対象5・12・15の第1回実績が第3〜5回の参考値を上回ります。第2回も全22対象で比較した上で、この参考組合せでは単独最大となる対象はありません。", "",
        "## MIAと希少回答の扱い", "",
        "MIAの対象別TPRは区分帯のみ公開され、正確な対象別値は非公開です。AIAと同じ対象別maxや個別方式の勝敗は作りません。MIA全体を過去の回答構成ごとに比較すると第3回が最高で、MIA力も第3回の保守区間が第1・2・4・5回の区間／正確値を上回ります。第4回の総点はscores.jsonの正確値、第3・5回は4桁総点からの保守区間で区別しています。第5回MIA力と第4回の差は、第5回の正確な総点がないため未確定です。", "",
        "| 回 | MIA全体・過去構成の実値 | MIA力の精度と値 |", "|---|---:|---|",
    ]
    for record in mia_context:
        interval = record["MIA_power_conservative_interval"]
        precision = f"正確な総点から導出 {record['MIA_power_exact_derived']!r}" if record["MIA_power_exact_derived"] is not None else f"保守区間 [{interval['lower_decimal']}, {interval['upper_decimal']}]"
        lines.append(f"| {record['round']} | {record['mia_all_whole_configuration']!r} | {precision} |")
    lines += ["", f"**全1049行を過去第3回MIA構成のまま保持する条件**で、第1〜5回AIA参考平均を第3回MIA力区間に足した算術参考は **[{conditional_r3[0]}, {conditional_r3[1]}]**。第4回MIA構成の力{mia_context[3]['MIA_power_exact_derived']!r}を使う算術参考は **{conditional_r4!r}**です。第3回保持には固定5bit第1問の希少回答も含みます。個別特定済みの最終回答を使った値ではなく、組合せを採点・保証した値でもありません。", "",
        "希少質問回答と、個人所属を確定させた最終MIA回答CSVは別です。第5回後は保存仮定と原本照合のもと16配置が残り、陽性所属が確定した個人は本戦17のID115のみ。残り2問は未採用・未生成で、個人特定後の最終CSVの得点も未算出・CSV未生成です。[復号の前提と限界](../../MIA希少/第5回実結果_復号_20261010/説明.md)を参照してください。", "",
        "## 出典・再現", "",
        "第1〜4回は固定ref `" + FIXED_REF + "` の公式保存HTMLを正規GitHub fetch_fileで再取得し、既監査のSHA256・Git blobと再照合しました。第5回はユーザー提供Library原本です。全5HTMLはteam28・検証OK・採点完了で、同じ本戦22対象を含みます。HTMLと提出回の対応には保存記録・ユーザー説明を使用しています。各回サーバー入力ZIPの直接hash照合は未実施です。", "",
        "同じ22対象×326課題という母集団は[aia_comparison_audit.json](aia_comparison_audit.json)の原本キー監査で確認済みです。このスクリプトはHTMLの22対象集合を再確認し、回答CSV・個別真値・tokenを読みません。`aia_per_target` は全体／希少r_timeに床控除等を適用した補正後値であり、全体・希少r_timeの平均とは別です。", "",
        "```sh", "python -B reference_combination_r1to5.py --html R1.html R2.html R3.html R4.html R5.html --output-dir OUTPUT_DIRECTORY", "```", "",
        "5引数にはJSON記載のSHA256と一致する公式HTML原本を渡します。外部通信はなく、同名CSV・JSON・MDの3資料を再生成します。資料CSVは得点一覧であり、提出用F_aia.csvではありません。", "",
        "- [全精度・対象別CSV](reference_combination_r1to5.csv)", "- [入力hash・全同点・集計と検証JSON](reference_combination_r1to5.json)", "- [再現スクリプト](reference_combination_r1to5.py)",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "status": "PASS", "created_date": "2026-10-10",
        "scope": "Score-only AIA R1..5 per-target maxima and representative rounds; no answers assembled",
        "metric": "official corrected aia_per_target", "targets": list(TARGETS),
        "leaderboard_metric_context": {
            "reference_metric_label": "総得点に入る補正後AIA力の参考平均",
            "reference_is_LB_breakdown3": False,
            "LB_breakdown3": {"leaderboard_key": "score_4", "formal_key": "aia.r_time_all", "meaning": "mean unfloored AIA whole-channel R", "R5_actual": records[4]["result"]["aia"]["r_time_all"], "R5_display_4dp": "0.1478"},
            "score_3_is": "MIA rare, not UI breakdown3",
            "reference_LB_breakdown3_computable_from_saved_HTML": False,
            "reason": "HTML exposes corrected per-target clip(max(R_all,R_rare)-floor) and whole-channel aggregates, not the selected per-target unfloored R_all; transformation is not invertible",
            "additional_evidence_needed": "selected round/target unfloored R_all values or official rescoring of that concrete answer",
            "official_source": SCORING_MAPPING_SOURCE,
        },
        "input_sources": [r["source"] for r in records], "summaries": summaries,
        "mean_differences": comparisons, "per_target": rows,
        "representative_round_targets": representatives,
        "tie_rule": "all exact maxima recorded; latest representative, target8 R5 kept solely for convenience",
        "all_zero_target8_max_ties": [1, 2, 3, 4, 5],
        "representative_is_adoption_decision": False,
        "mia_whole_configuration_context": mia_context,
        "mia_whole_R3_all_and_power_highest_confirmed": True,
        "mia_targetwise_max_generated": False,
        "conditional_arithmetic_reference": {
            "R3_MIA_full1049_kept": {"total_interval_decimal": conditional_r3, "rare_question_kept": "fixed5bit question1", "condition": "entire old R3 MIA answer including rare cells unchanged"},
            "R4_MIA_power_plus_AIA_reference": conditional_r4,
            "is_scored_single_submission": False, "is_scoring_guarantee": False,
        },
        "rare_context": {"remaining_placements": 16, "condition": "saved membership model and original correspondence", "confirmed_positive_individual_ids": [{"main_team": 17, "record_id": 115}], "future_two_questions_adopted": False, "future_answers_generated": False, "final_individually_resolved_MIA_CSV_generated": False, "final_MIA_CSV_score_computed": False, "evidence": "../../MIA希少/第5回実結果_復号_20261010/説明.md"},
        "validation": {"pinned_input_SHA256_and_git_blob": "all5 PASS", "same_22_HTML_targets": True, "same_constants": True, "all110_AIA_values_finite_in_0_1": True, "max_ties_float_and_decimal_equal": True, "max_tie_target_count": 1, "CSV_rows_readback": 22, "CSV_all_numeric_values_and_ties_roundtrip": True, "sum_and_mean_independent_fsum_check": True},
        "population_context": {"source": "aia_comparison_audit.json", "inherited_verified_targets": 22, "inherited_verified_rows_per_target": 326, "inherited_pair_set_sha256": "1dec4c02e1b9545a7d5d239649df5d68583036dbea62bd69b6101781a36751bc", "standalone_script_checks_HTML_targets_only": True},
        "limits": ["Reference maxima are observed after scoring, not a scored single submission or a guarantee.", "Server input ZIP hashes are not directly checked; R5 mapping follows user-supplied original.", "MIA target TPR is banded; no targetwise MIA winner or max is inferred.", "MIA exact totals R2/R4 are inherited verified ledger records; R1/R3/R5 use conservative rounded-display intervals.", "Rare fixed-question CSV differs from final individual-resolution answer; future questions and final answer not created."],
        "prediction_files_read": False, "new_answers_generated": False, "token_accessed": False,
        "private_truth_accessed": False, "GitHub_write_performed": False,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()},
        "output_SHA256": {csv_path.name: sha256(csv_path.read_bytes()), md_path.name: sha256(md_path.read_bytes()), Path(__file__).name: sha256(Path(__file__).read_bytes())},
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "reference_mean": reference, "reference_sum_fsum": summaries["r1to5_reference"]["independent_sum_math_fsum"], "mean_differences": comparisons, "representative_round_targets": representatives, "files": [csv_path.name, json_path.name, md_path.name]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
