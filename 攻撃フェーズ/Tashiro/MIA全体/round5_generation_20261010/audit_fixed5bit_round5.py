"""Independently verify saved rounds 3/4 and fixed five-bit question 3 (round 5).

Read historical files only. No individual main-round truth is read.
Use the PWS project Python and pass --local-root if its checkout moved.
The optional --candidate verifies a completed round-5 CSV; this script does
not select or generate ordinary MIA ranks.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import importlib.util
from itertools import combinations
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TARGETS = tuple(range(18)) + (22, 25, 26, 27)
ACTIVE = (6, 10, 11, 16, 17, 25)
COUNTS = {6: 1, 10: 2, 11: 1, 16: 2, 17: 1, 25: 2}
HIGH_TOTAL = {6: 6, 10: 7, 11: 7, 16: 11, 17: 10, 25: 12}
CANDIDATES = (47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675)
NEGATIVE_RARE = (126, 136, 271, 273, 317, 440, 480, 625, 637, 643, 668, 814, 841)
QUESTIONS = {
    3: (115, 169, 264, 383, 462),
    4: (47, 115, 304, 462, 675),
    5: (47, 90, 264, 462, 495),
    6: (47, 76, 264, 304, 383),
    7: (47, 76, 90, 115, 169),
}
QI = ("age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture")
ROUND3_SHA256 = "6ca9604a8936fb47255ae0bd05c9f3958368f0e196bc2af258f8255cf39c08eb"
ROUND4_SHA256 = "45077532e5b5cf6109bf86ce67472fce0fac703584fa81a2f0027434fe581183"
COUNT_SHA256 = "411e00d38ffbe7cb38bb73c654b129ca6377a198ad9031e409cccdc19dd725a4"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob(path):
    body = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"Missing header: {path}")
        require(len(set(reader.fieldnames)) == len(reader.fieldnames), "Duplicate columns")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows), "Malformed CSV")
    return tuple(reader.fieldnames), rows


def key(row):
    return tuple(row[c] if c == "prefecture" else Decimal(row[c]) for c in QI)


def expected_value(rid, team, round_number):
    if team not in ACTIVE:
        return 0.0
    padding = set(NEGATIVE_RARE[:HIGH_TOTAL[team] - 5])
    return 0.4 if rid in set(QUESTIONS[round_number]) | padding else 0.1


def verify_csv(path):
    header, rows = read_csv(path)
    require(header == ("record_id",) + tuple(map(str, TARGETS)), "Target columns/order differ")
    require(len(rows) == 1049, "Wrong number of rows")
    require([row["record_id"] for row in rows] == [str(i) for i in range(1049)], "record_id order differs")
    for row in rows:
        for team in TARGETS:
            value = Decimal(row[str(team)])
            require(value.is_finite() and 0 <= value <= 1, "Nonfinite/out-of-range confidence")
    return header, rows


def verify_rare(rows, rare, round_number):
    differences = [
        [rid, team] for rid in rare for team in TARGETS
        if float(rows[rid][str(team)]) != expected_value(rid, team, round_number)
    ]
    require(not differences, f"Round {round_number} rare cells differ: {differences[:8]}")
    return {"round": round_number, "question_number": round_number - 2,
            "cells": len(rare) * len(TARGETS), "differences": len(differences),
            "high_ids": {str(team): [rid for rid in rare if expected_value(rid, team, round_number) == 0.4]
                         for team in ACTIVE}}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("/Users/taisei1105/pws2026-local"))
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--report", type=Path, default=HERE / "rare_round5_independent_audit.json")
    args = parser.parse_args()
    root = args.local_root.resolve()
    attack = root / "本戦/攻撃フェーズ/Tashiro"
    source = root / "work/mia_round3_publish_20261008/candidate/攻撃フェーズ/MIA/20261008_第3回MIA_固定5bitと非会員較正"
    p3 = source / "結果/F_mia.csv"
    p4 = attack / "20261009_第4回MIA_チーム別方式と固定5bit/結果/F_mia.csv"
    bpaths = list((root / "本戦").glob("配布*/PWSCup2026_final_team28_*/B_self.csv"))
    require(len(bpaths) == 1, "Ambiguous main-round B_self path")
    bpath = bpaths[0]
    qpath = attack / "20260928_配布データ基本分析/入力/aia_challenge_28.csv"
    declared = qpath.parent / "mia_columns.json"
    count_path = attack / "20261006_MIA第2回結果と人数導出/公開用/MIA/20261006_MIA第2回_seo02と希少同点/採点結果/MIA結果.json"
    spec = source / "仕様一覧.md"
    settings_path = source / "結果/設定.json"
    table = source / "結果/固定5bit符号.csv"
    certificate_path = source / "結果/希少監査_再検証.json"
    round4_provenance = HERE / "inputs/rare_provenance/round4_出典と照合.json"
    round4_reconciliation = HERE / "inputs/rare_provenance/round4_照合結果.json"
    require(digest(p3) == ROUND3_SHA256, "Round 3 original SHA256 differs")
    require(digest(p4) == ROUND4_SHA256, "Round 4 original SHA256 differs")
    require(digest(count_path) == COUNT_SHA256, "Previously derived count-source SHA256 differs")
    declared_targets = tuple(int(t) for t in json.loads(declared.read_text())["target_columns"] if int(t) != 28)
    require(declared_targets == TARGETS, "Official main-round target columns differ")
    _, b = read_csv(bpath)
    _, q = read_csv(qpath)
    require(len(b) == 1049 and [int(x["record_id"]) for x in b] == list(range(1049)), "Main B_self IDs/order differ")
    require(len(q) == 326, "Self AIA challenge row count differs")
    bkeys, qkeys = [key(x) for x in b], [key(x) for x in q]
    require(len(set(bkeys)) == 1049 and len(set(qkeys)) == 326, "Seven-QI duplicate")
    private = set(i for i, k in enumerate(bkeys) if k in set(qkeys))
    require(all(x["is_rare"] in ("0", "1") for x in b), "Invalid is_rare label")
    rare = tuple(i for i, x in enumerate(b) if x["is_rare"] == "1")
    known_rare = tuple(sorted(set(rare) & private))
    candidates = tuple(sorted(set(rare) - private))
    known_ordinary = tuple(sorted(private - set(rare)))
    ordinary = tuple(sorted(set(range(1049)) - private - set(rare)))
    require((len(private), len(rare), len(known_rare), len(candidates), len(known_ordinary), len(ordinary)) ==
            (163, 24, 13, 11, 150, 875), "Exclusive partition differs")
    require(candidates == CANDIDATES and known_rare == NEGATIVE_RARE, "Rare/padding IDs differ")
    counts = json.loads(count_path.read_text())["derived_rare_members"]
    require(counts == {str(t): COUNTS.get(t, 0) for t in TARGETS}, "Derived team counts differ")
    settings = json.loads(settings_path.read_text())
    require(settings["fixed_queries"] == {str(k): list(v) for k, v in QUESTIONS.items()}, "Saved fixed queries differ")
    require(settings["total_high"] == {str(t): HIGH_TOTAL[t] for t in ACTIVE}, "High totals differ")
    _, code_rows = read_csv(table)
    single = {rid: tuple(int(rid in QUESTIONS[r]) for r in range(3, 8)) for rid in CANDIDATES}
    require({int(x["record_id"]): x["bits"] for x in code_rows} ==
            {rid: "".join(map(str, bits)) for rid, bits in single.items()}, "Saved bit table differs")
    pair = {ids: tuple(sum(single[rid][j] for rid in ids) for j in range(5)) for ids in combinations(CANDIDATES, 2)}
    require(len(set(single.values())) == 11 and len(set(pair.values())) == 55, "Fixed question signature collision")
    cert = json.loads(certificate_path.read_text())
    require(cert["audit_status"] == "PASS" and cert["candidate_sha256"] == ROUND3_SHA256, "Saved certificate differs")
    require(cert["fixed_five_bit_signatures"]["questions"] == [list(QUESTIONS[r]) for r in range(3, 8)], "Certificate question order differs")
    require(cert["rare_csv_cells"]["total_high"] == [HIGH_TOTAL[t] for t in ACTIVE], "Certificate high totals differ")
    provenance = json.loads(round4_provenance.read_text())
    reconciliation = json.loads(round4_reconciliation.read_text())
    require(provenance["F_mia_sha256"] == ROUND4_SHA256 == reconciliation["sha256"], "Pinned round 4 provenance differs")
    require(reconciliation["decode"]["next_round"] == 5 and reconciliation["decode"]["next_fixed_question"] == list(QUESTIONS[5]), "Round 4 next-question reconciliation differs")
    _, rows3 = verify_csv(p3)
    _, rows4 = verify_csv(p4)
    history_checks = [verify_rare(rows3, rare, 3), verify_rare(rows4, rare, 4)]
    require(sum(float(x["9"]) != float(y["9"]) for x, y in zip(rows3, rows4)) == 0, "Haruna saved P01 originals differ")
    for rows in (rows3, rows4):
        require(all(float(rows[rid][str(t)]) == 0 for rid in known_ordinary for t in TARGETS), "Historical ordinary nonshared zero differs")
        require(all(0.6 < float(rows[rid][str(t)]) <= 0.9 for rid in ordinary for t in TARGETS), "Historical ordinary range differs")
    expected_csv = HERE / "rare_round5_expected.csv"
    with expected_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("record_id",) + tuple(map(str, TARGETS)))
        for rid in rare:
            writer.writerow((rid,) + tuple(expected_value(rid, t, 5) for t in TARGETS))
    partitions = {"ordinary_ids": list(ordinary), "known_nonshared_ordinary_ids": list(known_ordinary),
                  "rare_ids": list(rare), "known_nonshared_rare_ids": list(known_rare),
                  "unknown_rare_ids": list(candidates)}
    write_json(HERE / "rare_round5_partition.json", partitions)
    # Reuse only the independent historical auditor, never the builder.
    auditor_path = source / "audit_rare.py"
    module_spec = importlib.util.spec_from_file_location("_saved_independent_rare_auditor", auditor_path)
    auditor = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(auditor)
    auditor.ROOT, auditor.ATTACK, auditor.INPUT, auditor.COUNT_SOURCE = root, attack, qpath.parent, count_path
    _, auditor_rare, auditor_negative, auditor_unknown, original_check = auditor.original_inputs()
    require(tuple(auditor_rare) == rare and tuple(auditor_negative) == known_rare and tuple(auditor_unknown) == candidates,
            "Independent historical auditor partition differs")
    _, _, certificate_recheck = auditor.signatures()
    score_mia, official_mean, official_sources = auditor.official_functions()
    alphabet = auditor.check_score_alphabet(score_mia, official_mean)
    round5_rows = [dict(row) for row in rows3]
    for rid in rare:
        for t in TARGETS:
            round5_rows[rid][str(t)] = repr(expected_value(rid, t, 5))
    # The historical checker reads its first-question slot; replace it solely
    # for question-3 CSV/scorer checks, after verifying the original bit order.
    auditor.QUESTIONS = (QUESTIONS[5],) + auditor.QUESTIONS[1:]
    auditor.check_rare_cells(("record_id",) + tuple(map(str, TARGETS)), round5_rows, rare, known_rare, candidates)
    actual = auditor.check_actual_candidate_scores(round5_rows, b, rare, candidates, score_mia)
    candidate_check = None
    if args.candidate:
        candidate = args.candidate.resolve()
        _, rows = verify_csv(candidate)
        candidate_check = verify_rare(rows, rare, 5)
        require(all(float(rows[rid][str(t)]) == 0 for rid in known_ordinary for t in TARGETS), "Round 5 ordinary nonshared zero differs")
        require(all(0.6 < float(rows[rid][str(t)]) <= 0.9 for rid in ordinary for t in TARGETS), "Round 5 ordinary range differs")
        haruna_diff = sum(float(row["9"]) != float(original["9"]) for row, original in zip(rows, rows3))
        require(haruna_diff == 0, "Round 5 Haruna P01 full column changed")
        actual = auditor.check_actual_candidate_scores(rows, b, rare, candidates, score_mia)
        candidate_check.update(path=str(candidate), sha256=digest(candidate),
                               ordinary_nonshared_zero_cells=150 * 22, haruna_full_column_differences=haruna_diff,
                               ordinary_min=min(float(rows[rid][str(t)]) for rid in ordinary for t in TARGETS),
                               ordinary_max=max(float(rows[rid][str(t)]) for rid in ordinary for t in TARGETS),
                               rare_max=max(float(rows[rid][str(t)]) for rid in rare for t in TARGETS))
    watched = [p3, p4, bpath, qpath, declared, count_path, spec, settings_path, table, certificate_path,
               auditor_path, round4_provenance, round4_reconciliation]
    result = {
        "status": "PASS", "round": 5, "question_number": 3, "question_high_candidates": list(QUESTIONS[5]),
        "target_order": list(TARGETS), "rare_counts": counts,
        "high_total": {str(t): HIGH_TOTAL[t] for t in ACTIVE},
        "padding_ids": {str(t): list(NEGATIVE_RARE[:HIGH_TOTAL[t] - 5]) for t in ACTIVE},
        "exclusive_partition_counts": {"ordinary": 875, "known_nonshared_ordinary": 150, "rare": 24},
        "known_nonshared_all": 163, "known_nonshared_rare_overlap": 13,
        "original_zero_semantics": "既知非共有の通常150人のみ0。希少13人は希少あり6対象で人数調整に0.1/0.4を使用。163人全員0では保存5bit仕様と一致しない。",
        "overlap_nonzero_rare_cells": 13 * 6,
        "history_rare_checks": history_checks,
        "round3_vs_round4_haruna_column_differences": 0,
        "haruna_fixed_column_compatibility": "本戦9の希少人数0で24希少行は保存原本・第3問とも0。第3・第4回の全1049行が同値。",
        "expected_rare_csv": {"path": expected_csv.name, "rows": 24, "targets": 22, "cells": 528, "sha256": digest(expected_csv)},
        "signature_recheck": {"single_cases": 11, "pair_cases": 55, "single_collisions": 0, "pair_collisions": 0,
                              "certificate_legal_joint_configurations": certificate_recheck["legal_joint_configurations"]},
        "official_score_alphabet": alphabet, "question3_official_synthetic_scorer_checks": actual,
        "official_source_sha256": official_sources, "candidate_check": candidate_check,
        "inputs": [{"path": str(p), "sha256": digest(p), "git_blob_sha1": git_blob(p)} for p in watched],
        "auditor_sha256": digest(Path(__file__).resolve()),
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "seed": "deterministic; no random sampling"},
        "limitations": ["人数は第1・2回公式スコアと共有条件に基づく条件付き導出。個人の本戦所属正解は未使用。",
                        "第4回はユーザー提供CSV原本の対応付け。サーバー入力ZIPの直接hash照合は未実施。",
                        "第5回は今回作成する第3問。大会提出・採点・個人所属確定は行っていない。"],
        "submitted": False, "main_individual_truth_used": False,
    }
    write_json(args.report.resolve(), result)
    lines = [
        "# 第5回MIA希少：固定5bit第3問の独立監査",
        "", "## 前提", "",
        "第3回原本を第1問、第4回ユーザー提供原本を第2問として、保存設定・固定5bit符号表・希少証明書と照合した。今回の第5回は第3問であり、大会へ未提出。人数は第1・2回公式スコアと共有条件に基づく条件付き導出で、未知の本戦個人所属正解は用いていない。",
        "", "## 仮説", "",
        "第3問の共通高側候補は47・90・264・462・495。希少あり6対象はこの5人を0.4、残りの未知希少候補を0.1とし、既知希少非共有13人の先頭から保存人数分を0.4へ調整する。希少人数0の16対象は24行をすべて0とする。",
        "", "| 本戦対象 | 導出希少人数 | 高側総数 | 0.4へ調整する既知希少非共有ID |",
        "|---|---:|---:|---|",
    ]
    lines += [f"| {t} | {COUNTS[t]} | {HIGH_TOTAL[t]} | " + "・".join(map(str, NEGATIVE_RARE[:HIGH_TOTAL[t] - 5])) + " |"
              for t in ACTIVE]
    lines += [
        "", "**既知非共有163人全員の0化は保存仕様と両立しない。** 公開7属性の完全一致照合により163人と希少24人の重複は13人。排他的な区分は通常875人・通常非共有150人・希少24人。通常非共有150人は0にし、重複する希少13人は希少あり6対象の78セルで0.1/0.4を使用する。これは所属を会員と判断した値ではなく、質問応答を分離する人数調整である。",
        "", "## 分析方法", "",
        "B_selfとaia_challenge_28の公開7属性をDecimalの完全一致で照合し、希少候補と非共有行を再導出した。第3回CSVのSHA256は6ca9604a8936fb47255ae0bd05c9f3958368f0e196bc2af258f8255cf39c08eb、第4回原本は45077532e5b5cf6109bf86ce67472fce0fac703584fa81a2f0027434fe581183。第4回照合資料から次回番号5と質問5人を確認した。保存生成器を呼ばず、期待値を独立に構成した。",
        "", "保存独立監査器の原入力照合と公式score_miaを再実行。第3問の11単体・55組を各対象の人数条件で全検査し、合計198ケースと希少人数0の16ケースを確認した。公式平均採点の216応答は衝突0で、最小差は厳密に1/54000。保存5質問の単体11符号・2人組55符号も衝突0。通常値の変更で希少採点が変わらない198ケースも確認した。",
        "", "## 結論", "",
        "第3回・第4回の希少528セルずつ、計1,056セルは保存質問と完全一致。第5回用の期待値528セルをrare_round5_expected.csvに保存した。SHA256は" + digest(expected_csv) + "。",
        "", "HarunaDan@GU（本戦9）は希少人数0のため、保存原本・第3問とも24希少行が0。第3・第4回原本の全1049行が同値なので、全列保存と質問更新は両立する。",
    ]
    if candidate_check:
        lines += ["", f"最終CSVもPASS。希少528セル差分0、通常非共有150人×22対象の3,300セルが0、Haruna全1049行差分0。通常875人の値域は{candidate_check['ordinary_min']}〜{candidate_check['ordinary_max']}で全て0.6超、希少最大値は0.4。最終CSV SHA256は{candidate_check['sha256']}。"]
    else:
        lines += ["", "この報告は保存原本と第3問期待値の監査結果。最終CSVは、生成後に--candidateで指定して同監査を実行する。"]
    lines += ["", "第3回・第4回から第3問への希少更新は各36セル。最終CSVの通常部分には別途5対象の変更が入るため、第4回比の差分を5対象変更だけの効果と解釈しない。", "",
              "根拠：配布データの完全一致照合／保存提出CSV・質問表・証明書／公式コード。第4回原本はユーザー提供CSVと提出画面の対応付けであり、サーバー入力ZIPの直接hash照合はしていない。入力ファイルと監査器・実行環境のSHA256等は対応するJSON報告に記録した。", ""]
    args.report.resolve().with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "round": 5, "question": 3, "rare_cells": 528,
                      "expected_sha256": digest(expected_csv), "history_cells_checked": 1056,
                      "candidate_checked": candidate_check is not None, "report": str(args.report.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
