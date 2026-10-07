#!/usr/bin/env python3
"""第3回MIAの独立CSV・公式検証・入力不変監査。生成器はimportしない。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
QI7 = ["age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
ACTIVE = {6, 10, 11, 16, 17, 25}
EXPECTED_CANDIDATES = {47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675}
EXPECTED_RARE_NEGATIVE = {126, 136, 271, 273, 317, 440, 480, 625, 637, 643, 668, 814, 841}
INDEPENDENT_SOURCE_PINS = {
    "B_self.csv": "1339042b4e1937d5fbf482aa8ce4de3f67a02e4694f8237894ca59bf3f0d52b1",
    "aia_challenge_28.csv": "9d464fce3af708819cb6358c1ad33182472f37b9f26fc94fdcc7c769d70d4c26",
    "mia_columns.json": "bca9a100c90222eb774e57708db34ee3f0519a12abc0bb74d170bc68459ef181",
    "validate.py": "1bd625c30e1d08e5377629fc461f48c4ebf634b291eb86a8afd033565724667e",
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def allowed_path(name):
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    parts = {unicodedata.normalize("NFC", part) for part in path.parts}
    check(not ({"archive", "アーカイブ"} & parts), "archiveの読み取りは禁止")
    check(path.is_relative_to(ROOT), f"入力はプロジェクト内に限る: {path}")
    return path


def hash_dict(path):
    data = read_json(path)
    values = data.get("input_sha256")
    check(isinstance(values, dict) and bool(values), f"input_sha256辞書が必要: {path}")
    result = {}
    for name, digest in values.items():
        check(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), f"SHA形式不正: {name}")
        resolved = allowed_path(name)
        check(resolved not in result, f"同一入力の重複: {name}")
        result[resolved] = digest
    return result


def read_context():
    distributed = [p for p in (ROOT / "本戦").iterdir()
                   if unicodedata.normalize("NFC", p.name) == "配布データ"]
    check(len(distributed) == 1, "本戦配布データが一意でない")
    dist = distributed[0]
    own = list(dist.glob("PWSCup2026_final_team28_*"))
    check(len(own) == 1, "team28配布コホートが一意でない")
    kit = dist / "pwscup2026-kit-main_revised0912"
    inputs = HERE.parent / "20260928_配布データ基本分析/入力"
    paths = [own[0] / "B_self.csv", inputs / "aia_challenge_28.csv", inputs / "mia_columns.json",
             kit / "starter/src/pwscup2026/kit/validate.py"]
    for path in paths:
        check(sha(path) == INDEPENDENT_SOURCE_PINS[path.name], f"独立基準の入力SHAと不一致: {path}")
    b = pd.read_csv(paths[0], float_precision="round_trip")
    q = pd.read_csv(paths[1], float_precision="round_trip")
    declared = read_json(paths[2])
    check(b.record_id.is_unique and set(b.record_id) == set(range(1049)), "B_selfの行キー不正")
    check(set(b.is_rare) == {0, 1}, "is_rareは公式B_selfの0/1を使用する")
    check(not b.duplicated(QI7).any() and not q.duplicated(QI7).any(), "QI7照合が一意でない")
    private = set(map(int, b.merge(q[QI7], on=QI7, validate="one_to_one").record_id))
    rare = set(map(int, b.loc[b.is_rare == 1, "record_id"]))
    check((len(b), len(q), len(private), len(rare), len(private & rare)) == (1049, 326, 163, 24, 13),
          "配布Bと自AIA課題の人数不一致")
    check(rare - private == EXPECTED_CANDIDATES, "候補11人が独立照合と異なる")
    check(rare & private == EXPECTED_RARE_NEGATIVE, "既知希少非会員13人が独立照合と異なる")
    cols = [str(c) for c in declared["target_columns"] if str(c) != "28"]
    check(len(cols) == 22 and len(set(cols)) == 22, "対象22列が一意でない")
    check(set(map(int, cols)) == set(range(18)) | {22, 25, 26, 27}, "本戦対象チームが異なる")
    sys.path.insert(0, str(kit / "starter/src"))
    official = importlib.import_module("pwscup2026.kit.validate")
    check(Path(official.__file__).resolve() == paths[3].resolve(), "公式validateのimport先が異なる")
    return {"b": b, "q": q, "private": private, "rare": rare, "cols": cols,
            "sources": paths, "official": official, "declared": declared,
            "normal": set(map(int, b.record_id)) - private - rare}


def verify_rows(rows, context):
    """独立raw CSV解析。すべての属性対応はrecord_idによるjoinで扱う。"""
    errors = []
    expected_header = ["record_id", *context["cols"]]
    if not rows:
        return ["CSVが空"], None, {"called": False}
    if rows[0] != expected_header:
        errors.append("公式対象列の順序・過不足・重複・自列・非整数列に不一致")
    if len(rows) != 1050:
        errors.append("ヘッダー+1049行でない")
    if any(len(row) != 23 for row in rows):
        errors.append("各行は23フィールドでなければならない")
    if errors:
        return errors, None, {"called": False}
    ids = []
    values = []
    for line, row in enumerate(rows[1:], 2):
        if not re.fullmatch(r"0|[1-9][0-9]*", row[0]):
            errors.append(f"record_idは正準整数でなければならない: line {line}")
            continue
        ids.append(int(row[0]))
        try:
            nums = [float(cell) for cell in row[1:]]
        except ValueError:
            errors.append(f"確信度が数値でない: line {line}")
            continue
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in nums):
            errors.append(f"確信度はfinite [0,1]が必要: line {line}")
        values.append(nums)
    if ids != list(range(1049)):
        errors.append("record_idは0..1048の昇順一意でなければならない")
    if errors:
        return errors, None, {"called": False}
    frame = pd.DataFrame(values, columns=context["cols"])
    frame.insert(0, "record_id", ids)
    by_id = frame.set_index("record_id", verify_integrity=True)
    # Bの物理行番号を使わず、全グループをrecord_idの集合で照合する。
    normal = by_id.loc[sorted(context["normal"])]
    nonrare_private = by_id.loc[sorted(context["private"] - context["rare"])]
    rare = by_id.loc[sorted(context["rare"])]
    if not (normal.to_numpy() > .6).all():
        errors.append("未知通常875人のすべてのconfidenceは0.6超が必要")
    if not (nonrare_private.to_numpy() == 0).all():
        errors.append("非希少既知非会員150人は全対象で0が必要")
    if not (rare.to_numpy() <= .4).all():
        errors.append("希少24人のconfidenceは0.4以下が必要")
    inactive = [c for c in context["cols"] if int(c) not in ACTIVE]
    if not (rare[inactive].to_numpy() == 0).all():
        errors.append("希少陽性0の16対象は希少24行すべて0が必要")
    active_values = rare[[c for c in context["cols"] if int(c) in ACTIVE]].to_numpy()
    if not np.isin(active_values, [.1, .4]).all():
        errors.append("6対象の希少値は0.1/0.4の二値が必要")
    result = context["official"].validate_mia_submission(
        frame, {}, {"mia_columns": context["declared"], "team_id": 28,
                    "attacker_record_ids": set(map(int, context["b"].record_id))})
    errors.extend([f"公式MIA検証: {message}" for message in result.errors])
    return errors, frame, {"called": True, "function": "validate_mia_submission",
                           "source": str(context["sources"][3].relative_to(ROOT)),
                           "source_sha256": sha(context["sources"][3]),
                           "passed": bool(result.ok), "errors": list(result.errors)}


def mutation_checks(rows, context):
    cases = []
    normal_id = min(context["normal"])
    rare_id = min(context["rare"])
    nonrare_negative = min(context["private"] - context["rare"])
    row_by_id = {int(row[0]): i for i, row in enumerate(rows[1:], 1)}
    active_col = rows[0].index("6")
    inactive_col = rows[0].index("0")

    def run(name, edit):
        altered = [row.copy() for row in rows]
        edit(altered)
        errors, _, official = verify_rows(altered, context)
        check(bool(errors), f"不正CSVを受理した: {name}")
        cases.append({"case": name, "rejected": True, "errors": errors,
                      "official_called": official["called"]})

    def cell(rid, col, value):
        return lambda altered: altered[row_by_id[rid]].__setitem__(col, value)

    run("unknown_target_999", lambda altered: altered[0].__setitem__(1, "999"))
    run("self_team_28", lambda altered: altered[0].__setitem__(1, "28"))
    run("time_column_misuse", lambda altered: altered[0].__setitem__(1, "time"))
    run("time_hat_column_misuse", lambda altered: altered[0].__setitem__(1, "time_hat"))
    run("duplicate_target_header", lambda altered: altered[0].__setitem__(2, altered[0][1]))
    run("missing_target", lambda altered: [row.pop() for row in altered])
    run("reordered_target_header", lambda altered: altered[0].__setitem__(slice(1, 3), altered[0][1:3][::-1]))
    run("duplicate_record_id", lambda altered: altered[2].__setitem__(0, altered[1][0]))
    run("reordered_rows", lambda altered: altered.__setitem__(slice(1, 3), altered[1:3][::-1]))
    run("missing_row", lambda altered: altered.pop())
    run("extra_id", lambda altered: altered[-1].__setitem__(0, "1049"))
    run("extra_row", lambda altered: altered.append(["1049", *altered[-1][1:]]))
    run("fractional_id", lambda altered: altered[2].__setitem__(0, "1.5"))
    run("blank_id", lambda altered: altered[2].__setitem__(0, ""))
    for name, value in [("nan", "NaN"), ("positive_inf", "inf"), ("negative_inf", "-inf"),
                        ("negative_confidence", "-0.1"), ("above_one", "1.1"),
                        ("blank_confidence", ""), ("time_years_as_confidence", "5.5")]:
        run(name, cell(normal_id, active_col, value))
    run("too_few_fields", lambda altered: altered[1].pop())
    run("too_many_fields", lambda altered: altered[1].append("0"))
    run("normal_band_overlap", cell(normal_id, active_col, ".6"))
    run("known_normal_nonzero", cell(nonrare_negative, active_col, "0.0001"))
    run("rare_band_overlap", cell(rare_id, active_col, ".7"))
    run("inactive_rare_nonzero", cell(rare_id, inactive_col, ".1"))
    run("active_rare_wrong_level", cell(rare_id, active_col, ".2"))
    # 属性照合の物理行順依存がないことも検査する。
    reordered_b = context["b"].sample(frac=1, random_state=20261008)
    revised = dict(context)
    revised["b"] = reordered_b
    errors, _, _ = verify_rows(rows, revised)
    check(not errors, "B_selfの行順だけで属性照合が変わる")
    return {"cases": cases, "rejected_cases": len(cases), "all_rejected": True,
            "shuffled_B_self_same_result": True,
            "scope": "CSV構造・値域・値の配置を検査。0..1へ加工されたtimeの意味的混入は通常者の独立数値監査の対象。"}


def old_output_pins():
    """既存記録の保存SHAを基準に、前の提出物が変わっていないことを確認する。"""
    pins = {}
    summaries = []
    for name in ["20261005_MIA初回_GenLRA比較", "20261005_MIA希少_第1回同点データ",
                 "20261006_MIA第2回_seo02と希少同点"]:
        folder = HERE.parent / name / "結果"
        manifest = folder / "出力SHA256.json"
        check(manifest.exists(), f"既存出力SHAがない: {manifest}")
        declared = read_json(manifest)
        count = 0
        for filename, digest in declared.items():
            path = allowed_path(folder / filename)
            check(sha(path) == digest, f"既存出力が保存SHAと異なる: {path}")
            pins[path] = digest
            count += 1
        pins[manifest] = sha(manifest)
        summaries.append({"folder": str(folder.relative_to(ROOT)), "checked_outputs": count})
    return pins, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--pins", type=Path, default=HERE / "input_pins.json")
    args = parser.parse_args()
    check(Path(sys.prefix).resolve() == (ROOT / ".venv").resolve(), "ルート.venvのPythonだけを使用する")
    candidate = allowed_path(args.candidate)
    output = allowed_path(args.output)
    check(candidate.is_relative_to(HERE) and output.is_relative_to(HERE), "新しい第3回フォルダ内だけを扱う")
    check(candidate.name == "F_mia.csv", "候補名はF_mia.csv")
    check(not output.exists(), "監査記録を上書きしない。新しい--outputを指定する")
    manifest = allowed_path(args.manifest or candidate.parent / "inputs_manifest.json")
    pins_path = allowed_path(args.pins)
    pins = hash_dict(pins_path)
    used = hash_dict(manifest)
    check(pins == used, "builder input manifestと事前pin辞書が完全一致しない")
    for path, expected in pins.items():
        check(sha(path) == expected, f"入力pin不一致: {path}")
    context = read_context()
    check(all(path.resolve() in pins for path in context["sources"]), "必須独立入力がbuilder pinsにない")
    old_pins, old_summaries = old_output_pins()
    watched = dict(pins)
    watched.update(old_pins)
    for path in [candidate, manifest, pins_path, Path(__file__).resolve()]:
        watched[path] = sha(path)
    before = {path: sha(path) for path in watched}
    for path, expected in watched.items():
        check(before[path] == expected, f"監査開始前に入力変化: {path}")
    content = candidate.read_text(encoding="utf-8-sig")
    rows = list(csv.reader(io.StringIO(content, newline=""), strict=True))
    errors, frame, official = verify_rows(rows, context)
    check(not errors, "候補検証失敗: " + "; ".join(errors))
    mutations = mutation_checks(rows, context)
    after = {path: sha(path) for path in watched}
    check(before == after, "監査中に入力または既存成果物が変わった")
    check(not any(p.exists() for p in [candidate.parent / "F_aia.csv", candidate.parent / "token.txt"]),
          "この監査対象はMIA単体。F_aia/tokenは不要")
    check(not any(candidate.parent.glob("*.zip")), "この段階では提出ZIPを作らない")
    by_id = frame.set_index("record_id")
    values = by_id.loc[sorted(context["normal"])].to_numpy()
    report = {
        "passed": True, "audit": "independent integration audit; no builder imports",
        "completed_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "candidate": str(candidate.relative_to(ROOT)), "candidate_sha256": sha(candidate),
        "auditor_sha256": sha(Path(__file__)), "rows": 1049, "targets": 22, "self_team": 28,
        "official_validation": official,
        "independent_strict_format": {"canonical_record_ids": True, "duplicate_header_checked": True,
                                      "finite_unit_interval": True, "exact_target_header": True,
                                      "row_id_join_only": True},
        "cohort_groups": {"private": 163, "rare": 24, "private_rare": 13,
                          "unknown_rare": 11, "unknown_normal": 875, "private_nonrare": 150},
        "separated_confidence_ranges": {"normal_min": float(values.min()), "normal_max": float(values.max()),
                                        "rare_max": float(by_id.loc[sorted(context["rare"])].to_numpy().max()),
                                        "inactive_16_rare_all_zero": True, "private_nonrare_all_zero": True,
                                        "tpr_all_independent_of_rare_changes_if_members_le27": True,
                                        "normal_block_min_end_fpr": 848 / 1022},
        "mutation_tests": mutations,
        "input_pins_equal_builder_manifest": True, "inputs_unchanged_before_after": True,
        "checked_input_sha256": {str(path.relative_to(ROOT)): digest for path, digest in before.items()},
        "existing_output_records": old_summaries,
        "writes": "only this audit JSON; no candidate/old input changes, no F_aia/token/ZIP",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "candidate_sha256": report["candidate_sha256"],
                      "mutation_cases": mutations["rejected_cases"], "audit_json": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
