"""第3回保存値を保ち、5対象の通常列と固定5bit第3問だけを移植する。

入力は事前に検証したCSVのみ。所属ラベル、AIA、tokenを読まない。
既存出力は上書きしない。MIA単独ZIPは配布用であり大会提出ZIPではない。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

TEAMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 25, 26, 27]
SELECTED = [5, 7, 13, 14, 15]
R3_SHA = "6ca9604a8936fb47255ae0bd05c9f3958368f0e196bc2af258f8255cf39c08eb"
R4_SHA = "45077532e5b5cf6109bf86ce67472fce0fac703584fa81a2f0027434fe581183"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def build(args):
    import numpy as np
    import pandas as pd
    require(not args.output.exists(), "既存出力先を上書きしない")
    require(sha(args.round3) == R3_SHA and sha(args.round4) == R4_SHA, "提出原本のhash不一致")
    headers, rows = read(args.round3)
    require(headers == ["record_id"] + list(map(str, TEAMS)), "第3回列順不一致")
    require([int(r["record_id"]) for r in rows] == list(range(1049)), "第3回ID順不一致")
    _, selected_rows = read(args.selected)
    _, partitions = read(args.partitions)
    _, rare_rows = read(args.rare)
    part = {int(r["record_id"]): r["partition"] for r in partitions}
    require(set(part) == set(range(1049)), "区分ID不足/重複")
    ordinary = {k for k, v in part.items() if v == "ordinary"}
    negative = {k for k, v in part.items() if v == "nonrare_known_negative"}
    rare = {k for k, v in part.items() if v == "rare"}
    require((len(ordinary), len(negative), len(rare)) == (875, 150, 24), "区分人数不一致")
    require(len(selected_rows) == 875 and {int(r["record_id"]) for r in selected_rows} == ordinary,
            "選択列の通常ID不一致")
    require(len(rare_rows) == 24 and {int(r["record_id"]) for r in rare_rows} == rare,
            "希少質問ID不一致")
    for r in selected_rows:
        rid = int(r["record_id"])
        for t in SELECTED:
            value = float(r[str(t)])
            require(np.isfinite(value) and .6 < value <= .9, "通常の値域不一致")
            rows[rid][str(t)] = format(value, ".17g")
    for r in rare_rows:
        rid = int(r["record_id"])
        for t in TEAMS:
            value = float(r[str(t)])
            require(value in (0., .1, .4), "希少値不一致")
            rows[rid][str(t)] = format(value, ".17g")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    candidate = buffer.getvalue().encode("utf-8")
    back = pd.read_csv(io.BytesIO(candidate), float_precision="round_trip")
    r3 = pd.read_csv(args.round3, float_precision="round_trip")
    r4 = pd.read_csv(args.round4, float_precision="round_trip")
    require(np.isfinite(back.iloc[:, 1:]).all().all(), "非有限値")
    require(((back.iloc[:, 1:] >= 0) & (back.iloc[:, 1:] <= 1)).all().all(), "[0,1]違反")
    require(np.array_equal(back.record_id, r3.record_id) and list(back.columns) == headers, "順序不一致")
    unchanged = [str(t) for t in TEAMS if t not in SELECTED]
    require(np.array_equal(back.loc[sorted(ordinary), unchanged], r3.loc[sorted(ordinary), unchanged]),
            "5対象外の通常値変更")
    require(np.array_equal(back["9"], r3["9"]), "Haruna全列固定違反")
    require((back.loc[sorted(negative), list(map(str, TEAMS))] == 0).all().all(), "非希少非共有150人の非零")
    rare_expected = pd.read_csv(args.rare, float_precision="round_trip").set_index("record_id")
    require(np.array_equal(back.loc[rare_expected.index, list(map(str, TEAMS))], rare_expected[list(map(str, TEAMS))]),
            "希少528セル不一致")
    require(back.loc[sorted(ordinary), list(map(str, TEAMS))].min().min() > .6 and
            back.loc[sorted(rare), list(map(str, TEAMS))].max().max() <= .4, "通常希少分離違反")
    args.output.mkdir(parents=True)
    (args.output / "F_mia.csv").write_bytes(candidate)
    with zipfile.ZipFile(args.output / "MIA_round5_components.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        info = zipfile.ZipInfo("F_mia.csv", (2026, 10, 10, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        z.writestr(info, candidate)
    differences = []
    for t in TEAMS:
        for label, old in [("round3", r3), ("round4", r4)]:
            differences.append({"baseline": label, "main_team": t,
                "ordinary_changed_cells": int((back.loc[sorted(ordinary), str(t)] != old.loc[sorted(ordinary), str(t)]).sum()),
                "rare_changed_cells": int((back.loc[sorted(rare), str(t)] != old.loc[sorted(rare), str(t)]).sum()),
                "nonrare_known_negative_changed_cells": int((back.loc[sorted(negative), str(t)] != old.loc[sorted(negative), str(t)]).sum()),
                "all_changed_cells": int((back[str(t)] != old[str(t)]).sum())})
    pd.DataFrame(differences).to_csv(args.output / "differences.csv", index=False, lineterminator="\n")
    dump(args.output / "assembly_validation.json", {"passed": True, "submitted": False, "rows": 1049,
        "targets": TEAMS, "selected_ordinary_targets": SELECTED, "ordinary_count": 875,
        "nonrare_known_negative_count": 150, "rare_count": 24, "rare_overlap_known_negative_count": 13,
        "ordinary_outside_five_changed_cells": 0, "haruna_all_changed_cells": 0,
        "rare_expected_cells": 528, "rare_question_number": 3, "rare_question_submission_round": 5,
        "normal_min": float(back.loc[sorted(ordinary), list(map(str, TEAMS))].min().min()),
        "normal_max": float(back.loc[sorted(ordinary), list(map(str, TEAMS))].max().max()),
        "rare_max": float(back.loc[sorted(rare), list(map(str, TEAMS))].max().max()),
        "input_sha256": {"round3": sha(args.round3), "round4": sha(args.round4), "selected": sha(args.selected),
                         "partitions": sha(args.partitions), "rare_expected": sha(args.rare)},
        "F_mia_sha256": sha(args.output / "F_mia.csv"),
        "component_zip_sha256": sha(args.output / "MIA_round5_components.zip"),
        "archive_is_official_submission": False,
        "known_negative_exception": "希少と重なる13人は原仕様どおり質問値。非希少150人のみ常に0。",
        "round4_difference_note": "第4回通常は別のチーム別6方式。第4回比は5対象変更だけの効果ではない。"})
    print(json.dumps({"passed": True, "F_mia_sha256": sha(args.output / "F_mia.csv"), "submitted": False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["round3", "round4", "selected", "partitions", "rare", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    build(p.parse_args())
