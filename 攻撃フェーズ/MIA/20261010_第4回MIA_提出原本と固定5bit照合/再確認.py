"""第4回保存原本の読み取り専用監査。標準Pythonのみ、ファイル更新なし。"""
import csv
import hashlib
import html
import importlib.util
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / "20261008_第3回MIA_固定5bitと非会員較正"
ATTACK = HERE.parent.parent

def read_scores(round_number):
    text = (ATTACK / "AIA" / f"第{round_number}回" / "detailed_results.html").read_text()
    return json.loads(html.unescape(text.split("<pre>", 1)[1].split("</pre>", 1)[0]))

def main():
    data = (HERE / "F_mia.csv").read_bytes()
    expected_hash = "45077532e5b5cf6109bf86ce67472fce0fac703584fa81a2f0027434fe581183"
    assert hashlib.sha256(data).hexdigest() == expected_hash
    raw = list(csv.DictReader(data.decode("utf-8-sig").splitlines()))
    settings = json.loads((PREV / "結果" / "設定.json").read_text())
    teams = settings["teams"]
    assert len(raw) == 1049
    assert list(raw[0]) == ["record_id"] + list(map(str, teams))
    rows = {int(r["record_id"]): r for r in raw}
    assert sorted(rows) == list(range(1049))
    rare = settings["rare_candidates"] + settings["known_negative_rare"]
    q = set(settings["fixed_queries"]["4"])
    checked = 0
    for record_id in rare:
        for team in teams:
            padding = set(settings["known_negative_rare"][:settings["padding_count"].get(str(team), 0)])
            expected = 0 if team in settings["zero_rare_teams"] else (0.4 if record_id in q | padding else 0.1)
            assert abs(float(rows[record_id][str(team)]) - expected) <= 1e-12, (record_id, team)
            checked += 1
    spec = importlib.util.spec_from_file_location("round3_decoder", PREV / "結果読取.py")
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    values = [str(read_scores(k)["mia"]["tpr_rare"]) for k in (3, 4)]
    assert values == ["0.017203703703703704", "0.010814814814814814"]
    result = decoder.interpret(values)
    groups = [result["remaining"][str(t)]["candidate_assignments"] for t in decoder.TEAMS]
    legal = [p for p in itertools.product(*groups) if len({i for group in p for i in group}) == 9]
    assert len(legal) == 1296
    assert not result["all_identified"]
    print(json.dumps({"passed": True, "F_mia_sha256": expected_hash, "rare_cells_checked": checked,
                      "mismatches": 0, "legal_joint_placements": len(legal), "decode": result,
                      "server_input_hash_checked": False, "writes_files": False}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
