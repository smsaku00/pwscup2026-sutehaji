from __future__ import annotations

import argparse
import csv
import hashlib
import io
from pathlib import Path

SOURCE_SHA = "20159ce25b547e6644a938e1cac9bcac03fb5be6c7cad8508f102a4663e26a8c"
OUTPUT_SHA = "5b28253a5d64543257cbb411a6fa42352080c260509129a8e4f05c9b6ff3f184"
RARE = {47,76,90,115,126,136,169,264,271,273,304,317,383,440,462,480,495,625,637,643,668,675,814,841}
MEMBERS = {6:{90},10:{304,383},11:{169},16:{76,462},17:{115},25:{264,675}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round7", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.round7.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA, "第7回原本のSHA-256が不一致"
    assert not args.out.exists(), "既存ファイルを上書きしません"
    lines = raw.decode("utf-8").splitlines(keepends=True)
    header = next(csv.reader([lines[0]]))
    targets = [int(c.removeprefix("Team")) for c in header[1:]]
    assert len(targets) == 22 and len(lines) == 1050
    changed = []
    for line in lines[1:]:
        cells = next(csv.reader([line]))
        record_id = int(cells[0])
        if record_id in RARE:
            cells[1:] = ["1" if record_id in MEMBERS.get(t,set()) else "0" for t in targets]
            buf = io.StringIO(newline="")
            csv.writer(buf,lineterminator="\n").writerow(cells)
            changed.append(buf.getvalue())
        else:
            changed.append(line)
    result = (lines[0] + "".join(changed)).encode("utf-8")
    assert hashlib.sha256(result).hexdigest() == OUTPUT_SHA, "再現結果のSHA-256が不一致"
    with args.out.open("xb") as f:
        f.write(result)
    print("PASS", OUTPUT_SHA)

if __name__ == "__main__":
    main()
