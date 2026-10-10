"""Reproduce round-7 MIA from the two pinned official submission CSVs.

Use Python 3.12. No training, reranking, rescaling or competition submission.
Only the explicitly named output file is written; existing files are refused.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import math
from pathlib import Path

TARGETS = tuple(range(18)) + (22, 25, 26, 27)
ACTIVE = (6, 10, 11, 16, 17, 25)
QUESTION5 = (47, 76, 90, 115, 169)
NEGATIVE_RARE = (126, 136, 271, 273, 317, 440, 480, 625, 637, 643, 668, 814, 841)
UNKNOWN_RARE = (47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675)
RARE_IDS = frozenset(NEGATIVE_RARE + UNKNOWN_RARE)
PADDING_COUNTS = {6: 1, 10: 2, 11: 2, 16: 6, 17: 5, 25: 7}
ROUND1_SHA256 = "35ee16be3d0bb56bad2db8891aba3b75eb28be413cd8400bec165363d24499af"
ROUND6_SHA256 = "a5c378d31b44f1bd4d63ab8e812a54cd34a5c5242d06cfc57917ce04c86d9918"

def read_pinned(path: Path, expected_hash: str) -> list[list[str]]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError(f"Input hash differs: {path}")
    lines = raw.decode("utf-8").splitlines()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
    if rows != [line.split(",") for line in lines]:
        raise ValueError("Ambiguous CSV representation")
    if len(rows) != 1050 or rows[0] != ["record_id", *map(str, TARGETS)]:
        raise ValueError("Input shape or columns differ")
    if any(len(row) != 23 for row in rows):
        raise ValueError("Input row width differs")
    if [row[0] for row in rows[1:]] != list(map(str, range(1049))):
        raise ValueError("Input record_id order differs")
    return rows

def build(round1: Path, round6: Path) -> bytes:
    first = read_pinned(round1, ROUND1_SHA256)
    sixth = read_pinned(round6, ROUND6_SHA256)
    result = [row.copy() for row in first]
    for rid in range(1049):
        if rid not in RARE_IDS:
            result[rid + 1][10] = sixth[rid + 1][10]  # record_id + target columns 0..9
            continue
        for col, target in enumerate(TARGETS, start=1):
            if target not in ACTIVE:
                value = "0"
            else:
                high_ids = set(QUESTION5) | set(NEGATIVE_RARE[:PADDING_COUNTS[target]])
                value = "0.4" if rid in high_ids else "0.1"
            result[rid + 1][col] = value
    for row in result[1:]:
        if not all(math.isfinite(float(s)) and 0 <= float(s) <= 1 for s in row[1:]):
            raise ValueError("Nonfinite or out-of-range output")
    return ("\n".join(",".join(row) for row in result) + "\n").encode("utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--round6", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = build(args.round1, args.round6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as handle:
        handle.write(output)
    print(f"Created {args.out}; SHA256 {hashlib.sha256(output).hexdigest()}")

if __name__ == "__main__":
    main()
