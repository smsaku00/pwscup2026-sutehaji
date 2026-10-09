#!/usr/bin/env python3
"""Package a saved AIA candidate using a local Team-28 submission ZIP."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd

from reproduce_aia import METHODS, base_zip_bytes, build_submission, require, validate_answers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Output exists; use a new ZIP filename")
    targets = list(range(18)) + [22, 25, 26, 27]
    candidate = Path(__file__).resolve().parent / "generated" / args.method / "F_aia.csv"
    aia_bytes = candidate.read_bytes()
    frame = pd.read_csv(io.BytesIO(aia_bytes))
    validate_answers(frame, targets)
    mia_bytes, token = base_zip_bytes(args.base_submission, targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_submission(args.output, mia_bytes, aia_bytes, token)
    print(json.dumps({"submission": str(args.output), "method": args.method,
                      "rows": len(frame), "MIA_and_token_copied_byte_for_byte": True},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
