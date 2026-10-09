#!/usr/bin/env python3
"""Run the bundled official main-round format validator; never print a token."""
from __future__ import annotations
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "official_validator_snapshot"))
from pwscup2026.common.config import load_config
from pwscup2026.kit.validate import check_package, load_dist, validate_aia_submission, validate_mia_submission


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--attack-dir", type=Path, required=True)
    p.add_argument("--base-submission", type=Path)
    p.add_argument("--team-id", type=int, default=28)
    args = p.parse_args()
    cfg = load_config(Path(__file__).resolve().parent /
                      "official_validator_snapshot/pwscup2026/kit/kit_config.yaml")
    dist = load_dist(args.attack_dir, team_id=args.team_id)
    # Main-round record IDs are released as 0..1048. Check the copied matrix.
    dist["attacker_record_ids"] = set(range(1049))
    package = check_package(args.submission, cfg)
    if not package.ok:
        raise ValueError(package.errors)
    for name, function in [("F_aia.csv", validate_aia_submission), ("F_mia.csv", validate_mia_submission)]:
        result = function(package.frames[name], cfg, dist)
        if not result.ok:
            raise ValueError(result.errors)
    aia = package.frames["F_aia.csv"]
    values = aia.time_hat.to_numpy(float)
    if not np.isfinite(values).all() or not (((values >= 0) & (values <= 10)) | (values == 11)).all():
        raise ValueError("This candidate must contain finite 0..10 or 11 values")
    targets = {int(k) for k in dist["mia_columns"]["target_columns"]} - {args.team_id}
    if set(aia.target_k) != targets or len(aia) != 326 * len(targets):
        raise ValueError("Candidate target set differs from released main-round opponents")
    if args.base_submission:
        with zipfile.ZipFile(args.submission) as new, zipfile.ZipFile(args.base_submission) as old:
            if new.read("F_mia.csv") != old.read("F_mia.csv") or new.read("token.txt") != old.read("token.txt"):
                raise ValueError("MIA/token differs from the specified base ZIP")
    print(json.dumps({"format_validation": "PASS", "rows": len(aia),
        "opponents": len(targets), "answers_0_to_10": int((values != 11).sum()),
        "answers_11": int((values == 11).sum()),
        "base_MIA_token_match": True if args.base_submission else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
