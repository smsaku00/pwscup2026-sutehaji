"""検証済MIAと指定されたAIA・配布tokenからローカル提出ZIPを組む。

token内容/hashは表示・記録しない。ZIPはGitHub掲載しないこと。
このスクリプトは大会への送信を行わない。
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["mia", "aia", "token", "dist", "attacker-b", "kit-source", "validator", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), "既存ZIPを上書きしない")
    # 2026-10-10現在の公式main検証器。現行snapshotに合わせてfail-closed。
    body = args.validator.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    require(blob == "07284c2239bfcd98dbadd17c69345490461718dc", "公式検証器snapshotが違う")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.kit_source))
    import pwscup2026.kit
    spec = importlib.util.spec_from_file_location("pwscup2026.kit.round5_validator", args.validator)
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)
    cfg = validator.config_mod.load_config(args.kit_source / "pwscup2026/kit/kit_config.yaml")
    dist = validator.load_dist(args.dist, team_id=28)
    # 本番truthは不要。参加者配布B_selfとtargetsを参照する。
    dist["team_id"] = 28
    import pandas as pd
    dist["attacker_record_ids"] = set(pd.read_csv(args.attacker_b, usecols=["record_id"])["record_id"].astype(int))
    require("mia_columns" in dist and "challenge" in dist, "本戦の標的/チャレンジ参照が不足")
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, source in [("F_mia.csv", args.mia), ("F_aia.csv", args.aia), ("token.txt", args.token)]:
            info = zipfile.ZipInfo(name, (2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            z.writestr(info, source.read_bytes())
    pkg = validator.check_package(args.output, cfg)
    errors = list(pkg.errors)
    if pkg.ok:
        errors += validator.validate_mia_submission(pkg.frames["F_mia.csv"], cfg, dist).errors
        errors += validator.validate_aia_submission(pkg.frames["F_aia.csv"], cfg, dist).errors
    if errors:
        args.output.unlink()
        raise ValueError("公式ZIP検証不合格: " + "; ".join(errors))
    print(json.dumps({"passed": True, "submitted": False, "zip_created_locally": True,
                      "GitHub_upload_allowed": False, "members": ["F_mia.csv", "F_aia.csv", "token.txt"]}))


if __name__ == "__main__":
    main()
