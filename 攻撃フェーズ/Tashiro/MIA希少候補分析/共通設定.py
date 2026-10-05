"""入出力の指定と、公開時点の入力ハッシュ確認。秘密データは同梱しない。"""
from pathlib import Path
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent


def configure(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--attack-dir", type=Path, required=True,
                        help="C_*.csv、aia_challenge_28.csv、mia_columns.json のフォルダ")
    parser.add_argument("--b-self", type=Path, required=True, help="自チーム28のB_self.csv")
    parser.add_argument("--background", type=Path, required=True, help="本戦A_bg.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE/"再計算結果",
                        help="集計・説明・検証記録の出力先。既定はコード横の再計算結果")
    options = parser.parse_args()
    for name in ["attack_dir", "b_self", "background", "output_dir"]:
        setattr(options, name, getattr(options, name).expanduser().resolve())
    return options


def checked_paths(options):
    manifest = json.loads((HERE/"入力ハッシュ.json").read_text(encoding="utf-8"))
    paths = {}
    for name, expected in manifest["sha256"].items():
        path = (options.b_self if name == "B_self.csv" else
                options.background if name == "A_bg.csv" else options.attack_dir/name)
        if not path.is_file():
            raise FileNotFoundError(f"必要な入力がありません: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"公開時点と異なる入力です: {name} (expected={expected}, actual={actual})")
        paths[name] = path
    return paths, manifest
