"""掲載する分析ファイルを明示し、内容のハッシュを記録する。

GitHubやCodaBenchへ接続しない。公式入力・個人別予測・提出物・途中版は含めない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

RESULT_FOLDERS = ["結果", "追加結果_最終版", "補助結果", "参照結果_拡張版", "背景再標本化結果",
                  "学習結果_順位校正版", "属性部分集合結果", "低FPR結果", "固定候補の感度結果",
                  "非会員較正結果", "非会員較正の感度結果"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    if args.output_file is not None and args.output_file.exists():
        raise FileExistsError("既存の掲載記録を上書きしません。")
    d = args.analysis_dir
    selected = [d / "README.md", d / "説明.md", d / "目標と公式結果.json", d / "集計検証結果.json"]
    selected += sorted(d.glob("*.py"))
    for folder in RESULT_FOLDERS:
        selected += sorted((d / folder).glob("*.csv"))
    selected += sorted((d / "共有集計_確認版").iterdir())
    assert len(selected) == len(set(selected))
    entries = []
    for path in selected:
        relative = path.relative_to(d).as_posix()
        assert path.is_file() and path.suffix in (".md", ".py", ".json", ".csv", ".png", ".svg"), path
        assert path.name not in ("F_mia.csv", "F_aia.csv", "token.txt") and "アーカイブ" not in relative
        content = path.read_bytes()
        entries.append({"path": relative, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                        "git_blob_sha1": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
                        "kind": "figure" if path.suffix in (".png", ".svg") else "analysis"})
    result = {"created_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
              "repository": "smsaku00/pwscup2026-sutehaji", "branch": "main",
              "destination": "攻撃フェーズ/Tashiro/20261007_MIA全体改善の検討/",
              "purpose": "MIA全体の方法検討・集計・説明を共有する。提出ではない。",
              "files": entries, "total_bytes": sum(entry["bytes"] for entry in entries),
              "manifest_note": "この一覧自身は自己参照を避けてfilesに含めない。GitHubのtreeで各blobを照合する。",
              "codabench_access_or_submission": False}
    if args.output_file is not None:
        args.output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(entries), "total_bytes": result["total_bytes"],
                      "destination": result["destination"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
