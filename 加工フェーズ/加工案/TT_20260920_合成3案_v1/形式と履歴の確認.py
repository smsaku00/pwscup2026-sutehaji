#!/usr/bin/env python3
"""不正入力の拒否と旧出力保存を確認する。配布データや正式なCは更新しない。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import pandas as pd


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="新規の確認記録JSON")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("確認記録は新しいファイル名にしてください")
    records = []
    # 加工案直下の3方式だけを検査し、他メンバーのファイルは対象にしない。
    folders = ['TT-01-背景条件合成', 'TT-02-群集約合成', 'TT-03-多元ドナー合成']
    for folder in folders:
        script = here.parent / folder / (folder+".py")
        spec = importlib.util.spec_from_file_location("tt_check_"+script.stem, script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # 自分で生成したCに仮IDを付けた、検査用の入力を作る。
        cpath = script.parent/("C_"+script.stem+".csv")
        valid = pd.read_csv(cpath, float_precision="round_trip")
        valid.insert(0, "record_id", range(len(valid)))
        cases = {}
        bad = valid.copy(); bad.loc[0, "BMI"] = float("inf"); cases["非有限値"] = bad
        bad = valid.copy(); bad.loc[0, "SBP"] = None; cases["欠測"] = bad
        bad = valid.copy(); bad.loc[0, "BMI"] = 60; cases["値域外"] = bad
        bad = valid.copy(); bad["age"] = bad.age.astype(float); bad.loc[0, "age"] = 40.5; cases["整数条件違反"] = bad
        bad = valid.copy(); bad.loc[0, "sex"] = 2; cases["二値条件違反"] = bad
        bad = valid.copy(); bad.loc[0, "prefecture"] = "不明"; cases["県名不正"] = bad
        bad = valid.copy(); bad.loc[0, "record_id"] = bad.loc[1, "record_id"]; cases["重複ID"] = bad
        bad = valid.copy(); bad["is_rare"] = 0; cases["正解列の混入"] = bad
        cases["不足列"] = valid.drop(columns="BMI")
        bad = valid.copy(); bad.loc[0, ["onset", "death", "time"]] = [0, 0, 8]; cases["打ち切り不整合"] = bad
        bad = valid.copy(); bad.loc[0, "time"] = 0; cases["時刻不正"] = bad
        for label, df in cases.items():
            rejected = False
            try:
                module.validate(df, label)
            except ValueError:
                rejected = True
            records.append(dict(method=script.stem, check=label, ok=rejected))
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            # 重複列はpandasで名前が変わる前のヘッダー検査が拒否する必要がある。
            malformed = temp/"重複列.csv"
            malformed.write_text("record_id,record_id\n1,2\n", encoding="utf-8")
            rejected = False
            try:
                module.read_input(malformed)
            except ValueError:
                rejected = True
            records.append(dict(method=script.stem, check="CSV重複ヘッダー", ok=rejected))
            # 古い内容を失わず新しい内容を保存することを、一時ファイルで検証する。
            output = temp/"C_確認.csv"
            old, new = b"old-content\n", b"new-content\n"
            module.save_preserving(output, old)
            module.save_preserving(output, new)
            archived = temp/"旧出力"/("C_確認_"+hashlib.sha256(old).hexdigest()[:16]+".csv")
            records.append(dict(method=script.stem, check="旧出力の保持", ok=output.read_bytes() == new and archived.read_bytes() == old))
    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if not all(r["ok"] for r in records):
        raise RuntimeError("不正入力・履歴の確認に失敗しました")
    print(f"{len(records)}件の形式・履歴確認が成功")


if __name__ == "__main__":
    main()
