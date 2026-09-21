#!/usr/bin/env python3
"""正解を使わない動作確認。攻撃プログラムとは別プロセス・別ファイルで実行する。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path, help="main-process-20260912キットのルート")
    args = parser.parse_args()
    kit = args.kit.resolve()
    dist = kit / "participant_data"
    out = Path(__file__).resolve().parent
    scripts = sorted(out.glob("TT-*/TT-*.py"))
    if len(scripts) != 3:
        raise ValueError("3本の攻撃プログラムが必要です")
    # 参照するのは形式検査だけ。truthやB_selfは読み込まない。
    sys.path.insert(0, str(kit / "starter/src"))
    from pwscup2026.kit.validate import validate_aia_submission

    used = [dist / "B_practice.csv", dist / "A_bg.csv"]
    used += [dist / "targets" / f"{prefix}_{k}.csv" for k in (1, 2, 3)
             for prefix in ("C", "aia_challenge")]
    before = {str(p): digest(p) for p in used}
    summary = {"scope": "toy_functional_checks_only", "scoring_performed": False,
               "reference_code": "pwscup2026-kit-main_revised0912", "programs": []}
    with tempfile.TemporaryDirectory(prefix="tt-aia-verify-") as tmp:
        work = Path(tmp)
        # Bを一切開かないことを検査するため、内容がCSVでないファイルも使う。
        unused_b = work / "unused_B.csv"
        unused_b.write_text("この内容は攻撃者の学習入力ではありません。\n", encoding="utf-8")
        targets = [{"target_k": k, "c_path": str(dist / "targets" / f"C_{k}.csv"),
                    "aia_challenge_path": str(dist / "targets" / f"aia_challenge_{k}.csv")}
                   for k in (1, 2)]
        default_targets = work / "targets.json"
        save_json(default_targets, targets)
        official = work / "mia_columns.json"
        save_json(official, {"target_columns": ["1", "2", "99"]})
        expected = [(k, int(rid)) for k in (1, 2)
                    for rid in sorted(pd.read_csv(dist / "targets" / f"aia_challenge_{k}.csv").challenge_row_id)]
        for source in scripts:
            run_dir = work / source.stem
            run_dir.mkdir()
            script = run_dir / source.name
            shutil.copy2(source, script)
            name = source.stem
            output = run_dir / f"F_aia_{name}.csv"
            record = run_dir / f"実行記録_{name}.json"
            checks = []

            def execute(target_path=default_targets, bg_path=dist / "A_bg.csv",
                        b_path=dist / "B_practice.csv", extra=(), success=True):
                cmd = [sys.executable, "-B", str(script), "--b", str(b_path),
                       "--a-bg", str(bg_path), "--targets", str(target_path), *extra]
                proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
                if success and proc.returncode != 0:
                    raise AssertionError(f"{name}: {proc.stderr}")
                if not success and (proc.returncode != 2 or "エラー:" not in proc.stderr):
                    raise AssertionError(f"不正入力を理由付きで拒否できません: {name}: {proc.stderr}")
                return proc

            execute()
            first = output.read_bytes()
            run1 = json.loads(record.read_text())
            execute()
            run2 = json.loads(record.read_text())
            assert output.read_bytes() == first
            checks.append("同じ入力2回のCSVバイト列一致")
            df = pd.read_csv(output)
            assert list(df) == ["target_k", "challenge_row_id", "time_hat"]
            assert list(zip(df.target_k, df.challenge_row_id)) == expected
            assert not df[["target_k", "challenge_row_id"]].duplicated().any()
            assert np.isfinite(df.time_hat).all() and ((df.time_hat > 0) & (df.time_hat <= 10)).all()
            assert b"\r" not in first and not first.startswith(b"\xef\xbb\xbf")
            refs = {"mia_columns": {"target_columns": ["1", "2"]},
                    "challenge": {k: {rid for kk, rid in expected if kk == k} for k in (1, 2)}}
            valid = validate_aia_submission(df, {}, refs)
            assert valid.ok, valid.errors
            checks.append("652行の過不足・値域・UTF-8/LF・指定版公式形式検査")
            execute(b_path=unused_b)
            assert output.read_bytes() == first
            checks.append("Bを非CSVへ差し替えても同じ回答（内容不使用）")

            shuffled = []
            for target in targets[::-1]:
                k = target["target_k"]
                cf, qf = work / f"C_任意名_{k}.csv", work / f"課題_{k}.csv"
                pd.read_csv(target["c_path"]).iloc[::-1].to_csv(cf, index=False, float_format="%.17g")
                pd.read_csv(target["aia_challenge_path"]).iloc[::-1].to_csv(qf, index=False, float_format="%.17g")
                shuffled.append({"target_k": k, "c_path": cf.name, "aia_challenge_path": qf.name})
            # 元の文字列を保持して行だけ逆順にする。浮動小数の再保存誤差を混ぜない。
            for original, revised in zip(targets[::-1], shuffled):
                for key in ("c_path", "aia_challenge_path"):
                    lines = Path(original[key]).read_text().splitlines()
                    (work / revised[key]).write_text("\n".join([lines[0]] + lines[1:][::-1]) + "\n")
            shuf_path = work / "shuffled.json"
            save_json(shuf_path, shuffled)
            bg_lines = (dist / "A_bg.csv").read_text().splitlines()
            bg_shuf = work / "背景逆順.csv"
            bg_shuf.write_text("\n".join([bg_lines[0]] + bg_lines[1:][::-1]) + "\n")
            execute(shuf_path, bg_shuf)
            assert output.read_bytes() == first
            checks.append("対象順・C行順・課題行順・背景行順変更に不変／相対パスと任意C名")

            swapped = work / "swapped.json"
            save_json(swapped, [{"target_k": 3, "c_path": str(dist / "targets/C_3.csv"),
                                 "aia_challenge_path": str(dist / "targets/aia_challenge_3.csv")}])
            execute(swapped)
            changed = pd.read_csv(output)
            assert set(changed.target_k) == {3} and len(changed) == 326
            assert (run_dir / "履歴" / output.name).is_dir()
            checks.append("対象3へ差し替え／326行／旧CSV保存")
            execute(extra=["--target-set", "official", "--mia-columns", str(official), "--self-team-id", "99"])
            assert output.read_bytes() == first
            assert json.loads(record.read_text())["target_set"] == "official"
            checks.append("公式集合照合の仕組み（架空ID99を使った機能テスト）")

            invalid = work / "invalid.json"
            cases = [targets + [targets[0]], targets[:1],
                     [{"target_k": 1, "c_path": targets[0]["c_path"]}],
                     [{**targets[0], "truth_path": "forbidden.csv"}]]
            for i, case in enumerate(cases):
                save_json(invalid, case)
                extra = ["--target-set", "official", "--mia-columns", str(official), "--self-team-id", "99"] if i == 1 else []
                execute(invalid, extra=extra, success=False)
                assert output.read_bytes() == first
            save_json(invalid, targets + [{**targets[0], "target_k": 99}])
            execute(invalid, extra=["--target-set", "official", "--mia-columns", str(official), "--self-team-id", "99"], success=False)
            checks.append("重複対象・公式対象不足・自己対象・課題欠落・正解パス追加を拒否")

            module = load(script, "test_" + source.stem[:5].replace("-", "_"))
            base = pd.read_csv(targets[0]["aia_challenge_path"])
            bad_q = work / "bad_challenge.csv"
            errors = []
            v = base.copy(); v.loc[1, "challenge_row_id"] = v.loc[0, "challenge_row_id"]; errors.append(v)
            v = base.copy(); v.loc[0, "TG"] = np.nan; errors.append(v)
            v = base.copy(); v.loc[0, "ALT"] = np.inf; errors.append(v)
            v = base.copy(); v.loc[0, "age"] = 90; errors.append(v)
            v = base.copy(); v["is_rare"] = 0; errors.append(v)
            v = base.copy(); v["challenge_row_id"] = v.challenge_row_id.astype(float); v.loc[0, "challenge_row_id"] = 0.5; errors.append(v)
            for v in errors:
                v.to_csv(bad_q, index=False)
                try:
                    module.read_table(bad_q, "challenge")
                except module.InputError:
                    pass
                else:
                    raise AssertionError("不正課題の拒否に失敗")
            bad_q.write_text("challenge_row_id,challenge_row_id\n1,1\n")
            try:
                module.read_table(bad_q, "challenge")
            except module.InputError:
                pass
            else:
                raise AssertionError("重複ヘッダーの拒否に失敗")
            checks.append("重複ID・欠損・無限大・値域違反・非整数ID・評価列・重複列を拒否")
            assert abs(module.interval_mode([2.0, 2.8, 10.0], [0.3, 0.3, 0.4]) - 2.4) < 1e-12
            assert module.interval_mode([1.0, 2.0], [0.5, 0.5]) == 1.5
            assert module.interval_mode([2.0, 10.0], [0.5, 0.5]) == 10.0
            checks.append("±0.5年の区間選択・端点・同点処理の手計算例")
            if name.startswith("TT-03"):
                # 純粋な機能検査。同じ属性に同じtimeならそのtimeを返す。
                c, _ = module.read_table(Path(targets[0]["c_path"]), "c")
                q, _ = module.read_table(Path(targets[0]["aia_challenge_path"]), "challenge")
                bg, _ = module.read_table(dist / "A_bg.csv", "background")
                c["time"] = 4.0
                prediction, _ = module.predict(c, q.iloc[:5], bg, module.scale_from_background(bg))
                assert np.all(prediction == 4.0)
                checks.append("一定timeの人工入力で同じ値を復元")
            summary["programs"].append({"name": name, "code_sha256": digest(source),
                                        "csv_sha256": hashlib.sha256(first).hexdigest(),
                                        "rows": len(df), "checks": checks,
                                        "run_seconds": [run1["elapsed_seconds"], run2["elapsed_seconds"]],
                                        "peak_rss_mib": [run1["peak_rss_mib"], run2["peak_rss_mib"]],
                                        "environment": run1["environment"]})
            print(name + ": 全確認通過", flush=True)
    assert before == {str(p): digest(p) for p in used}
    summary["input_files_unchanged"] = True
    summary["input_sha256"] = {os.path.relpath(p, out): h for p, h in before.items()}
    save_json(out / "動作確認結果.json", summary)
    print("動作確認結果.json を保存。攻撃性能の採点は未実施。")


if __name__ == "__main__":
    main()
