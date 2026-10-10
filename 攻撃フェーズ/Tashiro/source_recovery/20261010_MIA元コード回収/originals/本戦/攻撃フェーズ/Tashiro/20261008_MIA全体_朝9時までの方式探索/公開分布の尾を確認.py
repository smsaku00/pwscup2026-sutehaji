"""公開 A/C の性別別・属性別の中央と裾を、所属正解を使わず記述する。

先に --write-plan で方法.json を保存し、その後に引数なしで計算する。
実行はプロジェクトルートの .venv/bin/python -B に限定する。
既存の加工特徴スクリプト、背景プロフィール、採点結果は読み込まない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ATTRIBUTES = ("age", "BMI", "SBP", "TG", "HDL", "ALT", "FPG")
PROBABILITIES = (.01, .05, .25, .5, .75, .95, .99)
QUANTILE_NAMES = ("q01", "q05", "q25", "q50", "q75", "q95", "q99")
METHOD = {
    "version": 1,
    "scope": "公開Cと各フェーズの公開Aのみ。予備戦20対象、本戦22対象の全42対象。",
    "attributes": list(ATTRIBUTES),
    "sex_groups": [0, 1],
    "numeric_values": "CSVの値をそのまま使用。TG/ALTを含め対数変換しない。",
    "csv_float_precision": "round_trip",
    "quantiles": list(PROBABILITIES),
    "quantile_method": "numpy.quantile(method='linear')",
    "central_iqr_ratio": "(C_q75-C_q25)/(A_q75-A_q25)",
    "range_5_95_ratio": "(C_q95-C_q05)/(A_q95-A_q05)",
    "range_1_99_ratio": "(C_q99-C_q01)/(A_q99-A_q01)",
    "tail_fractions": "C<A_q01 と C>A_q99 の各行数・比率。境界と等しい値は裾に含めない。",
    "median_shift_in_a_iqr": "(C_q50-A_q50)/(A_q75-A_q25)。符号を保持する。",
    "wasserstein_in_a_iqr": "経験分布の1次Wasserstein距離をAのIQRで割る。",
    "reference": "各フェーズのAを性別で条件付け。都道府県・年齢での再重み付けはしない。",
    "replicate_rows": "重複行を含む公開Cの各行を等重みで扱う。",
    "uncertainty": "信頼区間・有意性検定・ブートストラップは計算しない。",
    "limits": [
        "有限の公開標本の記述値であり、元のコホート構成と加工の効果を切り分けない。",
        "性別ごとの比較でも都道府県・年齢・他属性の標本構成差が残る。",
        "Q、B、所属正解、攻撃の採点結果を読まず、MIA/AIA精度は測定しない。",
        "1-99%の幅は裾の少数行に影響され、IQRが狭い属性では比率が大きくなりやすい。",
        "同番号の予備戦と本戦が同じ加工アルゴリズムであるとは仮定しない。",
    ],
    "membership_labels_used": False,
    "b_self_used": False,
    "attack_scores_used": False,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def inputs() -> list[tuple[str, Path, Path, int, int]]:
    prelim = ROOT / "予備戦/配布データ"
    main = ROOT / "本戦/配布データ"
    main_background = sorted(main.glob("PWSCup2026_final_team28_*/A_bg.csv"))
    assert len(main_background) == 1, main_background
    return [
        ("prelim", prelim / "PWSCup2026_prelim_common/A_bg.csv",
         prelim / "PWSCup2026_prelim_attack", 5, 20),
        ("main", main_background[0], HERE.parent / "20260928_配布データ基本分析/入力", 28, 22),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "公開分布の尾の診断結果")
    parser.add_argument("--write-plan", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    plan = out / "方法.json"
    code_path = Path(__file__).resolve()
    code_hash = sha(code_path)
    if args.write_plan:
        assert not out.exists(), out
        out.mkdir()
        plan.write_text(json_text({"created_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
                                   "code_sha256_before_execution": code_hash, "method": METHOD}), encoding="utf-8")
        print(json_text({"stage": "method_saved_before_execution", "path": str(plan),
                         "method_sha256": sha(plan)}), end="")
        return
    assert out.is_dir() and {p.name for p in out.iterdir()} == {plan.name}, out
    saved = json.loads(plan.read_text(encoding="utf-8"))
    assert saved["method"] == METHOD
    assert saved["code_sha256_before_execution"] == code_hash
    assert Path(sys.executable).parent == ROOT / ".venv/bin", sys.executable
    started = time.perf_counter()
    input_hashes: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []

    def read_public(path: Path) -> pd.DataFrame:
        resolved = path.resolve()
        assert resolved.name == "A_bg.csv" or resolved.name.startswith("C_")
        assert not any(word in str(resolved) for word in ("アーカイブ", "B_self", "pool_ids", "正解データ"))
        input_hashes[str(resolved)] = sha(resolved)
        data = pd.read_csv(resolved, float_precision="round_trip")
        assert not data[list(ATTRIBUTES) + ["sex"]].isna().any().any(), resolved
        assert set(data.sex.unique()) == {0, 1}, resolved
        return data

    for phase, background_path, target_dir, self_id, expected_targets in inputs():
        a = read_public(background_path)
        target_paths = sorted(p for p in target_dir.glob("C_*.csv") if int(p.stem.rsplit("_", 1)[1]) != self_id)
        assert len(target_paths) == expected_targets, (phase, len(target_paths))
        source_rows.append({"phase": phase, "source": "phase_A", "path": str(background_path.resolve()),
                            "rows": len(a), "sex0_rows": int((a.sex == 0).sum()), "sex1_rows": int((a.sex == 1).sum())})
        for path in target_paths:
            team = int(path.stem.rsplit("_", 1)[1])
            c = read_public(path)
            source_rows.append({"phase": phase, "source": f"C_{team}", "path": str(path.resolve()),
                                "rows": len(c), "sex0_rows": int((c.sex == 0).sum()), "sex1_rows": int((c.sex == 1).sum())})
            for sex in (0, 1):
                aa = a.loc[a.sex == sex, list(ATTRIBUTES)].to_numpy(float)
                cc = c.loc[c.sex == sex, list(ATTRIBUTES)].to_numpy(float)
                assert np.isfinite(aa).all() and np.isfinite(cc).all()
                aq = np.quantile(aa, PROBABILITIES, axis=0, method="linear")
                cq = np.quantile(cc, PROBABILITIES, axis=0, method="linear")
                for j, attribute in enumerate(ATTRIBUTES):
                    a_iqr = float(aq[4, j] - aq[2, j])
                    a_90 = float(aq[5, j] - aq[1, j])
                    a_98 = float(aq[6, j] - aq[0, j])
                    assert min(a_iqr, a_90, a_98) > 0
                    below = int((cc[:, j] < aq[0, j]).sum())
                    above = int((cc[:, j] > aq[6, j]).sum())
                    row: dict[str, object] = {
                        "phase": phase, "team": team, "sex": sex, "attribute": attribute,
                        "a_rows": len(aa), "c_rows": len(cc),
                        "a_iqr": a_iqr, "c_iqr": float(cq[4, j] - cq[2, j]),
                        "central_iqr_ratio": float((cq[4, j] - cq[2, j]) / a_iqr),
                        "a_range_5_95": a_90, "c_range_5_95": float(cq[5, j] - cq[1, j]),
                        "range_5_95_ratio": float((cq[5, j] - cq[1, j]) / a_90),
                        "a_range_1_99": a_98, "c_range_1_99": float(cq[6, j] - cq[0, j]),
                        "range_1_99_ratio": float((cq[6, j] - cq[0, j]) / a_98),
                        "c_below_a_q01_rows": below, "c_above_a_q99_rows": above,
                        "c_below_a_q01_fraction": below / len(cc), "c_above_a_q99_fraction": above / len(cc),
                        "c_outside_a_q01_q99_fraction": (below + above) / len(cc),
                        "median_shift_in_a_iqr": float((cq[3, j] - aq[3, j]) / a_iqr),
                        "wasserstein_in_a_iqr": float(wasserstein_distance(aa[:, j], cc[:, j]) / a_iqr),
                    }
                    row.update({"a_" + name: float(aq[k, j]) for k, name in enumerate(QUANTILE_NAMES)})
                    row.update({"c_" + name: float(cq[k, j]) for k, name in enumerate(QUANTILE_NAMES)})
                    rows.append(row)
    table = pd.DataFrame(rows).sort_values(["phase", "team", "sex", "attribute"])
    assert len(table) == 42 * 2 * len(ATTRIBUTES)
    assert not table.duplicated(["phase", "team", "sex", "attribute"]).any()
    assert np.isfinite(table.select_dtypes("number").to_numpy()).all()
    assert (table.c_below_a_q01_fraction.between(0, 1) & table.c_above_a_q99_fraction.between(0, 1)).all()
    for path, expected in input_hashes.items():
        assert sha(Path(path)) == expected, path
    assert sha(code_path) == code_hash
    table.to_csv(out / "性別属性ごとの中央と尾.csv", index=False, float_format="%.17g")
    pd.DataFrame(source_rows).to_csv(out / "公開入力の行数.csv", index=False)
    output_hashes = {p.name: sha(p) for p in sorted(out.iterdir())}
    record = {
        "completed_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "seconds": time.perf_counter() - started,
        "python": sys.executable, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "code_path": str(code_path), "code_sha256": code_hash,
        "method_saved_before_execution": True, "method_sha256": sha(plan),
        "input_sha256": input_hashes, "output_sha256": output_hashes,
        "output_hash_scope": "実行記録.json自身は再帰ハッシュを避けるため対象外。",
        "targets": 42, "descriptor_rows": len(table),
        "membership_labels_used": False, "b_self_used": False, "attack_scores_used": False,
        "cohort_vs_processing": "コホート構成と加工効果の因果的な分解ではない。",
        "attack_accuracy_measured": False,
    }
    (out / "実行記録.json").write_text(json_text(record), encoding="utf-8")
    print(json_text({"stage": "public_tail_descriptors_completed", "output": str(out),
                     "targets": 42, "rows": len(table), "seconds": record["seconds"]}), end="")


if __name__ == "__main__":
    main()
