"""このフォルダの保存結果と入力不変を検証する。新予測・提出物は作らない。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import 分析 as core

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    checks = dict(input_paths=set(), score_rows=0)
    for source, record in [("分析.py", "実行記録.json"), ("推定時間の寄与.py", "寄与検証_実行記録.json")]:
        data = json.loads((HERE/record).read_text())
        assert data["code_sha256"] == sha(HERE/source)
        if source == "分析.py":
            assert data["reference_code_sha256"] == sha(core.OLD/"分析.py")
            assert data["official_score_check"] == dict(official_function_checked=True, cases=63)
            assert data["changed_predictions"] == 0 and data["permutation_comparisons"] == 396
        else:
            assert data["prediction_code_sha256"] == sha(HERE/"分析.py")
        assert data["submission_created"] is False and data["codabench_submitted"] is False
        for path, expected in data["input_sha256"].items():
            assert sha(Path(path)) == expected
            checks["input_paths"].add(path)
    out = HERE/"集計"
    target = pd.read_csv(out/"予備戦_対象別.csv")
    average = pd.read_csv(out/"予備戦_20対象と18対象平均.csv")
    assert len(target) == 21*9
    metrics = ["r_all", "r_rare", "r_max", "p_m", "p_c", "r_unclipped"]
    for group, excluded in [("20対象_自5除外", [5]), ("18対象_1_2_5除外", [1,2,5])]:
        calculated = target[~target.team.isin(excluded)].groupby("method")[metrics].mean().sort_index()
        saved = average[average.group.eq(group)].set_index("method")[metrics].sort_index()
        assert np.max(np.abs(calculated.to_numpy()-saved.to_numpy())) < 1e-12
    p = pd.read_csv(out/"Cと背景の行順変更.csv")
    assert len(p) == 44*9 and p.changed_predictions.sum() == 0
    contribution = pd.read_csv(out/"時間関係の寄与_全反復.csv")
    assert len(contribution) == 20*(20*2+3)
    assert set(contribution[contribution.source.eq("time_shuffle")].repeat) == set(range(20))
    for col in ["r_all", "r_rare", "r_max", "p_m", "p_c"]:
        assert target[col].between(0, 1).all() and contribution[col].between(0, 1).all()
    checks["score_rows"] = len(target)+len(contribution)
    # 連続時間窓の最大質量選択が、密な格子による下界を下回らないことを確認。
    rng = np.random.default_rng(core.SEED+99)
    grid = np.linspace(0, 10, 10001)
    checks["window_optimizer_cases"] = 0
    for _ in range(100):
        t = np.sort(rng.uniform(0, 10, 16)); t[-3:] = 10
        w = rng.uniform(.1, 1, len(t)); w /= w.sum()
        bt = np.sort(rng.uniform(0, 10, 24)); bt[-6:] = 10
        bw = rng.uniform(.1, 1, len(bt)); bw /= bw.sum()
        for contrast in [False, True]:
            pred, optimum = core.best_window(t, w, bt if contrast else None, bw if contrast else None)
            mass = ((np.abs(grid[:, None]-t) <= .5)*w).sum(1)
            selected = float(w[np.abs(pred-t) <= .5].sum())
            if contrast:
                mass -= ((np.abs(grid[:, None]-bt) <= .5)*bw).sum(1)
                selected -= float(bw[np.abs(pred-bt) <= .5].sum())
            assert selected >= mass.max()-1e-10 and abs(selected-optimum) < 1e-10
            checks["window_optimizer_cases"] += 1
    forbidden = [p.name for p in HERE.rglob("*") if p.is_file() and
                 (p.name in {"F_aia.csv", "F_mia.csv", "token.txt"} or p.suffix == ".zip")]
    assert not forbidden
    checks["unique_input_hashes_verified"] = len(checks.pop("input_paths"))
    checks["score_and_summary_consistent"] = True
    checks["phi_reproduced"] = False
    checks["submission_artifacts_found"] = forbidden
    checks["verification_code_sha256"] = sha(Path(__file__))
    core.dump(HERE/"保存結果の検証.json", checks)
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == "__main__":
    run()
