"""同じ加工行への対応が競合する課題を、部分一対一対応で調べる。

元行を保持する加工に向く仮説の検証。合成Cに対応があるとは仮定しない。
提出CSVは作らず、予備戦公開正解による集計と本戦の動作診断だけを保存する。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
import 分析 as core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def partial_assignment(pair_score, selected_n):
    nq, nc = pair_score.shape
    # 加工行の使い回しを許さず、選択人数だけを固定する。
    dummy = nq - selected_n
    size = nc + dummy
    cost = np.zeros((size, size), dtype=float)
    cost[:nq, :nc] = -pair_score
    cost[nq:, nc:] = 1e6
    ri, ci = linear_sum_assignment(cost)
    assigned = ci[:nq]
    assert np.array_equal(ri, np.arange(size))
    selected = assigned < nc
    assert selected.sum() == selected_n
    assert np.unique(assigned[selected]).size == selected_n
    return selected, assigned


def predict(q, c, bg):
    b = core.raw(bg)
    scale = np.maximum(np.quantile(b, .75, axis=0) - np.quantile(b, .25, axis=0), 1e-5)
    x, y, z = (core.raw(v) / scale for v in (q, c, bg))
    pc = np.sort((x[:, None] - y[None]) ** 2, axis=2)
    pb = np.sort((x[:, None] - z[None]) ** 2, axis=2)
    mismatch_c = q.sex.to_numpy()[:, None] != c.sex.to_numpy()[None]
    mismatch_b = q.sex.to_numpy()[:, None] != bg.sex.to_numpy()[None]
    pred, diagnostics = {}, []
    for n in [2, 3, 5]:
        for cat in ["sex", "both"]:
            dc, db = pc[:, :, :n].mean(2), pb[:, :, :n].mean(2)
            dc[mismatch_c] = np.inf
            db[mismatch_b] = np.inf
            if cat == "both":
                dc[q.prefecture.to_numpy()[:, None] != c.prefecture.to_numpy()[None]] = np.inf
                db[q.prefecture.to_numpy()[:, None] != bg.prefecture.to_numpy()[None]] = np.inf
            bg_sorted = np.sort(db, axis=1)
            count = np.vstack([np.searchsorted(bg_sorted[i], dc[i], side="right") for i in range(len(q))])
            pair = -np.log((count + .5) / (len(bg) + 1))
            pair[~np.isfinite(dc)] = -1e4
            # 各課題の近傍12行まで。遠い行へ無理に対応させない。
            candidates = np.argsort(dc, axis=1, kind="stable")[:, :12]
            mask = np.zeros_like(pair, dtype=bool)
            np.put_along_axis(mask, candidates, True, axis=1)
            pair[~mask] = -1e4
            nn = dc.argmin(1)
            score = pair[np.arange(len(q)), nn]
            unique_ratio = np.unique(nn).size / len(q)
            for fraction in [.5, .75]:
                count_select = int(np.ceil(len(q) * fraction))
                for mode in ["independent", "one_to_one"]:
                    if mode == "independent":
                        selected, assigned = core.keep(score, fraction), nn
                    else:
                        selected, assigned = partial_assignment(pair, count_select)
                    row = np.minimum(assigned, len(c) - 1)
                    for base in ["long95", "assigned_time"]:
                        value = np.full(len(q), 9.5) if base == "long95" else c.time.to_numpy()[row]
                        name = f"matching/partial{n}_{cat}/{mode}/{base}/gate{fraction}"
                        pred[name] = np.where(selected, value, 0)
                        diagnostics.append(dict(method=name, nn_unique_ratio=unique_ratio,
                                                selected=int(selected.sum()),
                                                selected_bad_pair=int((pair[np.arange(len(q)), row][selected] <= -1e4).sum())))
    return pred, diagnostics


def run():
    start = time.time()
    td = next((ROOT / "予備戦/配布データ/結果").glob("正解*"))
    rows, diag = [], []
    sets = [("prelim", ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack", ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
            ("main", ROOT / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力", ROOT / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]
    for phase, folder, bgpath in sets:
        bg = core.read(bgpath)
        for cp in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            k = int(cp.stem.rsplit("_", 1)[1])
            q = core.read(folder / f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            preds, diagnostics = predict(q, core.read(cp), bg)
            diag += [dict(phase=phase, team=k, **d) for d in diagnostics]
            if phase == "prelim":
                truth = core.read(td / f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
                rows += [dict(team=k, method=name, **core.scores(p, truth)) for name, p in preds.items()]
            print(json.dumps(dict(phase=phase, team=k, seconds=round(time.time()-start, 1))), flush=True)
    out = HERE / "集計"
    table = pd.DataFrame(rows)
    table.to_csv(out / "対応競合_対象別.csv", index=False)
    table[~table.team.isin([1, 2, 5])].groupby("method")[["r_all", "r_rare", "r_max", "p_m", "p_c"]].mean().sort_values("r_all", ascending=False).to_csv(out / "対応競合_加工群平均.csv")
    pd.DataFrame(diag).to_csv(out / "対応競合_予測診断.csv", index=False)
    (HERE / "対応競合_実行記録.json").write_text(json.dumps(dict(seed=core.SEED, methods=len(preds), seconds=time.time()-start,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_sha256=core.HASHES,
        truth_used_for_predictions=False, submission_created=False), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(table[~table.team.isin([1, 2, 5])].groupby("method")[["r_all", "p_m", "p_c"]].mean().nlargest(12, "r_all").to_string(), flush=True)


if __name__ == "__main__":
    run()
