"""加工Cを使わず、課題QI・表記・行番号だけで標的を区別できるか。

公開正解を用いたチーム外評価。これはC由来の漏洩とは別の対照検証。
本戦の自チーム正解は最後の診断だけに使い、予測器の学習には渡さない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import 分析 as core
from 分布識別 import group_keep

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def features(q, bg, string_q):
    x, b = core.raw(q), core.raw(bg)
    center, sd = np.median(b, 0), np.maximum(b.std(0), 1e-5)
    scaled = np.clip((x - center) / sd, -10, 10)
    sex_specific = np.zeros_like(x)
    challenge_rank = np.zeros_like(x)
    bg_rank = np.zeros_like(x)
    for sex in q.sex.unique():
        m, bm = q.sex.eq(sex).to_numpy(), bg.sex.eq(sex).to_numpy()
        rb = b[bm]
        sex_specific[m] = (x[m] - np.median(rb, 0)) / np.maximum(rb.std(0), 1e-5)
        for j in range(5):
            challenge_rank[m, j] = rankdata(x[m, j], method="average") / (m.sum() + 1)
            bg_rank[m, j] = np.searchsorted(np.sort(rb[:, j]), x[m, j], side="right") / len(rb)
    core_x = np.column_stack([scaled, sex_specific, challenge_rank, bg_rank, q.sex.to_numpy(),
                             q.age.to_numpy() % 5, q.age.to_numpy() % 10])
    for pref in ["東京都", "大阪府", "神奈川県", "愛知県", "福岡県", "北海道", "宮崎県"]:
        core_x = np.column_stack([core_x, q.prefecture.eq(pref).to_numpy()])
    formatting = np.column_stack([string_q[c].str.split(".", regex=False).str[-1].str.len().to_numpy(float) for c in core.NUM])
    row = q.challenge_row_id.to_numpy(float)
    position = np.column_stack([row / len(q)] + [(row % n) / n for n in [2, 3, 5, 7, 11]])
    return {"QI_only": core_x, "QI_format_position": np.column_stack([core_x, formatting, position])}


def fit_model(name, train, label):
    if name == "ET":
        m = ExtraTreesClassifier(n_estimators=160, min_samples_leaf=32, max_depth=14, max_features=.8,
                                random_state=core.SEED, n_jobs=1)
    elif name == "HGB":
        m = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=8, min_samples_leaf=80,
                                          l2_regularization=3, learning_rate=.05, early_stopping=False, random_state=core.SEED)
    else:
        m = LogisticRegression(C=.1, max_iter=500, random_state=core.SEED)
    m.fit(train, label)
    return m


def run():
    start = time.time()
    folder = ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack"
    bg = core.read(ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
    td = next((ROOT / "予備戦/配布データ/結果").glob("正解*"))
    items = []
    for cp in sorted(folder.glob("aia_challenge_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        k = int(cp.stem.rsplit("_", 1)[1])
        q = core.read(cp).sort_values("challenge_row_id").reset_index(drop=True)
        sq = pd.read_csv(cp, dtype=str).sort_values("challenge_row_id", key=lambda s: s.astype(int)).reset_index(drop=True)
        truth = core.read(td / f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        items.append(dict(team=k, q=q, truth=truth, x=features(q, bg, sq)))
    groups = np.concatenate([np.full(len(v["q"]), v["team"]) for v in items])
    label = np.concatenate([v["truth"].label.eq("member").to_numpy() for v in items])
    # 多項式はQI本体だけ。行位置の高次展開による探索は行わない。
    arrays = {name: np.vstack([v["x"][name] for v in items]) for name in items[0]["x"]}
    poly = PolynomialFeatures(2, include_bias=False)
    arrays["QI_poly"] = poly.fit_transform(np.clip(arrays["QI_only"], -6, 6))
    rows, aucs, main_diag = [], [], []
    mfolder = ROOT / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
    mbg = core.read(mfolder / "A_bg.csv")
    mqpath = mfolder / "aia_challenge_28.csv"
    mq = core.read(mqpath).sort_values("challenge_row_id").reset_index(drop=True)
    msq = pd.read_csv(mqpath, dtype=str).sort_values("challenge_row_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    mx = features(mq, mbg, msq)
    mx["QI_poly"] = poly.transform(np.clip(mx["QI_only"], -6, 6))
    self_b = core.read(next((ROOT / "本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    self_label = mq.merge(self_b[core.QI + ["time"]], on=core.QI, how="left", validate="one_to_one").time.notna().to_numpy()
    assert self_label.sum() == 163
    for setting, x in arrays.items():
        models = ["logistic"] if setting == "QI_poly" else ["ET", "HGB", "logistic"]
        for model_name in models:
            p = np.zeros(len(label))
            for train, test in GroupKFold(7).split(x, groups=groups):
                model = fit_model(model_name, x[train], label[train])
                p[test] = model.predict_proba(x[test])[:, 1]
            name = setting + "/" + model_name
            for item in items:
                m = groups == item["team"]
                aucs.append(dict(team=item["team"], method=name, selection_auc=roc_auc_score(label[m], p[m])))
                for fraction in [.5, .75, .9]:
                    for grouping in ["all", "sex"]:
                        chosen = core.keep(p[m], fraction) if grouping == "all" else group_keep(p[m], item["q"], fraction)
                        pred = np.where(chosen, 9.5, 0)
                        rows.append(dict(team=item["team"], method=f"noC/{name}/{grouping}/gate{fraction}", **core.scores(pred, item["truth"])))
            model = fit_model(model_name, x, label)
            mp = model.predict_proba(mx[setting])[:, 1]
            main_diag.append(dict(method=name, selection_auc=roc_auc_score(self_label, mp), control_time_known=False))
            print(json.dumps(dict(method=name, seconds=round(time.time()-start, 1))), flush=True)
    out = HERE / "集計"
    table = pd.DataFrame(rows)
    table.to_csv(out / "課題単独_対象別.csv", index=False)
    table[~table.team.isin([1, 2, 5])].groupby("method")[["r_all", "r_rare", "r_max", "p_m", "p_c"]].mean().sort_values("r_all", ascending=False).to_csv(out / "課題単独_加工群平均.csv")
    pd.DataFrame(aucs).to_csv(out / "課題単独_標的選別.csv", index=False)
    pd.DataFrame(main_diag).to_csv(out / "課題単独_本戦自チーム.csv", index=False)
    (HERE / "課題単独_実行記録.json").write_text(json.dumps(dict(seed=core.SEED, fold_kind="GroupKFold7", code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_sha256=core.HASHES, seconds=time.time()-start,
        C_used=False, main_truth_used_for_training=False, submission_created=False), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(table[~table.team.isin([1, 2, 5])].groupby("method")[["r_all", "p_m", "p_c"]].mean().nlargest(12, "r_all").to_string(), flush=True)
    print(pd.DataFrame(main_diag).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
