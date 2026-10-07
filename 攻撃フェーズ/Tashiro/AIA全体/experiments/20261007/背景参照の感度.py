"""固定した対応競合候補を、別の公開背景参照で評価する。

今回の入力は予備戦背景5143行とキット背景5143行で、別の公開データ。
背景を合算したケースも評価し、同じ課題を未知正解の検証とは扱わない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import 分析 as core
from 対応競合の頑健性 import distance, pair_score, predict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SETTING = [("discrete", "one_to_one", .75), ("continuous", "one_to_one", .75),
           ("continuous", "sex_quota", .75), ("continuous", "one_to_one", .5),
           ("discrete", "independent", .75)]


def run():
    start = time.time()
    prior = core.read(ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
    kit = core.read(ROOT / "スターターキット/初版/participant_data/A_bg.csv")
    md = ROOT / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
    main = core.read(md / "A_bg.csv")
    assert len(prior) == len(kit) == len(main) == 5143
    k1 = pd.util.hash_pandas_object(prior[core.QI], index=False).to_numpy()
    k2 = pd.util.hash_pandas_object(kit[core.QI], index=False).to_numpy()
    assert np.intersect1d(k1, k2).size == 0
    union = pd.concat([prior, kit], ignore_index=True)
    assert np.intersect1d(pd.util.hash_pandas_object(union[core.QI], index=False),
                          pd.util.hash_pandas_object(main[core.QI], index=False)).size == 0
    td = next((ROOT / "予備戦/配布データ/結果").glob("正解*"))
    pf = ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack"
    own = core.read(next((ROOT / "本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    rows, diag, self_rows = [], [], []
    input_matches = []
    for phase, folder, sources in [("prelim", pf, {"prelim_5143": prior, "kit_5143": kit, "prior_and_kit_10286": union}),
                                   ("main", md, {"main_5143": main, "prior_and_kit_10286": union,
                                                  "all_public_15429": pd.concat([union, main], ignore_index=True)})]:
        all_q = []
        for cp in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            k = int(cp.stem.rsplit("_", 1)[1])
            q = core.read(folder / f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c = core.read(cp)
            all_q.append(q)
            predictions = {}
            if phase == "prelim":
                truth = core.read(td / f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            for ref_name, bg in sources.items():
                b = core.raw(bg)
                scale = np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0), 1e-5)
                dc, db = distance(q, c, scale), distance(q, bg, scale)
                pairs = pair_score(dc, db)
                for cdf, mode, fraction in SETTING:
                    method = f"fixed/{cdf}/{mode}/gate{fraction}"
                    chosen = predict(pairs[cdf], dc, q.sex.to_numpy(), c.sex.to_numpy(), mode, fraction)
                    prediction = np.where(chosen, 9.5, 0)
                    predictions[(ref_name, method)] = chosen
                    if phase == "prelim":
                        rows.append(dict(team=k, background=ref_name, method=method, **core.scores(prediction, truth)))
                    elif k == 28:
                        joined = q.merge(own[core.QI + ["time", "is_rare"]], on=core.QI, how="left", validate="one_to_one")
                        label = joined.time.notna().to_numpy()
                        assert label.sum() == 163
                        hits = np.abs(prediction - joined.time.to_numpy(float)) <= .5
                        rare = joined.is_rare.fillna(0).to_numpy(bool)
                        self_rows.append(dict(background=ref_name, method=method,
                            selection_auc_binary=roc_auc_score(label, chosen),
                            selected_member=int((chosen & label).sum()), selected_control=int((chosen & ~label).sum()),
                            hit_member=int(hits[label].sum()), p_m=float(hits[label].mean()),
                            rare_member=int(rare.sum()), hit_rare_member=int(hits[rare].sum()),
                            control_time_known=False))
            first_name = next(iter(sources))
            for (ref_name, method), chosen in predictions.items():
                baseline = predictions[(first_name, method)]
                diag.append(dict(phase=phase, team=k, background=ref_name, method=method,
                    selected=int(chosen.sum()), same_selection_as_phase_background=int((chosen == baseline).sum()),
                    selected_overlap=int((chosen & baseline).sum())))
            print(json.dumps(dict(phase=phase, team=k, seconds=round(time.time()-start, 1))), flush=True)
        combined = pd.concat(all_q)
        for fields_name, fields in [("QI7", core.QI), ("clinical4", ["SBP", "TG", "HDL", "ALT"])]:
            qh = pd.util.hash_pandas_object(combined[fields], index=False).to_numpy()
            for n, data in {"prelim_5143": prior, "kit_5143": kit, "main_5143": main}.items():
                dh = pd.util.hash_pandas_object(data[fields], index=False).to_numpy()
                input_matches.append(dict(phase=phase, fields=fields_name, background=n,
                    exact_matched_queries=int(np.isin(qh, dh).sum()), queries=len(combined)))
    out = HERE / "集計"
    table = pd.DataFrame(rows)
    table.to_csv(out / "背景参照感度_対象別.csv", index=False)
    table[~table.team.isin([1,2,5])].groupby(["background", "method"])[["r_all", "p_m", "p_c"]].mean().to_csv(out / "背景参照感度_加工群平均.csv")
    pd.DataFrame(diag).to_csv(out / "背景参照感度_選択変化.csv", index=False)
    pd.DataFrame(self_rows).to_csv(out / "背景参照感度_本戦自チーム.csv", index=False)
    pd.DataFrame(input_matches).to_csv(out / "背景参照感度_課題との一致.csv", index=False)
    (HERE / "背景参照感度_実行記録.json").write_text(json.dumps(dict(seed=core.SEED,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_sha256=core.HASHES,
        reference_rows=dict(prelim=len(prior), kit=len(kit), main=len(main)), prior_kit_QI_overlap=0, settings=SETTING,
        seconds=time.time()-start, main_truth_used_for_training=False, submission_created=False), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(table[~table.team.isin([1,2,5])].groupby(["background", "method"])[["r_all", "p_m", "p_c"]].mean().to_string(), flush=True)
    print(pd.DataFrame(self_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
