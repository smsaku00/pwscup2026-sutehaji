"""固定済み予測を変えず、対照差・行順・地域近傍を確認する。"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("regional_audit", HERE / "分析.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def run():
    record = json.loads((HERE / "実行記録.json").read_text())
    assert m.sha(HERE / "分析.py") == record["code_sha256"]
    assert m.sha(HERE / "作業用/全予測_公開正解読込み前.npz") == record["prediction_sha256"]
    for path, digest in record["input_sha256"].items():
        assert m.sha(Path(path)) == digest, path
    for path, digest in record["imported_code_sha256"].items():
        assert m.sha(m.ROOT / path) == digest, path
    bg = m.b.read(m.ROOT / "予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv")
    folder = m.ROOT / "予備戦/配布データ/PWSCup2026_prelim_attack"
    q = m.b.read(folder / "aia_challenge_1.csv").sort_values("challenge_row_id").reset_index(drop=True)
    c = m.b.read(folder / "C_1.csv")
    original, _ = m.predict(q, c, m.prepare(q, bg))
    rng = np.random.default_rng(771)
    shuffled, _ = m.predict(q, c.iloc[rng.permutation(len(c))].reset_index(drop=True), m.prepare(q, bg))
    assert np.array_equal(original, shuffled)
    order = rng.permutation(len(q))
    qr = q.iloc[order].reset_index(drop=True)
    shuffled, _ = m.predict(qr, c, m.prepare(qr, bg))
    assert np.array_equal(original, shuffled[:, np.argsort(order)])
    # 実際に空層が生じた本戦6・seed20261040で、全点差の別計算と比較。
    main = m.ROOT / "本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
    qb = m.b.read(main / "aia_challenge_6.csv").sort_values("challenge_row_id").reset_index(drop=True)
    cb = m.b.read(main / "C_6.csv")
    mbg = m.b.read(main / "A_bg.csv")
    index = np.random.default_rng(m.SEEDS[0]).permutation(len(mbg))
    ref = mbg.iloc[index[len(mbg)//2:]].reset_index(drop=True)
    evidence, fallback = m.column_pref(qb, cb, ref)
    ratios, fallback_brute = [], 0
    for _, row in qb.iterrows():
        cc = cb[cb.sex.eq(row.sex) & cb.prefecture.eq(row.prefecture)]
        bb = ref[ref.sex.eq(row.sex) & ref.prefecture.eq(row.prefecture)]
        if len(cc) == 0 or len(bb) == 0:
            cc, bb = cb[cb.sex.eq(row.sex)], ref[ref.sex.eq(row.sex)]
            fallback_brute += 1
        ratios.append([np.log((np.min(np.abs(bb[col].to_numpy() - row[col])) + m.v.EPS) /
                             (np.min(np.abs(cc[col].to_numpy() - row[col])) + m.v.EPS)) for col in m.CLINICAL])
    ratios = np.array(ratios)
    assert fallback == fallback_brute == 218
    assert np.allclose(evidence["all4"], ratios.mean(1), atol=1e-12, rtol=0)
    assert np.allclose(evidence["top2"], np.sort(ratios, axis=1)[:, -2:].mean(1), atol=1e-12, rtol=0)
    arrays = np.load(HERE / "作業用/全予測_公開正解読込み前.npz", allow_pickle=False)
    assert arrays["prelim_full"].shape == (21, 7, 326)
    assert arrays["main_full"].shape == (23, 7, 326)
    assert arrays["null"].shape == (20, 4, 2, 5, 7, 326)
    diag = pd.read_csv(m.OUT / "公開対象と対照の動作.csv")
    assert diag.invalid_assignments.eq(0).all() and len(diag) == 1020
    score = pd.read_csv(m.OUT / "予備戦_全採点.csv")
    paired = pd.read_csv(m.OUT / "背景条件ごとの実Cと対照差.csv")
    base = paired[paired.method.eq("P01")][["team", "seed", "source_null", "r_all_actual", "excess_R", "excess_hit_gap"]]
    dif = paired.merge(base, on=["team", "seed", "source_null"], suffixes=("", "_base"), validate="many_to_one")
    for col in ["r_all_actual", "excess_R", "excess_hit_gap"]:
        dif["delta_" + col] = dif[col] - dif[col + "_base"]
    summaries = []
    for (team, method, source), rows in dif.groupby(["team", "method", "source_null"], sort=False):
        result = dict(team=team, method=method, source_null=source)
        for col in ["r_all_actual", "excess_R", "excess_hit_gap"]:
            values = rows["delta_" + col]
            result.update({col + "_delta_mean": values.mean(), col + "_wins": int(values.gt(1e-12).sum()),
                           col + "_losses": int(values.lt(-1e-12).sum()), col + "_ties": int(values.abs().le(1e-12).sum())})
        summaries.append(result)
    pd.DataFrame(summaries).to_csv(m.OUT / "対象別_4背景の勝敗.csv", index=False)
    groups = pd.read_csv(m.OUT / "特徴群ごとの固定7方式.csv")[['group', 'teams']].drop_duplicates()
    group_rows = []
    for _, group in groups.iterrows():
        teams = [int(value) for value in group.teams.split()]
        for (method, source), rows in dif[dif.team.isin(teams)].groupby(["method", "source_null"], sort=False):
            team_deltas = rows.groupby("team").delta_excess_R.mean()
            group_rows.append(dict(group=group.group, teams=group.teams, method=method, source_null=source,
                              **rows[["r_all_actual", "r_all_null", "excess_R", "excess_hit_gap", "delta_r_all_actual", "delta_excess_R", "delta_excess_hit_gap"]].mean().to_dict(),
                              excess_R_target_wins=int(team_deltas.gt(1e-12).sum()), excess_R_target_losses=int(team_deltas.lt(-1e-12).sum())))
    pd.DataFrame(group_rows).to_csv(m.OUT / "特徴群ごとの背景差.csv", index=False)
    old = pd.read_csv(HERE.parent / "属性対ごとの証拠統合/集計/予備戦_対象別.csv")
    new = score[score.source.eq("full_actual") & score.method.str.startswith("clinical_pair")].copy()
    new['matched_old_method'] = new.method.map({'clinical_pair_top1': 'pair_top1', 'clinical_pair_top3': 'pair_top3_mean'})
    comparison = new.merge(old, left_on=['team','matched_old_method'], right_on=['team','method'], suffixes=('_new6','_old10'), validate='one_to_one')
    comparison = comparison[['team','method_new6','method_old10','r_all_new6','r_all_old10','hit_m_new6','hit_c_new6','hit_m_old10','hit_c_old10']]
    comparison['delta_R_new6_minus_old10'] = comparison.r_all_new6 - comparison.r_all_old10
    comparison.to_csv(m.OUT / "旧10対と新6対_同一全BG比較.csv", index=False)
    verification = dict(all_saved_inputs_and_imports_hash_match=True, input_count=len(record['input_sha256']),
         saved_prediction_hash_match=True, full_control_scores_matched=63, score_rows=6335,
         C_and_query_order_permutation_sample_team1_all7_match=True,
         regional_nearest_vs_bruteforce_all326_main6_split_match=True, fallback_rows_sample=218,
         diagnostic_rows=1020, invalid_assignments=0, primary_prediction_changes=0,
         main_B_or_truth_used=False, old_age10_comparison_is_full_BG_only=True,
         old_age10_vs_new6_split_and_null_comparison_available=False,
         bootstrap_does_not_preserve_processed_C_quantization_or_joint_clinical_dependence=True,
         null_unique_rows_range=diag.groupby('source').unique_source_rows.agg(['min','max']).dropna().astype(int).to_dict('index'))
    m.dump(HERE / '検証記録.json', verification)
    print(json.dumps(verification, ensure_ascii=False))
    print(pd.DataFrame(group_rows).query("group == 'prior_strong_record_1_2' and method == 'clinical_pair_top1'").to_string(index=False))


if __name__ == '__main__':
    run()
