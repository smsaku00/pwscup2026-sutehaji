"""部分対応の同点を残り属性で解き、近傍境界の同点行も含める。

同点解消の検証のため、部分2属性・性別一致・75%選択へ固定する。
実人物の対応を証明するものではない。提出用の個票は保存しない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from 対応競合の分析 import partial_assignment
from 対応競合の頑健性 import distance, pair_score
import 分析 as core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PERMUTATIONS = 10
EPSILON = 1e-10


def select(pair, dc, qs, cs, mode):
    # 12番目と同じ部分距離を持つ行を、行番号で切り落とさない。
    limit = np.partition(dc, 11, axis=1)[:, 11]
    valid = np.isfinite(dc) & (dc <= limit[:, None])
    p = np.where(valid, pair, -1e4)
    if mode == "global":
        return partial_assignment(p, int(np.ceil(len(qs)*.75)))[0]
    result = np.zeros(len(qs), bool)
    for sex in np.unique(qs):
        a, b = qs == sex, cs == sex
        result[a] = partial_assignment(p[a][:, b], int(np.ceil(a.sum()*.75)))[0]
    return result


def matrices(q, c, bg):
    b = core.raw(bg)
    scale = np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0), 1e-5)
    dc, db = distance(q, c, scale), distance(q, bg, scale)
    pair = pair_score(dc, db)["continuous"]
    full = cdist(core.raw(q)/scale, core.raw(c)/scale, "sqeuclidean")/5
    # 一致の主尺度を変えず、同点のときに5属性の距離を使う。
    pair -= EPSILON*np.minimum(full, 1000.)
    return pair, dc


def run():
    start = time.time()
    td = next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    own = core.read(next((ROOT/"本戦/配布データ").glob("PWSCup2026_final_team28_*/B_self.csv")))
    rows, perms, selfs, main_changes = [], [], [], []
    sets = [("prelim", ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack", ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
            ("main", ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力", ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv")]
    for phase, folder, bgpath in sets:
        bg = core.read(bgpath)
        for cp in sorted(folder.glob("C_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            k = int(cp.stem.rsplit("_", 1)[1])
            q = core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c = core.read(cp)
            pair, dc = matrices(q, c, bg)
            qs, cs = q.sex.to_numpy(), c.sex.to_numpy()
            for mode in ["global", "sex_quota"]:
                chosen = select(pair, dc, qs, cs, mode)
                method = "stable_matching/"+mode+"/long75"
                prediction = np.where(chosen, 9.5, 0)
                if phase == "prelim":
                    truth = core.read(td/f"aia_truth_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
                    rows.append(dict(team=k, method=method, **core.scores(prediction, truth)))
                elif k == 28:
                    j = q.merge(own[core.QI+["time", "is_rare"]], on=core.QI, how="left", validate="one_to_one")
                    m = j.time.notna().to_numpy()
                    hit = np.abs(prediction-j.time.to_numpy(float)) <= .5
                    r = j.is_rare.fillna(0).to_numpy(bool)
                    assert m.sum()==163 and r.sum()==13
                    selfs.append(dict(method=method, selected_member=int((chosen&m).sum()), selected_control=int((chosen&~m).sum()),
                        hit_member=int(hit[m].sum()), p_m=float(hit[m].mean()), hit_rare_member=int(hit[r].sum()), control_time_known=False))
                for repeat in range(PERMUTATIONS):
                    rng = np.random.default_rng(core.SEED + k*100 + repeat)
                    a, b = rng.permutation(len(q)), rng.permutation(len(c))
                    perm = select(pair[a][:,b], dc[a][:,b], qs[a], cs[b], mode)[np.argsort(a)]
                    changed = int((chosen != perm).sum())
                    rec = dict(phase=phase, team=k, method=method, repeat=repeat, selected_changed=changed)
                    if phase=="prelim": rec.update(core.scores(np.where(perm,9.5,0),truth))
                    perms.append(rec)
                if phase=="main": main_changes.append(dict(team=k, method=method, selected=int(chosen.sum()), long=int(chosen.sum()), zero=int((~chosen).sum())))
            print(json.dumps(dict(phase=phase, team=k, seconds=round(time.time()-start,1))), flush=True)
    out = HERE/"集計"
    table = pd.DataFrame(rows)
    table.to_csv(out/"同点修正_対象別.csv", index=False)
    table[~table.team.isin([1,2,5])].groupby("method")[["r_all", "r_rare", "r_max", "p_m", "p_c"]].mean().to_csv(out/"同点修正_加工群平均.csv")
    pd.DataFrame(perms).to_csv(out/"同点修正_行順変更.csv", index=False)
    pd.DataFrame(perms).groupby(["phase", "method"]).selected_changed.agg(["mean", "max"]).to_csv(out/"同点修正_行順変更平均.csv")
    pd.DataFrame(selfs).to_csv(out/"同点修正_本戦自チーム.csv", index=False)
    pd.DataFrame(main_changes).to_csv(out/"同点修正_本戦動作.csv", index=False)
    (HERE/"同点修正_実行記録.json").write_text(json.dumps(dict(seed=core.SEED, permutations=PERMUTATIONS, epsilon=EPSILON,
        near_limit=12, boundary_ties_included=True, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_sha256=core.HASHES,
        seconds=time.time()-start, main_truth_used_for_predictions=False, submission_created=False),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(table[~table.team.isin([1,2,5])].groupby("method")[["r_all", "p_m", "p_c"]].mean().to_string(), flush=True)
    print(pd.DataFrame(perms).groupby(["phase","method"]).selected_changed.agg(["mean","max"]).to_string(), flush=True)


if __name__=="__main__":
    run()
