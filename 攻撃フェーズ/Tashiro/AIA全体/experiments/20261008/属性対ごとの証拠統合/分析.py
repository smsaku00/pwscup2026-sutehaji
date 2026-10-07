"""固定属性ペアの背景距離を先に尺度調整し、同一C行内で統合する。

3統合則と基準だけに固定する。統合量は校正済みpではない。
全44Cと背景対照の予測を生成してから公開正解を読む。
提出個票・ZIP・本戦正解・CodaBenchは使用しない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, importlib.util, io, json, platform, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = HERE.parents[1]/"20261007_AIA全体向上の分析"
OUT = HERE/"集計"
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location("frozen_stable_matching", OLD/"同点に依存しない対応.py")
stable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable)
core = stable.core
PAIRS = list(combinations(range(5), 2))
METHODS = ["baseline_partial2", "pair_top1", "pair_top3_mean", "pair_all10_mean"]
SEED = 20261008
REPEATS = 20
HASHES = {}
PROTOCOL = dict(seed=SEED, pairs=[[core.NUM[a], core.NUM[b]] for a,b in PAIRS], methods=METHODS,
                numeric_transform="TG/ALT log1p; reference BG IQR scaling",
                distance="mean squared distance on each fixed pair",
                pair_score="same-query continuous BG distance CDF count; -log((interpolated count+.25)/(BG n+1))",
                integration="same C row: maximum, upper-three mean, all-ten mean",
                pair_scores_independent=False, integrated_score_is_calibrated_p=False,
                sex_hard=True, prefecture_used=False, selected_fraction_per_sex=.75,
                baseline_near_limit="adaptive partial2 distance smallest12 incl boundary ties",
                new_near_limit="integrated score largest12 incl boundary ties; before tiny tie break",
                tie_break="subtract 1e-10 times min(full5 squared distance/5,1000)",
                one_to_one=True, selected_answer=9.5, other_answer=0,
                primary_predictions_before_truth_loading=True,
                reference_controls="disjoint BG sample pool2571/ref2572; 20 C samples1049 each; same reference for actual and null C",
                main_truth_used=False, official_phi_reproduction=False,
                number_of_new_rules=3, submission_created=False)


def read(path):
    raw = Path(path).read_bytes()
    HASHES[str(path)] = hashlib.sha256(raw).hexdigest()
    return pd.read_csv(io.BytesIO(raw), float_precision="round_trip")


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def background(q, ref):
    raw_ref = core.raw(ref)
    scale = np.maximum(np.quantile(raw_ref,.75,axis=0)-np.quantile(raw_ref,.25,axis=0), 1e-5)
    x, y = core.raw(q)/scale, raw_ref/scale
    sq = (x[:, None]-y[None])**2
    mismatch = q.sex.to_numpy()[:, None] != ref.sex.to_numpy()[None]
    adaptive = np.sort(sq, axis=2)[:, :, :2].mean(2)
    adaptive[mismatch] = np.inf
    sorted_pairs = []
    for a,b in PAIRS:
        d = (sq[:, :, a]+sq[:, :, b])/2
        d[mismatch] = np.inf
        sorted_pairs.append(np.sort(d, axis=1))
    return dict(scale=scale, x=x, adaptive=adaptive, pair_refs=sorted_pairs, ref_n=len(ref))


def continuous_score(d, sorted_ref, denominator):
    score = np.empty_like(d)
    for i, (values, ref) in enumerate(zip(d, sorted_ref)):
        finite = ref[np.isfinite(ref)]
        assert len(finite)>0
        count = np.interp(values, np.r_[0., finite], np.r_[0., np.arange(1,len(finite)+1)], right=len(finite))
        score[i] = -np.log((count+.25)/(denominator+1))
    score[~np.isfinite(d)] = -1e4
    return score


def select(pair, valid, qsex, csex):
    result = np.zeros(len(qsex), bool)
    bad = 0
    for sex in np.unique(qsex):
        a, b = qsex==sex, csex==sex
        nselect = int(np.ceil(a.sum()*.75))
        limited = np.where(valid[a][:, b], pair[a][:, b], -1e4)
        chosen, assigned = stable.partial_assignment(limited, nselect)
        result[a] = chosen
        bad += int((limited[np.flatnonzero(chosen), assigned[chosen]]<=-9999).sum())
    return result, bad


def predict(q, c, bginfo):
    """公開入力だけ。属性ペアは同じC行から取り、正解/番号を使わない。"""
    y = core.raw(c)/bginfo["scale"]
    sq = (bginfo["x"][:, None]-y[None])**2
    mismatch = q.sex.to_numpy()[:, None] != c.sex.to_numpy()[None]
    adaptive = np.sort(sq,axis=2)[:, :, :2].mean(2)
    adaptive[mismatch] = np.inf
    baseline_score = stable.pair_score(adaptive, bginfo["adaptive"])["continuous"]
    pair_scores = []
    for (a,b), ref in zip(PAIRS, bginfo["pair_refs"]):
        d = (sq[:, :, a]+sq[:, :, b])/2
        d[mismatch] = np.inf
        pair_scores.append(continuous_score(d, ref, bginfo["ref_n"]))
    scores = np.stack(pair_scores)
    sorted_scores = np.sort(scores, axis=0)
    integrated = {"baseline_partial2":baseline_score, "pair_top1":sorted_scores[-1],
                  "pair_top3_mean":sorted_scores[-3:].mean(0), "pair_all10_mean":scores.mean(0)}
    full = cdist(bginfo["x"], y, "sqeuclidean")/5
    prediction, diagnostic = {}, []
    chosen_base = None
    for name, score in integrated.items():
        if name=="baseline_partial2":
            limit = np.partition(adaptive, 11, axis=1)[:, 11]
            valid = np.isfinite(adaptive)&(adaptive<=limit[:, None])
        else:
            eligible_score = np.where(mismatch, -np.inf, score)
            limit = np.partition(eligible_score, -12, axis=1)[:, -12]
            valid = ~mismatch & (eligible_score>=limit[:, None])
        adjusted = score-stable.EPSILON*np.minimum(full,1000.)
        chosen, bad = select(adjusted, valid, q.sex.to_numpy(), c.sex.to_numpy())
        if name=="baseline_partial2":
            chosen_base = chosen
            assert np.array_equal(chosen, stable.select(adjusted, adaptive, q.sex.to_numpy(), c.sex.to_numpy(), "sex_quota"))
        prediction[name] = np.where(chosen,9.5,0.)
        diagnostic.append(dict(method=name, selected=int(chosen.sum()), changed_from_baseline=int((chosen!=chosen_base).sum()),
                               selected_bad_pairs=bad, candidate_count_median=float(np.median(valid.sum(1))),
                               candidate_count_max=int(valid.sum(1).max())))
    return prediction, diagnostic


def run():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dump(HERE/"事前固定仕様.json", PROTOCOL)
    truth_folder = next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    prior_teams = sorted(int(p.stem.rsplit("_",1)[1]) for p in truth_folder.glob("aia_truth_*.csv"))
    assert len(prior_teams)==21
    pending, counterfactual, diag, permutations = [], [], [], []
    phases = [("prelim", ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack", ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv", prior_teams),
              ("main", ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力", ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv", None)]
    for phase, folder, bgpath, teams in phases:
        bg = read(bgpath)
        if teams is None:
            teams = sorted(int(p.stem.rsplit("_",1)[1]) for p in folder.glob("C_*.csv"))
        index = np.random.default_rng(SEED).permutation(len(bg))
        pool, ref = (bg.iloc[i].reset_index(drop=True) for i in [index[:len(bg)//2], index[len(bg)//2:]])
        assert set(index[:len(bg)//2]).isdisjoint(index[len(bg)//2:])
        samples = [pool.iloc[np.sort(np.random.default_rng(SEED+400+r).choice(len(pool),1049,False))].reset_index(drop=True) for r in range(REPEATS)]
        for team in teams:
            q = read(folder/f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c = read(folder/f"C_{team}.csv")
            bginfo = background(q,bg)
            pred, one_diag = predict(q,c,bginfo)
            diag += [dict(phase=phase,team=team,**r) for r in one_diag]
            pending.append(dict(phase=phase, team=team, ids=q.challenge_row_id.tolist(), pred=pred))
            # 別行順の公開入力で全統合則を計算し直す。
            for repeat in range(3):
                rng = np.random.default_rng(SEED+team*100+repeat)
                a,b = rng.permutation(len(q)), rng.permutation(len(c))
                shuffled, _ = predict(q.iloc[a],c.iloc[b],background(q.iloc[a],bg))
                for name,value in shuffled.items():
                    permutations.append(dict(phase=phase,team=team,repeat=repeat,method=name,
                                              selected_changed=int((value[np.argsort(a)]!=pred[name]).sum())))
            if phase=="prelim" and team not in [1,2,5]:
                splitinfo = background(q,ref)
                real, real_diag = predict(q,c,splitinfo)
                counterfactual.append(dict(team=team, source="actual_C", repeat=-1, pred=real))
                for repeat,sample in enumerate(samples):
                    value, null_diag = predict(q,sample,splitinfo)
                    counterfactual.append(dict(team=team,source="BG_C",repeat=repeat,pred=value))
            print(json.dumps(dict(stage="predictions_before_truth",phase=phase,team=team,seconds=round(time.time()-start,1))),flush=True)
    assert len(pending)==44
    # 全44Cおよび対照予測を作ってから初めてtruthファイルを読む。
    truths = {team:read(truth_folder/f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True) for team in prior_teams}
    rows, controls = [], []
    for rec in pending:
        if rec["phase"]=="prelim":
            truth = truths[rec["team"]]
            assert rec["ids"]==truth.challenge_row_id.tolist()
            rows += [dict(team=rec["team"],method=name,**core.scores(value,truth)) for name,value in rec["pred"].items()]
    for rec in counterfactual:
        controls += [dict(team=rec["team"],source=rec["source"],repeat=rec["repeat"],method=name,**core.scores(value,truths[rec["team"]])) for name,value in rec["pred"].items()]
    table, control = pd.DataFrame(rows), pd.DataFrame(controls)
    table.to_csv(OUT/"予備戦_対象別.csv",index=False)
    pd.DataFrame(diag).to_csv(OUT/"44Cの予測動作.csv",index=False)
    pd.DataFrame(permutations).to_csv(OUT/"行順変更.csv",index=False)
    control.to_csv(OUT/"独立背景_実Cと全反復.csv",index=False)
    summary, selections = [], []
    for scope, excluded in [("all20",[5]),("processed18",[1,2,5])]:
        one = table[~table.team.isin(excluded)]
        wide = one.pivot(index="team",columns="method",values="r_all")[METHODS]
        for method, rec in one.groupby("method",sort=False):
            delta = wide[method]-wide[METHODS[0]]
            summary.append(dict(scope=scope,method=method,n=len(rec),**rec[["r_all","r_rare","r_max","p_m","p_c"]].mean().to_dict(),
                                signed_hit_difference=float((rec.hit_m-rec.hit_c).mean()),improved=int(delta.gt(1e-12).sum()),worsened=int(delta.lt(-1e-12).sum())))
        for team in wide.index:
            means = wide.drop(index=team).mean()
            chosen = METHODS[int(np.argmax(means.to_numpy()))]
            rec = one[(one.team==team)&(one.method==chosen)].iloc[0]
            selections.append(dict(scope=scope,team=int(team),chosen_method=chosen,r_all=float(rec.r_all),baseline_r_all=float(wide.loc[team,METHODS[0]]),
                                   delta=float(rec.r_all-wide.loc[team,METHODS[0]]),training_means=json.dumps(means.to_dict(),sort_keys=True),
                                   signed_hit_difference=int(rec.hit_m-rec.hit_c)))
    pd.DataFrame(summary).to_csv(OUT/"20対象と18対象平均.csv",index=False)
    selection = pd.DataFrame(selections)
    selection.to_csv(OUT/"対象留保の4方式選択.csv",index=False)
    selection.groupby("scope")[["r_all","baseline_r_all","delta","signed_hit_difference"]].mean().to_csv(OUT/"対象留保の平均.csv")
    real = control[control.source.eq("actual_C")].drop(columns=["source","repeat"])
    null = control[control.source.eq("BG_C")].groupby(["team","method"])[["r_all","p_m","p_c","hit_m","hit_c"]].mean().reset_index()
    comparison = real.merge(null,on=["team","method"],suffixes=("_real","_null"))
    comparison["r_excess"] = comparison.r_all_real-comparison.r_all_null
    comparison.to_csv(OUT/"独立背景_対象別実Cと背景差.csv",index=False)
    comparison.groupby("method")[["r_all_real","r_all_null","r_excess","p_m_real","p_c_real","p_m_null","p_c_null"]].mean().to_csv(OUT/"独立背景_平均.csv")
    saved = read(OLD/"集計/同点修正_対象別.csv")
    saved = saved[saved.method.eq("stable_matching/sex_quota/long75")].set_index("team")
    base = table[table.method.eq(METHODS[0])].set_index("team")
    assert np.array_equal(saved.loc[base.index,"hit_m"],base.hit_m)
    assert np.array_equal(saved.loc[base.index,"hit_c"],base.hit_c)
    dump(HERE/"実行記録.json",dict(PROTOCOL,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         protocol_sha256=hashlib.sha256((HERE/"事前固定仕様.json").read_bytes()).hexdigest(),input_sha256=HASHES,
         imported_code_sha256={str(OLD/name):hashlib.sha256((OLD/name).read_bytes()).hexdigest() for name in ["同点に依存しない対応.py","対応競合の分析.py","対応競合の頑健性.py","分析.py"]},
         python=sys.executable,python_version=platform.python_version(),seconds=time.time()-start,
         baseline_hit_counts_match_saved=True,permutation_cases=len(permutations),
         max_selected_changed=max(rec["selected_changed"] for rec in permutations),
         max_selected_bad_pairs=max(rec["selected_bad_pairs"] for rec in diag),
         prediction_artifacts_written=False, main_scores_used=False))
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(selection.groupby("scope")[["r_all","baseline_r_all","delta"]].mean().to_string(),flush=True)
    print(comparison.groupby("method")[["r_all_real","r_all_null","r_excess"]].mean().to_string(),flush=True)


if __name__=="__main__":
    run()
