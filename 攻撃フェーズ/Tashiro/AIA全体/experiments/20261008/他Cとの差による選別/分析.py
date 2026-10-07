"""固定少数候補で、対象Cへの近さを同フェーズ他Cと比較する。

昨日のstable部分2属性・性別75%・一対一対応・9.5/0を基準にする。
公開Cのみから参照を決め、予測器へtruthを渡さない。提出物は作らない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/"AGENTS.md").exists())
OLD = HERE.parent.parent/"20261007_AIA全体向上の分析"
sys.path.insert(0, str(OLD))
import 分析 as core
from 同点に依存しない対応 import select
from 対応競合の頑健性 import distance, pair_score
from scipy.spatial.distance import cdist

SEED = 20261008
REPEATS = 20
PERMUTATIONS = 3
CONFIG = [("baseline", None, 0.), ("other_median_lambda025", .5, .25),
          ("other_median_lambda100", .5, 1.), ("other_q75_lambda025", .75, .25),
          ("other_q75_lambda100", .75, 1.)]
PLAN = dict(configurations=CONFIG, main_metric="r_all", part_dimension=2, log_TG_ALT=True,
    same_phase_C_only=True, reference_exclusion="same all-column row-set OR >=98% exact QI in phase A_bg",
    background_reuse_threshold=.98, C_source_labels_used=False, truth_used_for_predictions=False,
    same_phase_background_cdf=True, continuous_cdf=True, near=12, boundary_ties_included=True,
    tie_epsilon=1e-10, sex_quota=.75, time_selected=9.5, time_other=0.,
    null_pool_reference_disjoint=True, null_repeats=REPEATS, permutation_repeats=PERMUTATIONS,
    official_phi=False, submission_created=False, codabench_submitted=False)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def fingerprint(c):
    """全列の行集合を比較し、行順だけが違う同一Cも除外する。"""
    values = np.sort(pd.util.hash_pandas_object(c, index=False).to_numpy())
    return hashlib.sha256(values.tobytes()).hexdigest()


def scale_for(bg):
    b = core.raw(bg)
    return np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0), 1e-5)


def calibrated_nearest(dc, db):
    sorted_bg = np.sort(db, axis=1)
    dmin = dc.min(1)
    out = np.empty(len(dc))
    for i, (one, row) in enumerate(zip(dmin, sorted_bg)):
        finite = row[np.isfinite(row)]
        count = np.interp(one, np.r_[0.,finite], np.r_[0.,np.arange(1,len(finite)+1)], right=len(finite))
        out[i] = -np.log((count+.25)/(db.shape[1]+1)) if np.isfinite(one) else -1e4
    return out


def matrices(q, c, bg, scale, db=None):
    dc = distance(q,c,scale)
    if db is None:
        db = distance(q,bg,scale)
    pair = pair_score(dc,db)["continuous"]
    full = cdist(core.raw(q)/scale,core.raw(c)/scale,"sqeuclidean")/5
    pair -= 1e-10*np.minimum(full,1000.)
    return pair, dc


def corrections(q, collections, references, db, scale, permute=False):
    # 参照Cの各行も同一行の2属性距離で比較する。属性ごと別行の照合はしない。
    nearest = []
    for other in references:
        c = collections[other]
        if permute:
            rng = np.random.default_rng(SEED+other)
            c = c.iloc[rng.permutation(len(c))].reset_index(drop=True)
        dc = distance(q,c,scale)
        nearest.append(calibrated_nearest(dc,db))
    assert nearest, "比較できる他Cが必要"
    scores = np.column_stack(nearest)
    return {"median":np.quantile(scores,.5,axis=1),"q75":np.quantile(scores,.75,axis=1)}, scores


def predictions(pair, dc, q, c, correction):
    result = {}
    qs, cs = q.sex.to_numpy(),c.sex.to_numpy()
    for name, quantile, lam in CONFIG:
        penalty = 0. if quantile is None else lam*correction["median" if quantile==.5 else "q75"][:,None]
        chosen = select(pair-penalty,dc,qs,cs,"sex_quota")
        result[name] = np.where(chosen,9.5,0)
    return result


def leave_out(table, targets, group):
    records = []
    order = [v[0] for v in CONFIG]
    for held_out in targets:
        train = table[table.team.isin([v for v in targets if v != held_out])]
        mean = train.groupby("method").r_all.mean().reindex(order)
        selected = mean.idxmax()
        result = table[table.team.eq(held_out)&table.method.eq(selected)].iloc[0]
        records.append(dict(group=group,team=held_out,method=selected,training_R=float(mean.loc[selected]),
            **{k:float(result[k]) for k in ["r_all","r_rare","r_max","p_m","p_c"]}))
    return records


def run():
    start = time.time()
    HERE.mkdir(parents=True,exist_ok=True)
    out = HERE/"集計";out.mkdir(exist_ok=True)
    planpath = HERE/"事前固定候補.json"
    serial_plan = json.loads(json.dumps(PLAN))
    if planpath.exists():
        assert json.loads(planpath.read_text()) == serial_plan
    else:
        dump(planpath,serial_plan)
    prelim = ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack"
    main = ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力"
    items, exclusions, diags, perms, null_items = [], [], [], [], []
    for phase, folder, bgpath in [("prelim",prelim,ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv"),
                                  ("main",main,main/"A_bg.csv")]:
        bg = core.read(bgpath)
        scale = scale_for(bg)
        files = sorted(folder.glob("C_*.csv"),key=lambda p:int(p.stem.rsplit("_",1)[1]))
        collections = {int(cp.stem.rsplit("_",1)[1]):core.read(cp) for cp in files}
        hashes = {team:fingerprint(c) for team,c in collections.items()}
        bg_index = pd.MultiIndex.from_frame(bg[core.QI])
        reuse = {team:float(pd.MultiIndex.from_frame(c[core.QI]).isin(bg_index).mean()) for team,c in collections.items()}
        rng = np.random.default_rng(SEED+41)
        order = rng.permutation(len(bg))
        pool = bg.iloc[order[:len(bg)//2]].reset_index(drop=True)
        ref = bg.iloc[order[len(bg)//2:]].reset_index(drop=True)
        split_scale = scale_for(ref)
        draws = [np.sort(np.random.default_rng(SEED+800+r).choice(len(pool),1049,False)) for r in range(REPEATS)]
        for team,c in collections.items():
            q = core.read(folder/f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            references = []
            for other in collections:
                reason = "target" if other==team else ("same_C_rows" if hashes[other]==hashes[team]
                    else ("phase_BG_QI_reuse" if reuse[other]>=.98 else "included"))
                exclusions.append(dict(phase=phase,team=team,other_team=other,reason=reason,
                    background_QI_reuse_fraction=reuse[other],same_C_rowset=hashes[other]==hashes[team]))
                if reason=="included":references.append(other)
            db = distance(q,bg,scale)
            correction, nearest = corrections(q,collections,references,db,scale)
            pair,dc = matrices(q,c,bg,scale,db)
            pred = predictions(pair,dc,q,c,correction)
            items.append(dict(phase=phase,team=team,q=q,pred=pred,correction=correction))
            for name,v in pred.items():
                diags.append(dict(phase=phase,team=team,method=name,other_C_count=len(references),
                    selected=int((v==9.5).sum()),changed_vs_baseline=int((v!=pred["baseline"]).sum()),
                    mean_other_median=float(correction["median"].mean()),mean_other_q75=float(correction["q75"].mean())))
            # 課題と対象Cの行順を入れ替えて対応を再計算する。
            for r in range(PERMUTATIONS):
                rng = np.random.default_rng(SEED+team*10+r)
                aq,bp = rng.permutation(len(q)),rng.permutation(len(c))
                changed = predictions(pair[aq][:,bp],dc[aq][:,bp],q.iloc[aq],c.iloc[bp],
                    {k:v[aq] for k,v in correction.items()})
                for name,v in changed.items():
                    restored = v[np.argsort(aq)]
                    perms.append(dict(phase=phase,team=team,method=name,kind="query_targetC",repeat=r,
                        changed=int((restored!=pred[name]).sum())))
            other_perm,_ = corrections(q,collections,references,db,scale,permute=True)
            other_pred = predictions(pair,dc,q,c,other_perm)
            for name,v in other_pred.items():
                perms.append(dict(phase=phase,team=team,method=name,kind="other_C",repeat=0,
                    changed=int((v!=pred[name]).sum()),max_other_score_difference=float(max(
                        np.max(np.abs(correction[k]-other_perm[k])) for k in correction))))
            # 予備戦のみ独立pool/refを対照として使う。公式phiは推定しない。
            if phase=="prelim" and team!=5:
                dbref = distance(q,ref,split_scale)
                corr_split,_ = corrections(q,collections,references,dbref,split_scale)
                p_split,d_split = matrices(q,c,ref,split_scale,dbref)
                split_pred = predictions(p_split,d_split,q,c,corr_split)
                null_items += [dict(team=team,source="real_C_split_reference",repeat=-1,method=name,prediction=v)
                               for name,v in split_pred.items()]
                p_pool,d_pool = matrices(q,pool,ref,split_scale,dbref)
                for r,draw in enumerate(draws):
                    cb = pool.iloc[draw].reset_index(drop=True)
                    null_pred = predictions(p_pool[:,draw],d_pool[:,draw],q,cb,corr_split)
                    null_items += [dict(team=team,source="background_C",repeat=r,method=name,prediction=v)
                                   for name,v in null_pred.items()]
            print(json.dumps(dict(phase=phase,team=team,other_C_count=len(references),seconds=round(time.time()-start,1))),flush=True)
    td = next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    truths,rows,nrows = {},[],[]
    for item in items:
        if item["phase"]!="prelim":continue
        team = item["team"]
        truth = core.read(td/f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
        assert item["q"].challenge_row_id.tolist() == truth.challenge_row_id.tolist()
        truths[team] = truth
        for name,v in item["pred"].items():
            rows.append(dict(team=team,method=name,**core.scores(v,truth)))
    for item in null_items:
        nrows.append(dict(team=item["team"],source=item["source"],repeat=item["repeat"],method=item["method"],
            **core.scores(item["prediction"],truths[item["team"]])))
    table = pd.DataFrame(rows);null = pd.DataFrame(nrows)
    table.to_csv(out/"予備戦_対象別.csv",index=False)
    null.to_csv(out/"独立背景対照_全反復.csv",index=False)
    averages,holdouts,null_mean = [],[],[]
    for group,excluded in [("20対象_自5除外",[5]),("18対象_1_2_5除外",[1,2,5])]:
        t = table[~table.team.isin(excluded)]
        a = t.groupby("method")[["r_all","r_rare","r_max","p_m","p_c"]].mean().reset_index()
        a.insert(0,"group",group);a["targets"] = t.team.nunique();averages.append(a)
        holdouts += leave_out(table, sorted(t.team.unique()),group)
        n = null[~null.team.isin(excluded)].groupby(["source","method"])[["r_all","r_rare","r_max","p_m","p_c"]].mean().reset_index()
        n.insert(0,"group",group);null_mean.append(n)
    pd.concat(averages).to_csv(out/"予備戦_20対象と18対象平均.csv",index=False)
    h = pd.DataFrame(holdouts);h.to_csv(out/"チーム留保選択_対象別.csv",index=False)
    h.groupby("group")[["r_all","r_rare","r_max","p_m","p_c"]].mean().to_csv(out/"チーム留保選択_平均.csv")
    pd.concat(null_mean).to_csv(out/"独立背景対照_平均.csv",index=False)
    delta = []
    for (team,name),d in null.groupby(["team","method"]):
        actual = d[d.source.eq("real_C_split_reference")].iloc[0]
        b = d[d.source.eq("background_C")]
        delta.append(dict(team=team,method=name,actual_R=actual.r_all,background_mean_R=b.r_all.mean(),
            excess_R=actual.r_all-b.r_all.mean(),background_R_q95=b.r_all.quantile(.95),
            excess_pm=actual.p_m-b.p_m.mean(),excess_pc=actual.p_c-b.p_c.mean()))
    pd.DataFrame(delta).to_csv(out/"独立背景対照_対象別.csv",index=False)
    pd.DataFrame(diags).to_csv(out/"本戦と予備戦_選別診断.csv",index=False)
    pd.DataFrame(exclusions).to_csv(out/"他C参照の採否.csv",index=False)
    p = pd.DataFrame(perms);p.to_csv(out/"行順変更.csv",index=False)
    previous = core.read(OLD/"集計/同点修正_対象別.csv")
    oldbase = previous[previous.method.eq("stable_matching/sex_quota/long75")].sort_values("team")
    newbase = table[table.method.eq("baseline")].sort_values("team")
    assert oldbase.team.tolist()==newbase.team.tolist()
    assert np.max(np.abs(oldbase.r_all.to_numpy()-newbase.r_all.to_numpy()))<1e-12
    official = core.validate_scores(truths)
    reference_files = ["分析.py","対応競合の頑健性.py","同点に依存しない対応.py","対応競合の分析.py"]
    dump(HERE/"実行記録.json",dict(plan=serial_plan,seed=SEED,python=platform.python_version(),executable=sys.executable,
        seconds=time.time()-start,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_code_sha256={name:hashlib.sha256((OLD/name).read_bytes()).hexdigest() for name in reference_files},
        input_sha256=core.HASHES,official_score_check=official,baseline_matches_yesterday=True,
        permutation_cases=len(p),changed_rows=int(p.changed.sum()),null_pool_n=2571,null_reference_n=2572,
        B_self_read=False,main_truth_read=False,official_phi=False,submission_created=False,codabench_submitted=False))
    print(pd.concat(averages).to_string(index=False),flush=True)
    print(h.groupby("group")[["r_all","p_m","p_c"]].mean().to_string(),flush=True)
    print(pd.concat(null_mean).to_string(index=False),flush=True)
    print(p.groupby(["kind","method"]).changed.sum().to_string(),flush=True)


if __name__=="__main__":
    run()
