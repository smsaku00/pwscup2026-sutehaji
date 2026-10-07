"""既存P01に公開BG一致行の事前重みを適用する固定比較。

本戦3の公開特徴を観測した後の設計であり、独立検証ではない。
基準との全21予備戦比較と全23本戦動作診断。予備18対象は独立BG参照の
20反復対照も行う。提出個票・本戦正解・CodaBenchを使わない。
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib, importlib.util, io, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = HERE.parents[1]/"20261007_AIA全体向上の分析"
OUT = HERE/"集計"
sys.path.insert(0, str(OLD))
spec = importlib.util.spec_from_file_location("stable_matching_reference", OLD/"同点に依存しない対応.py")
stable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable)
core = stable.core
MODE = ["baseline", "BG_weight0.5", "BG_exclude"]
SEED = 20261008
REPEATS = 20
HASHES = {}
PROTOCOL = dict(seed=SEED, methods=MODE, tau_numeric_absolute=1e-8,
                exact_categories=["sex", "prefecture"], numerical_columns=core.NUM,
                known_background="phase BG union old-kit BG; raw QI approximate row match",
                source_feature_is_preprocessing=True, baseline_partial_attributes=2,
                baseline_sex_quota=.75, baseline_answer_selected=9.5,
                baseline_answer_other=0, near_limit=12, boundary_ties_included=True,
                weight_semantics="pair score plus log(0.5) for known-BG C row",
                exclude_semantics="remove known-BG C rows before near-neighbor boundary and assignment",
                shortage_semantics="selection count per sex min(ceil(query sex count*.75), retained C sex count); other answers 0",
                negative_repeats=REPEATS, negative_split="phase BG first floor(n/2) sample pool; rest reference; disjoint permuted indices",
                negative_sample_count=1049, main_truth_used=False,
                scoring_after_prediction_generation=True,
                posthoc_design="motivated by main team3 public-BG overlap; no method parameters selected by main scores",
                phi_official_reproduction=False, submission_created=False)


def read(path):
    data = Path(path).read_bytes()
    HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return pd.read_csv(io.BytesIO(data), float_precision="round_trip")


def known_rows(c, bg):
    x, y = c[core.NUM].to_numpy(float), bg[core.NUM].to_numpy(float)
    scale = np.maximum(y.std(0, ddof=1), 1e-5)
    d = cdist(x/scale, y/scale)
    same = (c.sex.to_numpy()[:, None]==bg.sex.to_numpy()[None, :]) & (c.prefecture.to_numpy()[:, None]==bg.prefecture.to_numpy()[None, :])
    d[~same] = np.inf
    j = d.argmin(1)
    return np.isfinite(d[np.arange(len(c)), j]) & (np.abs(x-y[j])<=1e-8).all(1)


def select(pair, dc, qs, cs, remove=None):
    if remove is None:
        remove = np.zeros(len(cs), bool)
    chosen = np.zeros(len(qs), bool)
    requested_total = bad = shortage = 0
    for sex in np.unique(qs):
        a, b = qs==sex, (cs==sex)&~remove
        nq, nc = int(a.sum()), int(b.sum())
        requested = int(np.ceil(nq*.75))
        requested_total += requested
        count = min(requested, nc)
        shortage += requested-count
        if count==0:
            continue
        distances = dc[a][:, b]
        limit = np.partition(distances, min(11, nc-1), axis=1)[:, min(11, nc-1)]
        valid = np.isfinite(distances) & (distances<=limit[:, None])
        limited = np.where(valid, pair[a][:, b], -1e4)
        one, assignment = stable.partial_assignment(limited, count)
        chosen[a] = one
        bad += int((limited[np.flatnonzero(one), assignment[one]]<=-9999).sum())
    return chosen, dict(requested=requested_total, selected=int(chosen.sum()), shortage=shortage, selected_bad_pairs=bad)


def predict(q, c, ref, flagged, db=None):
    """公開入力だけを受け取る。ラベル・正解timeは引数にない。"""
    if db is None:
        pair, dc = stable.matrices(q, c, ref)
    else:
        b = core.raw(ref)
        scale = np.maximum(np.quantile(b, .75, axis=0)-np.quantile(b, .25, axis=0), 1e-5)
        dc = stable.distance(q, c, scale)
        pair = stable.pair_score(dc, db)["continuous"]
        full = cdist(core.raw(q)/scale, core.raw(c)/scale, "sqeuclidean")/5
        pair -= stable.EPSILON*np.minimum(full, 1000.)
    qs, cs = q.sex.to_numpy(), c.sex.to_numpy()
    result, diagnostics = {}, []
    chosen_base = None
    for method in MODE:
        adjusted = pair + np.log(.5)*flagged[None, :] if method=="BG_weight0.5" else pair
        chosen, diagnostic = select(adjusted, dc, qs, cs, flagged if method=="BG_exclude" else None)
        if method=="baseline":
            chosen_base = chosen
            expected = stable.select(pair, dc, qs, cs, "sex_quota")
            assert np.array_equal(chosen, expected)
        result[method] = np.where(chosen, 9.5, 0.)
        diagnostics.append(dict(method=method, flagged_C=int(flagged.sum()), retained_C=int((~flagged).sum()) if method=="BG_exclude" else len(c),
                                changed_from_baseline=int((chosen!=chosen_base).sum()), **diagnostic))
    return result, diagnostics


def run():
    start = time.time()
    OUT.mkdir(exist_ok=True)
    # 採点実行前に固定し、成績を見て書き換えない。
    protocol_path = HERE/"公開BG比較_事前固定仕様.json"
    protocol_path.write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2)+"\n")
    td = next((ROOT/"予備戦/配布データ/結果").glob("正解*"))
    prior_teams = sorted(int(p.stem.rsplit("_", 1)[1]) for p in td.glob("aia_truth_*.csv"))
    assert len(prior_teams)==21
    kit = read(ROOT/"スターターキット/初版/participant_data/A_bg.csv")
    sets = [("prelim", ROOT/"予備戦/配布データ/PWSCup2026_prelim_attack", ROOT/"予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv", prior_teams),
            ("main", ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力", ROOT/"本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv", None)]
    primary, diagnostics, permutations, real_ref, null = [], [], [], [], []
    for phase, folder, bgpath, teams in sets:
        bg = read(bgpath)
        if teams is None:
            teams = sorted(int(p.stem.rsplit("_", 1)[1]) for p in folder.glob("C_*.csv"))
        idx = np.random.default_rng(SEED).permutation(len(bg))
        pool, ref = (bg.iloc[j].reset_index(drop=True) for j in [idx[:len(bg)//2], idx[len(bg)//2:]])
        assert set(idx[:len(bg)//2]).isdisjoint(idx[len(bg)//2:])
        samples = [pool.iloc[np.sort(np.random.default_rng(SEED+400+rep).choice(len(pool), 1049, False))].reset_index(drop=True) for rep in range(REPEATS)]
        for team in teams:
            c = read(folder/f"C_{team}.csv")
            q = read(folder/f"aia_challenge_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            flagged = known_rows(c, bg)|known_rows(c, kit)
            preds, diag = predict(q, c, bg, flagged)
            diagnostics += [dict(phase=phase, team=team, **rec) for rec in diag]
            # 実Cの予測をすべて生成した後で初めて正解をロード。
            truth = read(td/f"aia_truth_{team}.csv").sort_values("challenge_row_id").reset_index(drop=True) if phase=="prelim" else None
            if truth is not None:
                assert q.challenge_row_id.tolist()==truth.challenge_row_id.tolist()
                primary += [dict(team=team, method=name, **core.scores(pred, truth)) for name, pred in preds.items()]
            for repeat in range(3):
                rng = np.random.default_rng(SEED+int(team)*100+repeat)
                a, b = rng.permutation(len(q)), rng.permutation(len(c))
                shuffled, _ = predict(q.iloc[a], c.iloc[b], bg, flagged[b])
                for name, prediction in shuffled.items():
                    permutations.append(dict(phase=phase, team=team, repeat=repeat, method=name,
                                              selected_changed=int((prediction[np.argsort(a)]!=preds[name]).sum())))
            if phase=="prelim" and team not in [1, 2, 5]:
                raw_ref = core.raw(ref)
                scale = np.maximum(np.quantile(raw_ref, .75, axis=0)-np.quantile(raw_ref, .25, axis=0), 1e-5)
                db = stable.distance(q, ref, scale)
                rp, _ = predict(q, c, ref, flagged, db)
                real_ref += [dict(team=team, method=name, **core.scores(pred, truth)) for name, pred in rp.items()]
                for repeat, sample in enumerate(samples):
                    # すべて公開BG標本から採ったことが判明している行。
                    known = np.ones(len(sample), bool)
                    npred, ndiag = predict(q, sample, ref, known, db)
                    null += [dict(team=team, repeat=repeat, method=name, **core.scores(pred, truth)) for name, pred in npred.items()]
            print(json.dumps(dict(phase=phase, team=int(team), seconds=round(time.time()-start, 1))), flush=True)
    primary = pd.DataFrame(primary)
    primary.to_csv(OUT/"公開BG処理_予備戦対象別.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"公開BG処理_44C動作診断.csv", index=False)
    pd.DataFrame(permutations).to_csv(OUT/"公開BG処理_行順変更.csv", index=False)
    summaries = []
    for scope, excluded in [("all20", [5]), ("processed18", [1, 2, 5])]:
        selected = primary[~primary.team.isin(excluded)]
        baseline = selected[selected.method.eq("baseline")].set_index("team")
        for method, one in selected.groupby("method", sort=False):
            summary = dict(scope=scope, method=method, n=len(one), **one[["r_all", "r_rare", "r_max", "p_m", "p_c"]].mean().to_dict(),
                           signed_hit_difference=float((one.hit_m-one.hit_c).mean()),
                           improved=int((one.set_index("team").r_all-baseline.r_all).gt(1e-12).sum()),
                           worsened=int((one.set_index("team").r_all-baseline.r_all).lt(-1e-12).sum()))
            summaries.append(summary)
    pd.DataFrame(summaries).to_csv(OUT/"公開BG処理_20対象と18対象平均.csv", index=False)
    real_ref, null = pd.DataFrame(real_ref), pd.DataFrame(null)
    real_ref.to_csv(OUT/"公開BG処理_独立参照の実C.csv", index=False)
    null.to_csv(OUT/"公開BG処理_背景対照全反復.csv", index=False)
    per_target = real_ref.merge(null.groupby(["team", "method"])[["r_all", "p_m", "p_c", "hit_m", "hit_c"]].mean().reset_index(), on=["team", "method"], suffixes=("_real", "_null"))
    per_target["r_excess"] = per_target.r_all_real-per_target.r_all_null
    per_target.to_csv(OUT/"公開BG処理_実Cと背景Cの差.csv", index=False)
    per_target.groupby("method")[["r_all_real", "r_all_null", "r_excess", "p_m_real", "p_c_real", "p_m_null", "p_c_null"]].mean().to_csv(OUT/"公開BG処理_背景対照平均.csv")
    # 基準の保存結果と一致。新処理の有利不利に関係なく基準を動かさない。
    saved = read(OLD/"集計/同点修正_対象別.csv")
    saved = saved[saved.method.eq("stable_matching/sex_quota/long75")].set_index("team")
    baseline = primary[primary.method.eq("baseline")].set_index("team")
    assert np.array_equal(saved.loc[baseline.index, "hit_m"], baseline.hit_m)
    assert np.array_equal(saved.loc[baseline.index, "hit_c"], baseline.hit_c)
    imports = ["同点に依存しない対応.py", "対応競合の分析.py", "対応競合の頑健性.py", "分析.py"]
    record = dict(PROTOCOL, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(), input_sha256=HASHES,
                  imported_code_sha256={str(OLD/name):hashlib.sha256((OLD/name).read_bytes()).hexdigest() for name in imports},
                  python=sys.executable, python_version=platform.python_version(), seconds=time.time()-start,
                  baseline_hit_counts_match_saved=True, prediction_artifacts_written=False,
                  independent_BG_pool_n=len(pool), independent_BG_reference_n=len(ref),
                  row_permutation_cases=len(permutations), max_selected_changed=max(p["selected_changed"] for p in permutations))
    (HERE/"公開BG比較_実行記録.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(per_target.groupby("method")[["r_all_real", "r_all_null", "r_excess"]].mean().to_string(), flush=True)


if __name__=="__main__":
    run()
