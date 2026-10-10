"""較正65方式を7背景条件で比較する。回答CSV・ZIPは作成しない。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEGACY = HERE.parent / "20261007_MIA全体改善の検討"
STRENGTHS = (0,25,50,100,200,300,500,750,1000,2000,5000)
FPRS = (.005,.01,.02,.05)
REF = "flat_pref_sex_s500"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_module("分析", LEGACY / "分析.py")
old_cal = load_module("legacy_calibration", LEGACY / "非会員を使う順位較正の検討.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specs():
    result = [{"method":"raw", "family":"raw", "cross":0.},
              {"method":"raw_cross01", "family":"raw", "cross":.1}]
    for family in ("flat_pref_sex","flat_sex","flat_sex_age"):
        for strength in STRENGTHS:
            result.append({"method":f"{family}_s{strength}", "family":family,
                           "strength":strength, "cross":.1})
    for family in ("hier_pref_sex","hier_sex_age"):
        for parent in (100,500):
            for strength in (25,100,300,500,1000):
                result.append({"method":f"{family}_p{parent}_s{strength}", "family":family,
                               "strength":strength, "parent_strength":parent, "cross":.1})
    for strength in (25,50,100,200,500):
        for support in (100,200):
            result.append({"method":f"adaptive_pref_sex_s{strength}_min{support}",
                           "family":"adaptive_pref_sex", "strength":strength,
                           "min_support":support, "cross":.1})
    assert len(result) == 65 and len({s["method"] for s in result}) == 65
    return result


def stratified_subset(frame, fraction, seed):
    rng = np.random.default_rng(seed)
    indices = []
    for group in frame.groupby(["sex","prefecture"],sort=True).indices.values():
        size = max(1, int(round(len(group)*fraction)))
        indices.extend(rng.choice(group,size=size,replace=False).tolist())
    return frame.iloc[sorted(indices)].reset_index(drop=True)


def empirical_cdf(values, null):
    ordered, counts = np.unique(null,return_counts=True)
    return np.interp(values,ordered,(np.cumsum(counts)-.5*counts)/len(null),left=0.,right=1.)


def group_stats(values, b, null, controls, cols):
    global_cdf = empirical_cdf(values,null)
    local, count = global_cdf.copy(), np.zeros(len(b),dtype=int)
    bgroups = b.groupby(cols,sort=True).indices
    qgroups = controls.groupby(cols,sort=True).indices
    for key, indices in bgroups.items():
        qi = qgroups.get(key,[])
        count[indices] = len(qi)
        if len(qi):
            local[indices] = empirical_cdf(values[indices],null[qi])
    return local,count


def blend(local, count, fallback, strength):
    strength = np.broadcast_to(np.asarray(strength,dtype=float),count.shape)
    weight = np.divide(count,count+strength,out=np.zeros(len(count),dtype=float),where=count+strength>0)
    return weight*local+(1-weight)*fallback


def calibrated_variants(values,b,null,controls,configs):
    global_cdf = empirical_cdf(values,null)
    assert np.array_equal(global_cdf,old_cal.empirical_cdf(values,null))
    stats = {
        "pref_sex":group_stats(values,b,null,controls,["prefecture","sex"]),
        "sex":group_stats(values,b,null,controls,["sex"]),
        "sex_age":group_stats(values,b,null,controls,["sex","age_bin"]),
    }
    tie = rankdata(values,method="average")/len(values)
    result = {}
    for config in configs:
        family = config["family"]
        if family == "raw":
            result[config["method"]] = values
            continue
        group = "sex_age" if "sex_age" in family else "sex" if family=="flat_sex" else "pref_sex"
        local,count = stats[group]
        strength = config["strength"]
        fallback = global_cdf
        if family.startswith("hier_"):
            fallback = blend(*stats["sex"],global_cdf,config["parent_strength"])
        if family.startswith("adaptive_"):
            strength = strength*np.maximum(1.,config["min_support"]/np.maximum(count,1))
        answer = blend(local,count,fallback,strength)
        result[config["method"]] = .999*answer+.001*tie
    for strength in (100,500):
        expected = old_cal.calibrate(values,b,null,controls,["prefecture","sex"],strength)
        assert np.max(np.abs(result[f"flat_pref_sex_s{strength}"]-expected)) < 1e-14
    return result,stats


def predict(b,background,controls,cs,normal,configs,label):
    banks = {s["method"]:[] for s in configs}
    counts = []
    for index,(team,c) in enumerate(cs.items(),1):
        q = controls[team]
        query = pd.concat([b[list(common.QI7)],q[list(common.QI7)]],ignore_index=True)
        raw = common.core(background,query,c,common.NUM7,(10,))[10]
        values,null = raw[:len(b)],raw[len(b):]
        variants,stats = calibrated_variants(values,b,null,q,configs)
        for name,vector in variants.items():
            banks[name].append(rankdata(vector[normal],method="average")/normal.sum())
        for group,(local,count) in stats.items():
            counts.append({"condition":label,"team":team,"grouping":group,
                           "null_rows":len(q),"candidate_group_n_min":int(count.min()),
                           "candidate_group_n_median":float(np.median(count)),
                           "candidate_group_n_max":int(count.max()),"zero_support_candidates":int((count==0).sum())})
        if index%2==0 or index==len(cs):
            print(json.dumps({"stage":"predict","condition":label,"completed":index,"targets":len(cs)},ensure_ascii=False),flush=True)
    matrices = {}
    for config in configs:
        mat = np.column_stack(banks[config["method"]])
        if config["cross"]:
            med = np.column_stack([np.median(np.delete(mat,j,axis=1),axis=1) for j in range(len(cs))])
            mat = mat-config["cross"]*med
        assert np.isfinite(mat).all()
        matrices[config["method"]] = mat
    return matrices,counts


def setup_phase(root,phase):
    b,a,_,private,rare,cs,targets,self_id = common.phase_inputs(root,phase)
    b = b.copy()
    age_cuts = np.unique(np.quantile(a.age.to_numpy(),(.2,.4,.6,.8)))
    b["age_bin"] = np.searchsorted(age_cuts,b.age.to_numpy(),side="right")
    qs = {t:common.read(targets/f"aia_challenge_{t}.csv") for t in sorted(set(cs)|{self_id})}
    controls = {}
    for team in cs:
        q = pd.concat([v[list(common.QI7)] for t,v in qs.items() if t!=team],ignore_index=True)
        q["age_bin"] = np.searchsorted(age_cuts,q.age.to_numpy(),side="right")
        candidate_keys = pd.MultiIndex.from_frame(b.loc[~private & ~rare,list(common.QI7)])
        assert not candidate_keys.isin(pd.MultiIndex.from_frame(q[list(common.QI7)])).any()
        controls[team] = q
    return b,a,private,rare,cs,controls,age_cuts


def main_diagnostics(matrices,teams,b,normal):
    expected_path = HERE.parent/"20261008_第3回MIA_固定5bitと非会員較正/結果/F_mia.csv"
    actual = common.read(expected_path).set_index("record_id").loc[b.record_id]
    maximum_error = 0.
    reference = matrices[REF]
    for j,team in enumerate(teams):
        proposed = .6+.3*rankdata(reference[:,j],method="min")/normal.sum()
        maximum_error = max(maximum_error,float(np.max(np.abs(proposed-actual.loc[normal,str(team)].to_numpy()))))
    assert maximum_error < 1e-14
    rows=[]
    for name,mat in matrices.items():
        for count in (12,20):
            changes=[]
            for j,team in enumerate(teams):
                a=set(np.flatnonzero(rankdata(-reference[:,j],method="min")<=count))
                v=set(np.flatnonzero(rankdata(-mat[:,j],method="min")<=count))
                changed=len(v-a)
                changes.append(changed)
                rows.append({"method":name,"team":team,"top_count":count,"members_known":False,
                             "reference_selected_count":len(a),"current_selected_count":len(v),
                             "new_candidates":changed,"removed_candidates":len(a-v),
                             "rank_correlation":float(np.corrcoef(rankdata(reference[:,j]),rankdata(mat[:,j]))[0,1])})
    return rows,maximum_error


def evaluate(root,b,private,rare,teams,all_matrices):
    truthdir = common.child(common.child(root/"予備戦","配布データ")/"結果","正解データ")
    own = common.read(truthdir/"pool_ids_5.csv").set_index("record_id").loc[b.record_id]
    truth = np.column_stack([own.pool_id.isin(common.read(truthdir/f"pool_ids_{t}.csv").pool_id).to_numpy(int) for t in teams])
    assert (truth.sum(axis=1)<=1).all() and not truth[private].any()
    kit=common.child(common.child(root/"本戦","配布データ"),"pwscup2026-kit-main_revised0912")
    sys.path.insert(0,str(kit/"starter/src"))
    from pwscup2026.scoring.attack_score import _tpr_at_fpr
    import pwscup2026.scoring.attack_score as official_module
    common.INPUT_HASHES[str(Path(official_module.__file__).resolve())]=sha(official_module.__file__)
    rows=[]
    max_error=0.
    normal=~private & ~rare
    for condition,matrices in all_matrices.items():
        for name,mat in matrices.items():
            for j,team in enumerate(teams):
                confidence=common.confidence(mat[:,j],normal,rare,private)
                fpr,tpr,_=roc_curve(truth[:,j],confidence,drop_intermediate=False)
                for point in FPRS:
                    value=float(np.interp(point,fpr,tpr))
                    official,_=_tpr_at_fpr(confidence,truth[:,j],point)
                    max_error=max(max_error,abs(value-official))
                    assert abs(value-official)<1e-12
                    rows.append({"condition":condition,"method":name,"team":team,"fpr":point,
                                 "mia_all":value,"members":int(truth[:,j].sum())})
        print(json.dumps({"stage":"evaluate","condition":condition},ensure_ascii=False),flush=True)
    return pd.DataFrame(rows),max_error


def summarize(table,configs):
    config_map={s["method"]:s for s in configs}
    rows=[]
    for (condition,point),part in table.groupby(["condition","fpr"],sort=True):
        reference=part[part.method==REF].set_index("team").mia_all
        baseline=part[part.method=="raw"].set_index("team").mia_all
        other=reference.index[~reference.index.isin([1,2])]
        for name,group in part.groupby("method",sort=True):
            score=group.set_index("team").mia_all
            delta=score-reference
            d=delta.loc[other]
            base_delta=score.loc[other]-baseline.loc[other]
            positive=d[d>1e-12].sort_values(ascending=False)
            dropped=d.drop(positive.index[:2])
            rows.append({**config_map[name],"condition":condition,"fpr":float(point),
                         "all20":float(score.mean()),"other18":float(score.loc[other].mean()),
                         "delta_all_vs500":float(delta.mean()),"delta_other18_vs500":float(d.mean()),
                         "wins_other18_vs500":int((d>1e-12).sum()),"losses_other18_vs500":int((d< -1e-12).sum()),
                         "wins_all_vs500":int((delta>1e-12).sum()),"losses_all_vs500":int((delta< -1e-12).sum()),
                         "wins_other18_vs_raw":int((base_delta>1e-12).sum()),"losses_other18_vs_raw":int((base_delta< -1e-12).sum()),
                         "worst_target_delta_vs500":float(d.min()),
                         "largest_positive_share":float(positive.iloc[0]/positive.sum()) if len(positive) else 0.,
                         "delta_sum_after_removing_top2_gains":float(dropped.sum())})
    summary=pd.DataFrame(rows)
    robust=[]
    at01=summary[summary.fpr==.01]
    for name,group in at01.groupby("method",sort=True):
        sampled=group[group.condition!="full"]
        full=group[group.condition=="full"].iloc[0]
        robust.append({**config_map[name],"full_other18":float(full.other18),"full_all20":float(full.all20),
                       "full_other18_delta":float(full.delta_other18_vs500),
                       "full_wins_other18":int(full.wins_other18_vs500),"full_losses_other18":int(full.losses_other18_vs500),
                       "full_largest_positive_share":float(full.largest_positive_share),
                       "full_delta_sum_without_top2_gains":float(full.delta_sum_after_removing_top2_gains),
                       "sample6_mean_other18":float(sampled.other18.mean()),
                       "sample6_mean_delta":float(sampled.delta_other18_vs500.mean()),
                       "sample6_wins":int((sampled.delta_other18_vs500>1e-12).sum()),
                       "sample6_losses":int((sampled.delta_other18_vs500< -1e-12).sum()),
                       "sample6_worst_delta":float(sampled.delta_other18_vs500.min()),
                       "sample6_mean_target_wins":float(sampled.wins_other18_vs500.mean()),
                       "sample6_mean_target_losses":float(sampled.losses_other18_vs500.mean())})
    return summary,pd.DataFrame(robust)


def selection_diagnostic(table):
    part=table[(table.condition=="full") & (table.fpr==.01) & ~table.team.isin([1,2])]
    pivot=part.pivot(index="team",columns="method",values="mia_all").sort_index()
    methods=sorted(pivot.columns)
    records=[]
    for team in pivot.index:
        train=pivot.drop(index=team)
        average=train.mean()
        best=float(average.max())
        selected=next(name for name in methods if abs(float(average[name])-best)<1e-12)
        records.append({"held_out_team":int(team),"selected_method":selected,"training_other17_mean":best,
                        "held_out_score":float(pivot.loc[team,selected]),"reference_score":float(pivot.loc[team,REF]),
                        "delta":float(pivot.loc[team,selected]-pivot.loc[team,REF]),
                        "note":"同じ予備戦を再利用した方式選択の感度診断。独立検証ではない。"})
    return pd.DataFrame(records)


def verify_saved_baselines(table):
    source=LEGACY/"非会員較正の感度結果/較正感度_対象別比較.csv"
    saved=common.read(source)
    mapping={"flat_pref_sex_s100":"Q_pref_sex100_cross01","flat_pref_sex_s500":"Q_pref_sex500_cross01",
             "raw":"baseline","raw_cross01":"baseline_cross01"}
    errors=[]
    for new,old in mapping.items():
        for condition in table.condition.unique():
            actual=table[(table.condition==condition)&(table.method==new)&(table.fpr==.01)].set_index("team").mia_all
            expected=saved[saved.method==condition+"__"+old].set_index("team").mia_all
            assert set(actual.index)==set(expected.index)
            error=float(np.max(np.abs(actual.loc[expected.index]-expected)))
            assert error<1e-12
            errors.append({"condition":condition,"method":new,"maximum_error":error})
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output-dir",type=Path,default=HERE/"結果")
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("既存結果には上書きしません。新しい出力先を指定してください。")
    started=time.perf_counter()
    started_jst=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    configs=specs()
    initial_pins={str(p.resolve()):sha(p) for p in (Path(__file__),HERE/"実験方針.json",LEGACY/"分析.py",LEGACY/"非会員を使う順位較正の検討.py")}
    b,a,private,rare,cs,controls,age_cuts=setup_phase(args.root,"prelim")
    references=[("full",1.,0,a)]+[(f"fraction{frac}_seed{seed}",frac,seed,stratified_subset(a,frac,seed))
                                 for frac in (.9,.8,.5) for seed in (20261007,20261008)]
    all_matrices={}
    counts=[]
    reference_rows=[]
    for label,fraction,seed,background in references:
        matrix,ct=predict(b,background,controls,cs,~private & ~rare,configs,label)
        all_matrices[label]=matrix
        counts.extend(ct)
        reference_rows.append({"condition":label,"fraction":fraction,"seed":seed,"background_rows":len(background)})
    mb,ma,mp,mr,mcs,mcontrols,main_age_cuts=setup_phase(args.root,"main")
    main_matrices,main_counts=predict(mb,ma,mcontrols,mcs,~mp & ~mr,configs,"main")
    counts.extend(main_counts)
    main_top,main_reproduction=main_diagnostics(main_matrices,list(mcs),mb,~mp & ~mr)
    print(json.dumps({"stage":"predictions_fixed_before_truth","methods":len(configs),"references":len(references),
                      "main_round3_max_error":main_reproduction},ensure_ascii=False),flush=True)
    table,official_error=evaluate(args.root,b,private,rare,list(cs),all_matrices)
    summary,robust=summarize(table,configs)
    selection=selection_diagnostic(table)
    reproductions=verify_saved_baselines(table)
    for path,expected in {**initial_pins,**common.INPUT_HASHES}.items():
        assert sha(path)==expected
    args.output_dir.mkdir(parents=True)
    tables={"対象別TPR.csv":table,"集約比較.csv":summary,"頑健性比較.csv":robust,
            "対象を外した方式選択.csv":selection,"本戦順位の変化.csv":pd.DataFrame(main_top),
            "較正の標本数.csv":pd.DataFrame(counts),"背景条件.csv":pd.DataFrame(reference_rows)}
    for name,frame in tables.items():
        frame.to_csv(args.output_dir/name,index=False,float_format="%.17g")
    record={"started_jst":started_jst,"completed_jst":datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
            "seconds":time.perf_counter()-started,"python":sys.executable,"python_version":platform.python_version(),
            "methods":len(configs),"prelim_background_conditions":len(references),"prelim_targets":len(cs),
            "main_targets":len(mcs),"prelim_age_cuts":age_cuts.tolist(),"main_age_cuts":main_age_cuts.tolist(),
            "input_and_code_sha256":{**initial_pins,**common.INPUT_HASHES},
            "official_tpr_max_error":official_error,"main_round3_max_error":main_reproduction,
            "saved_baseline_reproductions":reproductions,"main_truth_used":False,
            "submission_files_written":False,"coda_submission":False,
            "selection_note":"多数設定と既存予備戦正解の再利用。背景抽出・対象除外も独立コホートの検証ではない。",
            "outputs_sha256":{name:sha(args.output_dir/name) for name in tables}}
    (args.output_dir/"実行記録.json").write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(robust.sort_values(["sample6_mean_delta","full_other18"],ascending=False).head(12).to_json(orient="records",force_ascii=False),flush=True)
    print(json.dumps({"stage":"complete","seconds":record["seconds"],"output_dir":str(args.output_dir)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
