"""保存予測・採点・仮説確率の検証と、固定した背景参照差の集計。
候補・係数・スコア・予測は変更しない。個々の課題行は共有しない。
"""
from __future__ import annotations
import hashlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"集計"
SOURCE=HERE.parent/"臨床値集合の複数近傍"
OLD=HERE.parents[1]/"20261007_AIA全体向上の分析"
HASHES={}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    data=Path(path).read_bytes()
    HASHES[str(path)]=hashlib.sha256(data).hexdigest()
    return pd.read_csv(io.BytesIO(data),float_precision="round_trip")


def run():
    pred=json.loads((HERE/"予測実行記録.json").read_text())
    score=json.loads((HERE/"採点実行記録.json").read_text())
    cfg=json.loads((HERE/"固定条件.json").read_text())
    assert pred["code_sha256"]==score["code_sha256"]==sha(HERE/"分析.py")
    assert pred["config_sha256"]==score["config_sha256"]==sha(HERE/"固定条件.json")
    assert all(sha(path)==h for path,h in pred["input_sha256"].items())
    assert all(sha(path)==h for path,h in pred["imports_sha256"].items())
    assert all(sha(path)==h for path,h in score["public_prelim_truth_sha256"].items())
    assert score["prediction_receipt_sha256"]==sha(HERE/"予測実行記録.json")
    assert pred["source_prediction_receipt_sha256"]==sha(SOURCE/"予測実行記録.json")
    assert all(sha(SOURCE/path)==h for path,h in pred["source_prediction_hashes"].items())
    assert cfg["smoothing"]==64 and cfg["new_candidate_count"]==2 and cfg["last_method_addition"]
    assert pred["all_predictions_saved_before_truth_read"] and not pred["in_first39_frozen_pool"]
    changes=[]
    masks=0
    for item in pred["predictions"]:
        assert sha(HERE/item["path"])==item["sha256"]
        data=np.load(HERE/item["path"],allow_pickle=False)
        assert data["real"].shape==(2,5,326)
        assert len(np.unique(data["challenge_row_ids"]))==326
        if item["phase"]=="prelim" and item["team"]!=5:
            assert data["null"].shape==(2,20,5,326)
        else:
            assert data["null"].size==0
        for sex in [0,1]:
            rows=data["sex"]==sex
            quota=int(np.ceil(rows.sum()*.75))
            assert (data["real"][:,:,rows].sum(-1)==quota).all()
            if data["null"].size:
                assert (data["null"][:,:,:,rows].sum(-1)==quota).all()
        for sex in ["all",0,1]:
            rows=np.ones(326,bool) if sex=="all" else data["sex"]==sex
            for mi,method in enumerate(cfg["methods"]):
                half,full=data["real"][0,mi,rows],data["real"][1,mi,rows]
                changes.append(dict(phase=item["phase"],team=item["team"],sex=str(sex),method=method,q_n=int(rows.sum()),
                    changed=int((half!=full).sum()),jaccard=float((half&full).sum()/(half|full).sum()),
                    selected_split=int(half.sum()),selected_full=int(full.sum())))
        masks+=2*5+(2*20*5 if data["null"].size else 0)
    table=read(OUT/"予備戦_固定5候補と20null対象別.csv")
    assert len(table)==21*2*5+20*20*2*5
    assert table[table.source.eq("BG_null_C")].groupby(["team","context","method"]).repeat.nunique().eq(20).all()
    assert table.signed_hit_difference.eq(table.hit_m-table.hit_c).all()
    formula=np.clip((table.hit_m-table.hit_c)/(163-table.hit_c),0,1)
    assert np.max(np.abs(formula-table.r_all))<1e-12
    pm,pc=table.hit_m_rare/13,table.hit_c_rare/13
    rare=np.clip((pm-pc-1.64*np.sqrt(pm*(1-pm)/13+pc*(1-pc)/13))/np.maximum(1-pc,1e-9),0,1)
    assert np.max(np.abs(rare-table.r_rare))<1e-12
    assert np.max(np.abs(np.maximum(rare,table.r_all)-table.r_max))<1e-12
    average=read(OUT/"20対象と18対象の平均.csv")
    assert len(average)==20
    for row in average.itertuples(index=False):
        excluded=[5] if row.scope=="prelim20" else [1,2,5]
        use=table[~table.team.isin(excluded)&table.context.eq(row.context)&table.method.eq(row.method)]
        real,null=use[use.source.eq("real_C")],use[use.source.eq("BG_null_C")]
        assert len(real)==(20 if row.scope=="prelim20" else 18)
        for name,value in [("real_r_all",real.r_all.mean()),("null_r_all",null.r_all.mean()),("r_excess",real.r_all.mean()-null.r_all.mean()),
                           ("real_signed_hit_difference",real.signed_hit_difference.mean()),("null_signed_hit_difference",null.signed_hit_difference.mean()),
                           ("signed_hit_excess",real.signed_hit_difference.mean()-null.signed_hit_difference.mean())]:
            assert abs(getattr(row,name)-value)<1e-12
    lt=read(OUT/"固定5候補_LOTO対象別.csv")
    assert len(lt)==76
    for row in lt.itertuples(index=False):
        excluded=[5] if row.scope=="prelim20" else [1,2,5]
        matrix=table[table.source.eq("real_C")&table.context.eq(row.context)&~table.team.isin(excluded)].pivot(index="team",columns="method",values="r_all").reindex(columns=cfg["methods"])
        training=matrix.drop(index=row.held_team).mean().to_numpy()
        chosen=int(np.flatnonzero(training>=training.max()-1e-12)[0])
        assert row.selected_method==cfg["methods"][chosen]
        assert abs(row.train_R-training[chosen])<1e-12 and abs(row.held_R-matrix.loc[row.held_team,row.selected_method])<1e-12
    counts=read(OUT/"全44C_最小距離と背景人数の別実装照合.csv")
    perms=read(OUT/"行順変更.csv")
    reproduced=read(OUT/"初回3方式との実Cとnull採点一致.csv")
    assert len(counts)==88 and counts.maximum_C_distance_difference.eq(0).all() and counts.maximum_BG_count_difference.eq(0).all()
    assert len(perms)==2200 and perms.changed.eq(0).all()
    assert len(reproduced)==2526 and reproduced.maximum_score_difference.lt(1e-12).all()
    initial_sampling=read(SOURCE/"集計/初回条件の同性別null抽出監査.csv")
    sampling=read(OUT/"初回条件の同性別null抽出監査.csv")
    assert len(sampling)==800 and sampling.same_sex_counts.all() and sampling.split_pool_ref_disjoint.all()
    joined=sampling.merge(initial_sampling,on=["team","context","repeat"],suffixes=("_new","_source"))
    assert len(joined)==800 and joined.original_BG_index_sha256_new.eq(joined.original_BG_index_sha256_source).all()
    assert sampling.null_sample_contained_in_current_reference.eq(sampling.context.eq("full_reference_overlap")).all()
    old=read(OLD/"集計/同点修正_対象別.csv")
    old=old[old.method.eq("stable_matching/sex_quota/long75")].set_index("team")
    full=table[table.source.eq("real_C")&table.context.eq("full_reference_overlap")&table.method.eq(cfg["methods"][0])].set_index("team")
    assert len(full)==21
    for name in ["r_all","r_rare","r_max","hit_m","hit_c","hit_m_rare","hit_c_rare"]:
        assert np.max(np.abs(full[name]-old.loc[full.index,name]))<1e-12
    background=pd.DataFrame(changes)
    background.to_csv(OUT/"全BGとsplit参照の選択変化_対象別性別.csv",index=False)
    summary=[]
    for scope,phase,excluded in [("prelim20","prelim",[5]),("processed18","prelim",[1,2,5]),("main23","main",[])]:
        use=background[background.phase.eq(phase)&~background.team.isin(excluded)]
        for (sex,method),one in use.groupby(["sex","method"]):
            summary.append(dict(scope=scope,sex=sex,method=method,targets=len(one),changed_mean=float(one.changed.mean()),
                changed_min=int(one.changed.min()),changed_max=int(one.changed.max()),jaccard_mean=float(one.jaccard.mean()),
                jaccard_min=float(one.jaccard.min()),jaccard_max=float(one.jaccard.max())))
    pd.DataFrame(summary).to_csv(OUT/"全BGとsplit参照の選択変化_平均と範囲.csv",index=False)
    small=json.loads((HERE/"小型_有限人口確率の別実装照合.json").read_text())
    assert small["maximum_comb_error"]<1e-12 and small["maximum_enumeration_error"]<1e-12
    shared=[]
    for path in OUT.glob("*.csv"):
        columns=pd.read_csv(path,nrows=0).columns
        assert "challenge_row_id" not in columns and "challenge_row_ids" not in columns
        shared.append(path.name)
    forbidden=[str(p) for p in HERE.rglob("*") if p.name in ["F_aia.csv","F_mia.csv","token.txt"] or p.suffix==".zip"]
    assert not forbidden
    result=dict(code_sha256=sha(__file__),prediction_code_sha256=pred["code_sha256"],checked_prediction_input_hashes=len(pred["input_sha256"]),
        checked_import_hashes=len(pred["imports_sha256"]),checked_truth_hashes=len(score["public_prelim_truth_sha256"]),checked_prediction_masks=masks,
        checked_score_rows=len(table),checked_average_rows=len(average),checked_LOTO_rows=len(lt),checked_count_cases=len(counts),
        checked_permutation_cases=len(perms),matched_initial_score_cases=len(reproduced),matched_null_index_cases=len(joined),
        old_full21_scores_match=True,small_math_comb_cases=small["math_comb_cases"],small_enumeration_cases=small["full_subset_enumeration_cases"],
        added_fixed_background_diagnostic_rows=len(background),verification_input_sha256=HASHES,share_csv_files=shared,
        mathematical_results_changed=False,sharing_predictions=False,in_first39_frozen_pool=False,posthoc_inspiration=True,
        last_method_addition=True,forbidden_submission_files=forbidden)
    (HERE/"検証記録.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print({key:value for key,value in result.items() if key not in ["verification_input_sha256","share_csv_files"]})


if __name__=="__main__":
    run()
