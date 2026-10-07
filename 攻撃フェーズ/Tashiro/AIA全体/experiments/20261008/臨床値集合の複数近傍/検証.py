"""固定7方式の保存結果を検証し、背景参照差を保存maskから集計する。
補助表の背景重複の列名だけを明確にする。
予測・距離・得点・null索引・候補条件は変更しない。
"""
from __future__ import annotations
import hashlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"集計"
HASHES={}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    content=Path(path).read_bytes()
    HASHES[str(path)]=hashlib.sha256(content).hexdigest()
    return pd.read_csv(io.BytesIO(content),float_precision="round_trip")


def run():
    pred=json.loads((HERE/"予測実行記録.json").read_text())
    score=json.loads((HERE/"採点実行記録.json").read_text())
    cfg=json.loads((HERE/"固定条件.json").read_text())
    assert pred["code_sha256"]==score["code_sha256"]==sha(HERE/"分析.py")
    assert pred["config_sha256"]==score["config_sha256"]==sha(HERE/"固定条件.json")
    assert all(sha(p)==h for p,h in pred["input_sha256"].items())
    assert all(sha(p)==h for p,h in pred["imports_sha256"].items())
    assert all(sha(p)==h for p,h in score["public_prelim_truth_sha256"].items())
    assert sha(HERE/"予測実行記録.json")==score["prediction_receipt_sha256"]
    assert pred["all_predictions_saved_before_truth_read"] and not pred["in_first39_frozen_pool"]
    assert cfg["neighbour_k"]==[1,3,8] and cfg["new_candidate_count"]==4
    masks=0
    background_changes=[]
    for item in pred["predictions"]:
        assert sha(HERE/item["path"])==item["sha256"]
        data=np.load(HERE/item["path"],allow_pickle=False)
        assert data["real"].shape==(2,7,326)
        assert len(np.unique(data["challenge_row_ids"]))==326
        if item["phase"]=="prelim" and item["team"]!=5:
            assert data["null"].shape==(2,20,7,326)
        else:
            assert data["null"].size==0
        for sex in [0,1]:
            mask=data["sex"]==sex
            quota=int(np.ceil(mask.sum()*.75))
            assert data["real"][:,:,mask].sum(-1).tolist()==[[quota]*7]*2
            if data["null"].size:
                assert (data["null"][:,:,:,mask].sum(-1)==quota).all()
        for sex in ["all",0,1]:
            selected_rows=np.ones(326,bool) if sex=="all" else data["sex"]==sex
            for mi,method in enumerate(cfg["methods"]):
                half,full=data["real"][0,mi,selected_rows],data["real"][1,mi,selected_rows]
                union=(half|full).sum()
                background_changes.append(dict(phase=item["phase"],team=item["team"],sex=str(sex),method=method,
                    q_n=int(selected_rows.sum()),selected_split=int(half.sum()),selected_full=int(full.sum()),
                    changed=int((half!=full).sum()),jaccard=float((half&full).sum()/union)))
        masks+=2*7+(2*20*7 if data["null"].size else 0)
    table=read(OUT/"予備戦_対象別と20反復null.csv")
    assert len(table)==21*2*7+20*20*2*7
    assert table[table.source.eq("real_C")].groupby(["team","context"]).method.nunique().eq(7).all()
    assert table[table.source.eq("BG_null_C")].groupby(["team","context","method"]).repeat.nunique().eq(20).all()
    assert table.signed_hit_difference.eq(table.hit_m-table.hit_c).all()
    formula=np.clip((table.hit_m-table.hit_c)/(163-table.hit_c),0,1)
    assert np.max(np.abs(formula-table.r_all))<1e-12
    pm,pc=table.hit_m_rare/13,table.hit_c_rare/13
    rare=np.clip((pm-pc-1.64*np.sqrt(pm*(1-pm)/13+pc*(1-pc)/13))/np.maximum(1-pc,1e-9),0,1)
    assert np.max(np.abs(rare-table.r_rare))<1e-12
    assert np.max(np.abs(np.maximum(table.r_all,table.r_rare)-table.r_max))<1e-12
    averages=read(OUT/"20対象と18対象の平均.csv")
    assert len(averages)==2*2*7
    for row in averages.itertuples(index=False):
        excluded=[5] if row.scope=="prelim20" else [1,2,5]
        one=table[~table.team.isin(excluded)&table.context.eq(row.context)&table.method.eq(row.method)]
        real,null=one[one.source.eq("real_C")],one[one.source.eq("BG_null_C")]
        assert len(real)==(20 if row.scope=="prelim20" else 18)
        for name,actual in [("real_r_all",real.r_all.mean()),("null_r_all",null.r_all.mean()),
                            ("r_excess",real.r_all.mean()-null.r_all.mean()),
                            ("real_signed_hit_difference",real.signed_hit_difference.mean()),
                            ("null_signed_hit_difference",null.signed_hit_difference.mean()),
                            ("signed_hit_excess",real.signed_hit_difference.mean()-null.signed_hit_difference.mean())]:
            assert abs(getattr(row,name)-actual)<1e-12
    lt=read(OUT/"固定7候補_LOTO対象別.csv")
    assert len(lt)==76
    for row in lt.itertuples(index=False):
        excluded=[5] if row.scope=="prelim20" else [1,2,5]
        matrix=table[table.source.eq("real_C")&table.context.eq(row.context)&~table.team.isin(excluded)].pivot(index="team",columns="method",values="r_all").reindex(columns=cfg["methods"])
        training=matrix.drop(index=row.held_team).mean().to_numpy()
        selected=np.flatnonzero(training>=training.max()-1e-12)[0]
        assert row.selected_method==cfg["methods"][selected]
        assert abs(row.train_R-training[selected])<1e-12
        assert abs(row.held_R-matrix.loc[row.held_team,row.selected_method])<1e-12
    distances=read(OUT/"全44CとBG参照_最近k距離の別実装照合.csv")
    perms=read(OUT/"行順変更.csv")
    reproduction=read(OUT/"初回3方式との全実Cとnull採点一致.csv")
    assert len(distances)==44*3*3 and distances.maximum_distance_difference.eq(0).all()
    assert len(perms)==44*2*7*5 and perms.changed.eq(0).all()
    assert len(reproduction)==3*2*(21+20*20) and reproduction.maximum_score_difference.lt(1e-12).all()
    # 元のTrueは、初回splitのpool/refの非重複。full参照ではnullが含まれる。
    # この補助表だけを明確な列名に整え、数学的な結果や標本索引は変えない。
    sampling_path=OUT/"初回条件の同性別null抽出監査.csv"
    sampling=read(sampling_path)
    if "pool_ref_disjoint" in sampling:
        assert sampling.pool_ref_disjoint.all()
        sampling=sampling.rename(columns={"pool_ref_disjoint":"split_pool_ref_disjoint"})
    sampling["null_sample_disjoint_from_current_reference"]=sampling.context.eq("split_independent")
    sampling["null_sample_contained_in_current_reference"]=sampling.context.eq("full_reference_overlap")
    assert len(sampling)==20*20*2 and sampling.same_sex_counts.all() and sampling.n.eq(1049).all()
    assert sampling.groupby(["team","repeat"]).original_BG_index_sha256.nunique().eq(1).all()
    sampling.to_csv(sampling_path,index=False)
    background=pd.DataFrame(background_changes)
    background.to_csv(OUT/"全BGとsplit参照の選択変化_対象別性別.csv",index=False)
    background_summary=[]
    distance_rates=[]
    attributes=read(OUT/"第k近傍距離と重複の対象別属性別.csv")
    for scope,phase,excluded in [("prelim20","prelim",[5]),("processed18","prelim",[1,2,5]),("main23","main",[])]:
        use=background[background.phase.eq(phase)&~background.team.isin(excluded)]
        for (sex,method),one in use.groupby(["sex","method"]):
            background_summary.append(dict(scope=scope,sex=sex,method=method,targets=len(one),
                changed_mean=float(one.changed.mean()),changed_min=int(one.changed.min()),changed_max=int(one.changed.max()),
                jaccard_mean=float(one.jaccard.mean()),jaccard_min=float(one.jaccard.min()),jaccard_max=float(one.jaccard.max())))
        use=attributes[attributes.phase.eq(phase)&~attributes.team.isin(excluded)]
        for (context,k),one in use.groupby(["context","k"]):
            distance_rates.append(dict(scope=scope,context=context,k=k,q_attribute_combinations=int(one.q_n.sum()),
                C_distance_le_1e8=int(one.C_distance_le_1e8.sum()),BG_distance_le_1e8=int(one.BG_distance_le_1e8.sum()),
                C_near_zero_share=float(one.C_distance_le_1e8.sum()/one.q_n.sum()),
                BG_near_zero_share=float(one.BG_distance_le_1e8.sum()/one.q_n.sum())))
    pd.DataFrame(background_summary).to_csv(OUT/"全BGとsplit参照の選択変化_平均と範囲.csv",index=False)
    pd.DataFrame(distance_rates).to_csv(OUT/"第k近傍の距離一致率_集計.csv",index=False)
    shared=[]
    for path in OUT.glob("*.csv"):
        columns=pd.read_csv(path,nrows=0).columns
        assert "challenge_row_id" not in columns and "challenge_row_ids" not in columns
        shared.append(path.name)
    forbidden=[str(p) for p in HERE.rglob("*") if p.name in ["F_aia.csv","F_mia.csv","token.txt"] or p.suffix==".zip"]
    assert not forbidden
    result=dict(code_sha256=sha(__file__),prediction_code_sha256=pred["code_sha256"],
        checked_prediction_input_hashes=len(pred["input_sha256"]),checked_import_hashes=len(pred["imports_sha256"]),
        checked_truth_hashes=len(score["public_prelim_truth_sha256"]),checked_prediction_masks=masks,
        checked_score_rows=len(table),checked_average_rows=len(averages),checked_LOTO_rows=len(lt),
        checked_distance_cases=len(distances),checked_permutation_cases=len(perms),matched_original_score_cases=len(reproduction),
        checked_null_sample_context_cases=len(sampling),metadata_normalized=True,mathematical_results_changed=False,
        added_fixed_background_diagnostic_rows=len(background),added_distance_rate_rows=len(distance_rates),
        null_sample_indices_changed=False,final_sampling_table_sha256=sha(sampling_path),
        verification_input_sha256=HASHES,sharing_predictions=False,share_csv_files=shared,
        in_first39_frozen_pool=False,posthoc_inspiration=True,forbidden_submission_files=forbidden)
    (HERE/"検証記録.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print({key:value for key,value in result.items() if key not in ["verification_input_sha256","share_csv_files"]})


if __name__=="__main__":
    run()
