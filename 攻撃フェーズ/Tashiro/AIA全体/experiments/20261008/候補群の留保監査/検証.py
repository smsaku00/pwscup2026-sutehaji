"""保存集計だけでhash・基準・留保選択・oracle・記述bootstrapを確認する。"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):return pd.read_csv(HERE/name)
def near(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)))<1e-12

def main():
    frozen=json.loads((HERE/"候補プール固定.json").read_text())
    assert sha(HERE/"分析.py")==frozen["analysis_code_sha256"]
    assert sha(HERE/"監査条件.json")==frozen["audit_protocol_sha256"]
    for s in frozen["sources"]:assert sha(Path(s["absolute_path"]))==s["sha256"]
    for name,value in frozen["snapshot_sha256"].items():assert sha(HERE/name)==value
    record=json.loads((HERE/"監査実行記録.json").read_text())
    for name,value in record["output_sha256"].items():assert sha(HERE/name)==value
    catalog=read("候補の役割一覧.csv");main=read("共通全BG_全候補対象別.csv")
    pool=frozen["primary_pool"];assert len(pool)==39 and pool[0]=="anchor:P01"
    assert set(pool)==set(catalog[catalog.eligible_primary].candidate_id)
    assert set(catalog[catalog.eligible_primary].role)=={"anchor_baseline","new_fixed_attack"}
    assert not any("posthoc" in s for s in pool)
    assert not main.duplicated(["team","candidate_id"]).any()
    for method in pool:assert main[main.candidate_id.eq(method)].team.nunique()==21
    near(main.r_all,np.clip((main.hit_m-main.hit_c)/np.maximum(163-main.hit_c,1e-9),0,1))
    near(main.signed_hit_gap,main.hit_m-main.hit_c)
    near(main.r_all,main.unclipped_R_all+main.zero_clip_uplift)
    checks=read("共通基準_的中数とR一致.csv");assert len(checks)==9 and checks.hit_counts_match.all()
    assert checks.targets.eq(21).all() and checks.maximum_score_error.lt(1e-12).all()
    anchor=main[main.candidate_id.eq(pool[0])].set_index("team").sort_index()
    for name in catalog[catalog.role.eq("baseline_duplicate")].candidate_id:
        duplicate=main[main.candidate_id.eq(name)].set_index("team").sort_index()
        near(duplicate[["hit_m","hit_c","r_all"]],anchor[["hit_m","hit_c","r_all"]])
    loto=read("集計/global_LOTO_対象別.csv");oracle=read("集計/targetwise_oracle_対象別.csv")
    bootstrap=read("集計/paired_bootstrap_記述幅.csv");selection_checks=0;oracle_checks=0;bootstrap_checks=0
    for group,exclude,offset in [("prelim20",[5],0),("processed18",[1,2,5],1)]:
        d=main[~main.team.isin(exclude)];p=d.pivot(index="team",columns="candidate_id",values="r_all").sort_index()
        a=p[pool].to_numpy();baseline=a[:,0];n=len(a);assert n==20-offset*2
        actual=loto[loto.group.eq(group)].set_index("team").loc[p.index]
        selected=[]
        for row,team in enumerate(p.index):
            training=(a.sum(axis=0)-a[row])/(n-1)
            chosen=int(np.flatnonzero(training>=training.max()-1e-12)[0])
            assert actual.loc[team,"candidate_id"]==pool[chosen]
            near([actual.loc[team,"r_all"],actual.loc[team,"training_R"]],[a[row,chosen],training[chosen]])
            selected.append(a[row,chosen]);selection_checks+=1
        series={name:a[:,i] for i,name in enumerate(pool)};series["procedure:global_LOTO"]=np.array(selected)
        for kind,names in [("primary_fixed_attack",pool),("all_saved_roles",frozen["all_saved_roles_pool"])]:
            matrix=p[names].to_numpy();observed=oracle[oracle.group.eq(group)&oracle.pool.eq(kind)].set_index("team").loc[p.index]
            assert observed.truth_seen_to_choose.all() and not observed.deployable_performance.any()
            for row,team in enumerate(p.index):
                chosen=int(np.flatnonzero(matrix[row]>=matrix[row].max()-1e-12)[0])
                assert observed.loc[team,"candidate_id"]==names[chosen]
                near(observed.loc[team,"r_all"],matrix[row,chosen]);oracle_checks+=1
            if kind=="primary_fixed_attack":series["diagnostic:targetwise_oracle"]=matrix.max(axis=1)
        rng=np.random.default_rng(2026100821+offset);draws=rng.integers(0,n,(10000,n))
        observed=bootstrap[bootstrap.group.eq(group)].set_index("candidate_or_procedure")
        assert len(observed)==len(series)
        for name,value in series.items():
            delta=value-baseline;samples=delta[draws].mean(axis=1)
            near(observed.loc[name,["mean_delta_R","descriptive_q025","descriptive_q975"]].to_numpy(float),
                [delta.mean(),np.quantile(samples,.025),np.quantile(samples,.975)]);bootstrap_checks+=1
        near(d.groupby("candidate_id").r_all.mean().sort_index(),
            d.sample(frac=1,random_state=17).groupby("candidate_id").r_all.mean().sort_index())
    near(anchor[~anchor.index.isin([1,2,5])].r_all.mean(),.1117359225973019)
    assert bootstrap.paired_resamples.eq(10000).all() and not bootstrap.independent_generalization_CI.any()
    assert not bootstrap.p_value_computed.any() and not bootstrap.method_choice_refitted.any()
    for folder,protocol,run,code in [("既存分岐へのpool差替","追加監査条件固定.json","追加監査実行記録.json","凍結候補での既存分岐監査.py"),
        ("後付け対照の条件監査","条件と入力hash.json","実行記録.json","後付け対照の別条件記録.py"),
        ("留保選択の的中差分解","条件と入力hash.json","実行記録.json","留保選択の的中差分解.py")]:
        p=HERE/folder;spec=json.loads((p/protocol).read_text());executed=json.loads((p/run).read_text())
        for name,value in spec["input_sha256"].items():assert sha(Path(name))==value
        assert sha(p/protocol)==executed["protocol_sha256"]
        for name,value in executed["output_sha256"].items():assert sha(p/name)==value
        assert sha(HERE/code)==executed["code_sha256"]
    leak=read("既存分岐へのpool差替/外側正解の遮断確認.csv")
    assert len(leak)==4 and leak.choices_changed.eq(0).all() and leak.policy_changed.eq(0).all()
    repro=read("既存分岐へのpool差替/旧8候補_同一規則の再現.csv")
    assert len(repro)==4 and repro.old8_rule_reproduced.all() and repro.maximum_score_error.lt(1e-12).all()
    decomposed=read("留保選択の的中差分解/選択済み方式の対象別分解.csv")
    assert len(decomposed)==114
    near(decomposed.delta_r_all,decomposed.delta_unclipped_R_all+decomposed.delta_zero_clip_uplift)
    near(decomposed.delta_signed_hit_gap,decomposed.delta_hit_m-decomposed.delta_hit_c)
    report=dict(status="PASS",source_hashes=len(frozen["sources"]),baseline_aliases=9,primary_candidates=39,
        global_LOTO_rows_checked=selection_checks,oracle_rows_checked=oracle_checks,
        bootstrap_rows_recomputed=bootstrap_checks,bootstrap_repeats=10000,
        old8_rule_replications=4,heldout_score_interventions=4,selected_count_decompositions=114,team_row_order_invariance=True,
        verifier_sha256=sha(Path(__file__)),individual_predictions_read=False,main_truth_read=False,B_self_read=False)
    (HERE/"検証結果.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))

if __name__=="__main__":main()
