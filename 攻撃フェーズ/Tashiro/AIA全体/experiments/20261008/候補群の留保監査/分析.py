"""保存された固定方式を条件別に集め、候補プール固定後に選択の楽観を監査する。

原本、個票、B_self、本戦truthは読まない。方式追加・提出物・GitHub掲載を行わない。
inventory -> freeze -> aggregate の順。freeze後の原本/コード変更はエラーにする。
"""
from __future__ import annotations
import argparse,hashlib,json,platform,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
OLD=BASE.parent/"20261007_AIA全体向上の分析"
SEED=2026100821
BOOTSTRAPS=10000
METRICS=["r_all","r_rare","r_max","p_m","p_c","hit_m","hit_c","hit_m_rare","hit_c_rare"]
DERIVED=["signed_hit_gap","unclipped_R_all","zero_clip_uplift"]
ANCHOR="anchor:P01"

# ファミリー/条件の区別は統一した選択結果を見る前に、この表で定義する。
SPECS=[
 dict(family="順位尺度と分布補正",id="rank",csv="順位尺度と分布補正/結果/対象別.csv",baseline="baseline_partial2",
      code="順位尺度と分布補正/分析.py",protocol="順位尺度と分布補正/結果/固定比較条件.json",filter={},primary_family=True),
 dict(family="条件付き時間推定",id="time",csv="条件付き時間推定/集計/予備戦_対象別.csv",baseline=None,
      code="条件付き時間推定/分析.py",protocol="条件付き時間推定/事前固定候補.json",filter={},primary_family=True),
 dict(family="公開BG一致行の処理",id="public_bg",csv="加工特徴と方式選択/集計/公開BG処理_予備戦対象別.csv",baseline="baseline",
      code="加工特徴と方式選択/公開BG一致行の比較.py",protocol="加工特徴と方式選択/公開BG比較_事前固定仕様.json",filter={},primary_family=True),
 dict(family="人物別較正",id="person",csv="背景対照での選別較正/集計/予備戦_対象別とnull反復.csv",baseline="baseline",
      code="背景対照での選別較正/人物較正.py",protocol="背景対照での選別較正/固定条件.json",
      filter={"context":"full_reference_overlap","source":"real_C"},primary_family=False),
 dict(family="人物別較正_厳密計算",id="person_exact",csv="背景対照での選別較正/超幾何による同じ帰無分布の厳密計算/集計/厳密版_対象別とnull反復.csv",baseline=None,
      code="背景対照での選別較正/超幾何による同じ帰無分布の厳密計算/厳密較正.py",
      protocol="背景対照での選別較正/超幾何による同じ帰無分布の厳密計算/追加固定条件.json",
      filter={"source":"real_C"},primary_family=False),
 dict(family="他Cとの差による選別",id="other_C",csv="他Cとの差による選別/集計/予備戦_対象別.csv",baseline="baseline",
      code="他Cとの差による選別/分析.py",protocol="他Cとの差による選別/事前固定候補.json",filter={},primary_family=True),
 dict(family="C抽出による選別の安定性",id="subsample",csv="C抽出による選別の安定性/集計/予備戦_対象別.csv",baseline="baseline",
      code="C抽出による選別の安定性/分析.py",protocol="C抽出による選別の安定性/事前固定候補.json",filter={},primary_family=True),
 dict(family="属性対ごとの証拠統合",id="pair",csv="属性対ごとの証拠統合/集計/予備戦_対象別.csv",baseline="baseline_partial2",
      code="属性対ごとの証拠統合/分析.py",protocol="属性対ごとの証拠統合/事前固定仕様.json",filter={},primary_family=True),
 dict(family="C行の背景対比",id="row",csv="C行の背景対比/結果/対象別と背景反復.csv",baseline="baseline",
      code="C行の背景対比/分析.py",protocol="C行の背景対比/結果/固定条件.json",
      filter={"source":"real_C_full"},primary_family=True),
 dict(family="属性値集合による選別",id="column",csv="属性値集合による選別/集計/予備戦_対象別と20反復null.csv",baseline="baseline_partial2_matching",
      code="属性値集合による選別/分析.py",protocol="属性値集合による選別/固定条件.json",
      filter={"context":"full_reference_overlap","source":"real_C"},primary_family=False),
]

SIDE=[
 ("rank","順位尺度と分布補正/結果/独立背景対照_全反復.csv",{},"rank_independent_split"),
 ("public_bg","加工特徴と方式選択/集計/公開BG処理_独立参照の実C.csv",{},"publicBG_split_actual"),
 ("public_bg","加工特徴と方式選択/集計/公開BG処理_背景対照全反復.csv",{},"publicBG_split_null"),
 ("person","背景対照での選別較正/集計/予備戦_対象別とnull反復.csv",{"context":"split_independent"},"person_independent_split"),
 ("person","背景対照での選別較正/集計/予備戦_対象別とnull反復.csv",{"context":"full_reference_overlap","source":"BG_null_C"},"person_overlapping_reference_null"),
 ("person_exact","背景対照での選別較正/超幾何による同じ帰無分布の厳密計算/集計/厳密版_対象別とnull反復.csv",{"source":"BG_null_C"},"person_exact_null_same_pool"),
 ("other_C","他Cとの差による選別/集計/独立背景対照_全反復.csv",{},"otherC_independent_split"),
 ("subsample","C抽出による選別の安定性/集計/独立背景対照_全反復.csv",{},"subsample_independent_split"),
 ("pair","属性対ごとの証拠統合/集計/独立背景_実Cと全反復.csv",{},"pair_independent_split"),
 ("row","C行の背景対比/結果/対象別と背景反復.csv",{"source":["real_C_split","background_C"]},"row_independent_split"),
 ("column","属性値集合による選別/集計/予備戦_対象別と20反復null.csv",{"context":"split_independent"},"column_independent_split"),
 ("column","属性値集合による選別/集計/予備戦_対象別と20反復null.csv",{"context":"full_reference_overlap","source":"BG_null_C"},"column_overlapping_reference_null"),
 ("time","条件付き時間推定/集計/時間関係の寄与_全反復.csv",{"source":"time_shuffle"},"time_shuffle_same_sex"),
]

POLICY=dict(main_metric="r_all",main_reference="whole phase BG, same target/query/truth, verify saved baseline hit counts",
    score_stage="official raw R before phi; final phi unknown",secondary_metrics=["r_rare","r_max"],
    main_scope_prelim20="exclude5",main_scope_processed18="exclude1,2,5",score_tolerance=1e-12,
    pool="anchor plus all newly fixed attack candidates under common fullBG; no baseline duplicate/control/historical comparator",
    secondary_oracle_pool="all saved common-fullBG roles including controls and historical comparators; descriptive only",
    tie="anchor priority, then registry family order, then method lexical order",tie_tolerance=1e-12,
    bootstrap_repeats=BOOTSTRAPS,bootstrap_seed=SEED,bootstrap_unit="paired target",
    bootstrap_interpretation="descriptive 2.5/97.5 percentiles, not p-values or independent generalization CI; do not refit method choice",
    reused_public_prelim_truth=True,candidate_design_independent_of_prior_prelim_results=False,
    pool_frozen_after_family_outcomes=True,freeze_before_unified_global_choice=True,
    posthoc_diagnostics_not_in_pool=True,final_global_summary_written=False,GitHub_published=False,
    B_self_read=False,individual_predictions_read=False,main_truth_read=False,submission_created=False)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def now():return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def filtered(df,rules):
    mask=np.ones(len(df),bool)
    for key,value in rules.items():
        mask &= df[key].isin(value).to_numpy() if isinstance(value,list) else df[key].eq(value).to_numpy()
    return df[mask].copy()


def validate_score(df):
    assert set(METRICS).issubset(df.columns)
    for k in ["hit_m","hit_c"]:assert df[k].between(0,163).all() and (df[k]%1==0).all()
    for k in ["hit_m_rare","hit_c_rare"]:assert df[k].between(0,13).all() and (df[k]%1==0).all()
    assert np.max(np.abs(df.p_m-df.hit_m/163))<1e-12
    assert np.max(np.abs(df.p_c-df.hit_c/163))<1e-12
    r=np.clip((df.p_m-df.p_c)/np.maximum(1-df.p_c,1e-9),0,1)
    assert np.max(np.abs(r-df.r_all))<1e-12
    pm,pc=df.hit_m_rare/13,df.hit_c_rare/13
    se=np.sqrt(pm*(1-pm)/13+pc*(1-pc)/13)
    rare=np.clip((pm-pc-1.64*se)/np.maximum(1-pc,1e-9),0,1)
    assert np.max(np.abs(rare-df.r_rare))<1e-12
    assert np.max(np.abs(np.maximum(r,rare)-df.r_max))<1e-12


def role(spec,method):
    if method==spec["baseline"]:return "baseline_duplicate",False
    if spec["id"]=="time":
        if method=="BG/raw_k40_window" or method=="constant9.5":return "C_free_control",False
        if method=="BG/box_k40_window":return "C_grid_shape_control",False
    return "new_fixed_attack",True


def collect(complete):
    files={}
    def register(path,kind):
        path=path.resolve()
        assert path.exists(),str(path)
        key=str(path.relative_to(BASE)) if path.is_relative_to(BASE) else str(path)
        if key not in files:files[key]=dict(path=key,absolute_path=str(path),kind=kind,sha256=sha(path),bytes=path.stat().st_size)
        return key
    def read_csv(path):
        register(path,"aggregate_csv");return pd.read_csv(path)
    old=read_csv(OLD/"集計/同点修正_対象別.csv")
    base=old[old.method.eq("stable_matching/sex_quota/long75")].sort_values("team").copy()
    validate_score(base);teams=base.team.tolist();assert len(teams)==21
    main=base[["team"]+METRICS].copy()
    main["family"]="共通基準";main["family_id"]="anchor";main["method"]="P01";main["candidate_id"]=ANCHOR
    main["canonical_candidate_id"]=ANCHOR;main["role"]="anchor_baseline";main["eligible_primary"]=True
    main["reference_context"]="common_full_phaseBG";main["family_primary_condition"]=True
    main["source_csv"]=str(OLD/"集計/同点修正_対象別.csv");main["pool_order"]=0
    allmain=[main];allside=[];checks=[];pending=[]
    catalog=[dict(candidate_id=ANCHOR,family="共通基準",family_id="anchor",method="P01",role="anchor_baseline",
        eligible_primary=True,canonical_candidate_id=ANCHOR,source_csv=str(OLD/"集計/同点修正_対象別.csv"),
        primary_family_condition=True,pool_order=0,targets=21)]
    for order,spec in enumerate(SPECS,1):
        csv=BASE/spec["csv"]
        if not csv.exists():
            pending.append(spec["family"])
            if complete:raise FileNotFoundError(f"未完了family: {spec['family']}")
            continue
        register(BASE/spec["code"],"prediction_code");register(BASE/spec["protocol"],"family_protocol")
        df=filtered(read_csv(csv),spec["filter"])
        validate_score(df)
        assert not df.duplicated(["team","method"]).any()
        for method,g in df.groupby("method",sort=True):
            assert sorted(g.team.tolist())==teams,f"21対象不一致: {spec['family']}:{method}"
            r,eligible=role(spec,method)
            candidate=f"{spec['id']}:{method}"
            d=g[["team"]+METRICS].copy()
            for key,value in dict(family=spec["family"],family_id=spec["id"],method=method,candidate_id=candidate,
                canonical_candidate_id=ANCHOR if r=="baseline_duplicate" else candidate,role=r,eligible_primary=eligible,
                reference_context="common_full_phaseBG",family_primary_condition=spec["primary_family"],
                source_csv=spec["csv"],pool_order=order).items():d[key]=value
            if r=="baseline_duplicate":
                ordered=g.sort_values("team")
                errors={m:float(np.max(np.abs(ordered[m].to_numpy()-base[m].to_numpy()))) for m in METRICS}
                assert max(errors.values())<1e-12,f"baseline不一致:{spec['family']}"
                checks.append(dict(family=spec["family"],source=spec["csv"],method=method,targets=21,
                    hit_counts_match=True,maximum_score_error=max(errors.values())))
            catalog.append(dict(candidate_id=candidate,family=spec["family"],family_id=spec["id"],method=method,
                role=r,eligible_primary=eligible,canonical_candidate_id=ANCHOR if r=="baseline_duplicate" else candidate,
                source_csv=spec["csv"],primary_family_condition=spec["primary_family"],pool_order=order,targets=21))
            allmain.append(d)
    # 保存済み比較方式を消さず、現在の新方式poolとは別役割で保持する。
    historical="加工特徴と方式選択/集計/固定方式の予備戦得点.csv"
    h=read_csv(BASE/historical);validate_score(h)
    register(BASE/"加工特徴と方式選択/分析.py","historical_comparator_code")
    register(BASE/"加工特徴と方式選択/事前固定仕様.json","historical_comparator_protocol")
    for method,g in h.groupby("method",sort=True):
        r="baseline_duplicate" if method=="stable_matching/sex_quota/long75" else "historical_comparator"
        candidate="historical:"+method
        d=g[["team"]+METRICS].copy()
        for key,value in dict(family="過去固定方式の比較",family_id="historical",method=method,candidate_id=candidate,
            canonical_candidate_id=ANCHOR if r=="baseline_duplicate" else candidate,role=r,eligible_primary=False,
            reference_context="common_full_phaseBG",family_primary_condition=False,source_csv=historical,pool_order=50).items():d[key]=value
        assert sorted(g.team.tolist())==teams
        if r=="baseline_duplicate":
            assert np.max(np.abs(g.sort_values("team")[METRICS].to_numpy()-base[METRICS].to_numpy()))<1e-12
            checks.append(dict(family="過去固定方式の比較",source=historical,method=method,targets=21,
                hit_counts_match=True,maximum_score_error=float(np.max(np.abs(g.sort_values("team")[METRICS].to_numpy()-base[METRICS].to_numpy())))))
        catalog.append(dict(candidate_id=candidate,family="過去固定方式の比較",family_id="historical",method=method,
            role=r,eligible_primary=False,canonical_candidate_id=ANCHOR if r=="baseline_duplicate" else candidate,
            source_csv=historical,primary_family_condition=False,pool_order=50,targets=21))
        allmain.append(d)
    # 時間寄与の1対照は初期6候補結果閲覧後の対照。選択poolに入れない。
    aux="条件付き時間推定/集計/時間関係の寄与_全反復.csv"
    a=filtered(read_csv(BASE/aux),{"source":"fixed95","method":"gate75_long95"});validate_score(a)
    register(BASE/"条件付き時間推定/推定時間の寄与.py","mechanism_control_code")
    register(BASE/"条件付き時間推定/寄与検証_固定手順.json","mechanism_control_protocol")
    d=a[["team"]+METRICS].copy()
    candidate="time_control:gate75_long95"
    for key,value in dict(family="時間関係の寄与対照",family_id="time_control",method="gate75_long95",candidate_id=candidate,
        canonical_candidate_id=candidate,role="posthoc_mechanism_control",eligible_primary=False,
        reference_context="common_full_phaseBG",family_primary_condition=False,source_csv=aux,pool_order=51).items():d[key]=value
    allmain.append(d)
    catalog.append(dict(candidate_id=candidate,family="時間関係の寄与対照",family_id="time_control",method="gate75_long95",
        role="posthoc_mechanism_control",eligible_primary=False,canonical_candidate_id=candidate,source_csv=aux,
        primary_family_condition=False,pool_order=51,targets=len(a)))
    for family,csv,rules,condition in SIDE:
        if not (BASE/csv).exists():
            if complete:raise FileNotFoundError(csv)
            continue
        d=filtered(read_csv(BASE/csv),rules);validate_score(d)
        d=d[[v for v in ["team","context","source","repeat","method"] if v in d]+METRICS].copy()
        d["family_id"]=family;d["condition_id"]=condition;d["source_csv"]=csv
        if "context" not in d:d["context"]=condition
        if "source" not in d:d["source"]="real_C_split" if "actual" in condition else "background_C"
        if "repeat" not in d:d["repeat"]=-1
        d["not_for_primary_global_selection"]=True;allside.append(d)
    for name in ["分析.py","対応競合の頑健性.py","同点に依存しない対応.py","対応競合の分析.py"]:
        register(OLD/name,"anchor_reference_code")
    main=pd.concat(allmain,ignore_index=True);side=pd.concat(allside,ignore_index=True)
    for d in [main,side]:
        d["signed_hit_gap"]=d.hit_m-d.hit_c
        d["unclipped_R_all"]=(d.p_m-d.p_c)/np.maximum(1-d.p_c,1e-9)
        d["zero_clip_uplift"]=d.r_all-d.unclipped_R_all
    return main,side,pd.DataFrame(catalog),pd.DataFrame(checks),files,pending


def inventory():
    HERE.mkdir(parents=True,exist_ok=True)
    main,side,catalog,checks,files,pending=collect(False)
    checks.to_csv(HERE/"基準一致_収集準備.csv",index=False)
    dump(HERE/"収集準備.json",dict(time=now(),families_collected=catalog.family_id.unique().tolist(),pending=pending,
        main_candidate_records=len(catalog),baseline_checks=len(checks),source_files=list(files.values()),
        global_selection_computed=False,candidate_pool_frozen=False))
    print(json.dumps(dict(families=catalog.family_id.unique().tolist(),pending=pending,baseline_checks=len(checks)),ensure_ascii=False))


def freeze():
    HERE.mkdir(parents=True,exist_ok=True)
    assert not (HERE/"候補プール固定.json").exists(),"一度固定した候補プールは上書きしない"
    main,side,catalog,checks,files,pending=collect(True);assert not pending
    dump(HERE/"監査条件.json",dict(policy=POLICY,specs=SPECS,side_conditions=SIDE))
    catalog.to_csv(HERE/"候補の役割一覧.csv",index=False)
    main.to_csv(HERE/"共通全BG_全候補対象別.csv",index=False)
    side.to_csv(HERE/"別条件_対象別と反復.csv",index=False)
    checks.to_csv(HERE/"共通基準_的中数とR一致.csv",index=False)
    ordered=catalog.sort_values(["pool_order","candidate_id"])
    primary=ordered[ordered.eligible_primary].candidate_id.tolist()
    assert primary[0]==ANCHOR and len(primary)==len(set(primary))
    frozen=dict(frozen_at=now(),policy=POLICY,primary_pool=primary,all_saved_roles_pool=ordered[~ordered.role.eq("baseline_duplicate")].candidate_id.tolist(),
        sources=list(files.values()),analysis_code_sha256=sha(Path(__file__)),audit_protocol_sha256=sha(HERE/"監査条件.json"),
        snapshot_sha256={name:sha(HERE/name) for name in ["候補の役割一覧.csv","共通全BG_全候補対象別.csv","別条件_対象別と反復.csv","共通基準_的中数とR一致.csv"]},
        common_teams=sorted(main[main.candidate_id.eq(ANCHOR)].team.tolist()),
        truth_design_independent=False,pool_frozen_before_unified_selection=True)
    dump(HERE/"候補プール固定.json",frozen)
    print(json.dumps(dict(frozen_at=frozen["frozen_at"],primary_candidates=len(primary),saved_roles=len(ordered),source_files=len(files)),ensure_ascii=False))


def verified_frozen():
    f=json.loads((HERE/"候補プール固定.json").read_text())
    assert sha(Path(__file__))==f["analysis_code_sha256"]
    assert sha(HERE/"監査条件.json")==f["audit_protocol_sha256"]
    for entry in f["sources"]:assert sha(Path(entry["absolute_path"]))==entry["sha256"],entry["path"]
    for name,expected in f["snapshot_sha256"].items():assert sha(HERE/name)==expected,name
    return f


def choose(values,order):
    maximum=float(np.max(values));near=np.flatnonzero(values>=maximum-1e-12)
    return int(near[0]) # poolはanchorを先頭に固定済み。


def aggregate():
    frozen=verified_frozen();main=pd.read_csv(HERE/"共通全BG_全候補対象別.csv")
    out=HERE/"集計";out.mkdir(exist_ok=True)
    summaries,loto,oracle,bs,family_results,side_results,gap_results,long_records=[],[],[],[],[],[],[],[]
    for group,excluded in [("prelim20",[5]),("processed18",[1,2,5])]:
        d=main[~main.team.isin(excluded)].copy();teams=sorted(d[d.candidate_id.eq(ANCHOR)].team.unique())
        tagged=d.copy();tagged.insert(0,"group",group);long_records.append(tagged)
        anchor=d[d.candidate_id.eq(ANCHOR)].set_index("team").reindex(teams)
        means=d.groupby(["family","candidate_id","method","role","eligible_primary"])[METRICS+DERIVED].mean().reset_index()
        means.insert(0,"group",group);means["targets"]=len(teams);means["delta_R_vs_anchor"]=means.r_all-anchor.r_all.mean();summaries.append(means)
        pivot=d.pivot(index="team",columns="candidate_id",values="r_all").reindex(index=teams)
        primary=frozen["primary_pool"];assert pivot[primary].notna().all().all()
        matrix=pivot[primary].to_numpy();base=matrix[:,0]
        for candidate,g in d.groupby("candidate_id",sort=True):
            aligned=g.set_index("team").reindex(teams);assert aligned.hit_m.notna().all()
            gap_results.append(dict(group=group,candidate_id=candidate,targets=len(teams),rows_per_class=163*len(teams),
                total_hit_m=int(aligned.hit_m.sum()),total_hit_c=int(aligned.hit_c.sum()),
                delta_hit_m_vs_anchor=int((aligned.hit_m-anchor.hit_m).sum()),
                delta_hit_c_vs_anchor=int((aligned.hit_c-anchor.hit_c).sum()),
                total_signed_hit_gap=int(aligned.signed_hit_gap.sum()),
                delta_signed_hit_gap_vs_anchor=int((aligned.signed_hit_gap-anchor.signed_hit_gap).sum()),
                mean_R_all=float(aligned.r_all.mean()),mean_unclipped_R=float(aligned.unclipped_R_all.mean()),
                mean_zero_clip_uplift=float(aligned.zero_clip_uplift.mean()),
                delta_R_vs_anchor=float((aligned.r_all-anchor.r_all).mean()),
                delta_unclipped_R_vs_anchor=float((aligned.unclipped_R_all-anchor.unclipped_R_all).mean()),
                delta_zero_clip_uplift_vs_anchor=float((aligned.zero_clip_uplift-anchor.zero_clip_uplift).mean()),
                improved_targets=int((aligned.r_all>anchor.r_all+1e-12).sum()),
                worsened_targets=int((aligned.r_all<anchor.r_all-1e-12).sum()),
                equal_targets=int((np.abs(aligned.r_all-anchor.r_all)<=1e-12).sum())))
        selected_values=[];oracle_values=[]
        for row,team in enumerate(teams):
            training=np.delete(matrix,row,axis=0).mean(0)
            chosen=choose(training,primary);oracle_chosen=choose(matrix[row],primary)
            selected_values.append(matrix[row,chosen]);oracle_values.append(matrix[row,oracle_chosen])
            loto.append(dict(group=group,team=team,candidate_id=primary[chosen],training_R=training[chosen],
                r_all=matrix[row,chosen],baseline_R=base[row],delta=matrix[row,chosen]-base[row]))
            oracle.append(dict(group=group,pool="primary_fixed_attack",team=team,candidate_id=primary[oracle_chosen],
                r_all=matrix[row,oracle_chosen],baseline_R=base[row],delta=matrix[row,oracle_chosen]-base[row],
                truth_seen_to_choose=True,deployable_performance=False))
        all_pool=[v for v in frozen["all_saved_roles_pool"] if v in pivot and pivot[v].notna().all()]
        for row,team in enumerate(teams):
            pos=choose(pivot[all_pool].iloc[row].to_numpy(),all_pool);value=float(pivot.loc[team,all_pool[pos]])
            oracle.append(dict(group=group,pool="all_saved_roles",team=team,candidate_id=all_pool[pos],
                r_all=value,baseline_R=base[row],delta=value-base[row],truth_seen_to_choose=True,deployable_performance=False))
        # 全固定候補の差と、選択後に保存したLOTO/oracleの差に同じpaired抽出を使う。
        rng=np.random.default_rng(SEED+(0 if group=="prelim20" else 1))
        draws=rng.integers(0,len(teams),size=(BOOTSTRAPS,len(teams)))
        series={v:matrix[:,i] for i,v in enumerate(primary)}
        series["procedure:global_LOTO"]=np.array(selected_values)
        series["diagnostic:targetwise_oracle"]=np.array(oracle_values)
        for name,values in series.items():
            delta=values-base;samples=delta[draws].mean(1)
            bs.append(dict(group=group,candidate_or_procedure=name,mean_delta_R=float(delta.mean()),
                descriptive_q025=float(np.quantile(samples,.025)),descriptive_q975=float(np.quantile(samples,.975)),
                targets=len(teams),paired_resamples=BOOTSTRAPS,method_choice_refitted=False,
                independent_generalization_CI=False,p_value_computed=False))
        for family,g in d[d.eligible_primary].groupby("family",sort=True):
            ids=[v for v in primary if v in set(g.candidate_id)]
            m=pivot[ids].mean(0).to_numpy();pos=choose(m,ids)
            family_results.append(dict(group=group,family=family,best_saved_candidate=ids[pos],
                mean_R=float(m[pos]),baseline_mean_R=float(base.mean()),delta=float(m[pos]-base.mean()),
                selected_using_same_benchmark=True,new_independent_validation=False))
    pd.concat(summaries).to_csv(out/"共通全BG_20対象と18対象の全候補平均.csv",index=False)
    pd.concat(long_records).to_csv(out/"共通全BG_20対象と18対象の対象別longform.csv",index=False)
    pd.DataFrame(gap_results).to_csv(out/"的中差と0切り上げ_全候補.csv",index=False)
    lt=pd.DataFrame(loto);oc=pd.DataFrame(oracle)
    lt.to_csv(out/"global_LOTO_対象別.csv",index=False);lt.groupby("group")[["r_all","baseline_R","delta"]].mean().to_csv(out/"global_LOTO_平均.csv")
    oc.to_csv(out/"targetwise_oracle_対象別.csv",index=False);oc.groupby(["group","pool"])[["r_all","baseline_R","delta"]].mean().to_csv(out/"targetwise_oracle_平均.csv")
    pd.DataFrame(bs).to_csv(out/"paired_bootstrap_記述幅.csv",index=False)
    pd.DataFrame(family_results).to_csv(out/"family内最大値_同じベンチでの選択.csv",index=False)
    side=pd.read_csv(HERE/"別条件_対象別と反復.csv")
    for group,excluded in [("prelim20",[5]),("processed18",[1,2,5])]:
        sd=side[~side.team.isin(excluded)]
        avg=sd.groupby(["family_id","condition_id","context","source","method"])[METRICS+DERIVED].mean().reset_index()
        avg.insert(0,"group",group);side_results.append(avg)
    pd.concat(side_results).to_csv(out/"別条件_平均_主表へ混ぜない.csv",index=False)
    outputs={str(p.relative_to(HERE)):sha(p) for p in out.glob("*.csv")}
    dump(HERE/"監査実行記録.json",dict(time=now(),python=platform.python_version(),executable=sys.executable,
        frozen_pool_sha256=sha(HERE/"候補プール固定.json"),code_sha256=sha(Path(__file__)),output_sha256=outputs,
        primary_candidates=len(frozen["primary_pool"]),scoring_rows=len(main),side_rows=len(side),
        bootstrap_repeats=BOOTSTRAPS,selection_optimism_diagnosed=True,independent_validation=False,
        B_self_read=False,individual_predictions_read=False,main_truth_read=False,submission_created=False,GitHub_published=False))
    print(lt.groupby("group")[["r_all","baseline_R","delta"]].mean().to_string())
    print(oc.groupby(["group","pool"])[["r_all","baseline_R","delta"]].mean().to_string())


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--stage",choices=["inventory","freeze","aggregate"],required=True)
    stage=ap.parse_args().stage
    {"inventory":inventory,"freeze":freeze,"aggregate":aggregate}[stage]()
