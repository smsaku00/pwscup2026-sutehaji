"""既存の6特徴・1分岐・nested選択に、凍結候補poolだけを差し替える。

結果閲覧後の追加監査。新方式・新特徴・新閾値候補を追加しない。
原本スクリプト全体をimportせず、I/Oのない既存4関数のみASTで抽出する。
保存済み対象集計とC特徴だけを読み、個票/本戦truth/B_selfは読まない。
"""
from __future__ import annotations
import ast,hashlib,json,platform,sys,time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
SOURCE=BASE/"加工特徴と方式選択"
OUT=HERE/"既存分岐へのpool差替"
SEED=20261008
ROUTE_FEATURES=["lab_round2_share","lab_unique_share","corr_gap_bg_sex","q_C_full_nn_median","time_long_share","BG_QI7_copy_share"]
PENALTIES=[0.,.025,.05,.1]
MIN_LEAF=4
FUNCTIONS=["predict_policy","fit_policy","choose_penalty","processing_group"]
OLD_NAMES=["partial2_matching","S4_nn","full_sex_pref_nn","partial4_nn","partial2_density","GenLRA_nn","gaussian_source","constant95"]
OLD_IDS=["anchor:P01","historical:S4_log_iqr_soft/nn_all","historical:log_iqr_both/nn_all",
    "historical:partial4_log_iqr_both/nn_all","historical:density/partial2_sex_k3/long_gate0.75",
    "historical:GenLRA/ordinal7h7/NN/gate0.75","historical:sex/linear_gaussian/pooled/long0.75","historical:constant_9.5"]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def load_functions():
    tree=ast.parse((SOURCE/"分析.py").read_text())
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ["ROUTE_FEATURES","PENALTIES","MIN_LEAF"]}
    assert assignments==dict(ROUTE_FEATURES=ROUTE_FEATURES,PENALTIES=PENALTIES,MIN_LEAF=MIN_LEAF)
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in FUNCTIONS]
    assert set(n.name for n in selected)==set(FUNCTIONS)
    module=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]))
    namespace=dict(np=np,PENALTIES=PENALTIES,MIN_LEAF=MIN_LEAF)
    exec(compile(module,str(SOURCE/"分析.py"),"exec"),namespace)
    digest=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest()
    return namespace,digest

def nested(x,y,teams,names,func,groups=None,score_perturbation=None):
    rows=[];units=teams if groups is None else np.unique(groups)
    for unit in units:
        test=teams==unit if groups is None else groups==unit;train=~test
        work=y.copy()
        if score_perturbation is not None:work[test]=score_perturbation
        penalty,inner=func["choose_penalty"](x[train],work[train])
        policy=func["fit_policy"](x[train],work[train],penalty)
        chosen=func["predict_policy"](policy,x[test])
        for team,method,rowid in zip(teams[test],chosen,np.flatnonzero(test)):
            rows.append(dict(team=int(team),chosen_method=names[method],r_all=float(y[rowid,method]),
                baseline_r_all=float(y[rowid,0]),delta=float(y[rowid,method]-y[rowid,0]),
                heldout_group=str(unit) if groups is not None else "single_team",
                feature=None if policy["feature"] is None else ROUTE_FEATURES[policy["feature"]],
                threshold=policy["threshold"],left_method=names[policy["left_method"]],
                right_method=names[policy["right_method"]],penalty=penalty,
                inner_means=json.dumps(inner,sort_keys=True)))
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)

def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    assert not (OUT/"追加監査条件固定.json").exists(),"追加監査条件を上書きしない"
    frozen=json.loads((HERE/"候補プール固定.json").read_text())
    assert sha(HERE/"分析.py")==frozen["analysis_code_sha256"]
    for name,expected in frozen["snapshot_sha256"].items():assert sha(HERE/name)==expected
    features_path=SOURCE/"集計/公開Cの加工特徴_44組.csv"
    func,func_sha=load_functions()
    inputs=[SOURCE/"分析.py",SOURCE/"事前固定仕様.json",features_path,
        HERE/"候補プール固定.json",HERE/"共通全BG_全候補対象別.csv",HERE/"集計/global_LOTO_対象別.csv"]
    for scope in ["all20","processed18"]:
        for name in [f"nested_LTO_{scope}.csv",f"加工群を外した検証_{scope}.csv"]:inputs.append(SOURCE/"集計"/name)
    record=dict(fixed_at=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),code_sha256=sha(Path(__file__)),
        source_function_AST_sha256=func_sha,source_functions=FUNCTIONS,
        input_sha256={str(p):sha(p) for p in inputs},primary_candidates=frozen["primary_pool"],
        route_features=ROUTE_FEATURES,penalties=PENALTIES,min_leaf=MIN_LEAF,feature_shuffle_repeats=50,seed=SEED,
        new_method_added=False,new_feature_added=False,new_threshold_grid_added=False,
        posthoc_after_global_selection_results=True,independent_design_validation=False,
        outer_group_out_inner_still_leave_team_out=True,individual_predictions_read=False,
        main_truth_read=False,B_self_read=False,submission_created=False,GitHub_published=False)
    dump(OUT/"追加監査条件固定.json",record)
    f=pd.read_csv(features_path);pf=f[f.phase.eq("prelim")].set_index("team")
    assert np.isfinite(pf[ROUTE_FEATURES].to_numpy(float)).all()
    assert (func["processing_group"](pf)==pf.processing_group.to_numpy()).all()
    scores=pd.read_csv(HERE/"共通全BG_全候補対象別.csv")
    pivot=scores.pivot(index="team",columns="candidate_id",values="r_all")
    global_loto=pd.read_csv(HERE/"集計/global_LOTO_対象別.csv")
    summaries=[];all_rows=[];repro=[];leak=[];groups=[];bootstrap=[];shuffles=[]
    for group,scope,excluded in [("prelim20","all20",[5]),("processed18","processed18",[1,2,5])]:
        teams=pivot.index[~pivot.index.isin(excluded)].to_numpy();x=pf.loc[teams,ROUTE_FEATURES].to_numpy(float)
        labels=pf.loc[teams,"processing_group"].to_numpy();names=frozen["primary_pool"]
        y=pivot.loc[teams,names].to_numpy(float);assert np.isfinite(y).all()
        groups.extend(dict(group=group,processing_group=name,targets=int((labels==name).sum())) for name in np.unique(labels))
        oldy=pivot.loc[teams,OLD_IDS].to_numpy(float)
        for evaluation,glab,file in [("nested_LTO",None,f"nested_LTO_{scope}.csv"),
            ("processing_group_out",labels,f"加工群を外した検証_{scope}.csv")]:
            recreated=nested(x,oldy,teams,OLD_NAMES,func,glab)
            original=pd.read_csv(SOURCE/"集計"/file).sort_values("team").reset_index(drop=True)
            assert recreated.chosen_method.equals(original.chosen_method)
            assert np.max(np.abs(recreated.r_all-original.r_all))<1e-12
            assert np.max(np.abs(recreated.penalty-original.penalty))<1e-12
            assert recreated.left_method.equals(original.left_method) and recreated.right_method.equals(original.right_method)
            assert np.max(np.abs(recreated.threshold.fillna(-999)-original.threshold.fillna(-999)))<1e-12
            repro.append(dict(group=group,evaluation=evaluation,targets=len(teams),method_match=True,
                maximum_score_error=float(np.max(np.abs(recreated.r_all-original.r_all))),old8_rule_reproduced=True))
            result=nested(x,y,teams,names,func,glab);result.insert(0,"group",group);result.insert(1,"evaluation",evaluation)
            all_rows.append(result)
            hidden=nested(x,y,teams,names,func,glab,score_perturbation=999.)
            assert result.chosen_method.equals(hidden.chosen_method)
            assert result.penalty.equals(hidden.penalty)
            assert result.left_method.equals(hidden.left_method) and result.right_method.equals(hidden.right_method)
            assert np.max(np.abs(result.threshold.fillna(-999)-hidden.threshold.fillna(-999)))<1e-12
            leak.append(dict(group=group,evaluation=evaluation,targets=len(teams),heldout_scores_replaced_by=999.,
                choices_changed=0,policy_changed=0))
            baseline=y[:,0];glob=global_loto[global_loto.group.eq(group)].set_index("team").loc[teams,"r_all"].to_numpy()
            summaries.append(dict(group=group,evaluation=evaluation,targets=len(teams),router_R=float(result.r_all.mean()),
                baseline_R=float(baseline.mean()),global_LOTO_R=float(glob.mean()),
                delta_vs_anchor=float(result.delta.mean()),delta_vs_global_LOTO=float((result.r_all.to_numpy()-glob).mean()),
                changed=int(result.chosen_method.ne(names[0]).sum()),improved=int(result.delta.gt(1e-12).sum()),
                worsened=int(result.delta.lt(-1e-12).sum()),independent_validation=False))
            rng=np.random.default_rng(2026100821+(group=="processed18"));draws=rng.integers(0,len(teams),(10000,len(teams)))
            for comparison,values in [("anchor",baseline),("global_LOTO",glob)]:
                delta=result.r_all.to_numpy()-values;samples=delta[draws].mean(1)
                bootstrap.append(dict(group=group,evaluation=evaluation,comparison=comparison,
                    mean_delta_R=float(delta.mean()),descriptive_q025=float(np.quantile(samples,.025)),
                    descriptive_q975=float(np.quantile(samples,.975)),paired_resamples=10000,
                    method_choice_refitted=False,independent_generalization_CI=False,p_value_computed=False))
        if group=="processed18":
            rng=np.random.default_rng(SEED)
            for repeat in range(50):
                shuffled=nested(x[rng.permutation(len(x))],y,teams,names,func)
                shuffles.append(dict(repeat=repeat,router_R=float(shuffled.r_all.mean()),delta_vs_anchor=float(shuffled.delta.mean()),
                    exploratory_only=True,p_value_computed=False))
    pd.concat(all_rows).to_csv(OUT/"凍結39候補_nested選択の対象別.csv",index=False)
    pd.DataFrame(summaries).to_csv(OUT/"凍結39候補_nested選択の平均.csv",index=False)
    pd.DataFrame(repro).to_csv(OUT/"旧8候補_同一規則の再現.csv",index=False)
    pd.DataFrame(leak).to_csv(OUT/"外側正解の遮断確認.csv",index=False)
    pd.DataFrame(groups).to_csv(OUT/"加工群の対象数.csv",index=False)
    pd.DataFrame(bootstrap).to_csv(OUT/"nested差のpaired_bootstrap記述幅.csv",index=False)
    pd.DataFrame(shuffles).to_csv(OUT/"既存50反復の特徴入れ替え対照.csv",index=False)
    for path,expected in record["input_sha256"].items():assert sha(Path(path))==expected
    dump(OUT/"追加監査実行記録.json",dict(code_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/"追加監査条件固定.json"),
        output_sha256={p.name:sha(p) for p in OUT.glob("*.csv")},seconds=time.time()-start,python=sys.executable,
        python_version=platform.python_version(),main_predictions_created=False,independent_validation=False))
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=="__main__":main()
