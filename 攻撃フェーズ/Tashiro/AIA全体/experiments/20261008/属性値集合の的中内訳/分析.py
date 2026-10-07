"""既存AIA326課題の選別切替を公開予備真値の区分で数える。新方式なし。"""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/"属性値集合による選別"
spec=importlib.util.spec_from_file_location("frozen_column_hit_diagnostic",SOURCE/"分析.py")
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
METHODS=["column_all4_mean","column_top2_mean"];CONTEXTS=exp.CONTEXTS
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=HERE/"集計";out.mkdir(exist_ok=True)
    rec=json.loads((SOURCE/"予測実行記録.json").read_text());assert rec["code_sha256"]==digest(SOURCE/"分析.py") and rec["config_sha256"]==digest(SOURCE/"固定条件.json")
    for path,h in rec["input_sha256"].items():assert digest(path)==h
    for item in rec["predictions"]:assert digest(SOURCE/item["path"])==item["sha256"]
    td=next((exp.ROOT/"予備戦/配布データ/結果").glob("正解*"))
    rows=[];selection=[];checks=[];main_public=[];truthhash={}
    scores=pd.read_csv(SOURCE/"集計/予備戦_対象別と20反復null.csv")
    scores=scores[scores.source=="real_C"].set_index(["team","context","method"])
    for phase,folder,_ in exp.SETS:
        for item in [x for x in rec["predictions"] if x["phase"]==phase]:
            k=item["team"];q=exp.core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            data=np.load(SOURCE/item["path"]);sex=q.sex.to_numpy()
            if phase=="main":
                for ci,context in enumerate(CONTEXTS):
                    for mi,m in enumerate(METHODS,start=1):
                        for value in [0,1]:
                            a=sex==value
                            main_public.append(dict(team=k,context=context,method=m,sex=value,Q_rows=int(a.sum()),selected=int(data["real"][ci,mi,a].sum()),
                               selected_fraction=float(data["real"][ci,mi,a].mean()),truth_used=False))
                continue
            tp=td/f"aia_truth_{k}.csv";truth=exp.core.read(tp).sort_values("challenge_row_id").reset_index(drop=True);truthhash[str(tp)]=digest(tp)
            assert q.challenge_row_id.tolist()==truth.challenge_row_id.tolist()
            time=truth.time.to_numpy();assert ((time>=0)&(time<=10)).all()
            bins={"early_0_to_half":time<=.5,"long_9_to_10":time>=9,"other":(time>.5)&(time<9)}
            assert np.stack(list(bins.values())).sum(axis=0).min()==1 and np.stack(list(bins.values())).sum(axis=0).max()==1
            labels=truth.label.to_numpy()
            for ci,context in enumerate(CONTEXTS):
                baseline=data["real"][ci,0];base_hit=np.abs(np.where(baseline,9.5,0)-time)<=.5
                for mi,m in enumerate(METHODS,start=1):
                    selected=data["real"][ci,mi];hit=np.abs(np.where(selected,9.5,0)-time)<=.5
                    branch=(selected&bins["long_9_to_10"])|(~selected&bins["early_0_to_half"]);assert np.array_equal(hit,branch)
                    switches={"0_to_9.5":~baseline&selected,"9.5_to_0":baseline&~selected,"same":baseline==selected}
                    for label in ["member","control"]:
                        a=labels==label;key="hit_m" if label=="member" else "hit_c"
                        base_saved=scores.loc[(k,context,"baseline_partial2_matching"),key];new_saved=scores.loc[(k,context,m),key]
                        assert int(base_hit[a].sum())==int(base_saved) and int(hit[a].sum())==int(new_saved)
                        checks.append(dict(team=k,context=context,method=m,label=label,Q_rows=int(a.sum()),old_hits=int(base_hit[a].sum()),
                            new_hits=int(hit[a].sum()),hit_difference=int(hit[a].sum()-base_hit[a].sum()),matches_saved_score=True))
                        for value in [0,1]:
                            labsex=a&(sex==value)
                            selection.append(dict(team=k,context=context,method=m,label=label,sex=value,Q_rows=int(labsex.sum()),
                                old_selected=int((baseline&labsex).sum()),new_selected=int((selected&labsex).sum()),
                                selected_change=int((selected&labsex).sum()-(baseline&labsex).sum())))
                            for name,tmask in bins.items():
                                for switch,smask in switches.items():
                                    mask=labsex&tmask&smask
                                    rows.append(dict(team=k,context=context,method=m,label=label,sex=value,time_type=name,switch=switch,
                                      n=int(mask.sum()),old_selected=int((baseline&mask).sum()),new_selected=int((selected&mask).sum()),
                                      old_hits=int(base_hit[mask].sum()),new_hits=int(hit[mask].sum()),
                                      delta_hits=int(hit[mask].sum()-base_hit[mask].sum())))
    detail=pd.DataFrame(rows);sel=pd.DataFrame(selection);check=pd.DataFrame(checks)
    for key,z in detail.groupby(["team","context","method","label"]):
        saved=check[(check.team==key[0])&(check.context==key[1])&(check.method==key[2])&(check.label==key[3])]
        assert len(saved)==1 and int(z.n.sum())==163 and int(z.delta_hits.sum())==int(saved.hit_difference.iloc[0])
    # 同性別の選別人数を固定したため、MとCの選別人数差の和は各対象・sexで0。
    assert sel.groupby(["team","context","method","sex"]).selected_change.sum().eq(0).all()
    detail.to_csv(out/"選別切替_対象所属性別時間帯.csv",index=False)
    sel.to_csv(out/"選択会員数と対照数_対象性別.csv",index=False)
    check.to_csv(out/"区分の的中差と保存済み採点の照合.csv",index=False)
    pd.DataFrame(main_public).to_csv(out/"本戦_新方式の公開選択人数だけ.csv",index=False)
    totals=[];stotals=[]
    for scope,exclude in [("prelim20",[5]),("processed18",[1,2,5]),("all21_diagnostic",[])]:
        d=detail[~detail.team.isin(exclude)];s=sel[~sel.team.isin(exclude)]
        g=d.groupby(["context","method","label","sex","time_type","switch"])[["n","old_selected","new_selected","old_hits","new_hits","delta_hits"]].sum().reset_index()
        g.insert(0,"group",scope);totals.append(g)
        h=s.groupby(["context","method","label","sex"])[["Q_rows","old_selected","new_selected","selected_change"]].sum().reset_index()
        h.insert(0,"group",scope);stotals.append(h)
    total=pd.concat(totals);stotal=pd.concat(stotals)
    total.to_csv(out/"選別切替_区分別合計.csv",index=False)
    stotal.to_csv(out/"選択会員数と対照数_合計.csv",index=False)
    collapsed=total.groupby(["group","context","method","label"])[["n","old_selected","new_selected","old_hits","new_hits","delta_hits"]].sum().reset_index()
    collapsed.to_csv(out/"所属性別時間帯の合計照合.csv",index=False)
    expected={("column_all4_mean","member"):3,("column_all4_mean","control"):-27,("column_top2_mean","member"):-4,("column_top2_mean","control"):-23}
    z=collapsed[(collapsed.group=="processed18")&(collapsed.context=="full_reference_overlap")]
    for key,value in expected.items():
        actual=int(z[(z.method==key[0])&(z.label==key[1])].delta_hits.iloc[0]);assert actual==value
    exp.writej(HERE/"実行記録.json",dict(code_sha256=digest(__file__),diagnostic_protocol_sha256=digest(HERE/"診断条件.json"),
        source_code_sha256=digest(SOURCE/"分析.py"),source_prediction_receipt_sha256=digest(SOURCE/"予測実行記録.json"),
        public_prelim_truth_sha256=truthhash,prediction_cache_hashes_verified=True,public_input_files_unchanged=len(rec["input_sha256"]),
        formula_and_saved_score_checks=len(check),prelim_targets=21,main_public_targets=23,individual_ID_or_truth_exported=False,
        main_truth_used=False,B_self_used=False,new_methods=0,predictions_or_parameters_changed=False,excluded_from_39_pool=True,
        all4_M_plus3_C_minus27_and_top2_M_minus4_C_minus23_reproduced=True,
        note="AIA326課題の診断。MIA1049候補順位・希少11人選びの性能ではない。真会員確率や時間の因果関係ではない。"))
    print(z.to_string(index=False))
    print(stotal[(stotal.group=="processed18")&(stotal.context=="full_reference_overlap")].groupby(["method","label"])[["old_selected","new_selected","selected_change"]].sum().to_string())
if __name__=="__main__":main()

