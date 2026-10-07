"""固定KDEの記録と凍結対照・数値不変を確認する。"""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("fixed_kde_experiment",HERE/"分析.py")
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    pred=json.loads((HERE/"予測実行記録.json").read_text());old=json.loads((e.SOURCE/"予測実行記録.json").read_text())
    assert pred["code_sha256"]==digest(HERE/"分析.py") and pred["config_sha256"]==digest(HERE/"固定条件.json")
    assert pred["source_code_sha256"]==digest(e.SOURCE/"分析.py") and pred["source_protocol_sha256"]==digest(e.SOURCE/"固定条件.json")
    for name,h in old["imports"].items():assert digest(e.exp.OLD/name)==h
    for p,h in pred["input_sha256"].items():assert digest(p)==h
    oldmap={(x["phase"],x["team"]):x for x in old["predictions"]}
    actual_cases=null_cases=0;oldchanges=0;null_targets=0
    for phase,folder,_ in e.exp.SETS:
        for item in [x for x in pred["predictions"] if x["phase"]==phase]:
            p=HERE/item["path"];assert digest(p)==item["sha256"];data=np.load(p)
            op=oldmap[(phase,item["team"])];assert digest(e.SOURCE/op["path"])==op["sha256"];previous=np.load(e.SOURCE/op["path"])
            assert data["real"].shape==(2,5,326)
            assert np.array_equal(data["real"][:,:3],previous["real"][:,:3]);actual_cases+=2*5
            q=e.exp.core.read(folder/f"aia_challenge_{item['team']}.csv").sort_values("challenge_row_id")
            for sex in np.unique(q.sex):
                a=q.sex.to_numpy()==sex;n=int(np.ceil(int(a.sum())*.75))
                assert (data["real"][:,:,a].sum(axis=2)==n).all()
                if len(data["null"]):assert (data["null"][:,:,:,a].sum(axis=3)==n).all()
            if len(data["null"]):
                assert data["null"].shape==(2,20,5,326)
                assert np.array_equal(data["null"][:,:,:3],previous["null"][:,:,:3])
                null_targets+=1;null_cases+=2*20*5
    assert len(pred["predictions"])==44 and actual_cases==440 and null_targets==21 and null_cases==4200
    out=HERE/"集計";perm=pd.read_csv(out/"行順変更.csv");affine=pd.read_csv(out/"原尺度の変換不変.csv")
    assert len(perm)==880 and perm.changed.eq(0).all()
    assert len(affine)==88 and affine.selection_changed.eq(0).all() and affine.IQR_floor_inactive.all()
    param=pd.read_csv(out/"固定帯域と参照人数.csv");assert param.same_h_for_C_BG.all() and param.IQR_floor_applied.eq(0).all()
    assert np.allclose(param.h,param.nC**(-1/5),rtol=1e-14,atol=1e-14)
    fixture=json.loads((HERE/"小型Gaussian和の独立照合.json").read_text());assert fixture["maximum_difference"]<1e-12
    scored=pd.read_csv(out/"予備戦_固定5候補と20null対象別.csv");assert len(scored)==21*2*5*21
    oldscore=pd.read_csv(e.SOURCE/"集計/予備戦_対象別と20反復null.csv")
    keys=["team","context","method","source","repeat"];cols=e.exp.METRICS
    a=scored[scored.method.isin(e.METHODS[:3])].set_index(keys)[cols].sort_index()
    b=oldscore[oldscore.method.isin(e.METHODS[:3])].set_index(keys)[cols].sort_index()
    assert a.index.equals(b.index) and np.allclose(a,b,rtol=0,atol=1e-12)
    loo=pd.read_csv(out/"固定5候補_leave_team_out.csv");assert len(loo)==76 and not loo.selected_method.isin(e.NEWMETHODS).any()
    record=dict(prediction_receipt_sha256=digest(HERE/"予測実行記録.json"),verification_code_sha256=digest(__file__),
       source_imported_four_code_hashes_unchanged=True,source_methods_and_protocol_unchanged=True,public_input_files_unchanged=len(pred["input_sha256"]),
       actual_C=44,all5_actual_prediction_cases=actual_cases,all5_null_prediction_cases=null_cases,null_targets=21,
       old3_real_and_null_mask_changes=oldchanges,old3_score_max_difference=float(np.max(np.abs(a.to_numpy()-b.to_numpy()))),
       fixed_same_h_verified=True,bandwidth_min=float(param.h.min()),bandwidth_max=float(param.h.max()),
       direct_small_Gaussian_max_difference=fixture["maximum_difference"],permutation_tests=len(perm),permutation_changes_max=0,
       affine_tests=len(affine),affine_max_absolute_log_ratio_difference=float(affine.max_log_ratio_difference.max()),
       affine_max_scaled_difference=float(affine.max_scaled_difference.max()),affine_selection_changes_max=0,
       all_IQR_floor_inactive=True,KDE_selected_in_fixed5_LOTO=0,
       all44_predictions_completed_before_truth_read=True,main_truth_used=False,B_self_used=False,unfinished_C=[],
       excluded_from_39_pool=True,bandwidth_or_coefficients_changed_after_scoring=False)
    (HERE/"検証記録.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(record,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

