"""保存したmask/集計/hashだけで行結合診断を照合する。"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def near(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)))<1e-12

def main():
    rec=json.loads((HERE/"予測実行記録.json").read_text());score=json.loads((HERE/"採点実行記録.json").read_text())
    assert sha(HERE/"分析.py")==rec["code_sha256"]==score["code_sha256"]
    assert sha(HERE/"固定条件.json")==rec["protocol_sha256"]==score["protocol_sha256"]
    assert sha(HERE/"予測実行記録.json")==score["prediction_receipt_sha256"]
    for path,digest in rec["input_sha256"].items():assert sha(path)==digest
    for path,digest in rec["source_dependency_sha256"].items():assert sha(path)==digest
    for name,digest in score["output_sha256"].items():assert sha(HERE/"集計"/name)==digest
    assert len(rec["predictions"])==44 and sum(p["phase"]=="prelim" for p in rec["predictions"])==21
    transformed_checked=0
    for item in rec["predictions"]:
        assert sha(HERE/item["path"])==item["sha256"]
        with np.load(HERE/item["path"]) as z:
            assert z["original"].shape==(3,326) and z["shuffled"].shape==(20,3,326)
            assert z["original"].dtype==bool and z["shuffled"].dtype==bool
            assert np.array_equal(z["shuffled"][:,1:],np.repeat(z["original"][None,1:],20,axis=0))
            assert (z["shuffled"].sum(axis=2)==z["original"].sum(axis=1)[None,:]).all()
            transformed_checked+=20
    inv=pd.read_csv(HERE/"集計/値集合の証拠と選別の厳密不変.csv")
    preserve=pd.read_csv(HERE/"集計/性別内周辺値と非臨床列の保存.csv")
    assert len(inv)==len(preserve)==880 and inv.evidence_all6_arrays_exact.all()
    assert inv.all4_selected_changed.eq(0).all() and inv.top2_selected_changed.eq(0).all()
    assert preserve.same_sex_clinical_value_multisets_preserved.all() and preserve.other_columns_unchanged.all()
    repro=pd.read_csv(HERE/"集計/既存3方式の全44C再現.csv")
    assert len(repro)==44 and repro.baseline_pair_exact.all() and repro.baseline_mask_changed.eq(0).all() and repro.column_mask_changed.eq(0).all()
    ch=pd.read_csv(HERE/"集計/全44C_選別変化とJaccard.csv")
    assert len(ch)==44*20*3*3 and ch.selected_original.eq(ch.selected_shuffled).all()
    near(ch.jaccard,(ch.selected_original-ch.selected_changed/2)/(ch.selected_original+ch.selected_changed/2))
    assert ch[ch.method.ne("baseline_partial2_matching")].selected_changed.eq(0).all()
    table=pd.read_csv(HERE/"集計/原本Cと臨床列交換_予備対象別と20反復.csv")
    assert len(table)==21*3*21 and not table.duplicated(["team","method","source","repeat"]).any()
    near(table.p_m,table.hit_m/163);near(table.p_c,table.hit_c/163)
    near(table.r_all,np.clip((table.p_m-table.p_c)/np.maximum(1-table.p_c,1e-9),0,1))
    pm,pc=table.hit_m_rare/13,table.hit_c_rare/13
    rare=np.clip((pm-pc-1.64*np.sqrt(pm*(1-pm)/13+pc*(1-pc)/13))/np.maximum(1-pc,1e-9),0,1)
    near(table.r_rare,rare);near(table.r_max,np.maximum(table.r_all,rare))
    near(table.signed_hit_gap,table.hit_m-table.hit_c);near(table.r_all,table.unclipped_R_all+table.zero_clip_uplift)
    for (team,method),g in table.groupby(["team","method"]):
        assert len(g)==21
        if method!="baseline_partial2_matching":
            keys=["r_all","r_rare","r_max","p_m","p_c","hit_m","hit_c","hit_m_rare","hit_c_rare"]
            assert np.array_equal(g[keys].to_numpy(),np.repeat(g[g.source.eq("original_C")][keys].to_numpy(),21,axis=0))
    dd=pd.read_csv(HERE/"集計/臨床列交換の影響_予備対象別.csv")
    near(dd.delta_r_all,dd.delta_unclipped_R_all+dd.delta_zero_clip_uplift)
    pool=HERE.parent/"候補群の留保監査/候補プール固定.json"
    assert sha(pool)=="2615d649e932eb58decbbd3c01ab9163e19a85fc1ed9b1e2aa55bfb9acc4b6c8"
    report=dict(status="PASS",verifier_sha256=sha(__file__),transformed_C_predictions_checked=transformed_checked,
        column_evidence_invariance_checks=880,column_selection_invariance_checks=1760,
        same_sex_marginals_preserved=880,all_nonclinical_columns_preserved=880,
        original_masks_reproduced=44*3,public_prelim_score_rows=1323,
        all44x20_saved_before_truth=True,main_truth_read=False,B_self_read=False,initial_39_pool_sha256=sha(pool),
        independent_validation=False,real_team_shuffle_not_inferred=True)
    (HERE/"検証記録.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(report,ensure_ascii=False))

if __name__=="__main__":main()
