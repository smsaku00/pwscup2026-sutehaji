"""32背景感度の保存状態・件数・選別人数・元方式凍結を確認する。"""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("column_background_audit",HERE/"分析.py")
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    pred=json.loads((HERE/"予測実行記録.json").read_text());scored=json.loads((HERE/"採点実行記録.json").read_text())
    assert pred["code_sha256"]==digest(HERE/"分析.py") and pred["protocol_sha256"]==digest(HERE/"固定プロトコル.json")
    assert pred["source_code_sha256"]==digest(audit.SOURCE/"分析.py")
    assert pred["source_protocol_sha256"]==digest(audit.SOURCE/"固定条件.json")
    assert pred["matched_reference_setup_code_sha256"]==digest(audit.BASELINE/"分析.py")
    for p,h in pred["input_sha256"].items():assert digest(p)==h
    assert scored["prediction_receipt_sha256"]==digest(HERE/"予測実行記録.json")
    assert len(pred["predictions"])==44 and pred["completed_seeds"]==list(range(20261008,20261040)) and not pred["unfinished_seeds"]
    nc=0;realcases=nullcases=0
    for phase,folder,_ in audit.exp.SETS:
        for item in [x for x in pred["predictions"] if x["phase"]==phase]:
            cache=HERE/item["path"];assert digest(cache)==item["sha256"];data=np.load(cache)
            assert data["real"].shape==(32,2,326) and data["full"].shape==(2,326)
            q=audit.exp.core.read(folder/f"aia_challenge_{item['team']}.csv").sort_values("challenge_row_id")
            for sex in np.unique(q.sex):
                a=q.sex.to_numpy()==sex;n=int(np.ceil(int(a.sum())*.75))
                assert (data["real"][:,:,a].sum(axis=2)==n).all() and (data["full"][:,a].sum(axis=1)==n).all()
                if len(data["null"]):assert (data["null"][:,:,:,a].sum(axis=3)==n).all()
            realcases+=32*2
            if len(data["null"]):assert data["null"].shape==(32,20,2,326);nc+=1;nullcases+=32*20*2
    assert nc==20 and realcases==2816 and nullcases==25600
    out=HERE/"集計"
    real=pd.read_csv(out/"32参照_実C対象別.csv");null=pd.read_csv(out/"32参照_同性別BG_C全反復.csv")
    assert len(real)==1344 and len(null)==25600
    assert not real.duplicated(["team","seed","method"]).any() and not null.duplicated(["team","seed","method","repeat"]).any()
    perms=pd.read_csv(out/"行順変更.csv");assert len(perms)==2816 and perms.changed.eq(0).all()
    full=pd.read_csv(out/"全44C_凍結値集合方式の再現.csv");assert len(full)==44 and full.changed_original_full_BG_masks.eq(0).all()
    sample=pd.read_csv(out/"BG分割と抽出順の独立実装照合.csv")
    assert len(sample)==2560 and sample.setup_matches_independent_implementation.all() and sample.pool_reference_overlap.eq(0).all()
    hist=pd.read_csv(out/"本戦_選択頻度の対象別性別分布.csv")
    summary=pd.read_csv(out/"本戦_選択頻度の対象別性別要約.csv")
    for key,g in hist.groupby(["team","method","sex"]):
        s=summary[(summary.team==key[0])&(summary.method==key[1])&(summary.sex==key[2])]
        assert len(s)==1 and int(g.q_rows.sum())==int(s.q_n.iloc[0])
    comp=pd.read_csv(out/"同一分割での旧基準比較.csv");assert len(comp)==128
    assert (comp.targets_column==comp.targets_baseline).all()
    counts={}
    for m,g in comp[comp.scope=="processed18"].groupby("method"):
        counts[m]={metric:int((g[metric]>0).sum()) for metric in ["real_R_minus_baseline","excess_R_minus_baseline","signed_excess_minus_baseline"]}
    receipt=dict(prediction_receipt_sha256=digest(HERE/"予測実行記録.json"),score_receipt_sha256=digest(HERE/"採点実行記録.json"),
       verification_code_sha256=digest(__file__),completed_seeds=pred["completed_seeds"],unfinished_seeds=[],
       public_input_files_unchanged=len(pred["input_sha256"]),original_fixed_methods_unchanged=True,
       actual_prediction_cases=realcases,null_prediction_cases=nullcases,permutation_cases=len(perms),permutation_changed_max=0,
       null_target_count=nc,sex75_quota_verified=True,source_full_BG_mask_reproduction_changed_max=0,
       split_and_draw_order_matches_independent_implementation_cases=len(sample),post_result_audit=True,
       main_truth_used=False,B_self_used=False,selection_candidate=False,choose_best_seed=False,
       processed18_positive_seed_count_vs_baseline=counts)
    (HERE/"検証記録.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

