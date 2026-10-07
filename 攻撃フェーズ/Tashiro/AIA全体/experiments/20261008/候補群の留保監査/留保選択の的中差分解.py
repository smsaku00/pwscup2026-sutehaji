"""保存済みLOTO/nested選択を的中数・0切り上げ前後の差に分解する。"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"留保選択の的中差分解"
FIELDS=["r_all","hit_m","hit_c","signed_hit_gap","unclipped_R_all","zero_clip_uplift"]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/"条件と入力hash.json").exists(),"分解記録を上書きしない"
    paths=[HERE/"共通全BG_全候補対象別.csv",HERE/"集計/global_LOTO_対象別.csv",
        HERE/"既存分岐へのpool差替/凍結39候補_nested選択の対象別.csv",HERE/"候補プール固定.json"]
    spec=dict(code_sha256=sha(Path(__file__)),input_sha256={str(p):sha(p) for p in paths},
        selected_method_not_changed=True,candidate_pool_not_changed=True,posthoc_decomposition_only=True,
        individual_predictions_read=False,main_truth_read=False,B_self_read=False)
    dump(OUT/"条件と入力hash.json",spec)
    outcomes=pd.read_csv(paths[0]).set_index(["team","candidate_id"])
    loto=pd.read_csv(paths[1]);loto["evaluation"]="global_LOTO";loto=loto.rename(columns={"candidate_id":"chosen_method"})
    nested=pd.read_csv(paths[2]);chosen=pd.concat([loto,nested],ignore_index=True)
    rows=[]
    for s in chosen.itertuples(index=False):
        result=outcomes.loc[(s.team,s.chosen_method)];base=outcomes.loc[(s.team,"anchor:P01")]
        assert abs(result.r_all-s.r_all)<1e-12
        row=dict(group=s.group,evaluation=s.evaluation,team=s.team,chosen_method=s.chosen_method)
        for field in FIELDS:
            row[field]=float(result[field]);row["baseline_"+field]=float(base[field]);row["delta_"+field]=float(result[field]-base[field])
        assert abs(row["delta_r_all"]-row["delta_unclipped_R_all"]-row["delta_zero_clip_uplift"])<1e-12
        rows.append(row)
    d=pd.DataFrame(rows);d.to_csv(OUT/"選択済み方式の対象別分解.csv",index=False)
    summary=[]
    for (group,evaluation),g in d.groupby(["group","evaluation"]):
        summary.append(dict(group=group,evaluation=evaluation,targets=len(g),rows_per_class=163*len(g),
            mean_R_all=float(g.r_all.mean()),baseline_R_all=float(g.baseline_r_all.mean()),
            mean_delta_R_all=float(g.delta_r_all.mean()),
            mean_delta_unclipped_R_all=float(g.delta_unclipped_R_all.mean()),
            mean_delta_zero_clip_uplift=float(g.delta_zero_clip_uplift.mean()),
            delta_hit_m=int(g.delta_hit_m.sum()),delta_hit_c=int(g.delta_hit_c.sum()),
            delta_signed_hit_gap=int(g.delta_signed_hit_gap.sum()),
            improved_targets=int(g.delta_r_all.gt(1e-12).sum()),worsened_targets=int(g.delta_r_all.lt(-1e-12).sum()),
            equal_targets=int(g.delta_r_all.abs().le(1e-12).sum()),independent_validation=False))
    s=pd.DataFrame(summary);s.to_csv(OUT/"留保選択の的中差と0切り上げ平均.csv",index=False)
    for path,expected in spec["input_sha256"].items():assert sha(Path(path))==expected
    dump(OUT/"実行記録.json",dict(code_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/"条件と入力hash.json"),
        output_sha256={p.name:sha(p) for p in OUT.glob("*.csv")},python=sys.executable,candidate_pool_unchanged=True))
    print(s.to_string(index=False))

if __name__=="__main__":main()
