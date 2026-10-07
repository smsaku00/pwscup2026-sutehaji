"""後付け3反証を主poolから外したまま別の条件表に記録する。"""
from __future__ import annotations
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
FAMILY=HERE.parent/"属性値集合による選別"
OUT=HERE/"後付け対照の条件監査"
METRICS=["r_all","r_rare","r_max","p_m","p_c","hit_m","hit_c","hit_m_rare","hit_c_rare"]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/"条件と入力hash.json").exists(),"後付け記録を上書きしない"
    files=[FAMILY/"集計/予備戦_対象別と20反復null.csv",FAMILY/"後付けの交絡反証/集計/後付け対照_対象別と20反復null.csv",
        FAMILY/"後付けの交絡反証/追加固定条件.json",FAMILY/"後付けの交絡反証/交絡の反証.py",HERE/"候補プール固定.json"]
    spec=dict(recorded_at=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),code_sha256=sha(Path(__file__)),
        input_sha256={str(p):sha(p) for p in files},posthoc_after_outcomes=True,in_primary_pool=False,
        in_global_LOTO=False,conditions_kept_separate=True,individual_predictions_read=False,
        main_truth_read=False,B_self_read=False,submission_created=False)
    dump(OUT/"条件と入力hash.json",spec)
    original=pd.read_csv(files[0]);extra=pd.read_csv(files[1])
    original["role"]="original_fixed_candidate_or_baseline";extra["role"]="posthoc_falsification_only"
    d=pd.concat([original,extra],ignore_index=True)
    assert not d.duplicated(["team","context","source","repeat","method"]).any()
    assert np.max(np.abs(d.p_m-d.hit_m/163))<1e-12 and np.max(np.abs(d.p_c-d.hit_c/163))<1e-12
    unclip=(d.p_m-d.p_c)/np.maximum(1-d.p_c,1e-9)
    assert np.max(np.abs(d.r_all-np.clip(unclip,0,1)))<1e-12
    d["signed_hit_gap"]=d.hit_m-d.hit_c;d["unclipped_R_all"]=unclip;d["zero_clip_uplift"]=d.r_all-unclip
    d["eligible_primary_global_selection"]=False
    d.to_csv(OUT/"別条件_当初4候補と後付け3対照_全対象反復.csv",index=False)
    means=[];gaps=[];actual_long=[]
    for group,excluded in [("prelim20",[5]),("processed18",[1,2,5])]:
        use=d[~d.team.isin(excluded)]
        avg=use.groupby(["context","source","method","role"])[METRICS+["signed_hit_gap","unclipped_R_all","zero_clip_uplift"]].mean().reset_index()
        avg.insert(0,"group",group);means.append(avg)
        actual=use[use.source.eq("real_C")].copy();actual.insert(0,"group",group);actual_long.append(actual)
        for context,cd in actual.groupby("context"):
            base=cd[cd.method.eq("baseline_partial2_matching")].set_index("team").sort_index()
            for method,md in cd.groupby("method"):
                aligned=md.set_index("team").reindex(base.index);assert aligned.hit_m.notna().all()
                gaps.append(dict(group=group,context=context,method=method,role=aligned.role.iloc[0],targets=len(base),
                    rows_per_class=163*len(base),R_all=float(aligned.r_all.mean()),
                    delta_R_vs_context_baseline=float((aligned.r_all-base.r_all).mean()),
                    delta_hit_m=int((aligned.hit_m-base.hit_m).sum()),delta_hit_c=int((aligned.hit_c-base.hit_c).sum()),
                    delta_signed_hit_gap=int((aligned.signed_hit_gap-base.signed_hit_gap).sum()),
                    delta_unclipped_R=float((aligned.unclipped_R_all-base.unclipped_R_all).mean()),
                    delta_zero_clip_uplift=float((aligned.zero_clip_uplift-base.zero_clip_uplift).mean()),
                    improved_targets=int((aligned.r_all>base.r_all+1e-12).sum()),
                    worsened_targets=int((aligned.r_all<base.r_all-1e-12).sum()),
                    equal_targets=int((np.abs(aligned.r_all-base.r_all)<=1e-12).sum()),
                    in_primary_pool=False,in_global_LOTO=False))
    pd.concat(means).to_csv(OUT/"別条件_反復平均.csv",index=False)
    pd.concat(actual_long).to_csv(OUT/"別条件_実C対象別20と18.csv",index=False)
    g=pd.DataFrame(gaps);g.to_csv(OUT/"別条件_実Cの的中差と0切り上げ.csv",index=False)
    for path,value in spec["input_sha256"].items():assert sha(Path(path))==value
    dump(OUT/"実行記録.json",dict(code_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/"条件と入力hash.json"),
        output_sha256={p.name:sha(p) for p in OUT.glob("*.csv")},python=sys.executable,primary_pool_unchanged=True))
    print(g[g.group.eq("processed18")][["context","method","R_all","delta_signed_hit_gap"]].to_string(index=False))

if __name__=="__main__":main()
