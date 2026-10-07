"""今回の固定較正・厳密較正の保存結果と入力不変を確認する。再学習・提出なし。"""
from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
HYPER=HERE/"超幾何による同じ帰無分布の厳密計算"
ROOT=Path("/Users/taisei1105/pws2026-local")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={}
for rp in [HERE/"予測実行記録.json",HYPER/"厳密予測実行記録.json"]:
    r=json.loads(rp.read_text())
    code=HERE/"人物較正.py" if rp.parent==HERE else HYPER/"厳密較正.py"
    assert sha(code)==r["code_sha256"]
    if rp.parent==HERE:
        assert sha(HERE/"固定条件.json")==r["config_sha256"]
        old=ROOT/"本戦/攻撃フェーズ/Tashiro/20261007_AIA全体向上の分析"
        for name,d in r["imports"].items():assert sha(old/name)==d
    else:
        assert sha(HYPER/"追加固定条件.json")==r["additional_conditions_sha256"]
        assert sha(HERE/"固定条件.json")==r["parent_fixed_conditions_sha256"]
    for p,d in r["input_sha256"].items():
        assert sha(Path(p))==d;sources[p]=d
    key="predictions" if "predictions" in r else "prediction_entries"
    for x in r[key]:assert sha(rp.parent/x["path"])==x["sha256"]
    assert not r["main_truth_used"] and not r["truth_used_for_predictions"]
a=pd.read_csv(HERE/"集計/予備戦_対象別とnull反復.csv")
h=pd.read_csv(HYPER/"集計/厳密版_対象別とnull反復.csv")
hh=[]
for context in a.context.unique():
    z=h.copy();z["context"]=context;hh.append(z)
allrows=pd.concat([a]+hh,ignore_index=True)
bg=allrows[allrows.team.isin([18,19,29])].groupby(["team","context","method","source"])[["r_all","p_m","p_c"]].mean().reset_index()
bg["BG_source_note"]=bg.team.map({18:"phase BGコピーの特徴",19:"kit BGコピーの特徴",29:"phase/kit BG混合の特徴"})
bg.to_csv(HERE/"集計/BG由来C3対象の較正結果.csv",index=False)
orig=pd.read_csv(HERE/"集計/保存済み基準採点の一致.csv")
rep=pd.read_csv(HERE/"集計/全44C基準予測の一致.csv")
perm=pd.read_csv(HERE/"集計/行順変更.csv")
hperm=pd.read_csv(HYPER/"集計/厳密版_行順変更.csv")
ranks=pd.read_csv(HYPER/"集計/CDF経由経験順位と距離順位の一致.csv")
mathcheck=json.loads((HYPER/"組合せ全列挙との照合.json").read_text())
assert len(rep)==44 and rep.baseline_selection_changed.max()==0
assert perm.changed.max()==hperm.selected_changed.max()==0
assert ranks.split_CDF_vs_direct_rank_changes.max()==ranks.full_CDF_vs_direct_rank_changes.max()==0
assert orig.maximum_saved_metric_difference.max()<1e-12
for rp in [HERE/"採点実行記録.json",HYPER/"厳密採点実行記録.json"]:
    r=json.loads(rp.read_text());assert not r["main_truth_used"]
    for p,d in r["public_prelim_truth_sha256"].items():assert sha(Path(p))==d
checks=dict(input_files_unchanged=len(sources),baseline44_prediction_differences=0,prelim21_saved_score_max_diff=float(orig.maximum_saved_metric_difference.max()),
    original_permutation_cases=len(perm),original_permutation_max_change=int(perm.changed.max()),exact_permutation_cases=len(hperm),
    exact_permutation_max_change=int(hperm.selected_changed.max()),direct_distance_and_CDF_rank44_match=True,
    small_enum_conditions=len(mathcheck["small_pool"]),small_enum_max_float_error=mathcheck["small_max_error"],
    large_integer_combination_cases=len(mathcheck["large_integer_combination"]),large_max_error=mathcheck["large_max_error"],
    original_candidate_count=5,additional_exact_candidates=1,main_truth_used=False,submitted=False,is_official_phi=False)
(HERE/"検証結果.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2)+"\n")
manifest=[]
for p in sorted(HERE.rglob("*")):
    if p.is_file() and "作業用" not in p.parts and p.name!="公開対象のSHA256.json":
        manifest.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)))
(HERE/"公開対象のSHA256.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(checks,ensure_ascii=False))
