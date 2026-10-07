"""保存済み解析の入力・コード・採点照合を確認する。モデルは再学習しない。"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PAIRS={
    "分析.py":"実行記録.json", "学習比較.py":"公開正解学習_実行記録.json",
    "多尺度学習.py":"多尺度学習_実行記録.json", "密度差時間窓.py":"密度差時間窓_実行記録.json",
    "背景差し替え監査.py":"背景差し替え_実行記録.json", "分布識別.py":"分布識別_実行記録.json",
    "時間逆探索.py":"時間逆探索_実行記録.json", "GenLRAのAIA適用.py":"GenLRA_AIA_実行記録.json",
    "対応競合の分析.py":"対応競合_実行記録.json", "課題単独の漏洩監査.py":"課題単独_実行記録.json",
    "対応競合の頑健性.py":"対応競合頑健性_実行記録.json", "背景参照の感度.py":"背景参照感度_実行記録.json",
    "同点に依存しない対応.py":"同点修正_実行記録.json", "属性と選択人数の比較.py":"属性と選択人数_実行記録.json",
    "集約と監査.py":"集約監査_実行記録.json", "公式結果からの追加推定.py":"公式結果の追加推定.json"}


def run():
    hashes={};code_records=0;input_checks=0
    for code,record in PAIRS.items():
        d=json.loads((HERE/record).read_text(encoding="utf-8"))
        assert hashlib.sha256((HERE/code).read_bytes()).hexdigest()==d["code_sha256"],code
        code_records+=1
        for name,expected in d.get("input_sha256",{}).items():
            p=Path(name)
            if p not in hashes:hashes[p]=hashlib.sha256(p.read_bytes()).hexdigest()
            assert hashes[p]==expected,str(p)
            input_checks+=1
    learning=json.loads((HERE/"公開正解学習_実行記録.json").read_text())
    assert hashlib.sha256((HERE/"作業用/比較特徴量.npz").read_bytes()).hexdigest()==learning["input_cache_sha256"]
    gen=json.loads((HERE/"GenLRA_AIA_実行記録.json").read_text())
    assert hashlib.sha256((HERE.parent/"20261005_MIA初回_GenLRA比較/gen_lra.py").read_bytes()).hexdigest()==gen["source_genlra_sha256"]
    scoring=json.loads((HERE/"検証結果.json").read_text())
    assert scoring=={"official_function_checked":True,"cases":63}
    published=pd.read_csv(HERE/"集計/予備戦公開上位回答_対象別.csv")
    prior=pd.read_csv(ROOT/"予備戦/結果分析/集計/全チーム再採点.csv")
    x=published.merge(prior[["attacker","target","aia_all","aia_rare","aia_score"]],left_on=["attacker","team"],right_on=["attacker","target"],validate="one_to_one")
    assert len(x)==100
    differences=[float((x[a]-x[b]).abs().max()) for a,b in [("r_all","aia_all"),("r_rare","aia_rare"),("r_max","aia_score")]]
    assert max(differences)<1e-12
    permutations=pd.read_csv(HERE/"集計/同点修正_行順変更.csv")
    assert len(permutations)==880 and permutations.selected_changed.max()==0
    stable=pd.read_csv(HERE/"集計/同点修正_対象別.csv")
    original=pd.read_csv(HERE/"集計/背景参照感度_対象別.csv")
    a=stable[stable.method.eq("stable_matching/sex_quota/long75")].set_index("team")
    b=original[original.background.eq("prelim_5143")&original.method.eq("fixed/continuous/sex_quota/gate0.75")].set_index("team")
    assert len(a)==len(b)==21
    assert np.allclose(a.loc[b.index,"r_all"],b.r_all,rtol=0,atol=1e-12)
    nine=json.loads((HERE/"公式結果の追加推定.json").read_text())
    assert nine["target9_candidates"]==1 and nine["same_truth_and_phi_assumed"]
    for r,s in [(nine["target9_raw_R_round1"],.7322253012880818),(nine["target9_raw_R_round2"],.7462012678502552)]:
        assert abs(r-nine["target9_phi"]-s)<1e-12
    assert not list(HERE.rglob("F_aia.csv")) and not list(HERE.rglob("F_mia.csv"))
    assert not list(HERE.rglob("token.txt")) and not list(HERE.rglob("*.zip"))
    result=dict(code_records_checked=code_records,input_hash_checks=input_checks,unique_input_paths=len(hashes),
        official_score_cases=63,published_result_pairs=len(x),published_result_max_abs_diff=max(differences),
        learned_cache_hash_matches=True,frozen_genlra_hash_matches=True,permutation_cases=len(permutations),
        permutation_selection_changes=0,stable_sex_quota_matches_original_21_targets=True,
        target9_reconstruction_verified=True,submission_files_created=False,codabench_submitted=False,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/"保存結果の検証.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    run()
