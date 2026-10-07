"""保存集計、基準再現、入力とコードのハッシュを検証する。"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OLD=HERE.parent.parent/"20261007_AIA全体向上の分析"


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    record=json.loads((HERE/"実行記録.json").read_text())
    assert record["code_sha256"]==sha(HERE/"分析.py")
    for name,expected in record["reference_code_sha256"].items():assert sha(OLD/name)==expected
    for path,expected in record["input_sha256"].items():assert sha(Path(path))==expected
    assert not any("B_self" in path or "アーカイブ" in path for path in record["input_sha256"])
    assert record["official_score_check"]==dict(official_function_checked=True,cases=63)
    assert record["submission_created"] is False and record["codabench_submitted"] is False
    assert record["plan"]==json.loads((HERE/"事前固定候補.json").read_text())
    out=HERE/"集計"
    table=pd.read_csv(out/"予備戦_対象別.csv")
    null=pd.read_csv(out/"独立背景対照_全反復.csv")
    average=pd.read_csv(out/"予備戦_20対象と18対象平均.csv")
    assert len(table)==21*5 and len(null)==20*21*5
    metrics=["r_all","r_rare","r_max","p_m","p_c"]
    for key in metrics:assert table[key].between(0,1).all() and null[key].between(0,1).all()
    for group,excluded in [("20対象_自5除外",[5]),("18対象_1_2_5除外",[1,2,5])]:
        calc=table[~table.team.isin(excluded)].groupby("method")[metrics].mean().sort_index()
        saved=average[average.group.eq(group)].set_index("method")[metrics].sort_index()
        assert np.max(np.abs(calc.to_numpy()-saved.to_numpy()))<1e-12
    previous=pd.read_csv(OLD/"集計/同点修正_対象別.csv")
    a=previous[previous.method.eq("stable_matching/sex_quota/long75")].sort_values("team")
    b=table[table.method.eq("baseline")].sort_values("team")
    assert a.team.tolist()==b.team.tolist()
    assert np.max(np.abs(a[metrics].to_numpy()-b[metrics].to_numpy()))<1e-12
    p=pd.read_csv(out/"行順変更.csv")
    assert len(p)==44*4*5 and p.changed.sum()==0
    assert p[p.kind.eq("other_C")].max_other_score_difference.max()==0
    forbidden=[p.name for p in HERE.rglob("*") if p.is_file() and
               (p.name in {"F_aia.csv","F_mia.csv","token.txt"} or p.suffix==".zip")]
    assert not forbidden
    result=dict(input_hashes_verified=len(record["input_sha256"]),reference_codes_verified=4,
        score_rows=len(table)+len(null),same_baseline_as_yesterday=True,permutation_comparisons=len(p),
        changed_rows=0,mean_summaries_consistent=True,official_score_comparisons=63,
        no_B_self_or_archive_read=True,submission_artifacts_found=forbidden,official_phi_reproduced=False,
        verification_code_sha256=sha(Path(__file__)))
    (HERE/"保存結果の検証.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":run()
