"""固定32参照集計の範囲・ハッシュ・頻度集計・意味を確認する。

参照の再分割数を新しい検証対象数とは数えない。分布表の補助メタデータを
意味が明確な列へ整える。R・的中人数・選択・seedは一切変更しない。
"""
from __future__ import annotations
import hashlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"集計"
HASHES={}


def read(path):
    raw=Path(path).read_bytes()
    HASHES[str(path)]=hashlib.sha256(raw).hexdigest()
    return pd.read_csv(io.BytesIO(raw),float_precision="round_trip")


def run():
    record=json.loads((HERE/"実行記録.json").read_text())
    assert hashlib.sha256((HERE/"分析.py").read_bytes()).hexdigest()==record["code_sha256"]
    assert hashlib.sha256((HERE/"固定プロトコル.json").read_bytes()).hexdigest()==record["protocol_sha256"]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in record["input_sha256"].items())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in record["imported_code_sha256"].items())
    expected_seeds=record["seeds"]
    assert len(expected_seeds)==32 and expected_seeds==list(range(20261008,20261008+32))
    real=read(OUT/"32参照_実C対象別.csv")
    null=read(OUT/"32参照_同性別BG_C全反復.csv")
    paired=read(OUT/"32参照_実Cと対照差_対象別.csv")
    summary=read(OUT/"32参照_20対象と18対象のseed別平均.csv")
    diagnostic=read(OUT/"同性別BG_Cの候補制限監査.csv")
    assert len(real)==21*32 and real.groupby("team").seed.nunique().eq(32).all()
    assert len(null)==20*32*20 and null.groupby(["team","seed"]).repeat.nunique().eq(20).all()
    assert len(paired)==20*32
    assert set(real.seed)==set(null.seed)==set(expected_seeds)
    assert diagnostic.sex_counts_match_actual_C.all() and diagnostic.pool_reference_disjoint.all()
    assert diagnostic.sample_n.eq(1049).all()
    assert diagnostic.selected.eq(245).all() and diagnostic.selected_bad_pairs.eq(0).all()
    assert ((diagnostic.sample_sex0+diagnostic.sample_sex1)==1049).all()
    assert diagnostic.groupby("team")[["sample_sex0","sample_sex1"]].nunique().eq(1).all().all()
    assert len(summary)==64 and summary.groupby("scope").seed.nunique().eq(32).all()
    assert summary[summary.scope.eq("all20")].targets.eq(20).all()
    assert summary[summary.scope.eq("processed18")].targets.eq(18).all()
    for row in summary.itertuples(index=False):
        use=paired[paired.seed.eq(row.seed)&~paired.team.isin([5] if row.scope=="all20" else [1,2,5])]
        assert abs(use.r_all_real.mean()-row.real_r_all)<1e-12
        assert abs(use.r_all_null.mean()-row.null_r_all)<1e-12
        assert abs(use.r_excess.mean()-row.r_excess)<1e-12
        assert abs(use.signed_hit_difference_real.mean()-row.real_signed_hit_difference)<1e-12
        assert abs(use.signed_hit_difference_null.mean()-row.null_signed_hit_difference)<1e-12
    freqpath=HERE/record["working_main_frequency_file"]
    assert freqpath.parent==HERE/"作業用"
    freq=np.load(freqpath,allow_pickle=False)
    assert freq["selected_in_32_ref"].shape==(23,326)
    assert np.array_equal(freq["seeds"],expected_seeds)
    assert (freq["selected_in_32_ref"]<=32).all()
    assert freq["selected_in_32_ref"].sum(1).tolist()==[245*32]*23
    assert freq["full_BG_selected"].sum(1).tolist()==[245]*23
    hist=read(OUT/"本戦_32参照での選択頻度の対象別性別分布.csv")
    freq_summary=read(OUT/"本戦_32参照での選択頻度の対象別性別要約.csv")
    histogram_cases=0
    for i,team in enumerate(freq["teams"]):
        for sex in ["all","0","1"]:
            mask=np.ones(326,bool) if sex=="all" else freq["sex"][i]==int(sex)
            values=freq["selected_in_32_ref"][i,mask]
            one=hist[(hist.team==team)&hist.sex.eq(sex)].sort_values("selected_in_32_ref")
            assert one.selected_in_32_ref.tolist()==list(range(33))
            assert one.q_rows.tolist()==[(values==n).sum() for n in range(33)]
            assert abs(one.q_share.sum()-1)<1e-12
            one_summary=freq_summary[(freq_summary.team==team)&freq_summary.sex.eq(sex)].iloc[0]
            assert one_summary.q_n==mask.sum()
            assert abs(one_summary.frequency_mean-(values/32).mean())<1e-12
            histogram_cases+=1
    # 元の補助列はboolean Falseで、独立対象数を表す列ではなかった。
    # スコア・seedを保ったまま、対象数と再分割の意味を明示する列へ修正する。
    distribution_path=OUT/"32参照の感度分布_独立検証区間ではない.csv"
    distribution=read(distribution_path)
    assert len(distribution)==16
    for row in distribution.itertuples(index=False):
        values=summary.loc[summary.scope.eq(row.scope),row.metric].to_numpy()
        actual=np.r_[values.mean(),values.std(ddof=1),np.quantile(values,[0,.05,.25,.5,.75,.95,1])]
        saved=np.array([row.mean,row.sd,row.minimum,row.p05,row.q25,row.median,row.q75,row.p95,row.maximum])
        assert np.max(np.abs(actual-saved))<1e-12
    if "independent_validation_target_count" in distribution:
        assert (~distribution.independent_validation_target_count.astype(bool)).all()
        distribution=distribution.drop(columns="independent_validation_target_count")
    distribution["fixed_attack_target_count"]=distribution.scope.map({"all20":20,"processed18":18})
    distribution["same_target_set_reused"]=True
    distribution["independent_validation_units_from_resampling"]=False
    distribution.to_csv(distribution_path,index=False)
    assert distribution.reference_resamplings.eq(32).all()
    assert not distribution.independent_validation_units_from_resampling.any()
    forbidden=[str(p) for p in HERE.rglob("*") if p.name in ["F_aia.csv","F_mia.csv","token.txt"] or p.suffix==".zip"]
    assert not forbidden
    share_csv=[]
    for path in OUT.glob("*.csv"):
        columns=pd.read_csv(path,nrows=0).columns
        assert "challenge_row_id" not in columns and "challenge_row_ids" not in columns
        share_csv.append(path.name)
    output=dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                analysis_code_sha256=record["code_sha256"],checked_primary_input_hashes=len(record["input_sha256"]),
                checked_import_hashes=len(record["imported_code_sha256"]),verification_input_sha256=HASHES,
                checked_real_cases=len(real),checked_null_cases=len(null),checked_seed_scopes=len(summary),
                checked_distribution_metrics=len(distribution),
                checked_main_histograms=histogram_cases,sharing_frequency_rows=False,
                distribution_metadata_normalized=True,mathematical_results_changed=False,
                final_distribution_sha256=hashlib.sha256(distribution_path.read_bytes()).hexdigest(),
                max_null_selected_bad_pairs=int(diagnostic.selected_bad_pairs.max()),share_csv_files=share_csv,
                forbidden_submission_files=forbidden)
    (HERE/"検証記録.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print({key:value for key,value in output.items() if key not in ["verification_input_sha256","share_csv_files"]})


if __name__=="__main__":
    run()
