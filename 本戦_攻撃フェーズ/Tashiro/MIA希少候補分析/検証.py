"""保存された全1,584ケースを、cdistを使わず原票から再計算する。"""
from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from 共通設定 import configure, checked_paths
NUM = ["age","BMI","SBP","TG","HDL","ALT","FPG"]
QI = ["age","SBP","TG","HDL","ALT","sex","prefecture"]
TOL = 1e-12


def read_raw(path):
    with path.open(newline="",encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return pd.DataFrame([{k: v if k == "prefecture" else float(v) for k,v in r.items()} for r in rows])


def trans(df):
    a = df[NUM].to_numpy(float).copy()
    a[:,3] = np.log1p(a[:,3])
    a[:,5] = np.log1p(a[:,5])
    return a


def main():
    options = configure("保存された値を独立した数値演算で検証します")
    output = options.output_dir
    paths, _ = checked_paths(options)
    run = json.loads((output/"実行記録.json").read_text())
    assert hashlib.sha256((HERE/"分析.py").read_bytes()).hexdigest() == run["analysis_sha256"]
    assert hashlib.sha256((HERE/"レポート.py").read_bytes()).hexdigest() == run["report_code_sha256"]
    for name, h in run["support_sha256"].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == h
    for path,h in run["input_sha256"].items():
        assert hashlib.sha256(paths[path].read_bytes()).hexdigest() == h
    bpath = paths["B_self.csv"]
    b = read_raw(bpath)
    q = read_raw(paths["aia_challenge_28.csv"])
    bg = read_raw(paths["A_bg.csv"])
    qkeys = set(q[QI].itertuples(index=False,name=None))
    private = b[(b.is_rare==1)&b[QI].apply(tuple,axis=1).isin(qkeys)]
    assert set(private.record_id)==set(run["known_nonmember_record_ids"])
    candidates = set(b.loc[(b.is_rare==1)&(~b.record_id.isin(private.record_id)),"record_id"])
    assert candidates==set(run["candidate_record_ids"])
    scale = np.diff(np.percentile(trans(bg),[25,75],axis=0),axis=0)[0]/1.349
    assert np.allclose(scale,run["background_IQR_div_1_349"],atol=0,rtol=1e-14)
    detail = pd.read_csv(output/"集計/全設定診断.csv",float_precision="round_trip")
    pairs = pd.read_csv(output/"集計/候補242組.csv",float_precision="round_trip")
    assert len(detail)==1584 and len(pairs)==242
    assert pairs.membership.eq("未確定").all()
    assert set(map(tuple,pairs[["record_id","team"]].to_numpy()))=={(r,t) for r in candidates for t in run["target_teams"]}
    b_arr, bg_arr = trans(b), trans(bg)
    max_error, max_tie_spec_span, cases, ties = 0.,0.,0,0
    for team, team_data in detail.groupby("team"):
        c = read_raw(paths[f"C_{int(team)}.csv"])
        c_arr = trans(c)
        for method, group in team_data.groupby("method"):
            cat = [] if method == "numeric7_L2" else ["sex","smoking"]
            l1 = method == "mixed10_L1"
            if l1: cat += ["prefecture"]
            denominator = 7+len(cat)
            def dist(point, reference, point_values, reference_values):
                diff = (reference_values-point_values)/scale
                value = np.sum(np.abs(diff) if l1 else diff**2,axis=1)
                for col in cat:
                    value += reference[col].to_numpy()!=point[col]
                value /= denominator
                return value if l1 else np.sqrt(value)
            for z in group.itertuples():
                pos = int(np.flatnonzero(b.record_id.to_numpy()==z.record_id)[0])
                point = b.iloc[pos]
                dc = dist(point,c,b_arr[pos],c_arr)
                sorted_values=np.sort(dc)
                for actual, expected in [(sorted_values[0],z.d1),(sorted_values[1],z.d2),(sorted_values[4],z.d5)]:
                    err=abs(actual-expected);max_error=max(max_error,err);assert err<2e-12
                nearest=int(z.nearest_C_row0)
                assert dc[nearest]<=dc.min()+TOL
                back = dist(c.iloc[nearest],b,c_arr[nearest],b_arr)
                rank=int((back<z.d1-TOL).sum()+1)
                assert rank==z.C_to_B_rank_min
                back[pos]=np.inf
                other=float(back.min())
                assert abs(other-z.nearest_competitor_distance)<2e-12
                sp=float(np.log((other+TOL)/(dc[nearest]+TOL)))
                max_error=max(max_error,abs(sp-z.specificity));assert abs(sp-z.specificity)<2e-11
                db = dist(point,bg,b_arr[pos],bg_arr)
                assert abs(db.min()-z.query_to_A_bg_distance)<2e-12
                cb = dist(c.iloc[nearest],bg,c_arr[nearest],bg_arr)
                assert int((cb<z.d1-TOL).sum())==z.A_bg_rows_closer_to_this_C
                mask=(c.sex==point.sex)&(c.prefecture==point.prefecture)
                assert mask.sum()==z.C_same_sex_pref_count
                if mask.any(): assert abs(dc[mask].min()-z.same_sex_pref_d1)<2e-12
                else: assert pd.isna(z.same_sex_pref_d1)
                nearest_ties=np.flatnonzero(np.isclose(dc,dc.min(),atol=TOL,rtol=0))
                assert len(nearest_ties)==z.nearest_tie_count
                if len(nearest_ties)>1:
                    ties+=1;allsp=[]
                    for cp in nearest_ties:
                        back=dist(c.iloc[cp],b,c_arr[cp],b_arr);back[pos]=np.inf
                        allsp.append(np.log((back.min()+TOL)/(dc[cp]+TOL)))
                    max_tie_spec_span=max(max_tie_spec_span,float(np.ptp(allsp)))
                for label, cols in [("exact13_C_rows",list(c.columns)),("exact_num7_C_rows",NUM),("exact_mixed9_C_rows",NUM+["sex","smoking"])]:
                    assert int((c[cols]==point[cols]).all(axis=1).sum())==getattr(z,label)
                cases+=1
            nonmembers=group[group.role=="known_nonmember"]
            assert len(nonmembers)==13
            for z in group[group.role=="candidate"].itertuples():
                assert int((nonmembers.d1<=z.d1+TOL).sum())==z.known13_distance_le_count
                assert int((nonmembers.specificity>=z.specificity-TOL).sum())==z.known13_specificity_ge_count
                assert int(((nonmembers.d1<=z.d1+TOL)&(nonmembers.specificity>=z.specificity-TOL)).sum())==z.known13_both_at_least_as_strong_count
    # 同率最短のC行が複数あるケースでも固有性が変わらないことを確認。
    assert max_tie_spec_span<2e-11,max_tie_spec_span
    primary=detail[(detail.method=="mixed9_L2")&(detail.role=="candidate")].set_index(["record_id","team"])
    p=pairs.set_index(["record_id","team"])
    assert np.allclose(p.primary_d1,primary.loc[p.index,"d1"],rtol=0,atol=1e-14)
    assert np.allclose(p.primary_specificity,primary.loc[p.index,"specificity"],rtol=0,atol=1e-14)
    # 自チーム28の72ケースも、上の独立距離関数で確認する。
    own = pd.read_csv(output/"集計/自チーム28検証.csv", float_precision="round_trip")
    assert len(own) == 72 and own.membership.eq("既知会員").all()
    c = read_raw(paths["C_28.csv"])
    c_arr = trans(c)
    self_cases = 0
    for method, group in own.groupby("method"):
        cat = [] if method == "numeric7_L2" else ["sex", "smoking"]
        l1 = method == "mixed10_L1"
        if l1: cat += ["prefecture"]
        denominator = 7 + len(cat)
        for z in group.itertuples():
            pos = int(np.flatnonzero(b.record_id.to_numpy() == z.record_id)[0])
            assert z.other_team_candidate == (z.record_id in candidates)
            dc = dist(b.iloc[pos], c, b_arr[pos], c_arr)
            cp = int(z.nearest_C_row0)
            assert abs(dc.min()-z.d1) < 2e-12 and dc[cp] <= dc.min()+TOL
            back = dist(c.iloc[cp], b, c_arr[cp], b_arr)
            assert int((back < z.d1-TOL).sum()+1) == z.C_to_B_rank_min
            back[pos] = np.inf
            assert abs(back.min()-z.nearest_competitor_distance) < 2e-12
            expected = np.log((back.min()+TOL)/(dc[cp]+TOL))
            assert abs(expected-z.specificity) < 2e-11
            self_cases += 1
    selections = json.loads((HERE/"候補選定.json").read_text())
    summary = pd.read_csv(output/"集計/チーム別候補.csv", dtype={"candidate_ids": str})
    assert len(summary) == 23
    for s in selections:
        row = summary[summary.team == s["team"]].iloc[0]
        assert row.candidate_ids == ";".join(map(str, s["candidate_ids"]))
        assert set(s["candidate_ids"]) <= candidates
    classes = pd.read_csv(output/"集計/希少24行の区分.csv")
    assert list(classes.columns) == ["record_id", "is_rare", "区分", "自課題ID"]
    assert set(classes.record_id) == set(b.loc[b.is_rare==1, "record_id"])
    text=(output/"説明.md").read_text()
    for link in re.findall(r"\]\(([^)]+)\)",text):
        if link == "検証結果.json":
            continue  # この実行が全検査に成功した後、末尾で作成する。
        assert link.startswith("https://") or (output/link).exists(),link
    for path,h in run["input_sha256"].items():
        assert hashlib.sha256(paths[path].read_bytes()).hexdigest()==h
    result={"status":"passed","independent_cases":cases,"method":"stdlib CSV + explicit numpy arithmetic; no scipy cdist, no analysis-module import",
            "max_numeric_error":max_error,"nearest_tie_cases":ties,"max_specificity_span_over_tied_C":max_tie_spec_span,
            "input_hashes_verified":len(run["input_sha256"]),"unknown_pairs":len(pairs),
            "self_team_cases": self_cases, "plot_visual_check": "See publication verification; not checked by this script",
            "output_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((output/"集計").glob("*.csv"))}}
    (output/"検証結果.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="output_sha256"},ensure_ascii=False))


if __name__=="__main__":
    main()
