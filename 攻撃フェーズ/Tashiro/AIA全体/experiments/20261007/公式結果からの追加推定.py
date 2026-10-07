"""2回の公式AIA結果から、対象9の補正前Rと床を絞る。

同一の課題・床が両回に使われたことを前提とする。
本人の所属や個別時間を復元せず、対象全体の正解人数だけを推定する。
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import hashlib, html, json, re
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
N=163
TARGETS=22
TOL=1e-12


def read_result(n):
    p=HERE/"参照"/f"第{n}回_detailed_results.html"
    d=json.loads(html.unescape(re.search(r"<pre>(.*?)</pre>",p.read_text(encoding="utf-8"),re.S).group(1)))
    assert len(d["aia_per_target"])==TARGETS
    return d,p


def hit_counts(value):
    result=[]
    for denominator in range(1,N+1):
        numerator=value*denominator
        if numerator.denominator!=1:continue
        c=N-denominator;m=c+int(numerator)
        if 0<=m<=N:result.append((m,c))
    return result


def candidates(first,second,universe):
    v1,v2=first,second
    delta=v2-v1
    floats=np.array(list(map(float,universe)))
    found=[]
    for i,f1 in enumerate(universe):
        r1=float(f1);r2=r1+delta
        if r1<v1-TOL or r2<v2-TOL or r2>1+TOL:continue
        start,end=np.searchsorted(floats,[r2-TOL,r2+TOL])
        for j in range(max(0,start-1),min(len(floats),end+2)):
            if abs(floats[j]-r2)<=TOL:
                found.append((f1,universe[j],r1-v1))
    return found


def run():
    d1,p1=read_result(1);d2,p2=read_result(2)
    v1,v2=[d["aia_per_target"]["9"] for d in [d1,d2]]
    rare_upper=[d["aia"]["r_time_rare"]*TARGETS for d in [d1,d2]]
    # 各希少Rは非負。単一対象の希少Rは全対象の合計以下。
    assert all(v>r for v,r in zip([v1,v2],rare_upper))
    # 床は非負なので、両回の対象9は全体Rがmaxの勝者。
    universe=sorted(set(Fraction(a,b) for b in range(1,N+1) for a in range(b+1)))
    found=candidates(v1,v2,universe)
    assert len(found)==1
    r1,r2,phi=found[0]
    assert abs(float(r2)-v2-phi)<TOL
    counts1,counts2=hit_counts(r1),hit_counts(r2)
    assert len(counts1)==len(counts2)==1
    rows=[]
    for n,d,r,counts,p in [(1,d1,r1,counts1[0],p1),(2,d2,r2,counts2[0],p2)]:
        m,c=counts
        mh=d["aia"]["p_m"]*TARGETS*N;ch=d["aia"]["p_c"]*TARGETS*N
        assert abs(mh-round(mh))<1e-8 and abs(ch-round(ch))<1e-8
        rest_m,rest_c=int(round(mh))-m,int(round(ch))-c
        rows.append(dict(round=n,target9_R_raw=float(r),target9_R_fraction=str(r),target9_phi=phi,
            target9_hit_member=m,target9_hit_control=c,target9_p_m=m/N,target9_p_c=c/N,
            other21_hit_member_total=rest_m,other21_hit_control_total=rest_c,other21_hit_gap=rest_m-rest_c,
            other21_p_m=rest_m/(21*N),other21_p_c=rest_c/(21*N),
            other21_mean_R_raw=(TARGETS*d["aia"]["r_time_all"]-float(r))/21,
            assumption_same_truth_and_phi_across_rounds=True,html_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    table=pd.DataFrame(rows)
    table.to_csv(HERE/"集計"/"対象9の補正前Rと他21対象.csv",index=False)
    goal=.1432
    prior=pd.read_csv(HERE/"集計"/"同点修正_加工群平均.csv").set_index("method").loc["stable_matching/sex_quota/long75","r_all"]
    result=dict(N_member_control=N,targets=TARGETS,same_truth_and_phi_assumed=True,
        rational_values_checked=len(universe),tolerance=TOL,target9_candidates=len(found),
        target9_raw_R_round1=float(r1),target9_raw_R_round2=float(r2),target9_phi=phi,
        rare_upper_bound_rounds=rare_upper,channel9_is_all_in_both_rounds=True,
        published_highest_raw_AIA=goal,required_other21_mean_R=(TARGETS*goal-float(r2))/21,
        raw_AIA_mean_gain_max_by_target9_alone=(1-float(r2))/TARGETS,
        final_AIA_mean_gain_max_by_target9_alone=(1-float(r2))/TARGETS,
        hypothetical_prior_mean_transferred_to_other21=float(prior),
        hypothetical_full_mean_if_transfer_holds=(float(r2)+21*float(prior))/22,
        hypothetical_transfer_is_not_main_score_estimate=True,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),submission_created=False)
    (HERE/"公式結果の追加推定.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    run()
