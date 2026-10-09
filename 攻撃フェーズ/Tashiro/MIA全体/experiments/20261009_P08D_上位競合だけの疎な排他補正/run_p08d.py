"""P08-D: Sparse exclusivity correction for colliding P01 top-K candidates.

No hard assignment and no changes to fixed rare MIA. For each own ordinary
candidate i present in the top-K P01 ranked LISTS of two or more target
cohorts, choose the maximum *public-score* P01 as the leader; for any
other collided target whose P01 score lags by at least the fixed margin,
deduct the fixed penalty. The leader and all non-colliding target cells
remain UNMODIFIED. The ordering among candidates for each target may change
because other candidates were penalized.

This tests a precise hypothesis unlike P02's full 886-person matching or
generic cross-median/max-rival penalties. Record collision opportunities
before any released preliminary truth labels read, then report full
official-compatible TPR@FPR=.005/.01/.02 for target-inference scopes
all20 and without1_2 separately on four public A_bg backgrounds.

Official preliminary pool_ids is read only after ALL scores AND prediction
hashes are frozen. Research method design arose after extensive previous
use of same exposed preliminary scores and membership truth: even with
temporal label separation this is NOT independent generalization. Main
submission, main hidden truth, rare 5bit are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from openpyxl import load_workbook

BASE="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_全対象共同推定/run_experiment.py"
BASE_SHA="28b6a7c6106d89c9bcca6ca138c0aaae23c0b7c03e8dc299c34efd98c9bd4cb3"
OFFICIAL="公式データ/予備戦/結果/公開用結果表_20260909.xlsx"
BACKGROUNDS=("full","sub80_seed20261009","sub80_seed20261010","sub80_seed20261011")
TOP_K=(10,20,40)
MARGINS=(.02,.05,.10)
PENALTIES=(.05,.15)
TARGET_SCOPES=("all20","without1_2")
EVAL_FPR=.01
METHODS=("P01",)+tuple(
    f"sparse_top{k}_margin{margin:g}_penalty{penalty:g}"
    for k in TOP_K for margin in MARGINS for penalty in PENALTIES
)
assert len(METHODS)==19


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def writej(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def load_p01(root):
    p=root/BASE
    if sha(p)!=BASE_SHA:
        raise RuntimeError("Pinned P01 source drift")
    sp=importlib.util.spec_from_file_location("p08d_pin_P01",p)
    mod=importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def P01_score(base,cal,normal,selected):
    r=base.ranks(cal[normal][:,selected])
    return r-.1*base.rivals(r,np.median)


def eligibility(scores,k):
    # Score ties form a single equivalence group. If >K tied at boundary,
    # retaining the entire tie group ensures no artificial row-ID preference.
    return np.column_stack([
        rankdata(-scores[:,j],method="min")<=k for j in range(scores.shape[1])
    ])


def collision_analysis(scores,k):
    """Label-free collision structure with diagnostic counts."""
    mask=eligibility(scores,k)
    counts=mask.sum(axis=1)
    many=counts>=2
    leader=np.argmax(np.where(mask,scores,-np.inf),axis=1)
    leader_score=scores[np.arange(len(scores)),leader]
    gap=leader_score[:,None]-scores
    collided=mask & many[:,None]
    loser=collided & (np.arange(scores.shape[1])[None,:]!=leader[:,None])
    rows={
        "nominal_k":int(k),
        "n_candidates":len(scores),
        "targets":scores.shape[1],
        "actual_topk_cells":int(mask.sum()),
        "nonzero_topk_candidates":int((counts>=1).sum()),
        "collision_candidates_ge2":int(many.sum()),
        "collision_candidates_ge3":int((counts>=3).sum()),
        "collision_cells":int(collided.sum()),
        "collision_loser_cells":int(loser.sum()),
        "leader_tie_rows":int((many & ((np.isclose(scores,leader_score[:,None],rtol=0,atol=1e-13)&mask).sum(axis=1)>1)).sum()),
        "collision_margin_median":float(np.median(gap[loser])) if loser.any() else 0.,
        "collision_margin_p90":float(np.quantile(gap[loser],.9)) if loser.any() else 0.,
        "max_collision_counts":int(counts.max()),
    }
    return mask,many,leader,leader_score,loser,gap,rows


def sparse_penalty(scores,k,margin,penalty,parts=None):
    """Only ranked top-K loser cells with sufficiently large score gap change."""
    if parts is None:
        parts=collision_analysis(scores,k)
    mask,many,leader,leading,loser,gap,diag=parts
    eligible=loser & (gap>=margin-1e-12)
    updated=scores.copy()
    updated[eligible]-=penalty
    # Strict identity for every untouched cell and collision winners.
    assert np.array_equal(updated[~eligible],scores[~eligible])
    assert np.array_equal(updated[np.arange(len(scores)),leader],
                          scores[np.arange(len(scores)),leader])
    return updated,{
        **diag,"gap_cutoff":float(margin),"deduction":float(penalty),
        "penalized_cells":int(eligible.sum()),
        "candidates_with_penalty":int(eligible.any(axis=1).sum()),
        "targets_with_penalty":int(eligible.any(axis=0).sum()),
        "max_abs_score_change":float(np.max(np.abs(updated-scores))),
        "fraction_score_cells_unchanged":float((~eligible).mean()),
    }


def panel(base,cal,normal,indices):
    p01=P01_score(base,cal,normal,indices)
    scores={"P01":p01}
    stats=[]
    for k in TOP_K:
        parts=collision_analysis(p01,k)
        for margin in MARGINS:
            for penalty in PENALTIES:
                meth=f"sparse_top{k}_margin{margin:g}_penalty{penalty:g}"
                values,diag=sparse_penalty(p01,k,margin,penalty,parts)
                scores[meth]=values
                stats.append({"method":meth,**diag})
    assert set(scores)==set(METHODS)
    return scores,stats


def old_parts(root):
    wb=load_workbook(root/OFFICIAL,read_only=True,data_only=True)
    mia={};aia={}
    for row in list(wb["攻撃_標的別_MIA"].values)[1:]:
        if row[0]==5:mia[int(row[1])]=(float(row[4] or 0),float(row[5] or 0))
    for row in list(wb["攻撃_標的別_AIA"].values)[1:]:
        if row[0]==5:aia[int(row[1])]=float(row[9] or 0)
    assert len(mia)==len(aia)==20
    score=sum(max(*mia[t])+aia[t] for t in mia)/20
    assert abs(score-.3297953270354882)<1e-12
    return mia,aia,score


def run(root,out):
    if out.exists():
        raise FileExistsError("Refusing to overwrite results directory")
    out.mkdir(parents=True)
    start=time.perf_counter()
    base=load_p01(root)
    common,calib,folder,paths,b,a,qs,cs,teams,private,rare,normal=base.load_inputs(root)
    assert len(teams)==20 and int(normal.sum())==875
    active=[t for t in teams if t not in (1,2)]
    assert len(active)==18
    all_predictions=[]
    hashes=[]
    ops=[]
    for bg in BACKGROUNDS:
        if bg=="full":aa=a
        else:
            rng=np.random.default_rng(int(bg.split("seed")[1]))
            ix=np.sort(rng.choice(len(a),int(.8*len(a)),replace=False))
            aa=a.iloc[ix]
        _,cal=base.raw_scores(common,calib,b,aa,qs,cs,teams)
        scopes=[]
        for inf,target_ids in (("all20",teams),("without1_2",active)):
            indices=[teams.index(t) for t in target_ids]
            methods,diags=panel(base,cal,normal,indices)
            scopes.append((inf,target_ids,methods))
            for d in diags:
                ops.append({"background":bg,"inference":inf,**d})
            for method,mat in methods.items():
                hashes.append({
                  "background":bg,"inference":inf,"method":method,
                  "shape":list(mat.shape),
                  "sha256":hashlib.sha256(np.ascontiguousarray(
                               mat,dtype="<f8").tobytes()).hexdigest(),
                })
        all_predictions.append((bg,scopes))
        print("FROZEN COLLISIONS + PREDICTIONS:",bg,
              "background_n",len(aa),"elapsed",round(time.perf_counter()-start,2),flush=True)
    pd.DataFrame(ops).to_csv(out/"collision_opportunity.csv",index=False,float_format="%.17g")
    writej(out/"prediction_hashes.json",hashes)
    frozen=datetime.now(timezone.utc).isoformat()
    print("BEFORE TRUTH: all public-only collision structure and score hashes frozen. NOW score released prelim membership.",flush=True)
    truth,truth_audit,truthpaths=base.audit_truth(folder,b,private,rare,teams)
    assert truth_audit["self_rows_in_multiple_available_other_cohorts"]==0
    rows=[]
    for bg,scopes in all_predictions:
        for inf,target_ids,methods in scopes:
            for method,values in methods.items():
                confidence=base.confidence(values,normal,private,rare)
                assert np.all(confidence[private]==0)
                assert np.all(confidence[rare&~private]==.4)
                for j,t in enumerate(target_ids):
                    rows.append({
                      "background":bg,"inference":inf,"team":t,"method":method,
                      **base.metrics(confidence[:,j],truth[:,teams.index(t)])
                    })
    by=pd.DataFrame(rows)
    by.to_csv(out/"by_target.csv",index=False,float_format="%.17g")
    summaries=[]
    for (bg,inf),group in by.groupby(["background","inference"],sort=False):
        ref=group[group.method=="P01"].set_index("team")["tpr_0.01"]
        for method,g in group.groupby("method",sort=False):
            p=g.set_index("team")["tpr_0.01"]
            baseline=ref.reindex(p.index)
            diff=p-baseline
            summaries.append({
                "background":bg,"inference":inf,"method":method,
                "targets":len(g),"mean_tpr":float(p.mean()),
                "mean_delta_P01":float(diff.mean()),
                "TPR_ge010":int((p>=.1-1e-12).sum()),
                "new_ge010":int(((p>=.1-1e-12)&(baseline<.1-1e-12)).sum()),
                "lost_ge010":int(((p<.1-1e-12)&(baseline>=.1-1e-12)).sum()),
                "wins":int((diff>1e-12).sum()),
                "losses":int((diff< -1e-12).sum()),
                "band2plus":int((g.band>=2).sum()),
                "mean_tpr_0005":float(g["tpr_0.005"].mean()),
                "mean_tpr_002":float(g["tpr_0.02"].mean()),
            })
    S=pd.DataFrame(summaries)
    S.to_csv(out/"summary.csv",index=False,float_format="%.17g")
    expected={"all20":.08079772079772081,"without1_2":.025495409939854385}
    for inf,reference in expected.items():
        x=S[(S.background=="full")&(S.inference==inf)&(S.method=="P01")]
        if abs(x.mean_tpr.item()-reference)>1e-12:
            raise RuntimeError(f"P01 regression mismatch {inf}: {x.mean_tpr.item()}")

    # Summarize top-K actual redistributions; target identifiers are not used
    # to condition the penalty. Team-specific outcomes are read only now.
    st=by[by.inference=="without1_2"].assign(hit=lambda x:x["tpr_0.01"]>=.1-1e-12)
    stable=st.groupby(["team","method"],as_index=False).agg(
      hit_backgrounds=("hit","sum"),
      min_tpr=("tpr_0.01","min"),
      mean_tpr=("tpr_0.01","mean")
    )
    stable["hit_all4"]=stable.hit_backgrounds.eq(4)
    stable.to_csv(out/"threshold_stability.csv",index=False,float_format="%.17g")

    mia,aia,official=old_parts(root)
    virt=[]
    evaluated=by[(by.background=="full")&(by.inference=="all20")]
    for method,group in evaluated.groupby("method"):
        all_tp=group.set_index("team")["tpr_0.01"]
        # Old preliminary Team1/2 high scores retained, not subject to P08-D.
        proposal={t:(mia[t][0] if t in (1,2) else float(all_tp.loc[t]))
                  for t in mia}
        attack=sum(max(proposal[t],mia[t][1])+aia[t] for t in mia)/20
        virt.append({
            "method":method,"hypothetical_prelim_attack":float(attack),
            "delta_official_prelim":float(attack-official),
            "gap_official_prelim_3rd":.352845805596271-attack,
            "MIA_all_mean20":float(np.mean(list(proposal.values()))),
            "TPR010_teams20":sum(v>=.1-1e-12 for v in proposal.values()),
            "Team1_2_old_official_preserved":True,
            "KDEA_KDEB_not_used":True,
        })
    V=pd.DataFrame(virt)
    V.to_csv(out/"virtual_attack_P01_only.csv",index=False,float_format="%.17g")
    writej(out/"protocol.json",{
        "experiment":"P08D topK contender-only sparse exclusivity score penalty on P01",
        "topk":TOP_K,"minimum_leader_loser_gaps":MARGINS,
        "penalties":PENALTIES,"n_methods":len(METHODS),
        "sparse_semantics":"only loser of a >=2-cohort candidate collision in per-target topK, gap >= margin; leader/noncollided unchanged",
        "tied_topK":"retain whole tie group with rankdata(method=min)",
        "primary_inference":"without1_2; excluded from other-target median AND evaluation",
        "secondary_inference":"all20",
        "backgrounds":BACKGROUNDS,
        "score_hashes_frozen_utc":frozen,
        "official_prelim_truth_only_after_freeze":True,
        "public_truth_audit_max_other_memberships":truth_audit["self_max_other_cohorts"],
        "hypothetical_score_reference":"P01-only virtual attack, not P04 KDE overlay",
        "AIA_rare_official_frozen":True,
        "old_Team1_2_official_full_TPR_preserved_in_virtual":True,
        "past_prelim_membership_truth_repeatedly_examined":True,
        "external_main_test":False,
        "no_main_inputs":True,"no_main_submission":True,
        "no_individual_predictions_saved":True,
        "input_sha256":{str(p.relative_to(root)):sha(p) for p in
           paths+[root/BASE,root/OFFICIAL,Path(__file__).resolve()]},
        "truth_files_sha256_after_freeze":{str(p.relative_to(root)):sha(p) for p in truthpaths},
        "versions":{"numpy":np.__version__,"pandas":pd.__version__,
                    "python":platform.python_version()},
    })
    print("OPPORTUNITY:")
    print(pd.DataFrame(ops)[lambda x:(x.background=="full")&
                            (x.inference=="without1_2")&
                            (x.method.str.contains("margin0.02_penalty0.05"))].to_string(index=False))
    print("PRIMARY FULL:")
    print(S[(S.background=="full")&(S.inference=="without1_2")].to_string(index=False))
    print("VIRTUAL P01 ONLY:")
    print(V.sort_values("hypothetical_prelim_attack",ascending=False).head(10).to_string(index=False))
    print("seconds",time.perf_counter()-start)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.root.resolve(),a.out.resolve())
