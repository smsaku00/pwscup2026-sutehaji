"""Unlabeled main-round public applicability diagnostics for the P05 hypotheses.

Only PUBLIC main A_bg, C_0.. and aia_challenge_0.. plus previous PUBLIC
P04 grouping are loaded. No B_self, membership labels, AIA hidden controls,
F_mia, submitted confidences, or CodaBench feedback is ever accessed.

IMPORTANT DOMAIN SHIFT: Preliminary copy-rate was matched against A1 + legacy
starter-kit A2. Main has only the officially supplied A_bg. Thus a low main
copy-rate is NOT evidence of a low A1+A2 copy-rate and cannot validate the gate.
Public Q nearest RMS uses main A_bg SD, while preliminary threshold is frozen.
These are a public-data transfer-screen only, NOT real membership TPR.
"""
from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Mirrors the aggregate-only P05 public features; no private B.
NUM5=("age","SBP","TG","HDL","ALT")
FULL13=("age","BMI","SBP","TG","HDL","ALT","FPG","time",
        "sex","prefecture","smoking","onset","death")
GROUP="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_TPR010到達と加工特徴転移検証/p04_main_public/main_P04_public_target_families.csv"
PRELIM_FEATURE="攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_非学習方式と加工特徴_背景耐性/公開加工特徴.csv"
ATTACK="公式データ/PWSCup2026_final_attack"
N_TARGETS=22
RMS_LIMIT=.05


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def exact_keys(data):
    return pd.MultiIndex.from_frame(data[list(FULL13)].round(8))


def matched_q_c_nearest(q,c,scale):
    v=np.full(len(q),np.inf)
    for sex in sorted(q.sex.unique()):
        idx=q.sex.eq(sex).to_numpy()
        subset=c[c.sex==sex]
        if not len(subset):continue
        qx=q.loc[idx,list(NUM5)].to_numpy(float)/scale
        cx=subset[list(NUM5)].to_numpy(float)/scale
        distance=cdist(qx,cx,"euclidean").min(axis=1)/np.sqrt(len(NUM5))
        v[idx]=distance
    if not np.isfinite(v).all():
        raise RuntimeError("Some query sex absent from C")
    return v


def run(root,out):
    if out.exists():raise FileExistsError(str(out))
    out.mkdir(parents=True)
    attack=root/ATTACK
    home=list((root/"公式データ").glob("PWSCup2026_final_team28_*"))
    assert len(home)==1
    refpath=home[0]/"A_bg.csv"
    ref=pd.read_csv(refpath,float_precision="round_trip")
    std=ref[list(NUM5)].std(ddof=1).to_numpy(float)
    assert np.isfinite(std).all() and (std>0).all()
    public_ref_keys=exact_keys(ref)
    original_path=root/GROUP
    p04=pd.read_csv(original_path)
    assert len(p04)==N_TARGETS and len(set(p04.team))==N_TARGETS
    prelim_path=root/PRELIM_FEATURE
    prelim=pd.read_csv(prelim_path,encoding="utf-8-sig")
    PRELIM_LIMIT=float(prelim[~prelim.team.isin([1,2,12])]
                       ["Q_sexmatched_QI5_RMS_median"].median())
    assert abs(PRELIM_LIMIT-.19859443409635053)<1e-12
    result=[]
    sources={str(refpath.relative_to(root)):sha(refpath),
             GROUP:sha(original_path),PRELIM_FEATURE:sha(prelim_path)}
    for r in p04.itertuples(index=False):
        t=int(r.team)
        cpath=attack/f"C_{t}.csv";qpath=attack/f"aia_challenge_{t}.csv"
        sources[str(cpath.relative_to(root))]=sha(cpath)
        sources[str(qpath.relative_to(root))]=sha(qpath)
        c=pd.read_csv(cpath,float_precision="round_trip")
        q=pd.read_csv(qpath,float_precision="round_trip")
        assert len(c)==1049 and len(q)==326
        matches=int(exact_keys(c).isin(public_ref_keys).sum())
        frac=matches/len(c)
        closest=matched_q_c_nearest(q,c,std)
        nnear=int((closest<RMS_LIMIT).sum())
        qmed=float(np.median(closest))
        unique=float(np.median([c[k].nunique(dropna=False)/len(c)
                                for k in NUM5]))
        method=str(r.public_P04_family)
        copy=(frac>=.90)
        close=(method=="KDE_A" and nnear>=2 and unique>=.90 and
               qmed<PRELIM_LIMIT)
        selected=("P01" if copy else "P03_Cnn" if close else method)
        result.append({
            "team":t,
            "base_P04_family":method,
            "main_public_A_bg_only_full13_copy_rows":matches,
            "main_public_A_bg_only_full13_copy_fraction":frac,
            "full13_copy_ge90pct":int(copy),
            "main_sexmatched_QC_close_005":nnear,
            "main_sexmatched_QC_RMS_median":qmed,
            "fixed_prelim_17_public_QC_RMS_median":PRELIM_LIMIT,
            "main_C_median_numeric5_unique_fraction":unique,
            "P05_close_proxy_flag":int(close),
            "P05_combined_PUBLIC_only_family":selected,
            "main_imprint_margin":float(r.imprint_margin),
            "main_home_margin":float(r.home_margin)
        })
        print(f"main public P05 group {t}: {method}→{selected}, copying {matches}/{len(c)}",flush=True)
    df=pd.DataFrame(result).sort_values("team")
    df.to_csv(out/"main_public_P05_proxy.csv",index=False,float_format="%.17g")
    from collections import Counter
    report={
        "scope":"PUBLIC main round feature projection, no personal outcomes, no TPR",
        "main_target_count":N_TARGETS,
        "public_P04_counts":dict(Counter(df.base_P04_family)),
        "P05_proxy_counts":dict(Counter(df.P05_combined_PUBLIC_only_family)),
        "new_public_background_copy_flags":df.loc[df.full13_copy_ge90pct==1,"team"].tolist(),
        "new_public_nearest_Q_flags":df.loc[df.P05_close_proxy_flag==1,"team"].tolist(),
        "changed_public_families":df.loc[df.base_P04_family!=df.P05_combined_PUBLIC_only_family,
                                         "team"].tolist(),
        "copy_reference_WARNING":"Prelim public copy fraction used A1 PLUS legacy A2; main only released A_bg is available. Negative matches do not prove absence of legacy-data copying.",
        "QC_nearest_reference":"sex-matched public QI5 RMS using MAIN A_bg raw per-column SD, prelim threshold frozen at median17",
        "confidence":"public-feature hypothesis screening; NOT independent validation",
        "unknown_main_TPR":True,
        "no_main_B_self":True,"no_membership_truth":True,
        "no_hidden_AIA_labels":True,"no_F_mia_created":True,
        "no_submission_changed":True,
        "input_sha256":sources
    }
    (out/"main_public_diagnostics.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in [
        "main_target_count","public_P04_counts","P05_proxy_counts",
        "new_public_background_copy_flags","new_public_nearest_Q_flags",
        "changed_public_families"]},ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.root.resolve(),a.out.resolve())
