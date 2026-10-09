"""Project PRELIMINARY public AIA-imprint grouping onto MAIN public C/AIA targets.

No B_self data is read; no membership label, score matrix or submission created.
This is a public-data applicability check, not a main-round TPR evaluation.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

NUM = ("age", "SBP", "TG", "HDL", "ALT")
COMMON_RELPATH = Path("攻撃フェーズ/Tashiro/MIA全体/experiments/20261007/分析.py")
COMMON_SHA = "25cdf97bee3adbda9a5fd8b0793fa0c3f98102366bd1944386fb619d04569aad"


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compute(root, out):
    if out.exists():
        raise FileExistsError("Do not overwrite: " + str(out))
    out.mkdir(parents=True)
    common_path = root / COMMON_RELPATH
    if digest(common_path) != COMMON_SHA:
        raise RuntimeError("Public-distance definition has changed")
    sys.path.insert(0, str(common_path.parent))
    common = importlib.import_module("分析")
    prior = root / "攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/group_thresholds.json"
    group_params = json.loads(prior.read_text(encoding="utf-8"))
    attack = root / "公式データ/PWSCup2026_final_attack"
    home_dirs = sorted((root / "公式データ").glob("PWSCup2026_final_team28_*"))
    if len(home_dirs) != 1:
        raise RuntimeError("Missing unique official main-round Team28 distribution")
    abg_path = home_dirs[0] / "A_bg.csv"
    bg = pd.read_csv(abg_path, float_precision="round_trip")
    tcols = json.loads((attack / "mia_columns.json").read_text())["target_columns"]
    targets = sorted(set(map(int, tcols)) - {28})
    assert len(targets) == 22
    scales = common.scales(bg, NUM)
    sample_rng = np.random.default_rng(20261009)
    bgindex = np.sort(sample_rng.choice(len(bg), 1049, replace=False))
    BGA = common.numbers(bg.iloc[bgindex], NUM) / scales
    rows = []
    originals = [abg_path, attack / "mia_columns.json", common_path, prior]
    threshold = float(group_params["imprint_median"])
    for team in targets:
        qpath = attack / f"aia_challenge_{team}.csv"
        cpath = attack / f"C_{team}.csv"
        originals.extend([qpath,cpath])
        q = pd.read_csv(qpath, float_precision="round_trip")
        c = pd.read_csv(cpath, float_precision="round_trip")
        assert len(q) == 326 and len(c) == 1049
        home = set(q.prefecture.astype(str).unique())
        if len(home) != 1:
            raise RuntimeError(f"Target {team} AIA not single prefecture")
        region=next(iter(home))
        Q=common.numbers(q, NUM)/scales
        C=common.numbers(c, NUM)/scales
        q2c=cdist(Q,C,"sqeuclidean").min(axis=1)
        q2bg=cdist(Q,BGA,"sqeuclidean").min(axis=1)
        score=float(np.log((np.median(q2bg)+1e-6)/(np.median(q2c)+1e-6)))
        homefrac=float(c.prefecture.astype(str).eq(region).mean())
        uniq=float(np.median([c[col].nunique(dropna=False)/len(c) for col in NUM]))
        rows.append({
            "team":team,
            "aia_home":region,
            "main_imprint_score":score,
            "prelim_imprint_cutoff":threshold,
            "margin_from_prelim_cutoff":score-threshold,
            "imprint_high_prelim_threshold":int(score>=threshold),
            "C_home_fraction":homefrac,
            "C_median_numeric_unique_fraction":uniq,
            "quantized_C":int(uniq<.90),
            "Q_to_C_nn_median":float(np.median(q2c)),
            "Q_to_bg_nn_median":float(np.median(q2bg)),
            "C_rows":len(c),"AIA_rows":len(q)
        })
        print(f"MAIN public target {team}: imprint={score:.6f}",flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(out / "main_public_group_projection.csv",index=False,float_format="%.17g")
    payload={
        "status":"Unlabeled applicability diagnostics, NOT main attack performance",
        "source":"official main public C/AIA/A_bg",
        "main_target_count":len(targets),
        "preliminary_cutoff_public_only":threshold,
        "imprint_high_main_teams":result.loc[result.imprint_high_prelim_threshold==1,"team"].astype(int).tolist(),
        "imprint_low_main_teams":result.loc[result.imprint_high_prelim_threshold==0,"team"].astype(int).tolist(),
        "main_above_cutoff":int(result.imprint_high_prelim_threshold.sum()),
        "quantized_main":int(result.quantized_C.sum()),
        "within_005_of_boundary":result.loc[result.margin_from_prelim_cutoff.abs()<.05,"team"].astype(int).tolist(),
        "main_imprint_median":float(result.main_imprint_score.median()),
        "prelim_imprint_median":threshold,
        "group_shift":"Public AIA/C distribution may differ across preliminary and main rounds",
        "main_membership_truth_available":False,
        "main_TPR_not_computed":True,
        "main_predictions_not_generated":True,
        "token_not_read":True,
        "self_B_not_read":True,
        "input_hashes":{str(p.relative_to(root)):digest(p) for p in originals}
    }
    (out / "main_public_diagnostics.json").write_text(
        json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:payload[k] for k in [
        "main_target_count","main_above_cutoff","imprint_high_main_teams",
        "imprint_low_main_teams","within_005_of_boundary"]},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    a=parser.parse_args()
    compute(a.root.resolve(),a.output_dir.resolve())
