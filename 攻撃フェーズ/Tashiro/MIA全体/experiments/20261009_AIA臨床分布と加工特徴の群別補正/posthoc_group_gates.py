"""Post-selection target-group gates, reusing already frozen per-target MIA scores.

AIA-imprint high grouping was public-feature-only; deciding to use the Cnn method
in that group followed inspection of exploratory preliminary MIA outcomes.
Thus ALL gated results are post-hoc, NOT an independent test.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

CHOICES = {
    "P01": ("P01", "always"),
    "Cnn_all": ("Cnn_add010", "always"),
    "Cnn_imprint_high": ("Cnn_add010", "imprint_high"),
    "Cnn_imprint_low_control": ("Cnn_add010", "imprint_low"),
    "Cnn_imprint_high_LOO_public": ("Cnn_add010", "imprint_high_loo"),
    "Cnn_C_home_high_control": ("Cnn_add010", "home_high"),
    "Cnn_quantized_C_control": ("Cnn_add010", "quantized"),
    "AIA_residual_imprint_high": ("AIA_residual_add010", "imprint_high")
}
MEASURES = ("tpr_0.005","tpr_0.01","tpr_0.02","auc",
            "top10_expected_hits","top12_expected_hits","top25_expected_hits","band")


def file_hash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def choose(method, feature, peer_imprints):
    candidate, rule = CHOICES[method]
    if rule == "always":
        return candidate
    is_high = bool(feature.aia_imprint_high)
    if rule == "imprint_high": active = is_high
    elif rule == "imprint_low": active = not is_high
    elif rule == "home_high": active = bool(feature.c_home_above_median)
    elif rule == "quantized": active = bool(feature.quantized_c)
    elif rule == "imprint_high_loo":
        active = feature.AIA_imprint_log_BG_to_C_median_Q_NN >= np.median(peer_imprints)
    else: raise ValueError(rule)
    return candidate if active else "P01"


def calculate(experiment_dir, outdir):
    if outdir.exists():
        raise FileExistsError("Cannot overwrite output " + str(outdir))
    outdir.mkdir(parents=True)
    original = experiment_dir / "results"
    scores = pd.read_csv(original / "by_target.csv")
    desc = pd.read_csv(original / "public_target_groups.csv").set_index("team")
    targets = sorted(t for t in desc.index.tolist() if t not in (1,2))
    assert len(targets)==18 and desc.loc[targets,"aia_imprint_high"].sum()==9
    records=[]
    for (view,infer,team), subset in scores[scores.team.isin(targets)].groupby(
        ["background","inference","team"],sort=False
    ):
        lookup=subset.set_index("method")
        assert len(lookup)==12
        peer=desc.loc[[t for t in targets if t!=team],
                      "AIA_imprint_log_BG_to_C_median_Q_NN"].to_numpy(float)
        for method in CHOICES:
            source=choose(method,desc.loc[team],peer)
            row=lookup.loc[source]
            records.append({"background":view,"inference":infer,"team":team,
                            "method":method,"source_method":source,
                            **{k:float(row[k]) for k in MEASURES}})
    out=pd.DataFrame(records)
    assert len(out)==(4*18+18)*len(CHOICES)
    groupstats=[]
    for (view,infer), subset in out.groupby(["background","inference"],sort=False):
        b=subset[subset.method=="P01"].set_index("team")["tpr_0.01"]
        for method,g in subset.groupby("method",sort=False):
            s=g.set_index("team")["tpr_0.01"].reindex(targets)
            d=s-b.reindex(targets)
            x=d.to_numpy(float)
            groupstats.append({
                "background":view,"inference":infer,"evaluation":"without1_2",
                "method":method,"targets":len(s),"mean_tpr":float(s.mean()),
                "mean_delta_P01":float(d.mean()),
                "wins":int((x>1e-12).sum()),"ties":int((abs(x)<=1e-12).sum()),
                "losses":int((x< -1e-12).sum()),
                "min_target_LOO_mean_delta":float(min((x.sum()-v)/17 for v in x)),
                "max_target_LOO_mean_delta":float(max((x.sum()-v)/17 for v in x)),
                "applied_target_count":int((g.source_method!="P01").sum()),
                "mean_tpr_FPR005":float(g["tpr_0.005"].mean()),
                "mean_tpr_FPR020":float(g["tpr_0.02"].mean()),
                "band2plus":int((g.band>=2).sum())
            })
    summary=pd.DataFrame(groupstats)
    for inf,expected in [("without1_2",0.025495409939854385),
                         ("all20",0.029775245330800885)]:
        z=summary[(summary.background=="full")&
                  (summary.inference==inf)&(summary.method=="P01")].mean_tpr.item()
        assert abs(z-expected)<1e-12,(inf,z,expected)
    stability=[]
    primary=out[out.inference=="without1_2"]
    for method in CHOICES:
        for team in targets:
            p=primary[(primary.team==team)&(primary.method=="P01")].set_index("background")["tpr_0.01"]
            v=primary[(primary.team==team)&(primary.method==method)].set_index("background")["tpr_0.01"]
            d=v-p.reindex(v.index)
            assert len(d)==4
            stability.append({
                "team":team,"method":method,
                "aia_imprint_high":int(desc.loc[team,"aia_imprint_high"]),
                "improved_backgrounds":int((d>1e-12).sum()),
                "worsened_backgrounds":int((d< -1e-12).sum()),
                "min_delta":float(d.min()),"max_delta":float(d.max()),
                "mean_delta":float(d.mean())
            })
    out.to_csv(outdir/"gated_by_target.csv",index=False,float_format="%.17g")
    summary.to_csv(outdir/"gated_summary.csv",index=False,float_format="%.17g")
    pd.DataFrame(stability).to_csv(outdir/"gated_target_stability.csv",
                                   index=False,float_format="%.17g")
    info={
        "status":"Post-hoc exploratory method selection from inspected preliminary labels",
        "method_switching":"Public AIA/C target features; whole target column from already frozen prediction",
        "no_new_person_predictions":True,"selection_bias":True,
        "rules":CHOICES,
        "group_threshold_excludes_teams_1_2":True,
        "imprint_high_targets":[int(t) for t in targets if desc.loc[t,"aia_imprint_high"]],
        "source_scored_rows_sha256":file_hash(original/"by_target.csv"),
        "source_public_features_sha256":file_hash(original/"public_target_groups.csv"),
        "independent_validation":False
    }
    (outdir/"gated_protocol.json").write_text(
        json.dumps(info,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(summary[(summary.background=="full")&
                  (summary.inference=="without1_2")].to_string(index=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment-dir",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args()
    calculate(a.experiment_dir.resolve(),a.output_dir.resolve())
