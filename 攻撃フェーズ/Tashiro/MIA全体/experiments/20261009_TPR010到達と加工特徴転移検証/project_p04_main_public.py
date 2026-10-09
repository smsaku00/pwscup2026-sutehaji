"""P04 main-round *public-only* method-family projection.

Reads ONLY previously committed public C/AIA/A_bg descriptive features from
the main round, and PRELIMINARY public-feature median thresholds. Does not
read B_self, any membership truths, hidden AIA labels, actual submit CSV or
CodaBench results. Does NOT compute an attack score or promise TPR>=0.10.
Post-hoc preliminary chosen method-family orientation: exploratory only.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path

MAIN="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/main_public/main_public_group_projection.csv"
PRELIM="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/group_thresholds.json"
A="q13_median_iqr_sex0_kde_ratio_h2__raw__cross0.1"
B="q7_robust_adaptivek15_h1__age100__cross0.1"
P01="Q_prefecture_sex_shrink500_cross01"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(score,home,i_cut,h_cut):
    if score<i_cut:return "P01"
    if home>=h_cut:return "KDE_A"
    return "KDE_B"


def run(root,out):
    if out.exists():raise FileExistsError("Refuse overwrite "+str(out))
    out.mkdir(parents=True)
    p=root/MAIN
    q=root/PRELIM
    thresholds=json.loads(q.read_text())
    i=float(thresholds["imprint_median"])
    h=float(thresholds["c_home_median"])
    with p.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==22 and len({r["team"] for r in rows})==22
    assert "28" not in {r["team"] for r in rows}
    projected=[]
    for v in rows:
        team=int(v["team"])
        s=float(v["main_imprint_score"])
        f=float(v["C_home_fraction"])
        classification=select(s,f,i,h)
        projected.append({
            "team":team,
            "main_AIA_imprint":s,
            "preliminary_imprint_median":i,
            "imprint_margin":s-i,
            "main_C_home_fraction":f,
            "preliminary_C_home_fraction_median":h,
            "home_margin":f-h,
            "public_P04_family":classification,
            "model_formula":P01 if classification=="P01" else A if classification=="KDE_A" else B,
            "imprint_near_005":int(abs(s-i)<.005),
            "imprint_near_0050":int(abs(s-i)<.05),
            "home_near_005":int(abs(f-h)<.005),
            "home_near_0050":int(abs(f-h)<.05),
        })
    projected.sort(key=lambda x:x["team"])
    with (out/"main_P04_public_target_families.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(projected[0]))
        writer.writeheader()
        writer.writerows(projected)
    counts={name:sum(row["public_P04_family"]==name for row in projected)
            for name in ("P01","KDE_A","KDE_B")}
    cases=[]
    for di in (-.05,-.02,-.005,0,.005,.02,.05):
        for dh in (-.05,-.01,-.005,0,.005,.01,.05):
            updated={v["team"]:select(v["main_AIA_imprint"],v["main_C_home_fraction"],
                                        i+di,h+dh) for v in projected}
            changed=[tid for tid,cl in updated.items()
                     if cl!=next(r for r in projected if r["team"]==tid)["public_P04_family"]]
            cases.append({
                "imprint_shift":di,"C_home_shift":dh,"changes":len(changed),
                "changed_teams":";".join(map(str,changed)),
                "KDE_A":list(updated.values()).count("KDE_A"),
                "KDE_B":list(updated.values()).count("KDE_B"),
                "P01":list(updated.values()).count("P01")
            })
    with (out/"main_P04_public_threshold_sensitivity.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    report={
        "status":"PUBLIC GROUP PROJECTION ONLY; NO MAIN ATTACK OR TPR MEASURED",
        "round":"PWS CUP 2026 main 22 targets",
        "target_counts":counts,
        "group_preliminary_cutoff":{"AIA_imprint":i,"C_home_fraction":h},
        "families":{name:[r["team"] for r in projected if r["public_P04_family"]==name]
                    for name in ("P01","KDE_A","KDE_B")},
        "near_preliminary_imprint_cutoff_005":[r["team"] for r in projected if r["imprint_near_005"]],
        "near_preliminary_imprint_cutoff_0050":[r["team"] for r in projected if r["imprint_near_0050"]],
        "near_home_cutoff_005":[r["team"] for r in projected if r["home_near_005"]],
        "near_home_cutoff_0050":[r["team"] for r in projected if r["home_near_0050"]],
        "threshold_perturbations":len(cases),
        "preliminary_selection_bias":"KDE A/B orientation was chosen post-hoc after seeing published preliminary truth",
        "same_team_number_not_same_processing_proof":True,
        "predicted_0.10_success_count":None,
        "main_B_self_read":False,
        "main_labels_read":False,
        "main_candidate_scores_generated":False,
        "competition_submission_changed":False,
        "rare_5bit_changed":False,
        "sources_sha256":{MAIN:digest(p),PRELIM:digest(q)}
    }
    (out/"main_P04_public_group_audit.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    run(a.root.resolve(),a.out.resolve())
