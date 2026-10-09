"""Post-selected AIA imprint high-group gates for existing 22 fixed MIA methods.

All input data are previously published AGGREGATE metrics plus PUBLIC group
features. No individual predictions, target-membership training or scoring.
Selecting the method after reviewing these preliminary outcomes is post-hoc;
performance cannot be generalized as an independent test.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

HIST="攻撃フェーズ/Tashiro/MIA全体/experiments/20261008_非学習方式と加工特徴_背景耐性/固定方式_全背景評価.csv"
GROUP="攻撃フェーズ/Tashiro/MIA全体/experiments/20261009_AIA臨床分布と加工特徴の群別補正/results/public_target_groups.csv"
BASE="Q_prefecture_sex_shrink500_cross01"
TARGETS=[3,4,6,7,8,9,10,13,17,18,19,20,22,23,24,26,29]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    root,out=args.root.resolve(),args.out.resolve()
    if out.exists():raise FileExistsError("Refuse overwrite")
    out.mkdir(parents=True)
    historic=pd.read_csv(root/HIST,encoding="utf-8-sig")
    pub=pd.read_csv(root/GROUP)
    assert len(pub)==20
    high=set(pub[(pub.aia_imprint_high==1) & pub.team.isin(TARGETS)].team)
    assert len(high)==9, high
    df=historic[np.isclose(historic.fpr,.01)&historic.team.isin(TARGETS)].copy()
    methods=sorted(df.method.unique())
    views=sorted(df.condition.unique())
    assert len(TARGETS)==17 and len(methods)==22 and len(views)==7
    assert len(df)==17*22*7
    m=df.set_index(["team","method","condition"]).tpr
    def get(team,method,view):
        return float(m.loc[(team,method,view)])
    metrics=[]
    summary=[]
    for method in methods:
        target_stats={}
        for team in TARGETS:
            output=[get(team,method if team in high else BASE,view) for view in views]
            before=[get(team,BASE,view) for view in views]
            f=views.index("full")
            target_stats[team]={
                "full_tpr":output[f],
                "full_delta_P01":output[f]-before[f],
                "hit_full":int(output[f]>=.1-1e-12),
                "hit_all7":int(all(x>=.1-1e-12 for x in output)),
                "hit_bg_count":sum(int(x>=.1-1e-12) for x in output),
                "method_if_high":method if team in high else BASE,
                "min_background_tpr":min(output),
            }
            metrics.append({"team":team,"method":method,"aia_imprint_high":int(team in high),
                            **target_stats[team]})
        values=list(target_stats.values())
        summary.append({
            "method_for_high_group":method,"high_targets":len(high),
            "low_targets":len(TARGETS)-len(high),
            "full_hit_ge010":sum(x["hit_full"] for x in values),
            "all7_hit_ge010":sum(x["hit_all7"] for x in values),
            "new_full_hit_teams": ";".join(str(t) for t,x in target_stats.items()
                     if x["hit_full"] and get(t,BASE,"full")<.1-1e-12),
            "full_hit_teams": ";".join(str(t) for t,x in target_stats.items() if x["hit_full"]),
            "stable_teams": ";".join(str(t) for t,x in target_stats.items() if x["hit_all7"]),
            "mean_full_tpr":sum(x["full_tpr"] for x in values)/len(values),
            "mean_full_delta_P01":sum(x["full_delta_P01"] for x in values)/len(values),
            "full_improved":sum(x["full_delta_P01"]>1e-12 for x in values),
            "full_worsened":sum(x["full_delta_P01"]< -1e-12 for x in values),
            "min_7bg_hit_target_count":min(
                sum(get(team,method if team in high else BASE,view)>=.1-1e-12
                    for team in TARGETS) for view in views),
            "max_7bg_hit_target_count":max(
                sum(get(team,method if team in high else BASE,view)>=.1-1e-12
                    for team in TARGETS) for view in views),
        })
    df_summary=pd.DataFrame(summary)
    df_summary.to_csv(out/"all_22_high_group_methods.csv",index=False,float_format="%.17g")
    pd.DataFrame(metrics).to_csv(out/"all_22_high_group_targets.csv",index=False,float_format="%.17g")
    with (out/"protocol.json").open("w",encoding="utf-8") as f:
        json.dump({
            "status":"POSTHOC high-group method comparison; NOT a held-out independent test",
            "candidate_methods":22,"backgrounds":views,"target_count":17,
            "excluded_teams":[1,2,12],"source_p03_group_high":sorted(high),
            "low_group_score_source":BASE,
            "decision_rule":"For each fixed method use it for all 9 AIA-imprint-high target columns, P01 for 8 low; evaluate complete panel",
            "target_membership_labels_in_method_assignment":False,
            "method_selection_after_results":True,
            "main_membership_not_used":True,"main_submission_not_changed":True,
            "source_sha256":{HIST:sha(root/HIST),GROUP:sha(root/GROUP)},
        },f,indent=2,ensure_ascii=False)
    base_row=df_summary[df_summary.method_for_high_group==BASE].iloc[0]
    assert base_row.full_hit_ge010==2 and base_row.all7_hit_ge010==0
    print(df_summary.sort_values(["all7_hit_ge010","full_hit_ge010","mean_full_tpr"],
                                ascending=False).head(16).to_string(index=False))
    print("OUTPUT",out)


if __name__=="__main__":
    main()
