"""県・性別と性別・年齢帯の較正を混ぜる追加24方式。既存結果を上書きしない。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
spec=importlib.util.spec_from_file_location("calibration_sweep",PARENT/"分析.py")
sweep=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=sweep
spec.loader.exec_module(sweep)
common=sweep.common


def configurations():
    names={"raw","raw_cross01","flat_pref_sex_s100","flat_pref_sex_s200","flat_pref_sex_s500",
           "flat_sex_age_s100","flat_sex_age_s300"}
    base=[s for s in sweep.specs() if s["method"] in names]
    extra=[]
    for county in (100,200,500):
        for age in (100,300):
            for weight in (.1,.25,.5,.75):
                extra.append({"method":f"mix_county{county}_age{age}_w{weight:g}","family":"mixed",
                              "county_strength":county,"age_strength":age,"age_weight":weight,"cross":.1})
    assert len(base)==7 and len(extra)==24
    return base,base+extra


def predict(b,a,controls,cs,normal,base,configs,label):
    banks={s["method"]:[] for s in configs}
    counts=[]
    for index,(team,c) in enumerate(cs.items(),1):
        q=controls[team]
        query=pd.concat([b[list(common.QI7)],q[list(common.QI7)]],ignore_index=True)
        raw=common.core(a,query,c,common.NUM7,(10,))[10]
        variants,stats=sweep.calibrated_variants(raw[:len(b)],b,raw[len(b):],q,base)
        for config in configs:
            if config["family"]=="mixed":
                county=variants[f"flat_pref_sex_s{config['county_strength']}"]
                age=variants[f"flat_sex_age_s{config['age_strength']}"]
                weight=config["age_weight"]
                values=(1-weight)*county+weight*age
            else:
                values=variants[config["method"]]
            banks[config["method"]].append(rankdata(values[normal],method="average")/normal.sum())
        if index%4==0 or index==len(cs):
            print(json.dumps({"stage":"predict_mix","condition":label,"completed":index,"targets":len(cs)}),flush=True)
    matrices={}
    for config in configs:
        mat=np.column_stack(banks[config["method"]])
        if config["cross"]:
            med=np.column_stack([np.median(np.delete(mat,j,axis=1),axis=1) for j in range(len(cs))])
            mat=mat-config["cross"]*med
        assert np.isfinite(mat).all()
        matrices[config["method"]]=mat
    return matrices


def tie_sensitive_selection(table):
    part=table[(table.condition=="full")&(table.fpr==.01)&~table.team.isin([1,2])]
    pivot=part.pivot(index="team",columns="method",values="mia_all").sort_index()
    rows=[]
    for team in pivot.index:
        train=pivot.drop(index=team).mean()
        best=float(train.max())
        ties=sorted(name for name in train.index if abs(train[name]-best)<1e-12)
        selected=sweep.REF if sweep.REF in ties else ties[0]
        scores=pivot.loc[team,ties]
        rows.append({"held_out_team":int(team),"selected_method":selected,"training_best":best,
                     "tied_best_methods":len(ties),"tied_best_names":";".join(ties),
                     "held_out_score":float(pivot.loc[team,selected]),"reference_score":float(pivot.loc[team,sweep.REF]),
                     "delta":float(pivot.loc[team,selected]-pivot.loc[team,sweep.REF]),
                     "tie_held_out_min":float(scores.min()),"tie_held_out_max":float(scores.max()),
                     "note":"同点時は基準500を優先。予備戦再利用の感度診断で、独立検証ではない。"})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=sweep.ROOT)
    parser.add_argument("--output-dir",type=Path,default=HERE/"結果")
    args=parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError("既存結果は上書きしません。")
    started=time.perf_counter()
    started_jst=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()
    base,configs=configurations()
    pinpaths=[Path(__file__),HERE/"実験方針.json",PARENT/"分析.py",PARENT/"実験方針.json",
              sweep.LEGACY/"分析.py",sweep.LEGACY/"非会員を使う順位較正の検討.py"]
    initial_pins={str(p.resolve()):sweep.sha(p) for p in pinpaths}
    b,a,private,rare,cs,controls,cuts=sweep.setup_phase(args.root,"prelim")
    refs=[("full",a)]+[(f"fraction{f}_seed{seed}",sweep.stratified_subset(a,f,seed))
                        for f in (.9,.8,.5) for seed in (20261007,20261008)]
    matrices={label:predict(b,aa,controls,cs,~private & ~rare,base,configs,label) for label,aa in refs}
    mb,ma,mp,mr,mcs,mcontrols,mcuts=sweep.setup_phase(args.root,"main")
    main_matrices=predict(mb,ma,mcontrols,mcs,~mp & ~mr,base,configs,"main")
    main_top,main_error=sweep.main_diagnostics(main_matrices,list(mcs),mb,~mp & ~mr)
    print(json.dumps({"stage":"mix_predictions_fixed_before_truth","methods":len(configs),"main_round3_max_error":main_error}),flush=True)
    table,official_error=sweep.evaluate(args.root,b,private,rare,list(cs),matrices)
    summary,robust=sweep.summarize(table,configs)
    reproduction=sweep.verify_saved_baselines(table)
    first=common.read(PARENT/"結果/対象別TPR.csv")
    overlap=table.merge(first,on=["condition","method","team","fpr"],suffixes=("_new","_first"))
    repeat_error=float((overlap.mia_all_new-overlap.mia_all_first).abs().max())
    assert len(overlap)==7*7*20*4 and repeat_error<1e-12
    combined=pd.concat([first,table[table.method.str.startswith("mix_")]],ignore_index=True)
    selected=tie_sensitive_selection(combined)
    for path,expected in {**initial_pins,**common.INPUT_HASHES}.items(): assert sweep.sha(path)==expected
    args.output_dir.mkdir(parents=True)
    tables={"対象別TPR.csv":table,"集約比較.csv":summary,"頑健性比較.csv":robust,
            "本戦順位の変化.csv":pd.DataFrame(main_top),"全89方式_対象を外した方式選択.csv":selected}
    for name,frame in tables.items(): frame.to_csv(args.output_dir/name,index=False,float_format="%.17g")
    record={"started_jst":started_jst,"completed_jst":datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
            "seconds":time.perf_counter()-started,"python":sys.executable,"python_version":platform.python_version(),
            "new_methods":24,"methods_in_run":31,"unique_methods_total":89,
            "input_and_code_sha256":{**initial_pins,**common.INPUT_HASHES},
            "official_tpr_max_error":official_error,"main_round3_max_error":main_error,
            "first_run_reproduction_max_error":repeat_error,"saved_baseline_reproductions":reproduction,
            "main_truth_used":False,"submission_files_written":False,"coda_submission":False,
            "note":"最初の結果を見て追加した探索。独立検証ではない。",
            "outputs_sha256":{name:sweep.sha(args.output_dir/name) for name in tables}}
    (args.output_dir/"実行記録.json").write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(robust.sort_values(["sample6_mean_delta","full_other18"],ascending=False).head(10).to_json(orient="records",force_ascii=False),flush=True)
    print(json.dumps({"stage":"mix_complete","seconds":record["seconds"],"output_dir":str(args.output_dir)}),flush=True)


if __name__=="__main__": main()
