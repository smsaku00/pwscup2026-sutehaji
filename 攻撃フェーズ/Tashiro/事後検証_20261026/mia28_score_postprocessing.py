#!/usr/bin/env python3
"""Post-release MIA whole-channel rank transformations, self target=28.

Uses published all-cohort labels outside target28 to select an attacker's score
transformation, then evaluates on target28. NOT a valid competition-time
selection because other-target truth was not then published; this is a
leave-target-out diagnostic to avoid target28 truth-based method selection.
"""
from posthoc_analysis import *
from collections import defaultdict
import csv,json

METHODS=['original','reverse','extreme','center','binary_upper','binary_lower']
K=28
OUT=HERE/'self_target28_results'
OUT.mkdir(exist_ok=True)


def map_score(raw,method):
    if method=='original':return raw
    if method=='reverse':return -raw
    if method=='extreme':return np.abs(raw-np.median(raw))
    if method=='center':return -np.abs(raw-np.median(raw))
    if method=='binary_upper':return (raw>np.quantile(raw,.9)).astype(float)
    if method=='binary_lower':return (raw<np.quantile(raw,.1)).astype(float)
    raise ValueError(method)


def main():
    results=[];full=[]
    for attacker in ID:
        if attacker==K:continue
        mi,_=read_submission(attacker)
        prof={}
        for target in ID:
            if target==attacker:continue
            raw=np.asarray([float(mi[j][str(target)]) for j in range(1049)])
            prof[target]={m:mia_eval(attacker,target,map_score(raw,m)) for m in METHODS}
        # Choose transformation on other targets only, never target28.
        score_by_method={m:float(np.mean([prof[k][m]['tpr_all'] for k in ID if k not in [attacker,K]])) for m in METHODS}
        selected=max(METHODS,key=lambda m:score_by_method[m])
        local=prof[K]
        results.append({'attacker':attacker,'selected_using_21_other_targets':selected,
                        'train_avg_selected':score_by_method[selected],
                        'train_avg_original':score_by_method['original'],
                        'tpr28_original':local['original']['tpr_all'],
                        'tpr28_selected':local[selected]['tpr_all'],
                        'delta':local[selected]['tpr_all']-local['original']['tpr_all'],
                        'tpr28_best_oracle':max(local[m]['tpr_all'] for m in METHODS),
                        'tpr28_best_oracle_method':max(METHODS,key=lambda m:local[m]['tpr_all'])})
        for m in METHODS:
            full.append({'attacker':attacker,'method':m,
                         'other21_avg_tpr_all':score_by_method[m],
                         'target28_tpr_all':local[m]['tpr_all'],
                         'target28_auc':local[m]['auc']})
    with (OUT/'mia28_postprocess_loto.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
    with (OUT/'mia28_postprocess_methods.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=full[0].keys());w.writeheader();w.writerows(full)
    summary={'attacker_count':len(results),
       'target28_original_mean_tpr_all':float(np.mean([r['tpr28_original'] for r in results])),
       'target28_other_target_selected_mean':float(np.mean([r['tpr28_selected'] for r in results])),
       'target28_oracle_mean':float(np.mean([r['tpr28_best_oracle'] for r in results])),
       'target28_postprocess_improved_count':sum(r['delta']>1e-12 for r in results),
       'target28_postprocess_worsened_count':sum(r['delta']<-1e-12 for r in results),
       'target28_selected_max':max(results,key=lambda r:r['tpr28_selected']),
       'target28_oracle_max':max(results,key=lambda r:r['tpr28_best_oracle'])}
    (OUT/'mia28_postprocess_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for r in sorted(results,key=lambda r:-r['tpr28_selected']):print(r)

if __name__=='__main__':main()