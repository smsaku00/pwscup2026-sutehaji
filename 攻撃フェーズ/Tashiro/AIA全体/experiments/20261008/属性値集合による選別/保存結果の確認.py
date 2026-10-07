"""本実験の記録を読み、別1D実装・列ごとの行交換・固定条件を照合する。"""
from pathlib import Path
import json,hashlib,importlib.util,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("column_set_experiment",HERE/"分析.py")
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def brute_evidence(q,c,ref):
    dc=np.empty((len(q),4));db=np.empty_like(dc)
    for sex in np.unique(q.sex):
        a=np.flatnonzero(q.sex.to_numpy()==sex)
        for j,col in enumerate(exp.COLS):
            qc=q[col].to_numpy()[a,None]
            dc[a,j]=np.abs(qc-c.loc[c.sex==sex,col].to_numpy()[None,:]).min(axis=1)
            db[a,j]=np.abs(qc-ref.loc[ref.sex==sex,col].to_numpy()[None,:]).min(axis=1)
    r=np.log((db+1e-8)/(dc+1e-8))
    # top2はsortを避け、6組の和の最大から独立照合。
    pairs=np.stack([r[:,i]+r[:,j] for i in range(4) for j in range(i+1,4)],axis=1)
    return dict(dc=dc,db=db,ratio=r,all4=np.sum(r,axis=1)/4,top2=pairs.max(axis=1)/2,exact=(dc<=1e-8).sum(axis=1))
def python_select(q,ev,method):
    selected=np.zeros(len(q),dtype=bool)
    for sex in np.unique(q.sex):
        ii=np.flatnonzero(q.sex.to_numpy()==sex).tolist()
        if method==exp.METHODS[1]:key=lambda i:(-ev["all4"][i],int(q.challenge_row_id.iloc[i]))
        elif method==exp.METHODS[2]:key=lambda i:(-ev["top2"][i],int(q.challenge_row_id.iloc[i]))
        else:key=lambda i:(-int(ev["exact"][i]),-ev["all4"][i],int(q.challenge_row_id.iloc[i]))
        ordered=sorted(ii,key=key)
        selected[ordered[:int(np.ceil(len(ii)*.75))]]=True
    return selected
def main():
    rec=json.loads((HERE/"予測実行記録.json").read_text())
    assert rec["code_sha256"]==digest(HERE/"分析.py") and rec["config_sha256"]==digest(HERE/"固定条件.json")
    for p,sha in rec["input_sha256"].items():assert digest(p)==sha
    permutations=[];independent=[];observed=0
    for phase,folder,bgp in exp.SETS:
        bg=exp.core.read(bgp);order=np.random.default_rng(exp.CFG["split_seed"]+(phase=="main")).permutation(len(bg))
        ref=bg.iloc[order[len(bg)//2:]].reset_index(drop=True)
        for item in [x for x in rec["predictions"] if x["phase"]==phase]:
            k=item["team"];p=HERE/item["path"];assert digest(p)==item["sha256"];data=np.load(p)
            q=exp.core.read(folder/f"aia_challenge_{k}.csv").sort_values("challenge_row_id").reset_index(drop=True)
            c=exp.core.read(folder/f"C_{k}.csv");observed+=1
            shuffled=c.copy();rng=np.random.default_rng(2026100841+k+(phase=="main")*100000)
            # sex列を維持し、臨床列を各性別の内部で別々に交換。列の周辺集合は不変。
            for sex in np.unique(c.sex):
                ii=np.flatnonzero(c.sex.to_numpy()==sex)
                for col in exp.COLS:shuffled.loc[ii,col]=c.loc[rng.permutation(ii),col].to_numpy()
            for ci,context in enumerate(exp.CONTEXTS):
                r=ref if ci==0 else bg
                original=exp.evidence(q,c,r);after=exp.evidence(q,shuffled,r)
                for key in ["dc","db","ratio","all4","top2","exact"]:assert np.array_equal(original[key],after[key])
                direct=brute_evidence(q,c,r)
                for key in ["dc","db","ratio","all4","top2","exact"]:
                    assert np.allclose(original[key],direct[key],rtol=1e-14,atol=1e-14)
                for mi,method in enumerate(exp.METHODS[1:],start=1):
                    new=exp.independent_select(q,after,method);ind=python_select(q,direct,method)
                    diff=int((new!=data["real"][ci,mi]).sum());d2=int((ind!=data["real"][ci,mi]).sum())
                    assert diff==0 and d2==0
                    permutations.append(dict(phase=phase,team=k,context=context,method=method,changed_after_independent_column_permutation=diff))
                    independent.append(dict(phase=phase,team=k,context=context,method=method,changed_vs_brute_force_python_sort=d2))
                for mi,method in enumerate(exp.METHODS):
                    for sex in np.unique(q.sex):
                        a=q.sex.to_numpy()==sex;assert int(data["real"][ci,mi,a].sum())==int(np.ceil(int(a.sum())*.75))
    assert observed==44
    pd.DataFrame(permutations).to_csv(HERE/"集計/性別内の列独立交換.csv",index=False)
    pd.DataFrame(independent).to_csv(HERE/"集計/距離と選別の別実装照合.csv",index=False)
    receipt=dict(prediction_receipt_sha256=digest(HERE/"予測実行記録.json"),verification_code_sha256=digest(__file__),
       unchanged_public_input_files=len(rec["input_sha256"]),actual_C_count=observed,
       independent_distance_ratio_and_python_sort_checks=len(independent),independent_selection_changed_max=0,
       independent_clinical_column_permutation_checks=len(permutations),column_shuffle_selection_changed_max=0,
       same_sex_selection_quota_verified=True,truth_read=False,main_truth_used=False,B_self_used=False,
       note="列独立交換は値集合方式の不変性の照合で、実Cが実際にこの加工だったことの証拠ではない")
    exp.writej(HERE/"検証結果.json",receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=="__main__":main()

