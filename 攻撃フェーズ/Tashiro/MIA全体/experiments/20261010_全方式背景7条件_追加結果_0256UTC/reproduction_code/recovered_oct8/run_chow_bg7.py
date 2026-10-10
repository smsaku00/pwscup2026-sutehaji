"""Original Chow-Liu three-form diagnostics; keep the source no-GT guard.

Only input routing and split-size assertions are generalized to pooled BGs.
No membership truth, attack scores, per-row outputs, or NPZ are read/written.
"""
from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np,pandas as pd
from scipy.stats import spearmanr,rankdata
from benchmark import data,u,HERE

CHOW=HERE/'chow_runtime';u.HERE=CHOW
spec=importlib.util.spec_from_file_location('recovered_chow',CHOW/'離散依存木の密度比を確認.py');tree=importlib.util.module_from_spec(spec);spec.loader.exec_module(tree)
ORIGINAL_SPLIT=tree.split.split_a

def generalized_split(d):
    a=d['a'];order=np.random.default_rng(tree.split.SEED).permutation(len(a));assignment=np.empty(len(a),int)
    assignment[order[:len(a)//2]]=0;assignment[order[len(a)//2:]]=1
    parts=[a.iloc[np.flatnonzero(assignment==fold)].reset_index(drop=True) for fold in range(2)]
    assert np.bincount(assignment).tolist()==[len(a)//2,len(a)-len(a)//2]
    assert not pd.MultiIndex.from_frame(parts[0][list(u.base.QI7)]).isin(pd.MultiIndex.from_frame(parts[1][list(u.base.QI7)])).any()
    return assignment,parts

def main():
    out=HERE/'chow/prelim';out.mkdir(parents=True,exist_ok=True)
    d=data('prelim');a,p=ORIGINAL_SPLIT(d);aa,pp=generalized_split(d);assert np.array_equal(a,aa)
    for x,y in zip(p,pp):pd.testing.assert_frame_equal(x,y,check_exact=True)
    tree.split.split_a=generalized_split
    qa=tree.numerical_qa();qa['generalized_split_original_bit_exact']=True
    matrices=u.matrices;captured={};records=[]
    sources=[Path(__file__),*CHOW.glob('*.py')]
    plan={'input_sha256_before_execution':{},'support_code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'backgrounds':['prelim','starter','main','starter+prelim','starter+main','prelim+main','starter+prelim+main'],'settings':3,'phase':'prelim','membership_labels_used':False,'score_policy':'original guard preserved; public-null CRPS/PIT and rank diagnostics only','changes':'canonical input cache and variable-n half split; original seed/order/group definitions retained'}
    (out/'fixed_protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    for bg in plan['backgrounds']:
        start=time.perf_counter();d=data(bg);u.setup=lambda phase='prelim':d;u.base.INPUT_HASHES.clear()
        def capture(banks,teams,cross=(.1,)):
            result=matrices(banks,teams,cross);captured[bg]=result
            return result
        u.matrices=capture
        dest=out/bg;dest.mkdir(exist_ok=True)
        tree.phase_run('prelim',dest,qa,plan)
        records.append({'background':bg,'seconds':time.perf_counter()-start,'settings':3,'targets':20,'labels_used':False,'prediction_hashes':{k:hashlib.sha256(np.ascontiguousarray(v,dtype='<f8').tobytes()).hexdigest() for k,v in captured[bg].items()}})
        (dest/'prediction_record.json').write_text(json.dumps(records[-1],ensure_ascii=False,indent=2)+'\n')
        print('CHOW COMPLETE',bg,records[-1]['seconds'],flush=True)
    rows=[]
    for bg,models in captured.items():
        for method,mat in models.items():
            for j,t in enumerate(d['teams']):
                ref=captured['prelim'][method][:,j];value=mat[:,j]
                ref_head=set(np.flatnonzero(rankdata(-ref,method='min')<=12));head=set(np.flatnonzero(rankdata(-value,method='min')<=12))
                rows.append({'background':bg,'reference_background':'prelim','method':method,'team':t,'spearman':float(spearmanr(value,ref).statistic),'top12_tie_inclusive_overlap':len(head&ref_head),'top12_tie_inclusive_rows':len(head),'reference_top12_rows':len(ref_head),'membership_accuracy_unknown':True})
    pd.DataFrame(rows).to_csv(out/'background_rank_diagnostics.csv',index=False,float_format='%.17g')
    (out/'completion.json').write_text(json.dumps({'complete':True,'settings_per_bg':3,'backgrounds':7,'condition_panels':21,'membership_scoring':False,'source_no_GT_guard_retained':True,'runs':records},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
