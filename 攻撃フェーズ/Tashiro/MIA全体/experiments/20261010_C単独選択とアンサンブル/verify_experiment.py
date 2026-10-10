"""Independent schema, official scorer, heldout-label exclusion and output audit."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,inspect,ast
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import run_experiment as r
sys.path.insert(0,'/workspace/shared/mia_bg7_inputs/official_scoring/src')
from pwscup2026.scoring.attack_score import _tpr_at_fpr

def main():
    assert json.loads((HERE/'completion.json').read_text())['complete']
    b,private,rare,normal,y=r.load_eval();schema=json.loads((r.OCT/'cache_schema.json').read_text());mapping=pd.read_csv(r.OCT/schema['row_mapping'])
    assert r.sha(r.OCT/schema['row_mapping'])==schema['row_mapping_sha256']
    assert np.array_equal(mapping.record_id,b.loc[normal,'record_id'])
    assert np.array_equal(mapping.B_row,np.flatnonzero(normal))
    audit=json.loads((HERE/'source_score_audit.json').read_text());checks=[]
    for bank in audit['banks']:
        if bank['family'] in r.FOLDERS:
            path=r.CAL/bank['family']/bank['background']/'実行記録.json'
            rec=json.loads(path.read_text());assert rec['complete'] and rec['models']==bank['candidate_count']
        else:
            path=r.OCT/bank['family']/'prelim'/bank['background']/'prediction_record.json'
            rec=json.loads(path.read_text());assert rec['truth_read'] is False
            assert rec['scores_sha256']==bank['score_file_sha256']
        assert bank['record_sha256']==r.sha(path)
        checks.append({'family':bank['family'],'background':bank['background'],'source_record_hash_passed':True,'file_hash_matches_original_record':bank['family'] not in r.FOLDERS})
    by=pd.read_csv(HERE/'heldout_by_target.csv',float_precision='round_trip');npz=np.load(HERE/'private_cache/heldout_prediction_scores.npz');errors=[]
    for row in by.to_dict('records'):
        s=npz[f"{row['scope']}::{row['team']}::{row['strategy']}"];conf=np.zeros(len(b));conf[rare&~private]=.4;conf[normal]=.6+.3*rankdata(s,method='min')/normal.sum()
        truth=y[:,r.TEAMS.index(row['team'])];v=_tpr_at_fpr(conf,truth,.01)[0];errors.append(abs(v-row['tpr_0.01']))
    assert max(errors)<1e-12
    features=pd.read_csv(HERE/'C_only_features.csv').drop(columns='team').to_numpy();m=np.load(HERE/'private_cache/candidate_metrics.npy');plans=json.loads((HERE/'private_cache/selection_plan.json').read_text());nested=[]
    for plan in plans:
        t=plan['target_column'];train=plan['train_columns'];a=m.copy();a[:,t]=np.linspace(1,9,len(m));inner=[]
        for v in train:
            indices,_=r.choose_all(a,features,[j for j in train if j!=v],v);inner.append(a[indices,v])
        best=int(np.argmax(np.mean(inner,axis=0)));ix,_=r.choose_all(a,features,train,t)
        assert ix[best]==plan['selected']['C_gate_nested'];nested.append({'scope':plan['scope'],'team':r.TEAMS[t],'changing_heldout_metrics_leaves_nested_gate_unchanged':True})
    feature_node=ast.parse(inspect.getsource(r.features));names={n.id for n in ast.walk(feature_node) if isinstance(n,ast.Name)}
    forbidden={'TEAMS','ROOT','OCT','CAL','WORK','BGS','meta','m','truth','normal','private','rare'};assert not names&forbidden
    result={'passed':True,'source_banks':checks,'oct8_normal_record_mapping_passed':True,'official_heldout_score_checks':len(errors),'max_score_error':max(errors),'nested_gate_heldout_perturbation_checks':nested,'feature_function_forbidden_input_names_absent':True,'all20_targets':20,'without1_2_targets':18,'population':'all_fixed_1049','normal_ranks':875}
    r.dump(HERE/'independent_validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_banks','nested_gate_heldout_perturbation_checks']},ensure_ascii=False))
if __name__=='__main__':main()
