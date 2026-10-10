"""Validate immutable inputs, overlap multiplicities, fixed conditions and metrics."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import run_ablation as run

def test(pool='starter+prelim',output_dir=None):
    out=Path(output_dir) if output_dir else HERE/pool
    audit=json.loads((HERE/'input_hash_verification.json').read_text())
    assert audit['original_file_count']==142
    assert all(x['expected']==x['actual']==run.sha(x['path']) for x in audit['checks'])
    overlap=pd.read_csv(HERE/'prelim_all_teams_overlap.csv')
    assert len(overlap)==63 and overlap.team.nunique()==21
    actual={(int(r.team),r.source):(int(r.full13_unique),int(r.full13_c_rows)) for r in overlap.itertuples() if r.full13_unique}
    assert actual=={(18,'prelim'):(429,429),(19,'starter'):(117,129),(29,'starter'):(28,40),(29,'prelim'):(238,255)}
    extra=overlap[(overlap.full13_unique==0)&(overlap.qi7_unique>0)]
    assert len(extra)==1 and int(extra.iloc[0].team)==24 and int(extra.iloc[0].qi7_unique)==1
    assert (pd.read_csv(HERE/'background_candidate_overlap_check.csv').bg_tuple_overlap==0).all()
    pairs=pd.read_csv(HERE/'background_pair_identity_check.csv')
    assert (pairs.bare_record_id_intersection==5143).all() and (pairs.full13_tuple_intersection==0).all() and (pairs.qi7_tuple_intersection==0).all()
    freeze=json.loads((out/'prediction_freeze.json').read_text());assert freeze['conditions']==19
    protocol=json.loads((out/'fixed_protocol.json').read_text())
    for rel,digest in protocol['code_hashes'].items():
        source=HERE/rel
        if run.sha(source)!=digest and rel=='run_ablation.py':source=HERE/'run_ablation_prediction_frozen.py'
        assert run.sha(source)==digest
    original=json.loads((out/'original.json').read_text());assert original['fit_details']['original_raw_bitwise_regression'] and original['fit_details']['P01_bitwise_regression']
    records=freeze['records'];assert len({x['config']['condition'] for x in records})==19
    for rec in records:
        config=rec['config'];path=out/(config['condition']+'.npz')
        assert config['protocol_sha256']==run.sha(out/'fixed_protocol.json')
        assert run.sha(path)==rec['matrix_sha256']
        assert config['remaining']+config['removed']==10286
        assert sum(config['removed_by_source'].values())==config['removed']
        if config['target']:
            assert config['removed']=={18:429,19:117,29:266}[config['target']]
        with np.load(path) as z:
            for j,method in enumerate(run.METHODS):
                assert z[str(j)].shape==(875,20) and np.isfinite(z[str(j)]).all()
                assert run.msha(z[str(j)])==rec['prediction_hashes'][method]
    m=json.load(open('/workspace/shared/mia_bg7_inputs/manifest.json'));data=run.load(m)
    b,cs,qs,q,qsource,private,rare,normal,frames,teams=data
    y,_,_=run.base.audit_truth(Path(m['phases']['prelim']['attack_dir']).parent,b,private,rare,teams)
    by=pd.read_csv(out/'by_condition.csv',float_precision='round_trip');assert len(by)==84
    expected=pd.read_csv(HERE/'existing_bg7_P01_affected_teams.csv')
    for row in by.itertuples():
        assert row.n=={'all_fixed':1049,'normal':875}[row.population]
        with np.load(out/(row.condition+'.npz')) as z:score=z[str(run.METHODS.index(row.method))]
        conf=run.base.confidence(score,normal,private,rare)[:,teams.index(row.team)]
        mask=np.ones(len(b),bool) if row.population=='all_fixed' else normal
        s=conf[mask];labels=y[mask,teams.index(row.team)];assert len(s)==row.n and int(labels.sum())==row.members
        order=np.argsort(-s,kind='stable');ss=s[order];yy=labels[order]
        ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
        f=np.r_[0,np.cumsum(1-yy)[ends]/(len(yy)-yy.sum())];t=np.r_[0,np.cumsum(yy)[ends]/yy.sum()]
        value=float(np.interp(.01,f,t))
        reported=by.loc[(by.condition==row.condition)&(by.team==row.team)&(by.method==row.method)&(by.population==row.population),'tpr_0.01'].item()
        assert abs(value-reported)<1e-12
        if row.kind=='original' and row.method=='P01' and row.population=='all_fixed':
            gold=expected[(expected.background==pool)&(expected.team==row.team)]['tpr_0.01'].item();assert abs(value-gold)<1e-12
    summary=pd.read_csv(out/'ablation_summary.csv');assert len(summary)==36 and (summary.random_repeats==5).all()
    result={'passed':True,'immutable_original_files':142,'overlap_pairs':63,'frozen_conditions':19,'score_matrices':38,'independently_recomputed_metric_rows':84,'original_P01_bitwise_match':True}
    run.dump(out/'validation.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path);args=parser.parse_args();test(output_dir=args.output_dir)
