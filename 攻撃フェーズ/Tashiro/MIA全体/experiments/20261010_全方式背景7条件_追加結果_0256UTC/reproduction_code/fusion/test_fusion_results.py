"""Verify complete output grid, original regression and unchanged input bytes."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
BG={'starter','prelim','main','starter+prelim','starter+main','prelim+main','starter+prelim+main'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def msha(v):return hashlib.sha256(np.ascontiguousarray(v,dtype='<f8').tobytes()).hexdigest()


def main():
    checks=[]
    config=json.loads((HERE/'fixed_protocol.json').read_text())
    manifest_path=Path('/workspace/shared/mia_bg7_inputs/manifest.json')
    assert sha(manifest_path)==config['input_manifest_sha256']
    manifest=json.loads(manifest_path.read_text())
    for item in manifest['files']:assert sha(item['path'])==item['sha256']
    for item in manifest['subsets'].values():
        assert sha(item['path'])==item['sha256']
        assert sha(item['identity_path'])==item['identity_sha256']
    checks.append({'test':'input_bytes_unchanged','passed':True,'files':len(manifest['files']),'backgrounds':7})
    count=0
    for phase in ('prelim','main'):
        for bg in BG:
            rows=json.loads((HERE/phase/bg/'prediction_hashes.json').read_text())
            rec=json.loads((HERE/phase/bg/'prediction_record.json').read_text())
            assert not rec['membership_truth_read'] and sha(rec['artifact'])==rec['artifact_sha256']
            assert len(rows)==(60 if phase=='prelim' else 30)
            arrays=np.load(rec['artifact'])
            for row in rows:
                x=arrays[row['key']]
                assert x.shape==tuple(row['shape'])==(875,len(row['teams']))
                assert np.isfinite(x).all() and msha(x)==row['sha256']
                assert len(row['teams'])=={'all20':20,'legacy18':18,'all22':22}[row['inference']]
            count+=len(rows)
    assert count==630
    checks.append({'test':'all_prediction_panels_and_hashes','passed':True,'condition_panels':count})
    pred=json.loads((HERE/'historical_prediction_regression.json').read_text())
    metric=json.loads((HERE/'historical_metric_regression.json').read_text())
    assert len(pred)==35 and all(x['exact_match'] for x in pred)
    assert len(metric)==640 and all(x['passed'] for x in metric)
    checks.append({'test':'historical_regressions','passed':True,'exact_prediction_hashes':35,'target_metric_checks':640,'max_metric_error':max(x['absolute_error'] for x in metric)})
    df=pd.read_csv(HERE/'prelim_by_target.csv')
    assert len(df)==23940 and set(df.background)==BG and set(df.inference)=={'all20','legacy18'}
    assert not df.duplicated(['background','family','inference','method','team','population']).any()
    assert set(df.population)=={'all_fixed','nonrare','normal'}
    assert df.groupby('population').n.unique().map(lambda x:len(x)==1).all()
    assert df.groupby('population').n.first().to_dict()=={'all_fixed':1049,'nonrare':1025,'normal':875}
    assert np.isfinite(df.select_dtypes(include='number')).all().all()
    counts=df.groupby(['background','family','inference','population','method']).team.nunique()
    assert all(n=={'all20':20,'legacy18':18}[key[2]] for key,n in counts.items())
    checks.append({'test':'prelim_rows_populations_target_coverage','passed':True,'rows':len(df)})
    main=pd.read_csv(HERE/'main_rank_qa.csv')
    assert len(main)==4620 and set(main.background)==BG and main.team.nunique()==22
    assert not main.duplicated(['background','family','inference','method','team']).any()
    assert not any(c.startswith(('tpr','fpr','auc')) for c in main.columns)
    assert np.isfinite(main.select_dtypes(include='number')).all().all()
    same=main[main.background=='main']
    assert np.allclose(same.spearman,1,rtol=0,atol=1e-12) and (same.top20_overlap==20).all()
    assert (main.n==875).all() and main.finite.all()
    checks.append({'test':'main_rank_only_coverage_and_self_identity','passed':True,'rows':len(main)})
    freeze=json.loads((HERE/'prelim_prediction_freeze.json').read_text())
    assert len(freeze['prediction_files'])==7
    assert sha(freeze['protocol_path'])==freeze['protocol_sha256']
    assert sha(freeze['input_manifest_path'])==freeze['input_manifest_sha256']
    for item in freeze['prediction_files']:assert sha(item['path'])==item['sha256']
    checks.append({'test':'prelim_freeze_seal_still_valid','passed':True})
    payload={'passed':True,'checks':checks}
    (HERE/'test_results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(payload,ensure_ascii=False))


if __name__=='__main__':main()
