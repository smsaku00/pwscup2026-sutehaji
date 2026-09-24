from 採点共通 import *
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PYTHON=ROOT/'スターターキット/初版/starter/.venv/bin/python'
def mia_job(p,panel,aid):
    folder=HERE/'mia'/panel['id']/aid;folder.mkdir(parents=True,exist_ok=True);meta=folder/'結果.json'
    if meta.exists():
        r=load(meta);assert sha(r['prediction'])==r['prediction_sha256'];return r
    if aid=='M04':src=Path(p['mia4']);assets=[x for x in src.parent.glob('*.py') if x!=src]
    else:
        a=next(x for x in p['attacks'] if x['id']==aid);src=Path(a['local_path']);assets=[src.parent/x['name'] for x in a['assets']]
    script=Path(copy(src,folder/src.name))
    for asset in assets:copy(asset,folder/asset.name)
    save(folder/'targets.json',[dict(target_k=c['target_id'],c_path=c['path'],aia_challenge_path=p['panels'][0]['challenge']) for c in p['cases']])
    cmd=[str(PYTHON),'-B',str(script),'--b',panel['b'],'--a-bg',p['a_bg'],'--targets',str(folder/'targets.json')]
    with (folder/'実行.log').open('w') as log:r=subprocess.run(cmd,cwd=folder,stdout=log,stderr=subprocess.STDOUT)
    fp=folder/('F_mia_'+script.stem+'.csv');assert r.returncode==0 and fp.exists(),str(folder)
    d=read(fp);t=read(panel['truth']);assert len(d)==1049 and d.record_id.is_unique and set(d.record_id)==set(t.record_id)
    assert set(d.columns)=={'record_id'}|{str(c['target_id']) for c in p['cases']}
    sys.path.insert(0,str(Path(p['kit'])/'starter/src'))
    from pwscup2026.scoring.attack_score import _tpr_at_fpr
    rows=[];checks=0
    for c in p['cases']:
        pred=d.set_index('record_id').loc[t.record_id,str(c['target_id'])].to_numpy(float)
        assert np.isfinite(pred).all() and (pred>=0).all() and (pred<=1).all()
        tpr,auc=_tpr_at_fpr(pred,t.is_member.to_numpy(int),.01)
        rows.append(dict(case=c['id'],panel=panel['id'],attack=aid,TPR_at_FPR01=tpr,AUC=auc))
        for ref in p['mia_references']:
            if (ref['case'],ref['panel'],ref['attack'])!=(c['id'],panel['id'],aid):continue
            original=read(ref['prediction']);col=next(x for x in original.columns if x!='record_id')
            expected=original.set_index('record_id').loc[t.record_id,col].to_numpy(float)
            np.testing.assert_allclose(pred,expected,atol=1e-10,rtol=0);checks+=1
    result=dict(panel=panel['id'],attack=aid,rows=rows,reference_checks=checks,prediction=str(fp),prediction_sha256=sha(fp),command=cmd)
    save(meta,result);return result
