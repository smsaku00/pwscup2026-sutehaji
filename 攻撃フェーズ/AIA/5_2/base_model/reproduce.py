#!/usr/bin/env python3
"""Shared baseline functions used by the 5_2 reproduction entry point.

Reads main public background/C/challenge only. No main B time/labels, network,
or competition submission. MIA and token are copied, never modeled or printed.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,sys,zipfile
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE=Path(__file__).resolve().parent
import round5_engine as ca

def read_named(source,name):
    source=Path(source)
    if source.is_dir():
        found=[p for p in source.rglob(name) if p.is_file()]
        if len(found)!=1:raise ValueError(f'{source}: expected one {name}, got {len(found)}')
        return found[0].read_bytes()
    if source.name==name:return source.read_bytes()
    with zipfile.ZipFile(source) as z:
        found=[i for i in z.infolist() if Path(i.filename).name==name and not i.is_dir() and not i.filename.startswith('__MACOSX/')]
        if len(found)!=1:raise ValueError(f'{source.name}: expected one {name}, got {len(found)}')
        if found[0].file_size>50_000_000:raise ValueError('Unexpectedly large input')
        return z.read(found[0])

def stage(args):
    root=args.work_dir.resolve();manifest={}
    def write(source,name,directory):
        raw=read_named(source,name);p=root/directory/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        manifest[p.relative_to(root).as_posix()]=hashlib.sha256(raw).hexdigest()
    write(args.prelim_common,'A_bg.csv','tmp/aia/02')
    for k in ca.old.TEAMS:
        for prefix in ['C','aia_challenge']:write(args.prelim_attack,f'{prefix}_{k}.csv','tmp/aia/05')
        write(args.prelim_truth,f'aia_truth_{k}.csv','tmp/aia/08/正解データ')
    write(args.main_team,'A_bg.csv','tmp/main_team28')
    write(args.main_team,'schema.json','tmp/main_team28')
    write(args.main_attack,'mia_columns.json','tmp/main_attack')
    mc=json.loads((root/'tmp/main_attack/mia_columns.json').read_text())
    for k in sorted(map(int,mc['target_columns'])):
        for prefix in ['C','aia_challenge']:write(args.main_attack,f'{prefix}_{k}.csv','tmp/main_attack')
    (root/'input_sha256.json').write_text(json.dumps(manifest,indent=2))
    return root

def csv_bytes(D,pred):
    s=io.StringIO(newline='');w=csv.writer(s,lineterminator='\n');w.writerow(['target_k','challenge_row_id','time_hat'])
    for k in sorted(pred):
        for row,v in zip(D[k][0].challenge_row_id,pred[k]):w.writerow([k,int(row),format(float(v),'.17g')])
    return s.getvalue().encode()

def generate_selected(root):
    pbg,PD=ca.read_phase(root,'prelim');bg,D=ca.read_phase(root,'main')
    PF=ca.old.feature_table(pbg,PD)[0];F=ca.old.feature_table(bg,D)[0]
    truth={k:pd.read_csv(root/f'tmp/aia/08/正解データ/aia_truth_{k}.csv').sort_values('challenge_row_id') for k in PD}
    for k in PD:
        assert PD[k][0].challenge_row_id.tolist()==truth[k].challenge_row_id.tolist()
        assert int(truth[k].label.eq('member').sum())==163 and int(truth[k].is_rare.sum())==26
    X=np.concatenate([PF[k][ca.COLS].to_numpy() for k in PD]);y=np.concatenate([truth[k].label.eq('member').to_numpy() for k in PD])
    T=np.concatenate([F[k][ca.COLS].to_numpy() for k in D]);p={}
    for kind in ['lr0.04','lr10']:
        a=ca.model_prob(X,y,T,kind);p[kind]={k:a[i*326:(i+1)*326] for i,k in enumerate(D)}
    rf=RandomForestClassifier(n_estimators=180,min_samples_leaf=3,max_features=1.,random_state=20261010,n_jobs=2,class_weight='balanced_subsample')
    rf.fit(np.concatenate([ca.clinical(PD[k][0]) for k in PD]),np.concatenate([truth[k].is_rare.to_numpy() for k in PD]))
    result={'lr10_75_rare_soft':{}};audit=[];diagnostics=[]
    for k,(q,c,_) in D.items():
        if k==28:continue
        r=rf.predict_proba(ca.clinical(q))[:,1];mask=ca.rare_mask(q,r)
        base=ca.binary(q,p['lr10'][k],.75,11)
        v=ca.rare_modify(q,base,mask,p['lr0.04'][k]);result['lr10_75_rare_soft'][k]=v
        assert np.array_equal(v[~mask],base[~mask])
        audit.append(dict(team=k,method='lr10_75_rare_soft',rare_candidates=int(mask.sum()),changed=int((v!=base).sum())))
        diagnostics.append(pd.DataFrame({'target_k':k,'challenge_row_id':q.challenge_row_id,'prefecture':q.prefecture,'sex':q.sex,'age':q.age,'membership_LR10':p['lr10'][k],'membership_LR004':p['lr0.04'][k],'rare_score':r,'rare_candidate':mask,'base_time_hat':base,'time_hat':v,'changed_by_rare_group':v!=base}))
    return D,result,audit,pd.concat(diagnostics,ignore_index=True)

def make_zip(path,aia,mia,token):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,raw in [('F_mia.csv',mia),('F_aia.csv',aia),('token.txt',token)]:
            info=zipfile.ZipInfo(name,date_time=(2026,10,10,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o600<<16;z.writestr(info,raw)

def validate(path,attack_dir):
    sys.path.insert(0,str(HERE/'official_validator_snapshot'))
    from pwscup2026.common.config import load_config
    from pwscup2026.kit.validate import check_package,load_dist,validate_aia_submission,validate_mia_submission
    cfg=load_config(HERE/'official_validator_snapshot/pwscup2026/kit/kit_config.yaml')
    dist=load_dist(attack_dir,team_id=28);dist['attacker_record_ids']=set(range(1049))
    package=check_package(path,cfg)
    if not package.ok:raise ValueError(package.errors)
    for name,fn in [('F_aia.csv',validate_aia_submission),('F_mia.csv',validate_mia_submission)]:
        checked=fn(package.frames[name],cfg,dist)
        if not checked.ok:raise ValueError(checked.errors)
    aia=package.frames['F_aia.csv'];targets=set(map(int,dist['mia_columns']['target_columns']))-{28}
    assert set(aia.target_k)==targets and len(aia)==326*len(targets)==7172
    assert np.isin(aia.time_hat,[9.5,11]).all()
    return {'format':'PASS','aia_rows':len(aia),'targets':len(targets),'mia_rows':len(package.frames['F_mia.csv'])}

