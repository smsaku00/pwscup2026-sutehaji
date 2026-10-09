#!/usr/bin/env python3
"""PWS CUP 2026 AIA: held-out-cohort member/control evidence and constrained decisions.

All features/predictions depend on public Q/C/A_bg only. Truth is loaded into an isolated
scoring routine after frozen predictions are made; one cohort's labels never train its model.
No participant private B, secret token, submission bundle, or CodaBench action.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

SEED=20261009
CLIN=['SBP','TG','HDL','ALT']
TEAMS=[1,2,3,4,5,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]
GROUPS=[('5y',5),('10y',10)]
POLICY=['base_0','base_11','cut_control_025','cut_control_040','pair_only_025','pair_only_040','multi_extreme_025','multi_extreme_040','global_half']

def clipping_r(a,b):
    return float(np.clip((a-b)/max(1-b,1e-9),0,1))

def rrare(a,b):
    se=np.sqrt((a*(1-a)+b*(1-b))/13)
    return float(np.clip((a-b-1.64*se)/max(1-b,1e-9),0,1))

def read_data(root):
    paths=root/'公式データ'/'予備戦'
    bg=pd.read_csv(paths/'PWSCup2026_prelim_common/A_bg.csv')
    D={}
    for k in TEAMS:
        q=pd.read_csv(paths/f'PWSCup2026_prelim_attack/aia_challenge_{k}.csv').sort_values('challenge_row_id').reset_index(drop=True)
        c=pd.read_csv(paths/f'PWSCup2026_prelim_attack/C_{k}.csv')
        t=pd.read_csv(paths/f'結果/正解データ/aia_truth_{k}.csv').sort_values('challenge_row_id').reset_index(drop=True)
        assert len(q)==len(t)==326 and len(c)==1049
        assert q.challenge_row_id.equals(t.challenge_row_id), k
        assert int((t.label=='member').sum())==int((t.label=='control').sum())==163
        assert int(((t.label=='member')&t.is_rare.eq(1)).sum())==int(((t.label=='control')&t.is_rare.eq(1)).sum())==13
        D[k]=(q,c,t)
    return bg,D

def vtrans(df,cols=CLIN):
    x=df[list(cols)].to_numpy(dtype=float).copy()
    for n in ('TG','ALT'):
        if n in cols: x[:,list(cols).index(n)] = np.log1p(np.maximum(0,x[:,list(cols).index(n)]))
    return x

def near_1d(s,v):
    s=np.asarray(s,float)
    if not len(s): return np.full(len(v),1e7)
    s=np.sort(s)
    k=np.searchsorted(s,v,side='left')
    out=np.full(len(v),np.inf)
    ok=k<len(s); out[ok]=np.abs(v[ok]-s[k[ok]])
    ok=k>0;out[ok]=np.minimum(out[ok],np.abs(v[ok]-s[k[ok]-1]))
    return out

def k_neighbors(src,dest,k=10):
    if len(src)==0: return np.full((len(dest),k),1e4)
    dis,_=cKDTree(src).query(dest,k=min(k,len(src)),workers=1)
    dis=np.asarray(dis,float)
    if dis.ndim==1:dis=dis[:,None]
    if dis.shape[1]<k:dis=np.pad(dis,((0,0),(0,k-dis.shape[1])),constant_values=1e4)
    return dis

def c_partial2(q5,c5,sexQ,sexC):
    out=np.full(len(q5),1e5)
    for s in (0,1):
        iq=np.where(sexQ==s)[0];ic=np.where(sexC==s)[0]
        if not len(iq) or not len(ic):continue
        # For each donor, keep the two smallest of age and clinical squared deviations.
        p=((q5[iq,None,:]-c5[None,ic,:])**2)
        d=np.partition(p,1,axis=2)[:,:,:2].mean(axis=2)
        out[iq]=np.sqrt(d.min(axis=1))
    return out

def feature_table(bg,D):
    b=vtrans(bg)
    iqr=np.maximum(np.quantile(b,.75,axis=0)-np.quantile(b,.25,axis=0),1e-5)
    age_scale=max(float(np.quantile(bg.age,.75)-np.quantile(bg.age,.25)),1e-5)
    otherpool={}
    # Same public C across different target datasets; stable subsampling.
    for k,(q,c,t) in D.items():otherpool[k]=c.iloc[::6,:].copy()
    sets={};labels={};
    for k,(q,c,t) in D.items():
        x=vtrans(q);cx=vtrans(c)
        c5=np.column_stack((c.age.to_numpy(float)/age_scale,cx/iqr))
        q5=np.column_stack((q.age.to_numpy(float)/age_scale,x/iqr))
        bx=np.column_stack((bg.age.to_numpy(float)/age_scale,b/iqr))
        same=q.sex.to_numpy(int); cs=c.sex.to_numpy(int); bs=bg.sex.to_numpy(int)
        x4=x/iqr;cx4=cx/iqr;b4=b/iqr
        dC=np.full((len(q),10),1e4);dB=dC.copy();dO=dC.copy()
        dC5=np.full(len(q),1e4);dB5=dC5.copy()
        pool=pd.concat([s for j,s in otherpool.items() if j!=k],ignore_index=True)
        ox=vtrans(pool)/iqr; os=pool.sex.to_numpy(int)
        for s in [0,1]:
            ii=np.where(same==s)[0]
            if not len(ii):continue
            dC[ii]=k_neighbors(cx4[cs==s],x4[ii],10)
            dB[ii]=k_neighbors(b4[bs==s],x4[ii],10)
            dO[ii]=k_neighbors(ox[os==s],x4[ii],10)
            dC5[ii]=k_neighbors(c5[cs==s],q5[ii],1)[:,0]
            dB5[ii]=k_neighbors(bx[bs==s],q5[ii],1)[:,0]
        colC=np.zeros((len(q),4));colB=colC.copy()
        for s in (0,1):
            ii=np.where(same==s)[0]
            if not len(ii):continue
            for j in range(4):
                colC[ii,j]=near_1d(cx[cs==s,j]/iqr[j],x[ii,j]/iqr[j])
                colB[ii,j]=near_1d(b[bs==s,j]/iqr[j],x[ii,j]/iqr[j])
        colEvidence=np.log((colB+0.001)/(colC+0.001))
        # Distinct positive and negative evidence: 1-NN C and public background / other Cs.
        part2=c_partial2(q5,c5,same,cs)
        log=lambda x:np.log1p(np.maximum(x,0))
        f=pd.DataFrame({
            'member_near4':-log(dC[:,0]),'member_knn4':-log(dC[:,4]),
            'member_near5':-log(dC5),'member_partial2':-log(part2),
            'bg_near4':-log(dB[:,0]),'bg_knn4':-log(dB[:,4]),
            'bg_near5':-log(dB5),'otherc_near4':-log(dO[:,0]),
            'contrast_near4':log(dB[:,0])-log(dC[:,0]),
            'contrast_knn4':log(dB[:,4])-log(dC[:,4]),
            'contrast_near5':log(dB5)-log(dC5),
            'contrast_otherc':log(dO[:,0])-log(dC[:,0]),
            'col4':np.mean(colEvidence,axis=1),
            'col2':np.sort(colEvidence,axis=1)[:,-2:].mean(axis=1),
            'col1':colEvidence.max(axis=1),
            'col4_min':colEvidence.min(axis=1),
            'age':q.age.to_numpy(float), 'sex':same.astype(float),
            'raw_sbp':x[:,0]/iqr[0], 'raw_tg':x[:,1]/iqr[1],
            'raw_hdl':x[:,2]/iqr[2], 'raw_alt':x[:,3]/iqr[3],
        })
        for j,n in enumerate(CLIN):
            f[f'c_{n.lower()}']=-log(colC[:,j]);f[f'bg_{n.lower()}']=-log(colB[:,j]);f[f'ratio_{n.lower()}']=colEvidence[:,j]
        base_cols=['member_near4','member_knn4','member_near5','member_partial2',
                   'bg_near4','bg_knn4','contrast_near4','contrast_knn4','contrast_near5',
                   'contrast_otherc','col4','col2','col1']
        for n in base_cols:
            f['rank_'+n]=rankdata(f[n].to_numpy(),method='average')/len(f)
        f['team']=k;f['row_id']=q.challenge_row_id.to_numpy(int)
        assert np.isfinite(f.to_numpy(float)).all()
        sets[k]=f;labels[k]=t
    return sets,labels

def modeling(feats, truths):
    # Explicit feature provenance: BG-only is a source-artifact negative control.
    qbase=['age','sex','raw_sbp','raw_tg','raw_hdl','raw_alt']
    member=['member_near4','member_knn4','member_near5','member_partial2',
            'rank_member_near4','rank_member_near5','rank_member_partial2']+[f'c_{n.lower()}' for n in CLIN]
    bg=['bg_near4','bg_knn4','bg_near5','rank_bg_near4','rank_bg_knn4']+[f'bg_{n.lower()}' for n in CLIN]
    ratio=['contrast_near4','contrast_knn4','contrast_near5','col4','col2','col1','col4_min',
           'rank_contrast_near4','rank_col4','rank_col2','rank_col1']+[f'ratio_{n.lower()}' for n in CLIN]
    other=['otherc_near4','contrast_otherc','rank_contrast_otherc']
    specifications={
       'q_only':('logistic',qbase),
       'bg_only':('logistic',qbase+bg),
       'c_only':('logistic',member),
       'c_bg':('logistic',member+bg+ratio),
       'c_bg_other':('logistic',member+bg+ratio+other),
       'c_bg_hgb':('hgb',member+bg+ratio),
    }
    allpred={k:{} for k in feats}
    for scheme,(family,cols) in specifications.items():
        for k in feats:
            train_keys=[j for j in feats if j!=k]
            tr=pd.concat([feats[j] for j in train_keys],ignore_index=True)
            Y=np.concatenate([(truths[j].label=='member').to_numpy(int) for j in train_keys])
            if family=='logistic':model=make_pipeline(StandardScaler(),LogisticRegression(C=.04,max_iter=1500,solver='lbfgs'))
            else:model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=10,
                    min_samples_leaf=85,l2_regularization=20,learning_rate=.045,random_state=SEED)
            model.fit(tr[cols].to_numpy(float),Y)
            p=model.predict_proba(feats[k][cols].to_numpy(float))[:,1]
            assert np.isfinite(p).all()
            allpred[k][scheme]=p
    return allpred,specifications

def pairs(q,w):
    d={}
    for i,r in q.iterrows():
        k=(str(r.prefecture),int(r.sex),int(r.age//w))
        d.setdefault(k,[]).append(i)
    return d

def top75(q,score):
    chosen=set()
    for sex in (0,1):
        ix=np.where(q.sex.to_numpy(int)==sex)[0]
        arr=sorted(ix,key=lambda i:(-float(score[i]),int(q.challenge_row_id.iloc[i])))
        chosen.update(arr[:math.ceil(.75*len(arr))])
    return chosen

def do_policy(q,prob,baseline,ptype,w=5,zero=0):
    selected=set(baseline)
    assert 0<=zero<=11
    if ptype=='base_0':return np.where(np.isin(np.arange(len(q)),list(selected)),9.5,0).astype(float),[]
    if ptype=='base_11':return np.where(np.isin(np.arange(len(q)),list(selected)),9.5,11).astype(float),[]
    pred=np.where(np.isin(np.arange(len(q)),list(selected)),9.5,0).astype(float)
    if ptype=='global_half':
        a=sorted(range(len(q)),key=lambda i:(-prob[i],int(q.challenge_row_id.iloc[i])))
        return np.where(np.isin(np.arange(len(q)),a[:163]),9.5,11).astype(float),[]
    if ptype.startswith('cut_control_'):
        cut=.25 if ptype.endswith('025') else .4
        changed=[]
        for i in sorted(selected):
            if prob[i]<cut:pred[i]=11;changed.append((i,'threshold'))
        return pred,changed
    groups=pairs(q,w)
    cut=.25 if ptype.endswith('025') else .4
    changed=[]
    for grp in groups.values():
        if len(grp)<2 or len(grp)%2:continue
        lo=sorted(grp,key=lambda i:(prob[i],int(q.challenge_row_id.iloc[i])))
        hi=lo[::-1]
        if ptype.startswith('pair_only') and len(grp)!=2:continue
        # Balanced groups give an opportunity for RELATIVE evidence, but no label is forced.
        m=len(lo)//2
        # Require group-internal contrast; a control candidate has probability clearly below 0.5.
        for j in range(m):
            a=lo[j];b=hi[j]
            if prob[a]>=cut or prob[b]<.60 or prob[b]-prob[a]<.25:continue
            if a in selected:
                pred[a]=11; changed.append((a,'group'))
    return pred,changed

def metrics(p,t):
    m=t.label.eq('member').to_numpy();c=~m;rar=t.is_rare.to_numpy(bool)
    hit=np.abs(np.asarray(p)-t.time.to_numpy(float))<=.5
    pm=hit[m].mean();pc=hit[c].mean();pmr=hit[m&rar].mean();pcr=hit[c&rar].mean()
    ra=clipping_r(pm,pc);rr=rrare(pmr,pcr)
    return {'r_all':ra,'r_rare':rr,'r_max':max(ra,rr),'pm':pm,'pc':pc,
            'pmrare':pmr,'pcrare':pcr,'hit_member':int(hit[m].sum()),'hit_control':int(hit[c].sum()),
            'hit_member_rare':int(hit[m&rar].sum()),'hit_control_rare':int(hit[c&rar].sum())}

def bootstrap_gain(df,scheme,policy='pair_only_040',n=20000):
    sub=df[(df.scheme==scheme)&(df.policy.isin(['base_0',policy])) & df.team.ne(5)]
    pivot=sub.pivot(index='team',columns='policy',values='r_all').dropna()
    d=pivot[policy].to_numpy()-pivot['base_0'].to_numpy();rng=np.random.default_rng(SEED)
    ix=rng.integers(0,len(d),size=(n,len(d)))
    ci=np.quantile(d[ix].mean(axis=1),[.025,.975])
    return [float(d.mean()),float(ci[0]),float(ci[1]),int((d>1e-12).sum()),int((d<-1e-12).sum()),int((np.abs(d)<=1e-12).sum())]

def main(out,archive,root):
    out.mkdir(parents=True,exist_ok=True)
    if archive:
        assert not root or not root.exists(), 'Do not overwrite an existing input directory'
        root=out/'private_inputs'
        root.mkdir(parents=True,exist_ok=True)
        with ZipFile(archive) as z:
            # Only expect official input manifest, no private files.
            for f in z.namelist():
                if f.startswith('公式データ/予備戦/') or f=='input_sha256.json':z.extract(f,root)
    assert root and root.exists(),root
    # Official GitHub checkout has the repository SHA256SUMS rather than the
    # GitHub Actions artifact's input_sha256.json.
    saved=root/'input_sha256.json'
    if saved.exists():
        manifest=json.loads(saved.read_text())
    else:
        official=root/'公式データ/予備戦'
        parsed={name:d for d,name in (line.strip().split(maxsplit=1)
              for line in (official/'SHA256SUMS.txt').read_text().splitlines() if line.strip())}
        input_files=list((official/'PWSCup2026_prelim_attack').glob('C_*.csv'))
        input_files+=list((official/'PWSCup2026_prelim_attack').glob('aia_challenge_*.csv'))
        input_files+=list((official/'結果/正解データ').glob('aia_truth_*.csv'))
        input_files+=[official/'PWSCup2026_prelim_common/A_bg.csv',official/'PWSCup2026_prelim_common/schema.json']
        assert len(input_files)==65
        manifest={p.relative_to(root).as_posix():parsed[p.relative_to(official).as_posix()] for p in input_files}
    shaok=sum(hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in manifest.items())
    assert len(manifest)==65 and shaok==len(manifest)
    bg,D=read_data(root)
    print('Input checks:',shaok,'files, teams:',len(D),flush=True)
    feats,truths=feature_table(bg,D)
    pred,specs=modeling(feats,truths)
    public=[]
    for k,(q,c,t) in D.items():
        f=feats[k]
        z=pd.DataFrame({'team':k,'row_id':q.challenge_row_id,'sex':q.sex,
            'prefecture':q.prefecture,'age':q.age,
            'near4':f['member_near4'],'col4':f['col4'],'col2':f['col2']})
        for nm,v in pred[k].items(): z[nm]=v
        public.append(z)
    pd.concat(public,ignore_index=True).to_csv(out/'public_scores.csv',index=False)
    rows=[];group_audit=[];direction=[];changed_rows=[];calibration=[]
    for k in TEAMS:
        q=D[k][0];t=truths[k];y=t.label.eq('member').to_numpy(int)
        s={
            'near4':feats[k]['member_near4'].to_numpy(),
            'col4':feats[k]['col4'].to_numpy(),
            'col2':feats[k]['col2'].to_numpy(),
            **pred[k]
        }
        mask={p:top75(q,score) for p,score in s.items()}
        for name,prob in pred[k].items():
            e=(y-prob);calibration.append({'team':k,'model':name,'auc':roc_auc_score(y,prob),
                'brier':brier_score_loss(y,prob),'mae':np.mean(np.abs(e)),'pred_min':prob.min(),'pred_max':prob.max(),
                'low025':int((prob<.25).sum()),'low040':int((prob<.4).sum())})
        pg=pairs(q,5)
        for key,idx in pg.items():
            g=t.iloc[idx]; n=len(idx);mm=int(g.label.eq('member').sum());cc=n-mm
            group_audit.append({'team':k,'prefecture':key[0],'sex':key[1],'age_bin':key[2]*5,
                'n':n,'m':mm,'c':cc,'balanced':mm==cc,'candidate2':n==2,
                'has_rare_member':int(((g.label=='member')&(g.is_rare==1)).sum())})
            if n!=2:continue
            record={'team':k,'row_a':int(q.challenge_row_id.iloc[idx[0]]),'row_b':int(q.challenge_row_id.iloc[idx[1]]),
                    'truth_balanced':mm==1}
            for model,score in s.items():
                a,b=idx
                direction_pred=(score[a]>score[b])
                if score[a]==score[b]:result=float('nan')
                else:result=float(y[a]==direction_pred)
                record[f'{model}_correct']=result
                record[f'{model}_gap']=float(abs(score[a]-score[b]))
            direction.append(record)
        for sch,score in s.items():
            base=mask[sch]
            # Model probability is used to decide selective interventions; score rank is base.
            p=pred[k].get(sch,pred[k]['c_bg_other'])
            for policy in POLICY:
                vals,modified=do_policy(q,p,base,policy,w=5)
                z=metrics(vals,t)
                row={'team':k,'scheme':sch,'policy':policy,'changed_count':int((vals!=do_policy(q,p,base,'base_0')[0]).sum()),**z}
                rows.append(row)
                if modified:
                    for i,why in modified:
                        prev=9.5 if i in base else 0
                        hitpre=abs(prev-t.time.iloc[i])<=.5
                        hitpost=abs(vals[i]-t.time.iloc[i])<=.5
                        changed_rows.append({'team':k,'scheme':sch,'policy':policy,
                            'row_id':int(q.challenge_row_id.iloc[i]),'reason':why,
                            'true_label':str(t.label.iloc[i]),'true_is_rare':int(t.is_rare.iloc[i]),
                            'old_hit':bool(hitpre),'new_hit':bool(hitpost),
                            'old_time_hat':float(prev),'new_time_hat':float(vals[i]),
                            'p_member_oof':float(p[i])})
    d=pd.DataFrame(rows);d.to_csv(out/'per_target.csv',index=False)
    g=pd.DataFrame(group_audit);g.to_csv(out/'group_audit.csv',index=False)
    pairdf=pd.DataFrame(direction);pairdf.to_csv(out/'pair_direction.csv',index=False)
    cr=pd.DataFrame(changed_rows);cr.to_csv(out/'changed_rows.csv',index=False)
    ca=pd.DataFrame(calibration);ca.to_csv(out/'model_calibration.csv',index=False)
    for label,ids in [('20', [k for k in TEAMS if k!=5]),('18',[k for k in TEAMS if k not in (1,2,5)])]:
        v=d[d.team.isin(ids)].groupby(['scheme','policy'],sort=False).agg(
            n=('team','nunique'),r_all=('r_all','mean'),r_rare=('r_rare','mean'),
            r_max=('r_max','mean'),pm=('pm','mean'),pc=('pc','mean'),changed=('changed_count','sum')).reset_index()
        v.to_csv(out/f'summary_{label}.csv',index=False)
    # Model/pair contrasts: test whether independent evidence actually helps.
    cb=[]
    for m in pred[TEAMS[0]]:
        v0=ca[(ca.model==m)&ca.team.ne(5)]
        cb.append({'model':m,'mean_auc':v0.auc.mean(),'mean_brier':v0.brier.mean(),'cohorts':len(v0)})
    pd.DataFrame(cb).to_csv(out/'model_summary.csv',index=False)
    consistency=[]
    for mod in s:
        v0=pairdf[pairdf.team.ne(5)]
        valid=v0[f'{mod}_correct'].dropna()
        consistency.append({'score':mod,'pairs':len(valid),'orientation_accuracy':float(valid.mean()) if len(valid) else None})
    pd.DataFrame(consistency).to_csv(out/'pair_summary.csv',index=False)
    boosts=[]
    for sch in s:
        for pol in ('cut_control_025','cut_control_040','pair_only_025','pair_only_040','multi_extreme_025','multi_extreme_040'):
            boosts.append(dict(scheme=sch,policy=pol,mean_gain_and_ci=bootstrap_gain(d,sch,pol)))
    (out/'confidence_intervals.json').write_text(json.dumps(boosts,ensure_ascii=False,indent=2))
    (out/'method_manifest.json').write_text(json.dumps({'seed':SEED,'input_files_verified':shaok,'input_sha256':manifest,
            'models':{k:{'family':a,'columns':b}for k,(a,b)in specs.items()},
            'policy':POLICY,'source':'official preliminary 2026-10-09 snapshot','truth_used_for_feature_training_on_target':False,
            'study_is_posthoc':True,'codabench_submitted':False},ensure_ascii=False,indent=2))
    print('Model summary:',pd.DataFrame(cb).round(4).to_string(index=False),flush=True)
    print('Pair direction:',pd.DataFrame(consistency).round(4).to_string(index=False),flush=True)
    print('Mean scores main 20:',(pd.read_csv(out/'summary_20.csv').query("policy in ['base_0','cut_control_040','pair_only_040','multi_extreme_040']").pivot(index='scheme',columns='policy',values='r_all')).round(4).to_string(),flush=True)
    print('Output files:',len(list(out.glob('*'))),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(__file__).parent/'input')
    ap.add_argument('--out',type=Path,default=Path(__file__).parent/'output')
    args=ap.parse_args()
    main(args.out,None,args.root)
