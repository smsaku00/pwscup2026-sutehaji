"""2026-09-11公開結果の再評価。入力は読み取り専用、出力はこのフォルダだけ。
実行: プロジェクトのスターターキット既存venvで python -B 分析.py。Google Sheets作成は別途builderで実施。
"""
from pathlib import Path
import sys, io, json, hashlib, zipfile, importlib.util
import numpy as np
import pandas as pd
import openpyxl
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_curve, auc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT/'予備戦/配布データ'
RESULT = DATA/'結果'
TRUTH = next(RESULT.glob('正解*'))
DIST = DATA/'PWSCup2026_prelim_attack'
ANALYSES = ROOT/'予備戦/攻撃フェーズ/分析'
RECORDS = ROOT/'予備戦/攻撃フェーズ/提出記録'
sys.path.insert(0, str(ROOT/'スターターキット/改訂版_20260824/starter/src'))
from pwscup2026.scoring.attack_score import score_mia, score_aia
CFG = {'scoring': {'low_fpr': .01, 'aia_rare_se_z': 1.64}, 'aia': {'tau_time': .5}}
NUM = ['age','SBP','TG','HDL','ALT']
QI = NUM+['sex','prefecture']
FEATURES = ['age','sex','prefecture','BMI','SBP','TG','HDL','ALT','smoking','FPG','time','onset','death']
HASHES = {}

def read(p):
    data = Path(p).read_bytes(); HASHES[str(Path(p).relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return pd.read_csv(io.BytesIO(data), float_precision='round_trip')

def jread(p):
    data=Path(p).read_bytes(); HASHES[str(Path(p).relative_to(ROOT))]=hashlib.sha256(data).hexdigest()
    return json.loads(data)

def frame(ws):
    rows=list(ws.values); return pd.DataFrame(rows[1:],columns=rows[0])

def zipframes(p):
    with zipfile.ZipFile(p) as z:
        out=[]
        for f in ['F_mia.csv','F_aia.csv']:
            name=next(n for n in z.namelist() if n.split('/')[-1]==f)
            data=z.read(name); HASHES[str(Path(p).relative_to(ROOT))+'!'+f]=hashlib.sha256(data).hexdigest()
            out.append(pd.read_csv(io.BytesIO(data),float_precision='round_trip'))
        return out

def mia_eval(attacker,target,f):
    ids=pools[attacker].set_index('record_id').loc[f.record_id]
    y=ids.pool_id.isin(pools[target].pool_id).to_numpy(int)
    rare=ids.is_rare.to_numpy(bool); conf=f[str(target)].to_numpy(float)
    r=score_mia(conf,y,CFG,rare_mask=rare); b=r.breakdown
    # Independent sklearn check covers ROC interpolation and tie handling.
    if 0<y.sum()<len(y):
        fp,tp,_=roc_curve(y,conf,drop_intermediate=False)
        assert abs(np.interp(.01,fp,tp)-b['tpr_at_low_fpr'])<1e-12
    return dict(attacker=attacker,target=target,n_pos=int(y.sum()),n_rare=int(rare.sum()),n_rare_pos=int(y[rare].sum()),
        mia_all=b['tpr_at_low_fpr'],mia_rare=b['tpr_rare'],mia_score=r.score,auc=b['auc'])

def aia_eval(attacker,target,pred):
    t=truths[target]; m=t.label.eq('member').to_numpy(); rare=t.is_rare.to_numpy(bool)
    pred=np.asarray(pred,float); assert len(pred)==len(t) and np.isfinite(pred).all()
    r=score_aia(None,pred[m],t.onset.to_numpy()[m],t.time.to_numpy()[m],None,pred[~m],t.onset.to_numpy()[~m],t.time.to_numpy()[~m],CFG,member_rare_mask=rare[m],control_rare_mask=rare[~m])
    b=r.breakdown; hit=np.abs(pred-t.time.to_numpy())<=.5
    row=dict(attacker=attacker,target=target,aia_all=b['r_time_all'],aia_rare=b['r_time_rare'],aia_rare_raw=b['r_time_rare_raw'],aia_score=r.score)
    for label,mask in [('m',m),('c',~m),('mr',m&rare),('cr',~m&rare)]:
        row['n_'+label]=int(mask.sum()); row['hit_'+label]=int(hit[mask].sum()); row['p_'+label]=float(hit[mask].mean())
        row['abstain_'+label]=int(((pred>10.5)|(pred<-.5))[mask].sum())
    independent=max(0,(row['p_m']-row['p_c'])/max(1-row['p_c'],1e-9))
    assert abs(independent-row['aia_all'])<1e-12
    return row

def evaluate(attacker,mf,af,method,status):
    assert not mf.record_id.duplicated().any()
    assert set(mf.record_id)==set(pools[attacker].record_id)
    assert not af.duplicated(['target_k','challenge_row_id']).any()
    rows=[]
    for k in teams:
        if k==attacker: continue
        row=mia_eval(attacker,k,mf)
        f=af.loc[af.target_k.eq(k)].set_index('challenge_row_id')
        # A missing entire target is reported explicitly; no invented predictions.
        if len(f):
            assert set(f.index)==set(truths[k].challenge_row_id)
            pred=f.loc[truths[k].challenge_row_id,'time_hat'].to_numpy()
            row.update(aia_eval(attacker,k,pred))
        else: row['aia_missing']=True
        row.update(method=method,status=status)
        rows.append(row)
    return rows

def keys(df,cols):
    return [tuple(round(float(v),8) if isinstance(v,(float,np.floating)) else v for v in row) for row in df[cols].itertuples(index=False,name=None)]

def distance(q,c,block='both',log=False):
    qx=q[NUM].to_numpy(float).copy(); cx=c[NUM].to_numpy(float).copy(); bx=bg[NUM].to_numpy(float).copy()
    if log:
        qx[:,[2,4]]=np.log1p(qx[:,[2,4]]);cx[:,[2,4]]=np.log1p(cx[:,[2,4]]);bx[:,[2,4]]=np.log1p(bx[:,[2,4]])
    sd=np.std(bx,axis=0,ddof=1)
    d=cdist(qx/sd,cx/sd)/np.sqrt(len(NUM))
    if block in ['sex','both']: d[q.sex.to_numpy()[:,None]!=c.sex.to_numpy()[None,:]]=np.inf
    if block in ['prefecture','both']: d[q.prefecture.to_numpy()[:,None]!=c.prefecture.to_numpy()[None,:]]=np.inf
    empty=~np.isfinite(d).any(axis=1)
    if empty.any(): d[empty]=cdist(qx[empty]/sd,cx/sd)/np.sqrt(len(NUM))
    return d

def clean(obj):
    if isinstance(obj,dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list,np.ndarray)): return [clean(v) for v in obj]
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,np.bool_):return bool(obj)
    return obj

def save(name,df):
    df.to_csv(HERE/'集計'/f'{name}.csv',index=False)

def main():
    global pools,truths,teams,bg,kit,cs,chs,summary
    (HERE/'集計').mkdir(parents=True,exist_ok=True)
    source=RESULT/'公開用結果表_20260909.xlsx'; HASHES[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    w=openpyxl.load_workbook(source,data_only=True)
    summary=frame(w['総括']); teams=summary.loc[summary.U.notna(),'コホート番号'].astype(int).tolist()
    meta=dict(list(w['メタ'].values)[1:])
    pools={k:read(TRUTH/f'pool_ids_{k}.csv').sort_values('record_id') for k in teams}
    truths={k:read(TRUTH/f'aia_truth_{k}.csv').sort_values('challenge_row_id').reset_index(drop=True) for k in teams}
    cs={k:read(DIST/f'C_{k}.csv') for k in teams}; chs={k:read(DIST/f'aia_challenge_{k}.csv').sort_values('challenge_row_id').reset_index(drop=True) for k in teams}
    bg=read(DATA/'PWSCup2026_prelim_common/A_bg.csv'); kit=read(ROOT/'スターターキット/初版/participant_data/A_bg.csv')
    bself=read(next(DATA.glob('PWSCup2026_prelim_team05*/B_self.csv'))).sort_values('record_id').reset_index(drop=True)
    print('Inputs:',len(bg),len(kit),'background rows,',len(teams),'teams',flush=True)
    for k in teams:
        assert HASHES[str((DIST/f'C_{k}.csv').relative_to(ROOT))]==meta[f'sha256.defense.C_{k}.csv']
        assert list(chs[k].challenge_row_id)==list(truths[k].challenge_row_id)
        assert truths[k].label.value_counts().to_dict()=={'member':163,'control':163} or truths[k].label.value_counts().to_dict()=={'control':163,'member':163}
    official=[]; submitted={}
    for attacker in teams:
        z=RESULT/'攻撃提出物'/f'cohort{attacker:02}_F.zip'; mf,af=zipframes(z); submitted[attacker]=(mf,af)
        for f in ['F_mia.csv','F_aia.csv']:
            assert HASHES[str(z.relative_to(ROOT))+'!'+f]==meta[f'sha256.attack.{attacker}.{f}']
        official+=evaluate(attacker,mf,af,'公開採用回答','公式採用')
    official=pd.DataFrame(official)
    om=frame(w['攻撃_標的別_MIA']); oa=frame(w['攻撃_標的別_AIA'])
    discrepancies=[]
    for sheet,cols in [(om,{'TPR（全体）':'mia_all','TPR（希少な候補だけ）':'mia_rare','t_j（悪いほう）':'mia_score','AUC':'auc','標的のメンバー数':'n_pos','希少なメンバー数':'n_rare_pos'}),
                       (oa,{'R（全体）':'aia_all','R（希少な患者だけ）':'aia_rare','R（希少・誤差の範囲を差し引く前）':'aia_rare_raw','p_m':'p_m','p_c':'p_c','p_m_rare':'p_mr','p_c_rare':'p_cr','失点（悪いほう）':'aia_score'})]:
        for _,v in sheet.iterrows():
            r=official.loc[(official.attacker==v['攻撃者'])&(official.target==v['標的'])].iloc[0]
            for original,col in cols.items():
                if pd.isna(v[original]) and pd.isna(r[col]):continue
                if pd.isna(v[original]) or pd.isna(r[col]) or abs(v[original]-r[col])>1e-10:discrepancies.append([int(v['攻撃者']),int(v['標的']),col,v[original],r[col]])
    assert not discrepancies, discrepancies[:5]
    power=official.groupby('attacker')[['mia_all','mia_rare','mia_score','aia_all','aia_rare','aia_score']].mean().reset_index()
    # Public official table averages the targets actually present for AIA.
    # Attacker 4 has 19 AIA targets; preserve the observed denominator.
    power['total']=power.mia_score+power.aia_score
    official_power=summary.set_index('コホート番号')['攻撃力']
    power['official_total']=power.attacker.map(official_power)
    power['error']=power.total-power.official_total
    assert power.error.abs().max()<1e-10, power.to_string()
    histories=[]
    for n in range(1,9):
        folder=next(RECORDS.glob(f'{n}_*')); mp=list(folder.rglob('F_mia.csv')); ap=list(folder.rglob('F_aia.csv'))
        mf=read(min(mp,key=lambda p:len(p.parts)));af=read(min(ap,key=lambda p:len(p.parts)))
        histories+=evaluate(5,mf,af,f'第{n}回','提出済み')
    history=pd.DataFrame(histories)
    round8=history[history.method.eq('第8回')].set_index('target')
    adopted=official[official.attacker.eq(5)].set_index('target')
    assert np.allclose(round8[['mia_score','aia_score']],adopted[['mia_score','aia_score']],rtol=0,atol=1e-12)
    save('全チーム再採点',official);save('公式総括',summary);save('公式攻撃力照合',power);save('過去8回_対象別',history)
    print('Official scoring reproduced:',len(om),len(oa),'detail rows; power error',power.error.abs().max(),flush=True)

    # Pre-release saved candidates. Never run legacy scripts that mutate submissions.
    variants=[]; mf8,af8=submitted[5]
    p21=jread(next(ANALYSES.glob('21*/比較結果.json')));p22=jread(next(ANALYSES.glob('22*/比較結果.json')))
    hash_errors=[]
    for data in [p21,p22]:
        for path,digest in data.get('inputs_sha256',data.get('input_hashes',{})).items():
            p=ROOT/path
            if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:hash_errors.append(path)
    # Submission README/summary were updated after the saved analysis. Numerical
    # caches remain usable only if every non-document input still matches.
    assert not [p for p in hash_errors if not p.endswith('.md')],hash_errors
    assert p21['record_ids']==mf8.record_id.tolist()==p22['record_ids']
    rare=pools[5].is_rare.to_numpy(bool)
    private_keys=set(keys(chs[5].loc[truths[5].label.eq('member')],QI))
    private=np.array([v in private_keys for v in keys(bself,QI)])
    assert private.sum()==163
    candidate_docs=[]
    for dim,key in [(7,'qi7_gen'),(9,'attrs9'),(10,'attrs10_gen')]:
        for mode in ['生順位','非希少のみ差替']:
            mf=mf8.copy()
            for k in teams:
                if k==5:continue
                raw=np.array(p22['raw_new_scores'][str(k)][key] if dim==9 else p21['raw_scores'][str(k)][key])
                conf=rankdata(raw,method='average')/(len(raw)+1)
                if mode=='生順位':mf[str(k)]=conf
                elif k not in [1,2]:
                    mask=~rare&~private
                    mf.loc[mask,str(k)]=.1+.8*rankdata(raw[mask],method='average')/(mask.sum()+1)
                    mf.loc[~rare&private,str(k)]=0
            name=f'Gen{dim}_{mode}'
            variants+=evaluate(5,mf,af8,name,'未提出・公開前保存スコア')
            candidate_docs.append([name,'MIA',f'分析21/22の保存値。{mode}。差替版はTeam1・2と希少を第8回に固定。','公開前のスコア。組込版の評価は公開後'])
    for plan,assign in p22['plans'].items():
        if plan=='round6':continue
        mf=mf8.copy()
        for k,pair in assign.items():
            # Pre-specified top two candidates. Lower ranks cannot affect FPR=.01
            # when the rare nonmembers number <=24; preserve them at zero.
            mf.loc[rare,k]=0
            mf.loc[mf.record_id.eq(pair[0]),k]=.0002
            mf.loc[mf.record_id.eq(pair[1]),k]=.0001
        name='希少案_'+plan
        variants+=evaluate(5,mf,af8,name,'未提出・公開前固定候補')
        candidate_docs.append([name,'MIA','分析22に保存された各対象上位2人をそのまま使用。非希少は第8回。','公開後の真値による採点'])
    for p in next(ANALYSES.glob('10_*')).glob('*.zip'):
        mf,af=zipframes(p);name=p.stem
        variants+=evaluate(5,mf,af,name,'未採用・既存ZIP')
        candidate_docs.append([name,'MIA+AIA',str(p.relative_to(ROOT)),'旧案を読取のみ。採用・提出を意味しない'])
    for number,pattern in [(16,'期待希少TPRと順位.json'),(18,'候補順位と期待希少TPR.json'),(19,'候補順位と期待希少TPR.json')]:
        source=next(ANALYSES.glob(f'{number}*/{pattern}'))
        saved=pd.DataFrame(jread(source))
        for model,group in saved.groupby('model',sort=False):
            mf=mf8.copy()
            assert len(group)==20
            for _,v in group.iterrows():
                k=str(int(v.target_k)); mf.loc[rare,k]=0
                mf.loc[mf.record_id.eq(v.best_unique_first),k]=.0002
                mf.loc[mf.record_id.eq(v.best_unique_second),k]=.0001
            name=f'分析{number}_{model}'
            variants+=evaluate(5,mf,af8,name,'未提出・公開前のモデル固定候補')
            candidate_docs.append([name,'MIA',str(source.relative_to(ROOT)),f'保存済み最良上位2人を固定。期待値による順位を真値で再採点。非希少とAIAは第8回。'])

    # Existing AIA ablations from analysis 15: import pure inference functions only.
    aia_source=next(ANALYSES.glob('15_*/分析.py'))
    HASHES[str(aia_source.relative_to(ROOT))]=hashlib.sha256(aia_source.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('prior_aia_ablation',aia_source)
    prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
    metric=prior.Metric(bg,NUM)
    # Four already-defined blocking approaches, now evaluated on actual controls.
    aia_predictions={}; structural=[]; feature_stats=[]; self_diagnostics=[]
    bg_full=set(keys(bg,FEATURES));kit_full=set(keys(kit,FEATURES));bg_qi=set(keys(bg,QI));kit_qi=set(keys(kit,QI))
    no_pref=[x for x in FEATURES if x!='prefecture'];clin=['BMI','SBP','TG','HDL','ALT','FPG']
    bg_np=set(keys(bg,no_pref));kit_np=set(keys(kit,no_pref));bg_clin=set(keys(bg,clin));kit_clin=set(keys(kit,clin))
    for k in teams:
        c=cs[k]; q=chs[k]; t=truths[k]; m=t.label.eq('member').to_numpy(); rr=t.is_rare.to_numpy(bool)
        d=distance(q,c); idx=d.argmin(1); dm=d[np.arange(len(q)),idx]
        dbg=distance(q,bg);dkit=distance(q,kit)
        bnearest=dbg.argmin(1); knearest=dkit.argmin(1)
        cqi=set(keys(c,QI))
        corr=c[NUM].corr(method='spearman').to_numpy()
        row=dict(target=k,n_c=len(c),duplicate_rows=int(c.duplicated(FEATURES).sum()),
                 member_exact_qi=int(sum(v in cqi for v in keys(q[m],QI))),
                 control_exact_qi=int(sum(v in cqi for v in keys(q[~m],QI))),
                 member_d_median=float(np.median(dm[m])),control_d_median=float(np.median(dm[~m])),
                 member_close005=int((dm[m]<.05).sum()),control_close005=int((dm[~m]<.05).sum()),
                 nn_membership_auc=float(auc(*roc_curve(m,-dm,drop_intermediate=False)[:2])),
                 member_time_nn_hits=int((np.abs(c.time.to_numpy()[idx[m]]-t.time.to_numpy()[m])<=.5).sum()),
                 control_time_nn_hits=int((np.abs(c.time.to_numpy()[idx[~m]]-t.time.to_numpy()[~m])<=.5).sum()),
                 censored_rate=float(((c.onset==0)&(c.death==0)).mean()),death_rate=float(c.death.mean()),
                 time_unique=int(c.time.nunique()),time_min=float(c.time.min()),time_max=float(c.time.max()),
                 public_bg_rows=int(sum(v in bg_full for v in keys(c,FEATURES))),
                 public_kit_rows=int(sum(v in kit_full for v in keys(c,FEATURES))),
                 public_bg_qi=int(sum(v in bg_qi for v in keys(c,QI))),
                 public_kit_qi=int(sum(v in kit_qi for v in keys(c,QI))),
                 public_bg_no_pref=int(sum(v in bg_np for v in keys(c,no_pref))),
                 public_kit_no_pref=int(sum(v in kit_np for v in keys(c,no_pref))),
                 public_bg_clinical6=int(sum(v in bg_clin for v in keys(c,clin))),
                 public_kit_clinical6=int(sum(v in kit_clin for v in keys(c,clin))),
                 mean_abs_spearman_qi5=float(np.abs(corr[np.triu_indices(5,1)]).mean()))
        # Pairwise linkage residuals, with truth used only for the post-hoc diagnostic.
        for label,mask in [('member',m),('rare_member',m&rr),('control',~m)]:
            paired=c.iloc[idx[mask]].reset_index(drop=True); orig=q.loc[mask].reset_index(drop=True)
            for col in NUM:
                residual=paired[col].to_numpy()-orig[col].to_numpy()
                feature_stats.append(dict(target=k,group=label,column=col,metric='QI最近傍残差',n=len(residual),median=float(np.median(residual)),q10=float(np.quantile(residual,.1)),q90=float(np.quantile(residual,.9)),exact_rate=float(np.isclose(residual,0,atol=1e-8,rtol=0).mean())))
        for col in FEATURES:
            a=set(keys(bg,[col]));b=set(keys(kit,[col]));v=keys(c,[col]);
            feature_stats.append(dict(target=k,group='C全行',column=col,metric='値の種類と背景支持',n=len(c),unique=int(c[col].nunique()),bg_support=float(np.mean([x in a for x in v])),kit_support=float(np.mean([x in b for x in v]))))
        structural.append(row)
        for block in ['none','sex','prefecture','both']:
            dd=distance(q,c,block); pred=c.time.to_numpy()[dd.argmin(1)]
            name='AIA_NN_'+block; aia_predictions[(name,k)]=pred
            if k!=5:
                mr=round8.loc[k].to_dict();mr.update(aia_eval(5,k,pred));mr.update(method=name,status='未提出比較・既存方式を全対象再実行');variants.append(mr)
        inferred,_,_,_,qprob,options=prior.aia_inference(k,chs,cs,bg,metric)
        # These two ablations were specified before the release of truth data.
        extras={'AIA_標的確率0.5':prior.optimize_time(np.full(len(q),.5),options)[0],
                'AIA_C時間窓のみ':np.array([values[np.argmax(cm)] for values,cm,bm in options])}
        for name,pred in extras.items():
            aia_predictions[(name,k)]=pred
            if k!=5:
                mr=round8.loc[k].to_dict();mr.update(aia_eval(5,k,pred));mr.update(method=name,status='未提出比較・公開前定義のAIA対照');variants.append(mr)
        # Constant baseline audits background-independent predictability.
        pred=np.full(len(q),9.75);aia_predictions[('AIA_定数9.75',k)]=pred
        if k!=5:
            mr=round8.loc[k].to_dict();mr.update(aia_eval(5,k,pred));mr.update(method='AIA_定数9.75',status='対照用の固定方式');variants.append(mr)
        if k==5:
            ownkey=set(keys(bself,FEATURES));row['own_B_exact_rows']=int(sum(v in ownkey for v in keys(c,FEATURES)))
            for a,(mf,af) in submitted.items():
                if a==5:continue
                f=af.loc[af.target_k.eq(5)].set_index('challenge_row_id');pred=f.loc[t.challenge_row_id,'time_hat'].to_numpy()
                hit=np.abs(pred-t.time.to_numpy())<=.5
                for label,mask in [('全体標的',m),('全体対照',~m),('希少標的',m&rr),('希少対照',~m&rr)]:
                    self_diagnostics.append(dict(attacker=a,group=label,n=int(mask.sum()),hits=int(hit[mask].sum()),constant975_hits=int((np.abs(9.75-t.time.to_numpy()[mask])<=.5).sum()),
                        predicted_high=int(((pred>=9.5)&(pred<=10.5))[mask].sum()),predicted_abstain=int(((pred>10.5)|(pred<-.5))[mask].sum()),
                        constant_but_lost=int(((np.abs(9.75-t.time.to_numpy())<=.5)&~hit)[mask].sum()),gained_nonconstant=int(((np.abs(9.75-t.time.to_numpy())>.5)&hit)[mask].sum())))
    variants=pd.DataFrame(variants);structural=pd.DataFrame(structural); feature_stats=pd.DataFrame(feature_stats)
    save('未採用手法_対象別',variants);save('加工推定_チーム別',structural);save('加工推定_属性別',feature_stats);save('自チームAIA_正解内訳',pd.DataFrame(self_diagnostics))
    save('自チーム被攻撃',official[official.target.eq(5)].sort_values('aia_score',ascending=False))
    allself=pd.concat([history,variants],ignore_index=True)
    methods=allself.groupby(['method','status'],sort=False).agg(n_targets=('target','size'),mia_all=('mia_all','mean'),mia_rare=('mia_rare','mean'),mia_score=('mia_score','mean'),aia_all=('aia_all','mean'),aia_rare=('aia_rare','mean'),aia_score=('aia_score','mean')).reset_index()
    methods['total']=methods.mia_score+methods.aia_score
    methods['delta_round8']=methods.total-float(power.loc[power.attacker.eq(5),'total'].iloc[0])
    save('手法比較_集約',methods)
    # Hindsight selection is explicitly distinct from pre-release fixed candidates.
    oracle_rows=[]
    for label,df in [('提出8回から事後選択',history),('今回比較候補から事後選択',allself)]:
        for k,g in df.groupby('target'):
            mm=g.loc[g.mia_score.idxmax()];aa=g.loc[g.aia_score.idxmax()]
            oracle_rows.append(dict(scope=label,target=k,mia_method=mm.method,mia_score=mm.mia_score,aia_method=aa.method,aia_score=aa.aia_score,total_contribution=(mm.mia_score+aa.aia_score)/20))
    save('事後選択の上限',pd.DataFrame(oracle_rows))
    # Check actual C5 provenance by complete file hashes, not an old README label.
    lineage=[]
    for p in (ROOT/'予備戦/加工フェーズ/提出記録').rglob('C.csv'):
        candidate=read(p)
        ckeys=set(keys(candidate,FEATURES))
        lineage.append(dict(path=str(p.relative_to(ROOT)),sha256=HASHES[str(p.relative_to(ROOT))],
           file_equal=HASHES[str(p.relative_to(ROOT))]==meta['sha256.defense.C_5.csv'],
           shared_full_rows=sum(v in ckeys for v in keys(cs[5],FEATURES))))
    save('自チーム加工版の照合',pd.DataFrame(lineage))
    save('候補の定義',pd.DataFrame(candidate_docs,columns=['method','attack','definition','interpretation']))
    audit=dict(official_mia_rows=len(om),official_aia_rows=len(oa),all_pair_rows=len(official),aia_missing=official.loc[official.aia_score.isna(),['attacker','target']].to_dict('records'),
               official_score_discrepancies=discrepancies,official_power_max_error=float(power.error.abs().max()),
               background_counts={'prelim':len(bg),'kit':len(kit)},background_exact_overlap=len(set(keys(bg,FEATURES))&set(keys(kit,FEATURES))),
               all_C_hashes_match=True,all_published_F_hashes_match=True,saved_candidate_numerical_input_hashes_match=True,changed_document_hashes=sorted(set(hash_errors)),
               self_C=structural.loc[structural.target.eq(5)].to_dict('records')[0],
               self_C_exact_candidate=[r['path'] for r in lineage if r['file_equal']],
               output_methods=len(methods),inputs_sha256=HASHES)
    (HERE/'検証結果.json').write_text(json.dumps(clean(audit),ensure_ascii=False,indent=2))
    print(methods[['method','mia_score','aia_score','total','delta_round8']].to_string(index=False),flush=True)
    print('Structural\n',structural.to_string(index=False),flush=True)

if __name__=='__main__':main()
