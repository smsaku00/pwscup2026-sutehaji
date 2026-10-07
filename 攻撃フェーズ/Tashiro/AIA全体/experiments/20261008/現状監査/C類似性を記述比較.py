from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
BASE=Path('/Users/taisei1105/pws2026-local')
DEST=BASE/'本戦/攻撃フェーズ/Tashiro/20261008_AIA_C特徴と方式探索/現状監査'
PHASES={'main':(BASE/'本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力',BASE/'本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv'),'prelim':(BASE/'予備戦/配布データ/PWSCup2026_prelim_attack',BASE/'予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv')}
kit=BASE/'スターターキット/初版/participant_data/A_bg.csv'
NUM=['age','SBP','TG','HDL','ALT']
records=[]; inputs=[]
def sha(p):
    b=p.read_bytes(); return dict(path=str(p.relative_to(BASE)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def transform(f):
    x=f[NUM].to_numpy(float).copy();x[:,[2,4]]=np.log1p(np.maximum(x[:,[2,4]],0));return x
def nd(a,xa,b,xb):
    d=np.full(len(a),np.inf)
    for sex in [0,1]:
        ia=a.sex.eq(sex).to_numpy();ib=b.sex.eq(sex).to_numpy()
        if ia.any() and ib.any():d[ia]=cKDTree(xb[ib]).query(xa[ia])[0]/np.sqrt(xa.shape[1])
    return d
stable=pd.read_csv(BASE/'本戦/攻撃フェーズ/Tashiro/20261007_AIA全体向上の分析/集計/同点修正_対象別.csv')
stable=stable[stable.method.eq('stable_matching/sex_quota/long75')].set_index('team')
baseline=pd.read_csv(BASE/'本戦/攻撃フェーズ/Tashiro/20261007_AIA全体向上の分析/集計/予備戦_全設定と対象別.csv')
baseline=baseline[baseline.method.eq('S4_log_iqr_soft/nn_gate0.75_0')].set_index('team')
for phase,(src,bgp) in PHASES.items():
    bg=pd.read_csv(bgp);xbg=transform(bg);iqr=np.maximum(np.subtract(*np.percentile(xbg,[75,25],axis=0)),1e-8);xbg=xbg/iqr;kitbg=pd.read_csv(kit);xkit=transform(kitbg)/iqr
    inputs.extend([sha(bgp),sha(kit)])
    for p in sorted(src.glob('C_*.csv'),key=lambda p:int(p.stem[2:])):
        k=int(p.stem[2:]);c=pd.read_csv(p);q=pd.read_csv(src/f'aia_challenge_{k}.csv');xc=transform(c)/iqr;xq=transform(q)/iqr
        dc=nd(c,xc,bg,xbg);dk=nd(c,xc,kitbg,xkit);dq=nd(q,xq,c,xc);db=nd(q,xq,bg,xbg);dqa=nd(q,xq[:,1:],c,xc[:,1:])
        inputs.extend([sha(p),sha(src/f'aia_challenge_{k}.csv')])
        r=dict(phase=phase,team=k,C_rows=len(c),age_unique=int(c.age.nunique()),age_values=sorted(c.age.unique().tolist()) if c.age.nunique()<18 else None,
        numeric_unique_min=min(int(c[a].nunique()) for a in NUM[1:]),numeric_unique_max=max(int(c[a].nunique()) for a in NUM[1:]),
        male_C=float(c.sex.eq(1).mean()),male_Q=float(q.sex.eq(1).mean()),C_pref_Q_share=float(c.prefecture.isin(q.prefecture.unique()).mean()),C_time_ge9=float(c.time.ge(9).mean()),
        C_to_phaseBG_median=float(np.median(dc)),C_to_phaseBG_d_lt001=float((dc<.001).mean()),C_to_kitBG_d_lt001=float((dk<.001).mean()),
        Q_to_C_median=float(np.median(dq)),Q_to_BG_median=float(np.median(db)),Q_median_bg_over_c=float(np.median(db/np.maximum(dq,1e-12))),
        Q_to_C_without_age_median=float(np.median(dqa)))
        if phase=='prelim':
            r.update(P01_raw_all=float(stable.loc[k,'r_all']),S4_base_raw_all=float(baseline.loc[k,'r_all']))
        records.append(r)
out=dict(scope='記述・観測後の類似性比較。特徴閾値/タイプから方式選択する規則を性能評価した結果ではない。相手真値不使用。本戦のP01採用先の成績との対応は非無作為割当。',features='age・SBP・log1p(TG)・HDL・log1p(ALT)を各フェーズBGのIQRで標準化。同一性別内の最近傍距離RMS。prefectureは距離に含めない。d_lt001の閾値.001は近接診断。',records=records,inputs=inputs)
DEST.mkdir(parents=True,exist_ok=True)
(DEST/'予備戦と本戦のC類似性_記述比較.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'record_count':len(records),'selected':[r for r in records if (r['phase']=='main' and r['team'] in [3,6,8,9,10,14]) or(r['phase']=='prelim' and r['team'] in [1,2,18,19,24,26,29])]},ensure_ascii=False))
