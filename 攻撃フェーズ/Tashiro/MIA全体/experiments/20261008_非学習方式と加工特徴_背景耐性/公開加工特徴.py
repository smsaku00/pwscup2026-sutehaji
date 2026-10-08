"""公開C/A/Qから加工の痕跡を集約する。所属・AIA正解は読まない。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import background_common as bc

HERE=Path(__file__).resolve().parent
CLINICAL=['BMI','SBP','TG','HDL','ALT','FPG']
NUMERIC=['age']+CLINICAL
FULL=list(bc.u.NUM13)+list(bc.u.CAT13)

def keys(frame,cols):
    return pd.MultiIndex.from_frame(frame[cols].round(8))

def mean_abs_spearman(frame,cols):
    correlation=frame[cols].corr(method='spearman').to_numpy()
    return float(np.abs(correlation[np.triu_indices(len(cols),1)]).mean())

def nearest_rms(query,c,scale):
    values=np.full(len(query),np.inf)
    for sex in sorted(query.sex.unique()):
        mask=query.sex.to_numpy()==sex;release=c[c.sex==sex]
        if not len(release):continue
        nn=NearestNeighbors(n_neighbors=1).fit(release[list(bc.u.NUM7)].to_numpy()/scale)
        values[mask]=nn.kneighbors(query.loc[mask,list(bc.u.NUM7)].to_numpy()/scale)[0][:,0]/np.sqrt(5)
    assert np.isfinite(values).all()
    return values

def main():
    data=bc.setup();a1=data['a'];a2=bc.u.base.read(bc.ROOT/'スターターキット/初版/participant_data/A_bg.csv')
    public=pd.concat([a1,a2],ignore_index=True);scale=a1[list(bc.u.NUM7)].std(ddof=1).to_numpy()
    public_keys={label:keys(public,cols) for label,cols in [('clinical6',CLINICAL),('full13',FULL)]}
    a1keys=keys(a1,FULL);a2keys=keys(a2,FULL)
    rows=[]
    for team in data['teams']:
        if team==12:continue
        c=data['cs'][team];q=data['qs'][team]
        cq=nearest_rms(q,c,scale);ca=nearest_rms(a1,c,scale)
        row=dict(team=team,rows=len(c),precision='numeric rounded8 for public value/row/duplicate matches',
            numeric7_mean_abs_spearman=mean_abs_spearman(c,NUMERIC),
            QI5_mean_abs_spearman=mean_abs_spearman(c,list(bc.u.NUM7)),
            A1_numeric7_mean_abs_spearman=mean_abs_spearman(a1,NUMERIC),
            public_A_clinical6_joint_rows=int(keys(c,CLINICAL).isin(public_keys['clinical6']).sum()),
            public_A_full13_rows=int(keys(c,FULL).isin(public_keys['full13']).sum()),
            A1_full13_rows=int(keys(c,FULL).isin(a1keys).sum()),A2_full13_rows=int(keys(c,FULL).isin(a2keys).sum()),
            A2_except_prefecture_rows=int(keys(c,[x for x in FULL if x!='prefecture']).isin(keys(a2,[x for x in FULL if x!='prefecture'])).sum()),
            full13_duplicates=int(c[FULL].round(8).duplicated().sum()),
            Q_rows=len(q),Q_sexmatched_QI5_RMS_lt001=int((cq<.01).sum()),Q_sexmatched_QI5_RMS_lt005=int((cq<.05).sum()),
            A1_rows=len(a1),A1_sexmatched_QI5_RMS_lt005=int((ca<.05).sum()),
            Q_sexmatched_QI5_RMS_median=float(np.median(cq)),
            A1_sexmatched_QI5_RMS_median=float(np.median(ca)))
        for col in NUMERIC+['time','smoking','onset','death']:
            row[col+'_unique']=int(c[col].nunique())
        for col in CLINICAL:
            row[col+'_public_value_fraction']=float(c[col].round(8).isin(public[col].round(8)).mean())
        rows.append(row)
    frame=pd.DataFrame(rows);path=HERE/'公開加工特徴.csv'
    content=frame.to_csv(index=False,float_format='%.17g').encode('utf-8-sig')
    if path.exists():assert path.read_bytes()==content,path
    else:path.write_bytes(content)
    record=dict(labels_used=False,training=False,rounding_digits=8,rows=len(frame),
        input_sha256=bc.u.base.INPUT_HASHES,script_sha256=bc.sha(__file__),output_sha256=bc.sha(path),
        distance='sex一致、raw QI5をA1標本SDで除算、5次元RMS。Qの会員/非会員ラベルは不使用。',
        inference_limit='公開素材への一致・有限支持・相関・近傍の集約値。加工の生成式や元個人の対応を確定しない。')
    (HERE/'公開特徴記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(frame[frame.team.isin([1,2,4,18,19,23,24,29])][['team','public_A_full13_rows','A2_except_prefecture_rows','full13_duplicates','age_unique','time_unique','numeric7_mean_abs_spearman','Q_sexmatched_QI5_RMS_lt005']].to_string(index=False))

if __name__=='__main__':main()
