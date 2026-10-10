"""公開C・A・AIA課題だけで予備戦と本戦の加工特徴を整理する。

B_self・会員正解・採点結果は読み込まない。チーム名から加工法を断定しない。
"""
from __future__ import annotations
import argparse,json,platform,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
import 共通 as u

NUMERIC=('age','BMI','SBP','TG','HDL','ALT','FPG')
ALL13=u.NUM13+u.CAT13
BOOTSTRAPS=199
SEED=20261008


def inputs(phase):
    if phase=='prelim':
        distributed=u.base.child(u.ROOT/'予備戦','配布データ')
        targets=distributed/'PWSCup2026_prelim_attack'
        a=u.base.read(distributed/'PWSCup2026_prelim_common/A_bg.csv')
        self_id=5
    else:
        distributed=u.base.child(u.ROOT/'本戦','配布データ')
        own=next(distributed.glob('PWSCup2026_final_team28_*'))
        targets=u.HERE.parent/'20260928_配布データ基本分析/入力'
        a=u.base.read(own/'A_bg.csv');self_id=28
    cs={int(p.stem.rsplit('_',1)[1]):u.base.read(p) for p in sorted(targets.glob('C_*.csv'))}
    cs={t:c for t,c in cs.items() if t!=self_id}
    qs={t:u.base.read(targets/f'aia_challenge_{t}.csv') for t in cs}
    return a,cs,qs


def matched(frame,reference,columns):
    return pd.MultiIndex.from_frame(frame[list(columns)]).isin(pd.MultiIndex.from_frame(reference[list(columns)]))


def spearman(frame):
    values=np.column_stack([rankdata(frame[col].to_numpy(),method='average') for col in NUMERIC])
    return np.corrcoef(values,rowvar=False)


def spread_intervals(a,c,seed):
    rng=np.random.default_rng(seed)
    aa=a[list(NUMERIC)].to_numpy(float);cc=c[list(NUMERIC)].to_numpy(float)
    aq=np.quantile(aa,[.25,.75],axis=0);cq=np.quantile(cc,[.25,.75],axis=0)
    observed=(cq[1]-cq[0])/(aq[1]-aq[0])
    draws=[]
    for _ in range(BOOTSTRAPS):
        sa=aa[rng.integers(len(aa),size=len(aa))]
        sc=cc[rng.integers(len(cc),size=len(cc))]
        qa=np.quantile(sa,[.25,.75],axis=0);qc=np.quantile(sc,[.25,.75],axis=0)
        draws.append((qc[1]-qc[0])/(qa[1]-qa[0]))
    low,high=np.quantile(draws,[.025,.975],axis=0)
    return aq,cq,observed,low,high


def nearest_public(a,c,q):
    distances_c=np.full(len(q),np.nan);distances_a=np.full(len(q),np.nan)
    support=np.zeros(len(q),int)
    for key,indices in q.groupby(['prefecture','sex'],sort=True).indices.items():
        prefecture,sex=key
        ac=a[a.sex==sex]
        scale=np.quantile(u.base.numbers(ac,u.NUM7),.75,axis=0)-np.quantile(u.base.numbers(ac,u.NUM7),.25,axis=0)
        assert (scale>0).all()
        query=u.base.numbers(q.iloc[indices],u.NUM7)/scale
        for reference,dest in [(c,distances_c),(a,distances_a)]:
            local=reference[(reference.prefecture==prefecture)&(reference.sex==sex)]
            if reference is c:support[indices]=len(local)
            if len(local):
                d=cdist(query,u.base.numbers(local,u.NUM7)/scale,'euclidean')/np.sqrt(len(u.NUM7))
                dest[indices]=d.min(axis=1)
    return dict(q_matching_count_min=int(support.min()),q_matching_count_median=float(np.median(support)),
                q_rows_with_5_matching_c=int((support>=5).sum()),q_rows_without_matching_c=int((support==0).sum()),
                q_to_c_nearest_median=float(np.nanmedian(distances_c)),q_to_a_nearest_median=float(np.nanmedian(distances_a)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=u.HERE/'公開加工特徴の比較結果')
    args=parser.parse_args();assert not args.output_dir.exists()
    started=time.perf_counter()
    kit=u.ROOT/'スターターキット/初版/participant_data/A_bg.csv'
    kit_a=u.base.read(kit)
    phases={phase:inputs(phase) for phase in ('prelim','main')}
    overview=[];spreads=[];correlations=[];sources=[]
    for phase,(a,cs,qs) in phases.items():
        for label,reference in [('phase_A',a),('kit_A',kit_a)]:
            sources.append(dict(phase=phase,source=label,rows=len(reference),unique13=len(reference.drop_duplicates(list(ALL13))),
                                overlapping_13_rows_with_phase_A=int(matched(reference,a,ALL13).sum())))
        for index,(team,c) in enumerate(cs.items(),1):
            q=qs[team];qprefs=sorted(q.prefecture.unique());assert len(qprefs)==1
            top=c.prefecture.value_counts();pair_deltas=[]
            record=dict(phase=phase,team=team,c_rows=len(c),q_rows=len(q),q_prefecture=qprefs[0],
                        c_in_q_prefecture=int((c.prefecture==qprefs[0]).sum()),c_top_prefecture=str(top.index[0]),
                        c_top_prefecture_rows=int(top.iloc[0]),c_prefecture_count=int(len(top)),
                        c_qi7_duplicate_excess=int(c.duplicated(list(u.base.QI7)).sum()),
                        c_all13_duplicate_excess=int(c.duplicated(list(ALL13)).sum()),
                        q_to_c_exact_qi7_rows=int(matched(q,c,u.base.QI7).sum()),
                        c_to_phase_a_exact_13=int(matched(c,a,ALL13).sum()),
                        c_to_phase_a_exact_qi7=int(matched(c,a,u.base.QI7).sum()),
                        c_to_kit_a_exact_13=int(matched(c,kit_a,ALL13).sum()),
                        c_to_kit_a_exact_qi7=int(matched(c,kit_a,u.base.QI7).sum()),
                        c_sex1_fraction=float(c.sex.mean()),q_sex1_fraction=float(q.sex.mean()))
            record.update(nearest_public(a,c,q))
            for sex in (0,1):
                aa=a[a.sex==sex];cc=c[c.sex==sex]
                aq,cq,ratio,low,high=spread_intervals(aa,cc,SEED+100*team+sex)
                for j,col in enumerate(NUMERIC):
                    spreads.append(dict(phase=phase,team=team,sex=sex,attribute=col,a_rows=len(aa),c_rows=len(cc),
                                        a_iqr=float(aq[1,j]-aq[0,j]),c_iqr=float(cq[1,j]-cq[0,j]),iqr_ratio=float(ratio[j]),
                                        bootstrap_percentile_low=float(low[j]),bootstrap_percentile_high=float(high[j])))
                ca,ccorr=spearman(aa),spearman(cc)
                for j in range(len(NUMERIC)):
                    for k in range(j+1,len(NUMERIC)):
                        delta=float(ccorr[j,k]-ca[j,k]);pair_deltas.append(delta)
                        correlations.append(dict(phase=phase,team=team,sex=sex,attribute1=NUMERIC[j],attribute2=NUMERIC[k],
                                                 a_spearman=float(ca[j,k]),c_spearman=float(ccorr[j,k]),delta=delta))
            record['sexwise_correlation_delta_rms']=float(np.sqrt(np.mean(np.square(pair_deltas))))
            overview.append(record)
            u.log(stage='public_processing_profile',phase=phase,completed=index,targets=len(cs))
    table=pd.DataFrame(overview);spread=pd.DataFrame(spreads);corr=pd.DataFrame(correlations)
    comparisons=[]
    common=sorted(set(table[table.phase=='prelim'].team)&set(table[table.phase=='main'].team))
    for team in common:
        pair=table[table.team==team].set_index('phase')
        for column in ['c_to_phase_a_exact_13','c_qi7_duplicate_excess','c_in_q_prefecture','sexwise_correlation_delta_rms']:
            comparisons.append(dict(team=team,feature=column,prelim=float(pair.loc['prelim',column]),main=float(pair.loc['main',column]),
                                    change=float(pair.loc['main',column]-pair.loc['prelim',column]),same_processing_algorithm_unconfirmed=True))
        for sex in (0,1):
            values=spread[(spread.team==team)&(spread.sex==sex)&(spread.attribute=='HDL')].set_index('phase')
            comparisons.append(dict(team=team,feature=f'HDL_iqr_ratio_sex{sex}',prelim=float(values.loc['prelim','iqr_ratio']),
                                    main=float(values.loc['main','iqr_ratio']),change=float(values.loc['main','iqr_ratio']-values.loc['prelim','iqr_ratio']),
                                    same_processing_algorithm_unconfirmed=True))
    out=args.output_dir;out.mkdir()
    table.to_csv(out/'対象別の公開特徴.csv',index=False,float_format='%.17g')
    spread.to_csv(out/'性別ごとの分布幅.csv',index=False,float_format='%.17g')
    corr.to_csv(out/'性別ごとの属性間相関.csv',index=False,float_format='%.17g')
    pd.DataFrame(sources).to_csv(out/'公開背景の一致範囲.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'同番号のフェーズ間変化.csv',index=False,float_format='%.17g')
    assert not any('B_self' in p or 'pool_ids' in p or '正解データ' in p for p in u.base.INPUT_HASHES)
    for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h,p
    record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,
                code_path=str(Path(__file__).resolve()),code_sha256=u.sha(__file__),input_sha256=u.base.INPUT_HASHES,
                output_sha256={p.name:u.sha(p) for p in out.iterdir()},python=sys.executable,python_version=platform.python_version(),
                membership_labels_used=False,b_self_used=False,bootstraps=BOOTSTRAPS,seed=SEED,
                bootstrap_meaning='公開データ内の再標本化による記述的区間。加工アルゴリズムやMIA性能の区間ではない。',
                matching_meaning='公開背景と同じ13項目の行があるという観測。コピー加工の確定ではない。',
                cohort_vs_processing='元のコホート構成と加工の効果は公開Cだけでは切り分けられない。')
    (out/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    u.log(stage='public_profile_complete',seconds=record['seconds'],targets=len(table))


if __name__=='__main__':main()
