"""公開非会員の所有者を丸ごと保留し、固定300・500の分布較正を確認。

この結果で強度を選び直さない。会員正解・MIA採点は一切使用しない。
"""
from __future__ import annotations
import argparse,json,platform,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import 共通 as u

public=u.module('fixed_public_null_calibration',u.HERE/'公開非会員だけで較正を選ぶ.py')
STRENGTHS=(300,500)


def phase_run(phase,out):
    started=time.perf_counter();data=u.setup(phase)
    owner,_,duplicates=public.public_folds(data)
    selection_path=u.HERE/'公開非会員の較正結果'/phase/'選択結果.json'
    selected=u.base.read_json(selection_path)
    assert selected['primary_strength']=='300',selected
    rows=[];coverage=[];raws={};banks={}
    for index,(target,c) in enumerate(data['cs'].items(),1):
        raw=u.base.core(data['a'],data['query'],c,u.NUM7,(10,))[10]
        raws[target]=raw;banks[target]=public.calibrated(raw,data,target,500)
        ii=data['controls'][target];values=raw[ii];q=data['query'].iloc[ii]
        qowner=owner[ii]
        scale=np.quantile(values,.75)-np.quantile(values,.25)
        if scale<=np.finfo(float).eps:scale=max(float(np.std(values)),1.)
        values=values/scale
        for heldowner in np.unique(qowner):
            train=qowner!=heldowner;held=~train
            global_null=values[train]
            losses={s:np.full(int(held.sum()),np.nan) for s in STRENGTHS}
            pits={s:np.full(int(held.sum()),np.nan) for s in STRENGTHS}
            held_global=np.flatnonzero(held);heldq=q.iloc[held_global]
            counts=np.zeros(len(held_global),int)
            for (prefecture,sex),local_indices in heldq.groupby(['prefecture','sex'],sort=True).indices.items():
                group= (q.prefecture.to_numpy()==prefecture)&(q.sex.to_numpy()==sex)
                local=values[train&group];observed=values[held_global[local_indices]]
                a,b,d=public.crps_coefficients(global_null,local,observed)
                g=u.cal.empirical_cdf(observed,global_null)
                l=u.cal.empirical_cdf(observed,local) if len(local) else g
                counts[local_indices]=len(local)
                for s in STRENGTHS:
                    w=len(local)/(len(local)+s)
                    losses[s][local_indices]=a+2*w*b+w*w*d
                    pits[s][local_indices]=g+w*(l-g)
            for s in STRENGTHS:
                assert np.isfinite(losses[s]).all() and np.isfinite(pits[s]).all()
                rows.append(dict(phase=phase,target=target,heldout_null_owner=int(heldowner),strength=s,
                                 crps=float(losses[s].mean()),heldout_rows=len(counts),
                                 rows_with_zero_group_support=int((counts==0).sum()),local_null_n_min=int(counts.min()),
                                 local_null_n_median=float(np.median(counts)),
                                 pit_above_99=float((pits[s]>.99).mean()),pit_above_95=float((pits[s]>.95).mean())))
        for (prefecture,sex),indices in data['b'].loc[data['normal']].groupby(['prefecture','sex'],sort=True).indices.items():
            mask=(q.prefecture.to_numpy()==prefecture)&(q.sex.to_numpy()==sex)
            n=int(mask.sum());owners=int(len(np.unique(qowner[mask])))
            coverage.append(dict(phase=phase,target=target,prefecture=prefecture,sex=int(sex),candidate_rows=len(indices),
                                 null_rows=n,null_owners=owners,local_weight_300=n/(n+300),local_weight_500=n/(n+500),
                                 c_same_group_rows=int(((c.prefecture==prefecture)&(c.sex==sex)).sum()),
                                 membership_accuracy_unknown=True))
        u.log(stage='leave_null_owner_out',phase=phase,completed_targets=index,targets=len(data['teams']))
    reference=u.matrices({'fixed500':banks},data['teams'],cross=(.1,))['fixed500__cross0.1']
    old_path=u.HERE/'公開非会員の較正結果'/phase/'内部順位だけ.npz'
    u.base.INPUT_HASHES[str(old_path)]=u.sha(old_path);old=np.load(old_path)
    j=list(old['names']).index('fixed500__cross0.1')
    error=float(np.max(np.abs(reference-old['scores'][j])));assert error==0.,error
    table=pd.DataFrame(rows);groups=pd.DataFrame(coverage)
    selected_rows=table[~table.target.isin([1,2])] if phase=='prelim' else table
    summary=[]
    for s in STRENGTHS:
        part=selected_rows[selected_rows.strength==s]
        summary.append(dict(phase=phase,strength=s,targets=part.target.nunique(),owner_target_cases=len(part),
                            mean_crps=float(part.crps.mean()),mean_pit_above_99=float(part.pit_above_99.mean()),
                            mean_pit_above_95=float(part.pit_above_95.mean()),
                            zero_group_support_fraction=float(part.rows_with_zero_group_support.sum()/part.heldout_rows.sum()),
                            membership_labels_used=False,used_for_new_strength_selection=False))
    dest=out/phase;dest.mkdir()
    table.to_csv(dest/'課題所有者を保留した較正.csv',index=False,float_format='%.17g')
    groups.to_csv(dest/'候補の較正根拠の範囲.csv',index=False,float_format='%.17g')
    pd.DataFrame(summary).to_csv(dest/'固定条件の集計.csv',index=False,float_format='%.17g')
    np.savez_compressed(dest/'候補と公開課題の生スコア.npz',teams=np.array(data['teams']),raw=np.column_stack([raws[t] for t in data['teams']]),
                        query_owner=owner,b_rows=np.array(len(data['b'])))
    for p,h in u.base.INPUT_HASHES.items():assert u.sha(p)==h,p
    assert not any('pool_ids' in p or '正解データ' in p for p in u.base.INPUT_HASHES)
    record=dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),seconds=time.perf_counter()-started,
                code_path=str(Path(__file__).resolve()),code_sha256=u.sha(__file__),input_sha256=dict(u.base.INPUT_HASHES),
                support_code_sha256={str(p):u.sha(p) for p in [u.HERE/'共通.py',u.HERE/'公開非会員だけで較正を選ぶ.py',u.OLD/'分析.py',u.CAL/'分析.py']},
                output_sha256={p.name:u.sha(p) for p in dest.iterdir()},membership_labels_used=False,
                strengths=STRENGTHS,new_strength_selection=False,baseline_rank_max_error=error,duplicate_public_qi7_rows=duplicates,
                python=sys.executable,python_version=platform.python_version(),
                meaning='地域を含む公開Q所有者が未知の場合の非会員分布較正。MIA精度や転移精度は評価していない。')
    (dest/'実行記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    u.log(stage='leave_null_owner_out_complete',phase=phase,summary=summary,seconds=record['seconds'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=u.HERE/'非会員課題の丸ごと保留結果')
    args=parser.parse_args();assert not args.output_dir.exists();args.output_dir.mkdir()
    for phase in ('prelim','main'):phase_run(phase,args.output_dir)


if __name__=='__main__':main()
