"""固定済み背景感度の集計。攻撃の再実行・設定選択はしない。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
P01='Q_prefecture_sex_shrink500_cross01'
FOCUS={
 1:['global_nn_k1',P01],2:['global_nn_k1',P01],
 3:['q13_robust_lofC_k100__raw__cross0','drop_ALT_cross01'],
 4:['legacy_redteam_event_time_exact','legacy_redteam_event_time_near'],
 6:['q7_robust_lofC_k100__age100__cross0'],
 7:['q7_sex_bins40_s10_logratio__county500__cross0'],
 8:['q13_pref_sex_bins10_s10_logratio__raw__cross0.1'],
 9:['q7_quantile_sex1_kde_ratio_h0.5__county500__cross0.1'],10:[P01],
 13:['q13_median_iqr_sex0_kde_ratio_h2__raw__cross0.1','q7_median_iqr_sex1_dcr_k1__raw__cross0'],
 17:['q7_rank_lof_ratio_k100__age100__cross0.1'],
 18:['q13_mean_sd_sex0_kde_witness_h2__raw__cross0'],
 19:['legacy_saved_work_20260901以前__prelim_attack__r2__F_mia'],
 20:['q7_robust_adaptivek15_h1__age100__cross0.1'],
 22:['q7_rank_kde_witness_h0.5__age100__cross0'],
 23:['q7_all_bins20_s1000_logratio__county500__cross0.1'],
 24:['q7_robust_kde_ratio_h1__raw__cross0.1',P01],
 26:['q13_quantile_sex1_kde_ratio_h0.5__raw__cross0','q13_robust_adaptivek5_h0.5__raw__cross0'],
 29:['q7_all_bins20_s1000_zscore__county500__cross0.1'],
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    inputs=[HERE/(name+'_results.csv') for name in ['geometry','hist_local','gen','legacy']]
    table=pd.concat([pd.read_csv(p) for p in inputs],ignore_index=True)
    assert not table.duplicated(['method','condition','team','fpr']).any()
    assert table.tpr.between(0,1).all() and np.isfinite(table.tpr).all()
    assert (table.above_010.astype(bool)==(table.tpr>.1)).all()
    assert set(table.team)==set(FOCUS) and not table.team.isin([5,12]).any()
    assert len(table)==22*7*19*3 and table.method.nunique()==22
    base=table[(table.method==P01)&(table.fpr==.01)].set_index(['team','condition']).tpr
    rows=[]
    for (team,method),group in table.groupby(['team','method'],sort=True):
        part=group[group.fpr==.01].set_index('condition')
        assert len(part)==7 and 'full' in part.index
        v=part.tpr;sample=v.drop('full');full=float(v.loc['full'])
        fp05=group[group.fpr==.005].tpr;fp02=group[group.fpr==.02].tpr
        deltas=np.array([float(row.tpr)-float(base.loc[team,c]) for c,row in part.iterrows()])
        mode='A非依存対照' if method=='legacy_redteam_event_time_exact' else 'A1のみ変更・A2/gates固定' if method.startswith('legacy_saved_') else 'A1行抽出'
        rows.append(dict(team=int(team),method=method,focal=method in FOCUS[team],
            background_test_mode=mode,full_tpr=full,
            min_tpr=float(v.min()),median_tpr=float(v.median()),max_tpr=float(v.max()),
            success7=int((v>.1).sum()),sample6_success=int((sample>.1).sum()),
            sample6_min_tpr=float(sample.min()),sample6_mean_tpr=float(sample.mean()),
            full_tp_equivalent=full*int(part.members.iloc[0]),members=int(part.members.iloc[0]),
            full_delta_vs_P01=full-float(base.loc[team,'full']),
            worst_delta_vs_P01=float(deltas.min()),
            fpr005_min=float(fp05.min()),fpr005_max=float(fp05.max()),
            fpr020_min=float(fp02.min()),fpr020_max=float(fp02.max()),
            selection='前回の予備戦結果から固定済み。方式選択の外部検証ではない。'))
    stats=pd.DataFrame(rows);focal=stats[stats.focal].copy()
    for path,frame in [(HERE/'固定方式_全背景評価.csv',table),(HERE/'背景耐性_全対象全方式.csv',stats),(HERE/'背景耐性_チーム別固定候補.csv',focal)]:
        content=frame.to_csv(index=False,float_format='%.17g').encode('utf-8-sig')
        if path.exists():assert path.read_bytes()==content,('既存集計と不一致',path)
        else:path.write_bytes(content)
    record={'input_sha256':{p.name:sha(p) for p in inputs},'methods':22,'conditions_per_method':7,
        'report_teams':sorted(FOCUS),'fprs':[.005,.01,.02],'target_rows':len(table),
        'focus_pairs':len(focal),'focal_all7_positive':focal[focal.success7==7][['team','method','min_tpr']].to_dict('records'),
        'success_rule':'official unrounded TPR > 0.10 at FPR .01',
        'replicate_limit':'同じAからの依存した6標本。event exactはA非依存。旧r2はA2固定。独立評価ではない。',
        'script_sha256':sha(__file__)}
    (HERE/'集計記録.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(focal[['team','method','full_tpr','min_tpr','max_tpr','success7','fpr005_min']].to_string(index=False))

if __name__=='__main__':main()
