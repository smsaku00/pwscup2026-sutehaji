"""本戦22対象の公開特徴と公開評価バンドを保存済み集約値から結合する。

個別正解/予測/患者行を読まない。特徴による攻撃方式の選択や再採点をしない。
全対象に同じ列を設け、MIA全体の正確な対象TPR/的中人数は不明のままにする。
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'本戦公開特徴の集約表結果'
FILES = {
    'profile': '公開加工特徴の比較結果/対象別の公開特徴.csv',
    'numeric': '公開値の刻みとカテゴリ結果/数値の刻みと集中.csv',
    'tuple': '公開背景との属性単位一致結果/属性組と行全体の公開値一致.csv',
    'highorder': '公開背景との高次属性一致結果/高次属性の同時一致と独立周辺期待.csv',
    'tail': '公開分布の尾の診断結果/性別属性ごとの中央と尾.csv',
    'bands': '公式提出のMIA全体記録結果/本戦の対象別公開バンド.csv',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(frame, **filters):
    chosen = frame
    for field, value in filters.items():
        chosen = chosen[chosen[field].eq(value)]
    assert len(chosen) == 1, (filters, len(chosen))
    return chosen.iloc[0]


def main():
    assert not OUT.exists()
    hashes = {str(HERE/relative):sha(HERE/relative) for relative in FILES.values()}
    tables = {key:pd.read_csv(HERE/relative) for key, relative in FILES.items()}
    for key in tables:
        tables[key] = tables[key][tables[key].phase.eq('main')]
    rows = []
    for _, profile in tables['profile'].sort_values('team').iterrows():
        team = int(profile.team)
        age = one(tables['numeric'], team=team, source='C', group='all', attribute='age')
        row = dict(team=team, c_rows=int(profile.c_rows), age_distinct_values=int(age.distinct_count),
                   age_min=age.value_min, age_max=age.value_max,
                   c_qi7_duplicate_excess=int(profile.c_qi7_duplicate_excess),
                   c_rows_in_Q_prefecture=int(profile.c_in_q_prefecture),
                   c_sex1_fraction=profile.c_sex1_fraction, q_sex1_fraction=profile.q_sex1_fraction,
                   sexwise_correlation_delta_rms=profile.sexwise_correlation_delta_rms)
        for source in ['main_A', 'prelim_A', 'kit_A', 'union3_A']:
            for attribute_set in ['clinical6', 'QI7', 'all13']:
                matched = one(tables['tuple'], team=team, reference_source=source,
                              attribute_set=attribute_set)
                row[source+'_'+attribute_set+'_matching_C_rows'] = int(matched.exact_matching_c_rows)
        triple = one(tables['highorder'], team=team, reference_source='main_A', group='all',
                     attribute_set='HDL+FPG+SBP') if 'HDL+FPG+SBP' in tables['highorder'].attribute_set.values else one(
                         tables['highorder'], team=team, reference_source='main_A', group='all',
                         attribute_set='SBP+HDL+FPG')
        row['main_A_SBP_HDL_FPG_matching_C_rows'] = int(triple.exact_matching_c_rows)
        row['main_A_SBP_HDL_FPG_independent_marginal_expected_rows'] = triple.independent_marginals_expected_matching_c_rows
        for sex in [0, 1]:
            tail = one(tables['tail'], team=team, sex=sex, attribute='HDL')
            for field in ['central_iqr_ratio', 'range_5_95_ratio', 'range_1_99_ratio', 'c_outside_a_q01_q99_fraction']:
                row['HDL_sex'+str(sex)+'_'+field] = tail[field]
        for round_id in [1, 2, 3]:
            band = one(tables['bands'], target=team, round=round_id)
            for field in ['band', 'tpr_lower', 'lower_inclusive', 'tpr_upper', 'upper_inclusive']:
                row['round'+str(round_id)+'_'+field] = band[field]
        row.update(processing_algorithm_confirmed=False, exact_target_TPR_available=False,
                   exact_correct_person_count_available=False, attack_method_selected_from_this_table=False)
        rows.append(row)
    assert len(rows) == 22 and len({r['team'] for r in rows}) == 22
    OUT.mkdir()
    pd.DataFrame(rows).to_csv(OUT/'本戦22対象の公開特徴と公式MIA全体バンド.csv', index=False, float_format='%.17g')
    metadata = dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                    kind='derived_technical_table_not_final_prose', source_files=FILES,
                    field_definitions=dict(
                        numerical_features='public processed-C descriptors or typed public-source support matches',
                        independent_marginal_expectation='within-C empirical independent with-replacement column draws; descriptive hypothetical comparison',
                        main_A_nonmember_guarantee='same-phase person-set guarantee only; attribute tuples are not person identity',
                        historical_or_kit_A='no nonmember guarantee transferred to main targets',
                        official_band='upper inclusive: band1 [0,.25], band2 (.25,.5], band3 (.5,.75], band4 (.75,1]',
                        TPR_or_correct_person_counts='exact target values unavailable from bands',
                    ), targets=22, selection_performed=False, new_score_or_truth_used=False,
                    team_ID_not_processing_method_identity=True, formal_MIA_improvement_established=False)
    (OUT/'列の意味と制限.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    record = dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                  code_path=str(Path(__file__).resolve()), code_sha256=sha(Path(__file__)),
                  input_sha256=hashes, output_sha256={p.name:sha(p) for p in OUT.iterdir()},
                  individual_rows_read=False, membership_truth_read=False, attack_selection=False,
                  final_prose_or_GitHub_created=False)
    for name, expected in hashes.items():
        assert sha(Path(name)) == expected
    (OUT/'実行記録.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), targets=22, exact_target_TPR=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
