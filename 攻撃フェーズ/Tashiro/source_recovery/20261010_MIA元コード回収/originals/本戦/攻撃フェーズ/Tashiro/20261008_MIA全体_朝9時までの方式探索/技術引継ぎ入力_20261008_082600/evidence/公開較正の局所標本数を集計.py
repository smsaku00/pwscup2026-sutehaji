"""保存済み集団別CSVから局所支持数と縮約重みを再集計する。

個人行/所属正解/予測順位を読み込まず、方式選択・MIA採点をしない。
1/(n+1)は交換可能性を仮定した単純順位較正の数学的分解能の例。
現在の混合CDFをconformal p-valueや候補FPRとして扱わない。
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE/'公開較正の局所標本数結果'
BINS = [(-1, 0, '0'), (0, 9, '1-9'), (9, 49, '10-49'),
        (49, 98, '50-98'), (98, 299, '99-299'), (299, np.inf, '300+')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    files = [HERE/'公開背景の分割較正結果'/phase/'通常候補のQとAの支持範囲.csv'
             for phase in ['prelim', 'main']]
    code_hash = sha(Path(__file__))
    OUT.mkdir()
    plan = dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                code_sha256=code_hash, input_sha256={str(p):sha(p) for p in files},
                source='saved group aggregates, fixed full-public-control support diagnostic',
                prelim_targets_excluded=[1, 2], count_units='candidate exposures across targets and two A folds',
                not_unique_people=True, bins=[(a, None if np.isinf(b) else b, c) for a, b, c in BINS],
                simple_rank_resolution='p_min=1/(n+1), only as a mathematical illustration under exchangeability',
                applied_to_existing_CDF=False, candidate_FPR_or_membership_accuracy=False,
                new_method_or_threshold_selection=False)
    (OUT/'事前固定条件.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    rows = []
    sums = []
    for path in files:
        phase = path.parent.name
        table = pd.read_csv(path)
        if phase == 'prelim':
            table = table[~table.target.isin([1, 2])]
        assert table.candidate_rows.sum() == (18 if phase == 'prelim' else 22)*875*2
        for source, column in [('Q_county_sex', 'q_county_sex_rows'),
                               ('A_county_sex', 'a_county_sex_rows'),
                               ('Q_sex_age', 'q_sex_age_rows'),
                               ('A_sex_age', 'a_sex_age_rows')]:
            n = table[column].to_numpy(dtype=int)
            count = table.candidate_rows.to_numpy(dtype=int)
            assert (n>=0).all() and (count>0).all()
            for lo, hi, label in BINS:
                selected = (n>lo)&(n<=hi)
                exposures = int(count[selected].sum())
                rows.append(dict(phase=phase, source=source, support_count_bin=label,
                                 candidate_exposures=exposures,
                                 fraction_of_phase_exposures=exposures/int(count.sum()),
                                 target_fold_group_cells=int(selected.sum()),
                                 count_is_unique_people=False))
            sums.append(dict(phase=phase, source=source, candidate_exposures=int(count.sum()),
                             minimum_support=int(n.min()), maximum_support=int(n.max()),
                             mean_support_weighted_by_exposures=float(np.average(n, weights=count)),
                             zero_support_exposures=int(count[n==0].sum()),
                             support_below_99_exposures=int(count[n<99].sum()),
                             local_only_rank_can_resolve_p_at_or_below_01_exposures=int(count[n>=99].sum()),
                             rank_resolution_is_hypothetical_not_existing_method=True,
                             count_is_unique_people=False))
    pd.DataFrame(rows).to_csv(OUT/'集団別支持数の分布.csv', index=False, float_format='%.17g')
    pd.DataFrame(sums).to_csv(OUT/'支持数のフェーズ別集約.csv', index=False, float_format='%.17g')
    weights = [dict(local_n=n, strength=s, local_weight=n/(n+s), broad_parent_weight=s/(n+s))
               for n in [0, 10, 25, 50, 100, 200, 300, 500, 1000]
               for s in [25, 50, 100, 200, 300, 500, 750, 1000, 2000, 5000]]
    pd.DataFrame(weights).to_csv(OUT/'局所標本数と縮約係数の重み.csv', index=False, float_format='%.17g')
    resolution = [dict(alpha=a, necessary_calibration_count=int(np.ceil(1/a)-1),
                       smallest_possible_p=1/(int(np.ceil(1/a)-1)+1),
                       actual_candidate_FPR_control=False)
                  for a in [.005, .01, .02, .05]]
    assert resolution[1]['necessary_calibration_count'] == 99
    (OUT/'単純順位較正の分解能の数学例.json').write_text(json.dumps(dict(
        formula='p=(1+number of calibration scores at least as large as query)/(n+1)',
        assumption='exchangeable calibration and test scores',
        required_count_is_necessary_not_sufficient=True, exchangeability_established_here=False,
        examples=resolution, applied_to_existing_CDF=False,
        shrinking_or_source_mixtures_need_separate_argument=True), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    record = dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                  code_path=str(Path(__file__).resolve()), code_sha256=code_hash,
                  input_sha256=plan['input_sha256'],
                  output_sha256={p.name:sha(p) for p in OUT.iterdir()},
                  membership_labels_used=False, individual_rows_read=False,
                  new_model_or_coefficient_selection=False, final_prose_or_external_action=False)
    for name, expected in record['input_sha256'].items():
        assert sha(Path(name)) == expected
    assert sha(Path(__file__)) == code_hash
    (OUT/'実行記録.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), rows=len(rows), labels_used=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
