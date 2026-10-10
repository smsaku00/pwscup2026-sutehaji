"""公開A/C/Qだけの同値行を数える。独立なコホートの証明には使わない。"""
from __future__ import annotations
import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import 共通 as u

profile = u.module('frozen_overlap_public_profile', u.HERE / '公開Cの特徴を整理.py')


def keys(frame, columns):
    return set(frame[list(columns)].itertuples(index=False, name=None))


def overlap(a, b, columns):
    ka, kb = keys(a, columns), keys(b, columns)
    common = ka & kb
    # 集合の一致数と、重複込みの行数を分ける。別実装のMultiIndexと照合。
    left = sum(x in kb for x in a[list(columns)].itertuples(index=False, name=None))
    right = sum(x in ka for x in b[list(columns)].itertuples(index=False, name=None))
    assert left == int(profile.matched(a, b, columns).sum())
    assert right == int(profile.matched(b, a, columns).sum())
    return dict(left_rows=len(a), right_rows=len(b), left_unique=len(ka),
                right_unique=len(kb), shared_unique=len(common),
                left_matching_rows=left, right_matching_rows=right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=u.HERE / '公開データのフェーズ間重なり結果')
    args = parser.parse_args()
    assert not args.output_dir.exists()
    started = time.perf_counter()
    out = args.output_dir
    out.mkdir()
    plan = dict(created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                code_sha256=u.sha(__file__), keys=['QI7', 'all13'],
                sources=['prelim_A', 'main_A', 'kit_A', 'all_public_C', 'all_public_Q_including_self'],
                membership_labels_used=False, b_self_used=False,
                interpretation='Same observed fields, not identity or stochastic independence proof.')
    (out / '方法.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    phases = {phase: profile.inputs(phase) for phase in ('prelim', 'main')}
    kit = u.base.read(u.ROOT / 'スターターキット/初版/participant_data/A_bg.csv')
    for phase, own in [('prelim', 5), ('main', 28)]:
        a, cs, qs = phases[phase]
        if phase == 'prelim':
            target = u.base.child(u.ROOT / '予備戦', '配布データ') / 'PWSCup2026_prelim_attack'
        else:
            target = u.HERE.parent / '20260928_配布データ基本分析/入力'
        qs[own] = u.base.read(target / f'aia_challenge_{own}.csv')
        assert all(len(q) == 326 for q in qs.values())
    summary, c_pairs, q_pairs = [], [], []
    pre_a, pre_c, pre_q = phases['prelim']
    main_a, main_c, main_q = phases['main']
    sources = dict(prelim_A=pre_a, main_A=main_a, kit_A=kit,
                   prelim_C=pd.concat(pre_c.values(), ignore_index=True),
                   main_C=pd.concat(main_c.values(), ignore_index=True),
                   prelim_Q=pd.concat(pre_q.values(), ignore_index=True),
                   main_Q=pd.concat(main_q.values(), ignore_index=True))
    pairs = [('prelim_A', 'main_A'), ('prelim_A', 'kit_A'), ('main_A', 'kit_A'),
             ('prelim_C', 'main_C'), ('prelim_Q', 'main_Q')]
    for label, cols in [('QI7', u.base.QI7), ('all13', profile.ALL13)]:
        for left, right in pairs:
            if 'Q' in left and label == 'all13':
                continue
            summary.append(dict(left=left, right=right, keys=label,
                                **overlap(sources[left], sources[right], cols)))
        kit_keys = keys(kit, cols)
        for pt, pc in pre_c.items():
            for mt, mc in main_c.items():
                common = keys(pc, cols) & keys(mc, cols)
                c_pairs.append(dict(prelim_team=pt, main_team=mt, same_team_id=pt == mt,
                                    keys=label, shared_in_kit_unique=len(common & kit_keys),
                                    shared_outside_kit_unique=len(common - kit_keys),
                                    **overlap(pc, mc, cols)))
    for pt, pq in pre_q.items():
        for mt, mq in main_q.items():
            q_pairs.append(dict(prelim_team=pt, main_team=mt, same_team_id=pt == mt,
                                **overlap(pq, mq, u.base.QI7)))
    pd.DataFrame(summary).to_csv(out / '公開集合全体の一致.csv', index=False)
    pd.DataFrame(c_pairs).to_csv(out / '公開Cのフェーズ間一致.csv', index=False)
    pd.DataFrame(q_pairs).to_csv(out / '公開Qのフェーズ間一致.csv', index=False)
    assert not any(any(s in p for s in ['B_self', 'pool_ids', '正解データ', 'result.txt']) for p in u.base.INPUT_HASHES)
    for p, h in u.base.INPUT_HASHES.items():
        assert u.sha(p) == h, p
    support = [Path(profile.__file__), Path(u.__file__), u.OLD / '分析.py', u.CAL / '分析.py']
    record = dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                  seconds=time.perf_counter() - started, code_path=str(Path(__file__).resolve()),
                  code_sha256=u.sha(__file__), input_sha256=u.base.INPUT_HASHES,
                  support_code_sha256={str(p): u.sha(p) for p in support},
                  output_sha256={p.name: u.sha(p) for p in out.iterdir()},
                  python=sys.executable, python_version=platform.python_version(),
                  membership_labels_used=False, b_self_used=False,
                  qa='Every overlap was independently checked using tuple sets and MultiIndex membership.',
                  meaning='Observed value overlap only; no proof of distinct patient identity, distributional independence, or MIA transfer.')
    (out / '実行記録.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    u.log(stage='public_phase_overlap_complete', seconds=record['seconds'], summary=summary)


if __name__ == '__main__':
    main()
