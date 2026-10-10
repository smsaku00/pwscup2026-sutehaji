#!/usr/bin/env python3
"""第1〜7回の実提出回答から、チーム別最高点の326行を選び第8回AIAを再現する。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FIELDS = ['target_k', 'challenge_row_id', 'time_hat']
TARGETS = list(range(18)) + [22, 25, 26, 27]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_answer(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDS, path
        rows = {}
        for row in reader:
            key = (int(row['target_k']), int(row['challenge_row_id']))
            assert key not in rows, (path, key)
            assert math.isfinite(float(row['time_hat'])), (path, key)
            rows[key] = row['time_hat']
    assert set(rows) == {(k, r) for k in TARGETS for r in range(326)}, path
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attack-dir', type=Path, required=True,
                        help='公式配布のpublic main attackディレクトリ（AIA形式検証用）')
    parser.add_argument('--out', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = HERE / 'inputs'
    expected = json.loads((HERE / 'input_sha256.json').read_text())
    assert set(expected) == {f.name for f in inputs.iterdir() if f.is_file()}
    for name, digest in expected.items():
        assert sha(inputs / name) == digest, ('input hash mismatch', name)
    answers = {i: read_answer(inputs / f'round{i}_F_aia.csv') for i in range(1, 8)}
    scores = {k: {} for k in TARGETS}
    with (inputs / 'round1to5_aia_scores.csv').open(newline='') as f:
        historical = list(csv.DictReader(f))
    assert sorted(int(r['main_team']) for r in historical) == TARGETS
    for row in historical:
        k = int(row['main_team'])
        for i in range(1, 6):
            scores[k][i] = float(row[f'r{i}_aia_corrected'])
    for i in [5, 6, 7]:
        official = json.loads((inputs / f'round{i}_aia_scores.json').read_text())['aia_per_target']
        assert sorted(map(int, official)) == TARGETS
        for k in TARGETS:
            if i == 5:
                assert math.isclose(scores[k][i], official[str(k)], rel_tol=0, abs_tol=1e-15)
            else:
                scores[k][i] = float(official[str(k)])
    for team_scores in scores.values():
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in team_scores.values())
    chosen = {}
    selection = []
    for k in TARGETS:
        maximum = max(scores[k].values())
        tied = [i for i in range(1, 8) if scores[k][i] == maximum]
        chosen[k] = max(tied)  # 同点時は新しい実提出を採用。今回の最高点に同点なし。
        selection.append({'target_k': k, 'selected_round': chosen[k],
                          'best_corrected_score': maximum, 'tied_best_rounds': ';'.join(map(str, tied)),
                          **{f'round{i}_corrected_score': scores[k][i] for i in range(1, 8)}})
    # CSVの数値文字列をそのまま移し、桁落ちや新たな回答変更を避ける。
    output = args.out / 'F_aia.csv'
    with output.open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(FIELDS)
        for k in TARGETS:
            for row_id in range(326):
                writer.writerow([k, row_id, answers[chosen[k]][k, row_id]])
    final = read_answer(output)
    for key, value in final.items():
        assert value == answers[chosen[key[0]]][key]
    with (args.out / 'team_selection.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(selection[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(selection)
    import pandas as pd
    sys.path.insert(0, str(HERE / 'official_validator_snapshot'))
    from pwscup2026.common.config import load_config
    from pwscup2026.kit.validate import load_dist, validate_aia_submission
    cfg = load_config(HERE / 'official_validator_snapshot/pwscup2026/kit/kit_config.yaml')
    check = validate_aia_submission(pd.read_csv(output), cfg, load_dist(args.attack_dir, team_id=28))
    assert check.ok, check.errors
    reference = math.fsum(max(v.values()) for v in scores.values()) / len(TARGETS)
    audit = {
        'scope': 'AIA only', 'format_validation': 'PASS', 'rows': len(final),
        'target_count': len(TARGETS), 'rows_per_target': 326,
        'source_block_exact_string_match': True,
        'round8_aia_sha256': sha(output), 'input_sha256': expected,
        'selected_teams_by_round': {str(i): [k for k in TARGETS if chosen[k] == i] for i in range(1, 8)},
        'reference_mean_corrected_aia': reference,
        'reference_note': '過去のチーム別補正後最高点の平均。第8回の新規実採点値ではない。',
        'official_round8_score_obtained': False,
        'changed_numeric_rows_from_round7': sum(float(v) != float(answers[7][key]) for key, v in final.items()),
        'tie_rule': 'latest round', 'tied_target_count': sum(len(r['tied_best_rounds'].split(';')) > 1 for r in selection),
    }
    (args.out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if k != 'input_sha256'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
