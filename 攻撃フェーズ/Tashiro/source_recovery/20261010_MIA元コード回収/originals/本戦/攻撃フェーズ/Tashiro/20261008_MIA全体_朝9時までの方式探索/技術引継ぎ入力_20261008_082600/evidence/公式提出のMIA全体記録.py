"""保存済み公式HTMLからMIA全体の履歴だけを抽出。再採点・方式調整はしない。"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCES = HERE.parent / '20261008_AIA_C特徴と方式探索/現状監査/入力'
EDGES = (0., .25, .5, .75, 1.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / '公式提出のMIA全体記録結果')
    args = parser.parse_args()
    assert not args.output_dir.exists()
    args.output_dir.mkdir()
    rows, bands, inputs = [], [], {}
    official = ROOT / '本戦/配布データ/pwscup2026-kit-main_revised0912/starter/src/pwscup2026/kit/codabench/scoring_io.py'
    inputs[str(official)] = sha(official)
    # 公開バンド境界の実装のみ読む。正解ファイル/採点関数を実行しない。
    text = official.read_text()
    assert 'MIA_BAND_EDGES = (0.25, 0.5, 0.75)' in text
    assert 'if v <= lo:' in text and 'if v <= mid:' in text and 'if v <= hi:' in text
    for round_number in (1, 2, 3):
        path = SOURCES / f'第{round_number}回_detailed_results.html'
        inputs[str(path)] = sha(path)
        source = html.unescape(path.read_text())
        block = re.search(r'<pre>(.*?)</pre>', source, flags=re.S)
        assert block is not None
        value = json.loads(block.group(1))
        assert value['constants']['low_fpr'] == .01
        assert value['constants']['mia_band_edges'] == list(EDGES[1:-1])
        rows.append(dict(phase='main', round=round_number, low_fpr=.01,
                         mia_all_tpr_at_low_fpr=value['mia']['tpr_at_low_fpr'],
                         mia_auc=value['mia']['auc'], source_kind='saved_official_result',
                         exact_per_target_tpr_available=False))
        for target, band in value['mia_per_target'].items():
            band = int(band)
            assert 1 <= band <= 4
            bands.append(dict(phase='main', round=round_number, target=int(target), band=band,
                              tpr_lower=EDGES[band-1], lower_inclusive=band == 1,
                              tpr_upper=EDGES[band], upper_inclusive=True,
                              exact_tpr_unknown=True, correct_person_count_unknown=True))
    pd.DataFrame(rows).to_csv(args.output_dir / '本戦のMIA全体公式履歴.csv', index=False, float_format='%.17g')
    pd.DataFrame(bands).sort_values(['round', 'target']).to_csv(args.output_dir / '本戦の対象別公開バンド.csv', index=False, float_format='%.17g')
    assert all(sha(Path(p)) == h for p, h in inputs.items())
    record = dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                  code_path=str(Path(__file__).resolve()), code_sha256=sha(Path(__file__)),
                  input_sha256=inputs, output_sha256={p.name: sha(p) for p in args.output_dir.iterdir()},
                  scope='MIA全体公式履歴のみ。AIA実験・希少チャネルを集計せず、採点も実行しない。',
                  membership_labels_read=False, method_selection=False,
                  evidence='Saved official result and official public-band implementation.',
                  band_meaning='Upper endpoint inclusive; exact TPR and exact number of correct people are not disclosed.')
    (args.output_dir / '実行記録.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(stage='official_mia_history_extracted', rounds=3, band_rows=len(bands)), ensure_ascii=False))


if __name__ == '__main__':
    main()
