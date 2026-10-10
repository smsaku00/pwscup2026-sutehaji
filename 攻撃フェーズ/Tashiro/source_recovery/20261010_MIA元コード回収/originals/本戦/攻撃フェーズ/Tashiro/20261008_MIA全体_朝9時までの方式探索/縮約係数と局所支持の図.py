"""縮約係数の数学と既存局所支持の集約だけを図にする。最終文章は書かない。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE/'縮約係数と局所支持の図結果'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    source = HERE/'公開較正の局所標本数結果/集団別支持数の分布.csv'
    expected = sha(source)
    table = pd.read_csv(source)
    OUT.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), constrained_layout=True)
    grid = np.geomspace(25, 5000, 300)
    for n in [10, 50, 100, 300, 500, 1000]:
        axes[0].plot(grid, n/(n+grid), label=f'local n={n}')
    axes[0].axvline(100, color='#666666', linewidth=.9, linestyle=':')
    axes[0].axvline(500, color='#666666', linewidth=.9, linestyle=':')
    axes[0].set(xscale='log', ylim=(0, 1), xlabel='Shrinkage strength (lambda)',
                ylabel='Local CDF weight n / (n + lambda)',
                title='Weight rule; no accuracy measurement')
    axes[0].legend(fontsize=8, frameon=False, loc='upper right')
    axes[0].grid(alpha=.18)
    categories = ['0', '1-9', '10-49', '50-98', '99-299', '300+']
    phases = [('prelim', 'Q_county_sex'), ('prelim', 'A_county_sex'),
              ('main', 'Q_county_sex'), ('main', 'A_county_sex')]
    colors = ['#d1495b', '#ed9a9c', '#edc691', '#e3dfaf', '#95c6c4', '#3d8a99']
    bottom = np.zeros(4)
    for label, color in zip(categories, colors):
        values = []
        for phase, control in phases:
            selected = table[(table.phase==phase)&(table.source==control)&(table.support_count_bin==label)]
            assert len(selected)==1
            values.append(float(selected.iloc[0].fraction_of_phase_exposures)*100)
        axes[1].bar(np.arange(4), values, bottom=bottom, color=color, label=label, width=.68)
        bottom += values
    assert np.max(np.abs(bottom-100))<1e-10
    axes[1].set(xticks=np.arange(4), xticklabels=['Prelim\nQ', 'Prelim\nSplit A', 'Main\nQ', 'Main\nSplit A'],
                ylim=(0, 100), ylabel='Share of candidate exposures (%)',
                title='County x sex support counts')
    axes[1].legend(title='Local n', fontsize=8, title_fontsize=8, frameon=False,
                   bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.savefig(OUT/'shrinkage_and_support.png', dpi=170)
    fig.savefig(OUT/'shrinkage_and_support.svg')
    plt.close(fig)
    assert sha(source)==expected
    record = dict(completed_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                  code_path=str(Path(__file__).resolve()), code_sha256=sha(Path(__file__)),
                  input_sha256={str(source):expected}, output_sha256={p.name:sha(p) for p in OUT.iterdir()},
                  repeated_exposures_are_unique_people=False,
                  original_Q_support='target-owner excluded only; full-public prediction controls, not outer-owner heldout supports',
                  graph_is_MIA_accuracy_or_FPR=False, methods_selected=False,
                  membership_truth_or_individual_rows_read=False, final_prose_or_external_action=False)
    (OUT/'実行記録.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(OUT), final_prose_created=False), ensure_ascii=False))


if __name__=='__main__':
    main()
