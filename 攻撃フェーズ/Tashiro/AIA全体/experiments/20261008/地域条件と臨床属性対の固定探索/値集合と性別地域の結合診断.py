"""公開Cの列値が、同性別・同地域の公開BGにも残るかを数える。

主実験の採点後の説明用診断。群・方式・係数を選ぶ検証には使わない。
背景はphaseBGと旧キットBGの和集合。値の一致は許容差1e-8。
行対応・加工コード・人物対応を証明する統計量ではない。
"""
from pathlib import Path
import importlib.util
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('regional_values', HERE / '分析.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def run():
    kit = m.b.read(m.ROOT / 'スターターキット/初版/participant_data/A_bg.csv')
    phases = [('prelim', m.ROOT / '予備戦/配布データ/PWSCup2026_prelim_attack',
               m.ROOT / '予備戦/配布データ/PWSCup2026_prelim_common/A_bg.csv'),
              ('main', m.ROOT / '本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力',
               m.ROOT / '本戦/攻撃フェーズ/Tashiro/20260928_配布データ基本分析/入力/A_bg.csv')]
    rows = []
    for phase, folder, bgpath in phases:
        bg = pd.concat([m.b.read(bgpath), kit], ignore_index=True)
        for file in sorted(folder.glob('C_*.csv'), key=lambda p: int(p.stem.rsplit('_', 1)[1])):
            c = m.b.read(file)
            row = dict(phase=phase, team=int(file.stem.rsplit('_', 1)[1]), rows=len(c), cells=len(c)*4)
            for keys, name in [([], 'any'), (['sex'], 'sex'), (['sex', 'prefecture'], 'sex_prefecture')]:
                matched = 0
                for _, part in c.groupby(keys, sort=True) if keys else [(None, c)]:
                    subset = bg
                    for key in keys:
                        subset = subset[subset[key].eq(part[key].iloc[0])]
                    for col in m.CLINICAL:
                        if len(subset):
                            matched += int((m.v.nearest1d(part[col], subset[col]) <= 1e-8).sum())
                row[name + '_matched_cells'] = matched
                row[name + '_share'] = matched / row['cells']
            rows.append(row)
    table = pd.DataFrame(rows)
    assert len(table) == 44
    assert (table.sex_prefecture_share <= table.sex_share).all()
    assert (table.sex_share <= table.any_share).all()
    table.to_csv(m.OUT / '公開44C_値集合と性別地域の結合.csv', index=False)
    m.dump(HERE / '結合診断記録.json', dict(post_primary_score_explanatory_diagnostic=True, rows=44,
             criterion='each clinical cell has BG value within1e-8, optionally same sex/prefecture',
             BG_union='phase BG plus old-kit BG', new_method_switching=False,
             validation_on_new_teams=False, truth_used=False,
             input_sha256={str(Path(x).relative_to(m.ROOT)):h for x,h in m.b.HASHES.items()},
             code_sha256=m.sha(Path(__file__))))
    print(table[(table.phase == 'prelim') & table.team.isin([18,19,24,29])].to_string(index=False))


if __name__ == '__main__':
    run()
