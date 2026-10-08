"""10/8探索の保存表集計と、個別採点未保存の公開方式のメモリ再採点。

既存ファイルへ書き込まない。個人別予測・正解は返却も保存もしない。
離散依存木は元実行器を呼ばず、公開入力を使う純粋計算部分だけを再利用。
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

EXPERIMENT = Path('本戦/攻撃フェーズ/Tashiro/20261008_MIA全体_朝9時までの方式探索')
QI7 = ['age', 'sex', 'prefecture', 'SBP', 'TG', 'HDL', 'ALT']
LEARNED = {
    '低FPR学習結果', '別系統特徴の学習結果', '学習の分割感度結果',
    '学習の密度比乱数結果', '昨日の学習_18対象結果', '特徴除去の学習結果',
    '特徴除去の背景結果', '背景変更の学習結果', '複数ビューの学習結果',
    '複数ビューの対象保留結果',
}
FIXED_CONTROLS = {'gen7_county500', 'fixed_county500', 'fixed_gen7', 'fixed_gen7_cross01'}
SMOOTH_SOURCES = {
    'source0': 'gen7_reference__county500__cross0.1',
    'source1': 'q7_cat4_nnratio_k3__raw__cross0',
    'source2': 'q7_robust_sex1_full_reg0.01_gmm1__county500__cross0.1',
    'source3': 'q13_robust_sex0_diag_reg0.01_gmm16__raw__cross0.1',
    'source4': 'q13_robust_lofC_k100__raw__cross0',
    'source5': '学習の分割感度結果:median20_fourfold_C0.1_five_seed_mean',
    'source6': '学習の密度比乱数結果:without_kde_fourfold_ten_centers_five_people_median',
}


def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).exists() else {}


def _metadata(folder, top):
    record = _read_json(folder / '実行記録.json')
    specs_path = folder / '方式一覧.csv'
    if not specs_path.exists():
        specs_path = top / '方式一覧.csv'
    specs = {}
    if specs_path.exists():
        for row in _read_csv(specs_path):
            name = row.get('method') or row.get('raw_method')
            if name:
                specs[name] = {k: v for k, v in row.items() if v and k not in ('method', 'raw_method')}
    return record, specs


def _attributes(name, spec):
    if spec.get('num_attributes'):
        return spec['num_attributes'] + ';' + spec.get('cat_attributes', '')
    if spec.get('input_columns'):
        return spec['input_columns'] + ' (Gen-LRA7/9・性別Gen-LRAからの順位特徴)'
    if spec.get('input_features') or spec.get('features'):
        value = spec.get('input_features') or spec.get('features')
        if re.fullmatch(r'[0-9.]+', value):
            return 'QI7/QI9/QI13の既存12系統予測（Gen-LRAなし）×自身順位/他対象中央値、' + value + '特徴'
        return value
    if any(token in name for token in ('all12','nogen','without_kde','median20','median18','full_C','same_background_C')):
        return 'QI7/QI9/QI13の既存12系統予測（Gen-LRAなし）、方式名に示す特徴除去/集約'
    if 'q13' in name or 'all13' in name or spec.get('attributes') in ('13', 'q13'):
        return 'age;sex;prefecture;BMI;SBP;TG;HDL;ALT;FPG;smoking;onset;time;death'
    if 'q9' in name or 'gen9' in name or spec.get('attributes') in ('9', 'q9'):
        return 'age;sex;prefecture;BMI;SBP;TG;HDL;ALT;FPG'
    return ';'.join(QI7)


def _canonical(name, family):
    # Source inspection, rather than equal TPR, establishes these aliases.
    checked_aliases = {
        'q7_robust_nnratio_k10__raw__cross0': 'global_nn_k10',
        'q7_sex_nnratio_k10__raw__cross0': 'sex_nn_k10',
        'q7_robust_kde_ratio_h0.5__raw__cross0': 'global_point_h0.5',
        'q7_robust_kde_ratio_h1__raw__cross0': 'global_point_h1.0',
        'q7_robust_kde_ratio_h2__raw__cross0': 'global_point_h2.0',
        'gen7_reference__age100__cross0.1': 'flat_sex_age_s100',
    }
    if name in checked_aliases:
        return checked_aliases[name]
    if ((family == '公開非会員の較正結果' and name == 'fixed500__cross0.1')
            or (family == '広い集団への非会員較正結果' and name == 'flat500__cross0.1')
            or (family == '公開課題の除外に対する順位感度結果' and name == 'full::flat500__cross0.1')):
        # Only the Gen-LRA7 Q-prefecture/sex pipeline is aliased. Gaussian
        # calibration and owner-removal scenarios retain independent IDs.
        return 'Q_prefecture_sex_shrink500_cross01'
    if name in ('gen7_county500', 'fixed_county500', 'gen7_reference__county500__cross0.1'):
        return 'Q_prefecture_sex_shrink500_cross01'
    if name == 'gen7_reference__county500__cross0':
        return 'Q_prefecture_sex_shrink500'
    if name in ('fixed_gen7', 'gen7_reference__raw__cross0'):
        return 'global7'
    if name in ('fixed_gen7_cross01', 'gen7_reference__raw__cross0.1'):
        return 'global7_cross_loo_0.1'
    if family == '滑らかな対象間補正結果' and name.endswith('_unchanged'):
        return _canonical(SMOOTH_SOURCES[name.split('_')[0]], 'distance_or_density')
    if family in LEARNED or family == '滑らかな対象間補正結果':
        return family + ':' + name
    return name


def _class_and_split(name, family, spec, record):
    learned = family in LEARNED and name not in FIXED_CONTROLS
    if family == '滑らかな対象間補正結果':
        learned = name.startswith(('source5_', 'source6_'))
    if not learned:
        return 'C', '会員正解は採点のみ。公開A/C/Qから順位・較正・公開データ選択。'
    split = str(spec.get('target_split', ''))
    if not split:
        split = '対象LOO' if family == '複数ビューの対象保留結果' or 'LOO' in name else '対象4分割'
    labels = record.get('label_usage') or record.get('training_labels') or record.get('membership_labels') or record.get('extra', {}).get('label_usage', '')
    if name.startswith('all20_reproduction_'):
        detail = '学習候補20対象（1/2含む）。人物5分割×対象4分割。テスト人物・対象の正解を学習から除外。'
    else:
        detail = '学習候補18対象（1/2除外、12含む）。人物5分割×' + split + '。テスト人物・対象の正解を全ビューで学習から除外。'
    if labels:
        detail += str(labels)
    return 'B', detail


def _record(team, name, condition, value, members, family, source, spec, run, verification):
    evaluation_class, split = _class_and_split(name, family, spec, run)
    background = '公開予備戦A（母集団5143行）; 条件='+str(condition)
    if run.get('background_n') is not None:
        background += '; 参照行数='+str(run['background_n'])
    fraction = run.get('fraction')
    if fraction is not None:
        background += f'; fraction={fraction}; seed={run.get("background_seed")}'
    if family in ('複数ビューの学習結果', '複数ビューの対象保留結果'):
        background += '; training_context=' + spec.get('training_context', spec.get('training_contexts', '方式名参照'))
    params = dict(spec)
    params['saved_method'] = name
    params['target_reference_scope'] = ('18 (Team1/2を除外)' if ('median18' in name or 'excluding1and2' in name) else '20 (Team12含む)')
    note = '探索的な既存設定。公開予備戦正解を反復使用した方式選択の影響が残る。'
    if evaluation_class == 'B':
        note += '交差検証Bだが全探索を包む独立した外側検証ではない。'
    return dict(team=int(team), method=_canonical(name, family), original_method=name, condition=condition,
                tpr=float(value), members=int(members), attributes=_attributes(name, spec),
                params=json.dumps(params, ensure_ascii=False, sort_keys=True), background=background,
                evaluation_class=evaluation_class, split=split,
                transductive='評価対象公開C、875候補公開属性、他対象公開C/Qを参照。ラベルは分割条件に従う。',
                source=source, verification=verification, family=family, notes=note)


def _inputs(root):
    distribution = next((root/'予備戦').glob('配布*'))
    own = next(distribution.glob('PWSCup2026_prelim_team05_*'))
    b = pd.read_csv(own/'B_self.csv', float_precision='round_trip')
    q = pd.read_csv(distribution/'PWSCup2026_prelim_attack/aia_challenge_5.csv', float_precision='round_trip')
    private = pd.MultiIndex.from_frame(b[QI7]).isin(pd.MultiIndex.from_frame(q[QI7]))
    rare = b.is_rare.to_numpy(bool)
    normal = ~private & ~rare
    assert normal.sum() == 875 and private.sum() == 163
    return b, private, rare, normal


def _score_models(root, names, teams, arrays, source, family, condition, run=None, spec=None, ids=None, verification='保存順位を公式で今回再採点'):
    from official_audit import load_official, load_truth
    official = load_official(root)
    truth = load_truth(root)
    b, private, rare, normal = _inputs(root)
    if ids is not None:
        assert np.array_equal(b.loc[normal, 'record_id'].to_numpy(), ids), source
    assert arrays.shape == (len(names), 875, len(teams))
    rows = []
    for name, matrix in zip(names, arrays):
        for j, team in enumerate(teams):
            confidence = np.zeros(len(b))
            confidence[normal] = .6 + .3 * rankdata(matrix[:, j], method='min') / 875
            confidence[rare & ~private] = .4
            value, _ = official._tpr_at_fpr(confidence, truth[int(team)], .01)
            rows.append(_record(team, str(name), condition, value, int(truth[int(team)].sum()),
                                family, source, spec or {}, run or {}, verification))
    return rows


def _missing_npz(root, base):
    rows, notes = [], []
    for path in sorted(base.rglob('*.npz')):
        relative = path.relative_to(base)
        if '技術引継ぎ入力_' in str(relative) or (path.parent/'対象別TPR.csv').exists():
            continue
        if '/main/' in str(relative) or path.name.startswith('main_'):
            continue
        family = relative.parts[0]
        # Phase classification predicts preliminary/main source, not membership.
        if family in ('予備戦本戦の特徴差結果', '本戦の順位診断結果', '代表特徴の再現結果'):
            continue
        if path.name == '本戦の内部順位.npz' or family == '非会員課題の丸ごと保留結果':
            continue
        with np.load(path, allow_pickle=False) as cache:
            if not {'names', 'teams'} <= set(cache.files):
                continue
            teams = cache['teams']
            if set(map(int, teams)) != set([1,2,3,4,6,7,8,9,10,12,13,17,18,19,20,22,23,24,26,29]):
                notes.append('本戦キャッシュは予備戦評価へ混入しない: ' + str(relative))
                continue
            names = cache['names']
            key = 'cross_scores' if 'cross_scores' in cache.files else 'scores'
            if key not in cache.files:
                continue
            arrays = cache[key]
            ids = cache['record_ids'] if 'record_ids' in cache.files else None
            run = _read_json(path.parent/'実行記録.json')
            if arrays.ndim == 4:
                for seed, value in zip(cache['seeds'], arrays):
                    rows.extend(_score_models(root, names, teams, value, str(path.relative_to(root)),
                                family, 'A_split_seed'+str(seed), run=run, ids=ids,
                                spec={'A_split_seed': int(seed)}))
            else:
                assert arrays.ndim == 3
                condition = 'A_split_seed20261008' if '分割' in family or family in ('共有中順位座標の密度比結果','局所支持区間の密度比結果') else 'full'
                rows.extend(_score_models(root, names, teams, arrays, str(path.relative_to(root)),
                                         family, condition, run=run, ids=ids))
            notes.append('個別TPR欠落を保存順位から公式再採点: ' + str(relative) + ' ('+str(len(names))+'方式)')
    return rows, notes


def gather(root):
    """全方式の対象別レコードと探索留保を返す。成果物の書込は呼出元担当。"""
    root = Path(root).resolve()
    base = root / EXPERIMENT
    rows, notes = [], []
    csv_files = 0
    for path in sorted(base.rglob('*.csv')):
        relative = path.relative_to(base)
        if '技術引継ぎ入力_' in str(relative):
            continue
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'method', 'team', 'fpr', 'mia_all'} <= set(reader.fieldnames or []):
                continue
            csv_files += 1
            family = relative.parts[0]
            run, specs = _metadata(path.parent, base/family)
            condition = run.get('condition') or (relative.parts[-2] if len(relative.parts)>2 and relative.parts[-2] not in ('採点',) else 'full')
            for row in reader:
                if abs(float(row['fpr'])-.01) > 1e-14:
                    continue
                name = row['method']
                raw_name = name.split('__')[0]
                spec = specs.get(name) or specs.get(raw_name) or {}
                rows.append(_record(row['team'], name, condition, float(row['mia_all']),
                                     int(row['members']), family, str(path.relative_to(root)), spec, run,
                                     '保存対象別TPR。保存順位との今回の独立再採点は公式採点監査参照。'))
    notes.append(f'10/8探索の元対象別CSV {csv_files}ファイル、FPR=.01の元レコード {len(rows):,}行。技術引継ぎ複製と再集計平均表を重複採用しない。')
    extra, more = _missing_npz(root, base)
    rows.extend(extra); notes.extend(more)
    # Different raw-score variables retain their names even where their TPR agrees.
    deduplicated = {}
    for row in rows:
        key = (row['method'], row['condition'], row['team'], row['evaluation_class'], row['split'])
        if key in deduplicated:
            old = deduplicated[key]
            if abs(old['tpr']-row['tpr']) > 1e-12:
                # A source-specific pipeline is part of the condition; no silent merge.
                row['condition'] += '::' + row['family']
                key = (row['method'], row['condition'], row['team'], row['evaluation_class'], row['split'])
            else:
                continue
        deduplicated[key] = row
    rows = list(deduplicated.values())
    # The old discrete-PMF run deliberately did not export individual score arrays.
    # Run only its pure publicly fitted prediction, in an isolated process whose
    # audit guard cannot affect subsequent cache loading in the calling process.
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--tree-memory', str(root)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('離散依存木のメモリ再現に失敗: ' + result.stderr)
    tree = json.loads(result.stdout)
    rows.extend(tree['records']); notes.extend(tree['notes'])
    notes.extend([
        '学習方式は人物5分割×対象4分割または対象LOOのB。all20_reproductionも対象正解を除いたBであり、all20は学習候補集合の意味。',
        '滑らかな補正source5/source6はBの学習済みOOF予測を受け継ぐ。source0..4はC。',
        '47背景の分率・seed、full/same_background訓練、背景47view/中心10view訓練、median20/median18を方式または条件として区別。',
        '共有背景二分割、Aだけ/Qだけ/Q+A、局所区間・Gaussian・離散依存木は会員ラベル非依存C。公開Cと候補属性を使うトランスダクティブ処理を含む。',
        'フェーズ識別器npzは予備戦/本戦データ源の判別でありMIA方式ではないためTPRに変換しない。本戦22対象順位と公式帯は予備戦の個別TPRとして使用しない。',
        '非会員課題丸ごと保留は較正診断のみで新しいMIA予測を定義しない。強度300/500の候補予測は別途保存較正方式に含む。',
        '代表特徴キャッシュは元方式の再現対照であり新しい方式として重複算入しない。',
        'Team12は学習・他対象中央値には残す。報告行の除外は統合時に行う。Team1/2を学習から除く方式も採点は20対象を保存している。',
        '10/7と10/8の近傍比は共通IQR尺度・カテゴリ罰則1・同式だが、10/8の共通median中心化の数値実装差で通常順位が変わる。旧public計算対保存875順位の最大差: global k1=.056571428571428606、k3=.0011428571428571122、sex k1=.16285714285714292、k3=.0011428571428571122。全20対象TPR@.01は一致するが別実装として保持。',
        '完全一致alias検算: global/sex近傍比k10、global点KDE h0.5/1/2は、旧public計算と10/8保存通常順位を20対象で照合し最大差0。Gen7由来Q500のfixed500/flat500/full::flat500は同じforeignQ集合（自Team5を含むQ所有者21集合から対象所有者除外）とcross他19対象中央値0.1、保存スコア配列最大差0。Gaussianの同名較正は統合しない。',
        'Gen7 age100 cross0.1は固定fullA年齢20/40/60/80分位、Q性別×年齢CDFを100で全体CDFへ縮約、同じraw tie .001・他19対象中央値0.1のflat_sex_age_s100と同式。20対象保存TPR最大差0。',
    ])
    return rows, notes


def _tree_memory(root):
    """元離散PMFの公開予測だけを再計算し、保存上位順位集計と検算。"""
    root = Path(root).resolve()
    base = root/EXPERIMENT
    # Load label/scoring data solely for downstream evaluation before the original
    # diagnostic's no-GT audit hook is installed. Pure predictions below receive
    # only the public setup data, with no access to truth or scoring functions.
    from official_audit import load_truth, load_official
    truth = load_truth(root)
    official = load_official(root)
    b, private, rare, normal = _inputs(root)
    sys.path.insert(0, str(base))
    path = base/'離散依存木の密度比を確認.py'
    saved = _read_json(base/'離散依存木の密度比結果/prelim/実行記録.json')
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual_sha == saved['code_sha256']
    expected_drift = _read_csv(base/'離散依存木の密度比結果/prelim/独立形との上位12変化.csv')
    specification = importlib.util.spec_from_file_location('band_discrete_tree', path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    u, split = module.u, module.split
    data = u.setup('prelim')
    owner, _, _ = split.wide.pub.public_folds(data)
    assignment, parts = split.split_a(data)
    a_start = len(data['query'])
    queries = [pd.concat([data['query'][list(u.base.QI7)+['age_bin']],
                         parts[1-f][list(u.base.QI7)+['age_bin']]], ignore_index=True) for f in range(2)]
    contexts = [module.contexts(part) for part in parts]
    background = [module.density(parts[f], queries[f], contexts[f])[0] for f in range(2)]
    banks = {form: {} for form in module.FORMS}
    for target, c in data['cs'].items():
        predictions = []
        for fold in range(2):
            raw = module.density(c, queries[fold], contexts[fold])[0] - background[fold]
            predictions.append(np.stack([split.predict_cdf(raw[k], queries[fold], data, target, owner, a_start)[0][module.CAL_INDEX] for k in range(3)]))
        mean = (predictions[0]+predictions[1])/2
        for k, form in enumerate(module.FORMS):
            banks[form][target] = u.rank(mean[k], data['normal'])
    matrices = u.matrices(banks, data['teams'], cross=(.1,))
    control = matrices[module.FORMS[0]+'__cross0.1']
    max_corr_error = 0.
    for saved_row in expected_drift:
        j = data['teams'].index(int(saved_row['target']))
        form_key = saved_row['form'] if saved_row['form'].endswith('__cross0.1') else saved_row['form']+'__cross0.1'
        current = matrices[form_key][:,j]
        old_top = set(np.flatnonzero(rankdata(-control[:,j],method='min')<=12))
        new_top = set(np.flatnonzero(rankdata(-current,method='min')<=12))
        for label, value in [('independent_head_rows',len(old_top)),('form_head_rows',len(new_top)),('added',len(new_top-old_top)),('removed',len(old_top-new_top))]:
            assert int(saved_row[label]) == value, (saved_row['target'], saved_row['form'],label)
        corr = float(np.corrcoef(rankdata(control[:,j]),rankdata(current))[0,1])
        if saved_row['spearman']:
            max_corr_error = max(max_corr_error,abs(corr-float(saved_row['spearman'])))
    assert max_corr_error < 1e-12
    rows = []
    for name, matrix in matrices.items():
        for j, team in enumerate(data['teams']):
            confidence = np.zeros(len(b)); confidence[normal] = .6+.3*rankdata(matrix[:,j],method='min')/875
            confidence[rare & ~private] = .4
            value,_ = official._tpr_at_fpr(confidence,truth[team],.01)
            rows.append(_record(team,name,'A_split_seed20261008',value,int(truth[team].sum()),
                '離散依存木の密度比結果', str(path.relative_to(root)), module.METHOD, saved,
                '元公開予測計算をメモリ再現。保存上位12集計60行と照合、公式で新規個別採点。'))
    notes = [f'離散依存木3方式を元コードから予測だけメモリ再現。元code SHA一致、保存上位12変化60行の人数・入替数一致、Spearman最大誤差 {max_corr_error:.3g}。個別CSV/npz未保存を補完。']
    print(json.dumps({'records':rows,'notes':notes},ensure_ascii=False,allow_nan=False))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--tree-memory':
        _tree_memory(sys.argv[2])
    else:
        raise SystemExit('import gather(root) from the aggregate runner')
