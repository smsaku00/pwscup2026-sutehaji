"""移動前のMIA保存物と既存純関数の予備戦公式再採点。

旧スクリプトのmainは実行しない。個票・入力・大会提出物を保存しない。
gather(root) -> (集計recordのlist, 探索/欠落noteのlist)。
既存13密度推定器の内部BIC/正則化選択は元コードの定義を維持する。
評価する所属ラベルは推定器へ渡さない。
"""
from __future__ import annotations

import concurrent.futures
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
NUM7 = ['age', 'BMI', 'SBP', 'TG', 'HDL', 'ALT', 'FPG']
QI7 = ['age', 'SBP', 'TG', 'HDL', 'ALT', 'sex', 'prefecture']
ATTR9 = QI7 + ['BMI', 'FPG']
COV10 = ATTR9 + ['smoking']
ALL13 = COV10 + ['time', 'onset', 'death']


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gather(root, *, use_cache=True):
    root = Path(root).resolve()
    cache = HERE / 'historical_cache.json'
    if use_cache and cache.exists():
        stored = json.loads(cache.read_text())
        if all((root / p).exists() and _sha(root / p) == h
               for p, h in stored['source_hashes'].items()):
            return stored['records'], stored['notes']
    audit = _module('_historical_official', HERE / 'official_audit.py')
    official = audit.load_official(root)
    truths = audit.load_truth(root)
    data = root / '予備戦/配布データ'
    bpath = next(data.glob('PWSCup2026_prelim_team05_*/B_self.csv'))
    b = pd.read_csv(bpath, float_precision='round_trip')
    ids = b.record_id.to_numpy()
    rare = b.is_rare.to_numpy(bool)
    bgpath = data / 'PWSCup2026_prelim_common/A_bg.csv'
    bg = pd.read_csv(bgpath, float_precision='round_trip')
    dist = data / 'PWSCup2026_prelim_attack'
    teams = sorted(truths)
    cs = {k: pd.read_csv(dist / f'C_{k}.csv', float_precision='round_trip') for k in teams}
    q = pd.read_csv(dist / 'aia_challenge_5.csv', float_precision='round_trip')
    matched = b.merge(q[QI7], on=QI7, validate='one_to_one')
    private = b.record_id.isin(matched.record_id).to_numpy()
    assert private.sum() == 163 and len(b) == 1049 and len(bg) == 5143
    records, notes, source_paths = [], [], {bpath, bgpath, dist / 'aia_challenge_5.csv', Path(__file__), HERE/'official_audit.py'}
    source_paths.update(official._source_paths)
    truth_dir = next((data/'結果').glob('正解*'))
    source_paths.update(truth_dir.glob('pool_ids_*.csv'))
    source_paths.update(dist / f'C_{k}.csv' for k in teams)
    signature_to_records = {}

    def note(scope, status, reason, source):
        notes.append(dict(scope=scope, status=status, reason=reason, source=str(source)))

    def add_matrix(name, scores, *, source, attributes, params, family,
                   condition='予備戦公開正解・20対象', evaluation_class='C',
                   split='所属ラベルは採点だけに使用', transductive='評価対象Cを参照',
                   verification='保存個別scoreを現公式で再採点', background='予備戦A_bg 5143行',
                   detail='', selection='事前固定方式', saved=None, extra=None):
        scores = np.asarray(scores, float)
        if scores.shape != (len(ids), len(teams)) or not np.isfinite(scores).all():
            raise ValueError((name, scores.shape, 'finite required'))
        # 順位signatureは確認情報。別アルゴリズムの統合判断には使わない。
        ranks = np.column_stack([rankdata(scores[:, j], method='min') for j in range(len(teams))])
        signature = hashlib.sha256(ranks.astype('<i8').tobytes()).hexdigest()
        # 別アルゴリズムが同じ順位を出しても別方式として残す。
        # 明らかな提出コピー/全体不変の希少・AIA別名だけを上流の明示表で集約。
        current = []
        for j, team in enumerate(teams):
            tpr, auc = official._tpr_at_fpr(scores[:, j], truths[team], .01)
            if saved is not None:
                assert abs(tpr - float(saved[team])) < 1e-12, (name, team, tpr, saved[team])
            record = dict(team=int(team), method=name, condition=condition, tpr=float(tpr),
                          members=int(truths[team].sum()), attributes=attributes, params=params,
                          background=background, evaluation_class=evaluation_class, split=split,
                          transductive=transductive, source=str(source), verification=verification,
                          family=family, notes=detail, selection=selection, auc=float(auc),
                          rank_signature=signature)
            if extra:
                record.update(extra)
            records.append(record)
            current.append(record)
        signature_to_records[signature] = current

    # 1. 予備戦1〜8回の保存回答。全体に影響しない希少/AIA変更は別方式にしない。
    history_path = root / '予備戦/結果分析/集計/過去8回_対象別.csv'
    history = pd.read_csv(history_path)
    source_paths.add(history_path)
    submissions = root / '予備戦/攻撃フェーズ/提出記録'
    collapsed = {1: [1, 4, 5, 6], 2: [2, 3], 7: [7, 8]}
    for number, aliases in collapsed.items():
        folder = next(submissions.glob(f'{number}_*'))
        paths = list(folder.rglob('F_mia.csv'))
        path = min(paths, key=lambda p: len(p.parts))
        source_paths.add(path)
        f = pd.read_csv(path, float_precision='round_trip').set_index('record_id').loc[ids]
        saved = history.loc[history.method.eq(f'第{number}回')].set_index('target').mia_all.to_dict()
        if number == 1:
            attributes = COV10
            params = 'NN順位0.9+KDE順位0.1; team2/23区間距離;18/19/29公開素材選択補正;非会員163人0'
            family = '旧提出NN・密度比・対象別公開C補正'
        elif number == 2:
            attributes = '生成時の属性条件が未確定（保存回答からの採点は可能）'
            params = '旧提出保存回答。20対象の順位をそのまま再採点'
            family = '旧提出第2/3回'
        else:
            attributes = {'nonrare': QI7, 'rare': ATTR9, 'team1_2': COV10}
            params = 'Gen-LRA7;Team1/2第1回維持;非会員163人0;希少は別順位'
            family = 'Gen-LRA・予備戦対象別履歴方針'
        add_matrix(f'legacy_submission_{number}', f[[str(k) for k in teams]].to_numpy(),
                   source=str(path.relative_to(root)), attributes=attributes, params=params, family=family,
                   saved=saved, verification='保存F_miaから現公式再採点、過去対象別TPR一致',
                   detail=f'全体が同じ第{aliases}回を集約。対象別維持は予備戦の履歴依存。',
                   selection='予備戦履歴に依存する対象別方針', extra={'submission_aliases': aliases})
    # 2. 9月7日の全保存生score。距離・点密度比・喫煙分解も含む。
    p21 = root / '予備戦/攻撃フェーズ/分析/21_MIA全体と希少のGenLRA比較/比較結果.json'
    p22 = root / '予備戦/攻撃フェーズ/分析/22_GenLRAの7属性9属性10属性比較/比較結果.json'
    source_paths.update([p21, p22])
    d21, d22 = json.loads(p21.read_text()), json.loads(p22.read_text())
    assert d21['record_ids'] == ids.tolist() == d22['record_ids']
    specs = [(p21, d21['raw_scores'], key) for key in d21['raw_scores']['1'] if key != 'round6']
    specs += [(p22, d22['raw_new_scores'], key) for key in d22['raw_new_scores']['1']]
    for path, mapping, key in specs:
        attributes = QI7 if key.startswith('qi7') else COV10 if key.startswith('attrs10') or key.startswith('smoking') else ATTR9
        family = 'Gen-LRA混合カーネル' if 'gen' in key or key.startswith('attrs9') or key.startswith('smoking') else '同空間の近傍距離' if 'distance' in key else '同空間の点密度比'
        add_matrix('legacy_raw_' + key, np.column_stack([mapping[str(k)][key] for k in teams]),
                   source=str(path.relative_to(root)) + '#' + key, attributes=attributes,
                   params=('最近傍k=1;負の標準化混合L2距離' if 'distance' in key else
                           '全C/全背景の点KDE対数密度比' if 'point' in key else
                           'C近傍k=10の対数密度変化和')+';倍率1;TG/ALT log1p;背景IQR/1.349;混合カテゴリrho=.98;生score・0化なし',
                   family=family, detail='公開所属真値より前の保存値。今回の真値は採点だけ。')
    # 保存された非希少差替は、公開真値から後で対象別方針を組み込んだ参考条件。
    # 希少案・AIA案24+等は全体TPRを変えないため、ここへ重複追加しない。
    unmatched = root / '予備戦/結果分析/集計/未採用手法_対象別.csv'
    old = pd.read_csv(unmatched)
    source_paths.add(unmatched)
    for dim in [7, 9, 10]:
        method = f'Gen{dim}_非希少のみ差替'
        rows = old[old.method.eq(method)]
        for row in rows.itertuples():
            records.append(dict(team=int(row.target), method=f'legacy_gen{dim}_nonrare_historical_replace',
                condition='予備戦公開後の対象別固定差替', tpr=float(row.mia_all), members=int(row.n_pos),
                attributes=QI7 if dim==7 else ATTR9 if dim==9 else COV10,
                params='Team1/2と希少を第8回へ固定;残18対象の非希少・非専用875人だけ差替',
                background='予備戦A_bg 5143行', evaluation_class='A', split='公開真値を見た後の組込評価',
                transductive='評価対象Cと予備戦履歴を参照', source=str(unmatched.relative_to(root))+'#'+method,
                verification='公開真値再採点保存TPRを読取', family='Gen-LRA・事後対象別固定差替',
                selection='予備戦履歴/oracle組込参考', notes='未知本戦でこのteam番号の例外を再利用しない。'))
    note('未採用MIA希少/AIA変更', '全体不変・集約', '保存集計44条件のうちMIA全体生順位/差替6以外は全体不変。24希少モデル等を独立したMIA全体方式数へ加えない。', str(unmatched.relative_to(root)))

    # 3. 10月5日の8構成は純計算関数で再現し、160個の保存TPRと照合。
    initial_dir = root / '本戦/攻撃フェーズ/Tashiro/20261005_MIA初回_GenLRA比較'
    gen_path = initial_dir / 'gen_lra.py'
    saved_path = initial_dir / '結果/予備戦_対象別評価.csv'
    gen = _module('_historical_gen_initial', gen_path)
    source_paths.update([gen_path, saved_path])
    saved_initial = pd.read_csv(saved_path).set_index(['variant', 'team']).mia_all.to_dict()
    matrices = {name: np.empty((len(ids), len(teams))) for name in gen.VARIANTS}
    for j, k in enumerate(teams):
        _, answers = gen.calculate(bg, b.drop(columns='is_rare'), cs[k], rare, private)
        for name, conf in answers.items():
            matrices[name][:, j] = conf
    for name, scores in matrices.items():
        attributes = {'nonrare': QI7, 'rare': ATTR9} if name[0] in 'ABC' else COV10 if name[0]=='D' else ALL13
        add_matrix('initial_20261005_' + name, scores, source=str(saved_path.relative_to(root))+'#'+name,
                   attributes=attributes, params=f'k=10;{gen.VARIANTS[name][0]};{gen.VARIANTS[name][1]}',
                   family='Gen-LRA混合カーネル' if name.startswith('A') else 'Gen-LRA順序符号化Gaussian',
                   saved={k: saved_initial[name,k] for k in teams},
                   verification='純関数再計算、全20対象TPRが保存値と一致',
                   detail='A-Fは希少/非希少の値域分離、A-Gは公開専用非会員163人0。Hは0化なし。')

    # 4. 移動前のDOMIAS/未提出回答。旧mainは使わず保存ファイルだけ再採点。
    archive = root / 'アーカイブ/予備戦_攻撃フェーズ'
    legacy_files = sorted(archive.rglob('F_mia*.csv'))
    for path in legacy_files:
        source_paths.add(path)
        f = pd.read_csv(path, float_precision='round_trip')
        if not f.record_id.is_unique or set(f.record_id) != set(ids) or set(f.columns) != {'record_id', *map(str,teams)}:
            note(str(path.relative_to(root)), '対象不一致・除外', '予備戦B/20対象とIDまたは列が一致しない（練習課題等）。', str(path.relative_to(root)))
            continue
        f = f.set_index('record_id').loc[ids]
        diagpath = path.parent / 'domias.json'
        if diagpath.exists():
            source_paths.add(diagpath)
            diag = json.loads(diagpath.read_text())['_summary']
            params = {k: diag.get(k) for k in ['bw','zeroed','prior_weight','home_share_rate']}
            # 初期outのメタデータには後追加されたprior_weightがなく、値は0。
            params['prior_weight'] = diag.get('prior_weight', 0.0)
            name = 'legacy_domias_' + path.parent.name
            attrs = NUM7
            family = 'DOMIAS Gaussian KDE'
        else:
            name = 'legacy_saved_' + path.relative_to(archive).as_posix().replace('/', '__').removesuffix('.csv')
            attrs = '保存元README/コード参照（不明な列定義を推測しない）'
            params = '旧未提出・メンバー保存予測。生成条件の一部未確定'
            family = '旧保存MIA'
            if path.parent.name == 'zhou':
                attrs = ALL13
                params = '連続7列(BMI,SBP,TG,HDL,ALT,FPG,time)の独立1D KDE;TG/ALT log,time log1p;カテゴリ6列(age,sex,prefecture,smoking,onset,death)のlog密度比×.5;Laplace1;C1の公開QI近接120行を密度学習から除外'
                family = '独立周辺KDE・カテゴリ密度比'
            elif 'prelim_attack' in path.as_posix():
                attrs = NUM7+['time','prefecture']
                params = '他C行列double-centering;地元県事前;非地元上位12;time/距離指紋;保存回答そのまま'
                family = '旧Cross-C・公開構造・指紋組込'
        add_matrix(name, f[[str(k) for k in teams]].to_numpy(), source=str(path.relative_to(root)),
                   attributes=attrs, params=params, family=family,
                   detail='撤回済みのproxy結論は使わず、保存予測の大会真値TPRを今回再採点。')

    # 5. 旧attack libraryの既定方式。旧評価台のproxy数値は使わない。
    attack_path = archive / 'work_20260901以前/attacks.py'
    attack = _module('_historical_attacks', attack_path)
    source_paths.add(attack_path)
    simple = [('domias',attack.m1_domias), ('exact',attack.m2_exact_match),
              ('nn_gap',attack.m3_nn_gap), ('knn',attack.m4_knn_density),
              ('pref_prior',attack.m5_pref_prior), ('ensemble',attack.mia_ensemble)]
    for name, fn in simple:
        scores = np.column_stack([fn(cs[k], bg, b) for k in teams])
        add_matrix('legacy_library_' + name, scores, source=str(attack_path.relative_to(root))+'#'+fn.__name__,
                   attributes='prefecture' if name=='pref_prior' else NUM7[1:] if name=='exact' else NUM7+['prefecture'] if name=='ensemble' else NUM7,
                   params={'nn_gap':'kC=1,kA=2','knn':'k=15','exact':'clinical6,rel_tol=1e-9',
                           'ensemble':'exact3,nn1,knn1,DOMIAS.7,pref.5;exact gate .5',
                           'pref_prior':'log県周辺比','domias':'Gaussian KDE Scott'}[name],
                   family='旧近傍・密度比・一致', verification='旧純関数を固定既定値で再計算・現公式採点',
                   detail='過去はproxy台で検討。今回初めて同じ定義を予備戦大会20対象の真値で評価。')

    # 6. Cross-Cの固定6方式。保存されていない全順位を元定義で再現。
    cross_path = archive / 'work_20260901以前/cross_c_mia/cross_c_consensus_mia.py'
    cross = _module('_historical_cross', cross_path)
    source_paths.add(cross_path)
    mu, sd = bg[NUM7].mean().to_numpy(), bg[NUM7].std(ddof=0).to_numpy()
    clean, _ = cross.clean_public_pool(bg, cs, teams, mu, sd, .05)
    controls = cross.matched_controls(b, clean, mu, sd, 20260829)
    db1 = cross.distance_matrix(b, cs, teams, mu, sd, 1)
    dr1 = cross.distance_matrix(controls, cs, teams, mu, sd, 1)
    db5 = cross.distance_matrix(b, cs, teams, mu, sd, 5)
    dr5 = cross.distance_matrix(controls, cs, teams, mu, sd, 5)
    s1, s5 = cross.consensus_scores(db1, dr1), cross.consensus_scores(db5, dr5)
    own_cpath = dist / 'C_5.csv'
    source_paths.add(own_cpath)
    own_c = pd.read_csv(own_cpath, float_precision='round_trip')
    bridge_b = cross.bridge_distances(b, own_c, cs, teams, mu, sd)
    bridge_r = cross.bridge_distances(controls, own_c, cs, teams, mu, sd)
    sb = cross.consensus_scores(bridge_b, bridge_r)
    cross_scores = {'nn1_raw_closeness':-db1, 'nn1_public_null_z':s1['direct_z'],
                    'nn1_cross_specific':s1['cross_specific'], 'nn1_rank_residual':s1['rank_residual'],
                    'nn5_cross_specific':s5['cross_specific'], 'c5_bridge_cross_specific':sb['cross_specific']}
    for name, scores in cross_scores.items():
        add_matrix('legacy_cross_' + name, scores, source=str(cross_path.relative_to(root))+'#'+name,
                   attributes=NUM7, params='seed20260829;C近接A_bg除外閾値.05;同性/喫煙/県優先の対照1049',
                   family='Cross-C近傍比較', background=f'予備戦A_bg5143→C近接除外{len(clean)}→整合対照1049',
                   transductive='全20対象Cと自チームCを参照', verification='元純関数で全score再現・現公式採点',
                   detail='旧READMEのtime指紋/C間一致の結論は撤回済み。新評価はそのproxy陽性を使わない。')
    note('Cross-C capacity26', '診断割当・除外', '各対象26人を強制充填する集合で連続confidenceが未定義。新方式へ変換しない。', str(cross_path.relative_to(root)))

    # 7. 旧13密度推定器の既定設定。C/A分類のラベルは所属正解と別物。
    density_path = archive / '過去のコード/densratio/densities.py'
    density = _module('_historical_densities', density_path)
    source_paths.add(density_path)
    ref = (bg[NUM7].to_numpy()-mu)/sd
    query = (b[NUM7].to_numpy()-mu)/sd
    zoo = density.ratio_zoo()
    started = time.monotonic()
    with threadpool_limits(limits=1):
        reference_logs = {}
        for label, factory in zoo:
            estimator = factory()
            if isinstance(estimator, density.TwoDensityRatio):
                reference_logs[label] = estimator.maker().fit(ref).logpdf(query)

        def density_target(k):
            synthetic = (cs[k][NUM7].to_numpy()-mu)/sd
            result, failures = {}, []
            for label, factory in zoo:
                try:
                    estimator = factory()
                    if isinstance(estimator, density.TwoDensityRatio):
                        score = np.clip(estimator.maker().fit(synthetic).logpdf(query)-reference_logs[label], -200, 200)
                    else:
                        score = estimator.fit(synthetic, ref).log_ratio(query)
                    if not np.isfinite(score).all():
                        raise ArithmeticError('nonfinite scores')
                    result[label] = score
                except Exception as exc:
                    failures.append((label, type(exc).__name__ + ': ' + str(exc)))
            return k, result, failures

        density_results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(density_target, k) for k in teams]
            for future in concurrent.futures.as_completed(futures):
                k, result, failures = future.result()
                density_results[k] = result
                for label, error in failures:
                    note(f'{label}/Team{k}', '既存定義の実行失敗・未確認', error, str(density_path.relative_to(root)))
                print(f'旧密度比の公式再採点: {len(density_results)}/20対象 ({time.monotonic()-started:.0f}秒)', flush=True)
    for label, _ in zoo:
        available = [k for k in teams if label in density_results[k]]
        name = 'legacy_density_' + label.split()[0]
        if len(available) == len(teams):
            add_matrix(name, np.column_stack([density_results[k][label] for k in teams]),
                       source=str(density_path.relative_to(root))+'#'+label,
                       attributes=NUM7, params=label+';既存ratio_zoo既定値;種0;logscore±200 (two-density)',
                       family='密度比13設定', verification='既存推定器を全20対象で再計算・現公式採点',
                       split='所属ラベル不使用。分類器はC対Aの5fold、GMMはC/A内BIC。',
                       detail='旧保存Panel A/Bの代替真値TPRは除外。これは元方式を今回の大会対象で再評価した値。')
        else:
            for k in available:
                tpr, auc = official._tpr_at_fpr(density_results[k][label], truths[k], .01)
                records.append(dict(team=k,method=name,condition='予備戦公開正解・実行可能対象',tpr=float(tpr),members=int(truths[k].sum()),
                    attributes=NUM7,params=label+';既存ratio_zoo既定値',background='予備戦A_bg 5143行',evaluation_class='C',
                    split='所属ラベル不使用。C/A内推定だけ。',transductive='評価対象Cを参照',source=str(density_path.relative_to(root))+'#'+label,
                    verification='既存推定器再計算・現公式採点',family='密度比13設定',notes='実行不能対象を0と補完しない。',auc=float(auc),selection='事前固定方式'))

    note('旧proxy実験/本戦バトル', '大会20対象TPRとして除外',
         'densratio/mia_results・panelA2/B2、aia2mia/bench、atk*/mia、matrix2、仮想大会、バトルは自己C・代替対照・仮想加工/別評価対象。保存TPRをTeam5対大会20対象のTPRと混ぜない。再利用可能な方式は上の純関数で再評価。',
         'アーカイブ/予備戦_攻撃フェーズ/work_20260901以前;本戦/バトル')
    # 8. 加工監査の既存攻撃定義も再利用。壊れた旧harnessや加工実行器を
    # importせず、事前に読んだ純数値関数/必要な定数だけをASTから取り出す。
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    from sklearn.neighbors import NearestNeighbors
    from lifelines import CoxPHFitter

    def extracted(path, names, environment):
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
        assert {n.name for n in selected} == set(names)
        namespace = dict(environment)
        namespace['__name__'] = '_historical_extracted_' + path.stem
        sys.modules[namespace['__name__']] = SimpleNamespace(**namespace)
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec',flags=__import__('__future__').annotations.compiler_flag),namespace)
        source_paths.add(path)
        return SimpleNamespace(**namespace)

    env = dict(np=np,pd=pd,AT=attack,NearestNeighbors=NearestNeighbors,
               ExtraTreesClassifier=ExtraTreesClassifier,HistGradientBoostingClassifier=HistGradientBoostingClassifier,
               dataclass=dataclass,CoxPHFitter=CoxPHFitter,warnings=warnings)
    defense_path = archive/'work_20260901以前/defense.py'
    defense_env = dict(env,COX_COLS=['age','sex','BMI','SBP','log_TG','HDL','log_ALT','FPG','smoking'])
    defense = extracted(defense_path,['_design','fit_cox_raw'],defense_env)
    defense.HORIZON = 10.0
    env['D'] = defense
    clinical_path = archive/'work_20260901以前/mia_redteam.py'
    clinical_env = dict(env,CLINICAL_VIEWS=(tuple(NUM7),('age','BMI','FPG'),('age','TG','HDL','ALT'),
                                           ('BMI','SBP','FPG'),('TG','HDL','ALT','FPG')))
    clinical = extracted(clinical_path,['_shadow_features','shadow_c2st','_scaled','multiview_tail_linkage'],clinical_env)
    outcome_path = archive/'work_20260901以前/outcome_mia_redteam.py'
    outcome = extracted(outcome_path,['_event_code','event_time_exact','event_time_near','_joint_features',
                                     'joint_nn_gap','joint_knn','joint_shadow_c2st','_CauseModel','_exposure',
                                     '_fit_cause','survival_scores','_rank01'],env)
    outcome_seeds = [20260830+split*10000+method_index*100 for split in range(3) for method_index in range(3)]
    starting_red = time.monotonic()
    def red_target(k):
        values = {
            'shadow_c2st':clinical.shadow_c2st(cs[k],bg,b,seed=20260830+7001),
            'multiview_tail':clinical.multiview_tail_linkage(cs[k],bg,b,k=3),
            'event_time_exact':outcome.event_time_exact(cs[k],bg,b),
            'event_time_near':outcome.event_time_near(cs[k],bg,b),
            'joint_nn_gap':outcome.joint_nn_gap(cs[k],bg,b),
            'joint_knn':outcome.joint_knn(cs[k],bg,b,k=15),
        }
        values['survival_likelihood'],values['cox_residual'] = outcome.survival_scores(cs[k],bg,b)
        for seed in outcome_seeds:
            shadow = outcome.joint_shadow_c2st(cs[k],bg,b,seed=seed)
            values[f'joint_shadow_c2st_seed{seed}'] = shadow
            pieces = [outcome._rank01(values[name]) for name in ['event_time_near','joint_nn_gap','joint_knn','survival_likelihood','cox_residual']]
            pieces.append(outcome._rank01(shadow))
            values[f'outcome_ensemble_seed{seed}'] = np.mean(pieces,axis=0)
        return k, values
    red = {}
    with threadpool_limits(limits=1), concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(red_target,k) for k in teams]
        for future in concurrent.futures.as_completed(futures):
            k, values = future.result()
            red[k] = values
            print(f'旧転帰・臨床攻撃の公式再採点: {len(red)}/20対象 ({time.monotonic()-starting_red:.0f}秒)',flush=True)
    for name in red[teams[0]]:
        clinical_name = name in ['shadow_c2st','multiview_tail']
        add_matrix('legacy_redteam_'+name,np.column_stack([red[k][name] for k in teams]),
            source=str((clinical_path if clinical_name else outcome_path).relative_to(root))+'#'+name,
            attributes=(NUM7 if name=='multiview_tail' else NUM7+['sex','smoking']) if clinical_name else
                       ['time','onset','death'] if name.startswith('event_time') else [c for c in ALL13 if c!='prefecture'],
            params=('k=3;既存臨床5viewsの最大背景percentile' if name=='multiview_tail' else
                    'ExtraTrees240+HGB180;旧実行器seed'+str(20260830+7001) if name=='shadow_c2st' else
                    '元純関数定義・旧実行器の保存条件;'+name),
            family='旧臨床・転帰MIA',verification='旧pure定義を切り出して再計算・現公式採点',
            split='所属ラベル不使用。C対Aの分類ラベル/生存モデルだけ。',
            detail='旧加工監査と今回大会20対象の評価は区別。元実験はproxy真値を利用した同点jitterを加えていたが、今回は大会公式の同点を維持しjitterを加えない。')
    # 転帰を含む連続8列の既存NN差（submit_audit m6_outcome_nn）。
    submit_path = archive/'work_20260901以前/submit_audit.py'
    submit = extracted(submit_path,['_z','m6_outcome_nn'],dict(env,CONT8=NUM7+['time']))
    add_matrix('legacy_outcome_nn8',np.column_stack([submit.m6_outcome_nn(cs[k],bg,b) for k in teams]),
        source=str(submit_path.relative_to(root))+'#m6_outcome_nn',attributes=NUM7+['time'],
        params='kC=1,kA=2;A_bg SD;time込み8次元',family='旧転帰NN',
        verification='既存pure定義再計算・現公式採点')

    # 旧AIA課題を用いたMIA診断の5特徴。proxyTPRの保存値を混ぜず同じ
    # 性別別最近傍関数を大会B候補へ当てる。県条件は未定義として除く。
    bench_path = archive/'work_20260901以前/aia2mia/bench.py'
    source_paths.add(bench_path)
    from scipy.spatial import cKDTree
    num5 = ['age','SBP','TG','HDL','ALT']
    sd5 = bg[num5].std(ddof=1).to_numpy()
    sex_feature = {name:np.empty((len(ids),len(teams))) for name in ['d1','d1_pref','r12','dr1','dr5']}
    bx = b[num5].to_numpy()/sd5
    for j,k in enumerate(teams):
        for sex in [0,1]:
            mask = b.sex.to_numpy()==sex
            sc = cs[k][cs[k].sex.eq(sex)]
            sa = bg[bg.sex.eq(sex)]
            dc = cKDTree(sc[num5].to_numpy()/sd5).query(bx[mask],k=5)[0]
            da = cKDTree(sa[num5].to_numpy()/sd5).query(bx[mask],k=5)[0]
            sex_feature['d1'][mask,j] = -dc[:,0]
            sex_feature['r12'][mask,j] = -dc[:,0]/(dc[:,1]+1e-9)
            sex_feature['dr1'][mask,j] = np.log(da[:,0]+1e-9)-np.log(dc[:,0]+1e-9)
            sex_feature['dr5'][mask,j] = np.log(da[:,4]+1e-9)-np.log(dc[:,4]+1e-9)
        # 旧県条件はchallenge地元県に固定。B候補全県へ外挿すると未定義のため
        # この1特徴は大会全B対象として再定義せず、明示的に未確認とする。
    for name in ['d1','r12','dr1','dr5']:
        add_matrix('legacy_aia2mia_sex_'+name,sex_feature[name],source=str(bench_path.relative_to(root))+'#'+name,
            attributes=num5+['sex'],params='同性内cKDTree;ddof1;d1/r12/dr1/dr5元定義;eps1e-9',
            family='旧性別内近傍MIA',verification='元定義のB候補scoreを再計算・現公式採点',
            detail='proxy帰無の推定TPRではない。距離関数だけをそのままB候補へ適用。')
    note('aia2mia d1_pref','条件の大会Bへの外挿未定義・未確認','旧実験は各challengeの地元県に固定。全B候補へ県条件を変更する新定義は作らない。',str(bench_path.relative_to(root)))
    note('旧atkの固定公開入力条件', '補充再評価', 'historical_atk.pyが除染背景・sex/県block・距離比を補充。元seed1半分割は元rng消費順が定義された24/26/29の3対象だけ。', str((HERE/'historical_atk.py').relative_to(root)))
    note('旧Python/旧相対参照', 'main不実行', '残存コードの旧Python固定パス・work/data_prelim_team05等は書換えず、正規配布入力を読んで純関数のみ使用。', 'アーカイブ/予備戦_攻撃フェーズ')
    # 呼出側の任意保存用。個人別scoreや所属正解は含まれない。
    gather.last_source_hashes = {str(p.relative_to(root)): _sha(p) for p in source_paths}
    return records, notes


if __name__ == '__main__':
    task_root = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE.parents[4]
    rs, ns = gather(task_root, use_cache=False)
    payload = dict(records=rs, notes=ns, source_hashes=gather.last_source_hashes)
    destination = HERE/'historical_cache.json'
    # このタスクの中で作った集計cacheだけを更新する。元実験には書かない。
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(records=len(rs), methods=len({r['method'] for r in rs}),
                         over25=sorted({r['team'] for r in rs if r['tpr']>.25}), notes=len(ns)), ensure_ascii=False))
