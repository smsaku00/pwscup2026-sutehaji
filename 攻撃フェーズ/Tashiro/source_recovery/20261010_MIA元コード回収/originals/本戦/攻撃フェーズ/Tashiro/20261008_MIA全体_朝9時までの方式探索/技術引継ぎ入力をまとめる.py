"""保存済み集約値だけを別ディレクトリへ固定する。最終文章は作成しない。

このZIPは分析資料。CodaBench提出物ではない。個人行、所属正解、
内部順位NPZ、F_mia/F_aia/tokenを含めない。ChatGPT 6 Proの執筆入力用。
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

HERE = Path(__file__).resolve().parent

UNREFERENCED_JSON_HASHES = {
    '公開背景行の除外診断結果/事前固定条件.json': 'ea62523bc06f12627d5ba33c9fcb7e2a587fbb747078a240a2c3898dbea61d9b',
    '公開背景行の除外診断結果/完了記録.json': '46ab271737386ed63c3d19b1b4ac6a6658be6fc7632446cc38cd296027abae29',
    '公開背景の分割較正結果/事前固定条件.json': 'b234acbd100b244e62e4f2682045f2c0a1fcd9e265c6ef8a19b4fd7ffd86cafa',
    '公開背景の分割較正結果/完了記録.json': 'a15924abf4e62c162600d715a020192a1acf84f1874a5d8908b5ba06e110aca9',
    '公開背景の分割較正結果/二fold平均CDFの解釈上の制限.json': 'd1177c70aa4f423c84f9370e42cf6e19d4c7c673452024ef9efdde9cd9c75ef4',
    '局所支持区間の密度比結果/事前固定条件.json': '068c75f2d479355ae693cb0924bf74cd8fee1d51ed0a1b5cfdc064628443fff9',
    '局所支持区間の密度比結果/完了記録.json': 'a6789e9c8d7acc3b5467ec58879dced1c8c9e6768e2b27eac5aff7c23171da1f',
    '局所支持区間の密度比結果/異なる生スコアのCRPS比較制限.json': '15f2898f11ae728697071a7e34ded163ca846d79c783745addb351a9dc32280b',
    '公開非会員の較正結果/実験方針.json': 'f98d2382483f7a693c6a57c49372c6f57877e585c2ab63afc3d2e5be3b389bf0',
    '広い集団への非会員較正結果/実験方針.json': 'a258dcd0bac507e354207c4936af09c521a303d6d39a8b51e8094b61a4278f6e',
    '共有中順位座標の密度比結果/事前固定条件.json': '95a638716d4101d56c161e4c9257379bb9bb548461f41c8ef299107c45daea5b',
    '共有中順位座標の密度比結果/完了記録.json': '9c850909e4b0b06759f1572bbfec6d2e20c28c2d50ace3e143eebd8bf32ffc56',
    '離散依存木の密度比結果/事前固定条件.json': 'bfcc87e8bf07e5c4d4042a26d872e633ef9ff18ab763b49a762932efda51a83f',
    '公開背景三分割の独立非会員診断結果/完了記録.json': '24aca865c730df110f7255ac9dd795714994e7e53dc6f484f5c5c7dae2f92075',
    '公開背景三分割の独立非会員診断結果/fullA年齢層にテスト共変量を含む制限.json': 'f687e65c1e9409ffde3cc0ad7ee600a517b35b5c6633ff37886a79dbd126e886',
    '公開背景三分割の独立非会員診断結果/独立数式と集約の検証.json': '1a8b06f7518f3160fc8baff6abf08d655e9f8b804d9ba3c5139acba3fe3362d7',
    '三分割Atestの公開C一致層別診断結果/完了記録.json': '6a9de4d42327afdd74a4f185efd3cc20e0ed18ac18180dc6d83262a0ffd90c4a',
    '三分割Atestの公開C一致層別診断結果/独立集約と旧三分割対照の検証.json': '0e3626f2782e5d37c2a7be890ed1321a96596598886160844e2775bf38a777ff',
    '理想CDF縮約の条件付き尾率結果/事前固定計画.json': '1b8d913ac1bffbc4738ad95593c70bd284950877c0c467a0673615283bddadfc',
    '公開Aの有限標本p値結果/事前固定条件.json': '95203856e33a16aa8bdff26ce106a546200ab11bb1f5a0b6d41bd18dced143fb',
    '公開Aの有限標本p値結果/完了記録.json': '4639120e3971c622e9435f2f83593d1467535274fe5872804374781fc24e5835',
    '有限p値の整数解像度結果/事前固定計画.json': 'c94d742869d9aa59c83575d34dbee902c738a97caebb5cd9aa95fc1d227ee21c',
    '有限p値の整数解像度結果/整数解像度の数式と制限.json': '2b3c99094cab810219955b6da038cc59974f84793e668ed802fe2aa20092d5cb',
    '公開QとCの双方向支持必要条件_カテゴリ文字列対応結果/独立集約と出力範囲の検証.json': 'd28110f20f83d1967fdb26d6ab98ba6cee9a5355465d8ada4fb30e08fe8d6bbe',
    '公開QとCの重複数を保つ対応容量結果/独立容量と出力範囲の検証.json': '3e89e3867d82439530d8b9507d3cad0dc7630d109ab9b7e5e59147e54cb5affb',
}

PUBLIC_DIRS = [
    '公開加工特徴の比較結果', '公開分布の尾の診断結果',
    '公開値の刻みとカテゴリ結果', '公開背景との属性単位一致結果',
    '公開データのフェーズ間重なり結果', '公開非会員の較正結果',
    '非会員課題の丸ごと保留結果', '広い集団への非会員較正結果',
    '公開背景行の除外診断結果', '公開背景の分割較正結果',
    '公開課題の除外に対する順位感度結果', '局所支持区間の密度比結果',
    '通常順位を通した非会員診断結果', '公式提出のMIA全体記録結果',
    '公開較正の局所標本数結果',
    '本戦公開特徴の集約表結果',
    '共有中順位座標の密度比結果', '共有GaussianのA分割seed感度結果',
    '公開背景との高次属性一致結果', '離散依存木の密度比結果',
    '縮約係数と局所支持の図結果',
    '公開背景三分割の独立非会員診断結果',
    '三分割Atestの公開C一致層別診断結果', '理想CDF縮約の条件付き尾率結果',
    '公開Aの有限標本p値結果',
    '有限p値の整数解像度結果',
    '公開QとCの双方向支持必要条件_カテゴリ文字列対応結果',
    '公開QとCの重複数を保つ対応容量結果',
]

# 集約粒度を保ち、個別患者・内部予測値を出さない明示的allowlist。
CSV_NAMES = {
    '対象別の公開特徴.csv', '性別ごとの分布幅.csv', '性別ごとの属性間相関.csv',
    '性別属性ごとの中央と尾.csv', '数値の刻みと集中.csv',
    'カテゴリとtimeの支持.csv', '同番号の特徴変化.csv',
    '属性単位の公開値共有.csv', '属性組と行全体の公開値一致.csv',
    'time右端と非右端の刻み.csv', 'time区分別のonset_death件数.csv',
    '3背景の行支持重なり.csv', '公開集合全体の一致.csv',
    '公開Cのフェーズ間一致.csv', '強度選択の曲線.csv',
    '基準との順位差.csv', '所有者単位のCRPS.csv',
    '課題所有者を保留した較正.csv', '固定条件の集計.csv',
    '外側保留ごとの内側選択.csv', '課題所有者の外側保留.csv',
    '基準との上位順位差.csv', '固定六案の非会員診断.csv',
    'Q所有者等重みの診断パネル.csv', 'Qのみflatとの上位12変化.csv',
    '同一参照の生スコア検証.csv', '対象ごとの課題除外感度.csv',
    '課題除外ごとの順位差.csv', '固定三形の非会員診断.csv',
    '点Gaussianとの上位12変化.csv', '観測支持と仮定した区間幅.csv',
    '全対象共通の固定条件診断.csv', '公開背景一致と除外数.csv',
    '固定元方式の再現対照.csv', '対象別の非会員診断差.csv',
    '通常875行の上位12変化.csv',
    '同一参照の密度比検証.csv', 'フェーズ別の通常閾値超過.csv',
    '対象別の通常閾値超過.csv', '所有者別の通常閾値超過.csv',
    'B順位対照.csv', '本戦のMIA全体公式履歴.csv', '本戦の対象別公開バンド.csv',
    '集団別支持数の分布.csv', '支持数のフェーズ別集約.csv', '局所標本数と縮約係数の重み.csv',
    '本戦22対象の公開特徴と公式MIA全体バンド.csv',
    '数値floorのdataset発生数.csv', '同一共有座標の密度比検証.csv', 'robustLWとの上位12変化.csv',
    '固定三形の分割感度集約.csv', 'primary対象平均のseed別感度.csv',
    '四seedと元seedを含む上位集合の共通部分.csv', 'primaryCsourceのfloorとclamp集約.csv',
    '対象別seed順位感度.csv', '単独query一致の集約検証.csv',
    '高次属性の同時一致と独立周辺期待.csv', 'batchとsingleの生スコア検証.csv',
    '独立形との上位12変化.csv', '経験pairMIと木の辺.csv', '木構造とPMF正規化.csv',
    'Atestの対象fold別通常順位閾値超過.csv', '同一参照と凍結順位の数値QA.csv',
    'Atestのfold別固定六CDF診断.csv', '三分割の非重複と役割.csv',
    'Atestのfold別通常順位閾値超過.csv', 'Atestの対象fold別CRPSと単CDF診断.csv',
    'Atestの三fold集約通常順位閾値超過.csv',
    '公開C支持へのAtest一致三flag組合せ件数.csv', '元三分割対照と同一参照数値QA.csv',
    '一致層別の三fold対象exposure集約.csv', '一致層別の通常順位閾値超過.csv',
    '一致層別の単CDFとCRPS集約.csv',
    '固定グリッドの群別尾率.csv', '独立積分と分位点の検証.csv',
    '対象fold別の有限標本p値診断.csv', '公開生スコアとCounterの数値QA.csv',
    'fold別の有限標本p値診断.csv', '三fold集約の有限標本p値診断.csv',
    '群別の較正標本数と閾値到達可否.csv',
    '群とalpha別の整数解像度.csv', 'phaseとgroup別の期待尾率.csv',
    'foldとgroup別の期待尾率.csv',
    '本戦22の固定12属性集合と双方向支持必要条件.csv',
    '本戦22の固定12集合の最大対応容量とprivate不足下限.csv',
}

SOURCE_CODES = [
    '共通.py', '昨日の再評価.py', '昨日の学習を18対象で確認.py',
    '距離と分布の探索.py', '公開データ分類器の探索.py', '直接密度比の探索.py',
    '分布を揃えた距離の探索.py', '混合分布と局所密度の探索.py',
    '階層ヒストグラムの探索.py', '複数背景と乱数で学習.py',
    '複数ビューの対象保留確認.py', '公開非会員だけで較正を選ぶ.py',
    '非会員較正の広い集団を保留検証.py', '公開背景行を除いた診断.py',
    '非会員課題を丸ごと保留.py',
    '公開背景の分割較正.py', '局所支持区間の密度比.py',
    '通常順位を通した非会員診断.py', '公開課題の除外に対する順位感度.py',
    '公開Cの特徴を整理.py', '公開分布の尾を確認.py',
    '公開値の刻みとカテゴリを確認.py', '公開背景との属性単位一致.py',
    '公開データのフェーズ間重なり.py', '公式提出のMIA全体記録.py',
    '公開較正の局所標本数を集計.py',
    '本戦公開特徴の集約表を作る.py',
    '共有中順位座標の密度比.py', '共有GaussianのA分割seed感度.py',
    '公開背景との高次属性一致.py', '離散依存木の密度比を確認.py',
    '縮約係数と局所支持の図.py',
    '公開背景三分割の独立非会員診断.py',
    '三分割Atestの公開C一致層別診断.py', '理想CDF縮約の条件付き尾率を確認.py',
    '公開Aの有限標本p値を確認.py',
    '有限p値の整数解像度.py',
    '公開QとCの双方向支持必要条件_カテゴリ文字列対応.py',
    '公開QとCの重複数を保つ対応容量.py',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def csv_metadata(path):
    with path.open(newline='') as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        rows = sum(1 for _ in reader)
    prohibited = {'record_id', 'patient_id', 'A_row_position', 'c_row_position',
                  'row_position', 'query_row', 'candidate_index', 'is_member', 'membership_label'}
    assert not prohibited.intersection(columns), (path, columns)
    return dict(rows=rows, columns=columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--extra-public-dir', action='append', default=[])
    parser.add_argument('--extra-source-code', action='append', default=[])
    parser.add_argument('--extra-csv-name', action='append', default=[])
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert out.parent == HERE and not out.exists(), out
    for relative, expected in UNREFERENCED_JSON_HASHES.items():
        assert sha(HERE/relative) == expected, relative
    assets = set()
    for name in PUBLIC_DIRS + args.extra_public_dir:
        directory = HERE/name
        assert directory.is_dir(), directory
        for path in directory.rglob('*.json'):
            if path.name in {'Arefの共有座標と有限中順位境界.json', 'source別Gaussian固定fit.json',
                             'Arefの固定bin境界.json'}:
                # 共有座標/全fitは本文執筆に不要。run recordのhashだけを渡す。
                continue
            # JSONは実行条件・集約QA・支持値。個別候補順位を含めない。
            value = json.loads(path.read_text())
            if isinstance(value, list):
                assert path.relative_to(HERE).as_posix() in {
                    '局所支持区間の密度比結果/prelim/性別ごとの固定尺度と帯域.json',
                    '局所支持区間の密度比結果/main/性別ごとの固定尺度と帯域.json',
                }, path
                required = {'phase', 'fold', 'sex', 'reference_rows', 'h',
                            'numeric_scale', 'numeric_center', 'category_levels'}
                assert all(isinstance(item, dict) and required.issubset(item) for item in value), path
            else:
                assert isinstance(value, dict), path
            assets.add(path)
        for path in directory.rglob('*.csv'):
            if path.name in CSV_NAMES | set(args.extra_csv_name):
                csv_metadata(path)
                assets.add(path)
    for name in SOURCE_CODES + args.extra_source_code:
        path = HERE/name
        assert path.suffix == '.py' and path.is_file(), path
        assets.add(path)
    for name in ['縮約係数と局所支持の図結果/shrinkage_and_support.png',
                 '縮約係数と局所支持の図結果/shrinkage_and_support.svg',
                 '理想CDF縮約の条件付き尾率結果/条件付き尾率の曲線.png',
                 '理想CDF縮約の条件付き尾率結果/条件付き尾率の曲線.svg']:
        assets.add(HERE/name)
    for name in ['公開入力と非会員保証の範囲.json', '公開非会員の根拠と解釈の監査.json',
                 '独立監査の技術記録.json', '旧学習履歴の対応条件監査.json',
                 '公開密度比の追加環境と参照情報.json', '評価対象と通常分岐の定義.json',
                 '公開Q県集合の行対応必要条件.json',
                 '公開A分割の交換可能性と外部代表性.json',
                 '公開A分割の交換可能性と外部代表性_前提の補足.json']:
        assets.add(HERE/name)
    # 過去の探索値。正解を再読・再採点せず、保存済み集約CSVを固定する。
    for name in ['昨日の再評価結果/昨日の方式_18対象再評価.csv',
                 '昨日の再評価結果/全背景の比較.csv',
                 '複数ビューの対象保留結果/全条件の比較.csv',
                 '複数ビューの対象保留結果/方式一覧.csv',
                 '複数ビューの対象保留結果/実行記録.json',
                 '複数ビューの対象保留結果/実験方針.json',
                 '複数ビューの学習結果/全条件の比較.csv',
                 '複数ビューの学習結果/方式一覧.csv',
                 '複数ビューの学習結果/実行記録.json',
                 '複数ビューの学習結果/実験方針.json',
                 '再現監査_20261008_054516/18対象の保存結果.csv',
                 '再現監査_20261008_054516/実行記録.json']:
        assets.add(HERE/name)
    out.mkdir()
    manifest = []
    for path in sorted(assets):
        assert path.is_file() and 'アーカイブ' not in path.parts, path
        relative = path.relative_to(HERE)
        assert path.suffix in {'.json', '.csv', '.py', '.png', '.svg'}, path
        assert path.name not in {'F_mia.csv', 'F_aia.csv', 'token.txt'}, path
        dest = out/'evidence'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        expected = sha(path)
        assert sha(dest) == expected
        item = dict(source_relative_path=str(relative), packet_path=str(dest.relative_to(out)),
                    sha256=expected, bytes=path.stat().st_size)
        if path.suffix == '.csv':
            item.update(csv_metadata(path))
        manifest.append(item)
    old = pd.read_csv(HERE/'再現監査_20261008_054516/18対象の保存結果.csv')
    old = old[old.fpr.eq(.01)].copy()
    old['family'] = old.source.str.split('/').str[0]
    rows = []
    for family, part in old.groupby('family'):
        maximum = part.loc[part.other18.idxmax()]
        rows.append(dict(family=family, saved_condition_rows=len(part),
                         unique_method_names=part.method.nunique(),
                         exploratory_max_other18=maximum.other18,
                         corresponding_all20=maximum.all20,
                         historical_max_source=maximum.source,
                         historical_max_method=maximum.method,
                         independent_method_count=False,
                         independent_transfer_test=False,
                         proposal_or_adoption=False))
    pd.DataFrame(rows).to_csv(out/'探索履歴の集約.csv', index=False, float_format='%.17g')
    supervised = pd.read_csv(HERE/'複数ビューの学習結果/全条件の比較.csv')
    stronger = pd.read_csv(HERE/'複数ビューの対象保留結果/全条件の比較.csv')
    matched_history = {}
    for label, frame, method in [
        ('target_fourfold_balanced_full', supervised,
         'without_kde_balanced_background_and_centers_five_seed_mean'),
        ('target_label_LOO_balanced_full', stronger,
         'without_kde_LOO_balanced_background_and_centers_five_seed_mean'),
    ]:
        chosen = frame[frame.fpr.eq(.01) & frame.condition.eq('full') & frame.method.eq(method)]
        assert len(chosen) == 1
        matched_history[label] = chosen.iloc[0].to_dict()
    write_json(out/'対応する固定履歴条件.json', dict(
        records=matched_history,
        condition='full',
        adaptive_history=True,
        independent_transfer_performance=False,
        membership_labels_reread=False,
        selection_or_retuning_performed=False,
        global_adaptive_max_is_a_different_condition=True,
        highest_saved_other18_in_manyview_history=float(supervised.loc[supervised.fpr.eq(.01), 'other18'].max()),
        highest_saved_other18_in_target_LOO_history=float(stronger.loc[stronger.fpr.eq(.01), 'other18'].max()),
        manyview_historical_max_records=supervised[
            supervised.fpr.eq(.01) & supervised.other18.eq(supervised.loc[supervised.fpr.eq(.01), 'other18'].max())
        ].to_dict(orient='records'),
        target_LOO_historical_max_records=stronger[
            stronger.fpr.eq(.01) & stronger.other18.eq(stronger.loc[stronger.fpr.eq(.01), 'other18'].max())
        ].to_dict(orient='records'),
        best2_gain_units='sum of target-level TPR differences, not mean TPR',
    ))
    write_json(out/'技術入力仕様.json', dict(
        created_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        kind='technical_evidence_packet_not_final_report',
        scope='20261008 MIA全体 goal only',
        requested_final_author='ChatGPT 6 Pro',
        handoff_not_before_jst='2026-10-08T09:00:00+09:00',
        no_model_substitution=True,
        repository='smsaku00/pwscup2026-sutehaji',
        repository_private=True,
        provisional_summary_location='攻撃フェーズ/Tashiro/MIA全体/experiments/20261008/',
        summary_location_requires_fresh_repository_check=True,
        role='whole-MIA supplemental analysis/proposal; team adoption not established',
        no_codabench_submission=True,
        no_contest_answer_files=True,
        raw_patient_rows_included=False,
        membership_truth_included=False,
        individual_candidate_ranks_or_NPZ_included=False,
        package_is_self_contained_executable_reproduction=False,
        omitted_reproduction_dependencies='original public/own-private input rows and older support modules are not attached; frozen records pin their hashes',
        execution_by_final_author_requested=False,
        old_exploration='adaptive history; saved maxima are not independent transfer performance',
        method_counts='aliases, seeds and shared-cohort views are not independent methods/cohorts',
        primary_historical_evaluation=dict(targets=18, excluded_prelim_targets=[1, 2], fpr=.01),
        target_exclusions='membership-label training and primary evaluation exclude prelim1/2; public C/Q are not universally removed',
        fixed_cross_correction='other-target median coefficient .1 computed jointly over prelim20 or main22',
        primary_baseline_other18=.0297752453308016,
        adaptive_balanced_full_target_fourfold_other18=matched_history['target_fourfold_balanced_full']['other18'],
        matched_balanced_full_target_label_LOO_other18=matched_history['target_label_LOO_balanced_full']['other18'],
        target_LOO_still_transductive_public_C=True,
        public_controls='foreign Q or same-phase split-heldout A; own-target Q excluded',
        controls_disjointness_not_representativeness=True,
        no_membership_label_or_score_tuning_by_new_public_diagnostics=True,
        no_team_id_based_score_winner_stitching=True,
        separate_scales=['prelim20', 'prelim18', 'main22 official score'],
        crucial_limits=[
            'Lower public nonmember CRPS is not MIA/FPR improvement.',
            'CRPS comparison requires same raw variable and evaluation/scale protocol.',
            'Mean of two CDFs need not be uniform PIT; fold-average CRPS is not mixture-CDF CRPS.',
            'Q normal-branch head exceedance is not official FPR or correctness.',
            'Deletion stress inclusion is not correctness probability.',
            'Same numeric team ID does not establish same processing algorithm.',
            'Public source field overlap is not cross-phase person identity/disjointness proof.',
            'Historical/kit A has no established nonmember guarantee for current main cohorts.',
            'IQR describes central width; extreme tail claims require separate tail widths.',
            'Official per-target bands do not disclose exact TPR or correct-person counts.',
            'Clipped transformations, latent support intervals and public source-density models are hypotheses.',
            'The post-rank Q exceedance diagnostic covers full-A Gen with three fixed CDFs only.',
            'Three-A overlap counts 225/381 are not the two-A removal counts 133/228.',
            'Target4fold and targetLOO change both held-out granularity and training-target count.',
            'A three-way diagnostic holds out A test scores, but full-A age strata include test covariates.',
            'A test rows are not guaranteed independent of how public C was constructed.',
            'A-test normal-branch head exceedance is not candidate FPR or membership accuracy.',
            'Single-CDF p99 and top12 of B875 use different thresholds and reference distributions.',
            'A split seed stability is within the same cohort; high correlation does not guarantee stable heads.',
            'Discrete source trees are a categorical adaptation, not a replication of continuous forest density estimation.',
            'A/C support-match strata are post-discovery descriptions; clinical3 includes FPG outside QI7 and strata use a priority partition.',
            'A-row to C-support matches and C-row to A-support matches can have different counts/multiplicities.',
            'Population-CDF shrinkage examples are mathematical counterexamples, not estimates of actual-data FPR or method performance.',
            'Frozen C dependence alone does not negate random cal/test allocation symmetry inside the same finite A pool.',
            'Pure-A finite-p diagnosis has no Q mixture, candidate B rank or cross; it is a different calibration stage.',
            'Small-group finite-p resolution can force no rejections; low observed tail rate is not proof of MIA power.',
            'Pure-A diagnostic tests each A row in exactly one fold; p-values from other folds were not averaged.',
            'The finite-p integer expected tail assumes tie-free uniform conditional ranks; ties can make it more conservative, and one fixed allocation need not equal its expectation.',
            'Q/C support and matching capacity provide necessary conditions for stated value-preserving correspondence hypotheses, not recovered processing algorithms or membership predictions.',
            'Private-Q shortage lower bounds count occurrences in the unknown private163 subset; no individual private row is identified.',
            'Q contains no BMI or FPG: those 44 support cases are unavailable and blank, not zero matches.',
            'Passing either support or capacity conditions does not prove the unknown private subset can be preserved.',
        ],
        non_referenced_JSON_hashes_separately_pinned=UNREFERENCED_JSON_HASHES,
        final_prose_created_by_this_script=False,
        external_action_performed=False,
    ))
    write_json(out/'manifest.json', dict(
        code_path=str(Path(__file__).resolve()), code_sha256=sha(Path(__file__)),
        copied_assets=manifest,
        generated_assets={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        all_copy_hashes_match=True,
        manifest_does_not_hash_itself=True,
    ))
    zip_path = out.with_suffix('.zip')
    assert not zip_path.exists(), zip_path
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(out))
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        for item in manifest:
            assert hashlib.sha256(archive.read(item['packet_path'])).hexdigest() == item['sha256']
    print(json.dumps(dict(packet_directory=str(out), zip_path=str(zip_path),
                          zip_sha256=sha(zip_path), assets=len(manifest),
                          bytes=zip_path.stat().st_size, final_prose_created=False,
                          contest_submission=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
