"""全監査・固定hash合格後に候補を確定する。既存の結果を上書きしない。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import importlib.util
import json
import sys
import unicodedata

sys.dont_write_bytecode = True
FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[3]
CANDIDATE = FOLDER / '候補'
FINAL = FOLDER / '結果'
EXPECTED = '6ca9604a8936fb47255ae0bd05c9f3958368f0e196bc2af258f8255cf39c08eb'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def allowed(name):
    path = (ROOT / name).resolve()
    check(path.is_relative_to(ROOT), 'project外の確認を拒否')
    parts = {unicodedata.normalize('NFC', x) for x in path.parts}
    check(not ({'archive', 'アーカイブ'} & parts), 'archive読み取りを拒否')
    return path


def main():
    check(CANDIDATE.is_dir() and not FINAL.exists(), '新しい確定先が必要')
    check(sha(CANDIDATE / 'F_mia.csv') == EXPECTED, 'CSV SHA不一致')
    names = ['全体監査.json', '希少監査_再検証.json', '統合監査.json',
             '復号監査_ラベル版最終.json', '再作成一致.json']
    a, b, c, d, e = [read(CANDIDATE / n) for n in names]
    check(a['status'] == 'passed' and b['audit_status'] == 'PASS' and c['passed']
          and d['audit_status'] == 'PASS' and e['passed'], '監査未合格')
    check(a['candidate']['ordinary_cells'] == 19250 and a['candidate']['submit_values_exact']
          and a['candidate']['final_ranks_exact'], '全体一致の確認不備')
    check(a['candidate']['private_ordinary_zero_cells'] == 3300, '通常非会員0化不備')
    check(b['rare_csv_cells']['all_rare_cells_checked'] == 528
          and b['rare_csv_cells']['zero_member_team_cells_checked'] == 384, '希少0化の確認不備')
    check(c['official_validation']['passed'] and all(c['independent_strict_format'].values()),
          '形式監査不備')
    for audit, key, n in [(a, 'intentional_errors', 6), (b, 'deliberate_mutations', 10)]:
        check(len(audit[key]) == n and all(x['rejected'] for x in audit[key]), '改変拒否不備')
    check(c['mutation_tests']['rejected_cases'] == 28 and c['mutation_tests']['all_rejected'],
          '統合改変拒否不備')
    check(e['F_mia_byte_identical'] and all(e['byte_identical_files'].values())
          and all(e['npz_arrays_identical'].values()), '再作成不一致')
    check(d['all_five_mean_results_identify_nine_and_outside_two']
          and d['partial_prefix_true_candidate_retention'], '復号監査不備')
    label = d['recommended_labeled_input_checks']
    check(label['random_correctly_labeled_argument_permutations'] == 100
          and label['reversed_partial_label_cases'] == 4
          and label['all_restore_correct_membership_or_next_question'], 'ラベル並べ替え監査不備')
    check(len(label['rejections']) == 14 and all(x['rejected'] for x in label['rejections']),
          'ラベル拒否監査不備')
    check(len(d['rejections']) == 13 and all(x['rejected'] for x in d['rejections']), '復号拒否監査不備')
    check(sha(FOLDER / '結果読取.py') == d['decoder_sha256'], '読取器hash不一致')
    check(sha(FOLDER / 'decoder_audit.py') == d['auditor_sha256'], '復号監査器hash不一致')
    for name, digest in [('audit_main.py', '1ac3a9a4258586b7acb2cbafeb4a075850e3cd7e78dbcbd6efefd57dba16a5d6'),
                         ('audit_rare.py', b['auditor_sha256']), ('audit_integration.py', c['auditor_sha256'])]:
        check(sha(FOLDER / name) == digest, f'監査器hash不一致: {name}')
    check(b['candidate_sha256'] == c['candidate_sha256'] == e['F_mia_sha256'] == EXPECTED,
          '監査・再作成のCSV hash不一致')
    pins = read(FOLDER / 'input_pins.json')['input_sha256']
    check(len(pins) == 63 and pins == read(CANDIDATE / 'inputs_manifest.json')['input_sha256'],
          '入力固定・manifest不一致')
    watched = c['checked_input_sha256']
    for entries in [pins, watched]:
        for name, digest in entries.items():
            check(sha(allowed(name)) == digest, f'固定入力・過去提出物の変更: {name}')

    spec = importlib.util.spec_from_file_location('_round3_final_integration', FOLDER / 'audit_integration.py')
    integration = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = integration
    spec.loader.exec_module(integration)
    context = integration.read_context()
    with (CANDIDATE / 'F_mia.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    errors, frame, official = integration.verify_rows(rows, context)
    check(not errors and official['passed'], f'確定直前形式検証失敗: {errors}')
    original_settings = (CANDIDATE / '設定.json').read_bytes()
    with (CANDIDATE / '作成時設定.json').open('xb') as f:
        f.write(original_settings)
    CANDIDATE.rename(FINAL)
    settings = json.loads(original_settings)
    settings.update(status='作成・独立監査済み・未提出', official_format_passed=True, F_mia_sha256=EXPECTED)
    (FINAL / '設定.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    check(sha(FINAL / 'F_mia.csv') == EXPECTED, '確定移動後のCSV変更')
    with (FINAL / 'F_mia.csv').open(newline='', encoding='utf-8') as f:
        after_rows = list(csv.reader(f))
    errors, _, official_after = integration.verify_rows(after_rows, context)
    check(not errors and official_after['passed'] and after_rows == rows, '確定後形式検証失敗')
    for entries in [pins, watched]:
        for name, digest in entries.items():
            path = allowed(name)
            if path.is_relative_to(CANDIDATE):
                path = FINAL / path.relative_to(CANDIDATE)
            check(sha(path) == digest, f'確定後の監査対象変更: {name}')
    confirmation = {
        'passed': True, 'completed_jst': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        'round': 3, 'final_csv': str(FINAL / 'F_mia.csv'), 'F_mia_sha256': EXPECTED,
        'rows': 1049, 'targets': 22, 'own_team_excluded': 28,
        'rare_method': '固定5bitの第1問・事前固定した第3〜7回質問',
        'ordinary_method': 'Gen-LRA7 + 県性別非会員較正500 + 自対象除外中央値補正0.1',
        'zero_member_team_rare_cells': 384, 'ordinary_submit_cells_independently_exact': 19250,
        'rare_cells_independently_exact': 528, 'known_nonrare_negative_zero_cells': 3300,
        'audits': {n: {'sha256': sha(FINAL / n), 'passed': True} for n in names},
        'deliberate_csv_or_formula_mutations_rejected': {'ordinary': 6, 'rare': 10, 'integration': 28, 'total': 44},
        'decoder_checks': {'random_legal_configurations': 100, 'targeted_single_pair_coverage': 198,
                           'partial_histories': 400, 'invalid_history_cases_rejected': 13,
                           'labeled_argument_permutations': 100, 'invalid_labeled_inputs_rejected': 14},
        'reproduction': {'csv_byte_identical': True, 'byte_identical_generated_files': 8, 'identical_arrays': 9,
                        'settings_notice': '作成時設定.jsonは再作成一致の対象。設定.jsonのstatusと最終hash情報のみ確定時に更新。'},
        'official_validation_after_promotion': official_after,
        'pinned_inputs_checked_again': len(pins), 'all_watched_files_checked_again': len(watched),
        'inputs_and_old_submissions_unchanged': True, 'original_candidate_csv': str(CANDIDATE / 'F_mia.csv'),
        'historical_audit_path_notice': '監査報告の候補パスは監査時点の記録。結果へ移動後もCSVは同じSHA256。報告原本は書き換えていない。',
        'initial_rare_audit_failure_preserved': '希少監査.jsonの初回FAILは17桁文字列をDecimalで直接比較した監査器の誤検出。CSV変更なし。監査器修正後の希少監査_再検証.jsonがPASS。',
        'operational_requirement': '詳細結果の生のmia.tpr_rareを実提出CSV・採点原本・回番号と照合し、--round-meansで入力。誤ラベルは別の合法所属に見える場合があり、数字だけで原本対応は証明できない。',
        'source_sha256': {p.name: sha(p) for p in sorted(FOLDER.iterdir()) if p.is_file()},
        'performance_notice': '未知の本戦個人所属に対する改善量は未検証。形式と合意した計算・符号・復号の監査結果。',
        'scope': 'MIAのみ。F_aia/token/ZIPは未作成。大会へ未提出。', 'submitted': False, 'main_individual_truth_used': False,
    }
    write_new(FINAL / '最終確認.json', confirmation)
    write_new(FINAL / '出力SHA256.json', {p.name: sha(p) for p in sorted(FINAL.iterdir()) if p.is_file()})
    check(all(sha(FINAL / n) == digest for n, digest in read(FINAL / '出力SHA256.json').items()), '最終出力manifest不一致')
    print(json.dumps({'passed': True, 'final_csv': str(FINAL / 'F_mia.csv'), 'rows': len(frame),
                      'targets': len(frame.columns) - 1, 'F_mia_sha256': EXPECTED, 'submitted': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
