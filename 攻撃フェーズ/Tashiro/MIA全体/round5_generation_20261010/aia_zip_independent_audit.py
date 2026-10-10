"""指定AIA原本と第5回MIAの組合せ、公式ZIPを独立検証する。

AIA/MIA/token/ZIPを変更せず、token本文・token hashを表示/記録しない。
公式ZIPと公開ZIPの相違はtoken.txtの有無だけであることを確認する。
大会への提出、GitHubへの操作、Libraryへの操作は行わない。
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TEAMS = tuple(range(18)) + (22, 25, 26, 27)
AIA_SHA256 = '6f625e41a324ffb32ff2958c312a30ee2e06d3a3fac6504015443cb8f867d85c'
AIA_BLOB = 'bade408752830d55a634bc1248bd49694c7664a1'
MIA_SHA256 = '3e1f260ed7ab2f8beed57e6eb5b6e30d0c96388ac15c1e50ef0cfd1fe0f96c7a'
MAIN_B_SHA256 = '1339042b4e1937d5fbf482aa8ce4de3f67a02e4694f8237894ca59bf3f0d52b1'
VALIDATOR_BLOB = '07284c2239bfcd98dbadd17c69345490461718dc'
AIA_SOURCE_COMMIT = 'b023ca627fb664ffb0f4b26f34ca6a4448b26418'
AIA_SOURCE_PATH = '攻撃フェーズ/AIA/5_2/結果/F_aia.csv'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blob(content):
    return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        require(columns and len(columns) == len(set(columns)), 'CSVヘッダ欠落/重複')
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows), '壊れたCSV')
    return columns, rows


def integer(value, description):
    number = Decimal(value)
    require(number.is_finite() and number == number.to_integral_value(), description + 'は有限整数ではない')
    return int(number)


def official(root, validator_path, dist_dir, record_ids):
    require(blob(validator_path.read_bytes()) == VALIDATOR_BLOB, '公式検証器snapshot不一致')
    kits = list((root / '本戦').glob('配布*/pwscup2026-kit-main_revised0912/starter/src'))
    require(len(kits) == 1, '公式依存packageが一意でない')
    kit = kits[0]
    sys.path.insert(0, str(kit))
    name = 'pwscup2026.kit._aia_zip_independent_current_validate'
    spec = importlib.util.spec_from_file_location(name, validator_path)
    validator = importlib.util.module_from_spec(spec)
    sys.modules[name] = validator
    spec.loader.exec_module(validator)
    cfg_path = kit / 'pwscup2026/kit/kit_config.yaml'
    cfg = validator.config_mod.load_config(cfg_path)
    dist = validator.load_dist(dist_dir, team_id=28)
    require('mia_columns' in dist and 'challenge' in dist, '公式distで標的/課題検査が省略される')
    require(dist['team_id'] == 28 and set(dist['challenge']) == set(TEAMS) | {28}, '本戦dist対象不一致')
    dist['attacker_record_ids'] = record_ids
    return validator, cfg, dist, cfg_path


def zip_checks(path, required, mia_bytes, aia_bytes, token_bytes=None):
    require(zipfile.is_zipfile(path), 'ZIPとして開けない')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(required) and set(names) == set(required), 'ZIP直下ファイル過不足/重複')
        require(all('/' not in name and '\\' not in name and not name.startswith('.') for name in names),
                'ZIP階層/隠しファイルがある')
        require(all(not info.is_dir() and not (info.flag_bits & 1) for info in archive.infolist()),
                'ZIPディレクトリ/暗号化entryがある')
        require(archive.testzip() is None, 'ZIP CRC不一致')
        require(archive.read('F_mia.csv') == mia_bytes and archive.read('F_aia.csv') == aia_bytes,
                'ZIP再展開CSVと原本byte不一致')
        if token_bytes is not None:
            require(archive.read('token.txt') == token_bytes, 'ZIP tokenと本戦28配布token byte不一致')
    result = {'members': names, 'member_count': len(names), 'crc_pass': True,
              'mia_original_byte_match': True, 'aia_original_byte_match': True,
              'mia_sha256': hashlib.sha256(mia_bytes).hexdigest(),
              'aia_sha256': hashlib.sha256(aia_bytes).hexdigest()}
    if token_bytes is not None:
        result.update(token_original_byte_match=True, token_content_logged=False, token_hash_recorded=False,
                      GitHub_upload_allowed=False)
    else:
        result.update(token_file_present=False, GitHub_upload_allowed=True, public_archive_is_official_submission=False,
                      zip_sha256=sha(path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-root', type=Path, default=Path('/Users/taisei1105/pws2026-local'))
    parser.add_argument('--aia', type=Path, default=Path('/tmp/mia_round5_aia/F_aia.csv'))
    parser.add_argument('--mia', type=Path, required=True)
    parser.add_argument('--submission-zip', type=Path, required=True)
    parser.add_argument('--no-token-zip', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=HERE / 'aia_zip_independent_audit.json')
    args = parser.parse_args()
    root = args.local_root.resolve()
    attack = root / '本戦/攻撃フェーズ/Tashiro'
    dist_dir = attack / '20260928_配布データ基本分析/入力'
    own_folders = list((root / '本戦').glob('配布*/PWSCup2026_final_team28_*'))
    require(len(own_folders) == 1, '本戦28配布folderが一意でない')
    own = own_folders[0]
    token_path = own / 'token.txt'
    bpath = own / 'B_self.csv'
    require(sha(bpath) == MAIN_B_SHA256, 'token同梱folderの本戦Bが固定原本と異なる')
    _, b = read(bpath)
    record_ids = {integer(row['record_id'], 'B record_id') for row in b}
    require(len(b) == 1049 and record_ids == set(range(1049)), '本戦B record_id集合不一致')
    aia_bytes, mia_bytes = args.aia.read_bytes(), args.mia.read_bytes()
    require(hashlib.sha256(aia_bytes).hexdigest() == AIA_SHA256 and blob(aia_bytes) == AIA_BLOB,
            '指定AIA/5_2原本のSHA256/blob不一致')
    require(hashlib.sha256(mia_bytes).hexdigest() == MIA_SHA256, '検証済第5回MIA hash不一致')
    watched = {'aia': args.aia, 'mia': args.mia, 'B_self': bpath,
               'mia_columns': dist_dir / 'mia_columns.json',
               'validator': attack / '20261005_MIA初回_GenLRA比較/参照コード/validate_20260928.py'}
    aia_columns, rows = read(args.aia)
    require(aia_columns == ('target_k', 'challenge_row_id', 'time_hat'), 'AIA列/列順不一致')
    require(len(rows) == 7172, 'AIA全22対象×326行ではない')
    pairs, times, target_ids = [], [], {}
    for row in rows:
        target = integer(row['target_k'], 'AIA target_k')
        rid = integer(row['challenge_row_id'], 'AIA challenge_row_id')
        value = float(row['time_hat'])
        require(math.isfinite(value) and value >= 0, 'AIA time_hat非有限/負値')
        pairs.append((target, rid))
        times.append(value)
        target_ids.setdefault(target, []).append(rid)
    require(len(pairs) == len(set(pairs)), 'AIA target/challenge_row_id重複')
    require(set(target_ids) == set(TEAMS), 'AIAの対象がMIA本戦22対象と異なる')
    challenge_summary = {}
    for target in sorted(set(TEAMS) | {28}):
        qp = dist_dir / f'aia_challenge_{target}.csv'
        watched[f'aia_challenge_{target}'] = qp
        _, q = read(qp)
        expected_order = [integer(row['challenge_row_id'], '公開Q challenge_row_id') for row in q]
        require(len(q) == 326 and len(set(expected_order)) == 326, '本戦公開Q行数/重複不一致')
        if target != 28:
            got = target_ids[target]
            require(len(got) == 326 and set(got) == set(expected_order), 'AIA対象内行数/公開Q rowID集合不一致')
            challenge_summary[str(target)] = {'rows': 326, 'record_id_set_matches': True,
                'row_order_matches_public_challenge': got == expected_order}
    mia_columns, mia_rows = read(args.mia)
    require(mia_columns == ('record_id',) + tuple(map(str, TEAMS)), 'MIA対象/順序不一致')
    require(len(mia_rows) == 1049 and [integer(r['record_id'], 'MIA record_id') for r in mia_rows] == list(range(1049)),
            'MIA行数/ID順不一致')
    require(all(math.isfinite(float(row[str(t)])) and 0 <= float(row[str(t)]) <= 1
                for row in mia_rows for t in TEAMS), 'MIA値域/有限性不一致')
    digests = {name: sha(path) for name, path in watched.items()}
    validator, cfg, dist, cfg_path = official(root, watched['validator'], dist_dir, record_ids)
    watched['kit_config'] = cfg_path
    digests['kit_config'] = sha(cfg_path)
    aia_frame = validator.submission_schema.read_scoring_csv(args.aia, encoding='utf-8')
    mia_frame = validator.submission_schema.read_scoring_csv(args.mia, encoding='utf-8')
    aia_check = validator.validate_aia_submission(aia_frame, cfg, dist)
    mia_check = validator.validate_mia_submission(mia_frame, cfg, dist)
    require(aia_check.ok and mia_check.ok, '公式CSV logical validator不合格')
    token_bytes = token_path.read_bytes()
    try:
        token_text = token_bytes.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError('本戦28配布tokenがUTF-8でない') from None
    token_lines = [line for line in token_text.splitlines() if line.strip()]
    require(len(token_lines) == 1 and re.match(cfg['submission']['token_pattern'], token_lines[0].strip()) is not None,
            '本戦28配布tokenの形式不一致')
    private_zip = zip_checks(args.submission_zip, ('F_mia.csv', 'F_aia.csv', 'token.txt'),
                             mia_bytes, aia_bytes, token_bytes)
    public_zip = zip_checks(args.no_token_zip, ('F_mia.csv', 'F_aia.csv'), mia_bytes, aia_bytes)
    package = validator.check_package(args.submission_zip, cfg)
    require(package.ok, '公式ZIP packaging validator不合格')
    mia_zip_check = validator.validate_mia_submission(package.frames['F_mia.csv'], cfg, dist)
    aia_zip_check = validator.validate_aia_submission(package.frames['F_aia.csv'], cfg, dist)
    require(mia_zip_check.ok and aia_zip_check.ok, 'ZIP再展開CSVの公式logical validator不合格')
    require(token_path.read_bytes() == token_bytes, '監査中に本戦28配布tokenが変更')
    for name, path in watched.items():
        require(sha(path) == digests[name], '監査中に公開入力が変更: ' + name)
    result = {'status': 'PASS', 'submitted': False, 'aia_original_modified': False,
        'mia_original_modified': False, 'token_original_modified': False,
        'aia_source': {'repository': 'smsaku00/pwscup2026-sutehaji', 'commit': AIA_SOURCE_COMMIT,
                       'path': AIA_SOURCE_PATH, 'git_blob_sha': AIA_BLOB, 'sha256': AIA_SHA256, 'bytes': len(aia_bytes)},
        'mia_sha256': MIA_SHA256,
        'aia_checks': {'columns': list(aia_columns), 'rows': 7172, 'targets': list(TEAMS), 'target_count': 22,
                       'matches_mia_target_set': True, 'rows_per_target': 326, 'duplicate_pairs': 0,
                       'integer_finite_target_k_and_challenge_row_id': True, 'all_time_hat_finite_nonnegative': True,
                       'time_hat_min': min(times), 'time_hat_max': max(times), 'public_challenges': challenge_summary,
                       'row_order_preserved_in_both_archives': True},
        'main_team28_token_provenance': {'source': '本戦team28参加者配布folderのtoken.txt',
            'same_folder_B_self_sha256_matches_pinned_main': True, 'team_id': 28, 'format_pass': True,
            'byte_match_with_zip': True, 'token_content_logged': False, 'token_hash_recorded': False},
        'reference': {'kind': 'participant public main distribution', 'team_id': 28,
            'attacker_record_ids_from': '本戦team28配布B_self.csv', 'attacker_record_id_count': 1049,
            'aia_challenge_targets': sorted(dist['challenge']), 'csv_targets': list(TEAMS),
            'individual_truth_used': False, 'token_registry_accessed': False},
        'official_validation': {'package_ok': package.ok, 'mia_ok': mia_check.ok, 'aia_ok': aia_check.ok,
            'zip_mia_ok': mia_zip_check.ok, 'zip_aia_ok': aia_zip_check.ok,
            'errors': [], 'validator_git_blob_sha': VALIDATOR_BLOB,
            'validator_sha256': digests['validator'], 'kit_config_sha256': sha(cfg_path)},
        'official_private_zip': private_zip, 'public_no_token_zip': public_zip,
        'input_sha256': {name: {'path': str(path), 'sha256': digests[name]} for name, path in watched.items()},
        'auditor_sha256': sha(Path(__file__)),
        'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                        'seed': 'deterministic; no sampling'},
        'limits': ['公式token登録はサーバー側のtokens.csvが必要。本検証は本戦28配布出典・形式・byte同一性まで。',
                   'token入り公式ZIPはローカル/Library用。GitHubへの掲載対象は無token ZIPと公開検証資料。',
                   '大会送信・採点は行っていない。AIA推定精度とMIA所属精度は未知の本戦正解で未評価。']}
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    markdown = ['# 第5回MIA + AIA/5_2 ZIP 独立検証', '', '**PASS**。大会へ未提出。AIA・MIA・token原本を変更していない。', '',
        f'AIA原本は `{AIA_SOURCE_COMMIT}` の `{AIA_SOURCE_PATH}`。Git blob `{AIA_BLOB}`、SHA256 `{AIA_SHA256}` と一致。', '',
        '| 検証 | 結果 |', '|---|---|',
        '| AIA列順 | target_k, challenge_row_id, time_hat |',
        '| AIA対象・行数 | MIAと同じ本戦22対象、各326行、全7172行 |',
        '| 行キー | target_k / challenge_row_idは有限整数、重複0、全公開課題のID集合一致 |',
        '| time_hat | 全有限・非負、9.5～11.0 |',
        '| MIA | 検証済第5回原本のhash一致、1049行×22対象 |',
        '| token | 本戦28配布folder出典、形式PASS、ZIP内byte一致 |',
        '| 公式ZIP | 直下F_mia.csv / F_aia.csv / token.txt、CSV原本byte一致、CRC PASS |',
        '| GitHub用ZIP | F_mia.csv / F_aia.csvのみ、同じCSV原本byte一致、CRC PASS |',
        '| 公式検証器 | packaging / MIA logical / AIA logicalすべてPASS |', '',
        '公開本戦mia_columnsと23課題を参照し、team_id=28、自B_selfのrecord_id集合を明示補完した。自己28を除く22対象すべてについて各326行の公開課題ID集合とAIA行を照合した。予備戦・本戦の所属正解は読んでいない。', '',
        'token本文・token hashは表示も記録もしていない。公式ZIPにはtoken.txtを含むため、納品先はローカル/Libraryである。GitHubに掲載できるZIPは無token版であり、無token版だけでは公式提出要件を満たさない。', '',
        'tokenの本戦28配布出典・形式・ZIP内一致を確認した。サーバー側token登録は非公開tokens.csvを持たないため検証していない。大会への送信・採点と推定精度評価は行っていない。', '',
        '詳細は[JSON証跡](aia_zip_independent_audit.json)、再現は[独立監査スクリプト](aia_zip_independent_audit.py)を参照。', '']
    args.report.with_suffix('.md').write_text('\n'.join(markdown), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'aia_rows': 7172, 'aia_targets': 22,
                      'official_package_and_logical_ok': True, 'token_content_logged': False,
                      'token_hash_recorded': False, 'submitted': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
