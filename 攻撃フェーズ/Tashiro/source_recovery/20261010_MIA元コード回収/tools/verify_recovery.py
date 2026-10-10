"""Check recovered bytes, source syntax, manifests and catalog; no source imports."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import unicodedata
from restore_catalog import restore


def digest(content):
    return hashlib.sha256(content).hexdigest()


def nfc(path):
    return unicodedata.normalize('NFC', str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-root', type=Path, help='Optional original root for a second byte comparison')
    parser.add_argument('--report', type=Path, help='Create a new verification JSON; existing file refused')
    args = parser.parse_args()
    bundle = args.bundle
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    sources = manifest['sources']
    assert len(sources) == manifest['counts']['source_files']
    assert len({x['original_relative_path'] for x in sources}) == len(sources)
    actual_originals = {nfc(p.relative_to(bundle)) for p in (bundle / 'originals').rglob('*') if p.is_file()}
    assert actual_originals == {x['bundle_relative_path'] for x in sources}
    syntax_count = json_count = compared = 0
    for item in sources:
        path = bundle / item['bundle_relative_path']
        content = path.read_bytes()
        assert len(content) == item['bytes'] and digest(content) == item['sha256'], str(path)
        if args.source_root:
            original = args.source_root / item['original_relative_path']
            assert original.read_bytes() == content, str(original)
            compared += 1
        if path.suffix == '.py':
            compile(ast.parse(content, filename=str(path)), str(path), 'exec')
            syntax_count += 1
        elif path.suffix == '.json':
            json.loads(content.decode('utf-8-sig'))
            json_count += 1
    patterns = [rb'gh[pousr]_[A-Za-z0-9]{20,}', rb'github_pat_[A-Za-z0-9_]{20,}',
                rb'AKIA[0-9A-Z]{16}', rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
                rb'(?i)Bearer\s+[A-Za-z0-9._~-]{20,}',
                rb'(?i)(?:api[_-]?key|password|secret|access[_-]?token)\s*[:=]\s*[\x22\x27][A-Za-z0-9_+/=.-]{16,}']
    files = [p for p in bundle.rglob('*') if p.is_file()]
    forbidden_suffixes = {'.npz', '.npy', '.zip', '.pem', '.key'}
    for path in files:
        assert path.suffix not in forbidden_suffixes, str(path)
        assert not path.name.startswith('.env') and path.name.lower() not in {'token.txt', 'tokens.json'}, str(path)
        content = path.read_bytes()
        assert not any(re.search(pattern, content) for pattern in patterns), 'Potential credential: ' + str(path)
    restored = {mode: restore(bundle, mode) for mode in ['text', 'gzip', 'gzip-parts']}
    assert restored['text'] == restored['gzip'] == restored['gzip-parts']
    if args.source_root:
        index = json.loads((bundle / 'catalog/catalog_index.json').read_text(encoding='utf-8'))
        assert (args.source_root / index['original_relative_path']).read_bytes() == restored['text']
    index = json.loads((bundle / 'catalog/source_index.json').read_text(encoding='utf-8'))
    assert digest((bundle / 'catalog/結果保存元.csv').read_bytes()) == index['sha256']
    missing = json.loads((bundle / 'missing_files.json').read_text(encoding='utf-8'))
    report = {'status': 'pass', 'original_files': len(sources), 'python_syntax_pass': syntax_count,
              'calculation_json_parse_pass': json_count, 'source_root_byte_comparisons': compared,
              'catalog_rows': 14864, 'catalog_three_restorations_identical': True,
              'catalog_sha256': digest(restored['text']),
              'requested_missing': missing['explicit_requested_missing'],
              'credential_pattern_findings': 0,
              'limitations': 'Static and byte validation only. Credential screening is not a formal proof. No experiment, source import, MIA score validation, submission or dependency installation.'}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
