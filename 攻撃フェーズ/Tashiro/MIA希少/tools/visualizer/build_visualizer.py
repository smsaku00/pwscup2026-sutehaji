"""Build a reproducible, offline viewer with lazy inline GZIP certificates."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
from artifact_layout import artifact_path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / 'analysis' / 'full_depth4_20261007'
DEST = ROOT / 'docs' / 'visualizer'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proof', type=Path, default=PROOF)
    parser.add_argument('--output', type=Path, default=DEST / 'MIA_4query_tree.html')
    args = parser.parse_args()
    proof = args.proof
    manifest = read(proof / 'manifest.json')
    audit = read(proof / 'independent_full_audit.json')
    if manifest['q3_query'] != [590, 436, 1547, 440, 1095, 433]:
        raise ValueError('unexpected q3 question')
    data = {name: read(proof / filename) for name, filename in {
        'manifest': 'manifest.json', 'encoding': 'universal_score_encoding.json',
        'padding': 'padding_rule.json', 'summary': 'summary.json',
        'example': 'example_witness_code30.json',
    }.items()}
    data['q3Counts'] = {str(b['q3_code']): b['states'] for b in audit['branches']}
    if len(data['q3Counts']) != 209 or sum(data['q3Counts'].values()) != 2494800:
        raise ValueError('incomplete root outcomes')
    data['certificates'] = {}
    sources = []
    compressed = expanded = 0
    for orbit in manifest['orbits']:
        rep = orbit['representative']
        path = proof / 'orbits' / str(rep) / 'certificate.json.gz'
        packed = path.read_bytes()
        unpacked = gzip.decompress(packed)
        cert = json.loads(unpacked)
        if cert['q3_code'] != rep or cert['depth'] != 3:
            raise ValueError('wrong representative certificate')
        data['certificates'][str(rep)] = base64.b64encode(packed).decode('ascii')
        sources.append({'path': str(path.relative_to(proof)), 'bytes': len(packed),
                        'sha256': hashlib.sha256(packed).hexdigest()})
        compressed += len(packed)
        expanded += len(unpacked)
    for filename in ['manifest.json', 'independent_full_audit.json', 'summary.json',
                     'padding_rule.json', 'universal_score_encoding.json', 'example_witness_code30.json']:
        packed = (proof / filename).read_bytes()
        sources.append({'path': filename, 'bytes': len(packed), 'sha256': hashlib.sha256(packed).hexdigest()})
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    template = (HERE / 'viewer.template.html').read_text(encoding='utf-8')
    core = (HERE / 'tree_core.js').read_text(encoding='utf-8')
    html = template.replace('/*__CORE__*/', core).replace('/*__DATA__*/', payload)
    if '/*__CORE__*/' in html or '/*__DATA__*/' in html:
        raise ValueError('unreplaced placeholder')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding='utf-8')
    stats = {'format': 'single HTML with inline base64 GZIP; selected orbit inflated on demand',
             'certificate_count': len(manifest['orbits']), 'q3_branches': 209,
             'states': 2494800, 'certificate_gzip_bytes': compressed,
             'certificate_expanded_json_bytes': expanded,
             'certificate_base64_bytes': sum(len(v) for v in data['certificates'].values()),
             'html_bytes': args.output.stat().st_size,
             'html_gzip_bytes': len(gzip.compress(html.encode(), mtime=0)),
             'source_files': sources}
    report_path = artifact_path('size_report.json') if args.output.resolve() == (DEST / 'MIA_4query_tree.html').resolve() else args.output.parent / 'size_report.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in stats.items() if k != 'source_files'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
