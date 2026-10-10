"""Restore the existing catalog byte-for-byte; never run an experiment."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import io
import json


def digest(content):
    return hashlib.sha256(content).hexdigest()


def restore(bundle, mode='text'):
    index = json.loads((bundle / 'catalog/catalog_index.json').read_text(encoding='utf-8'))
    if mode == 'gzip':
        content = (bundle / index['full_gzip_path']).read_bytes()
        assert digest(content) == index['full_gzip_sha256']
        result = gzip.decompress(content)
    else:
        sections = index['text_parts'] if mode == 'text' else index['parts']
        buffers, expected_start, header = [], 1, None
        for section in sections:
            assert section['row_start_1based'] == expected_start
            content = (bundle / section['path']).read_bytes()
            expected_hash = section['sha256'] if mode == 'text' else section['sha256_gzip']
            assert digest(content) == expected_hash, section['path']
            if mode == 'gzip-parts':
                content = gzip.decompress(content)
                assert digest(content) == section['sha256_decompressed_csv']
            elif len(content) >= index['text_part_max_bytes_exclusive']:
                raise ValueError('Text part exceeds its recorded limit')
            new_header = content[:index['header_bytes']]
            assert digest(new_header) == index['header_sha256']
            if header is None:
                header = new_header
                buffers.append(header)
            assert new_header == header
            assert len(list(csv.DictReader(io.StringIO(content.decode('utf-8-sig'))))) == section['rows']
            buffers.append(content[index['header_bytes']:])
            expected_start = section['row_end_1based'] + 1
        assert expected_start - 1 == index['rows']
        result = b''.join(buffers)
    assert len(result) == index['original_bytes']
    assert digest(result) == index['original_sha256']
    assert len(list(csv.DictReader(io.StringIO(result.decode('utf-8-sig'))))) == index['rows']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--mode', choices=['text', 'gzip', 'gzip-parts'], default='text')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    content = restore(args.bundle, args.mode)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            stream.write(content)
    print(json.dumps({'mode': args.mode, 'bytes': len(content), 'sha256': digest(content),
                      'written': str(args.output) if args.output else None}, ensure_ascii=False))


if __name__ == '__main__':
    main()
