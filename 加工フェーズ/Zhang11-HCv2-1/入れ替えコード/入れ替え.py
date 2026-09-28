"""元のZhang11から、採用したTG交換1回だけを再現・検証する。

共有用の実行器。探索時の交換処理は「当時の交換処理.py」に収録。
依存: Python 3.12、pandas 2.3.3（検証時）。
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from 当時の交換処理 import replay

HERE = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def run(args):
    manifest = json.loads((HERE / '交換内容.json').read_text(encoding='utf-8'))
    source = args.input.read_bytes()
    if sha256(source) != manifest['input_sha256']:
        raise ValueError('入力が実験時の元Zhang11と一致しません。同梱の「元のZhang11.csv」を使ってください。')
    if args.output.resolve() in {args.input.resolve(), args.reference.resolve()}:
        raise ValueError('出力先に元データ・提出済みCを指定することはできません。')
    if not args.check_only and args.output.exists():
        raise ValueError('出力ファイルが既にあります。--output で新しい保存先を指定してください。')

    # 当時の replay() で交換し、当時と同じ設定でCSVを書き出す。
    changed = replay(args.input, manifest['operations'])
    buffer = io.BytesIO()
    changed.to_csv(buffer, index=False, encoding='utf-8-sig', float_format='%.17g')
    result = buffer.getvalue()

    # ハッシュだけでなく、変更したセルがTGの2セルのみであることも確認する。
    before = list(csv.reader(io.StringIO(source.decode('utf-8-sig'))))
    after = list(csv.reader(io.StringIO(result.decode('utf-8-sig'))))
    if len(before) != len(after) or before[0] != after[0]:
        raise ValueError('行数または列が変わっています。')
    differences = []
    for row, (old, new) in enumerate(zip(before[1:], after[1:]), start=1):
        if len(old) != len(new):
            raise ValueError('列数が変わっています。')
        differences.extend((row, before[0][col]) for col, (a, b) in enumerate(zip(old, new)) if a != b)
    if differences != [(316, 'TG'), (1049, 'TG')]:
        raise ValueError(f'予定外の変更です: {differences}')
    digest = sha256(result)
    if digest != manifest['output_sha256']:
        raise ValueError('生成結果が提出済みCのハッシュと一致しません。Python・pandasの版を確認してください。')

    reference = args.reference.read_bytes()
    if result != reference:
        raise ValueError('指定された提出済みCと、生成結果が一致しません。')
    if not args.check_only:
        # 既存ファイルを上書きしない。出力は全検証が成功してから行う。
        with args.output.open('xb') as stream:
            stream.write(result)
        print(f'保存先: {args.output}')
    print('OK: TGの2セルだけを交換しました。提出済みCとバイト単位で完全一致します。')
    print(f'SHA-256: {digest}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / '元のZhang11.csv')
    parser.add_argument('--output', type=Path, default=HERE / 'C_reproduced.csv')
    parser.add_argument('--reference', type=Path, default=HERE.parent / 'C.csv')
    parser.add_argument('--check-only', action='store_true', help='生成・照合をメモリ上で行い、ファイルを保存しない')
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'エラー: {error}\n')


if __name__ == '__main__':
    main()
