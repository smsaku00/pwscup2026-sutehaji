"""2回目の詳細結果HTMLから希少人数を読み取る。ファイル変更・提出はしない。

実行: .venv/bin/python -B <このファイル> /path/to/detailed_results.html
丸めた表示値や最も近い値への強制割当は使用しない。一致しなければ停止する。
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

HERE=Path(__file__).resolve().parent


def identify(value, cases, tolerance=1e-13):
    if not isinstance(value,(int,float)) or not 0 <= value <= 1:
        raise ValueError('有効な希少点数ではありません')
    hits=[case for case in cases if abs(case['round2_score']-value)<=tolerance]
    if len(hits)!=1:
        raise ValueError(f'一致する配置が{len(hits)}件です。提出したCSV・提出回・採点条件を確認してください。推測で次へ進みません。')
    return hits[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('detailed_results',type=Path)
    args=parser.parse_args()
    first_report_sha='1d06e508ae0e7fd8119bb6da34b205df8e1cd4b0e4aeaa621dba6cd47bcf6687'
    if hashlib.sha256(args.detailed_results.read_bytes()).hexdigest()==first_report_sha:
        raise ValueError('これは保存済みの1回目の結果です。2回目の詳細結果を指定してください。')
    out=HERE/'結果'
    hashes=json.loads((out/'出力SHA256.json').read_text())
    for name,digest in hashes.items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'保存ファイルが変わっています: {name}')
    report_text=args.detailed_results.read_text(encoding='utf-8')
    matches=re.findall(r'<pre[^>]*>(.*?)</pre>',report_text,re.S)
    objects=[]
    for block in matches:
        try:
            obj=json.loads(html.unescape(block))
        except json.JSONDecodeError:
            continue
        if isinstance(obj,dict) and isinstance(obj.get('mia'),dict) and 'tpr_rare' in obj['mia']:
            objects.append(obj)
    if len(objects)!=1:
        raise ValueError('詳細結果のmia.tpr_rareを一意に読み取れません')
    cases=json.loads((out/'2回目の点数と人数102通り.json').read_text())
    answer=identify(objects[0]['mia']['tpr_rare'],cases)
    print(json.dumps({'observed_rare_score':objects[0]['mia']['tpr_rare'],**answer,
                     'note':'この第2回F_miaを提出した結果であることが前提。次回CSVは自動作成しない。'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
