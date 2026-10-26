#!/usr/bin/env python3
"""Export published XLSX cell values into deterministic JSON via artifact_tool.

Use: python extract_official.py [path/to/公開用結果表_本戦_20261026.xlsx]
"""
import json
import sys
from pathlib import Path
from artifact_tool import Blob, SpreadsheetFile

HERE = Path(__file__).resolve().parent
SOURCE = (Path(sys.argv[1]).expanduser() if len(sys.argv)>1 else
          HERE.parents[2]/'公式データ'/'本戦'/'結果'/'公開用結果表_本戦_20261026.xlsx')
wb = SpreadsheetFile.import_xlsx(Blob.load(str(SOURCE)))
RANGES = {
    '最終得点':'A1:O24','総括':'A1:S24',
    '加工者x攻撃者_MIA':'A1:X24','加工者x攻撃者_AIA':'A1:X24',
    '攻撃_標的別_MIA':'A1:H507','攻撃_標的別_AIA':'A1:J507',
    'AIAの差し引く水準':'A1:C24','メタ':'A1:B107',
}
result = {}
for name, cell_range in RANGES.items():
    rows = wb.worksheets.get_item(name).get_range(cell_range).values
    cols = [str(v) if v is not None else '' for v in rows[0]]
    result[name] = [dict(zip(cols, row)) for row in rows[1:] if any(v is not None for v in row)]
(HERE/'official_table_data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,default=str), encoding='utf-8')
print('Official XLSX sheets extracted:', {k:len(v) for k,v in result.items()})