#!/usr/bin/env python3
"""Unpack the three official source artifacts and reproduce the post-hoc report.

Run this in the checked-out GitHub repository, with numpy and artifact_tool installed:
    python 攻撃フェーズ/Tashiro/事後検証_20261026/run_all.py

No unpublicized final-round data or competition service access is needed.
"""
from pathlib import Path
from zipfile import ZipFile
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SOURCES = Path(os.environ.get('PWS_OFFICIAL_FINAL_DATA', HERE.parents[2]/'公式データ'/'本戦'/'結果'))
for name in ('本戦正解データ_20261026.zip','攻撃提出物_20261026.zip'):
    source = SOURCES/name
    if not source.exists(): raise FileNotFoundError(source)
    with ZipFile(source) as archive:
        for item in archive.infolist():
            dst = (HERE/item.filename).resolve()
            if not dst.is_relative_to(HERE): raise ValueError('ZIP path traversal: '+item.filename)
        archive.extractall(HERE)
    print('Extracted',name)
workbook = SOURCES/'公開用結果表_本戦_20261026.xlsx'
subprocess.run([sys.executable,str(HERE/'extract_official.py'),str(workbook)],check=True)
subprocess.run([sys.executable,str(HERE/'posthoc_analysis.py')],check=True)
print('Verification and experiments complete:',HERE)