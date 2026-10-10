"""47背景の再実行入口。過去の実行器を保存し、必要なら新しい終了時刻を指定。"""
from __future__ import annotations
import argparse,sys
from datetime import datetime
from zoneinfo import ZoneInfo
import 共通 as u

def main():
 parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--until',help='例: 2026-10-09T08:40:00+09:00。省略すると全47条件を実行。')
 options,remaining=parser.parse_known_args();zone=ZoneInfo('Asia/Tokyo')
 until=datetime.fromisoformat(options.until) if options.until else datetime.max.replace(tzinfo=zone)
 if until.tzinfo is None:raise ValueError('--untilにはタイムゾーンを含める。')
 if until<=datetime.now(zone):raise ValueError('--untilは未来の時刻を指定する。')
 original=u.module('rerun_background_preserved',u.HERE/'背景変更で学習を確認.py')
 original.DEADLINE=until;u.base.INPUT_HASHES[str(__file__)]=u.sha(__file__)
 sys.argv=[sys.argv[0],*remaining]
 original.main()

if __name__=='__main__':main()
