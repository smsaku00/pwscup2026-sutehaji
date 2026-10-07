"""第3回MIAのみを作成する。新しい出力先のみ許可し、入力hashを固定する。

既存の検討コードのcore/calibrateだけを再利用する。本戦正解、AIA回答、tokenは読まない。
公式提出は行わない。実行: ルート .venv/bin/python -B 作成.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ACTIVE = [6, 10, 11, 16, 17, 25]
COUNTS = [1, 2, 1, 2, 1, 2]
TOTAL_HIGH = [6, 7, 7, 11, 10, 12]
QUERIES = [
    [115, 169, 264, 383, 462],
    [47, 115, 304, 462, 675],
    [47, 90, 264, 462, 495],
    [47, 76, 264, 304, 383],
    [47, 76, 90, 115, 169],
]
CANDIDATES = [47, 76, 90, 115, 169, 264, 304, 383, 462, 495, 675]
NEG_RARE = [126, 136, 271, 273, 317, 440, 480, 625, 637, 643, 668, 814, 841]
EXPECTED_TEAMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 25, 26, 27]
REFERENCE = HERE.parent / '20261007_MIA全体改善の検討'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def pins_check():
    pins = json.loads((HERE / 'input_pins.json').read_text())['input_sha256']
    for relative, expected in pins.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'入力または参照コードが固定hashと不一致: {relative}')
    return pins


def load_references():
    # このフォルダ以外の生成処理は呼ばない。参照コードのmainは実行しない。
    spec = importlib.util.spec_from_file_location('round3_reference_common', REFERENCE / '分析.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    previous = sys.modules.get('分析')
    sys.modules['分析'] = common
    try:
        spec = importlib.util.spec_from_file_location('round3_reference_calibration', REFERENCE / '非会員を使う順位較正の検討.py')
        calibration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(calibration)
    finally:
        if previous is None:
            del sys.modules['分析']
        else:
            sys.modules['分析'] = previous
    return common, calibration


def rare_design_tables():
    words = {i: tuple(int(i in q) for q in QUERIES) for i in CANDIDATES}
    pair_words = {p: tuple(sum(words[i][k] for i in p) for k in range(5)) for p in combinations(CANDIDATES, 2)}
    if len(set(words.values())) != 11 or len(set(pair_words.values())) != 55:
        raise ValueError('固定5bitの符号が衝突')
    encoding = []
    for h in product(*[range(n + 1) for n in COUNTS]):
        per_team = [Fraction((24-n)*v, 100*n*(m-v)) for n, m, v in zip(COUNTS, TOTAL_HIGH, h)]
        mean = sum(per_team, Fraction()) / 6
        encoding.append({'counts': list(h), 'mean_exact': str(mean), 'mean': float(mean)})
    means = sorted(Fraction(row['mean_exact']) for row in encoding)
    if len(set(means)) != 216 or min(b-a for a,b in zip(means,means[1:])) != Fraction(1,54000):
        raise ValueError('平均点の符号が衝突または間隔不一致')
    return words, pair_words, encoding


def build(output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(f'既存出力先を上書きしない: {output}')
    pins = pins_check()
    common, calibration = load_references()
    b, a, _, private, rare, cs, targets, self_team = common.phase_inputs(ROOT, 'main')
    declared = common.read_json(targets / 'mia_columns.json')
    teams = [int(t) for t in declared['target_columns'] if int(t) != self_team]
    if teams != EXPECTED_TEAMS or set(cs) != set(teams) or self_team != 28:
        raise ValueError('攻撃対象列が合意した22チームと不一致')
    ids = b.record_id.to_numpy(int)
    if not np.array_equal(ids, np.arange(1049)):
        raise ValueError('本番B_selfのID順が想定と異なる')
    normal = ~private & ~rare
    if (len(b), len(a), int(normal.sum()), int((private & ~rare).sum())) != (1049,5143,875,150):
        raise ValueError('B/A/通常候補/既知非会員の人数不一致')
    if sorted(ids[rare & ~private]) != CANDIDATES or sorted(ids[rare & private]) != NEG_RARE:
        raise ValueError('希少候補/既知非会員のID不一致')
    words, pair_words, encoding = rare_design_tables()
    qs = {t: common.read(targets / f'aia_challenge_{t}.csv') for t in sorted(teams + [28])}
    raw_columns, calibrated_columns, bank_counts = [], [], []
    for team in teams:
        controls = pd.concat([qs[t][list(common.QI7)] for t in sorted(qs) if t != team], ignore_index=True)
        if len(controls) != 7172 or controls.duplicated(list(common.QI7)).any():
            raise ValueError(f'Team{team}: 較正用非会員の人数/重複が不一致')
        if pd.MultiIndex.from_frame(b.loc[normal,list(common.QI7)]).isin(pd.MultiIndex.from_frame(controls)).any():
            raise ValueError(f'Team{team}: 未知通常候補が非会員bankと一致')
        query = pd.concat([b[list(common.QI7)], controls], ignore_index=True)
        combined = common.core(a, query, cs[team], common.NUM7, (10,))[10]
        raw = combined[:1049]
        null = combined[1049:]
        cal = calibration.calibrate(raw, b, null, controls, ['prefecture','sex'], 500.)
        raw_columns.append(raw)
        calibrated_columns.append(cal)
        bank_counts.append({'team': team, 'nonmember_rows': len(controls), 'excluded_challenge': team,
                            'included_self_challenge': 28, 'nonmember_unique_QI7': len(controls)})
        print(f'第3回 全体計算 {len(raw_columns)}/22: Team{team}', flush=True)
    raw_all = np.column_stack(raw_columns)
    calibrated_all = np.column_stack(calibrated_columns)
    ranks = np.column_stack([rankdata(col[normal], method='average') / 875 for col in calibrated_all.T])
    other_median = np.column_stack([np.median(np.delete(ranks,j,axis=1),axis=1) for j in range(22)])
    corrected = ranks - .1 * other_median
    final_rank = np.column_stack([rankdata(col,method='min') for col in corrected.T])
    final = np.zeros((1049,22), dtype=float)
    final[normal] = .6 + .3 * final_rank / 875
    high_ids = {}
    for team, m in zip(ACTIVE,TOTAL_HIGH):
        selected = sorted(QUERIES[0] + NEG_RARE[:m-5])
        high_ids[str(team)] = selected
        col = teams.index(team)
        final[rare,col] = .1
        final[np.isin(ids,selected),col] = .4
    # 他16対象の希少24行、既知通常非会員150行はゼロ初期値を維持。
    if not np.isfinite(final).all() or final.min() < 0 or final.max() > 1:
        raise ValueError('提出値が有限[0,1]でない')
    pins_check()
    for filename, digest in common.INPUT_HASHES.items():
        rel = str(Path(filename).relative_to(ROOT))
        if pins.get(rel) != digest:
            raise ValueError(f'固定manifestにない入力を使用: {rel}')
    output.mkdir(parents=True)
    frame = pd.DataFrame(final,columns=[str(t) for t in teams])
    frame.insert(0,'record_id',ids)
    frame.to_csv(output/'F_mia.csv',index=False,float_format='%.17g',lineterminator='\n')
    # 読み戻しをCSVの17桁精度で照合。行位置ではなくIDを鍵に照合する。
    back = pd.read_csv(output/'F_mia.csv',float_precision='round_trip')
    if list(back.columns) != list(frame.columns) or not np.array_equal(back.record_id,ids) or not np.array_equal(back.iloc[:,1:].to_numpy(),final):
        raise ValueError('CSVの読み戻しが計算値と不一致')
    trace_rows = []
    for j,team in enumerate(teams):
        for k,row in enumerate(np.flatnonzero(normal)):
            trace_rows.append({'team':team,'record_id':int(ids[row]),'raw7':float(raw_all[row,j]),
                              'calibrated':float(calibrated_all[row,j]),'ordinary_rank':float(ranks[k,j]),
                              'cross_corrected':float(corrected[k,j]),'submit_rank':float(final_rank[k,j])})
    pd.DataFrame(trace_rows).to_csv(output/'通常者計算過程.csv',index=False,float_format='%.17g')
    np.savez_compressed(output/'計算値.npz', teams=np.array(teams), record_ids=ids,
                        normal_ids=ids[normal], raw7=raw_all, calibrated=calibrated_all,
                        ordinary_rank=ranks, other_median=other_median, cross_corrected=corrected, submit_rank=final_rank)
    setting = {'round':3,'status':'作成済み・独立監査待ち・未提出','own_team':28,'teams':teams,
               'scope':'MIAのみ。AIA/token/ZIP/大会提出なし。','num_rows':1049,'num_targets':22,
               'ordinary_candidates':875,'nonrare_known_negative':150,'rare_records':24,
               'rare_candidates':CANDIDATES,'known_negative_rare':NEG_RARE,
               'question_teams':ACTIVE,'rare_member_counts':dict(zip(map(str,ACTIVE),COUNTS)),
               'zero_rare_teams':[t for t in teams if t not in ACTIVE],
               'fixed_queries':{str(i+3):q for i,q in enumerate(QUERIES)},
               'total_high':dict(zip(map(str,ACTIVE),TOTAL_HIGH)),
               'padding_count':dict(zip(map(str,ACTIVE),[m-5 for m in TOTAL_HIGH])),
               'round3_high_record_ids':high_ids,'rare_high':.4,'rare_low':.1,'zero_rare_value':0,
               'overall_method':'Gen-LRA7 + 県性別非会員較正500 + 自対象除外中央値補正0.1',
               'features':list(common.QI7),'k':10,'rho':.98,'shrink':500,'cross_alpha':.1,
               'cdf_tie_rule':'.999 * empirical_CDF + .001 * average_raw_rank / 1049',
               'ordinary_rank_rule':'average_rank / 875 before cross; min_rank after cross; true ties preserved',
               'overall_mapping':'.6 + .3 * min_rank / 875',
               'rare_mean_key':'mia.tpr_rare','mean_minimum_gap_exact':'1/54000',
               'main_truth_used':False,'submitted':False,
               'count_source':'20261006_MIA第2回結果と人数導出/公開用/MIA/20261006_MIA第2回_seo02と希少同点/採点結果/MIA結果.json'}
    json_write(output/'設定.json',setting)
    pd.DataFrame(bank_counts).to_csv(output/'非会員較正の対象.csv',index=False)
    pd.DataFrame([{'record_id':i,'bits':''.join(map(str,words[i]))} for i in CANDIDATES]).to_csv(output/'固定5bit符号.csv',index=False)
    pd.DataFrame([{'record_id_1':p[0],'record_id_2':p[1],'count_signature':','.join(map(str,w))} for p,w in pair_words.items()]).to_csv(output/'2人組55通り.csv',index=False)
    json_write(output/'平均点対応表.json',encoding)
    json_write(output/'inputs_manifest.json',{'input_sha256':pins,'all_pinned_inputs_unchanged':True})
    json_write(output/'実行記録.json',{'started_jst':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                'elapsed_seconds':time.perf_counter()-started,'python':sys.version,'executable':sys.executable,
                'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
                'builder_sha256':sha(__file__),'F_mia_sha256':sha(output/'F_mia.csv'),
                'references':{str(p.relative_to(ROOT)):sha(p) for p in [REFERENCE/'分析.py',REFERENCE/'非会員を使う順位較正の検討.py']},
                'main_truth_used':False,'submitted':False})
    pins_check()
    print(json.dumps({'candidate':str(output/'F_mia.csv'),'sha256':sha(output/'F_mia.csv'),'rows':1049,'targets':22},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'候補')
    build(parser.parse_args().output_dir.resolve())
