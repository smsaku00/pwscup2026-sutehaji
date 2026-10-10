"""承認済み第7回MIA: 非希少Gen-LRA7、希少Gen-LRA9（同時確率案ではない）。

実行: Codex同梱Pythonで python3 -B 作成.py
再確認: python3 -B 作成.py --no-write （ファイルを更新しない）
予想30〜120秒。20対象の生スコアを再計算し、CSV保存後も独立検査する。
NumPy・pandasは同梱環境、CSV著述は同梱artifact-tool。
提出記録・配布物・AIA・token・ZIPは作成も変更もしない。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ANALYSIS = ROOT/'予備戦/攻撃フェーズ/分析'
A21 = ANALYSIS/'21_MIA全体と希少のGenLRA比較'
A22 = ANALYSIS/'22_GenLRAの7属性9属性10属性比較'
spec = importlib.util.spec_from_file_location('source21', A21/'分析.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
h = a.h
NODE_ROOT = Path('/Users/taisei1105/.cache/codex-runtimes/codex-primary-runtime/dependencies/node')
BASELINE = a.R6/'F_mia.csv'
PLAN = 'gen9_after48'
# 承認した分析22の順位との照合用。スコア順位をこの表で上書きしない。
PAIRS = {1:[1043,870], 2:[480,870], 3:[211,870], 4:[870,356], 6:[1043,238],
    7:[238,870], 8:[238,59], 9:[738,238], 10:[480,211], 12:[870,1043],
    13:[238,1043], 17:[201,238], 18:[982,238], 19:[218,480], 20:[480,218],
    22:[870,356], 23:[201,982], 24:[201,1043], 26:[356,1043], 29:[201,218]}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_data(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    header = rows[0]
    ids = np.array([int(row[0]) for row in rows[1:]])
    values = np.array([[float(x) for x in row[1:]] for row in rows[1:]])
    return header, ids, values


def save(name, value):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n'
    with (HERE/name).open('x', encoding='utf-8') as stream:
        stream.write(encoded)


def check_protected(hashes, record_paths):
    assert all(h.sha256(ROOT/p) == digest for p, digest in hashes.items()), '入力ファイルの変更を検出'
    current = {str(p.relative_to(ROOT)) for p in a.RECORDS.rglob('*') if p.is_file()}
    assert current == set(record_paths), '提出記録のファイル集合が変化'


def compute():
    start = perf_counter()
    old, comparison = read(A21/'比較結果.json'), read(A22/'比較結果.json')
    hashes = comparison['input_hashes'].copy()
    for p, digest in old['inputs_sha256'].items():
        assert p not in hashes or hashes[p] == digest
        hashes[p] = digest
    for p in [A22/'比較結果.json', A22/'検証結果.json', BASELINE,
              HERE/'作成.py', HERE/'検証.py', HERE/'CSV作成.mjs']:
        hashes[str(p.relative_to(ROOT))] = h.sha256(p)
    protected = {str(p.relative_to(ROOT)): h.sha256(p) for p in a.RECORDS.rglob('*') if p.is_file()}
    hashes.update(protected)
    check_protected(hashes, protected)
    assert {int(k): pair for k, pair in comparison['plans'][PLAN].items()} == PAIRS
    data = h.load_inputs()
    groups, history = h.audit_history(data)
    header, ids, original = csv_data(BASELINE)
    teams = [int(k) for k in header[1:]]
    assert header[0] == 'record_id' and teams == data['teams'] and teams == list(PAIRS)
    assert original.shape == (1049,20) and ids.tolist() == old['record_ids'] == comparison['record_ids']
    b = pd.read_csv(a.DATA/'PWSCup2026_prelim_team05_ステテコは恥だが役に立つ/B_self.csv', float_precision='round_trip')
    assert b.record_id.tolist() == ids.tolist()
    bg = pd.read_csv(a.DATA/'PWSCup2026_prelim_common/A_bg.csv', float_precision='round_trip')
    challenge = pd.read_csv(a.DIST/'aia_challenge_5.csv', float_precision='round_trip')
    nums, cats = a.gen.FEATURES[0][1:]
    nums9 = ['age','BMI','SBP','TG','HDL','ALT','FPG']
    private_ids = set(b.merge(challenge[nums+cats], on=nums+cats, validate='one_to_one').record_id.astype(int))
    private, rare = np.isin(ids, list(private_ids)), b.is_rare.to_numpy(bool)
    nonrare_unknown = ~private & ~rare
    assert (len(bg), int(private.sum()), int(rare.sum()), int(nonrare_unknown.sum())) == (5143,163,24,875)
    eligible = data['ids']
    known = {(k,eligible[i]) for k,i in data['old'] | data['new']}
    known |= {(r['team'],r['record_id']) for r in old['new_negative_pairs']}
    assert len(known) == 48 and set(eligible) == set(ids[rare & ~private])
    pos = {int(rid): i for i, rid in enumerate(ids)}
    answer = original.copy()
    details, rare_rows = [], []
    max_error = max_error9 = 0.
    for j,k in enumerate(teams):
        c = pd.read_csv(a.DIST/f'C_{k}.csv', float_precision='round_trip')
        computed = a.compute(bg,c,b,nums,cats)['gen']
        raw = np.array(old['raw_scores'][str(k)]['qi7_gen'])
        error = float(np.max(np.abs(computed-raw)))
        max_error = max(max_error,error)
        assert error < 1e-10
        assert np.array_equal(a.rank(computed),a.rank(raw)), '再計算で順位が変化'
        raw9 = np.array(comparison['raw_new_scores'][str(k)]['attrs9'])
        computed9 = a.compute(bg,c,b.loc[rare],nums9,cats)['gen']
        error9 = float(np.max(np.abs(computed9-raw9[rare])))
        max_error9 = max(max_error9,error9)
        assert error9 < 1e-10
        assert np.array_equal(a.rank(computed9),a.rank(raw9[rare])), '希少9属性の再計算で順位が変化'
        assert np.array_equal(original[:,j], np.array(old['raw_scores'][str(k)]['round6']))
        cap = float(original[rare,j].max())
        assert 0 < cap < 1
        if k not in (1,2):
            # 同じ生スコアは同点のまま。全非希少候補を希少上限より上へ写す。
            ascending = pd.Series(raw[nonrare_unknown]).rank(method='min').to_numpy()
            answer[nonrare_unknown,j] = cap+(1-cap)*(.1+.8*ascending/875)
        assert (answer[private,j] == 0).all()
        answer[rare,j] = 0.
        remaining = [rid for rid in eligible if (k,rid) not in known]
        order = sorted(remaining, key=lambda rid:(-raw9[pos[rid]],rid))
        pair = order[:2]
        assert pair == PAIRS[k] and len(set(raw9[[pos[rid] for rid in order]])) == len(order)
        for rank0,rid in enumerate(order):
            answer[pos[rid],j] = cap*(.9-.8*rank0/len(order))
        values = [answer[pos[rid],j] for rid in order]
        assert all(x > y > 0 for x,y in zip(values,values[1:]))
        assert (answer[private,j] == 0).all()
        assert all(answer[pos[rid],j] == 0 for kk,rid in known if kk == k)
        protected_prefix = int(sum(answer[~rare,j] > answer[rare,j].max()))
        assert protected_prefix >= 38
        if k in (1,2):
            assert np.array_equal(answer[~rare,j],original[~rare,j])
        rank11 = {rid:int(1+sum(raw[pos[x]] > raw[pos[rid]] for x in eligible)) for rid in eligible}
        rank11_9 = {rid:int(1+sum(raw9[pos[x]] > raw9[pos[rid]] for x in eligible)) for rid in eligible}
        for rid in ids[rare]:
            rid = int(rid)
            rare_rows.append(dict(team=k,record_id=rid,private=rid in private_ids,
                status='x' if rid in private_ids or (k,rid) in known else '?',
                genlra7=float(raw[pos[rid]]),genlra7_rank11=rank11.get(rid),
                genlra9=float(raw9[pos[rid]]),genlra9_rank11=rank11_9.get(rid),
                submission_rank=order.index(rid)+1 if rid in order else None,
                confidence=float(answer[pos[rid],j])))
        full_order = a.order(answer[:,j],ids,nonrare_unknown)
        details.append(dict(team=k,nonrare_method='round6_unchanged' if k in (1,2) else 'GenLRA_QI7',
            rare_method='GenLRA_attrs9',rare_order=order,rare_top2=pair,cap=cap,rare_top2_confidence=values[:2],
            nonrare_top20=[int(ids[i]) for i in full_order[:20]],
            nonrare_above_rare=protected_prefix,
            changed_nonrare_cells=int(sum(answer[~rare,j] != original[~rare,j])),
            changed_rare_cells=int(sum(answer[rare,j] != original[rare,j]))))
        elapsed = perf_counter()-start
        print(f'{j+1}/20対象を再計算・組込済み。経過{elapsed:.1f}秒、残り計算約{elapsed/(j+1)*(19-j):.1f}秒',flush=True)
    assert np.isfinite(answer).all() and ((answer >= 0)&(answer <= 1)).all()
    assert int((answer[rare] == 0).sum()) == 308 and int((answer[rare] > 0).sum()) == 172
    model_summary = [dict(model=m['model'],**m['plans'][PLAN]) for m in comparison['rare_models']]
    meta = dict(status='作成済み・未提出',baseline=str(BASELINE.relative_to(ROOT)),
        overall_method='QI7/h1/k10/rho0.98; Team1,2 nonrare unchanged',rare_plan=PLAN,
        numeric=nums,categorical=cats,rare_numeric=nums9,rare_categorical=cats,
        log1p=['TG','ALT'],bandwidth_factor=1,neighbors=10,rho=.98,
        eligible_ids=eligible,private_ids=sorted(private_ids),private_rare_ids=sorted(ids[rare&private].tolist()),
        known_negative_pairs=[dict(team=k,record_id=rid) for k,rid in sorted(known)],
        teams=details,history=history,rare_models=model_summary,
        total_changed_cells=int(sum((answer != original).ravel())),
        changed_nonrare_cells=int(sum((answer[~rare] != original[~rare]).ravel())),
        changed_rare_cells=int(sum((answer[rare] != original[rare]).ravel())),
        genlra_recomputed_values=20980,max_genlra_recompute_error=max_error,
        genlra9_recomputed_values=480,max_genlra9_recompute_error=max_error9,
        protected_record_paths=sorted(protected),official_round6=old['official_round6'])
    matrix = dict(header=header,rows=[[int(rid),*answer[i].tolist()] for i,rid in enumerate(ids)])
    check_protected(hashes,protected)
    return matrix,meta,rare_rows,hashes,perf_counter()-start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-write',action='store_true')
    args = parser.parse_args()
    start = perf_counter()
    print('予想30〜120秒。MIAのみ作成。AIA・ZIP・提出記録は変更しません。',flush=True)
    names = ['提出行列.json','計算条件.json','希少24人のスコア.json','入力SHA256.json']
    if not args.no_write:
        assert not any((HERE/n).exists() for n in names+['F_mia.csv','検証結果.json','実行記録.json']), '既存出力あり。--no-writeで再確認してください'
    matrix,meta,rare_rows,hashes,seconds = compute()
    payloads = [matrix,meta,rare_rows,hashes]
    if args.no_write:
        for name,value in zip(names,payloads):
            assert read(HERE/name) == json.loads(json.dumps(value,ensure_ascii=False)),name
    else:
        # 全データがJSON化可能なことを確認してから書く。
        for value in payloads:
            json.dumps(value,ensure_ascii=False,allow_nan=False)
        for name,value in zip(names,payloads):
            save(name,value)
        # 共有用フォルダには依存ライブラリへのリンクを置かない。
        runtime = Path(tempfile.mkdtemp(prefix='pws-mia7-',dir='/private/tmp'))
        (runtime/'node_modules').symlink_to(NODE_ROOT/'node_modules',target_is_directory=True)
        subprocess.run([str(NODE_ROOT/'bin/node'),'--input-type=module','-e',
            (HERE/'CSV作成.mjs').read_text(),str(HERE)],cwd=runtime,check=True)
    verification = subprocess.run([sys.executable,'-B',str(HERE/'検証.py')],capture_output=True,text=True)
    if verification.returncode:
        print(verification.stdout,flush=True)
        print(verification.stderr,file=sys.stderr,flush=True)
        raise RuntimeError('保存後検査に失敗しました。CSVを提出しないでください。')
    verified = json.loads(verification.stdout)
    check_protected(hashes,meta['protected_record_paths'])
    if not args.no_write:
        save('検証結果.json',verified)
        save('実行記録.json',dict(estimated_seconds=[30,120],compute_seconds=seconds,total_seconds=perf_counter()-start,
            python=sys.executable,node=str(NODE_ROOT/'bin/node'),csv_sha256=h.sha256(HERE/'F_mia.csv')))
    print(json.dumps(dict(read_only=args.no_write,changed_cells=meta['total_changed_cells'],verification=verified,
        seconds=perf_counter()-start),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
