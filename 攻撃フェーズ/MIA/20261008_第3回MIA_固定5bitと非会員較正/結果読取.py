"""第3〜7回の固定5bit質問の公式希少平均点を読む。ファイルは書かない。

例: .venv/bin/python -B 結果読取.py --round-means 3=<第3回の生のmia.tpr_rare>
第3〜7回の5回分を回番号付きで渡すと個人所属を特定する。
入力はこのフォルダの固定質問・paddingと一致する実提出に限る。
回番号自体の誤記は常に検出できないため、採点原本の回との照合が必要。
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from itertools import combinations, product

TEAMS = (6,10,11,16,17,25)
N = (1,2,1,2,1,2)
M = (6,7,7,11,10,12)
IDS = (47,76,90,115,169,264,304,383,462,495,675)
QUESTIONS = (
    (115,169,264,383,462),
    (47,115,304,462,675),
    (47,90,264,462,495),
    (47,76,264,304,383),
    (47,76,90,115,169),
)
TOLERANCE = Fraction(1,10**12)


def labeled_values(entries):
    """回番号で並べ替える。第3回から連続した入力だけを受け付ける。"""
    by_round = {}
    for entry in entries:
        label, separator, value = entry.partition('=')
        if not separator or not label.isascii() or not label.isdecimal():
            raise ValueError('回番号付き入力は3=<生の平均点>の形式が必要です')
        round_number = int(label)
        if not 3 <= round_number <= 7:
            raise ValueError('回番号は第3〜7回が必要です')
        if round_number in by_round:
            raise ValueError(f'第{round_number}回の入力が重複しています')
        if not value.strip():
            raise ValueError(f'第{round_number}回の平均点が空です')
        by_round[round_number] = value
    expected = list(range(3,3+len(by_round)))
    if not by_round or sorted(by_round) != expected:
        raise ValueError('第3回から途中の回を欠かさず入力してください')
    return [by_round[k] for k in expected]


def decode_mean(value: str):
    try:
        point = Fraction(value)
    except (ValueError,ZeroDivisionError) as exc:
        raise ValueError('平均点は公式詳細結果の有限な数値を入力してください') from exc
    if not 0 <= point <= 1:
        raise ValueError('平均点は[0,1]の範囲が必要です')
    matches = []
    for counts in product(*[range(n+1) for n in N]):
        mean = sum((Fraction((24-n)*h,100*n*(m-h)) for n,m,h in zip(N,M,counts)),Fraction())/6
        if abs(mean-point) <= TOLERANCE:
            matches.append((counts,mean))
    if len(matches) != 1:
        raise ValueError('平均点が符号表に一意一致しません。実提出の設定・生の結果値・桁数を確認してください')
    return matches[0]


def compatible_assignment(groups):
    order = sorted(range(6),key=lambda i:len(groups[i]))
    def visit(k,used):
        if k == 6:
            return True
        for pair in groups[order[k]]:
            if not used.intersection(pair) and visit(k+1,used.union(pair)):
                return True
        return False
    return visit(0,set())


def interpret(values):
    if not 1 <= len(values) <= 5:
        raise ValueError('平均点は第3回から順に1〜5個が必要です')
    decoded = [decode_mean(v) for v in values]
    groups,remaining = [],{}
    for j,(team,n) in enumerate(zip(TEAMS,N)):
        valid = []
        for selected in combinations(IDS,n):
            if all(sum(i in QUESTIONS[k] for i in selected) == decoded[k][0][j] for k in range(len(values))):
                valid.append(selected)
        if not valid:
            raise ValueError(f'Team{team}: 履歴と一致する候補がありません')
        groups.append(valid)
        remaining[str(team)] = {'member_count':n,'possibilities':len(valid),'candidate_assignments':[list(p) for p in valid]}
    if not compatible_assignment(groups):
        raise ValueError('6チームの候補に重複禁止を満たす配置がありません。履歴または提出設定を確認してください')
    complete = all(len(v)==1 for v in groups)
    answer = {'read_rounds':list(range(3,3+len(values))),
              'observations':[{'round':k+3,'mean_input':values[k],'mean_exact':str(mean),
                               'selected_member_counts':dict(zip(map(str,TEAMS),counts))} for k,(counts,mean) in enumerate(decoded)],
              'remaining':remaining,'all_identified':complete,
              'other_16_teams_rare_members':0,
              'writes_files':False,'submitted':False}
    if complete:
        assignments = {str(t):list(g[0]) for t,g in zip(TEAMS,groups)}
        used = {i for group in assignments.values() for i in group}
        if len(used)!=9:
            raise ValueError('9人の所属数と不一致')
        answer['identified_members'] = assignments
        answer['not_in_any_target'] = sorted(set(IDS)-used)
    if len(values)<5:
        answer['next_round'] = 3+len(values)
        answer['next_fixed_question'] = list(QUESTIONS[len(values)])
        answer['next_padding_count'] = dict(zip(map(str,TEAMS),[m-5 for m in M]))
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--round-means',nargs='+',metavar='回番号=平均点',
                        help='推奨: 3=<生の点数> 4=<生の点数> …。引数順は問いません')
    inputs.add_argument('--means',nargs='+',metavar='平均点',
                        help='互換用: 第3回から順に入力。回番号付き入力を推奨します')
    args=parser.parse_args()
    try:
        values=labeled_values(args.round_means) if args.round_means is not None else args.means
        print(json.dumps(interpret(values),ensure_ascii=False,indent=2,allow_nan=False))
    except ValueError as exc:
        parser.error(str(exc))
