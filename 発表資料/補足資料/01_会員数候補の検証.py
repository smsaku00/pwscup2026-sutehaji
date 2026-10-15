#!/usr/bin/env python3
"""第1回102候補と第2回52候補を全探索し、8行・4行の記法と照合する。

Python 3.9以降、標準ライブラリのみ。
既定ではファイルを変更せず結果を標準出力へ出す。
--report PATH を指定した場合だけ検証結果JSONを保存する。
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from math import lcm
from pathlib import Path

TEAMS = (*range(18), 22, 25, 26, 27)
HIGH_COUNTS = (
    (19, 22, 17, 11, 22, 17, 11, 20, 19, 17, 24, 20, 13, 22, 15, 17, 22, 21, 16, 12, 18, 18),
    (12, 13, 23, 22, 11, 17, 23, 14, 20, 15, 13, 18, 16, 16, 23, 20, 14, 19, 24, 21, 23, 13),
)
EXACT_MEANS = (Fraction(227, 15200), Fraction(110863, 7674480))
RAW_MEANS = ("0.014934210526315791", "0.014445669283130584")
MAX_MEMBERS = 11
MATCH_TOLERANCE = Fraction(1, 10**13)
CSV_NAMES = ("01_第1回_会員数候補102通り.csv", "01_第2回_会員数候補52通り.csv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def team_score(high_count, member_count):
    """0人の0は平均の分子用。平均の分母からも必ず除外する。"""
    if member_count == 0:
        return Fraction(0)
    return Fraction(24 - member_count, 100 * (high_count - member_count))


def exact_mean(vector, round_index):
    active = sum(value > 0 for value in vector)
    require(active > 0, "全チーム0人は今回の有限平均の候補ではない")
    return sum((team_score(n, m) for n, m in zip(HIGH_COUNTS[round_index], vector)), Fraction(0)) / active


def select(pool, number, members):
    """同じプールから重複なしでチームを選ぶ。順序は区別しない。"""
    return [{members: selected} for selected in itertools.combinations(pool, number)]


def fixed(teams, members):
    return [{members: teams}]


def flexible(team):
    return [{1: (team,)}, {2: (team,)}]


def expand_rows():
    """本文のR1-A〜HとR2-A〜Dを、そのまま選択の直積で展開する。"""
    a, b, c, e, f = (3, 6), (2, 5, 9, 15), (26, 27), (7, 11), (1, 4, 13, 16)
    g, h, i, j = (2, 6, 14, 26), (1, 10, 27), (7, 16), (12, 13)
    definitions = (
        (
            ("R1-A", (select(a, 1, 1), select(e, 1, 1), fixed((17,), 1), fixed((25,), 2), select(f, 1, 2), flexible(10))),
            ("R1-B", (fixed((12, 22), 1), select(e, 1, 1), fixed((25,), 2), select(f, 1, 2), flexible(10))),
            ("R1-C", (select(a, 1, 1), fixed(e, 1), fixed((25, 17), 2), select(c, 1, 2), flexible(10))),
            ("R1-D", (fixed((22,), 1), select(e, 1, 1), fixed((12, 26, 27), 2), select(b, 1, 2))),
            ("R1-E", (fixed((17,), 1), select(a, 1, 2), select(b, 2, 2), ({1: (7,), 2: (11,)}, {1: (11,), 2: (7,)}))),
            ("R1-F", (select(e, 1, 1), fixed((17,), 1), fixed((25,), 2), select(b, 3, 2))),
            ("R1-G", (fixed(b, 1), select(e, 1, 1), fixed((12,), 2))),
            ("R1-H", (fixed((0, 8), 1), select(e, 1, 1), select(a, 1, 2), fixed(c, 2))),
        ),
        (
            ("R2-A", (fixed((11, 17), 1), select(g, 1, 1), fixed((25,), 2), select(h, 1, 2), select(i, 1, 2))),
            ("R2-B", (fixed((11, 17), 1), select(g, 1, 1), select(h, 2, 1), fixed((25,), 2))),
            ("R2-C", (fixed((4, 11, 17), 1), select(g, 1, 1), select(j, 1, 1), fixed((25,), 2))),
            ("R2-D", (fixed((3, 11), 1), select(g, 1, 1), fixed((4, 25), 2), select(j, 1, 2))),
        ),
    )
    rounds = []
    for rows in definitions:
        expanded = {}
        for label, factors in rows:
            vectors = []
            for product in itertools.product(*factors):
                counts, seen = dict.fromkeys(TEAMS, 0), set()
                for choice in product:
                    for members, selected in choice.items():
                        for team in selected:
                            require(team not in seen, f"{label}内でTeam{team}を二重選択")
                            seen.add(team)
                            counts[team] = members
                vectors.append(tuple(counts[team] for team in TEAMS))
            require(len(vectors) == len(set(vectors)), f"{label}内に重複")
            expanded[label] = set(vectors)
        rounds.append(expanded)
    return rounds


def enumerate_all(round_index):
    """22チーム各0・1・2人、合計11人以下を二分して漏れなく列挙する。"""
    sizes = HIGH_COUNTS[round_index]
    scale = lcm(*(team_score(n, m).denominator for n in sizes for m in (1, 2)))
    weights = tuple(tuple(int(team_score(n, m) * scale) for m in (0, 1, 2)) for n in sizes)
    halves = []
    for begin in (0, 11):
        entries = []
        for state in itertools.product((0, 1, 2), repeat=11):
            total = sum(state)
            if total <= MAX_MEMBERS:
                active = sum(m > 0 for m in state)
                score_sum = sum(weights[begin + k][m] for k, m in enumerate(state))
                entries.append((state, total, active, score_sum))
        halves.append(entries)
    right = defaultdict(list)
    for state, total, active, score_sum in halves[1]:
        right[active, score_sum].append((total, state))
    answers = []
    raw = Fraction(Decimal(RAW_MEANS[round_index]))
    for active_total in range(1, MAX_MEMBERS + 1):
        # 平均の公式生値から直接、許容差1e-13内の整数和をすべて求める。
        # 正解の分数や該当チーム数を探索への入力にしない。
        low = (raw - MATCH_TOLERANCE) * scale * active_total
        high = (raw + MATCH_TOLERANCE) * scale * active_total
        goal_min = -(-low.numerator // low.denominator)
        goal_max = high.numerator // high.denominator
        for goal in range(goal_min, goal_max + 1):
            for state, total, active, score_sum in halves[0]:
                if active > active_total:
                    continue
                for other_total, other_state in right.get((active_total - active, goal - score_sum), ()):
                    if total + other_total <= MAX_MEMBERS:
                        answers.append(state + other_state)
    require(len(answers) == len(set(answers)), "全探索に重複")
    return set(answers)


def read_csv(path, round_index):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    vectors, ids = [], []
    for row in rows:
        vector = tuple(int(row[f"Team{team}"]) for team in TEAMS)
        require(all(m in (0, 1, 2) for m in vector), "CSVの会員数は0・1・2人に限定")
        require(sum(vector) <= MAX_MEMBERS, "CSVの合計が11人を超える")
        require(sum(vector) == int(row["total_members"]), "CSVの合計人数が不一致")
        require(sum(m > 0 for m in vector) == int(row["active_teams"]), "CSVの該当チーム数が不一致")
        for q in (0, 1):
            require(exact_mean(vector, q) == Fraction(row[f"round{q + 1}_mean_exact"]), "CSVの平均分数が不一致")
            require((exact_mean(vector, q) == EXACT_MEANS[q]) == (row[f"matches_round{q + 1}"] == "true"), "CSVの一致判定が不一致")
        require(exact_mean(vector, round_index) == EXACT_MEANS[round_index], "該当回の平均に一致しない候補")
        require(row["teams_one_member"] == ";".join(str(t) for t, m in zip(TEAMS, vector) if m == 1), "1人のチーム欄が不一致")
        require(row["teams_two_members"] == ";".join(str(t) for t, m in zip(TEAMS, vector) if m == 2), "2人のチーム欄が不一致")
        vectors.append(vector)
        ids.append(row["candidate_id"])
    require(len(ids) == len(set(ids)), "CSV内の候補IDが重複")
    require(len(vectors) == len(set(vectors)), "CSV内の配置が重複")
    return set(vectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="指定時だけJSON結果を保存")
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    rows_by_round = expand_rows()
    sets, summaries = [], []
    for q in (0, 1):
        csv_set = read_csv(base / CSV_NAMES[q], q)
        exhaustive = enumerate_all(q)
        row_sets = rows_by_round[q]
        expanded = set().union(*row_sets.values())
        require(sum(map(len, row_sets.values())) == len(expanded), "表の行をまたぐ重複")
        require(csv_set == exhaustive == expanded, "全探索・CSV・記法の候補集合が不一致")
        require(len(csv_set) == (102, 52)[q], "候補件数が不一致")
        for vector in expanded:
            require(exact_mean(vector, q) == EXACT_MEANS[q], "展開した候補の平均が不一致")
        raw = Fraction(Decimal(RAW_MEANS[q]))
        require(abs(raw - EXACT_MEANS[q]) < Fraction(1, 10**13), "公式生値との許容差を超える")
        summaries.append({"round": q + 1, "candidate_count": len(csv_set), "row_counts": {label: len(vs) for label, vs in row_sets.items()}, "official_raw_mean": RAW_MEANS[q], "exact_mathematical_mean": str(EXACT_MEANS[q]), "exhaustive_matches_csv": True, "expanded_rows_match_csv": True, "missing": 0, "extraneous": 0, "duplicates": 0})
        sets.append(csv_set)
    common = sets[0] & sets[1]
    require(len(common) == 1, "共通配置が1通りでない")
    counts = {str(team): m for team, m in zip(TEAMS, next(iter(common)))}
    require({t: m for t, m in counts.items() if m} == {"6": 1, "10": 2, "11": 1, "16": 2, "17": 1, "25": 2}, "共通人数が不一致")
    # 本文の5チームを固定した部分例も、すべて候補集合内にあることを確認する。
    slice_choices = ((3, 6), (2, 6, 14, 26))
    for q, choices in enumerate(slice_choices):
        for team in choices:
            one = {11, 17, team}
            two = {10, 16, 25}
            vector = tuple(1 if t in one else 2 if t in two else 0 for t in TEAMS)
            require(vector in sets[q], "固定5チームの部分例が候補集合外")
    report = {"status": "OK", "conditions": {"teams": list(TEAMS), "members_per_team": [0, 1, 2], "maximum_members_total": MAX_MEMBERS, "score_match_tolerance": str(MATCH_TOLERANCE), "zero_member_teams_excluded_from_mean": True}, "rounds": summaries, "intersection_count": len(common), "common_team_counts": counts, "common_active_teams": sum(m > 0 for m in counts.values()), "common_total_members": sum(counts.values()), "fixed_five_team_slice_checked": True, "method": "standard-library exact Fraction arithmetic; independent exhaustive meet-in-the-middle enumeration from official raw means with tolerance1e-13; compare exhaustive set, CSV, and table expansion"}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.report:
        args.report.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
