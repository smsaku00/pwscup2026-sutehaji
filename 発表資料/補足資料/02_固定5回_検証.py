"""固定5回の選択符号と1回分の平均TPR対応表を標準ライブラリだけで再計算する。"""
from pathlib import Path
from fractions import Fraction
from itertools import combinations, product
import csv
import json

OUT = Path(__file__).resolve().parent
CODES = {
    47: "01111", 76: "00011", 90: "00101", 115: "11001", 169: "10001",
    264: "10110", 304: "01010", 383: "10010", 462: "11100", 495: "00100", 675: "01000",
}
EXPECTED_QUERIES = {
    3: [115, 169, 264, 383, 462],
    4: [47, 115, 304, 462, 675],
    5: [47, 90, 264, 462, 495],
    6: [47, 76, 264, 304, 383],
    7: [47, 76, 90, 115, 169],
}
# (Team, 会員総数m, 同点高側人数n)。各回で同じ設定を使う。
PARAMETERS = [(6, 1, 6), (10, 2, 7), (11, 1, 7), (16, 2, 11), (17, 1, 10), (25, 2, 12)]
# 第8回の保存復号回答。公式の個別秘匿正解一覧を入力したものではない。
DECODED_MEMBERS = {6: [90], 10: [304, 383], 11: [169], 16: [76, 462], 17: [115], 25: [264, 675]}
OBSERVED_MEANS = {3: "0.017203703703703704", 4: "0.010814814814814814", 5: "0.011166666666666667", 6: "0.010833333333333334", 7: "0.020148148148148148"}


def tpr(m, n, k):
    # FPR=0.01は最初の同点群を結ぶ線分上にある。
    return Fraction((24 - m) * k, 100 * m * (n - k))


def write_csv(name, fields, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    bits = {r: tuple(map(int, code)) for r, code in CODES.items()}
    assert len(bits) == len(set(bits.values())) == 11
    queries = {q: [r for r, code in CODES.items() if code[q - 3] == "1"] for q in range(3, 8)}
    assert queries == EXPECTED_QUERIES
    assert all(len(ids) == 5 for ids in queries.values())
    pair_rows = []
    sums = {}
    for a, b in combinations(bits, 2):
        summed = tuple(x + y for x, y in zip(bits[a], bits[b]))
        assert summed not in sums
        sums[summed] = (a, b)
        pair_rows.append({"record_id_1": a, "record_id_2": b, **dict(zip([f"round_{q}" for q in range(3, 8)], summed))})
    assert len(sums) == 55
    example = (1, 1, 0, 2, 0)
    assert sums[example] == (304, 383)
    mean_rows = []
    means = {}
    for counts in product(*(range(m + 1) for _, m, _ in PARAMETERS)):
        mean = sum((tpr(m, n, k) for (_, m, n), k in zip(PARAMETERS, counts)), Fraction(0)) / 6
        assert mean not in means
        means[mean] = counts
        row = {f"team_{t}_high_members": k for (t, _, _), k in zip(PARAMETERS, counts)}
        row |= {"mean_tpr_exact": str(mean), "mean_tpr_decimal": format(float(mean), ".17g")}
        mean_rows.append(row)
    assert len(means) == 216
    ordered = sorted(means)
    gap = min(b - a for a, b in zip(ordered, ordered[1:]))
    assert gap == Fraction(1, 54000)
    write_csv("02_固定5回_2人組55通り.csv", ["record_id_1", "record_id_2"] + [f"round_{q}" for q in range(3, 8)], pair_rows)
    write_csv("02_固定5回_216平均対応.csv", [f"team_{t}_high_members" for t, _, _ in PARAMETERS] + ["mean_tpr_exact", "mean_tpr_decimal"], sorted(mean_rows, key=lambda r: Fraction(r["mean_tpr_exact"])))
    actual = []
    for q in range(3, 8):
        counts = tuple(sum(bits[r][q - 3] for r in DECODED_MEMBERS[t]) for t, _, _ in PARAMETERS)
        mean = sum((tpr(m, n, k) for (_, m, n), k in zip(PARAMETERS, counts)), Fraction(0)) / 6
        assert means[mean] == counts
        error = abs(float(mean) - float(OBSERVED_MEANS[q]))
        assert error < 1e-16
        actual.append({"round": q, "counts": list(counts), "mean_exact": str(mean), "observed_mean": OBSERVED_MEANS[q], "rational_to_published_float_error": error})
    report = {
        "single_codes": len(bits), "single_code_collisions": 0,
        "pair_sums": len(sums), "pair_sum_collisions": 0,
        "selected_candidates_each_round": {q: len(ids) for q, ids in queries.items()},
        "queries": queries,
        "parameters": [{"team": t, "members": m, "high_group_size": n} for t, m, n in PARAMETERS],
        "predicted_mean_count": len(means), "predicted_mean_collisions": 0,
        "minimum_mean_gap_exact": str(gap),
        "example": {"team": 10, "high_member_counts": list(example), "pair": list(sums[example])},
        "actual_rounds_3_to_7": actual,
        "decoded_members": DECODED_MEMBERS,
        "round8_official_rare_tpr": "1.0",
        "scope": "Fixed design and published observation verification. Decoded members are from the saved round-8 answer; the official hidden individual membership table is not public.",
    }
    (OUT / "02_固定5回_検証結果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
