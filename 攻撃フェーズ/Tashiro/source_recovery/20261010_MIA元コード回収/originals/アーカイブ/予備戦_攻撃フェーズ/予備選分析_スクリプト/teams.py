# -*- coding: utf-8 -*-
"""参加チーム名とコホート割当。公式サイトの表を取り込み、salt から独立に再計算して検証する。"""
import hashlib, unicodedata

SALT = "pwscup2026-prelim-640835c70b0c16c5bae356b2"
SALT_COMMIT = "49c5074a597afc70cc659bff95764bb9724201f5c2a8f0bb1a4615e853eeaecb"

# (登録順, 公式コホート番号, 正式名, 略称)
TEAMS = [
    (17,  1, "Ritz", "Ritz"),
    ( 5,  2, "HarunaDan@GU", "HarunaDan"),
    ( 6,  3, "ふてねこ/HuteNeko", "ふてねこ"),
    (13,  4, "アノニメイジング・PRIダーマン / The Anonymazing Prider-Man", "PRIダーマン"),
    ( 8,  5, "ステテコは恥だが役に立つ", "ステテコ"),
    (11,  6, "Trailblazer", "Trailblazer"),
    (10,  7, "神神公社 / KU nu tau tau", "神神公社"),
    (12,  8, "CyberMantra", "CyberMantra"),
    ( 4,  9, "あかぎ団＠群大/AkagiDan@GU", "あかぎ団"),
    (22, 10, "『データ提供申請が半年経っても通らないので、もう全部生成することにしました "
             "～合成データ研究者のスローライフ～』（全部生成） / JustGenerateIt", "全部生成"),
    (16, 12, "k等分の花嫁", "k等分の花嫁"),
    ( 9, 13, "ステテコVanguard", "ステテコVG"),
    (15, 17, "あれ / are", "あれ"),
    (21, 18, "コログの森団/Koroks in the forest", "コログの森団"),
    ( 2, 19, "Asakatsu2026", "Asakatsu"),
    ( 3, 20, "さいふぉん-3/siphon-3", "さいふぉん-3"),
    (18, 21, "bot035", "bot035"),
    (20, 22, "映画しろうと diabetes島のひみつ / Shiroto", "映画しろうと"),
    ( 1, 23, "宮地研.exe", "宮地研.exe"),
    (19, 24, "アウトサイダー / outsider", "アウトサイダー"),
    (14, 26, "グラブジャムン/GulabJamun", "グラブジャムン"),
    ( 7, 29, "神ぼ大/jinbodai", "神ぼ大"),
]
SELF = 5

# CodaBench のユーザ名 → コホート番号（リーダーボードはチーム名でなくユーザ名で並ぶ）
CODABENCH_USER = {
    "gghatano":     20,   # 波多野卓磨 = さいふぉん-3
    "harunaDan@GU":  2,   # HarunaDan@GU
    "zhiyan":        5,   # 張知妍 = ステテコは恥だが役に立つ（自チーム）
}

# 予備戦・攻撃フェーズ リーダーボード（2026-08-27 18:27 JST 時点、27参加中2チーム提出）
# (ユーザ名, 得点, MIA全体, MIA希少, AIA全体, AIA希少)
LEADERBOARD = [
    ("gghatano",     0.1949, 0.0005, 0.25, 0.0918, 0.0414),
    ("harunaDan@GU", 0.0935, 0.0217, 0.00, 0.0718, 0.0254),
]
# ここから読めた較正値（採点コードで裏を取った版）:
#   得点 = mia_power + aia_power
#     mia_power = mean_20(score_mia)  , score_mia = max(tpr_all, tpr_rare)   [scoring_io.py:177]
#     aia_power = mean_20(score_aia)  , score_aia = max(R_all, R_rare)       [attack_score.py:135]
#     内訳1・2・3・4 は max を取る前の nanmean（希少側は陽性0の標的が NaN で分母から落ちる）
#
#   ★ D（希少陽性を持つ標的数）は確定できない。aia_power >= aia_all しか言えないため:
#       mia_power = 0.1949 - aia_power <= 0.1949 - 0.0918 = 0.1031
#       希少の寄与 Σ = 20 * mia_power - (tpr_all の寄与≈0.01) <= 2.06
#       D = Σ / 0.25 <= 8.2
#     共有ブロックの希少は1〜2人なので tpr_rare ∈ {0, 0.5, 1.0}、よって Σ ∈ {0.5,1.0,1.5,2.0}
#       → D ∈ {2, 4, 6, 8}。構造推定（自コホートで 7〜9）と合うのは D=8、次点 D=6。
#     なお D は攻撃者ごとの値（自分のコホートの希少がどの相手と共有されているかで決まる）。
#
#   AIA の床 ≈ 0.07〜0.09  … 両者とも MIA全体が偶然水準＝漏洩を見つけていないのに この値。
#                            clip 由来の下駄（E[R]=0.064/標的）の実測。
NAME  = {k: full  for _, k, full, _ in TEAMS}
SHORT = {k: short for _, k, _, short in TEAMS}

def label(k, width=None):
    s = SHORT.get(k, "?")
    if k == SELF: s += "（自）"
    return f"{k} {s}"

def verify(n_cohorts=30):
    """§3.2 の手順を再現: key = SHA-256(NFC(チーム名) ‖ salt), 番号 = key mod N、
    衝突したら空きが見つかるまで +1（登録順に処理）。"""
    assert hashlib.sha256(SALT.encode()).hexdigest() == SALT_COMMIT, "salt がコミットメントと不一致"
    taken, got = {}, {}
    for order, k, full, _ in sorted(TEAMS):
        key = hashlib.sha256(unicodedata.normalize("NFC", full).encode("utf-8") + SALT.encode("utf-8")).hexdigest()
        n = int(key, 16) % n_cohorts
        while n in taken:
            n = (n + 1) % n_cohorts
        taken[n] = full; got[k] = n
    return got

if __name__ == "__main__":
    print(f"salt コミットメント: 一致")
    got = verify()
    ok = sum(1 for _, k, _, _ in TEAMS if got[k] == k)
    print(f"再計算が公式表と一致: {ok}/{len(TEAMS)} チーム\n")
    for _, k, full, short in TEAMS:
        mark = "OK " if got[k] == k else f"NG({got[k]})"
        print(f"  {mark} コホート{k:3d}  {short:14s}  {full[:44]}")
