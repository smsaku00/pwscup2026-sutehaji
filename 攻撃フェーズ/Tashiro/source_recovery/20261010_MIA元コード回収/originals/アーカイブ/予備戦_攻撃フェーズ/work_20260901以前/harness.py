"""採点ハーネス: C(DataFrame) を渡すと公式採点器と同じ U と内訳を返す。

さらに、手元では採点できない「匿名性」の代理指標（参照MIA相当・参照AIA相当）も
自前の真値で近似して返す。使い方:

    from harness import Bench
    bench = Bench()
    bench.report(C)          # U と 4 観点 + プライバシー代理指標を印字
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

KIT = Path(__file__).resolve().parent.parent / "スターターキット/初版"
sys.path.insert(0, str(KIT / "starter" / "src"))

from pwscup2026.attack import aia as aia_mod  # noqa: E402
from pwscup2026.attack import mia as mia_mod  # noqa: E402
from pwscup2026.common import schema, submission_schema  # noqa: E402
from pwscup2026.kit.codabench.scoring_io import _canonicalize_c  # noqa: E402
from pwscup2026.rare import outlier as outlier_mod  # noqa: E402
from pwscup2026.scoring import attack_score  # noqa: E402
from pwscup2026.scoring.result import ReportLevel  # noqa: E402
from pwscup2026.scoring.utility import UtilityReference, score_utility  # noqa: E402

DIST = KIT / "participant_data"
AIA_PARAMS = {
    "qi_continuous": ["age", "SBP", "TG", "HDL", "ALT"],
    "block_keys": ["sex", "prefecture"],
    "k_nn_onset": 5,
}


class Bench:
    def __init__(self, dist: Path = DIST):
        self.dist = Path(dist)
        self.cfg = yaml.safe_load((self.dist / "scoring_config.yaml").read_text())
        self.cfg.setdefault("aia", {}).update(AIA_PARAMS)
        ref_d = json.loads((self.dist / "utility_ref.json").read_text())
        b_self = pd.read_csv(self.dist / "B_self.csv")
        self.is_rare = b_self["is_rare"].to_numpy().astype(bool)
        self.B = b_self.drop(columns=["is_rare"])
        self.ref = UtilityReference(
            mu=np.asarray(ref_d["mu"], dtype=float),
            sigma=np.asarray(ref_d["sigma"], dtype=float),
            band=dict(ref_d["band"]),
            b_is_rare=self.is_rare,
            km_signal_D=ref_d.get("km_signal_D"),
            km_floor=ref_d.get("km_floor"),
        )
        self.A_bg = pd.read_csv(self.dist / "A_bg.csv")

    # ------------------------------------------------------------------ #
    # 有用性（公式採点器と同一）
    # ------------------------------------------------------------------ #
    def score(self, C: pd.DataFrame) -> dict:
        c = C.drop(columns=["record_id"]) if "record_id" in C.columns else C
        c = _canonicalize_c(c[submission_schema.SUBMITTED_C_COLUMNS].copy())
        res = score_utility(c, self.B, self.ref, self.cfg, level=ReportLevel.BREAKDOWN)
        return {"U": float(res.score), "facets": dict(res.facets), "detail": res.detail}

    # ------------------------------------------------------------------ #
    # 匿名性の代理指標（手元近似・公式値ではない）
    # ------------------------------------------------------------------ #
    def mia_proxy(self, C: pd.DataFrame, n_neg: int = 1023, seed: int = 0) -> dict:
        """参照MIA(DOMIAS密度比)をCに当てたときのTPR@1%FPR。

        候補 = B の 26 行（メンバー）+ A_bg の n_neg 行（非メンバー）を 40 回引き直し、
        本番の 1 攻撃者ぶんの標本サイズを再現して平均する（本番は 25〜27 人 / 1049 人）。
        rare worst-case も本番と同様に max で取る。
        """
        rng = np.random.default_rng(seed)
        tprs, tprs_rare = [], []
        rare_idx = np.flatnonzero(self.is_rare)
        maj_idx = np.flatnonzero(~self.is_rare)
        a_idx_all = np.arange(len(self.A_bg))
        # A_bg 側の希少判定は B と同じ検出器で
        a_rare = self._rare_mask(self.A_bg)
        for _ in range(40):
            # メンバー26人: 希少の内数はコホート比率（24/1049）に合わせ 1 人程度
            m_rare = rng.choice(rare_idx, size=1, replace=False)
            m_maj = rng.choice(maj_idx, size=25, replace=False)
            m = np.concatenate([m_rare, m_maj])
            neg = rng.choice(a_idx_all, size=n_neg, replace=False)
            cand = pd.concat([self.B.iloc[m], self.A_bg.iloc[neg]], ignore_index=True)
            truth = np.concatenate([np.ones(len(m), int), np.zeros(len(neg), int)])
            rare_mask = np.concatenate([self.is_rare[m], a_rare[neg]])
            conf = mia_mod.overlap_mia(C, self.A_bg, cand, self.cfg)
            res = attack_score.score_mia(conf, truth, self.cfg, rare_mask=rare_mask)
            tprs.append(res.breakdown["tpr_at_low_fpr"])
            tr = res.breakdown["tpr_rare"]
            if np.isfinite(tr):
                tprs_rare.append(tr)
        t_all = float(np.mean(tprs))
        t_rare = float(np.mean(tprs_rare)) if tprs_rare else float("nan")
        beta = max(t_all, t_rare if np.isfinite(t_rare) else 0.0)
        a_mia = float(np.clip((beta / 0.01 - 1.0) / (12.0 - 1.0), 0.0, 1.0))
        return {"tpr_all": t_all, "tpr_rare": t_rare, "beta_raw": beta, "a_mia_upper": a_mia}

    def aia_proxy(self, C: pd.DataFrame, seed: int = 0, n_target: int = 163) -> dict:
        """参照AIA(最近傍)を C に当てたときの R = (p_m - p_c)/(1 - p_c)。

        標的 = B からランダムに n_target 行。対照 = A_bg から (sex, prefecture, 年齢帯,
        onset) をそろえてマッチした別人。公式の対照条件（県・性別・年齢帯・発症の有無）を
        近似する。
        """
        rng = np.random.default_rng(seed)
        tgt_idx = rng.choice(len(self.B), size=n_target, replace=False)
        tgt = self.B.iloc[tgt_idx].reset_index(drop=True)
        ctl = self._match_controls(tgt, rng)
        qi = ["challenge_row_id", "age", "SBP", "TG", "HDL", "ALT", "sex", "prefecture"]
        both = pd.concat([tgt, ctl], ignore_index=True)
        both["challenge_row_id"] = np.arange(len(both))
        _oc, time_hat = aia_mod.aia_predict(C, both[qi], self.A_bg, self.cfg)
        tau = float(self.cfg["aia"]["tau_time"])
        succ = np.abs(time_hat - both["time"].to_numpy(float)) <= tau
        nm = len(tgt)
        r = attack_score.R_from_rate(succ[:nm].astype(float), succ[nm:].astype(float))
        return {
            "R_time": float(r),
            "p_member": float(succ[:nm].mean()),
            "p_control": float(succ[nm:].mean()),
        }

    def _match_controls(self, tgt: pd.DataFrame, rng) -> pd.DataFrame:
        """A_bg から (sex, prefecture, 年齢帯5歳, onset) をそろえた対照を1対1で引く。"""
        a = self.A_bg.copy()
        a["_ab"] = (a["age"] // 5).astype(int)
        used: set[int] = set()
        rows = []
        for _, t in tgt.iterrows():
            ab = int(t["age"] // 5)
            for keys in (
                (a["sex"] == t["sex"]) & (a["prefecture"] == t["prefecture"]) & (a["_ab"] == ab) & (a["onset"] == t["onset"]),
                (a["sex"] == t["sex"]) & (a["_ab"] == ab) & (a["onset"] == t["onset"]),
                (a["onset"] == t["onset"]),
            ):
                pool = a.index[keys & ~a.index.isin(list(used))]
                if len(pool):
                    j = int(rng.choice(pool))
                    used.add(j)
                    rows.append(a.loc[j])
                    break
        return pd.DataFrame(rows).drop(columns=["_ab"]).reset_index(drop=True)

    def _rare_mask(self, df: pd.DataFrame) -> np.ndarray:
        m = outlier_mod.mahalanobis(df, self.ref.mu, self.ref.sigma, schema.CONTINUOUS)
        return m >= float(self.ref.band["lo"])

    def n_rare(self, C: pd.DataFrame) -> int:
        return int(self._rare_mask(C).sum())

    # ------------------------------------------------------------------ #
    def report(self, C: pd.DataFrame, privacy: bool = True, label: str = "") -> dict:
        out = self.score(C)
        f = out["facets"]
        d = out["detail"]
        print(f"===== {label} =====")
        print(f"  U = {out['U']:.4f}   " + "  ".join(f"{k}={v:.4f}" for k, v in f.items()))
        g = d["U_gen"]
        print(f"   U_gen : marg_mean={g['marginal_mean']:.4f} pmse={g['pmse_score']:.4f} "
              f"time_dz={g['time_dz']:.4f}  (marg_worst={g['marginal_worst']:.4f})")
        s = d["U_spec"]
        if s.get("cox_converged"):
            print(f"   U_spec: io={s['io_part']:.4f} tstr_cal={s['tstr_cal']:.4f} "
                  f"(c_tstr={s['tstr_c_index']:.4f} c_trtr={s['trtr_c_index']:.4f})")
        else:
            print(f"   U_spec: {s.get('error')}")
        r = d["U_rare"]
        print(f"   U_rare: mass*={r['mass_star']:.4f} prof={_fmt(r['prof_cal'])} "
              f"onset={_fmt(r['onset_cal'])} c2st={_fmt(r['c2st_cal'])} "
              f"| nC={r['rare_detected_C']} nB={r['rare_detected_B']} truth={r['rare_true_B']}")
        v = d["U_valid"]
        print(f"   U_valid: cens={v['censoring_realism_rate']:.4f} no_fab={v['no_fab']:.4f}")
        if privacy:
            m = self.mia_proxy(C)
            a = self.aia_proxy(C)
            print(f"   [privacy proxy] MIA tpr_all={m['tpr_all']:.4f} tpr_rare={m['tpr_rare']:.4f} "
                  f"-> a_mia~{m['a_mia_upper']:.3f} | AIA R={a['R_time']:.4f} "
                  f"(pm={a['p_member']:.3f} pc={a['p_control']:.3f})")
            out["mia"] = m
            out["aia"] = a
        return out


def _fmt(x) -> str:
    return "None " if x is None else f"{x:.4f}"


def worst_coefs(detail: dict, n: int = 4) -> list:
    per = detail["U_spec"].get("per_coefficient", {})
    return sorted(per.items(), key=lambda kv: kv[1]["contrib"])[:n]
